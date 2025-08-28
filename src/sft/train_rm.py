#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rm.fixed.py — Robust pairwise reward model trainer (v2)
Improvements vs prior version:
  * Proper AMP: uses torch.amp.autocast; disables GradScaler for bf16.
  * Param groups: higher LR for classification head via --head-lr (default 10x base).
  * Optional head-only warmup steps (--head-only-steps) to stabilize early training.
  * Optional prompt cleanup (--clean-encoder-prompts) to strip chat markers for encoder models.
  * Clearer logs; prints pairwise acc even without a val file if --eval-train is set.

Usage (typical):
  python train_rm.fixed.py \
    --model microsoft/deberta-v3-base \
    --data prefs/preferences.jsonl \
    --save-dir runs/rm/deberta-v3-base \
    --epochs 2 --batch-size 8 --grad-accum 2 \
    --lr 2e-5 --head-lr 1e-4 --warmup-ratio 0.03 \
    --max-length 512 --precision bf16 --clean-encoder-prompts
"""
from __future__ import annotations

import argparse, json, os, sys, math, random, time, re
from typing import List, Dict, Any, Optional

try:
    import yaml
except Exception:
    yaml = None

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

DEFAULTS = {
    "model": "microsoft/deberta-v3-base",
    "data": "prefs/preferences.jsonl",
    "val": None,
    "save_dir": "runs/rm",
    "batch_size": 16,
    "eval_batch_size": 32,
    "grad_accum": 1,
    "lr": 2e-5,
    "head_lr": None,             # default: 10x base lr
    "epochs": 2,
    "max_length": 512,
    "precision": "bf16",         # fp16 | bf16 | fp32
    "checkpointing": False,
    "num_workers": 2,
    "weight_decay": 0.0,
    "warmup_steps": 0,
    "warmup_ratio": 0.03,
    "seed": 42,
    "clip_grad_norm": 1.0,
    "head_only_steps": 0,        # if >0, train only head for first N optimizer updates
    "clean_encoder_prompts": False,
    "eval_train": False,         # if True and no val set, compute pairwise acc on a small train slice
}

CHAT_TAG_RE = re.compile(r"<\|/?(system|user|assistant|tool|observation|assistant_response)\|>", re.IGNORECASE)

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def is_decoder_only(model_type: str) -> bool:
    return model_type in {"llama", "qwen2", "qwen2_moe", "gpt2", "mpt", "mixtral", "phi", "gemma", "opt"}

def ensure_pad_and_sides(tok, model_type: str):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    if is_decoder_only(model_type):
        tok.padding_side = "left"; tok.truncation_side = "left"
    else:
        tok.padding_side = "right"; tok.truncation_side = "right"

def build_head_from_row(r: dict, clean: bool) -> str:
    ptxt = r.get("prompt")
    if isinstance(ptxt, str) and ptxt:
        if clean:
            # strip chat tags and collapse duplicated blank lines
            ptxt = CHAT_TAG_RE.sub("", ptxt)
            ptxt = re.sub(r"\n{3,}", "\n\n", ptxt).strip()
        return ptxt
    q = r.get("question", "")
    head = (
        "You are a factual QA system. Answer using ONLY the provided evidence.\n\n"
        f"Question:\n{q}\n\n"
        "Respond as JSON: {\"answer\": \"...\", \"citations\": [], \"confidence\": 0.5}\n\n"
        "Response:"
    )
    return head

def normalize_resp(sample: dict, key: str) -> str:
    parsed = sample.get(f"{key}_parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str) and parsed["text"]:
        return parsed["text"]
    txt = sample.get(key, "")
    if isinstance(txt, (dict, list)):
        return json.dumps(txt, ensure_ascii=False)
    return str(txt)

class PrefDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int, clean_prompts: bool, encoder_like: bool):
        self.rows = load_jsonl(path)
        self.tok = tokenizer
        self.max_length = max_length
        self.clean_prompts = clean_prompts and encoder_like

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        head = build_head_from_row(r, clean=self.clean_prompts)
        pos = head + "\n" + normalize_resp(r, "chosen")
        neg = head + "\n" + normalize_resp(r, "rejected")
        return {"pos": pos, "neg": neg}

def collate(batch, tokenizer: AutoTokenizer, max_length: int):
    max_len = min(max_length, getattr(tokenizer, "model_max_length", max_length))
    pos_texts = [b["pos"] for b in batch]
    neg_texts = [b["neg"] for b in batch]
    pos = tokenizer(pos_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    neg = tokenizer(neg_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return {"pos": pos, "neg": neg}

@torch.jit.script
def pairwise_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(-(pos_scores - neg_scores)).mean()

def unify_cfg(args, yml: Optional[dict]) -> dict:
    cfg = DEFAULTS.copy()
    if isinstance(yml, dict):
        if "base_model" in yml and "model" not in yml:
            yml = yml.copy(); yml["model"] = yml.pop("base_model")
        tr = yml.get("training") or {}
        if tr:
            cfg["batch_size"] = int(tr.get("batch_size", cfg["batch_size"]))
            cfg["eval_batch_size"] = int(tr.get("eval_batch_size", cfg["eval_batch_size"]))
            cfg["lr"] = float(tr.get("learning_rate", cfg["lr"]))
            cfg["epochs"] = int(tr.get("num_epochs", cfg["epochs"]))
            cfg["grad_accum"] = int(tr.get("gradient_accumulation_steps", cfg["grad_accum"]))
            cfg["weight_decay"] = float(tr.get("weight_decay", cfg["weight_decay"]))
            cfg["warmup_ratio"] = float(tr.get("warmup_ratio", cfg["warmup_ratio"]))
            cfg["warmup_steps"] = int(tr.get("warmup_steps", cfg["warmup_steps"]))
        data = yml.get("data") or {}
        if data:
            cfg["data"] = data.get("train_path", cfg["data"])
            cfg["val"] = data.get("eval_path", cfg["val"])
        for k in ("model","save_dir","max_length","precision","checkpointing","num_workers","seed","clip_grad_norm"):
            if k in yml: cfg[k] = yml[k]
    # CLI overrides
    for k in ["model","data","val","save_dir","batch_size","eval_batch_size","grad_accum","lr","head_lr","epochs",
              "max_length","precision","num_workers","weight_decay","warmup_steps","warmup_ratio","seed",
              "clip_grad_norm","head_only_steps","clean_encoder_prompts","checkpointing","eval_train"]:
        v = getattr(args, k.replace("-","_"), None)
        if v is not None:
            cfg[k] = v
    # Flatten nested model config if dict
    if isinstance(cfg.get("model"), dict):
        md = cfg["model"]
        name = md.get("name") or md.get("id") or md.get("base") or md.get("path")
        if not isinstance(name, str) or not name:
            raise SystemExit("Invalid 'model' section: expected a 'name' field with HF id or local path.")
        cfg["model"] = name
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--val", type=str, default=None)
    ap.add_argument("--save-dir", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--precision", type=str, default=None, choices=["fp16","bf16","fp32"])
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--warmup-ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--clip-grad-norm", type=float, default=None)
    ap.add_argument("--head-only-steps", type=int, default=None)
    ap.add_argument("--clean-encoder-prompts", action="store_true")
    ap.add_argument("--eval-train", action="store_true")
    args = ap.parse_args()

    yml = None
    if args.config:
        if yaml is None:
            print("[fatal] pyyaml not available but --config was provided", file=sys.stderr)
            sys.exit(2)
        with open(args.config, "r", encoding="utf-8") as f:
            yml = yaml.safe_load(f)

    cfg = unify_cfg(args, yml)
    os.makedirs(cfg["save_dir"], exist_ok=True)
    save_json(os.path.join(cfg["save_dir"], "train_config.json"), cfg)

    torch.manual_seed(int(cfg["seed"])); random.seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg["model"]
    hf_cfg = AutoConfig.from_pretrained(model_name, num_labels=1, problem_type="regression")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if is_decoder_only(hf_cfg.model_type):
        tok.padding_side = "left"
        tok.truncation_side = "left"
    else:
        tok.padding_side = "right"
        tok.truncation_side = "left"   # <-- change to 'left' so the answer is preserved

    ensure_pad_and_sides(tok, hf_cfg.model_type)
    eff_max_len = min(int(cfg["max_length"]), getattr(tok, "model_max_length", int(cfg["max_length"])) or int(cfg["max_length"]))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=hf_cfg)
    if cfg["checkpointing"] and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)

    # Precision
    amp_dtype = None
    if cfg["precision"] == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg["precision"] == "fp16":
        amp_dtype = torch.float16

    # Data
    if not os.path.exists(cfg["data"]):
        raise SystemExit(f"Training data not found: {cfg['data']}")
    encoder_like = not is_decoder_only(hf_cfg.model_type)
    train_ds = PrefDataset(cfg["data"], tok, eff_max_len, clean_prompts=cfg["clean_encoder_prompts"], encoder_like=encoder_like)
    val_ds = PrefDataset(cfg["val"], tok, eff_max_len, clean_prompts=cfg["clean_encoder_prompts"], encoder_like=encoder_like) if cfg["val"] else None
    coll = lambda b: collate(b, tok, eff_max_len)

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg["num_workers"]), pin_memory=(device=='cuda'),
        collate_fn=coll, persistent_workers=(int(cfg["num_workers"])>0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["eval_batch_size"] or cfg["batch_size"]), shuffle=False,
        num_workers=int(cfg["num_workers"]), pin_memory=(device=='cuda'),
        collate_fn=coll, persistent_workers=(int(cfg["num_workers"])>0)
    ) if val_ds else None

    # Optimizer with param groups (head vs backbone)
    head_kw = ("classifier", "score", "pooler.dense")
    head_params, body_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        (head_params if any(k in n for k in head_kw) else body_params).append(p)

    base_lr = float(cfg["lr"])
    head_lr = float(cfg["head_lr"]) if cfg["head_lr"] is not None else base_lr * 10.0
    optim_groups = [
        {"params": body_params, "lr": base_lr, "weight_decay": float(cfg["weight_decay"])},
        {"params": head_params, "lr": head_lr, "weight_decay": float(cfg["weight_decay"])}
    ]
    opt = torch.optim.AdamW(optim_groups)

    total_update_steps = math.ceil(len(train_loader) / max(1, int(cfg["grad_accum"]))) * int(cfg["epochs"])
    warmup_steps = int(cfg["warmup_steps"]) if cfg["warmup_steps"] else int((cfg["warmup_ratio"] or 0) * total_update_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)

    use_scaler = (amp_dtype is not None and device == "cuda" and amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print("============================================================")
    print("Reward Model Training (v2)")
    print("============================================================")
    print(f"Model: {model_name} (type={hf_cfg.model_type})")
    print(f"Train rows: {len(train_ds)} | Val rows: {len(val_ds) if val_ds else 0}")
    print(f"Batch size: {cfg['batch_size']} (accum={cfg['grad_accum']}) | Epochs: {cfg['epochs']}")
    print(f"Max length: {eff_max_len} | Precision: {cfg['precision']} | Device: {device}")
    print(f"LR (body/head): {base_lr:.2e} / {head_lr:.2e} | Warmup steps: {warmup_steps}")
    print(f"Head-only steps: {cfg['head_only_steps']} | Clean encoder prompts: {cfg['clean_encoder_prompts']}")
    print(f"Save dir: {cfg['save_dir']}")
    print("============================================================")

    best_val = float("inf")
    global_step = 0
    head_only_steps = int(cfg["head_only_steps"] or 0)

    for epoch in range(int(cfg["epochs"])):
        model.train()
        running = 0.0; t0 = time.time()
        for step, batch in enumerate(train_loader):
            pos = {k: v.to(device, non_blocking=True) for k, v in batch["pos"].items()}
            neg = {k: v.to(device, non_blocking=True) for k, v in batch["neg"].items()}
            if head_only_steps > 0 and global_step < head_only_steps:
                for p in body_params: p.requires_grad = False
            else:
                for p in body_params: 
                    if not p.requires_grad: p.requires_grad = True

            if use_scaler:
                with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                    pos_out = model(**pos).logits.squeeze(-1)
                    neg_out = model(**neg).logits.squeeze(-1)
                    loss = pairwise_loss(pos_out, neg_out) / max(1, int(cfg["grad_accum"]))
                scaler.scale(loss).backward()
            else:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and device=='cuda')):
                    pos_out = model(**pos).logits.squeeze(-1)
                    neg_out = model(**neg).logits.squeeze(-1)
                    loss = pairwise_loss(pos_out, neg_out) / max(1, int(cfg["grad_accum"]))
                loss.backward()

            running += loss.item()
            if (step + 1) % int(cfg["grad_accum"]) == 0:
                if float(cfg["clip_grad_norm"]) and cfg["clip_grad_norm"] > 0:
                    if use_scaler:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip_grad_norm"]))
                if use_scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

        avg_loss = (running * max(1,int(cfg["grad_accum"]))) / max(1, len(train_loader))
        print(f"[epoch {epoch+1}/{cfg['epochs']}] train loss: {avg_loss:.4f}  (time: {time.time()-t0:.1f}s)")

        # Validation
        if val_loader:
            model.eval()
            losses, acc = [], []
            with torch.no_grad():
                for batch in val_loader:
                    pos = {k: v.to(device, non_blocking=True) for k, v in batch["pos"].items()}
                    neg = {k: v.to(device, non_blocking=True) for k, v in batch["neg"].items()}
                    out_p = model(**pos).logits.squeeze(-1)
                    out_n = model(**neg).logits.squeeze(-1)
                    l = pairwise_loss(out_p, out_n)
                    losses.append(l.item())
                    acc.extend(((out_p > out_n).float().cpu().tolist()))
            v = sum(losses)/max(1,len(losses)); a = sum(acc)/max(1,len(acc))
            print(f"[epoch {epoch+1}] val loss: {v:.4f} | pairwise acc: {a:.3f}")
            if v < best_val:
                best_val = v
                path = os.path.join(cfg["save_dir"], "best")
                os.makedirs(path, exist_ok=True)
                model.save_pretrained(path); tok.save_pretrained(path)
                save_json(os.path.join(cfg["save_dir"], "best_metrics.json"), {"val_loss": v, "pairwise_acc": a})
                print(f"  ✓ saved best to {path}")
        elif cfg["eval_train"]:
            # quick sanity check on a small training slice
            model.eval()
            with torch.no_grad():
                n = 256 if len(train_ds) > 256 else len(train_ds)
                loader = DataLoader(torch.utils.data.Subset(train_ds, range(n)),
                                    batch_size=int(cfg["eval_batch_size"] or cfg["batch_size"]), collate_fn=coll)
                acc = []
                for batch in loader:
                    pos = {k: v.to(device) for k, v in batch["pos"].items()}
                    neg = {k: v.to(device) for k, v in batch["neg"].items()}
                    out_p = model(**pos).logits.squeeze(-1)
                    out_n = model(**neg).logits.squeeze(-1)
                    acc.extend(((out_p > out_n).float().cpu().tolist()))
                a = sum(acc)/max(1,len(acc))
                print(f"[epoch {epoch+1}] train-slice pairwise acc: {a:.3f}")

    # final save
    path = os.path.join(cfg["save_dir"], "final")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path); tok.save_pretrained(path)
    print(f"Saved final checkpoint to: {path}")

if __name__ == "__main__":
    main()

