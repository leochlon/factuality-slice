#!/usr/bin/env python3
# train_sft.py
"""
Train baseline with LoRA/QLoRA using standard Trainer.

Usage:
  python train_sft.py -c configs/sft_llama3.yaml
"""

import os, json, math, time, logging, inspect
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
    TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# Local
from prompt_schema import PromptTemplates

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_sft")


# ---------- IO ----------
def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


# ---------- Dataset ----------
class EvidenceQADataset(Dataset):
    """
    Formats data for answer-only loss (prompt tokens masked with -100).
    """
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        template_name: str = "default",
        max_seq_len: int = 2048,
        refusal_low_conf: float = 0.0,
        answer_conf: float = 0.85,
    ):
        self.data = data
        self.tok = tokenizer
        self.template = template_name
        self.max_len = max_seq_len
        self.refusal_low_conf = refusal_low_conf
        self.answer_conf = answer_conf
        self._pt = PromptTemplates()

    def __len__(self):
        return len(self.data)

    def _gold_json(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        ans = ex.get("answer", "")
        support = ex.get("support_spans", []) or []
        refusal_markers = {"[INSUFFICIENT_EVIDENCE]", "[OUTDATED_CONTEXT]", "[NO_EVIDENCE]"}
        is_refusal = isinstance(ans, str) and ans.strip().upper() in refusal_markers
        conf = self.refusal_low_conf if is_refusal else self.answer_conf
        n_chunks = len(ex.get("context_chunks", []))
        citations = [int(i) for i in support if isinstance(i, int) and 0 <= i < n_chunks]
        return {"answer": ans, "citations": sorted(set(citations)), "confidence": float(conf)}

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        ex = self.data[idx]
        prompt = self._pt.get_sft_prompt(
            question=ex["question"],
            evidence_chunks=ex["context_chunks"],
            template_name=self.template,
        )
        target_json = json.dumps(self._gold_json(ex), ensure_ascii=False)

        # Encode prompt and answer separately to build labels mask
        enc_prompt = self.tok(
            prompt,
            truncation=True,
            max_length=max(8, self.max_len - 128),   # leave space for JSON answer
            padding=False,
            add_special_tokens=False,
            return_tensors=None,
        )

        # Budget for answer (ensure positive)
        prompt_len = len(enc_prompt["input_ids"])
        answer_budget = max(8, self.max_len - prompt_len)

        eos = self.tok.eos_token or ""  # guard if eos_token is None
        enc_answer = self.tok(
            "\n" + target_json + eos,
            truncation=True,
            max_length=answer_budget,
            padding=False,
            add_special_tokens=False,
            return_tensors=None,
        )

        input_ids = enc_prompt["input_ids"] + enc_answer["input_ids"]
        labels = [-100] * len(enc_prompt["input_ids"]) + enc_answer["input_ids"]

        # Truncate to max_len just in case
        input_ids = input_ids[:self.max_len]
        labels = labels[:self.max_len]

        return {"input_ids": input_ids, "labels": labels}


# ---------- Utilities ----------
def ensure_pad_token(tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "right"


def count_tokens(tokenizer: AutoTokenizer, dataset: Dataset) -> int:
    sample_size = min(len(dataset), 100)
    if sample_size == 0:
        return 0
    total_in_sample = 0
    for i in range(sample_size):
        item = dataset[i]
        total_in_sample += len(item["input_ids"])
    avg = total_in_sample / sample_size
    return int(avg * len(dataset))


def guess_attn_implementation(want: str) -> Optional[str]:
    want = (want or "").lower()
    if want in ("flash_attention_2", "flash-attention-2", "flash2"):
        try:
            import flash_attn  # noqa: F401
            log.info("Flash Attention 2 found, will use.")
            return "flash_attention_2"
        except Exception:
            log.warning("Flash Attention 2 not found, falling back to SDPA.")
            return "sdpa"
    if want in ("sdpa", "eager"):
        return want
    return None  # let transformers decide


def filter_target_modules(model, requested: List[str]) -> List[str]:
    present_modules = {n.split(".")[-1] for n, _ in model.named_modules()}
    filtered = [m for m in requested if m in present_modules]
    if not filtered:
        log.warning(f"No exact LoRA targets found in {requested}; trying suffix match.")
        all_names = {n for n, _ in model.named_modules()}
        fallback = [m for m in requested if any(n.endswith(m) for n in all_names)]
        filtered = list(dict.fromkeys(fallback))
    if not filtered:
        log.error(f"Could not find any LoRA target modules from: {requested}")
    return filtered


# ---------- Data Collator & Callback ----------
class CustomDataCollatorForCausalLM:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        for f in features:
            assert "input_ids" in f and "labels" in f, f"Feature missing keys: {f.keys()}"

        max_len_in_batch = max(len(f["input_ids"]) for f in features)
        max_len = min(max_len_in_batch, self.max_length)
        if max_len % 8 != 0:
            max_len = ((max_len // 8) + 1) * 8

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"][:self.max_length]
            lbs = f["labels"][:self.max_length]
            pad_len = max(0, max_len - len(ids))

            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbs + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class ProgressCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        cur = 0 if state.epoch is None else int(state.epoch)
        tot = int(math.ceil(args.num_train_epochs))
        print(f"\n{'='*50}\nStarting Epoch {cur + 1}/{tot}\n{'='*50}")

    def on_epoch_end(self, args, state, control, **kwargs):
        cur = 0 if state.epoch is None else int(state.epoch)
        tot = int(math.ceil(args.num_train_epochs))
        print(f"\n{'='*50}\nCompleted Epoch {cur + 1}/{tot}")
        if state.log_history:
            latest = state.log_history[-1]
            loss = latest.get("eval_loss", latest.get("loss"))
            lr = latest.get("learning_rate")
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A"
            lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else "N/A"
            print(f"Loss: {loss_str} | Learning Rate: {lr_str}")
        print(f"{'='*50}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Use args.process_index to gate printing on rank 0
        if logs and getattr(args, "process_index", 0) == 0:
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A"
            lr_str = f"{lr:.2e}" if isinstance(lr, (int, float)) else "N/A"
            print(f"Step {state.global_step}: Loss={loss_str}, LR={lr_str}")


# ---------- Main ----------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=Path, required=True, help="Path to the config YAML file.")
    p.add_argument("--debug", action="store_true", help="Run in debug mode with fewer samples.")
    args = p.parse_args()

    cfg = load_yaml(args.config)

    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    # ---- Model & tokenizer ----
    model_name = cfg["model_name"]
    use_4bit = bool(cfg.get("load_in_4bit", True))
    torch_dtype = torch.bfloat16 if bool(cfg.get("bf16", True)) else torch.float16
    trust_remote_code = bool(cfg.get("trust_remote_code", False))

    attn_impl = guess_attn_implementation(cfg.get("attn_implementation", "flash_attention_2"))
    if "gemma-2" in model_name.lower():
        log.warning("Forcing 'eager' attention for Gemma2.")
        attn_impl = "eager"

    model_kwargs = dict(
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
    )

    if use_4bit:
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb_cfg
        except Exception as e:
            log.warning(f"4-bit loading failed: {e}. Falling back to {torch_dtype}.")
            use_4bit = False

    log.info(f"Loading base model: {model_name} (4-bit={use_4bit}, attn={attn_impl}, dtype={torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    ensure_pad_token(tok, model)

    max_seq_length = int(cfg["training"].get("max_seq_length", 2048))
    tok.model_max_length = max_seq_length

    # Train config on model
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # ---- Apply (Q)LoRA ----
    if use_4bit:
        try:
            model = prepare_model_for_kbit_training(model)
        except Exception as e:
            log.warning(f"prepare_model_for_kbit_training failed: {e}")

    lora_cfg_raw = cfg.get("lora", {})
    requested_targets = lora_cfg_raw.get("target_modules") or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]
    target_modules = filter_target_modules(model, requested_targets)

    lora_cfg = LoraConfig(
        r=int(lora_cfg_raw.get("r", 16)),
        lora_alpha=int(lora_cfg_raw.get("alpha", 32)),
        lora_dropout=float(lora_cfg_raw.get("dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    log.info(f"Applying LoRA to modules: {target_modules}")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- Data ----
    data_cfg = cfg.get("data", {})
    train_path = Path(data_cfg.get("train_path", "data/processed/train.jsonl"))
    val_path = Path(data_cfg.get("val_path", "data/processed/val.jsonl"))
    template = data_cfg.get("template", "default")
    max_train = 100 if args.debug else data_cfg.get("max_samples")
    max_val = 20 if args.debug else data_cfg.get("max_val_samples")

    train_rows = read_jsonl(train_path, max_rows=max_train)
    val_rows = read_jsonl(val_path, max_rows=max_val)

    train_ds = EvidenceQADataset(train_rows, tok, template_name=template, max_seq_len=max_seq_length)
    val_ds = EvidenceQADataset(val_rows, tok, template_name=template, max_seq_len=max_seq_length)

    log.info(f"Loaded {len(train_ds)} training and {len(val_ds)} validation examples.")

    # ---- Token counts & mixtures ----
    log.info("Estimating token counts...")
    t0 = time.time()
    train_token_count = count_tokens(tok, train_ds)
    val_token_count = count_tokens(tok, val_ds)
    log.info(f"Token counts: train≈{train_token_count:,}, val≈{val_token_count:,} (in {time.time()-t0:.1f}s)")

    train_mix = dict(Counter(r.get("metadata", {}).get("source", "unknown") for r in train_rows))
    val_mix = dict(Counter(r.get("metadata", {}).get("source", "unknown") for r in val_rows))

    # ---- TrainingArguments (version‑robust) ----
    out_dir = Path(cfg.get("output_dir", "checkpoints/sft"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_args_cfg = cfg.get("training", {})
    eval_enabled = len(val_ds) > 0

    # Build kwargs and include gradient_checkpointing_kwargs only if supported
    ta_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tr_args_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(tr_args_cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(tr_args_cfg.get("gradient_accumulation_steps", 64)),
        learning_rate=float(tr_args_cfg.get("learning_rate", 1.5e-5)),
        num_train_epochs=float(tr_args_cfg.get("num_train_epochs", 2)),
        weight_decay=float(tr_args_cfg.get("weight_decay", 0.1)),
        warmup_ratio=float(tr_args_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(tr_args_cfg.get("logging_steps", 10)),
        eval_steps=int(tr_args_cfg.get("eval_steps", 100)) if eval_enabled else None,
        save_steps=int(tr_args_cfg.get("save_steps", 100)),
        save_total_limit=int(tr_args_cfg.get("save_total_limit", 2)),
        lr_scheduler_type=str(tr_args_cfg.get("lr_scheduler_type", "cosine")),
        bf16=bool(cfg.get("bf16", True)),
        fp16=not bool(cfg.get("bf16", True)),
        gradient_checkpointing=True,
        optim="paged_adamw_32bit" if use_4bit else "adamw_torch",
        max_grad_norm=1.0,
        report_to=["none"],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        logging_dir=str(out_dir / "logs"),
        logging_first_step=True,
        log_level="info",
        disable_tqdm=False,
    )

    if "gradient_checkpointing_kwargs" in inspect.signature(TrainingArguments).parameters:
        ta_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    training_args = TrainingArguments(**ta_kwargs)

    data_collator = CustomDataCollatorForCausalLM(tokenizer=tok, max_length=max_seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if eval_enabled else None,
        data_collator=data_collator,
        tokenizer=tok,
        callbacks=[ProgressCallback()],
    )

    # ---- Print training info ----
    steps_per_epoch = math.ceil(
        len(train_ds) / max(1, training_args.per_device_train_batch_size)
    )
    opt_steps_per_epoch = math.ceil(
        steps_per_epoch / max(1, training_args.gradient_accumulation_steps)
    )
    total_steps = int(opt_steps_per_epoch * training_args.num_train_epochs)

    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Total training samples: {len(train_ds)}")
    print(f"Total validation samples: {len(val_ds)}")
    print(f"Batch size per device: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Total optimization steps (approx): {total_steps}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Output directory: {out_dir}")
    print("="*60 + "\n")

    # ---- Train ----
    log.info("Starting training...")
    try:
        trainer.train()
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        if trainer.state.log_history:
            merged_logs = {}
            for lg in trainer.state.log_history:
                merged_logs.update(lg)
            final_loss = merged_logs.get("eval_loss", merged_logs.get("loss"))
            print("Final Loss:", f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else "N/A")
            print("Total Steps:", trainer.state.global_step)
            print("Total Epochs:", f"{trainer.state.epoch:.2f}" if isinstance(trainer.state.epoch, (int, float)) else "N/A")
        print("="*60 + "\n")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    # ---- Save ----
    log.info(f"Saving final model and tokenizer to {out_dir}...")
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    # ---- Model card ----
    card = f"""# Fine-tuned Model (LoRA/QLoRA)

**Base model:** `{model_name}`
**Output dir:** `{out_dir}`
**Quantization:** {"QLoRA (4-bit nf4)" if use_4bit else "LoRA"}
**LoRA:** r={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, dropout={lora_cfg.lora_dropout}, targets={target_modules}
**bf16:** {cfg.get("bf16", True)}
**Grad checkpointing:** True
**Seq length:** {max_seq_length}

## Data mixture
**Train:** `{dict(train_mix)}`
**Val:** `{dict(val_mix)}`

## Token counts (estimated)
- Train tokens: {train_token_count:,}
- Val tokens: {val_token_count:,}

## Config
```yaml
{yaml.safe_dump(cfg, sort_keys=False)}
"""
    (out_dir / "model_card.md").write_text(card)
    log.info("Done.")

if __name__ == "__main__":
    raise SystemExit(main())