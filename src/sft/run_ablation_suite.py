#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation_suite.py

Orchestrates two ablation families using your existing scripts:
  1) Noise robustness tests  -> sft.evaluate_sft_fast_robust
  2) Data scaling curves     -> sft.train_rm (+ built-in validator)

Outputs:
  ablations/
    noise/
      datasets/<noise>-<level>.jsonl
      results_<noise>.json
      summary_noise.md
    scaling/
      rm/
        splits/
        runs/
        rm_scaling.json
        summary_scaling.md

Usage examples:

# 1) Noise robustness on your SFT
python -m sft.run_ablation_suite noise \
  --test data/processed/test.jsonl \
  --sft-checkpoint checkpoints/sft/gemma2-9b/checkpoint-60 \
  --base-model google/gemma-2-9b \
  --attn-impl eager \
  --buckets 2048,3072,4096,5120,6144,7168,7936 \
  --max-new-tokens 256

# 2) RM data scaling curves from preferences (auto 10% val split)
python -m sft.run_ablation_suite scaling \
  --prefs prefs/preferences_cb68999425.jsonl \
  --model microsoft/deberta-v3-base \
  --epochs 3 --batch-size 8 --grad-accum 2 \
  --lr 2e-5 --head-lr 1e-4 --warmup-ratio 0.03 \
  --sizes 128,256,512,768,1024,max

Notes:
- For Gemma‑2, prefer --attn-impl eager in eval for stability.
- Scaling curves default to RM (cheap). SFT scaling can be added later.
"""

import argparse, os, json, random, math, shutil, subprocess, sys, time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# ------------------------- IO utils -------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def run_cmd(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=check)

# ------------------------- Noise utils -------------------------
def _chunk_type(row):
    chunks = row.get("context_chunks", [])
    if not chunks:
        return "empty"
    return "dict" if isinstance(chunks[0], dict) else "str"

def _extract_text(ch):
    if isinstance(ch, dict):
        return ch.get("text", "")
    return str(ch)

def _make_chunk_like(proto, text, tag="distractor"):
    if proto == "dict":
        return {"id": f"{tag}-{random.randint(100000,999999)}", "text": text, "source": tag}
    else:
        return text

def inject_distractors(rows: List[Dict[str, Any]], k_values: List[int], rng: random.Random) -> Dict[str, List[Dict[str, Any]]]:
    # Build a pool of all chunk texts
    pool = []
    for r in rows:
        for ch in r.get("context_chunks", []):
            t = _extract_text(ch).strip()
            if t:
                pool.append(t)
    if not pool:
        pool = ["[Noisy distractor]"]

    out = {}
    for k in k_values:
        mutated = []
        for r in rows:
            kind = _chunk_type(r)
            new_row = dict(r)
            base = list(r.get("context_chunks", []))
            # sample K texts not identical to existing ones
            existing_texts = set(_extract_text(ch).strip() for ch in base)
            sampled = []
            for _ in range(k):
                t = rng.choice(pool)
                # try a few times to avoid duplicates
                tries = 0
                while t in existing_texts and tries < 5:
                    t = rng.choice(pool); tries += 1
                sampled.append(_make_chunk_like(kind, t, tag="distractor"))
            new_row["context_chunks"] = base + sampled
            mutated.append(new_row)
        out[str(k)] = mutated
    return out

def drop_support(rows: List[Dict[str, Any]], drop_fracs: List[float], rng: random.Random) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for frac in drop_fracs:
        mutated = []
        for r in rows:
            chunks = list(r.get("context_chunks", []))
            if not chunks:
                mutated.append(dict(r))
                continue
            n = len(chunks)
            # support indices may be list of ints or stored as support_spans
            supports = set()
            ss = r.get("support_spans", [])
            for s in ss:
                try:
                    si = int(s)
                    if 0 <= si < n:
                        supports.add(si)
                except Exception:
                    pass
            # if none, nothing to drop
            if not supports:
                mutated.append(dict(r))
                continue
            keep_support = set(idx for idx in supports if rng.random() > frac)
            # always keep non-support
            keep = [i for i in range(n) if i not in supports or i in keep_support]
            # build mapping old->new
            new_chunks = [chunks[i] for i in keep]
            idx_map = {old: new for new, old in enumerate(keep)}
            new_support = [idx_map[i] for i in sorted(keep_support) if i in idx_map]
            new_row = dict(r)
            new_row["context_chunks"] = new_chunks
            new_row["support_spans"] = new_support
            mutated.append(new_row)
        out[f"{frac:.2f}"] = mutated
    return out

def noisy_chars(text: str, rate: float, rng: random.Random) -> str:
    if rate <= 0: return text
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    out = []
    for ch in text:
        if rng.random() < rate:
            # mutate
            op = rng.choice(["swap", "drop", "noise"])
            if op == "swap" and out:
                out[-1], ch = ch, out[-1]
                out.append(ch)
            elif op == "drop":
                continue
            else:  # noise
                out.append(rng.choice(alphabet))
        else:
            out.append(ch)
    return "".join(out)

def perturb_text(rows: List[Dict[str, Any]], noise_rates: List[float], rng: random.Random) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for rate in noise_rates:
        mutated = []
        for r in rows:
            kind = _chunk_type(r)
            new_row = dict(r)
            new_chunks = []
            for ch in r.get("context_chunks", []):
                t = _extract_text(ch)
                tn = noisy_chars(t, rate, rng)
                new_chunks.append(_make_chunk_like(kind, tn, tag="noisy"))
            new_row["context_chunks"] = new_chunks
            mutated.append(new_row)
        out[f"{rate:.3f}"] = mutated
    return out

# ------------------------- Evaluator wrapper -------------------------
def run_eval_on(data_path: Path,
                sft_ckpt: str,
                base_model: str,
                out_json: Path,
                max_new_tokens: int = 256,
                fast: bool = True,
                buckets: Optional[str] = None,
                attn_impl: str = "eager",
                compare_baseline: bool = False):
    cmd = [
        sys.executable, "-m", "sft.evaluate_sft_fast_robust",
        "--model-path", sft_ckpt,
        "--base-model", base_model,
        "--test-data", str(data_path),
        "--max-new-tokens", str(max_new_tokens),
        "--attn-impl", attn_impl,
        "--output", str(out_json),
    ]
    if fast:
        cmd.append("--fast")
    if buckets:
        cmd.extend(["--buckets", buckets])
    if compare_baseline:
        cmd.append("--compare-baseline")
    run_cmd(cmd)
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ------------------------- Scaling (RM) -------------------------
def split_prefs(path: Path, val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    rows = read_jsonl(path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    nv = max(1, int(val_ratio * n))
    return rows[nv:], rows[:nv]

def subset(rows: List[dict], size: int) -> List[dict]:
    if size >= len(rows) or size < 0:
        return rows
    return rows[:size]

def is_decoder_only_name(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(k in name for k in ["llama", "qwen", "mpt", "gpt2", "phi", "mixtral", "gemma", "opt"])

def eval_rm_pairwise_acc(model_dir: Path, data_rows: List[dict], batch_size: int = 16, device: str = "cuda") -> Dict[str, Any]:
    """
    Lightweight scorer for the trained RM (deberta-style encoders).
    Computes pairwise accuracy on provided rows.
    """
    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Build prompts like train_rm.py
    def build_user_prompt(question: str, context_chunks: List[str]) -> str:
        evidence = []
        for i, ch in enumerate(context_chunks or [], start=1):
            if isinstance(ch, dict):
                text = ch.get("text", "")
            else:
                text = str(ch)
            evidence.append(f"[{i}] {text}")
        ev = "\n".join(evidence) if evidence else "[none]"
        sys_msg = "You are a helpful assistant. Use ONLY the EVIDENCE to answer. Respond in JSON with keys: answer, citations (list), confidence (0-1)."
        user = f"Question: {question}\n\nEVIDENCE (numbered):\n{ev}"
        return f"<|system|>\n{sys_msg}\n<|user|>\n{user}\n<|assistant|>\n"

    pairs = []
    for r in data_rows:
        p = r.get("prompt", {})
        q = p.get("question", "")
        ctx = p.get("context_chunks", [])
        head = build_user_prompt(q, ctx)

        def norm(side: str) -> str:
            parsed = r.get(f"{side}_parsed")
            if isinstance(parsed, dict) and "answer" in parsed:
                out = {
                    "answer": parsed.get("answer", ""),
                    "citations": parsed.get("citations", []),
                    "confidence": parsed.get("confidence", 0.5),
                }
                return json.dumps(out, ensure_ascii=False)
            txt = r.get(side, "")
            if isinstance(txt, (dict, list)):
                return json.dumps(txt, ensure_ascii=False)
            return str(txt)

        a = head + norm("chosen")
        b = head + norm("rejected")
        pairs.append((a, b))

    # batching
    acc_n = 0
    tot = 0
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        pos_texts = [p for p, _ in batch]
        neg_texts = [n for _, n in batch]
        pos = tok(pos_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        neg = tok(neg_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            ps = model(**pos).logits.squeeze(-1)
            ns = model(**neg).logits.squeeze(-1)
        acc_n += (ps > ns).sum().item()
        tot += len(batch)
    return {"pairs": tot, "pairwise_acc": (acc_n / max(1, tot))}

def run_rm_scaling(prefs_path: Path,
                   out_dir: Path,
                   model_name: str = "microsoft/deberta-v3-base",
                   sizes: List[str] = ["128","256","512","768","1024","max"],
                   epochs: int = 3,
                   batch_size: int = 8,
                   grad_accum: int = 2,
                   lr: float = 2e-5,
                   head_lr: float = 1e-4,
                   warmup_ratio: float = 0.03,
                   val_ratio: float = 0.1,
                   seed: int = 42,
                   device: str = "cuda") -> Dict[str, Any]:

    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = out_dir / "splits"
    runs_dir = out_dir / "runs"
    splits_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = split_prefs(prefs_path, val_ratio, seed)
    write_jsonl(splits_dir / "val.jsonl", val_rows)

    results = {"meta": {
                    "model": model_name, "epochs": epochs, "batch_size": batch_size,
                    "grad_accum": grad_accum, "lr": lr, "head_lr": head_lr,
                    "warmup_ratio": warmup_ratio, "val_ratio": val_ratio,
                    "train_size_total": len(train_rows), "val_size": len(val_rows),
                    "time": now()
               },
               "runs": []}

    for sz in sizes:
        if sz == "max":
            n = len(train_rows)
        else:
            n = max(1, min(int(sz), len(train_rows)))
        subset_rows = subset(train_rows, n)
        split_train = splits_dir / f"train_{n}.jsonl"
        write_jsonl(split_train, subset_rows)

        save_dir = runs_dir / f"rm_{model_name.split('/')[-1]}_{n}"
        # train RM with your script
        cmd = [
            sys.executable, "-m", "sft.train_rm",
            "--model", model_name,
            "--data", str(split_train),
            "--val", str(splits_dir / "val.jsonl"),
            "--save-dir", str(save_dir),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--grad-accum", str(grad_accum),
            "--lr", str(lr),
            "--head-lr", str(head_lr),
            "--warmup-ratio", str(warmup_ratio),
            "--max-length", "512",
            "--precision", "bf16",
            "--clean-encoder-prompts",
            "--head-only-steps", "20"
        ]
        # Some train_rm variants don't support eval-train; keep minimal args above.
        run_cmd(cmd)

        # prefer "best" if present
        best_dir = save_dir / "best"
        final_dir = save_dir / "final"
        model_dir = best_dir if best_dir.exists() else final_dir

        val_score = eval_rm_pairwise_acc(model_dir, val_rows, batch_size=32, device=device)
        results["runs"].append({
            "train_n": n,
            "model_dir": str(model_dir),
            "val_pairs": val_score["pairs"],
            "val_pairwise_acc": val_score["pairwise_acc"]
        })
        write_json(out_dir / "rm_scaling.json", results)

    # quick markdown
    lines = [
        "# RM Scaling Curve",
        f"- Model: **{model_name}**",
        f"- Train total: {results['meta']['train_size_total']}, Val: {results['meta']['val_size']}",
        "",
        "| Train pairs | Val pairwise acc |",
        "|---:|---:|",
    ]
    for r in sorted(results["runs"], key=lambda x: x["train_n"]):
        lines.append(f"| {r['train_n']} | {r['val_pairwise_acc']:.3f} |")
    (out_dir / "summary_scaling.md").write_text("\n".join(lines))
    return results

# (Judge ablations removed: dependency on external rlaif_judge.py)

# ------------------------- Noise driver -------------------------
def run_noise_robustness(test_path: Path,
                         out_root: Path,
                         sft_ckpt: str,
                         base_model: str,
                         attn_impl: str = "eager",
                         buckets: Optional[str] = None,
                         max_new_tokens: int = 256,
                         compare_baseline: bool = False,
                         seed: int = 42) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    datasets_dir = out_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rows = read_jsonl(test_path)
    report: Dict[str, Any] = {"meta": {
                                "test_path": str(test_path),
                                "sft_ckpt": sft_ckpt,
                                "base_model": base_model,
                                "attn_impl": attn_impl,
                                "max_new_tokens": max_new_tokens,
                                "seed": seed,
                                "time": now()
                              },
                              "runs": []}

    # 1) Distractor injection: K ∈ {0,2,4,8}
    dist_ks = [0,2,4,8]
    dist_map = inject_distractors(rows, dist_ks, rng)
    for k, mrows in dist_map.items():
        p = datasets_dir / f"distractors-{k}.jsonl"
        write_jsonl(p, mrows)
        out_json = out_root / f"results_distractors_{k}.json"
        metrics = run_eval_on(p, sft_ckpt, base_model, out_json,
                              max_new_tokens=max_new_tokens, fast=True,
                              buckets=buckets, attn_impl=attn_impl,
                              compare_baseline=compare_baseline)
        report["runs"].append({"noise": "distractors", "level": k, "data": str(p), "metrics_path": str(out_json), "metrics": metrics})

    # 2) Drop support: frac ∈ {0.0, 0.25, 0.5, 0.75}
    drop_fracs = [0.0, 0.25, 0.5, 0.75]
    drop_map = drop_support(rows, drop_fracs, rng)
    for frac, mrows in drop_map.items():
        p = datasets_dir / f"drop-support-{frac}.jsonl"
        write_jsonl(p, mrows)
        out_json = out_root / f"results_drop_support_{frac}.json"
        metrics = run_eval_on(p, sft_ckpt, base_model, out_json,
                              max_new_tokens=max_new_tokens, fast=True,
                              buckets=buckets, attn_impl=attn_impl,
                              compare_baseline=compare_baseline)
        report["runs"].append({"noise": "drop_support", "level": frac, "data": str(p), "metrics_path": str(out_json), "metrics": metrics})

    # 3) Text perturbation: rate ∈ {0.0, 0.01, 0.03, 0.05}
    rates = [0.0, 0.01, 0.03, 0.05]
    pert_map = perturb_text(rows, rates, rng)
    for rate, mrows in pert_map.items():
        p = datasets_dir / f"text-noise-{rate}.jsonl"
        write_jsonl(p, mrows)
        out_json = out_root / f"results_text_noise_{rate}.json"
        metrics = run_eval_on(p, sft_ckpt, base_model, out_json,
                              max_new_tokens=max_new_tokens, fast=True,
                              buckets=buckets, attn_impl=attn_impl,
                              compare_baseline=compare_baseline)
        report["runs"].append({"noise": "text_noise", "level": rate, "data": str(p), "metrics_path": str(out_json), "metrics": metrics})

    write_json(out_root / "noise_robustness.json", report)

    # Markdown summary (SFT only if present)
    lines = [
        "# Noise Robustness",
        f"- Base model: `{base_model}`",
        f"- SFT: `{sft_ckpt}`",
        "",
        "## Key metric (EM/F1) by noise",
        "",
        "| Noise | Level | EM | F1 | Refusal | Citation correctness |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    def pick(mdict):
        # adapt to your evaluate_sft_fast_robust.py output keys if needed
        if not mdict: return None
        # Prefer SFT metrics if compare_baseline was enabled:
        if isinstance(mdict, dict) and "sft" in mdict and isinstance(mdict["sft"], dict):
            return mdict["sft"]
        return mdict  # flat dict fallback

    for r in report["runs"]:
        m = pick(r.get("metrics"))
        if not m: continue
        em = m.get("em", m.get("EM", None))
        f1 = m.get("f1", m.get("F1", None))
        rr = m.get("refusal_rate", None)
        cc = m.get("citation_correctness", None)
        lines.append(f"| {r['noise']} | {r['level']} | "
                     f"{(em if em is not None else ''):.3f} | "
                     f"{(f1 if f1 is not None else ''):.3f} | "
                     f"{(rr if rr is not None else ''):.3f} | "
                     f"{(cc if cc is not None else ''):.3f} |")
    (out_root / "summary_noise.md").write_text("\n".join(lines))
    return report

# ------------------------- Main CLI -------------------------
def main():
    p = argparse.ArgumentParser(description="Ablation Suite: noise robustness and RM scaling curves.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # noise
    pn = sub.add_parser("noise", help="Run noise robustness ablations")
    pn.add_argument("--test", type=Path, required=True, help="Path to canonical test jsonl")
    pn.add_argument("--sft-checkpoint", type=str, required=True)
    pn.add_argument("--base-model", type=str, required=True)
    pn.add_argument("--attn-impl", type=str, default="eager")
    pn.add_argument("--buckets", type=str, default="2048,3072,4096,5120,6144,7168,7936")
    pn.add_argument("--max-new-tokens", type=int, default=256)
    pn.add_argument("--compare-baseline", action="store_true")
    pn.add_argument("--seed", type=int, default=42)
    pn.add_argument("--outdir", type=Path, default=Path("ablations/noise"))

    # scaling (RM)
    ps = sub.add_parser("scaling", help="Run preference data scaling curves for RM")
    ps.add_argument("--prefs", type=Path, required=True, help="preferences jsonl (chosen/rejected) from judging")
    ps.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    ps.add_argument("--sizes", type=str, default="128,256,512,768,1024,max")
    ps.add_argument("--epochs", type=int, default=3)
    ps.add_argument("--batch-size", type=int, default=8)
    ps.add_argument("--grad-accum", type=int, default=2)
    ps.add_argument("--lr", type=float, default=2e-5)
    ps.add_argument("--head-lr", type=float, default=1e-4)
    ps.add_argument("--warmup-ratio", type=float, default=0.03)
    ps.add_argument("--val-ratio", type=float, default=0.1)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--outdir", type=Path, default=Path("ablations/scaling/rm"))

    # all (noise + scaling)
    pa = sub.add_parser("all", help="Run noise and RM scaling with reasonable defaults.")
    pa.add_argument("--test", type=Path, required=True)
    pa.add_argument("--sft-checkpoint", type=str, required=True)
    pa.add_argument("--base-model", type=str, required=True)
    pa.add_argument("--attn-impl", type=str, default="eager")
    pa.add_argument("--buckets", type=str, default="2048,3072,4096,5120,6144,7168,7936")
    pa.add_argument("--max-new-tokens", type=int, default=256)
    pa.add_argument("--rm-prefs", type=Path, default=Path("prefs/preferences_cb68999425.jsonl"))
    pa.add_argument("--outdir", type=Path, default=Path("ablations"))

    args = p.parse_args()

    if args.cmd == "noise":
        run_noise_robustness(
            test_path=args.test,
            out_root=args.outdir,
            sft_ckpt=args.sft_checkpoint,
            base_model=args.base_model,
            attn_impl=args.attn_impl,
            buckets=args.buckets,
            max_new_tokens=args.max_new_tokens,
            compare_baseline=args.compare_baseline,
            seed=args.seed
        )

    elif args.cmd == "scaling":
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
        run_rm_scaling(
            prefs_path=args.prefs,
            out_dir=args.outdir,
            model_name=args.model,
            sizes=sizes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            head_lr=args.head_lr,
            warmup_ratio=args.warmup_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

    elif args.cmd == "all":
        root = args.outdir
        root.mkdir(parents=True, exist_ok=True)
        # 1) Noise
        run_noise_robustness(
            test_path=args.test,
            out_root=root / "noise",
            sft_ckpt=args.sft_checkpoint,
            base_model=args.base_model,
            attn_impl=args.attn_impl,
            buckets=args.buckets,
            max_new_tokens=args.max_new_tokens,
            compare_baseline=True,
            seed=42
        )
        # 2) RM scaling
        run_rm_scaling(
            prefs_path=args.rm_prefs,
            out_dir=root / "scaling" / "rm",
            model_name="microsoft/deberta-v3-base",
            sizes=["128","256","512","768","1024","max"],
            epochs=3, batch_size=8, grad_accum=2,
            lr=2e-5, head_lr=1e-4, warmup_ratio=0.03,
            val_ratio=0.1, seed=42
        )

if __name__ == "__main__":
    main()
