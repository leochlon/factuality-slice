#!/usr/bin/env python3
"""
make_prefs.py — Two-phase preference set builder with caching (FIXED)
- Robust judge JSON parsing & score normalization
- Correct ID alignment between generation and judging
- Uses PromptTemplates to ensure evidence-grounded prompts for both candidates
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm

# Quiet tokenizer spam
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass

# BitsAndBytes optional
try:
    from transformers import BitsAndBytesConfig
    _BNB = True
except Exception:
    _BNB = False

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Optional OutputValidator (graceful fallback) ----
_HAS_VALIDATOR = False
try:
    from sft.prompt_schema import OutputValidator  # real validator if present
    _HAS_VALIDATOR = True
except Exception:
    class _Parsed:
        def __init__(self, text: str):
            self.answer = None
            self.citations = []
            self.confidence = None
            self.response_type = type("T", (), {"value": "unknown"})
            self.validation_errors = ["validator_missing"]
    class OutputValidator:  # shim
        def __init__(self, strict_mode: bool = False): pass
        def validate_and_repair(self, text: str, n_evidence_chunks: int = 0):
            try:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                obj = json.loads(m.group(0)) if m else {}
            except Exception:
                obj = {}
            p = _Parsed(text)
            p.answer = obj.get("answer")
            p.citations = obj.get("citations", []) if isinstance(obj.get("citations"), list) else []
            p.confidence = obj.get("confidence")
            return p

# ---- Utilities ----
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path) -> List[dict]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def append_jsonl(path: Path, rows: Iterable[dict]):
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-_.]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "x"

def config_signature(args: argparse.Namespace) -> str:
    parts = dict(
        ms=args.max_samples, ng=args.n_generations, bs=args.batch_size,
        mxt=args.max_new_tokens, mit=args.max_input_tokens,
        t=args.temperature, tp=args.top_p, tk=args.top_k,
        sft=slug(args.sft_model or "none"),
        base=slug(args.base_model or "none"),
        data=slug(Path(args.data_path).name),
    )
    raw = json.dumps(parts, sort_keys=True).encode("utf-8")
    import hashlib
    return hashlib.sha1(raw).hexdigest()[:10]

def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    enc = tokenizer(text, truncation=True, max_length=max_tokens,
                    add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]
    if isinstance(ids[0], list): ids = ids[0]
    return tokenizer.decode(ids, skip_special_tokens=True)

def device_str() -> str:
    if torch.cuda.is_available():
        try: return torch.cuda.get_device_name(0)
        except Exception: return "cuda"
    return "cpu"

def find_latest_valid_cache(cache_root: Path) -> Optional[Path]:
    if not cache_root.exists(): return None
    cands = []
    for sub in cache_root.iterdir():
        if not sub.is_dir(): continue
        if (sub/"gens_sft.jsonl").exists() and (sub/"gens_base.jsonl").exists():
            cands.append((sub.stat().st_mtime, sub))
    if not cands: return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

# ---- Model Loading ----
@dataclass
class LoadedModel:
    name: str
    model: AutoModelForCausalLM
    tok: AutoTokenizer

def quant_config(use_8bit: bool, use_4bit: bool):
    if not _BNB or (not use_8bit and not use_4bit): return None
    if use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if use_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

def load_causal_lm(model_name: str, use_8bit: bool=False, use_4bit: bool=False) -> LoadedModel:
    qcfg = quant_config(use_8bit, use_4bit)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.truncation_side = "left"  # keep end of prompt on generation
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=qcfg,
    )
    return LoadedModel(model_name, model, tok)

# ---- Generation ----
def build_generate_kwargs(args) -> dict:
    do_sample = (args.temperature is not None) and (args.temperature > 0)
    g = dict(max_new_tokens=int(args.max_new_tokens),
             do_sample=bool(do_sample),
             pad_token_id=None, eos_token_id=None)
    if do_sample:
        if args.temperature is not None: g["temperature"] = float(args.temperature)
        if args.top_p is not None and 0 < args.top_p < 1.0: g["top_p"] = float(args.top_p)
        if args.top_k is not None and args.top_k > 0: g["top_k"] = int(args.top_k)
    return g

def run_generate_for_model(
    lm: LoadedModel,
    prompts: List[Tuple[int, str]],
    args: argparse.Namespace,
    model_tag: str,
    out_path: Path,
):
    ensure_dir(out_path.parent)
    done = set()
    if out_path.exists():
        for r in read_jsonl(out_path):
            try: done.add((int(r["id"]), int(r["gen_idx"])))
            except Exception: continue

    gen_kwargs = build_generate_kwargs(args)
    if lm.tok.pad_token_id is not None: gen_kwargs["pad_token_id"] = lm.tok.pad_token_id
    if lm.tok.eos_token_id is not None: gen_kwargs["eos_token_id"] = lm.tok.eos_token_id

    rows: List[dict] = []
    bs = max(1, int(args.batch_size))
    total = len(prompts) * int(args.n_generations)
    pbar = tqdm(total=total, desc=f"gen[{model_tag}]"); pbar.update(len(done))

    enc_prompts = [(pid, truncate_to_tokens(lm.tok, ptxt, int(args.max_input_tokens)))
                   for pid, ptxt in prompts]

    def flush():
        nonlocal rows
        if rows:
            append_jsonl(out_path, rows); rows = []

    for g in range(int(args.n_generations)):
        for i in range(0, len(enc_prompts), bs):
            batch = enc_prompts[i:i+bs]
            if all(((pid, g) in done) for pid, _ in batch):
                pbar.update(len(batch)); continue
            texts = [t for _, t in batch]
            toks = lm.tok(texts, return_tensors="pt", padding=True, truncation=True).to(lm.model.device)
            with torch.no_grad():
                out = lm.model.generate(**toks, **gen_kwargs)
            dec = lm.tok.batch_decode(out[:, toks["input_ids"].shape[1]:], skip_special_tokens=True)
            for (pid, _), txt in zip(batch, dec):
                if (pid, g) in done: continue
                rows.append(dict(id=int(pid), prompt=texts[0] if False else None,  # omit duplicate prompt
                                 model_tag=model_tag, model_name=lm.name, gen_idx=int(g), text=txt.strip()))
                if len(rows) >= 200: flush()
            pbar.update(len(batch))
    flush(); pbar.close()

# ---- Judge ----
JUDGE_SYS_PROMPT = (
    "You are a strict factuality judge evaluating evidence-grounded answers.\n"
    "Priorities: (1) factual correctness from EVIDENCE; (2) correct citations; "
    "(3) refuse when evidence is insufficient; (4) calibrated confidence.\n"
    "Return a single-line JSON: {\"winner\":\"A|B\",\"scoreA\":0.00-1.00,\"scoreB\":0.00-1.00,\"reason\":\"...\"}"
)

def _extract_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1: return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: return text[start:i+1]
    return None

def _coerce_json(s: str) -> Optional[dict]:
    if not s: return None
    try: return json.loads(s)
    except Exception: pass
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", s)
        if fixed.count('"') < 2 and "'" in fixed: fixed = fixed.replace("'", '"')
        return json.loads(fixed)
    except Exception: return None

def safe_float(x, default=0.0):
    try: return float(x)
    except Exception: return default

def normalize_score(v: float) -> float:
    if v > 1.0:
        if v <= 10: return v / 10.0
        if v <= 100: return v / 100.0
        return 1.0
    return max(0.0, min(1.0, v))

def parse_judge_json(txt: str) -> Optional[dict]:
    obj = _coerce_json(_extract_json(txt))
    if not isinstance(obj, dict):
        m = re.search(r"\b(A|B)\b", txt.upper())
        if not m: return None
        w = m.group(1)
        return {"winner": w, "scoreA": 1.0 if w=="A" else 0.0, "scoreB": 1.0 if w=="B" else 0.0, "reason": ""}
    # winner
    winner = (obj.get("winner") or obj.get("choice") or obj.get("selected") or obj.get("verdict") or "").strip().upper()
    if winner not in ("A","B"):
        m = re.search(r"\b(A|B)\b", txt.upper()); winner = m.group(1) if m else None
    if winner not in ("A","B"): return None
    # scores
    scoreA = obj.get("scoreA"); scoreB = obj.get("scoreB")
    scores = obj.get("scores") or obj.get("score") or {}
    if scoreA is None and isinstance(scores, dict):
        scoreA = scores.get("A") or scores.get("a") or scores.get(0)
    if scoreB is None and isinstance(scores, dict):
        scoreB = scores.get("B") or scores.get("b") or scores.get(1)
    margin = obj.get("margin") or obj.get("gap") or obj.get("delta")
    if (scoreA is None or scoreB is None) and margin is not None:
        m = normalize_score(safe_float(margin, 0.0))
        hi, lo = min(1.0, 0.5 + m/2), None
        lo = 1.0 - hi
        scoreA, scoreB = (hi, lo) if winner == "A" else (lo, hi)
    if scoreA is None or scoreB is None:
        scoreA = 1.0 if winner=="A" else 0.0
        scoreB = 1.0 - scoreA
    scoreA = normalize_score(safe_float(scoreA, 0.5))
    scoreB = normalize_score(safe_float(scoreB, 0.5))
    # tie detection
    if isinstance(obj.get("tie"), bool) and obj["tie"]: return None
    if re.search(r"\b(tie|equal|cannot\s+decide)\b", txt, re.I): return None
    return {"winner": winner, "scoreA": scoreA, "scoreB": scoreB,
            "reason": obj.get("reason") or obj.get("rationale") or ""}

def format_evidence(chunks):
    lines = []
    for i, ch in enumerate(chunks or []):
        if isinstance(ch, dict): s = ch.get("text") or ch.get("chunk") or ""
        else: s = str(ch)
        lines.append(f"[{i}] {s}")
    return "\n".join(lines) if lines else "[none]"

def judge_build_prompt(question: str, evidence_chunks: List, ansA: str, ansB: str) -> str:
    ev = format_evidence(evidence_chunks)
    return (
        f"<SYS>\n{JUDGE_SYS_PROMPT}\n</SYS>\n\n"
        f"<EVIDENCE>\n{ev}\n</EVIDENCE>\n\n"
        f"<QUESTION>\n{question}\n</QUESTION>\n\n"
        f"<CANDIDATE_A>\n{ansA}\n</CANDIDATE_A>\n\n"
        f"<CANDIDATE_B>\n{ansB}\n</CANDIDATE_B>\n\n"
        "Answer with the required JSON only."
    )

def run_judge_only(
    judge: LoadedModel,
    gens_sft_path: Path,
    gens_base_path: Path,
    args: argparse.Namespace,
    out_prefs: Path,
    out_stats: Path,
    id_meta: Dict[str, dict],
):
    sft_rows = read_jsonl(gens_sft_path)
    base_rows = read_jsonl(gens_base_path)
    if not sft_rows or not base_rows:
        raise SystemExit("No cached generations found; run with --phase generate first.")

    # id -> candidates (align by enumerated pid)
    from collections import defaultdict
    pool = defaultdict(lambda: {"sft": [], "base": []})
    for r in sft_rows:
        try: rid = int(r["id"])
        except Exception: continue
        pool[rid]["sft"].append((int(r.get("gen_idx", 0)), r.get("text", "")))
    for r in base_rows:
        try: rid = int(r["id"])
        except Exception: continue
        pool[rid]["base"].append((int(r.get("gen_idx", 0)), r.get("text", "")))

    for v in pool.values():
        v["sft"].sort(key=lambda x: x[0])
        v["base"].sort(key=lambda x: x[0])

    # cross pairs
    pairs = []
    for pid, v in pool.items():
        sft_cands = v["sft"][: int(args.n_generations)]
        base_cands = v["base"][: int(args.n_generations)]
        if not sft_cands or not base_cands: continue
        meta = id_meta.get(str(pid), {})
        q = meta.get("question") or ""
        chunks = meta.get("context_chunks", [])
        for (i, sft_txt) in sft_cands:
            for (j, base_txt) in base_cands:
                pairs.append((pid, q, chunks, sft_txt, base_txt))

    if not pairs:
        raise SystemExit("No candidate pairs constructed — check caches and n_generations.")

    bs = max(1, int(args.judge_batch_size or args.batch_size))
    gen_kwargs = dict(max_new_tokens=192, do_sample=False,
                      pad_token_id=judge.tok.pad_token_id,
                      eos_token_id=judge.tok.eos_token_id)

    ensure_dir(out_prefs.parent)
    if out_prefs.exists(): out_prefs.unlink()
    if out_stats.exists(): out_stats.unlink()

    ov = OutputValidator(strict_mode=False)
    written = kept = skipped = parse_failures = 0
    margins: List[float] = []

    for i in tqdm(range(0, len(pairs), bs), desc="judge"):
        chunk = pairs[i:i+bs]
        prompts_txt = [
            judge_build_prompt(q, chunks, a, b) for _, q, chunks, a, b in chunk
        ]
        toks = judge.tok(prompts_txt, return_tensors="pt", padding=True,
                         truncation=True, max_length=4096).to(judge.model.device)
        with torch.no_grad():
            out = judge.model.generate(**toks, **gen_kwargs)
        dec = judge.tok.batch_decode(out[:, toks["input_ids"].shape[1]:], skip_special_tokens=True)

        rows = []
        for (pid, q, chunks, sft_txt, base_txt), raw in zip(chunk, dec):
            written += 1
            obj = parse_judge_json(raw)
            if obj is None:
                parse_failures += 1; skipped += 1; continue

            scoreA = normalize_score(safe_float(obj.get("scoreA"), 0.5))
            scoreB = normalize_score(safe_float(obj.get("scoreB"), 0.5))
            margin = abs(scoreA - scoreB)
            winner = obj.get("winner")
            if winner not in ("A","B"):
                winner = "A" if scoreA >= scoreB else "B"

            if margin < float(args.min_margin):
                skipped += 1; continue

            # Winner “A” is SFT candidate by construction
            row = dict(
                id=int(pid),
                win=winner,
                scoreA=scoreA, scoreB=scoreB, margin=margin,
                judge=judge.name,
                prompt={"question": q, "context_chunks": chunks},
                chosen= sft_txt if winner == "A" else base_txt,
                rejected= base_txt if winner == "A" else sft_txt,
            )

            if args.output_mode == "rich":
                n_chunks = len(chunks) if isinstance(chunks, list) else 0
                parsedA = ov.validate_and_repair(sft_txt, n_evidence_chunks=n_chunks)
                parsedB = ov.validate_and_repair(base_txt, n_evidence_chunks=n_chunks)

                def pack(parsed, text):
                    return {
                        "text": text,
                        "answer": getattr(parsed, "answer", None),
                        "citations": getattr(parsed, "citations", []),
                        "confidence": getattr(parsed, "confidence", None),
                        "type": getattr(getattr(parsed, "response_type", None), "value", "unknown"),
                        "errors": getattr(parsed, "validation_errors", []),
                    }

                row["chosen_parsed"]   = pack(parsedA if winner=="A" else parsedB,
                                              sft_txt if winner=="A" else base_txt)
                row["rejected_parsed"] = pack(parsedB if winner=="A" else parsedA,
                                              base_txt if winner=="A" else sft_txt)

                # carry through a little metadata
                meta = id_meta.get(str(pid), {})
                row["question"] = meta.get("question")
                row["source"] = meta.get("source")
                row["gold_answer"] = meta.get("gold_answer")
                row["n_chunks"] = n_chunks

            rows.append(row)
            kept += 1; margins.append(margin)

        if rows:
            append_jsonl(out_prefs, rows)

    stats = dict(
        total_pairs=len(pairs), attempted=written, kept=kept, skipped=skipped,
        keep_rate=(kept/written) if written else 0.0,
        mean_margin=(sum(margins)/len(margins)) if margins else 0.0,
        parse_failures=parse_failures,
        parse_failure_rate=(parse_failures/written) if written else 0.0,
        judge_model=judge.name, output_mode=args.output_mode,
        has_validator=_HAS_VALIDATOR, phase="judge",
    )
    with out_stats.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("\nJudge summary:"); print(json.dumps(stats, indent=2))

# ---- Main ----
def main():
    p = argparse.ArgumentParser()
    # Data / IO
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="prefs")
    p.add_argument("--max-samples", type=int, default=600)
    p.add_argument("--per-source-cap", type=int, default=2000)

    # Models
    p.add_argument("--sft-model", type=str, default=None)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--judge-model", type=str, default=None)

    # Quantization
    p.add_argument("--use-8bit", action="store_true")
    p.add_argument("--use-4bit", action="store_true")
    p.add_argument("--judge-use-8bit", action="store_true")
    p.add_argument("--judge-use-4bit", action="store_true")

    # Generation config
    p.add_argument("--n-generations", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--max-input-tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)

    # Judge / filtering
    p.add_argument("--min-margin", type=float, default=0.12)
    p.add_argument("--judge-batch-size", type=int, default=None)

    # Flow control
    p.add_argument("--phase", choices=["generate", "judge", "all"], default="all")
    p.add_argument("--skip-if-cached", action="store_true", default=True)
    p.add_argument("--output-mode", choices=["lite", "rich"], default="lite")
    p.add_argument("--cache-sig", type=str, default=None)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(int(args.seed))

    out_dir = Path(args.output_dir); ensure_dir(out_dir)
    sig = config_signature(args)
    cache_root = out_dir / "cache"

    # Resolve cache dir
    if args.cache_sig:
        cache_dir = cache_root / args.cache_sig
        ensure_dir(cache_dir)
        resolved_sig = args.cache_sig
    else:
        cache_dir = cache_root / sig
        resolved_sig = sig

    gens_sft_path = cache_dir / "gens_sft.jsonl"
    gens_base_path = cache_dir / "gens_base.jsonl"
    prefs_path = out_dir / f"preferences_{resolved_sig}.jsonl"
    stats_path = out_dir / f"preference_stats_{resolved_sig}.json"

    # Load data and build prompts with evidence
    data_path = Path(args.data_path)
    data = read_jsonl(data_path)
    if not data: raise SystemExit(f"No data found at {args.data_path}")

    from sft.prompt_schema import PromptTemplates
    pt = PromptTemplates()

    prompts: List[Tuple[int, str]] = []
    id_meta: Dict[str, dict] = {}

    for i, ex in enumerate(data):
        # Always construct the same SFT-style prompt so SFT & BASE see identical evidence
        if ex.get("context_chunks"):
            pr = pt.get_sft_prompt(
                question=ex.get("question", ""),
                evidence_chunks=ex["context_chunks"],
                template_name="default"
            )
        else:
            # minimal fallback
            q = ex.get("question") or ex.get("prompt") or ex.get("input") or ""
            pr = f"## Question:\n{q}".strip()
        prompts.append((i, pr))
        id_meta[str(i)] = {  # <-- key by enumerated pid to align with gens
            "orig_id": ex.get("id"),
            "question": ex.get("question"),
            "source": (ex.get("metadata") or {}).get("source") if isinstance(ex.get("metadata"), dict) else None,
            "gold_answer": ex.get("answer"),
            "context_chunks": ex.get("context_chunks", []),
        }
        if len(prompts) >= int(args.max_samples): break

    print("============================================================")
    print("Preference Builder")
    print("============================================================")
    print(f"Device: {device_str()} | CUDA: {torch.cuda.is_available()}")
    print(f"Signature: {sig}")
    if args.cache_sig: print(f"Cache override: {args.cache_sig}")
    print(f"Using cache dir: {cache_dir}")
    print(f"Data: {args.data_path} | Prompts: {len(prompts)}")
    print(f"Output dir: {out_dir}")
    print(f"Phase: {args.phase}")
    print("============================================================\n")

    # Phase: generate
    if args.phase in ("generate", "all"):
        if not args.sft_model or not args.base_model:
            raise SystemExit("Generation phase requires --sft-model and --base-model.")
        ensure_dir(cache_dir)
        need_sft = not (args.skip_if_cached and gens_sft_path.exists())
        need_base = not (args.skip_if_cached and gens_base_path.exists())

        if need_sft:
            print(f"Loading SFT: {args.sft_model} (8bit={args.use_8bit}, 4bit={args.use_4bit})")
            sft_lm = load_causal_lm(args.sft_model, use_8bit=args.use_8bit, use_4bit=args.use_4bit)
            run_generate_for_model(sft_lm, prompts, args, "sft", gens_sft_path)
            del sft_lm; torch.cuda.empty_cache()
        else:
            print(f"[cache] Found SFT generations: {gens_sft_path}")

        if need_base:
            print(f"Loading BASE: {args.base_model} (8bit={args.use_8bit}, 4bit={args.use_4bit})")
            base_lm = load_causal_lm(args.base_model, use_8bit=args.use_8bit, use_4bit=args.use_4bit)
            run_generate_for_model(base_lm, prompts, args, "base", gens_base_path)
            del base_lm; torch.cuda.empty_cache()
        else:
            print(f"[cache] Found BASE generations: {gens_base_path}")

        print("\n✅ Generation phase complete.")
        try:
            print(f"  SFT cache : {gens_sft_path}  ({gens_sft_path.stat().st_size/1e6:.1f} MB)")
            print(f"  BASE cache: {gens_base_path}  ({gens_base_path.stat().st_size/1e6:.1f} MB)\n")
        except Exception:
            pass

    # Allow fallback to most recent cache in judge-only
    if args.phase in ("judge", "all"):
        if (not gens_sft_path.exists() or not gens_base_path.exists()) and args.phase == "judge":
            fallback = find_latest_valid_cache(cache_root)
            if fallback and fallback != cache_dir:
                print(f"[cache] Missing cache; falling back to: {fallback.name}")
                cache_dir = fallback
                gens_sft_path = cache_dir / "gens_sft.jsonl"
                gens_base_path = cache_dir / "gens_base.jsonl"
                resolved_sig = fallback.name
                prefs_path = out_dir / f"preferences_{resolved_sig}.jsonl"
                stats_path = out_dir / f"preference_stats_{resolved_sig}.json"

    # Phase: judge
    if args.phase in ("judge", "all"):
        if not args.judge_model:
            if args.phase == "judge":
                raise SystemExit("Judge phase requires --judge-model.")
            else:
                print("ℹ No --judge-model provided; skipping judge phase.")
                print("Done."); return

        print(f"Loading JUDGE: {args.judge_model} (8bit={args.judge_use_8bit}, 4bit={args.judge_use_4bit})")
        judge_lm = load_causal_lm(args.judge_model, use_8bit=args.judge_use_8bit, use_4bit=args.judge_use_4bit)
        run_judge_only(
            judge=judge_lm,
            gens_sft_path=gens_sft_path,
            gens_base_path=gens_base_path,
            args=args,
            out_prefs=prefs_path,
            out_stats=stats_path,
            id_meta=id_meta,
        )
        del judge_lm; torch.cuda.empty_cache()
        print("\n✅ Judge phase complete.")
        print(f"  Preferences : {prefs_path}")
        print(f"  Stats       : {stats_path}\n")

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user."); sys.exit(130)
