#!/usr/bin/env python3
"""
make_prefs.py — Two-phase preference set builder with caching.

Features
--------
- --phase {generate,judge,all}: split workflow so judge runs later by loading ONLY the judge.
- Separate quantization flags for generators and judge:
    * --use-8bit / --use-4bit             (applies to SFT/Base)
    * --judge-use-8bit / --judge-use-4bit (applies to Judge)
- Config-keyed caches for generations so we don’t redo work.
- Optional --cache-sig to reuse a specific cache; otherwise auto-fallback to latest valid cache for judge.
- Only pass generation kwargs valid for the chosen decoding mode to avoid
  "The following generation flags are not valid ..." spam.
- Output modes:
    * lite (default): compact pairs
    * rich: adds parsed fields (answer/citations/confidence/type) and dataset metadata

Outputs (under --output-dir, default `prefs/`):
- cache/<sig>/gens_sft.jsonl
- cache/<sig>/gens_base.jsonl
- preferences_<sig>.jsonl
- preference_stats_<sig>.json

Each line in gens_*.jsonl: {
  "id": int, "prompt": str, "model_tag": "sft"|"base", "gen_idx": int, "text": str, "model_name": str
}
Each line in preferences_*.jsonl (lite): {
  "id": int, "prompt": str, "chosen": str, "rejected": str,
  "win": "A"|"B", "scoreA": float, "scoreB": float, "margin": float,
  "judge": str
}
In rich mode, adds:
  "chosen_parsed": {...}, "rejected_parsed": {...},
  "question": str, "source": str|None, "support_spans": [...],
  "gold_answer": str|list|None, "n_chunks": int

Assumptions:
- Your data jsonl has at least one of: "prompt", "input", "question".
- If you have custom prompt templating, adapt `extract_prompt()` below to mirror your setup.
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

# Quiet down excessive TF / transformers chatter early
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


# --------------------------------
# Optional Output Validator import
# --------------------------------
_HAS_VALIDATOR = False
try:
    # Expecting your repo's prompt_schema.py to expose OutputValidator
    from prompt_schema import OutputValidator  # type: ignore
    _HAS_VALIDATOR = True
except Exception:
    # Fallback stub so --output-mode rich won't crash if validator is absent.
    class _Parsed:
        def __init__(self, text: str):
            self.answer = None
            self.citations = []
            self.confidence = None
            self.response_type = type("T", (), {"value": "unknown"})
            self.validation_errors = ["validator_missing"]
    class OutputValidator:  # type: ignore
        def __init__(self, strict_mode: bool = False):
            pass
        def validate_and_repair(self, text: str, n_evidence_chunks: int = 0):
            # Very light attempt to pull JSON-ish fields if present; otherwise empty.
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


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[dict]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # skip malformed lines silently
                continue
    return out


def write_jsonl(path: Path, rows: Iterable[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict]):
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-_.]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "x"


def config_signature(args: argparse.Namespace) -> str:
    """
    Capture ONLY knobs that change *generations* so the same sig can be reused for judging.
    """
    parts = dict(
        ms=args.max_samples,
        ng=args.n_generations,
        bs=args.batch_size,
        mxt=args.max_new_tokens,
        mit=args.max_input_tokens,
        t=args.temperature,
        tp=args.top_p,
        tk=args.top_k,
        sft=slug(args.sft_model or "none"),
        base=slug(args.base_model or "none"),
        data=slug(Path(args.data_path).name),
    )
    raw = json.dumps(parts, sort_keys=True)
    import hashlib
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return h


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=False,
        return_tensors=None
    )
    ids = enc["input_ids"][0] if isinstance(enc["input_ids"][0], list) else enc["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def extract_prompt(ex: dict) -> Optional[str]:
    """
    Adapt this to your schema if needed. We try common keys.
    """
    for k in ("prompt", "input", "question", "query"):
        if k in ex and ex[k]:
            return str(ex[k])
    if "context" in ex and "question" in ex:
        return f"{ex['question']}\n\nContext:\n{ex['context']}"
    return None


def build_generate_kwargs(args) -> dict:
    """
    Only include kwargs valid for the selected decoding mode.
    Avoids 'flags not valid' warnings from transformers.
    """
    do_sample = (args.temperature is not None) and (args.temperature > 0)
    gen_kwargs = dict(
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(do_sample),
        pad_token_id=None,  # set later from tokenizer
        eos_token_id=None,  # set later from tokenizer
    )
    if do_sample:
        if args.temperature is not None:
            gen_kwargs["temperature"] = float(args.temperature)
        if args.top_p is not None and 0 < args.top_p < 1.0:
            gen_kwargs["top_p"] = float(args.top_p)
        if args.top_k is not None and args.top_k > 0:
            gen_kwargs["top_k"] = int(args.top_k)
    return gen_kwargs


def device_str() -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "cuda"
    return "cpu"


def find_latest_valid_cache(cache_root: Path) -> Optional[Path]:
    """
    Returns newest cache dir that contains both gens files.
    """
    if not cache_root.exists():
        return None
    candidates = []
    for sub in cache_root.iterdir():
        if not sub.is_dir():
            continue
        sft = sub / "gens_sft.jsonl"
        base = sub / "gens_base.jsonl"
        if sft.exists() and base.exists():
            candidates.append((sub.stat().st_mtime, sub))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ----------------------------
# Model Loading
# ----------------------------

@dataclass
class LoadedModel:
    name: str
    model: AutoModelForCausalLM
    tok: AutoTokenizer


def quant_config(use_8bit: bool, use_4bit: bool):
    if not _BNB or (not use_8bit and not use_4bit):
        return None
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


def load_causal_lm(
    model_name: str,
    use_8bit: bool = False,
    use_4bit: bool = False,
) -> LoadedModel:
    qcfg = quant_config(use_8bit, use_4bit)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Some models have no pad token; default to eos
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=qcfg,
    )
    return LoadedModel(model_name, model, tok)


# ----------------------------
# Generation
# ----------------------------

def run_generate_for_model(
    lm: LoadedModel,
    prompts: List[Tuple[int, str]],
    args: argparse.Namespace,
    model_tag: str,
    out_path: Path,
):
    """
    prompts: list of (id, prompt_text)
    Writes jsonl rows to out_path incrementally (resume-safe).
    """
    ensure_dir(out_path.parent)
    # Resume support: load existing (id, gen_idx)
    done_keys = set()
    if out_path.exists():
        for r in read_jsonl(out_path):
            try:
                done_keys.add((int(r["id"]), int(r["gen_idx"])))
            except Exception:
                continue

    gen_kwargs = build_generate_kwargs(args)
    if lm.tok.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = lm.tok.pad_token_id
    if lm.tok.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = lm.tok.eos_token_id

    rows_to_append: List[dict] = []
    bs = max(1, int(args.batch_size))
    total = len(prompts) * int(args.n_generations)
    pbar = tqdm(total=total, desc=f"gen[{model_tag}]")
    pbar.update(len(done_keys))

    def flush():
        nonlocal rows_to_append
        if rows_to_append:
            append_jsonl(out_path, rows_to_append)
            rows_to_append = []

    # Pre-tokenize prompts with truncation to enforce context budget
    enc_prompts = []
    for pid, ptxt in prompts:
        truncated = truncate_to_tokens(lm.tok, ptxt, int(args.max_input_tokens))
        enc_prompts.append((pid, truncated))

    for g in range(int(args.n_generations)):
        for i in range(0, len(enc_prompts), bs):
            batch = enc_prompts[i : i + bs]
            if all(((pid, g) in done_keys) for pid, _ in batch):
                pbar.update(len(batch))
                continue

            texts = [b[1] for b in batch]
            toks = lm.tok(texts, return_tensors="pt", padding=True, truncation=True).to(lm.model.device)
            with torch.no_grad():
                out = lm.model.generate(
                    **toks,
                    **gen_kwargs,
                )
            # Slice off the prompt portion
            gen_texts = lm.tok.batch_decode(out[:, toks["input_ids"].shape[1]:], skip_special_tokens=True)

            for (pid, ptxt), gen_txt in zip(batch, gen_texts):
                key = (pid, g)
                if key in done_keys:
                    continue
                rows_to_append.append(
                    dict(
                        id=int(pid),
                        prompt=ptxt,
                        model_tag=model_tag,
                        model_name=lm.name,
                        gen_idx=int(g),
                        text=gen_txt.strip(),
                    )
                )
                if len(rows_to_append) >= 200:
                    flush()
            pbar.update(len(batch))
    flush()
    pbar.close()


# ----------------------------
# Judge
# ----------------------------

JUDGE_SYS_PROMPT = (
    "You are a strict, factual, and terse evaluator. "
    "Compare two candidate answers to the same user prompt and pick the better one.\n"
    "Rules:\n"
    " - Prioritize factual accuracy, directness, relevance, and safety.\n"
    " - Penalize hallucinations, vagueness, and over-refusal without cause.\n"
    " - If both are poor, choose the less bad one.\n"
    "Return ONLY a JSON object on one line with keys: "
    '{"winner":"A"|"B","scoreA":float,"scoreB":float,"reason":string}.'
)

def judge_build_prompt(prompt_text: str, ansA: str, ansB: str) -> str:
    return (
        f"<PROMPT>\n{prompt_text}\n</PROMPT>\n\n"
        f"<CANDIDATE_A>\n{ansA}\n</CANDIDATE_A>\n\n"
        f"<CANDIDATE_B>\n{ansB}\n</CANDIDATE_B>\n\n"
        "Respond with the required JSON only."
    )

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_judge_json(txt: str) -> Optional[dict]:
    m = _JSON_RE.search(txt)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict) and "winner" in obj and "scoreA" in obj and "scoreB" in obj:
            return obj
    except Exception:
        return None
    return None


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

    # Index by id -> candidate list
    from collections import defaultdict
    pool = defaultdict(lambda: {"prompt": None, "sft": [], "base": []})
    for r in sft_rows:
        try:
            rid = int(r["id"])
        except Exception:
            continue
        pool[rid]["prompt"] = r.get("prompt")
        pool[rid]["sft"].append((int(r.get("gen_idx", 0)), r.get("text", "")))
    for r in base_rows:
        try:
            rid = int(r["id"])
        except Exception:
            continue
        pool[rid]["prompt"] = r.get("prompt")
        pool[rid]["base"].append((int(r.get("gen_idx", 0)), r.get("text", "")))

    # Sort candidates by gen_idx for determinism
    for v in pool.values():
        v["sft"].sort(key=lambda x: x[0])
        v["base"].sort(key=lambda x: x[0])

    # Build cross-model pairs (n_generations^2 per sample)
    pairs = []
    for pid, v in pool.items():
        ptxt = v["prompt"]
        sft_cands = v["sft"][: int(args.n_generations)]
        base_cands = v["base"][: int(args.n_generations)]
        if not ptxt or not sft_cands or not base_cands:
            continue
        for i, sft_txt in sft_cands:
            for j, base_txt in base_cands:
                pairs.append((pid, ptxt, sft_txt, base_txt))

    if not pairs:
        raise SystemExit("No candidate pairs constructed — check caches and n_generations.")

    # Judge in batches; SFT is A, BASE is B (consistent)
    bs = max(1, int(args.judge_batch_size or args.batch_size))
    gen_kwargs = dict(
        max_new_tokens=192,
        do_sample=False,
        pad_token_id=judge.tok.pad_token_id,
        eos_token_id=judge.tok.eos_token_id
    )

    written = 0
    kept = 0
    margins: List[float] = []

    # Prepare writer
    ensure_dir(out_prefs.parent)
    if out_prefs.exists():
        out_prefs.unlink()
    if out_stats.exists():
        out_stats.unlink()

    ov = OutputValidator(strict_mode=False)

    for i in tqdm(range(0, len(pairs), bs), desc="judge"):
        chunk = pairs[i : i + bs]
        prompts_txt = []
        for pid, ptxt, sft_txt, base_txt in chunk:
            prompts_txt.append(f"<SYS>\n{JUDGE_SYS_PROMPT}\n</SYS>\n\n" + judge_build_prompt(ptxt, sft_txt, base_txt))
        toks = judge.tok(prompts_txt, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(judge.model.device)
        with torch.no_grad():
            out = judge.model.generate(**toks, **gen_kwargs)
        dec = judge.tok.batch_decode(out[:, toks["input_ids"].shape[1]:], skip_special_tokens=True)

        rows = []
        for (pid, ptxt, sft_txt, base_txt), raw in zip(chunk, dec):
            obj = parse_judge_json(raw) or {}
            winner = obj.get("winner")
            scoreA = float(obj.get("scoreA", 0.0))
            scoreB = float(obj.get("scoreB", 0.0))
            margin = abs(scoreA - scoreB)
            keep = margin >= float(args.min_margin)

            if winner not in ("A", "B"):
                # Fallback heuristic if judge fails: prefer longer answer slightly
                winner = "A" if len(sft_txt) >= len(base_txt) else "B"
                margin = 0.0
                keep = float(args.min_margin) <= 0.0

            row = dict(
                id=int(pid),
                prompt=ptxt,
                win=winner,
                scoreA=scoreA,
                scoreB=scoreB,
                margin=margin,
                judge=judge.name,
            )

            # Always include raw strings for chosen/rejected
            row["chosen"]   = sft_txt if winner == "A" else base_txt
            row["rejected"] = base_txt if winner == "A" else sft_txt

            if args.output_mode == "rich":
                meta = id_meta.get(str(pid), {})
                n_chunks = int(meta.get("n_chunks", 0))

                parsedA = ov.validate_and_repair(sft_txt, n_evidence_chunks=n_chunks)
                parsedB = ov.validate_and_repair(base_txt, n_evidence_chunks=n_chunks)

                def pack(parsed, text_is_A: bool):
                    return {
                        "text": sft_txt if text_is_A else base_txt,
                        "answer": getattr(parsed, "answer", None),
                        "citations": getattr(parsed, "citations", []),
                        "confidence": getattr(parsed, "confidence", None),
                        "type": getattr(getattr(parsed, "response_type", None), "value", "unknown"),
                        "errors": getattr(parsed, "validation_errors", []),
                    }

                chosen_parsed   = pack(parsedA if winner == "A" else parsedB, text_is_A=(winner == "A"))
                rejected_parsed = pack(parsedB if winner == "A" else parsedA, text_is_A=(winner == "A"))

                row["chosen_parsed"]   = chosen_parsed
                row["rejected_parsed"] = rejected_parsed
                row["question"]        = meta.get("question")
                row["source"]          = meta.get("source")
                row["support_spans"]   = meta.get("support_spans", [])
                row["gold_answer"]     = meta.get("gold_answer")
                row["n_chunks"]        = n_chunks

            written += 1
            if keep:
                rows.append(row)
                kept += 1
                margins.append(margin)

        if rows:
            append_jsonl(out_prefs, rows)

    # Stats
    stats = dict(
        total_pairs=len(pairs),
        judged=written,
        kept=kept,
        keep_rate=(kept / written) if written else 0.0,
        mean_margin=(sum(margins) / len(margins)) if margins else 0.0,
        judge_model=judge.name,
        output_mode=args.output_mode,
        has_validator=_HAS_VALIDATOR,
        phase="judge",
    )
    with out_stats.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("\nJudge summary:")
    print(json.dumps(stats, indent=2))


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    # Data / IO
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="prefs")
    p.add_argument("--max-samples", type=int, default=600)
    p.add_argument("--per-source-cap", type=int, default=2000)  # reserved for future use

    # Models
    p.add_argument("--sft-model", type=str, default=None)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--judge-model", type=str, default=None)

    # Quantization (generators)
    p.add_argument("--use-8bit", action="store_true")
    p.add_argument("--use-4bit", action="store_true")
    # Quantization (judge)
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

    # Output flavor
    p.add_argument("--output-mode", choices=["lite", "rich"], default="lite",
                   help="Emit compact pairs (lite) or include parsed chosen/rejected + dataset metadata (rich).")

    # Cache control
    p.add_argument("--cache-sig", type=str, default=None,
                   help="Override generation cache signature to use (for judge resume).")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    set_seed(int(args.seed))

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    sig = config_signature(args)
    cache_root = out_dir / "cache"

    # Resolve cache directory:
    # - If user provided --cache-sig, use that.
    # - Else use computed signature.
    # - In judge phase, if computed sig missing, fallback to latest valid cache.
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

    # Load and sample data (and id->meta for rich)
    data_path = Path(args.data_path)
    data = read_jsonl(data_path)
    if not data:
        raise SystemExit(f"No data found at {args.data_path}")

    prompts: List[Tuple[int, str]] = []
    id_meta: Dict[str, dict] = {}
    for i, ex in enumerate(data):
        pr = extract_prompt(ex)
        if pr:
            prompts.append((i, pr))
        # Build id -> meta (for rich output)
        sid = str(ex.get("id", i))
        id_meta[sid] = {
            "question": ex.get("question"),
            # many pipelines put source in metadata; adapt if different
            "source": ((ex.get("metadata") or {}).get("source") if isinstance(ex.get("metadata", None), dict) else None),
            "support_spans": ex.get("support_spans", []),
            "n_chunks": len(ex.get("context_chunks", [])) if isinstance(ex.get("context_chunks", None), list) else 0,
            "gold_answer": ex.get("answer"),
        }
        if len(prompts) >= int(args.max_samples):
            break

    if not prompts:
        raise SystemExit("Could not find any prompts in the dataset. Adapt extract_prompt().")

    print("============================================================")
    print("Preference Builder")
    print("============================================================")
    print(f"Device: {device_str()} | CUDA: {torch.cuda.is_available()}")
    print(f"Signature (computed): {sig}")
    if args.cache_sig:
        print(f"Cache override (--cache-sig): {args.cache_sig}")
    print(f"Using cache dir: {cache_dir}")
    print(f"Data: {args.data_path} | Prompts: {len(prompts)}")
    print(f"Output dir: {out_dir}")
    print(f"Phase: {args.phase}")
    print("============================================================\n")

    # ---------------- Phase: GENERATE ----------------
    if args.phase in ("generate", "all"):
        if not args.sft_model or not args.base_model:
            raise SystemExit("Generation phase requires --sft-model and --base-model.")

        ensure_dir(cache_dir)
        need_sft = True
        need_base = True
        if args.skip_if_cached and gens_sft_path.exists():
            sft_rows = read_jsonl(gens_sft_path)
            if len(sft_rows) >= int(args.max_samples) * int(args.n_generations) * 0.95:
                print(f"[cache] Found SFT generations: {gens_sft_path}")
                need_sft = False
        if args.skip_if_cached and gens_base_path.exists():
            base_rows = read_jsonl(gens_base_path)
            if len(base_rows) >= int(args.max_samples) * int(args.n_generations) * 0.95:
                print(f"[cache] Found BASE generations: {gens_base_path}")
                need_base = False

        if need_sft:
            print(f"Loading SFT: {args.sft_model} (8bit={args.use_8bit}, 4bit={args.use_4bit})")
            sft_lm = load_causal_lm(args.sft_model, use_8bit=args.use_8bit, use_4bit=args.use_4bit)
            run_generate_for_model(sft_lm, prompts, args, "sft", gens_sft_path)
            del sft_lm
            torch.cuda.empty_cache()

        if need_base:
            print(f"Loading BASE: {args.base_model} (8bit={args.use_8bit}, 4bit={args.use_4bit})")
            base_lm = load_causal_lm(args.base_model, use_8bit=args.use_8bit, use_4bit=args.use_4bit)
            run_generate_for_model(base_lm, prompts, args, "base", gens_base_path)
            del base_lm
            torch.cuda.empty_cache()

        print("\n✅ Generation phase complete.")
        try:
            sz_sft = gens_sft_path.stat().st_size / 1e6
            sz_base = gens_base_path.stat().st_size / 1e6
            print(f"  SFT cache : {gens_sft_path}  ({sz_sft:.1f} MB)")
            print(f"  BASE cache: {gens_base_path}  ({sz_base:.1f} MB)\n")
        except Exception:
            pass

    # If we are *only* judging and the computed cache doesn't exist, fallback to latest valid cache
    if args.phase in ("judge", "all"):
        if (not gens_sft_path.exists() or not gens_base_path.exists()) and not (args.phase == "all"):
            # attempt fallback discovery
            fallback = find_latest_valid_cache(cache_root)
            if fallback and fallback != cache_dir:
                print(f"[cache] Computed cache missing. Falling back to latest cache: {fallback.name}")
                cache_dir = fallback
                gens_sft_path = cache_dir / "gens_sft.jsonl"
                gens_base_path = cache_dir / "gens_base.jsonl"
                # also update output signature so filenames reflect the cache actually used
                resolved_sig = fallback.name
                prefs_path = out_dir / f"preferences_{resolved_sig}.jsonl"
                stats_path = out_dir / f"preference_stats_{resolved_sig}.json"

    # ---------------- Phase: JUDGE ----------------
    if args.phase in ("judge", "all"):
        if not args.judge_model:
            if args.phase == "judge":
                raise SystemExit("Judge phase requires --judge-model.")
            else:
                print("ℹ No --judge-model provided; skipping judge phase in 'all'.")
                print("Done.")
                return

        # Load ONLY the judge (SFT/Base are NOT loaded here)
        print(f"Loading JUDGE: {args.judge_model} "
              f"(8bit={args.judge_use_8bit}, 4bit={args.judge_use_4bit})")
        judge_lm = load_causal_lm(
            args.judge_model,
            use_8bit=args.judge_use_8bit,
            use_4bit=args.judge_use_4bit,
        )
        run_judge_only(
            judge=judge_lm,
            gens_sft_path=gens_sft_path,
            gens_base_path=gens_base_path,
            args=args,
            out_prefs=prefs_path,
            out_stats=stats_path,
            id_meta=id_meta,
        )
        del judge_lm
        torch.cuda.empty_cache()
        print("\n✅ Judge phase complete.")
        print(f"  Preferences : {prefs_path}")
        print(f"  Stats       : {stats_path}\n")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
