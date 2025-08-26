#!/usr/bin/env python3
# evaluate_sft_fast_robust.py
"""
Evaluation for an SFT model with configurable speed/robustness trade-offs.

"""

from __future__ import annotations

import json
import logging
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from prompt_schema import PromptTemplates, OutputValidator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("eval_sft")

# ---- Prefer efficient attention kernels (Flash / SDPA) ----
# Hint for PyTorch SDPA backends; harmless if unsupported.
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
except Exception:
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass

# TF32 OK on Ampere+ (A100)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def read_jsonl(p: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_rows and len(rows) >= max_rows:
                    break
    return rows


def _normalize_tokens(s: str) -> List[str]:
    import re, string
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s.split()


def em(a: str, b: str) -> float:
    return 1.0 if _normalize_tokens(a) == _normalize_tokens(b) else 0.0


def f1(a: str, b: str) -> float:
    A, B = _normalize_tokens(a), _normalize_tokens(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    from collections import Counter
    cA, cB = Counter(A), Counter(B)
    num = sum((cA & cB).values())
    if num == 0:
        return 0.0
    prec = num / max(1, sum(cA.values()))
    rec = num / max(1, sum(cB.values()))
    return 2 * prec * rec / (prec + rec)


def _token_length(tok: AutoTokenizer, text: str) -> int:
    # Fast, approximate token count; suitable for batching
    return len(tok.encode(text, add_special_tokens=True))


def _parse_buckets(arg: Optional[str]) -> List[int]:
    if not arg:
        # reasonable defaults up to ~8k (cap will still be enforced)
        return [1024, 2048, 3072, 4096, 5120, 6144, 7168, 7936]
    out: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            raise ValueError(f"Invalid bucket value: {part}")
    out = sorted(set(x for x in out if x > 0))
    if not out:
        raise ValueError("Empty --buckets after parsing.")
    return out


def _choose_bucket(length: int, buckets: List[int], prompt_cap: int) -> int:
    length = min(length, prompt_cap)
    for b in buckets:
        if b >= length:
            return min(b, prompt_cap)
    return prompt_cap


def run_model(
    model_path: str,
    data: List[Dict[str, Any]],
    template: str = "default",
    max_new_tokens: int = 256,
    batch_size: int = 4,
    temperature: float = 0.0,
    top_p: float = 1.0,
    prompt_cap: Optional[int] = None,   # if None: ctx - max_new_tokens
    fast: bool = False,                  # bucketed shapes + (optional) compile
    buckets: Optional[List[int]] = None, # used when fast=True
    attn_impl: str = "auto",             # "auto" | "sdpa" | "flash2"
) -> Tuple[List[Dict[str, Any]], AutoModelForCausalLM, AutoTokenizer]:
    # ---- Tokenizer ----
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"      # decoder-only best practice
    tok.truncation_side = "left"

    # ---- Model ----
    model_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if attn_impl == "flash2":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif attn_impl == "sdpa":
        model_kwargs["attn_implementation"] = "sdpa"
    # else: let Transformers pick

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    # Optional compile for speed (recommended when fast=True and buckets give stable shapes)
    compiled = False
    if fast:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
            compiled = True
            log.info("torch.compile enabled (fast mode).")
        except Exception as e:
            log.warning(f"torch.compile unavailable or failed, continuing without compile: {e}")

    # ---- Context window & prompt cap ----
    ctx = getattr(model.config, "max_position_embeddings", None)
    try:
        ctx = int(ctx) if ctx is not None else 8192
    except Exception:
        ctx = 8192
    if ctx <= 0 or ctx > 262144:
        ctx = 8192
    tok.model_max_length = ctx

    if prompt_cap is None:
        prompt_cap = max(16, ctx - int(max_new_tokens))
    else:
        prompt_cap = max(16, min(int(prompt_cap), ctx - int(max_new_tokens)))

    # ---- Build prompts & lengths, then sort by length to reduce padding ----
    pt = PromptTemplates()
    prompts = [pt.get_sft_prompt(ex["question"], ex["context_chunks"], template_name=template) for ex in data]

    lengths = [_token_length(tok, p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    inv_order = [0] * len(order)
    for new_pos, orig_idx in enumerate(order):
        inv_order[orig_idx] = new_pos

    sorted_prompts = [prompts[i] for i in order]
    sorted_data = [data[i] for i in order]
    sorted_lengths = [lengths[i] for i in order]

    results_by_sorted_idx: List[Optional[Dict[str, Any]]] = [None] * len(sorted_prompts)
    val = OutputValidator(strict_mode=False)

    # Decide generate() wrapping:
    # - In safe mode (fast=False), disable compilation around generate() to avoid recompile_limit errors.
    # - In fast mode, we rely on bucketed shapes (+optional compile), so we call generate() directly.
    def _gen_call(m, **kwargs):
        return m.generate(**kwargs)

    if not fast:
        try:
            from torch import compiler as _tcompiler
            @_tcompiler.disable
            def _safe_generate(m, **kwargs):
                return m.generate(**kwargs)
            _gen_call = _safe_generate
            log.info("Compiler disabled around generate() (robust mode).")
        except Exception:
            log.info("Could not disable compiler; continuing without wrapper.")

    do_sample = (temperature is not None) and (float(temperature) > 0.0)

    # Prepare buckets if fast mode
    if fast:
        buckets = buckets or [1024, 2048, 3072, 4096, 5120, 6144, 7168, 7936]
        buckets = [b for b in buckets if b <= prompt_cap]
        if not buckets:
            buckets = [prompt_cap]
        log.info(f"Fast mode buckets: {buckets} (prompt_cap={prompt_cap})")

    i = 0
    pbar = tqdm(total=len(sorted_prompts), desc="generate", dynamic_ncols=True)
    while i < len(sorted_prompts):
        bs = min(batch_size, len(sorted_prompts) - i)
        done_this_round = False

        while not done_this_round:
            try:
                batch_slice = slice(i, i + bs)
                batch_prompts = sorted_prompts[batch_slice]

                if fast:
                    # Bucket to a stable max_length
                    longest = min(max(sorted_lengths[batch_slice]), prompt_cap)
                    max_len = _choose_bucket(longest, buckets, prompt_cap)
                    pad_mode = "max_length"
                else:
                    # Robust mode: dynamic per-batch padding to the true longest (<= prompt_cap)
                    max_len = min(max(sorted_lengths[batch_slice]), prompt_cap)
                    pad_mode = "longest"

                inputs = tok(
                    batch_prompts,
                    return_tensors="pt",
                    padding=pad_mode,
                    truncation=True,
                    max_length=max_len,
                ).to(model.device)

                gen_kwargs = dict(
                    max_new_tokens=int(max_new_tokens),
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                    use_cache=True,
                )
                if do_sample:
                    gen_kwargs.update(dict(do_sample=True, temperature=float(temperature), top_p=float(top_p)))

                with torch.inference_mode():
                    gen = _gen_call(model, **inputs, **gen_kwargs)

                # Decode only new tokens
                cut = inputs["input_ids"].shape[1]
                decoded = tok.batch_decode(gen[:, cut:], skip_special_tokens=True)

                for j, raw in enumerate(decoded):
                    ex = sorted_data[i + j]
                    parsed = val.validate_and_repair(raw, n_evidence_chunks=len(ex.get("context_chunks", [])))
                    results_by_sorted_idx[i + j] = {
                        "raw": raw,
                        "validated": {
                            "answer": parsed.answer,
                            "citations": parsed.citations,
                            "confidence": parsed.confidence,
                            "type": parsed.response_type.value,
                            "errors": parsed.validation_errors,
                        }
                    }

                i += bs
                pbar.update(bs)
                done_this_round = True

            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                if bs > 1:
                    bs = max(1, bs // 2)
                    log.warning(f"OOM; reducing batch size, retrying with bs={bs}")
                else:
                    # last resort: shrink prompt cap
                    old_cap = int(prompt_cap)
                    prompt_cap = max(1024, prompt_cap - 512)
                    log.warning(f"OOM at bs=1; reducing prompt_cap from {old_cap} to {prompt_cap} and retrying.")
                    if fast:
                        buckets = [b for b in buckets if b <= prompt_cap] or [prompt_cap]
                    if prompt_cap <= 1024:
                        log.error("Still OOM after aggressive shrinking; aborting this batch.")
                        raise e

    pbar.close()

    # Reorder results to original dataset order
    results: List[Dict[str, Any]] = [None] * len(results_by_sorted_idx)  # type: ignore
    for sorted_idx, res in enumerate(results_by_sorted_idx):
        orig_idx = order[sorted_idx]
        results[orig_idx] = res  # type: ignore

    return results, model, tok


def metrics_from_results(results: List[Dict[str, Any]], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(results) == len(data)
    n = len(results)
    ems: List[float] = []
    f1s: List[float] = []
    halluc = 0
    cite_corr = 0
    refusals = 0

    for res, ex in zip(results, data):
        got = res["validated"]
        gold = ex.get("answer", "")
        ems.append(em(got["answer"], gold))
        f1s.append(f1(got["answer"], gold))

        if got["type"] == "refusal":
            refusals += 1

        n_chunks = len(ex.get("context_chunks", []))
        cits: List[int] = got["citations"]
        oob = any((c < 0 or c >= n_chunks) for c in cits)
        if got["type"] != "refusal" and (len(cits) == 0 or oob):
            halluc += 1

        support = set(int(i) for i in ex.get("support_spans", []) if isinstance(i, int))
        if cits:
            if not support:
                cite_corr += 0
            else:
                cite_corr += int(set(cits).issubset(support))

    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    num_cited = sum(1 for res in results if res["validated"]["citations"])

    return {
        "n": n,
        "em": mean(ems),
        "f1": mean(f1s),
        "refusal_rate": refusals / max(1, n),
        "hallucination_rate": halluc / max(1, n),
        "citation_correctness": cite_corr / max(1, num_cited) if num_cited > 0 else 0.0,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--test-data", type=Path, default=Path("data/processed/test.jsonl"))
    p.add_argument("--template", type=str, default="default")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--compare-baseline", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    # New knobs
    p.add_argument("--prompt-cap", type=int, default=None, help="Max prompt tokens before generation; default ctx - max_new_tokens")
    p.add_argument("--fast", action="store_true", help="Bucketed shapes + (optional) torch.compile for speed")
    p.add_argument("--buckets", type=str, default=None, help="Comma-separated bucket sizes, used with --fast")
    p.add_argument("--attn-impl", type=str, default="auto", choices=["auto", "sdpa", "flash2"], help="Attention impl preference")
    args = p.parse_args()

    data = read_jsonl(args.test_data, args.max_samples)

    # SFT model
    log.info(f"Evaluating {args.model_path} on {len(data)} samples...")
    results_sft, model_sft, tok_sft = run_model(
        model_path=args.model_path,
        data=data,
        template=args.template,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_cap=args.prompt_cap,
        fast=args.fast,
        buckets=_parse_buckets(args.buckets) if args.fast else None,
        attn_impl=args.attn_impl,
    )
    m_sft = metrics_from_results(results_sft, data)
    log.info(f"SFT metrics: {json.dumps(m_sft, indent=2)}")

    out: Dict[str, Any] = {"sft": m_sft}

    # Baseline compare
    if args.compare_baseline and args.base_model:
        log.info("Clearing GPU memory before loading base model...")
        try:
            del model_sft, tok_sft, results_sft
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        log.info(f"Evaluating BASE model {args.base_model}...")
        results_base, _, _ = run_model(
            model_path=args.base_model,
            data=data,
            template=args.template,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            prompt_cap=args.prompt_cap,
            fast=args.fast,
            buckets=_parse_buckets(args.buckets) if args.fast else None,
            attn_impl=args.attn_impl,
        )
        m_base = metrics_from_results(results_base, data)
        log.info(f"BASE metrics: {json.dumps(m_base, indent=2)}")
        out["base"] = m_base
        out["deltas"] = {
            "em": m_sft["em"] - m_base["em"],
            "f1": m_sft["f1"] - m_base["f1"],
            "hallucination_rate": m_base["hallucination_rate"] - m_sft["hallucination_rate"],
            "citation_correctness": m_sft["citation_correctness"] - m_base["citation_correctness"],
        }

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        print(json.dumps(out, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
