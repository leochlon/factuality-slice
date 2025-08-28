
# prompt_schema.fixed.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re

REFUSAL_TOKENS = {"[INSUFFICIENT_EVIDENCE]", "[OUTDATED_CONTEXT]", "[NO_EVIDENCE]"}


class PromptTemplates:
    """
    Templates for building SFT/eval prompts.
    Includes a verdict-aware instruction block.
    """

    def __init__(self) -> None:
        pass

    def _format_evidence(self, evidence_chunks: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for i, ch in enumerate(evidence_chunks or []):
            title = (ch.get("title") or "").strip()
            text = (ch.get("text") or "").strip()
            if title:
                lines.append(f"[{i}] {title}: {text}")
            else:
                lines.append(f"[{i}] {text}")
        return "\n".join(lines) if lines else "[no evidence provided]"

    def get_sft_prompt(self, question: str, evidence_chunks: List[Dict[str, Any]], template_name: str = "default") -> str:
        evidence = self._format_evidence(evidence_chunks)
        instructions = (
            "You are a careful QA assistant. Use ONLY the evidence provided below.\n"
            "Decide if the question is answerable from the evidence (ANSWERABLE), not supported (NEI), or outdated (OUTDATED).\n"
            "Return ONLY a JSON object with keys exactly: verdict, answer, citations, confidence.\n"
            "- verdict: one of \"ANSWERABLE\", \"NEI\", \"OUTDATED\".\n"
            "- answer: if verdict==ANSWERABLE, return the concise answer string; "
            "         if verdict==NEI, return \"[INSUFFICIENT_EVIDENCE]\"; "
            "         if verdict==OUTDATED, return \"[OUTDATED_CONTEXT]\".\n"
            "- citations: list of integer indices of evidence chunks that support the answer; empty for NEI/OUTDATED.\n"
            "- confidence: a float in [0,1].\n"
            "Do not add any extra keys or text."
        )
        return f"{instructions}\n\nQuestion:\n{question}\n\nEvidence:\n{evidence}\n\nJSON:"


# ---------- Evaluation-time validator ----------

class ResponseType(Enum):
    ANSWER = "answer"
    REFUSAL = "refusal"
    MALFORMED = "malformed"


@dataclass
class Validated:
    answer: str
    citations: List[int]
    confidence: float
    response_type: ResponseType
    validation_errors: List[str]


class OutputValidator:
    """
    Parses model generations and repairs into a normalized structure used by eval.
    Backward compatible with older outputs that omitted 'verdict'.
    """
    def __init__(self, strict_mode: bool = False) -> None:
        self.strict_mode = strict_mode

    # -- public API expected by evaluate_sft_fast_robust.py --
    def validate_and_repair(self, raw: str, n_evidence_chunks: Optional[int] = None) -> Validated:
        errors: List[str] = []
        obj = self._extract_json_obj(raw)
        if obj is None:
            # Heuristics for refusals in plain text
            text = (raw or "").strip()
            if any(tok in text for tok in REFUSAL_TOKENS):
                verdict = "NEI" if "[OUTDATED_CONTEXT]" not in text else "OUTDATED"
                ans = "[OUTDATED_CONTEXT]" if verdict == "OUTDATED" else "[INSUFFICIENT_EVIDENCE]"
                return Validated(answer=ans, citations=[], confidence=0.1, response_type=ResponseType.REFUSAL, validation_errors=["no_json"])
            # Otherwise: malformed
            return Validated(answer="", citations=[], confidence=0.0, response_type=ResponseType.MALFORMED, validation_errors=["no_json"])

        # Fields
        verdict = str(obj.get("verdict", "") or "").strip().upper()
        answer = obj.get("answer", "")
        citations = obj.get("citations", [])
        confidence = obj.get("confidence", None)

        # Backward compatibility: infer verdict from answer token if needed
        if not verdict:
            if isinstance(answer, str) and answer.strip().upper() in REFUSAL_TOKENS:
                verdict = "OUTDATED" if answer.strip().upper() == "[OUTDATED_CONTEXT]" else "NEI"
            else:
                verdict = "ANSWERABLE"

        # Normalize answer
        if not isinstance(answer, str):
            try:
                answer = str(answer)
                errors.append("answer_coerced_to_string")
            except Exception:
                answer = ""
                errors.append("answer_set_empty")

        # Normalize citations
        if isinstance(citations, list):
            try:
                citations = [int(c) for c in citations if isinstance(c, (int, float, str)) and str(c).strip().lstrip("-").isdigit()]
            except Exception:
                citations = []
                errors.append("citations_parse_fail")
        else:
            citations = []
            errors.append("citations_not_list")

        # Bound-check citations
        if isinstance(n_evidence_chunks, int) and n_evidence_chunks >= 0 and citations:
            citations = [c for c in citations if 0 <= c < n_evidence_chunks]

        # Normalize confidence
        try:
            confidence = float(confidence) if confidence is not None else 0.5
        except Exception:
            confidence = 0.5
            errors.append("confidence_parse_fail")
        confidence = max(0.0, min(1.0, confidence))

        # Enforce refusal logic
        rtype = ResponseType.ANSWER
        up_ans = answer.strip().upper()
        if verdict in {"NEI", "OUTDATED"} or up_ans in REFUSAL_TOKENS:
            rtype = ResponseType.REFUSAL
            # Force canonical refusal tokens
            answer = "[OUTDATED_CONTEXT]" if verdict == "OUTDATED" else "[INSUFFICIENT_EVIDENCE]"
            citations = []  # eval expects no citations on refusal

        # Minimal strict checks
        if self.strict_mode:
            # Require fields to exist and be of correct type; otherwise mark malformed
            if rtype != ResponseType.REFUSAL and (not answer or not isinstance(citations, list)):
                rtype = ResponseType.MALFORMED
                errors.append("strict_violation")

        # Deduplicate and sort citations for determinism
        citations = sorted(set(citations))

        return Validated(
            answer=answer,
            citations=citations,
            confidence=confidence,
            response_type=rtype,
            validation_errors=errors,
        )

    # -- helpers --
    def _extract_json_obj(self, raw: str) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, str):
            return None
        s = raw.strip()
        # Fast path
        try:
            return json.loads(s)
        except Exception:
            pass
        # Try to locate a JSON object substring
        obj = self._find_json_block(s)
        if obj is not None:
            return obj
        # Try repairing common issues: single quotes, trailing commas
        repaired = self._repair_common(s)
        if repaired is not None:
            return repaired
        return None

    def _find_json_block(self, s: str) -> Optional[Dict[str, Any]]:
        # naive bracket-matching to extract first {...}
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                # Try to tighten by scanning
                depth = 0
                for i, ch in enumerate(s[start:], start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            cand = s[start:i+1]
                            try:
                                return json.loads(cand)
                            except Exception:
                                break
        return None

    def _repair_common(self, s: str) -> Optional[Dict[str, Any]]:
        # Replace single quotes with double quotes in a conservative way
        candidate = s
        # Remove trailing content after final '}' to avoid decoder tails
        if "}" in candidate:
            candidate = candidate[: candidate.rfind("}") + 1]
        # heuristic: convert single quotes to double quotes for keys/strings
        candidate = re.sub(r"([{,\s])'([^']+?)'\s*:", r'\1"\2":', candidate)
        candidate = re.sub(r':\s*\'([^\']*?)\'', r': "\1"', candidate)
        # remove trailing commas before } or ]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            return None
