#!/usr/bin/env python3
"""
Factuality Slice - Prompt Templates and Output Schema
Handles prompt generation, output parsing, validation, and auto-repair
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

class ResponseType(Enum):
    """Types of valid responses"""
    ANSWER = "answer"
    REFUSAL = "refusal"
    
@dataclass
class ValidatedOutput:
    """Validated model output"""
    response_type: ResponseType
    answer: Optional[str]
    citations: List[int]
    confidence: float
    raw_output: str
    validation_errors: List[str]
    was_repaired: bool

class PromptTemplates:
    """Canonical prompt templates for SFT and evaluation"""
    
    # Main SFT prompt template
    SFT_TEMPLATE = """You are a factual question-answering system. Answer questions using ONLY the provided evidence.

## Instructions:
1. Answer using ONLY information from the evidence chunks
2. Cite evidence by chunk indices (e.g., [0, 2] for chunks 0 and 2)
3. Provide calibrated confidence (0.0 to 1.0)
4. If evidence is insufficient, refuse with confidence 0.0
5. Keep answers SHORT (under 120 characters)
6. Respond with **JSON only**, no extra text

## Evidence Chunks:
{evidence_chunks}

## Question:
{question}

## Required Output Format (valid JSON):
```json
{{
  "answer": "Your concise answer or '[INSUFFICIENT_EVIDENCE]' or '[OUTDATED_CONTEXT]'",
  "citations": [0],
  "confidence": 0.87
}}
```

## Response:"""

    # Alternative SFT template with more structure
    SFT_STRUCTURED_TEMPLATE = """Task: Answer the question using provided evidence.

Rules:
- Use ONLY the evidence chunks below
- Cite chunks by index
- Express confidence (0=uncertain, 1=certain)
- Refuse if evidence insufficient
- Maximum 120 characters for answer
- Output **JSON only** (no extra text)

Evidence:
{evidence_chunks}

Question: {question}

Format your response as JSON:
```json
{{"answer": "...", "citations": [0], "confidence": 0.72}}
```

Response:"""

    # Few-shot SFT template with examples
    SFT_FEWSHOT_TEMPLATE = """Answer questions using evidence. Output JSON with answer, citations, and confidence.

Example 1:
Evidence:
[0] The Eiffel Tower is located in Paris, France.
[1] It was completed in 1889.
Question: When was the Eiffel Tower built?
Response: {{"answer": "1889", "citations": [1], "confidence": 0.95}}

Example 2:
Evidence:
[0] Dolphins are marine mammals.
[1] The Pacific Ocean covers 63 million square miles.
Question: What is the population of dolphins in Antarctica?
Response: {{"answer": "[INSUFFICIENT_EVIDENCE]", "citations": [], "confidence": 0.0}}

Now answer:
Evidence:
{evidence_chunks}
Question: {question}
Response:"""

    # Judge prompt for RLAIF
    JUDGE_TEMPLATE = """Evaluate which answer is better based on factual accuracy and proper evidence use.

## Evaluation Criteria:
1. Factual correctness vs evidence
2. Proper citations (all claims supported)
3. Appropriate confidence calibration
4. Proper refusal when evidence insufficient
- Scores are integers in the range 0 to 10

## Evidence:
{evidence_chunks}

## Question:
{question}

## Answer A:
{answer_a}

## Answer B:
{answer_b}

## Output JSON (no extra text):
```json
{{
  "verdict": "A",
  "rationale": "Brief explanation",
  "scores": {{
    "A": {{"factuality": 9, "citations": 9, "confidence": 8}},
    "B": {{"factuality": 7, "citations": 6, "confidence": 7}}
  }}
}}
```

Verdict:"""

    # Simplified judge prompt
    JUDGE_SIMPLE_TEMPLATE = """Compare two answers for factual accuracy.

Evidence: {evidence_chunks}
Question: {question}

Answer A: {answer_a}
Answer B: {answer_b}

Output JSON only:
```json
{{"verdict": "A", "rationale": "A cites both supporting chunks correctly."}}
```

Verdict:"""

    @staticmethod
    def format_evidence_chunks(chunks: List[Dict[str, str]]) -> str:
        """Format evidence chunks for prompt"""
        formatted = []
        for i, chunk in enumerate(chunks):
            formatted.append(f"[{i}] {chunk['text']}")
        return "\n".join(formatted)

    @classmethod
    def get_sft_prompt(cls, question: str, evidence_chunks: List[Dict], 
                      template_name: str = "default") -> str:
        """Get formatted SFT prompt"""
        formatted_evidence = cls.format_evidence_chunks(evidence_chunks)
        
        templates = {
            "default": cls.SFT_TEMPLATE,
            "structured": cls.SFT_STRUCTURED_TEMPLATE,
            "fewshot": cls.SFT_FEWSHOT_TEMPLATE
        }
        
        template = templates.get(template_name, cls.SFT_TEMPLATE)
        return template.format(
            evidence_chunks=formatted_evidence,
            question=question
        )
    
    @classmethod
    def get_judge_prompt(cls, question: str, evidence_chunks: List[Dict],
                        answer_a: str, answer_b: str, 
                        template_name: str = "default") -> str:
        """Get formatted judge prompt"""
        formatted_evidence = cls.format_evidence_chunks(evidence_chunks)
        
        templates = {
            "default": cls.JUDGE_TEMPLATE,
            "simple": cls.JUDGE_SIMPLE_TEMPLATE
        }
        
        template = templates.get(template_name, cls.JUDGE_TEMPLATE)
        return template.format(
            evidence_chunks=formatted_evidence,
            question=question,
            answer_a=answer_a,
            answer_b=answer_b
        )

class OutputValidator:
    """Validates and repairs model outputs"""
    
    # Compiled refusal patterns (case-insensitive)
    REFUSAL_PATTERNS = [
        re.compile(r"\[(?:insufficient|no)[_\s]?evidence\]", re.IGNORECASE),
        re.compile(r"\[outdated[_\s]?context\]", re.IGNORECASE),
        re.compile(r"insufficient evidence", re.IGNORECASE),
        re.compile(r"cannot answer", re.IGNORECASE),
        re.compile(r"no evidence", re.IGNORECASE)
    ]
    
    # Configuration
    MAX_ANSWER_CHARS = 120
    
    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, require exact JSON with no repairs
        """
        self.strict_mode = strict_mode
    
    def _strip_code_fences(self, text: str) -> str:
        """Strip markdown code fences if present"""
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        return m.group(1) if m else text
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from potentially messy text"""
        # First try: Look for fenced JSON blocks
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        
        # Second try: Direct parsing of stripped text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Third try: Find smallest object containing "answer"
        for match in re.finditer(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL):
            try:
                data = json.loads(match.group(0))
                if 'answer' in data:
                    return data
            except json.JSONDecodeError:
                continue
        
        # Fourth try: Look for any JSON-like structure
        json_patterns = [
            r'\{[^{}]*"answer"[^{}]*\}',  # Simple JSON with answer
            r'\{(?:[^{}]|\{[^{}]*\})*\}',  # Nested JSON (be careful)
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if 'answer' in data:
                        return data
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def repair_citations(self, citations: Any, n_chunks: int) -> List[int]:
        """Repair citations to valid format"""
        if citations is None:
            return []
        
        repaired = []
        
        if isinstance(citations, list):
            for c in citations:
                if isinstance(c, int) and 0 <= c < n_chunks:
                    repaired.append(c)
                elif isinstance(c, str) and c.isdigit():
                    val = int(c)
                    if 0 <= val < n_chunks:
                        repaired.append(val)
        elif isinstance(citations, str):
            # Extract numbers from string like "[0, 2, 3]" or "0,2,3"
            numbers = re.findall(r'\d+', citations)
            for n in numbers:
                val = int(n)
                if 0 <= val < n_chunks:
                    repaired.append(val)
        elif isinstance(citations, int):
            if 0 <= citations < n_chunks:
                repaired.append(citations)
        
        # Deduplicate and sort
        return sorted(set(repaired))
    
    def repair_confidence(self, confidence: Any) -> float:
        """Repair confidence to valid float in [0, 1]"""
        if confidence is None:
            return 0.5  # Default to uncertain
        
        try:
            if isinstance(confidence, str):
                # Handle percentage strings like "95%" or "0.95"
                confidence = confidence.strip().rstrip('%')
                if '/' in confidence:  # Handle fractions
                    parts = confidence.split('/')
                    confidence = float(parts[0]) / float(parts[1])
                else:
                    confidence = float(confidence)
                    if confidence > 1.0:  # Assume percentage
                        confidence = confidence / 100.0
            else:
                confidence = float(confidence)
            
            # Clamp to [0, 1]
            return max(0.0, min(1.0, confidence))
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.5
    
    def is_refusal(self, answer: str) -> bool:
        """Check if answer is a refusal"""
        for pattern in self.REFUSAL_PATTERNS:
            if pattern.search(answer):
                return True
        return False
    
    def validate_judge(self, raw_output: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Validate judge JSON: {"verdict": "A"|"B"|"TIE", "rationale": str, "scores": {...}}.
        Returns (parsed_json_or_none, errors).
        """
        errors = []
        text = self._strip_code_fences(raw_output)
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = self.extract_json(raw_output)
            if data is None:
                return None, ["Judge: invalid or no JSON"]
        
        verdict = data.get("verdict")
        if verdict not in {"A", "B", "TIE"}:
            errors.append("Judge: verdict must be 'A', 'B', or 'TIE'")
        
        rationale = data.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            errors.append("Judge: rationale must be a non-empty string")
        
        # scores optional but, if present, must contain 0..10 ints
        scores = data.get("scores")
        if scores is not None:
            for key in ("A", "B"):
                if key in scores:
                    for k2 in ("factuality", "citations", "confidence"):
                        v = scores.get(key, {}).get(k2)
                        if v is not None and (not isinstance(v, int) or not 0 <= v <= 10):
                            errors.append(f"Judge: scores[{key}][{k2}] must be int 0..10")
        
        return (data if not errors else None), errors
    
    def validate_and_repair(self, raw_output: str, 
                          n_evidence_chunks: int) -> ValidatedOutput:
        """
        Validate and repair model output
        
        Args:
            raw_output: Raw model output string
            n_evidence_chunks: Number of evidence chunks (for citation validation)
            
        Returns:
            ValidatedOutput with parsed fields and validation status
        """
        errors = []
        was_repaired = False
        
        # STRICT MODE: No repairs, JSON must be valid
        if self.strict_mode:
            try:
                cleaned = self._strip_code_fences(raw_output).strip()
                data = json.loads(cleaned)
                
                # Validate required fields
                if not isinstance(data.get("answer"), str):
                    raise ValueError("answer must be string")
                if not isinstance(data.get("citations"), list):
                    raise ValueError("citations must be list")
                if not isinstance(data.get("confidence"), (int, float)):
                    raise ValueError("confidence must be number")
                
                # Validate citation bounds
                for c in data["citations"]:
                    if not isinstance(c, int) or c < 0 or c >= n_evidence_chunks:
                        raise ValueError(f"citation {c} out of bounds")
                
                # Validate confidence range
                if not 0 <= data["confidence"] <= 1:
                    raise ValueError("confidence must be in [0,1]")
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                return ValidatedOutput(
                    response_type=ResponseType.ANSWER,
                    answer="[PARSING_ERROR]",
                    citations=[],
                    confidence=0.0,
                    raw_output=raw_output,
                    validation_errors=[f"Strict mode: {str(e)}"],
                    was_repaired=False
                )
            
            # Strict mode success - still normalize citations
            answer = data["answer"]
            citations = sorted(set(data["citations"]))
            confidence = data["confidence"]
            
        else:
            # REPAIR MODE: Try to extract and fix
            data = self.extract_json(raw_output)
            
            if data is None:
                # Try to construct from raw text
                was_repaired = True
                errors.append("No valid JSON found, attempting reconstruction")
                
                # Look for answer-like content
                answer_match = re.search(r'answer["\s:]+([^",}]+)', raw_output, re.IGNORECASE)
                answer = answer_match.group(1).strip() if answer_match else "[MISSING_ANSWER]"
                
                # Look for citations
                citation_numbers = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', raw_output)
                citations = []
                if citation_numbers:
                    for num_str in citation_numbers[0].split(','):
                        try:
                            val = int(num_str.strip())
                            if 0 <= val < n_evidence_chunks:
                                citations.append(val)
                        except ValueError:
                            pass
                
                # Look for confidence
                conf_match = re.search(r'confidence["\s:]+([0-9.%/]+)', raw_output, re.IGNORECASE)
                confidence = self.repair_confidence(conf_match.group(1) if conf_match else 0.5)
                
                data = {
                    "answer": answer,
                    "citations": citations,
                    "confidence": confidence
                }
            
            # Validate and repair answer
            answer = data.get("answer", "")
            if not answer:
                errors.append("Missing answer field")
                answer = "[MISSING_ANSWER]"
                was_repaired = True
            
            # Enforce short answer
            if isinstance(answer, str) and len(answer) > self.MAX_ANSWER_CHARS:
                if not self.is_refusal(answer):  # Don't truncate refusals
                    errors.append(f"Answer truncated from {len(answer)} to {self.MAX_ANSWER_CHARS} chars")
                    answer = answer[:self.MAX_ANSWER_CHARS].rstrip() + "..."
                    was_repaired = True
            
            # Validate and repair citations
            original_citations = data.get("citations")
            citations = self.repair_citations(original_citations, n_evidence_chunks)
            if citations != original_citations:
                errors.append(f"Repaired citations from {original_citations} to {citations}")
                was_repaired = True
            
            # Validate and repair confidence
            original_confidence = data.get("confidence")
            confidence = self.repair_confidence(original_confidence)
            if confidence != original_confidence:
                errors.append(f"Repaired confidence from {original_confidence} to {confidence}")
                was_repaired = True
        
        # Check for refusal
        response_type = ResponseType.REFUSAL if self.is_refusal(answer) else ResponseType.ANSWER
        
        # STRICT MODE: No repairs, only validation
        if self.strict_mode:
            # In strict mode: do not mutate fields. Only record logic violations.
            if response_type == ResponseType.REFUSAL:
                if citations:
                    errors.append("Strict: refusal must not include citations")
                if confidence > 0.1:
                    errors.append("Strict: refusal confidence must be ≤ 0.1")
            else:
                if not citations and confidence > 0.5:
                    errors.append("Strict: non-refusal with no citations should not have high confidence")
            
            return ValidatedOutput(
                response_type=response_type,
                answer=answer,
                citations=citations,
                confidence=confidence,
                raw_output=raw_output,
                validation_errors=errors,
                was_repaired=False
            )
        
        # REPAIR MODE: Logic checks and repairs
        if response_type == ResponseType.REFUSAL:
            if citations:
                errors.append("Refusal should not have citations")
                citations = []
                was_repaired = True
            if confidence > 0.1:
                errors.append("Refusal should have low confidence")
                confidence = 0.0
                was_repaired = True
        else:
            if not citations and confidence > 0.5:
                errors.append("High confidence answer should have citations")
                confidence = min(confidence, 0.3)
                was_repaired = True
        
        return ValidatedOutput(
            response_type=response_type,
            answer=answer,
            citations=citations,
            confidence=confidence,
            raw_output=raw_output,
            validation_errors=errors,
            was_repaired=was_repaired
        )

class SchemaManager:
    """Manages prompt templates and output schemas"""
    
    def __init__(self, prompts_dir: Path = Path("prompts")):
        self.prompts_dir = prompts_dir
        self.prompts_dir.mkdir(exist_ok=True)
        self.templates = PromptTemplates()
        self.validator = OutputValidator()
    
    def save_templates(self):
        """Save all prompt templates to files"""
        # Save SFT templates
        sft_templates = {
            "default": PromptTemplates.SFT_TEMPLATE,
            "structured": PromptTemplates.SFT_STRUCTURED_TEMPLATE,
            "fewshot": PromptTemplates.SFT_FEWSHOT_TEMPLATE
        }
        
        for name, template in sft_templates.items():
            filepath = self.prompts_dir / f"sft_{name}.txt"
            with open(filepath, 'w') as f:
                f.write(template)
        
        # Save judge templates
        judge_templates = {
            "default": PromptTemplates.JUDGE_TEMPLATE,
            "simple": PromptTemplates.JUDGE_SIMPLE_TEMPLATE
        }
        
        for name, template in judge_templates.items():
            filepath = self.prompts_dir / f"judge_{name}.txt"
            with open(filepath, 'w') as f:
                f.write(template)
        
        # Save output schema
        schema = {
            "type": "object",
            "required": ["answer", "citations", "confidence"],
            "properties": {
                "answer": {
                    "type": "string",
                    "maxLength": 120,
                    "description": "Concise answer or refusal marker"
                },
                "citations": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "uniqueItems": True,
                    "description": "Indices of evidence chunks used"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Calibrated confidence score"
                }
            },
            "additionalProperties": False,
            "refusal_values": [
                "[INSUFFICIENT_EVIDENCE]",
                "[OUTDATED_CONTEXT]",
                "[NO_EVIDENCE]"
            ]
        }
        
        schema_path = self.prompts_dir / "output_schema.json"
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        # Save validation config
        config = {
            "validation": {
                "strict_mode": False,
                "auto_repair": True,
                "max_answer_chars": 120,
                "max_citation_index": "n_chunks - 1",
                "refusal_max_confidence": 0.1,
                "no_citation_max_confidence": 0.3
            },
            "templates": {
                "sft": list(sft_templates.keys()),
                "judge": list(judge_templates.keys())
            }
        }
        
        config_path = self.prompts_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Saved templates and schema to {self.prompts_dir}/")
    
    def test_validation(self) -> Dict[str, Any]:
        """Test validation on various output formats"""
        test_cases = [
            # Valid JSON
            ('{"answer": "Paris", "citations": [0, 2], "confidence": 0.9}', True),
            # JSON in code fence
            ('```json\n{"answer": "Paris", "citations": [0], "confidence": 0.9}\n```', True),
            # Missing citations
            ('{"answer": "Paris", "confidence": 0.9}', True),  # Should repair
            # Text with embedded JSON
            ('Sure! Here is the answer: {"answer": "42", "citations": [1], "confidence": 0.8}', True),
            # Refusal - uppercase
            ('{"answer": "[INSUFFICIENT_EVIDENCE]", "citations": [], "confidence": 0.0}', True),
            # Refusal - lowercase (test case sensitivity fix)
            ('{"answer": "[insufficient_evidence]", "citations": [], "confidence": 0.0}', True),
            # Invalid confidence percentage
            ('{"answer": "Yes", "citations": [0], "confidence": "95%"}', True),  # Should repair
            # Natural language
            ('The answer is probably Paris based on chunk 0', True),  # Should attempt repair
            # Long answer (should truncate)
            ('{"answer": "' + 'x' * 200 + '", "citations": [0], "confidence": 0.8}', True),
            # Complete garbage
            ('ajsdkfj klasdjf lkasdf', False)
        ]
        
        results = []
        for raw_output, should_parse in test_cases:
            validated = self.validator.validate_and_repair(raw_output, n_evidence_chunks=5)
            # Success = we got a valid output (even if repaired)
            success = validated.answer not in ["[PARSING_ERROR]", "[MISSING_ANSWER]"]
            
            results.append({
                "input": raw_output[:50] + "..." if len(raw_output) > 50 else raw_output,
                "expected_parse": should_parse,
                "parsed_successfully": success,
                "was_repaired": validated.was_repaired,
                "errors": validated.validation_errors
            })
        
        return {
            "test_results": results,
            "success_rate": sum(1 for r in results if r["parsed_successfully"]) / len(results)
        }

def inference_sample(data_path: Path, template: str = "default", 
                    strict: bool = False) -> Dict[str, Any]:
    """
    Sample inference with validation
    
    Simulates: fslice infer --sample
    """
    import random
    
    manager = SchemaManager()
    manager.validator.strict_mode = strict
    
    # Load a sample from data
    samples = []
    if data_path.exists():
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= 10:  # Load just a few
                    break
                samples.append(json.loads(line))
    
    if not samples:
        # Create dummy sample
        samples = [{
            "question": "What is the capital of France?",
            "context_chunks": [
                {"id": "0", "text": "France is a country in Europe."},
                {"id": "1", "text": "Paris is the capital of France."},
                {"id": "2", "text": "The Eiffel Tower is in Paris."}
            ],
            "answer": "Paris",
            "support_spans": [1]
        }]
    
    sample = random.choice(samples)
    
    # Generate prompt
    prompt = manager.templates.get_sft_prompt(
        question=sample["question"],
        evidence_chunks=sample["context_chunks"],
        template_name=template
    )
    
    # Simulate model outputs (various formats for testing)
    simulated_outputs = [
        '{"answer": "' + sample["answer"] + '", "citations": ' + str(sample["support_spans"]) + ', "confidence": 0.85}',
        f'```json\n{{"answer": "{sample["answer"]}", "citations": {sample["support_spans"]}, "confidence": 0.9}}\n```',
        f'The answer is {sample["answer"]} based on evidence chunks {sample["support_spans"]}. Confidence: 0.9',
        f'{{"answer": "{sample["answer"]}", "citations": {sample["support_spans"]}, "confidence": "90%"}}',  # Wrong format
        '{"answer": "[INSUFFICIENT_EVIDENCE]", "citations": [], "confidence": 0.0}',
        '{"answer": "[insufficient_evidence]", "citations": [], "confidence": 0.0}',  # Test lowercase
    ]
    
    results = []
    for raw_output in simulated_outputs:
        validated = manager.validator.validate_and_repair(
            raw_output, 
            n_evidence_chunks=len(sample["context_chunks"])
        )
        
        # Success = valid output produced (even if repaired)
        success = validated.answer not in ["[PARSING_ERROR]", "[MISSING_ANSWER]"]
        
        results.append({
            "raw": raw_output[:100],
            "validated": {
                "answer": validated.answer,
                "citations": validated.citations,
                "confidence": validated.confidence,
                "type": validated.response_type.value
            },
            "success": success,
            "was_repaired": validated.was_repaired,
            "errors": validated.validation_errors
        })
    
    # Calculate success rate (valid JSON produced)
    success_count = sum(1 for r in results if r["success"])
    
    return {
        "sample_question": sample["question"],
        "n_evidence_chunks": len(sample["context_chunks"]),
        "validation_results": results,
        "success_rate": success_count / len(results) if results else 0
    }

def main():
    """Main entry point"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Factuality Prompt & Schema Manager")
    parser.add_argument("command", choices=["save", "test", "infer", "judge"],
                       help="Command to run")
    parser.add_argument("--prompts-dir", type=Path, default=Path("prompts"),
                       help="Directory for prompt templates")
    parser.add_argument("--data-path", type=Path, default=Path("data/processed/val.jsonl"),
                       help="Path to data for inference")
    parser.add_argument("--template", default="default",
                       help="Template name to use")
    parser.add_argument("--strict", action="store_true",
                       help="Use strict validation mode (no repairs)")
    parser.add_argument("--jsonl", action="store_true",
                       help="Output JSONL format for infer command")
    
    args = parser.parse_args()
    
    manager = SchemaManager(args.prompts_dir)
    
    if args.command == "save":
        manager.save_templates()
        print("\n Templates and schema saved successfully")
        
    elif args.command == "test":
        print("Testing output validation...")
        results = manager.test_validation()
        print(f"\n Validation tests complete")
        print(f"   Success rate: {results['success_rate']*100:.1f}%")
        
        for i, result in enumerate(results["test_results"], 1):
            status = "✓" if result["parsed_successfully"] else "✗"
            repair = " (repaired)" if result["was_repaired"] else ""
            print(f"   {status} Test {i}: {result['input']}{repair}")
            if result["errors"]:
                print(f"      Errors: {', '.join(result['errors'])}")
    
    elif args.command == "infer":
        results = inference_sample(args.data_path, args.template, strict=args.strict)
        
        if args.jsonl:
            # Output JSONL format (machine-readable)
            for result in results["validation_results"]:
                output = {
                    "answer": result["validated"]["answer"],
                    "citations": result["validated"]["citations"],
                    "confidence": result["validated"]["confidence"],
                    "type": result["validated"]["type"],
                    "success": result["success"],
                    "errors": result["errors"] if result["errors"] else None
                }
                print(json.dumps(output))
            
            # Exit with error code if strict validation failed
            if args.strict:
                any_fail = any(not r["success"] for r in results["validation_results"])
                if any_fail:
                    sys.exit(2)
        else:
            # Human-readable output
            print(f"Inference validation complete")
            print(f"Question: {results['sample_question']}")
            print(f"Success rate: {results['success_rate']*100:.1f}%")
            
            if results['success_rate'] < 0.99:
                sys.exit(1)  # Exit with error if <99% success
    
    elif args.command == "judge":
        # Test judge validation
        validator = OutputValidator()
        test_outputs = [
            '{"verdict": "A", "rationale": "A is more accurate", "scores": {"A": {"factuality": 9, "citations": 8, "confidence": 9}, "B": {"factuality": 6, "citations": 5, "confidence": 7}}}',
            '{"verdict": "B", "rationale": "B has better evidence"}',
            '{"verdict": "TIE", "rationale": "Both are equally good"}',
            '{"verdict": "A/B", "rationale": "invalid"}',  # Invalid verdict
            '{"verdict": "A"}',  # Missing rationale
        ]
        
        print("Testing judge validation...")
        for i, output in enumerate(test_outputs, 1):
            data, errors = validator.validate_judge(output)
            status = "✓" if data else "✗"
            print(f"  {status} Test {i}: {output[:50]}...")
            if errors:
                print(f"     Errors: {', '.join(errors)}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())