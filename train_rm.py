#!/usr/bin/env python3
"""
train_rm.py - Reward Model Training with Anti-gaming Features

Trains a reward model on preference pairs using Bradley-Terry loss.
Includes citation validation and anti-gaming penalties.

Usage:
  python train_rm.py -c configs/rm.yaml
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    set_seed,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr

from prompt_schema import PromptTemplates, OutputValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_rm")

# Performance settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def fmt_metric(value, precision=3):
    """Format metric value for logging"""
    if isinstance(value, (float, int)):
        return f"{value:.{precision}f}"
    else:
        return str(value)


@dataclass
class RMConfig:
    """Reward model configuration"""
    base_model: str
    output_dir: Path
    learning_rate: float = 5e-6
    batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 1
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10
    load_in_8bit: bool = False
    seed: int = 42
    # Anti-gaming params - matching exact names
    citation_penalty_weight: float = 0.2
    confidence_penalty_weight: float = 0.1
    length_penalty_weight: float = 0.05
    overcitation_penalty_weight: float = 0.15
    span_penalty_weight: float = 0.25  # For support span overlap check
    
    @classmethod
    def from_yaml(cls, path: Path) -> "RMConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Flatten nested config
        flat = {}
        for key, val in data.items():
            if isinstance(val, dict):
                flat.update(val)
            else:
                flat[key] = val
        
        # Convert paths
        if "output_dir" in flat:
            flat["output_dir"] = Path(flat["output_dir"])
        
        return cls(**{k: v for k, v in flat.items() if k in cls.__dataclass_fields__})


def to_device(x: Any, device: torch.device) -> Any:
    """Recursively move tensors to device"""
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    else:
        return x


class RewardHead(nn.Module):
    """Reward head with dropout for regularization"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = torch.tanh(x)
        x = self.dropout(x)
        rewards = self.out_proj(x)
        return rewards


class RewardModel(nn.Module):
    """Reward model with anti-gaming features"""
    
    def __init__(
        self, 
        base_model_name: str,
        config: RMConfig,
        device: str = "cuda"
    ):
        super().__init__()
        self.config = config
        self.device = device
        
        # Load base model
        model_config = AutoConfig.from_pretrained(base_model_name)
        
        bnb_config = None
        if config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.base = AutoModel.from_pretrained(
            base_model_name,
            config=model_config,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
        )
        
        # Freeze base model
        for param in self.base.parameters():
            param.requires_grad = False
        
        # Add reward head with correct dtype
        base_dtype = next(self.base.parameters()).dtype
        self.reward_head = RewardHead(model_config.hidden_size)
        self.reward_head.to(device=device, dtype=base_dtype)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning rewards"""
        
        # Get base model outputs
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Pool last hidden states (using last token for causal LM)
        last_hidden = outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        pooled = last_hidden[
            torch.arange(batch_size, device=last_hidden.device),
            sequence_lengths
        ]
        
        # Compute rewards
        rewards = self.reward_head(pooled).squeeze(-1)
        
        return rewards


class PreferenceDataset(Dataset):
    """Dataset for preference pairs"""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        templates: Optional[PromptTemplates] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.templates = templates or PromptTemplates()
        self.validator = OutputValidator(strict_mode=False)
        
        # Load preferences
        self.pairs = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    pair = json.loads(line)
                    # Validate/repair outputs
                    n_chunks = len(pair.get("context_chunks", []))
                    chosen_validated = self.validator.validate_and_repair(
                        json.dumps(pair["chosen"]), n_chunks
                    )
                    rejected_validated = self.validator.validate_and_repair(
                        json.dumps(pair["rejected"]), n_chunks
                    )
                    
                    # Update with validated versions
                    pair["chosen"]["answer"] = chosen_validated.answer
                    pair["chosen"]["citations"] = chosen_validated.citations
                    pair["chosen"]["confidence"] = chosen_validated.confidence
                    pair["rejected"]["answer"] = rejected_validated.answer
                    pair["rejected"]["citations"] = rejected_validated.citations
                    pair["rejected"]["confidence"] = rejected_validated.confidence
                    
                    self.pairs.append(pair)
        
        logger.info(f"Loaded {len(self.pairs)} preference pairs from {data_path}")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def _format_response(self, output: Dict[str, Any]) -> str:
        """Format response for tokenization"""
        return json.dumps({
            "answer": output.get("answer", ""),
            "citations": output.get("citations", []),
            "confidence": output.get("confidence", 0.5)
        })
    
    def _compute_penalties(self, output: Dict[str, Any], pair: Dict[str, Any]) -> Dict[str, float]:
        """Compute penalty features for anti-gaming including span overlap"""
        penalties = {}
        
        citations = output.get("citations", [])
        answer = output.get("answer", "")
        confidence = output.get("confidence", 0.5)
        n_chunks = len(pair.get("context_chunks", []))
        
        # 1. Invalid citations (out of bounds)
        invalid = sum(1 for c in citations if not (0 <= c < n_chunks))
        penalties["invalid_citations"] = invalid / max(1, len(citations)) if citations else 0.0
        
        # 2. Over-citation (citing more than 70% of chunks)
        if n_chunks > 0 and len(citations) > 0.7 * n_chunks:
            penalties["overcitation"] = (len(citations) - 0.7 * n_chunks) / n_chunks
        else:
            penalties["overcitation"] = 0.0
        
        # 3. Length penalty
        max_len = 120
        if len(answer) > max_len:
            penalties["length"] = min(1.0, (len(answer) - max_len) / max_len)
        else:
            penalties["length"] = 0.0
        
        # 4. Confidence mismatch (high confidence with no citations)
        if confidence > 0.7 and not citations:
            penalties["confidence_mismatch"] = confidence - 0.3
        else:
            penalties["confidence_mismatch"] = 0.0
        
        # 5. Support span overlap check (KEY anti-gaming feature from spec)
        support_spans = pair.get("labels", {}).get("support_spans", pair.get("support_spans", []))
        
        if citations and support_spans:
            # Check how many cited chunks actually contain support evidence
            overlaps = 0
            for cite_idx in citations:
                if 0 <= cite_idx < n_chunks:
                    # Check if this citation index is in the support spans
                    if cite_idx in support_spans:
                        overlaps += 1
                    else:
                        # Also check for string overlap as fallback
                        chunk_text = pair["context_chunks"][cite_idx].get("text", "")
                        # Check if any support span text appears in this chunk
                        for span_idx in support_spans:
                            if isinstance(span_idx, int) and 0 <= span_idx < n_chunks:
                                support_text = pair["context_chunks"][span_idx].get("text", "")
                                # Simple substring check for overlap
                                if support_text and chunk_text and len(support_text) > 10:
                                    # Check for meaningful overlap (not just common words)
                                    if support_text[:50] in chunk_text or chunk_text[:50] in support_text:
                                        overlaps += 0.5  # Partial credit for text overlap
                                        break
            
            # Penalty = fraction of citations that don't overlap support
            penalties["span_miss_rate"] = 1.0 - (overlaps / max(1, len(citations)))
        else:
            penalties["span_miss_rate"] = 0.0
        
        # Clamp all penalties to [0, 1] for stability
        for k in penalties:
            penalties[k] = float(max(0.0, min(1.0, penalties[k])))
        return penalties
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        
        # Build prompt
        prompt = self.templates.get_sft_prompt(
            question=pair["question"],
            evidence_chunks=pair["context_chunks"]
        )
        
        # Format chosen and rejected
        chosen_text = prompt + "\n" + self._format_response(pair["chosen"])
        rejected_text = prompt + "\n" + self._format_response(pair["rejected"])
        
        # Tokenize
        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Compute anti-gaming features
        chosen_penalties = self._compute_penalties(pair["chosen"], pair)
        rejected_penalties = self._compute_penalties(pair["rejected"], pair)
        
        # Determine label (1 if chosen is better, 0 otherwise)
        margin = pair.get("labels", {}).get("margin", 0.0)
        label = 1 if margin > 0 else 0
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
            "chosen_penalties": {k: torch.tensor(v, dtype=torch.float32) 
                                for k, v in chosen_penalties.items()},
            "rejected_penalties": {k: torch.tensor(v, dtype=torch.float32) 
                                  for k, v in rejected_penalties.items()},
            "label": torch.tensor(label, dtype=torch.long),
            "margin": torch.tensor(margin, dtype=torch.float32),
            "chosen_confidence": torch.tensor(pair["chosen"].get("confidence", 0.5), dtype=torch.float32),
            "rejected_confidence": torch.tensor(pair["rejected"].get("confidence", 0.5), dtype=torch.float32),
            "source": pair.get("metadata", {}).get("source", "unknown")
        }


class RewardTrainer:
    """Trainer for reward model"""
    
    def __init__(
        self,
        model: RewardModel,
        config: RMConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.eval_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        else:
            self.eval_loader = None
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        num_training_steps = len(self.train_loader) * config.num_epochs
        num_warmup_steps = int(config.warmup_ratio * num_training_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Tracking
        self.global_step = 0
        self.best_eval_auc = 0.0
        self.train_losses = []
        self.eval_metrics = []
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute Bradley-Terry loss with anti-gaming penalties"""
        
        # Get rewards for chosen and rejected
        chosen_rewards = self.model(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"]
        )
        
        rejected_rewards = self.model(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"]
        )
        
        # Apply anti-gaming penalties with correct config key mapping
        penalty_map = {
            "invalid_citations": "citation_penalty_weight",
            "overcitation": "overcitation_penalty_weight", 
            "length": "length_penalty_weight",
            "confidence_mismatch": "confidence_penalty_weight",
            "span_miss_rate": "span_penalty_weight"
        }
        
        for penalty_name, config_key in penalty_map.items():
            if "chosen_penalties" in batch and penalty_name in batch["chosen_penalties"]:
                weight = getattr(self.config, config_key, 0.1)
                chosen_penalty = batch["chosen_penalties"][penalty_name]
                rejected_penalty = batch["rejected_penalties"][penalty_name]
                
                chosen_rewards = chosen_rewards - weight * chosen_penalty
                rejected_rewards = rejected_rewards - weight * rejected_penalty
        
        # Bradley-Terry loss
        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits).mean()
        
        return loss, {"logits": logits.detach(), "rewards_diff": logits.detach()}
    
    def compute_baseline_heuristic(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        """Compute simple baseline heuristic scores"""
        # Heuristic: prefer valid citations, less overcitation, shorter, higher confidence, better span overlap
        chosen_score = (
            (1.0 - batch["chosen_penalties"]["invalid_citations"]) 
            - batch["chosen_penalties"]["overcitation"]
            - batch["chosen_penalties"]["length"] 
            - batch["chosen_penalties"]["span_miss_rate"]  # Include span penalty in baseline
            + batch["chosen_confidence"]
        )
        
        rejected_score = (
            (1.0 - batch["rejected_penalties"]["invalid_citations"])
            - batch["rejected_penalties"]["overcitation"]
            - batch["rejected_penalties"]["length"]
            - batch["rejected_penalties"]["span_miss_rate"]  # Include span penalty in baseline
            + batch["rejected_confidence"]
        )
        
        return (chosen_score - rejected_score).cpu().numpy()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set with proper metrics"""
        if not self.eval_loader:
            return {}
        
        self.model.eval()
        
        # Collect all predictions and labels
        all_labels = []
        all_scores = []  # Reward differences (continuous)
        all_margins = []
        all_sources = []
        all_chosen_conf = []
        all_rejected_conf = []
        all_heuristic_scores = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                # Move to device
                batch = to_device(batch, self.model.device)
                
                loss, info = self.compute_loss(batch)
                total_loss += loss.item()
                
                # Collect predictions (continuous scores for AUC)
                reward_diff = info["rewards_diff"].cpu().numpy()
                all_scores.extend(reward_diff)
                
                # Labels
                all_labels.extend(batch["label"].cpu().numpy())
                
                # Other metadata
                all_margins.extend(batch["margin"].cpu().numpy())
                all_sources.extend(batch["source"])
                all_chosen_conf.extend(batch["chosen_confidence"].cpu().numpy())
                all_rejected_conf.extend(batch["rejected_confidence"].cpu().numpy())
                
                # Baseline heuristic
                heuristic = self.compute_baseline_heuristic(batch)
                all_heuristic_scores.extend(heuristic)
        
        # Convert to arrays
        y_true = np.array(all_labels)
        y_scores = np.array(all_scores)
        margins = np.array(all_margins)
        conf_diff = np.array(all_chosen_conf) - np.array(all_rejected_conf)
        heuristic_scores = np.array(all_heuristic_scores)
        
        # Core metrics
        metrics = {
            "eval_loss": total_loss / len(self.eval_loader),
            "accuracy": float(((y_scores > 0).astype(int) == y_true).mean())
        }
        
        # ROC-AUC with continuous scores (KEY FIX)
        if len(set(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
            
            # Baseline heuristic AUC
            metrics["heuristic_auc"] = float(roc_auc_score(y_true, heuristic_scores))
            
            # Check if RM beats baseline
            metrics["beats_baseline"] = metrics["roc_auc"] > metrics["heuristic_auc"]
        
        # Calibration curve (reliability)
        try:
            probs = 1 / (1 + np.exp(-y_scores))  # Convert to probabilities
            fraction_pos, mean_pred = calibration_curve(
                y_true, probs, n_bins=5, strategy="quantile"
            )
            metrics["calibration_curve"] = {
                "fraction_positive": fraction_pos.tolist(),
                "mean_predicted": mean_pred.tolist()
            }
            # Expected calibration error
            ece = np.abs(fraction_pos - mean_pred).mean()
            metrics["calibration_ece"] = float(ece)
        except Exception as e:
            logger.warning(f"Calibration computation failed: {e}")
        
        # Reward monotonicity with confidence (when correct)
        correct_mask = (y_true == 1)
        if correct_mask.sum() > 1:
            try:
                corr, p_value = spearmanr(y_scores[correct_mask], conf_diff[correct_mask])
                metrics["monotonicity_spearman"] = float(corr)
                metrics["monotonicity_p_value"] = float(p_value)
            except Exception as e:
                logger.warning(f"Monotonicity computation failed: {e}")
        
        # Known-wrong vs known-right separation (using margin as proxy)
        high_margin_mask = (margins > 0.5)
        low_margin_mask = (margins < -0.5)
        if high_margin_mask.any() and low_margin_mask.any():
            known_right_scores = y_scores[high_margin_mask].mean()
            known_wrong_scores = y_scores[low_margin_mask].mean() 
            metrics["known_separation"] = float(known_right_scores - known_wrong_scores)
        
        # Per-source accuracy
        from collections import defaultdict
        source_data = defaultdict(list)
        for i, src in enumerate(all_sources):
            source_data[src].append((y_true[i], y_scores[i]))
        
        for src, pairs in source_data.items():
            if len(pairs) > 0:
                src_true = np.array([p[0] for p in pairs])
                src_scores = np.array([p[1] for p in pairs])
                metrics[f"accuracy_{src}"] = float(((src_scores > 0) == src_true).mean())
        
        self.model.train()
        return metrics
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Total optimization steps: {len(self.train_loader) * self.config.num_epochs}")
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(progress):
                # Move to device
                batch = to_device(batch, self.model.device)
                
                # Forward pass
                loss, _ = self.compute_loss(batch)
                
                # Backward pass  
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                epoch_loss += loss.item()
                self.train_losses.append(loss.item())
                
                # Update progress
                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = np.mean(self.train_losses[-self.config.logging_steps:])
                    logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}")
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0 and self.eval_loader:
                    metrics = self.evaluate()
                    self.eval_metrics.append(metrics)
                    
                    logger.info(f"Step {self.global_step}:")
                    logger.info(f"  Loss: {fmt_metric(metrics['eval_loss'], 4)}")
                    logger.info(f"  Accuracy: {fmt_metric(metrics['accuracy'])}")
                    logger.info(f"  ROC-AUC: {fmt_metric(metrics.get('roc_auc', 'N/A'))}")
                    logger.info(f"  vs Baseline: {fmt_metric(metrics.get('heuristic_auc', 'N/A'))}")
                    
                    # Save best model based on AUC
                    current_auc = metrics.get("roc_auc", 0)
                    if current_auc > self.best_eval_auc:
                        self.best_eval_auc = current_auc
                        self.save_checkpoint("best")
                        logger.info(f"  New best model! AUC: {current_auc:.3f}")
                
                # Regular checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")
        
        # Final save
        self.save_checkpoint("final")
        
        # Save training history
        self.save_training_history()
    
    def save_checkpoint(self, tag: str):
        """Save model checkpoint"""
        save_dir = self.config.output_dir / tag
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save reward head only (base model is frozen)
        torch.save(
            self.model.reward_head.state_dict(),
            save_dir / "reward_head.pt"
        )
        
        # Save config
        with open(save_dir / "config.json", "w") as f:
            json.dump({
                "base_model": self.config.base_model,
                "step": self.global_step,
                "best_auc": self.best_eval_auc,
            }, f, indent=2)
        
        logger.info(f"Saved checkpoint to {save_dir}")
    
    def save_training_history(self):
        """Save training metrics"""
        # Handle Path serialization
        config_dict = {}
        for k, v in vars(self.config).items():
            if isinstance(v, Path):
                config_dict[k] = str(v)
            else:
                config_dict[k] = v
        
        history = {
            "train_losses": self.train_losses,
            "eval_metrics": self.eval_metrics,
            "config": config_dict,
        }
        
        with open(self.config.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)


def validate_reward_model(
    model: RewardModel,
    test_data_path: Path,
    tokenizer: AutoTokenizer,
    config: RMConfig
) -> Dict[str, Any]:
    """Comprehensive validation of trained reward model"""
    
    # Load test data
    test_dataset = PreferenceDataset(test_data_path, tokenizer, config.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    
    # Create a temporary trainer for evaluation
    trainer = RewardTrainer(model, config, test_dataset, None)
    trainer.eval_loader = test_loader
    
    # Run comprehensive evaluation
    results = trainer.evaluate()
    
    # Additional validation checks
    validation_passed = True
    checks = []
    
    # Check 1: AUC > baseline heuristic
    if "beats_baseline" in results:
        passed = results["beats_baseline"]
        checks.append(("AUC > baseline heuristic", passed))
        validation_passed &= passed
    
    # Check 2: Known separation > threshold
    if "known_separation" in results:
        passed = results["known_separation"] > 0.3
        checks.append(("Known wrong/right separation > 0.3", passed))
        validation_passed &= passed
    
    # Check 3: Monotonicity positive correlation
    if "monotonicity_spearman" in results:
        passed = results["monotonicity_spearman"] > 0.2
        checks.append(("Confidence monotonicity > 0.2", passed))
        validation_passed &= passed
    
    # Check 4: Calibration ECE < threshold
    if "calibration_ece" in results:
        passed = results["calibration_ece"] < 0.2
        checks.append(("Calibration ECE < 0.2", passed))
        validation_passed &= passed
    
    results["validation_checks"] = checks
    results["validation_passed"] = validation_passed
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to validate")
    args = parser.parse_args()
    
    # Load config
    config = RMConfig.from_yaml(args.config)
    set_seed(config.seed)
    
    # Setup paths
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RewardModel(config.base_model, config, device)
    model.to(device)
    
    if args.validate_only:
        # Load checkpoint
        checkpoint_dir = config.output_dir / (args.checkpoint or "best")
        model.reward_head.load_state_dict(
            torch.load(checkpoint_dir / "reward_head.pt", map_location=device)
        )
        
        # Run validation
        test_path = Path("prefs/preferences.jsonl")  # Or from config
        results = validate_reward_model(model, test_path, tokenizer, config)
        
        # Save results
        with open(config.output_dir / "validation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        print(f"Accuracy: {fmt_metric(results.get('accuracy', 'N/A'))}")
        print(f"ROC-AUC: {fmt_metric(results.get('roc_auc', 'N/A'))}")
        print(f"Baseline AUC: {fmt_metric(results.get('heuristic_auc', 'N/A'))}")
        print(f"Beats baseline: {results.get('beats_baseline', 'N/A')}")
        print(f"Known separation: {fmt_metric(results.get('known_separation', 'N/A'))}")
        print(f"Monotonicity: {fmt_metric(results.get('monotonicity_spearman', 'N/A'))}")
        print(f"Calibration ECE: {fmt_metric(results.get('calibration_ece', 'N/A'))}")
        print("\nValidation checks:")
        for check_name, passed in results.get("validation_checks", []):
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}")
        print("="*50)
        
        if not results.get("validation_passed", False):
            return 1
        
    else:
        # Load datasets
        train_path = Path("prefs/preferences.jsonl")
        
        # Create deterministic train/eval split if needed
        train_split_path = Path("prefs/preferences_train.jsonl")
        eval_path = Path("prefs/preferences_eval.jsonl")
        
        if not eval_path.exists() and train_path.exists():
            # Load all data
            with open(train_path) as f:
                all_data = [json.loads(line) for line in f if line.strip()]
            
            # Deterministic shuffle and split
            rng = np.random.RandomState(config.seed)
            indices = rng.permutation(len(all_data))
            split_idx = int(0.9 * len(all_data))
            train_indices = indices[:split_idx]
            eval_indices = indices[split_idx:]
            
            # Save splits (don't overwrite original)
            with open(train_split_path, 'w') as f:
                for idx in train_indices:
                    f.write(json.dumps(all_data[idx]) + '\n')
            
            with open(eval_path, 'w') as f:
                for idx in eval_indices:
                    f.write(json.dumps(all_data[idx]) + '\n')
            
            logger.info(f"Created splits: train={len(train_indices)}, eval={len(eval_indices)}")
            
            # Use the split version for training
            train_path = train_split_path
        elif train_split_path.exists():
            # Use existing split
            train_path = train_split_path
        
        train_dataset = PreferenceDataset(train_path, tokenizer, config.max_seq_length)
        eval_dataset = PreferenceDataset(eval_path, tokenizer, config.max_seq_length) if eval_path.exists() else None
        
        # Train
        trainer = RewardTrainer(model, config, train_dataset, eval_dataset)
        trainer.train()
        
        # Final validation
        if eval_path.exists():
            model.reward_head.load_state_dict(
                torch.load(config.output_dir / "best" / "reward_head.pt", map_location=device)
            )
            results = validate_reward_model(model, eval_path, tokenizer, config)
            
            with open(config.output_dir / "final_validation.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print("\n" + "="*50)
            print("TRAINING COMPLETE")
            print("="*50)
            print(f"Best ROC-AUC: {fmt_metric(results.get('roc_auc', 'N/A'))}")
            print(f"vs Baseline: {fmt_metric(results.get('heuristic_auc', 'N/A'))}")
            print(f"Final accuracy: {fmt_metric(results.get('accuracy', 'N/A'))}")
            print(f"All checks passed: {results.get('validation_passed', False)}")
            print(f"Output: {config.output_dir}")
            print("="*50)
            
            if not results.get("validation_passed", False):
                logger.warning("Some validation checks failed!")
                return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())