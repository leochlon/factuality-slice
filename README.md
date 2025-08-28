# Factuality Slice: Evidence-Grounded QA Training Pipeline

A comprehensive pipeline for training language models to provide factually accurate, evidence-grounded question answering with proper citation practices. This system implements supervised fine-tuning (SFT) followed by reinforcement learning from AI feedback (RLAIF) to reduce hallucinations and improve citation accuracy.

## Key Features

- **Evidence-grounded QA**: Models learn to answer questions using only provided evidence chunks
- **Citation training**: Explicit training on proper citation practices with anti-gaming penalties
- **Multi-source dataset**: Combines FEVER, HotpotQA, NQ-Open, and PopQA for diverse training
- **RLAIF pipeline**: Two-phase preference generation with configurable judge models
- **Reward model training**: Bradley-Terry reward modeling with anti-gaming features
- **Comprehensive evaluation**: Metrics for accuracy, hallucination rate, and citation correctness

## Dataset

The pipeline builds a unified dataset from multiple public QA sources:

| Source | License | Focus |
|--------|---------|-------|
| FEVER | CC-BY-SA 4.0 | Fact verification |
| HotpotQA | CC-BY-SA 4.0 | Multi-hop reasoning |
| NQ-Open | CC-BY 4.0 | Open-domain QA |
| PopQA | MIT | Long-tail entity questions |

## Install

Recommended: install in editable mode so `sft` commands and module imports work everywhere.

```bash
cd SFT
pip install -e .
```

Alternatively, you can install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Build the Dataset

```bash
# either entrypoint or module form
sft-build-data --data-dir data --seed 42 --accept-new-hash
# python -m sft.build_data --data-dir data --seed 42 --accept-new-hash
```

This downloads and processes data from FEVER, HotpotQA, NQ-Open, and PopQA, creating train/val/test splits with evidence chunks and citations.

### Step 2: Supervised Fine-Tuning (SFT)

```bash
sft-train-sft -c configs/sft_gemma2.yaml
# python -m sft.train_sft -c configs/sft_gemma2.yaml
```

Fine-tunes a base model using LoRA/QLoRA on the evidence-grounded QA task.

### Step 3: Evaluate SFT Model

```bash
sft-eval \
  --model-path checkpoints/sft/gemma2-9b/checkpoint-final \
  --base-model google/gemma-2-9b \
  --test-data data/processed/test.jsonl \
  --compare-baseline \
  --batch-size 4 \
  --max-new-tokens 256 \
  --output eval_results.json
```

### Step 4: Generate Preference Data

Generate candidate responses from SFT and base models:

```bash
sft-make-prefs --phase generate \
  --sft-model checkpoints/sft/gemma2-9b/checkpoint-final \
  --base-model google/gemma-2-9b \
  --data-path data/processed/train.jsonl \
  --max-samples 600 --n-generations 2 \
  --batch-size 12 --use-8bit
```

Judge the candidates to create preference pairs:

```bash
sft-make-prefs --phase judge \
  --judge-model Qwen/Qwen2.5-3B-Instruct \
  --data-path data/processed/train.jsonl \
  --max-samples 600 --n-generations 2 \
  --judge-use-8bit --output-mode rich
```

### Step 5: Train Reward Model

```bash
sft-train-rm -c configs/rm.yaml
# python -m sft.train_rm -c configs/rm.yaml
```

Trains a reward model on the preference pairs with anti-gaming penalties.

## Configuration Files

### SFT Configuration (`configs/sft_gemma2.yaml`)

```yaml
model_name: google/gemma-2-9b
output_dir: checkpoints/sft/gemma2-9b
load_in_4bit: true
bf16: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 64
  learning_rate: 1.5e-5
  num_train_epochs: 2
  max_seq_length: 2048
  warmup_ratio: 0.03
  weight_decay: 0.1

data:
  train_path: data/processed/train.jsonl
  val_path: data/processed/val.jsonl
  template: default
```

### Reward Model Configuration (`configs/rm.yaml`)

```yaml
base_model: google/gemma-2-9b
output_dir: checkpoints/rm
learning_rate: 5e-6
batch_size: 16
num_epochs: 3

# Anti-gaming penalties
citation_penalty_weight: 0.2
confidence_penalty_weight: 0.1
length_penalty_weight: 0.05
overcitation_penalty_weight: 0.15
span_penalty_weight: 0.25
```

## Pipeline Architecture

### Data Flow

1. **Data Building**: Raw datasets → Unified evidence-grounded format
2. **SFT Training**: Base model → Fine-tuned QA model
3. **Preference Generation**: SFT + Base outputs → Judge evaluations → Preference pairs
4. **Reward Training**: Preference pairs → Reward model
5. **RLHF/DPO** (optional): Use reward model or preferences for further training

### Output Format

Models are trained to produce structured JSON outputs:

```json
{
  "answer": "Paris",
  "citations": [0, 2],
  "confidence": 0.85
}
```

- `answer`: Direct answer or `[INSUFFICIENT_EVIDENCE]` for refusals
- `citations`: List of evidence chunk indices used
- `confidence`: Calibrated confidence score (0.0 to 1.0)

## Evaluation Metrics

The pipeline tracks multiple metrics to assess model performance:

- **Exact Match (EM)**: Exact string match with gold answer
- **F1 Score**: Token-level overlap with gold answer
- **Hallucination Rate**: Answers without proper citations
- **Citation Correctness**: Whether citations match gold support spans
- **Refusal Rate**: Frequency of insufficient evidence responses
- **Calibration**: How well confidence scores match actual accuracy

## Notebook

The `SFT.ipynb` notebook has been updated to use `python -m sft.*` calls so it works after `pip install -e .`.

1. Mount drive and authenticate with Hugging Face
2. Build the dataset
3. Train SFT model
4. Generate preferences (two-phase)
5. Train reward model

Each step includes progress tracking and intermediate validation.

## Ablations

Run robustness and scaling ablations via the bundled CLI. Use either the console script `sft-ablate` (after `pip install -e .`) or module form.

- Noise robustness:
  
  ```bash
  sft-ablate noise \
    --test data/processed/test.jsonl \
    --sft-checkpoint checkpoints/sft/gemma2-9b/checkpoint-60 \
    --base-model google/gemma-2-9b \
    --attn-impl eager \
    --buckets 2048,3072,4096,5120,6144,7168,7936 \
    --max-new-tokens 256 \
    --compare-baseline \
    --outdir ablations/noise
  # python -m sft.run_ablation_suite noise ...
  ```

- RM data scaling from preferences (auto 10% val split):
  
  ```bash
  sft-ablate scaling \
    --prefs prefs/preferences_cb68999425.jsonl \
    --model microsoft/deberta-v3-base \
    --epochs 3 --batch-size 8 --grad-accum 2 \
    --lr 2e-5 --head-lr 1e-4 --warmup-ratio 0.03 \
    --sizes 128,256,512,768,1024,max \
    --outdir ablations/scaling/rm
  ```

- Run both noise and scaling:
  
  ```bash
  sft-ablate all \
    --test data/processed/test.jsonl \
    --sft-checkpoint checkpoints/sft/gemma2-9b/checkpoint-60 \
    --base-model google/gemma-2-9b \
    --buckets 2048,3072,4096,5120,6144,7168,7936 \
    --max-new-tokens 256 \
    --rm-prefs prefs/preferences_cb68999425.jsonl \
    --outdir ablations
  ```

Outputs are written under the chosen `--outdir` with per-ablation summaries.

## Advanced Features

### Fast Evaluation Mode

Enable bucketed shapes and torch.compile for faster inference:

```bash
sft-eval \
  --fast \
  --buckets 2048,3072,4096,5120,6144,7168 \
  --attn-impl flash2
```

### Multi-GPU Training

The pipeline supports distributed training via Hugging Face Accelerate:

```bash
accelerate launch python -m sft.train_sft -c configs/sft_gemma2.yaml
```

### Custom Templates

Modify prompt templates in `src/sft/prompt_schema.py`:
- `SFT_TEMPLATE`: Default training template
- `SFT_STRUCTURED_TEMPLATE`: More structured format
- `SFT_FEWSHOT_TEMPLATE`: Few-shot examples included

## Troubleshooting

### Out of Memory Errors

- Reduce `per_device_train_batch_size` in config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable 8-bit quantization with `--use-8bit` flag
- Reduce `max_seq_length` if context is too long

### Slow Training

- Enable Flash Attention 2: `attn_implementation: flash_attention_2`
- Use mixed precision training: `bf16: true`
- Enable gradient checkpointing (default in configs)

### Poor Citation Quality

- Increase `citation_penalty_weight` in reward model config
- Add more diverse negative examples in preference generation
- Ensure sufficient evidence coverage in training data


## License

This project uses multiple datasets with different licenses. Please refer to individual dataset licenses when using the processed data:
- FEVER: CC-BY-SA 4.0
- HotpotQA: CC-BY-SA 4.0
- NQ-Open: CC-BY 4.0
- PopQA: MIT

The pipeline code itself is released under MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

## Contact

For questions or issues, please open a GitHub issue in this repository.
