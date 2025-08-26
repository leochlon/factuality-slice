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

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `transformers>=4.36.0`
- `peft>=0.7.0`
- `bitsandbytes>=0.41.0`
- `torch>=2.0.0`
- `datasets`
- `accelerate`
- `scikit-learn`
- `scipy`
- `pyyaml`
- `tqdm`

## Quick Start

### Step 1: Build the Dataset

```bash
python build_data.py --data-dir data --seed 42 --accept-new-hash
```

This downloads and processes data from FEVER, HotpotQA, NQ-Open, and PopQA, creating train/val/test splits with evidence chunks and citations.

### Step 2: Generate Prompt Templates

```bash
python prompt_schema.py save --prompts-dir prompts
```

Creates standardized prompt templates and output schemas for training.

### Step 3: Supervised Fine-Tuning (SFT)

```bash
python train_sft.py -c configs/sft_gemma2.yaml
```

Fine-tunes a base model using LoRA/QLoRA on the evidence-grounded QA task.

### Step 4: Evaluate SFT Model

```bash
python evaluate_sft_fast_robust.py \
  --model-path checkpoints/sft/gemma2-9b/checkpoint-final \
  --base-model google/gemma-2-9b \
  --test-data data/processed/test.jsonl \
  --compare-baseline \
  --batch-size 4 \
  --max-new-tokens 256 \
  --output eval_results.json
```

### Step 5: Generate Preference Data

Generate candidate responses from SFT and base models:

```bash
python make_prefs.py --phase generate \
  --sft-model checkpoints/sft/gemma2-9b/checkpoint-final \
  --base-model google/gemma-2-9b \
  --data-path data/processed/train.jsonl \
  --max-samples 600 --n-generations 2 \
  --batch-size 12 --use-8bit
```

Judge the candidates to create preference pairs:

```bash
python make_prefs.py --phase judge \
  --judge-model Qwen/Qwen2.5-3B-Instruct \
  --data-path data/processed/train.jsonl \
  --max-samples 600 --n-generations 2 \
  --judge-use-8bit --output-mode rich
```

### Step 6: Train Reward Model

```bash
python train_rm.py -c configs/rm.yaml
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

## Google Colab Usage

The `SFT.ipynb` notebook provides a complete training workflow for Google Colab:

1. Mount drive and authenticate with Hugging Face
2. Build the dataset
3. Train SFT model
4. Generate preferences (two-phase)
5. Train reward model

Each step includes progress tracking and intermediate validation.

## Advanced Features

### Fast Evaluation Mode

Enable bucketed shapes and torch.compile for faster inference:

```bash
python evaluate_sft_fast_robust.py \
  --fast \
  --buckets 2048,3072,4096,5120,6144,7168 \
  --attn-impl flash2
```

### Multi-GPU Training

The pipeline supports distributed training via Hugging Face Accelerate:

```bash
accelerate launch train_sft.py -c configs/sft_gemma2.yaml
```

### Custom Templates

Modify prompt templates in `prompt_schema.py`:
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