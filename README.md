# LM-from-Scratch: Research-Grade Small Language Model Training

A from-scratch implementation of transformer language model training with built-in ablation support, designed to produce research-quality insights at small scale (100M–300M parameters).

## Why This Project Exists

This is not a wrapper around HuggingFace Trainer. Every component—model, tokenizer training, data pipeline, training loop, evaluation—is written from scratch in PyTorch to demonstrate deep understanding of each piece.

## Project Structure

```
lm-from-scratch/
├── configs/           # YAML experiment configs
│   ├── base.yaml      # Default 150M config
│   └── ablations/     # One config per ablation
├── model/
│   ├── transformer.py # Full model: embeddings, attention, FFN, LM head
│   ├── attention.py   # MHA, GQA, MQA implementations
│   ├── positional.py  # RoPE, ALiBi, learned positional embeddings
│   ├── feedforward.py # SwiGLU, GELU FFN variants
│   └── norms.py       # Pre-norm vs post-norm wrappers
├── data/
│   ├── tokenizer.py   # BPE tokenizer from scratch
│   ├── dataset.py     # Memory-mapped streaming dataset
│   └── prepare.py     # Download & preprocess data
├── training/
│   ├── trainer.py     # Training loop with logging
│   ├── optimizer.py   # AdamW with decoupled weight decay
│   └── scheduler.py   # Cosine, linear, warmup schedules
├── evaluation/
│   ├── perplexity.py  # Validation perplexity
│   ├── benchmarks.py  # HellaSwag, ARC-Easy (few-shot)
│   └── analysis.py    # Loss curve plotting, ablation comparison
├── utils/
│   ├── logging.py     # Experiment tracking (CSV + optional W&B)
│   └── config.py      # Config loading & validation
├── train.py           # Main entry point
├── run_ablations.py   # Run full ablation suite
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Prepare data (downloads a FineWeb subset)
python -m data.prepare --output_dir ./data/processed --num_tokens 1B

# 2. Train base model
python train.py --config configs/base.yaml

# 3. Run ablation suite
python run_ablations.py --suite positional_encoding

# 4. Generate analysis plots
python -m evaluation.analysis --exp_dir ./experiments/
```

## Ablation Dimensions

| Dimension | Variants | What to measure |
|-----------|----------|-----------------|
| Positional encoding | RoPE, ALiBi, Learned | Loss, long-context perf |
| Attention | MHA, GQA (4 groups), GQA (2 groups) | Loss vs throughput |
| Activation | SwiGLU, GELU | Loss, param count tradeoff |
| Normalization | Pre-norm, Post-norm | Training stability, final loss |
| LR warmup | 100, 500, 1000, 2000 steps | Loss curves, instability events |
| Vocab size | 8K, 16K, 32K, 64K | Tokenizer fertility, downstream loss |

## Hardware Requirements

- **Minimum**: 1x RTX 3090/4090 (24GB) — trains 150M in ~12-18 hours on 1B tokens
- **Recommended**: 1x A100 40GB — trains 200M in ~6-8 hours on 2B tokens
- All configs include gradient accumulation for smaller GPUs
