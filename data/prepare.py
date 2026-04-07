"""Download and preprocess data into tokenized binary files.

Uses HuggingFace datasets to download a FineWeb subset, then tokenizes
with our BPE tokenizer and saves as memory-mapped numpy arrays.

Usage:
    python -m data.prepare --output_dir ./data/processed --num_tokens 500_000_000
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def prepare_data(output_dir: str, num_tokens: int = 500_000_000, vocab_size: int = 32000):
    from datasets import load_dataset
    from data.tokenizer import BPETokenizer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = out / "tokenizer"

    # --- Step 1: Download data ---
    print("Downloading FineWeb-Edu subset...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # --- Step 2: Train tokenizer on a sample ---
    if not (tok_path / "merges.json").exists():
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        tokenizer = BPETokenizer()
        # Collect text samples for tokenizer training
        train_texts = []
        for i, example in enumerate(ds):
            train_texts.append(example["text"])
            if i >= 10000:  # 10K docs is enough for BPE training
                break
        tokenizer.train(train_texts, vocab_size=vocab_size)
        tokenizer.save(str(tok_path))
    else:
        print("Loading existing tokenizer...")
        tokenizer = BPETokenizer.load(str(tok_path))

    # --- Step 3: Tokenize and save as binary ---
    print(f"Tokenizing {num_tokens/1e6:.0f}M tokens...")

    # Re-create iterator
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    all_tokens = []
    total = 0
    for example in tqdm(ds, desc="Tokenizing"):
        tokens = tokenizer.encode(example["text"])
        all_tokens.extend(tokens)
        total += len(tokens)
        if total >= num_tokens:
            break

    all_tokens = np.array(all_tokens[:num_tokens], dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens)/1e6:.1f}M")

    # Split into train/val (99/1)
    split_idx = int(len(all_tokens) * 0.99)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    # Save as memory-mapped binary files
    train_tokens.tofile(str(out / "train.bin"))
    val_tokens.tofile(str(out / "val.bin"))

    print(f"Saved: train={len(train_tokens)/1e6:.1f}M tokens, val={len(val_tokens)/1e6:.1f}M tokens")
    print(f"Output directory: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--num_tokens", type=int, default=500_000_000)
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()
    prepare_data(args.output_dir, args.num_tokens, args.vocab_size)
