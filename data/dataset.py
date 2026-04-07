"""Memory-mapped dataset for efficient large-scale token streaming.

Uses numpy memmap so we never load the full dataset into RAM.
This is how real training pipelines work at scale.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple


class TokenDataset(Dataset):
    """Memory-mapped token dataset. Serves random chunks of seq_len+1 tokens
    (input = chunk[:-1], target = chunk[1:] for next-token prediction).
    """

    def __init__(self, data_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        # Number of complete sequences we can extract
        self.n_samples = (self.n_tokens - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def build_dataloaders(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders from preprocessed token files."""
    train_path = Path(data_dir) / "train.bin"
    val_path = Path(data_dir) / "val.bin"

    assert train_path.exists(), f"Train data not found at {train_path}. Run data/prepare.py first."
    assert val_path.exists(), f"Val data not found at {val_path}. Run data/prepare.py first."

    train_ds = TokenDataset(str(train_path), seq_len)
    val_ds = TokenDataset(str(val_path), seq_len)

    print(f"Train: {train_ds.n_tokens/1e6:.1f}M tokens, {len(train_ds)} sequences")
    print(f"Val:   {val_ds.n_tokens/1e6:.1f}M tokens, {len(val_ds)} sequences")

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_cuda, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda, drop_last=True,
    )
    return train_loader, val_loader
