"""Main entry point for training.

Usage:
    python train.py --config configs/base.yaml
    python train.py --config configs/base.yaml --override model.pos_encoding=alibi --override name=alibi
"""

import argparse
import torch
import random
import numpy as np

from utils.config import ExperimentConfig
from training.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Train a language model from scratch")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--override", action="append", default=[],
                        help="Override config values, e.g. model.pos_encoding=alibi")
    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for o in args.override:
        key, val = o.split("=", 1)
        # Try to parse as number/bool
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
        overrides[key] = val

    if overrides:
        cfg = ExperimentConfig.from_overrides(args.config, **overrides)
    else:
        cfg = ExperimentConfig.from_yaml(args.config)

    cfg.save()
    set_seed(cfg.seed)

    print(f"\n{'='*60}")
    print(f"Experiment: {cfg.name}")
    print(f"Output: {cfg.exp_dir}")
    print(f"Model: {cfg.model.n_params/1e6:.0f}M params (estimated)")
    print(f"  pos_encoding={cfg.model.pos_encoding}, activation={cfg.model.activation}")
    print(f"  n_heads={cfg.model.n_heads}, n_kv_heads={cfg.model.n_kv_heads}")
    print(f"  norm={cfg.model.norm_type} ({cfg.model.norm_position}-norm)")
    print(f"{'='*60}\n")

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
