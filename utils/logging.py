"""Lightweight experiment logging to CSV + optional W&B."""

import csv
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any


class ExperimentLogger:
    def __init__(self, exp_dir: str, use_wandb: bool = False, wandb_project: str = "lm-scratch"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.exp_dir / "metrics.csv"
        self.writer = None
        self.file = None
        self.fieldnames = None
        self.use_wandb = use_wandb
        self.start_time = time.time()

        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=self.exp_dir.name, dir=str(self.exp_dir))
            except ImportError:
                print("wandb not installed, falling back to CSV only")
                self.use_wandb = False

    def log(self, metrics: Dict[str, Any], step: int):
        metrics["step"] = step
        metrics["elapsed_s"] = round(time.time() - self.start_time, 1)

        # CSV logging
        if self.writer is None:
            self.fieldnames = list(metrics.keys())
            self.file = open(self.log_path, "w", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        self.writer.writerow({k: metrics.get(k, "") for k in self.fieldnames})
        self.file.flush()

        # W&B logging
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def log_config(self, config: dict):
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def close(self):
        if self.file:
            self.file.close()
        if self.use_wandb:
            import wandb
            wandb.finish()
