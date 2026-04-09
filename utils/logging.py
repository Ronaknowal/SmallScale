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
        self.rows = []           # keep all rows in memory for potential rewrite
        self.fieldnames = []     # ordered list of known columns
        self.fieldnames_set = set()
        self.file = None
        self.writer = None
        self.use_wandb = use_wandb
        self.start_time = time.time()

        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=self.exp_dir.name, dir=str(self.exp_dir))
            except ImportError:
                print("wandb not installed, falling back to CSV only")
                self.use_wandb = False

    def _open_csv(self):
        """Open CSV file and write header."""
        if self.file:
            self.file.close()
        self.file = open(self.log_path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def _rewrite_csv(self):
        """Rewrite entire CSV with updated fieldnames (called when new columns appear)."""
        self._open_csv()
        for row in self.rows:
            self.writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self.file.flush()

    def log(self, metrics: Dict[str, Any], step: int):
        metrics["step"] = step
        metrics["elapsed_s"] = round(time.time() - self.start_time, 1)

        # Check for new columns
        new_keys = set(metrics.keys()) - self.fieldnames_set
        needs_rewrite = bool(new_keys)

        if new_keys:
            # Add new keys in sorted order (step and elapsed_s always first)
            for k in sorted(new_keys):
                if k not in ("step", "elapsed_s"):
                    self.fieldnames.append(k)
                    self.fieldnames_set.add(k)
            # Ensure step and elapsed_s are at the front
            for k in ("elapsed_s", "step"):
                if k in self.fieldnames:
                    self.fieldnames.remove(k)
                self.fieldnames.insert(0, k)
                self.fieldnames_set.add(k)

        self.rows.append(metrics)

        if needs_rewrite:
            # New columns detected — rewrite entire CSV so all rows have all columns
            self._rewrite_csv()
        else:
            # Just append the new row
            if self.writer is None:
                self._open_csv()
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
