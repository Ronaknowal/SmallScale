"""Training loop with gradient accumulation, mixed precision, and logging.

Written from scratch (no HuggingFace Trainer) to demonstrate full understanding
of the training process. Every decision is documented for the research write-up.
"""

import torch
import torch.nn as nn
import time
import math
from pathlib import Path
from typing import Optional
from contextlib import nullcontext

from utils.config import ExperimentConfig
from utils.logging import ExperimentLogger
from model.transformer import Transformer
from data.dataset import build_dataloaders
from training.optimizer import build_optimizer
from training.scheduler import get_lr


class Trainer:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        if torch.cuda.is_available() and torch.cuda.device_count() == 0:
            print("WARNING: CUDA is available but no devices found. Check CUDA_VISIBLE_DEVICES env var.")

        # Precision context
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[cfg.training.dtype]
        # AMP only works properly on CUDA
        self.use_amp = self.dtype != torch.float32 and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.dtype == torch.float16))
        self.amp_ctx = (
            torch.amp.autocast("cuda", dtype=self.dtype)
            if self.use_amp else nullcontext()
        )

        # Build components
        print(f"Building model ({cfg.model.n_params/1e6:.0f}M params estimated)...")
        self.model = Transformer(cfg.model).to(self.device)

        if cfg.training.compile and self.device == "cuda":
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        self.optimizer = build_optimizer(
            self.model, cfg.training.lr, cfg.training.weight_decay,
            cfg.training.beta1, cfg.training.beta2,
        )

        print("Building dataloaders...")
        self.train_loader, self.val_loader = build_dataloaders(
            cfg.data.data_dir, cfg.data.seq_len, cfg.data.batch_size, cfg.data.num_workers,
        )

        self.logger = ExperimentLogger(str(cfg.exp_dir))
        self.logger.log_config({"model": cfg.model.__dict__, "training": cfg.training.__dict__})

        # Training state
        self.step = 0
        self.best_val_loss = float("inf")
        self.tokens_seen = 0

    def _get_batch(self, loader_iter):
        """Get next batch, cycling the dataloader if exhausted."""
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(self.train_loader)
            x, y = next(loader_iter)
        return x.to(self.device), y.to(self.device), loader_iter

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation on val set, return mean loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        max_eval_batches = 50  # cap eval time

        for i, (x, y) in enumerate(self.val_loader):
            if i >= max_eval_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            with self.amp_ctx:
                out = self.model(x, targets=y)
            total_loss += out["loss"].item()
            n_batches += 1

        self.model.train()
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, name: str = "latest"):
        """Save model checkpoint."""
        ckpt_dir = self.cfg.exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{name}.pt"

        # Unwrap compiled model if needed
        raw_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

        torch.save({
            "step": self.step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "tokens_seen": self.tokens_seen,
            "config": self.cfg,
        }, path)

    def train(self):
        """Main training loop."""
        cfg = self.cfg.training
        print(f"\n{'='*60}")
        print(f"Starting training: {cfg.max_steps} steps")
        print(f"  Batch size: {self.cfg.data.batch_size} x {cfg.grad_accum_steps} accum = "
              f"{self.cfg.data.batch_size * cfg.grad_accum_steps} effective")
        print(f"  Seq length: {self.cfg.data.seq_len}")
        print(f"  Tokens/step: {self.cfg.data.batch_size * cfg.grad_accum_steps * self.cfg.data.seq_len:,}")
        print(f"  Precision: {cfg.dtype}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        self.model.train()
        loader_iter = iter(self.train_loader)

        t0 = time.time()
        running_loss = 0.0
        running_count = 0

        for self.step in range(1, cfg.max_steps + 1):
            # Update learning rate
            lr = get_lr(
                self.step, cfg.lr, cfg.min_lr,
                cfg.warmup_steps, cfg.max_steps, cfg.lr_schedule,
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation loop
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(cfg.grad_accum_steps):
                x, y, loader_iter = self._get_batch(loader_iter)

                with self.amp_ctx:
                    out = self.model(x, targets=y)
                    loss = out["loss"] / cfg.grad_accum_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

                self.tokens_seen += x.numel()

            # Gradient clipping
            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                ).item()
            else:
                grad_norm = 0.0

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += accum_loss
            running_count += 1

            # --- Logging ---
            if self.step % cfg.log_interval == 0:
                avg_loss = running_loss / running_count
                elapsed = time.time() - t0
                tokens_per_sec = self.tokens_seen / elapsed
                ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow

                self.logger.log({
                    "train/loss": round(avg_loss, 4),
                    "train/ppl": round(ppl, 2),
                    "train/lr": lr,
                    "train/grad_norm": round(grad_norm, 4),
                    "train/tokens_seen": self.tokens_seen,
                    "perf/tokens_per_sec": int(tokens_per_sec),
                }, step=self.step)

                print(
                    f"step {self.step:>6d} | loss {avg_loss:.4f} | ppl {ppl:8.2f} | "
                    f"lr {lr:.2e} | gnorm {grad_norm:.2f} | "
                    f"tok/s {tokens_per_sec/1000:.1f}K"
                )
                running_loss = 0.0
                running_count = 0

            # --- Evaluation ---
            if self.step % cfg.eval_interval == 0:
                val_loss = self.evaluate()
                val_ppl = math.exp(min(val_loss, 20))
                self.logger.log({
                    "val/loss": round(val_loss, 4),
                    "val/ppl": round(val_ppl, 2),
                }, step=self.step)
                print(f"  >>> val loss {val_loss:.4f} | val ppl {val_ppl:.2f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
                    print(f"  >>> New best! Saved checkpoint.")

            # --- Save ---
            if self.step % cfg.save_interval == 0:
                self.save_checkpoint("latest")

        # Final save
        self.save_checkpoint("final")
        self.logger.close()
        total_time = time.time() - t0
        print(f"\nTraining complete in {total_time/3600:.1f}h")
        print(f"Best val loss: {self.best_val_loss:.4f}")
