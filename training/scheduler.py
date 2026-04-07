"""Learning rate schedulers with warmup.

Cosine decay with warmup is standard. Linear decay is a useful ablation.
Warmup length is an important hyperparameter to sweep — too short causes
instability, too long wastes compute on suboptimal learning rates.
"""

import math


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
    schedule: str = "cosine",
) -> float:
    """Compute learning rate for a given step."""
    # Warmup phase: linear increase
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # After max_steps: hold at min_lr
    if step >= max_steps:
        return min_lr

    # Decay phase
    progress = (step - warmup_steps) / (max_steps - warmup_steps)

    if schedule == "cosine":
        # Cosine decay from max_lr to min_lr
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * coeff
    elif schedule == "linear":
        return max_lr - (max_lr - min_lr) * progress
    elif schedule == "constant":
        return max_lr
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
