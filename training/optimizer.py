"""AdamW optimizer with proper weight decay separation.

Key insight: weight decay should NOT be applied to biases, norms, or embeddings.
This is a common mistake that hurts performance. We explicitly separate
parameter groups.
"""

import torch


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float,
                    beta1: float, beta2: float) -> torch.optim.AdamW:
    # Separate parameters: decay vs no-decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't decay 1D params (biases, norms) or embeddings
        if param.dim() == 1 or "emb" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: {n_decay/1e6:.1f}M params with decay, {n_no_decay/1e6:.1f}M without")

    optimizer = torch.optim.AdamW(
        param_groups, lr=lr, betas=(beta1, beta2), fused=True
    )
    return optimizer
