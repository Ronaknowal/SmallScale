"""Normalization variants: RMSNorm and LayerNorm, pre-norm and post-norm.

Pre-norm (normalize before attention/FFN) is standard in modern LLMs as it
stabilizes training. Post-norm (original transformer) can achieve slightly
better final loss but is prone to training instability, especially without
careful LR tuning. Worth ablating to quantify this tradeoff at small scale.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019).

    Simpler than LayerNorm: no mean subtraction, no bias.
    Used in LLaMA, Mistral, Gemma. Slightly faster than LayerNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


def build_norm(norm_type: str, d_model: int) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(d_model)
    elif norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
