"""Feed-forward network variants: SwiGLU and GELU.

SwiGLU (Shazeer 2020) is used in LLaMA/Mistral and generally outperforms GELU
at equivalent compute, but uses 50% more parameters (3 matrices vs 2).
The research question: at small scale, does the parameter efficiency of GELU
close the gap, or does SwiGLU still win?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU: gate(x) * up(x) where gate uses SiLU activation.

    Uses 3 linear projections: gate, up, down.
    Total params: 3 * d_model * d_ff (vs 2 * d_model * d_ff for standard).
    To match param count, d_ff is typically set to 8/3 * d_model instead of 4 * d_model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GELUFFN(nn.Module):
    """Standard FFN with GELU activation.

    Uses 2 linear projections: up, down.
    Simpler and fewer params, but generally slightly worse than SwiGLU.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x))))


def build_ffn(activation: str, d_model: int, d_ff: int, dropout: float = 0.0) -> nn.Module:
    if activation == "swiglu":
        return SwiGLUFFN(d_model, d_ff, dropout)
    elif activation == "gelu":
        return GELUFFN(d_model, d_ff, dropout)
    else:
        raise ValueError(f"Unknown activation: {activation}")
