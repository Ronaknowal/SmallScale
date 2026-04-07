"""Positional encoding implementations: RoPE, ALiBi, Learned.

Each has different properties worth ablating:
- RoPE: Rotation-based, good extrapolation, standard in modern LLMs (LLaMA, Mistral)
- ALiBi: No learned params, linear bias on attention, used in BLOOM/MPT
- Learned: Classical approach, limited to max_seq_len, baseline reference
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from Su et al. 2021.

    Key insight: encodes relative position through rotation in complex plane,
    allowing the model to generalize to longer sequences than seen in training.
    We use the standard theta=10000 base frequency.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        # Precompute frequency bands: theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        # Cache cos and sin for efficiency
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len],  # (seq_len, dim)
            self.sin_cached[:seq_len],
        )


def apply_rotary_emb(
    x: torch.Tensor,  # (batch, n_heads, seq_len, head_dim)
    cos: torch.Tensor,  # (seq_len, head_dim)
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings by rotating pairs of dimensions."""
    # Split into even/odd pairs and rotate
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos1, sin1 = cos[..., : x1.shape[-1]], sin[..., : x1.shape[-1]]

    rotated = torch.cat([
        x1 * cos1 - x2 * sin1,
        x2 * cos1 + x1 * sin1,
    ], dim=-1)
    return rotated


class ALiBiSlopes(nn.Module):
    """Attention with Linear Biases (ALiBi) from Press et al. 2022.

    Key insight: instead of adding positional info to embeddings, ALiBi adds
    a linear penalty to attention scores based on distance between tokens.
    Each head gets a different slope, creating multi-scale position sensitivity.
    No learned parameters — the slopes are fixed geometric sequence.
    """

    def __init__(self, n_heads: int):
        super().__init__()
        # Slopes follow geometric sequence: 2^(-8/n_heads * i) for i in 1..n_heads
        slopes = self._get_slopes(n_heads)
        self.register_buffer("slopes", slopes, persistent=False)

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes. Uses the power-of-2 trick for non-power-of-2 heads."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            # For non-power-of-2, interpolate between nearest powers
            closest_power = 2 ** math.floor(math.log2(n_heads))
            slopes = (
                get_slopes_power_of_2(closest_power)
                + get_slopes_power_of_2(2 * closest_power)[0::2][: n_heads - closest_power]
            )
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns bias tensor of shape (n_heads, seq_len, seq_len).

        bias[h, i, j] = -slope_h * |i - j| (causal: only j <= i matters)
        """
        # Distance matrix: positions 0..seq_len-1
        pos = torch.arange(seq_len, device=self.slopes.device)
        # Relative distances (causal: token i attends to j where j <= i)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)  # (seq_len, seq_len)
        rel_pos = rel_pos.abs().float()
        # Apply per-head slopes: (n_heads, 1, 1) * (1, seq_len, seq_len)
        bias = -self.slopes.unsqueeze(1).unsqueeze(1) * rel_pos.unsqueeze(0)
        return bias  # (n_heads, seq_len, seq_len)


class LearnedPositionalEncoding(nn.Module):
    """Classical learned positional embeddings.

    Simple baseline. Cannot extrapolate beyond max_seq_len.
    Added to token embeddings before transformer layers.
    """

    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        return self.embedding(positions)  # (seq_len, d_model)
