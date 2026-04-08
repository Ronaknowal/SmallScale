"""Multi-Head Attention (MHA) and Grouped-Query Attention (GQA).

GQA (Ainslie et al. 2023) shares KV heads across query heads, reducing
KV cache size and compute. MHA is the special case where n_kv_heads == n_heads.
MQA is the other extreme where n_kv_heads == 1.

The key research question at small scale: does GQA's quality-speed tradeoff
still hold at 150M params, or does it only matter at 7B+?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from model.positional import RotaryEmbedding, apply_rotary_emb, ALiBiSlopes


class Attention(nn.Module):
    """Unified attention supporting MHA, GQA, and MQA via n_kv_heads parameter."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        pos_encoding: str = "rope",
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # how many Q heads per KV head
        self.scale = head_dim ** -0.5
        self.pos_encoding = pos_encoding

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # Positional encoding components
        if pos_encoding == "rope":
            self.rotary = RotaryEmbedding(head_dim, max_seq_len)
        elif pos_encoding == "alibi":
            self.alibi = ALiBiSlopes(n_heads)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match number of Q heads for GQA."""
        if self.n_rep == 1:
            return x  # MHA: no repetition needed
        bs, n_kv, seq, hd = x.shape
        x = x[:, :, None, :, :].expand(bs, n_kv, self.n_rep, seq, hd)
        return x.reshape(bs, self.n_heads, seq, hd)

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None,  # causal mask
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_heads, T, head_dim), k/v: (B, n_kv_heads, T, head_dim)

        # Apply positional encoding
        if self.pos_encoding == "rope":
            cos, sin = self.rotary(T)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        elif self.pos_encoding == "learned":
            pass  # Learned pos is added to embeddings, not here

        # Expand KV heads for GQA
        k = self._repeat_kv(k)  # (B, n_heads, T, head_dim)
        v = self._repeat_kv(v)

        # Use Flash Attention via PyTorch's SDPA — never materializes (B, H, T, T) matrix.
        # Manual path needed only for ALiBi (which requires a custom attention bias).
        if self.pos_encoding == "alibi":
            # ALiBi requires adding bias to attention scores — use manual path
            attn = (q @ k.transpose(-2, -1)) * self.scale
            alibi_bias = self.alibi(T)  # (n_heads, T, T)
            attn = attn + alibi_bias.unsqueeze(0)
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = attn @ v
        else:
            # Flash Attention path — O(T) memory instead of O(T^2)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,       # use is_causal flag instead
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)
