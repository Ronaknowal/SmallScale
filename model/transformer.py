"""Full transformer language model assembled from components.

Design decisions documented inline for the research write-up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from model.attention import Attention
from model.feedforward import build_ffn
from model.norms import build_norm
from model.positional import LearnedPositionalEncoding
from utils.config import ModelConfig


class TransformerBlock(nn.Module):
    """Single transformer block with configurable norm position."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_position = cfg.norm_position

        self.attn_norm = build_norm(cfg.norm_type, cfg.d_model)
        self.ffn_norm = build_norm(cfg.norm_type, cfg.d_model)

        self.attn = Attention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            pos_encoding=cfg.pos_encoding,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )
        self.ffn = build_ffn(cfg.activation, cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.norm_position == "pre":
            # Pre-norm: norm -> sublayer -> residual (LLaMA style)
            x = x + self.attn(self.attn_norm(x), mask=mask)
            x = x + self.ffn(self.ffn_norm(x))
        else:
            # Post-norm: sublayer -> residual -> norm (original transformer)
            x = self.attn_norm(x + self.attn(x, mask=mask))
            x = self.ffn_norm(x + self.ffn(x))
        return x


class Transformer(nn.Module):
    """Full causal language model.

    Architecture follows the LLaMA family design by default:
    - RMSNorm (pre-norm)
    - RoPE
    - SwiGLU
    - GQA (configurable)
    - Tied embeddings

    All components are swappable via config for ablation studies.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Token embeddings
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Learned positional embeddings (only if using "learned" pos encoding)
        self.pos_emb = None
        if cfg.pos_encoding == "learned":
            self.pos_emb = LearnedPositionalEncoding(cfg.d_model, cfg.max_seq_len)

        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ])

        # Final norm
        self.final_norm = build_norm(cfg.norm_type, cfg.d_model)

        # LM head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self._init_weights()

        # Build causal mask buffer
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_params_non_emb = n_params - self.tok_emb.weight.numel()
        print(f"Model initialized: {n_params/1e6:.1f}M total params, "
              f"{n_params_non_emb/1e6:.1f}M non-embedding params")

    def _init_weights(self):
        """Initialize weights following GPT-2 / LLaMA conventions.

        Key choices:
        - Normal init with std=0.02 for embeddings and most weights
        - Scaled init for output projections: std=0.02 / sqrt(2*n_layers)
          This prevents the residual stream from growing with depth.
        """
        for name, p in self.named_parameters():
            if p.dim() == 1:
                continue  # skip biases and norm weights
            if "o_proj" in name or "down_proj" in name:
                # Scale residual-contributing projections
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len)
        targets: Optional[torch.Tensor] = None,  # (batch, seq_len)
    ) -> dict:
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, f"Sequence length {T} exceeds max {self.cfg.max_seq_len}"

        # Token + position embeddings
        x = self.tok_emb(input_ids)  # (B, T, d_model)
        if self.pos_emb is not None:
            x = x + self.pos_emb(T)
        x = self.drop(x)

        # Causal mask for this sequence length
        mask = self.causal_mask[:, :, :T, :T]

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # for padding if needed
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple autoregressive generation for qualitative evaluation."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.cfg.max_seq_len:]
            out = self(idx_cond)
            logits = out["logits"][:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
