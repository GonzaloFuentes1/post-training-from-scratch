"""A deliberately small GPT used to test objectives end to end.

The objectives in this package are model-agnostic, but some invariants (packed
versus unpacked equivalence, gradient accumulation) only mean something when
gradients flow through attention. This model exists for those tests and for
the toy examples; it is not meant for real training.

Two choices make it suitable for exact tests:

* attention is written out explicitly and accepts an arbitrary boolean
  ``[B, T, T]`` mask, so packed sequences can be isolated with a
  block-diagonal mask;
* position embeddings are looked up from explicit ``position_ids``, so each
  packed document can restart at position 0 exactly as it would unpacked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TinyGPTConfig:
    vocab_size: int
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 2
    max_positions: int = 64


class _Block(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln_mlp = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        batch, seq_len, d_model = x.shape
        head_dim = d_model // self.n_heads
        q, k, v = self.qkv(self.ln_attn(x)).split(d_model, dim=-1)
        q, k, v = (
            t.view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2) for t in (q, k, v)
        )
        scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
        scores = scores.masked_fill(~attention_mask[:, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, d_model)
        x = x + self.proj(out)
        return x + self.mlp(self.ln_mlp(x))


class TinyGPT(nn.Module):
    """Pre-LayerNorm decoder-only transformer with learned positions."""

    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_positions, cfg.d_model)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        *,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Return ``[B, T, vocab]`` logits.

        Args:
            input_ids: ``[B, T]`` token ids.
            position_ids: ``[B, T]``; defaults to ``0..T-1`` for every row.
            attention_mask: ``[B, T, T]`` boolean, true where query ``i`` may
                attend to key ``j``. Defaults to a plain causal mask. Every
                row must allow at least one key or softmax returns NaN.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).expand(batch, seq_len)
        if attention_mask is None:
            causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
            attention_mask = causal.expand(batch, seq_len, seq_len)
        x = self.tok_emb(input_ids) + self.pos_emb(position_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.head(self.ln_f(x))
