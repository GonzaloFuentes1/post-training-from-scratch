"""Attention masks and position ids for packed sequences.

Packing concatenates several documents into one fixed-length row to avoid
wasting compute on padding. Done naively, every token can attend to the
documents packed before it and positions keep counting across document
boundaries, so the model is trained on contexts it will never see at
inference ("cross-contamination", Krell et al., 2021). The fix has two parts:

* a block-diagonal causal mask, so that a token only attends to earlier
  tokens of its own document;
* position ids that restart at 0 for every document.

With both, the packed forward pass is mathematically identical to running
each document on its own (verified in ``tests/sft/test_packed_equivalence.py``).

Segment ids encode the layout: ``0`` marks padding, and documents in a row are
numbered ``1, 2, ...`` in order.
"""

from __future__ import annotations

import torch
from torch import Tensor


def block_causal_mask(segment_ids: Tensor) -> Tensor:
    """``[N, T, T]`` boolean mask: query ``i`` may attend to key ``j``.

    Padding positions attend only to themselves. Their outputs are never
    used, but a row with no admissible key would make softmax return NaN, and
    NaN activations poison gradients even when their loss weight is zero.
    """
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must have shape [N, T]")
    seq_len = segment_ids.shape[-1]
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=segment_ids.device).tril()
    is_pad = segment_ids == 0
    mask = same_segment & causal & ~is_pad[:, :, None]
    eye = torch.eye(seq_len, dtype=torch.bool, device=segment_ids.device)
    return mask | (eye & is_pad[:, :, None])


def position_ids_from_segments(segment_ids: Tensor) -> Tensor:
    """Position of each token within its own document; 0 on padding."""
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must have shape [N, T]")
    seq_len = segment_ids.shape[-1]
    index = torch.arange(seq_len, device=segment_ids.device).expand_as(segment_ids)
    is_start = torch.ones_like(segment_ids, dtype=torch.bool)
    is_start[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    start_index = torch.where(is_start, index, torch.zeros_like(index)).cummax(dim=-1).values
    positions = index - start_index
    return torch.where(segment_ids == 0, torch.zeros_like(positions), positions)
