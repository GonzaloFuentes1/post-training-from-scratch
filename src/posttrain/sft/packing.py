"""Sequence packing with first-fit-decreasing bin packing.

Packing many short examples into fixed-length rows is the main throughput
lever for SFT on chat data, where lengths are heavily skewed. Choosing which
examples share a row is one-dimensional bin packing, which is NP-hard;
first-fit-decreasing (FFD) is the standard heuristic because it is simple,
deterministic and provably close to optimal: it never uses more than
``11/9 * OPT + 6/9`` rows (Dósa, 2007).

This implementation scans open bins linearly, which is ``O(n * bins)``. That is
fine for a reference; production packers find the first fitting bin with a
segment tree in ``O(n log n)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from posttrain.sft.attention import position_ids_from_segments


def first_fit_decreasing(lengths: Sequence[int], capacity: int) -> list[list[int]]:
    """Group item indices into bins whose total length is at most ``capacity``.

    Items are placed longest first (ties broken by index, so the result is
    deterministic) into the first bin with enough room. Returns lists of
    indices into ``lengths``, one list per bin, in order of bin creation.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    for i, length in enumerate(lengths):
        if not 0 < length <= capacity:
            raise ValueError(f"item {i} has length {length}, outside (0, {capacity}]")

    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    bins: list[list[int]] = []
    free: list[int] = []
    for i in order:
        for b, room in enumerate(free):
            if lengths[i] <= room:
                bins[b].append(i)
                free[b] -= lengths[i]
                break
        else:
            bins.append([i])
            free.append(capacity - lengths[i])
    return bins


@dataclass(frozen=True)
class PackedBatch:
    """Packed rows ready for a causal LM forward pass.

    Attributes:
        input_ids: ``[N, C]`` token ids, ``pad_id`` after the last document.
        labels: ``[N, C]`` next-token targets aligned with ``input_ids``
            (``labels[:, t]`` is predicted from position ``t``), set to
            ``ignore_index`` at document ends, on padding and wherever the
            loss mask excludes the target token.
        position_ids: ``[N, C]`` positions restarting at 0 for each document.
        segment_ids: ``[N, C]`` 0 for padding, ``1..k`` for the documents of a row.
    """

    input_ids: Tensor
    labels: Tensor
    position_ids: Tensor
    segment_ids: Tensor


def pack_sequences(
    sequences: Sequence[Sequence[int]],
    capacity: int,
    *,
    loss_masks: Sequence[Sequence[bool]] | None = None,
    pad_id: int = 0,
    ignore_index: int = -100,
) -> PackedBatch:
    """Pack token sequences into rows of length ``capacity`` with FFD.

    ``loss_masks[i][j]`` says whether token ``j`` of sequence ``i`` is a
    training target (for example only assistant tokens in chat SFT). The last
    token of each document never gets a target: predicting the first token of
    the *next* packed document would teach the model a transition that does
    not exist in the data.
    """
    if loss_masks is not None and len(loss_masks) != len(sequences):
        raise ValueError("loss_masks must have one entry per sequence")
    bins = first_fit_decreasing([len(s) for s in sequences], capacity)

    shape = (len(bins), capacity)
    input_ids = torch.full(shape, pad_id, dtype=torch.long)
    labels = torch.full(shape, ignore_index, dtype=torch.long)
    segment_ids = torch.zeros(shape, dtype=torch.long)
    for row, members in enumerate(bins):
        offset = 0
        for segment, index in enumerate(members, start=1):
            tokens = list(sequences[index])
            mask = [True] * len(tokens) if loss_masks is None else list(loss_masks[index])
            if len(mask) != len(tokens):
                raise ValueError(f"loss mask of sequence {index} has the wrong length")
            end = offset + len(tokens)
            input_ids[row, offset:end] = torch.tensor(tokens)
            segment_ids[row, offset:end] = segment
            for j in range(len(tokens) - 1):
                if mask[j + 1]:
                    labels[row, offset + j] = tokens[j + 1]
            offset = end
    return PackedBatch(
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids_from_segments(segment_ids),
        segment_ids=segment_ids,
    )
