"""Masked reductions and per-token log-probabilities.

Every objective in this package reduces over a ragged set of tokens stored in a
padded tensor. Getting the denominator or the mask of that reduction wrong is
the most common silent bug in post-training code: the loss still decreases, it
just optimises a different objective. These helpers exist so that the masking
logic is written and tested once.
"""

from __future__ import annotations

import torch
from torch import Tensor

Dim = int | tuple[int, ...] | None


def _check_mask(values: Tensor, mask: Tensor) -> None:
    if values.shape != mask.shape:
        raise ValueError(f"values {tuple(values.shape)} and mask {tuple(mask.shape)} differ")


def masked_sum(values: Tensor, mask: Tensor, dim: Dim = None) -> Tensor:
    """Sum of ``values`` over positions where ``mask`` is true.

    ``torch.where`` is used instead of ``values * mask`` because padded
    positions may legitimately hold non-finite numbers (for example ``-inf``
    log-probabilities of a pad token). ``inf * 0`` is NaN and would poison both
    the value and the gradient; selecting with ``where`` keeps the masked
    positions out of the computation graph entirely.
    """
    _check_mask(values, mask)
    selected = torch.where(mask.bool(), values, torch.zeros_like(values))
    return selected.sum() if dim is None else selected.sum(dim=dim)


def masked_mean(values: Tensor, mask: Tensor, dim: Dim = None) -> Tensor:
    """Mean of ``values`` over positions where ``mask`` is true.

    A slice with no valid positions returns 0 rather than NaN. That situation
    is routine (a fully truncated response, an empty micro-batch shard) and a
    single NaN would otherwise propagate into every parameter on the next
    optimizer step.
    """
    total = masked_sum(values, mask, dim)
    count = mask.bool().sum() if dim is None else mask.bool().sum(dim=dim)
    return total / count.clamp_min(1).to(total.dtype)


def token_logprobs(logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """Log-probability of each target token under ``logits``.

    ``logits[..., t, :]`` must already be aligned with ``targets[..., t]``; the
    causal shift is the caller's job because it differs between packed and
    unpacked layouts.

    The value is computed as ``logit[target] - logsumexp(logits)`` rather than
    by gathering from ``log_softmax(logits)``. Both are exact, but the former
    never materialises a second ``[..., vocab]`` tensor of log-probabilities,
    which dominates memory for large vocabularies.

    Positions where ``mask`` is false return exactly 0 and receive no gradient.
    Their target ids may be arbitrary (for example ``-100``), so callers do
    not need to sanitise padding before calling this function.
    """
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} do not align with targets {tuple(targets.shape)}"
        )
    if mask is None:
        if (targets < 0).any():
            raise ValueError("negative target ids require an explicit mask")
        safe_targets = targets
    else:
        _check_mask(targets, mask)
        safe_targets = torch.where(mask.bool(), targets, torch.zeros_like(targets))
    picked = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    logp = picked - torch.logsumexp(logits, dim=-1)
    if mask is None:
        return logp
    return torch.where(mask.bool(), logp, torch.zeros_like(logp))
