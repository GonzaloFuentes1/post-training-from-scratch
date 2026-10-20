"""Sequence log-likelihoods of responses.

Preference objectives compare whole responses, so they consume one scalar per
sequence: either the summed log-probability of the response tokens (DPO, IPO)
or its per-token average (SimPO). The prompt must be excluded through the
mask: it is shared by the chosen and rejected responses and only adds noise
to the comparison.
"""

from __future__ import annotations

from torch import Tensor

from posttrain.ops import masked_mean, masked_sum, token_logprobs


def sequence_logprobs(
    logits: Tensor, labels: Tensor, mask: Tensor, *, average: bool = False
) -> Tensor:
    """Summed or averaged log-probability of the masked tokens of each sequence.

    Args:
        logits: ``[B, T, V]`` logits where position ``t`` predicts ``labels[:, t]``.
        labels: ``[B, T]`` target ids; masked positions may hold any value.
        mask: ``[B, T]`` true on response tokens.
        average: Return the mean over response tokens instead of the sum. The
            average is what makes SimPO insensitive to response length.

    Returns:
        ``[B]`` log-likelihoods.
    """
    mask = mask.bool()
    logp = token_logprobs(logits, labels, mask)
    if average:
        return masked_mean(logp, mask, dim=-1)
    return masked_sum(logp, mask, dim=-1)
