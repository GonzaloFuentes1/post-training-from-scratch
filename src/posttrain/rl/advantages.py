r"""Group-relative advantages (GRPO, Dr.GRPO).

GRPO replaces PPO's learned value baseline with the mean reward of a group of
``G`` responses sampled for the same prompt:

.. math::

    A_i = \frac{r_i - \operatorname{mean}(r_{1..G})}{\operatorname{std}(r_{1..G}) + \epsilon}

Mean-centering is what makes this a baseline: it leaves the policy gradient
unbiased while removing the per-prompt reward offset. Dividing by the group
standard deviation is a separate choice. Dr.GRPO (Liu et al., 2025) points out
that it re-weights prompts by the inverse of their reward spread, so prompts
that are almost always solved or almost always failed (small std) receive the
largest updates. ``scale_by_std=False`` gives the Dr.GRPO variant.
"""

from __future__ import annotations

from torch import Tensor


def group_relative_advantages(
    rewards: Tensor,
    group_size: int,
    *,
    scale_by_std: bool = True,
    eps: float = 1e-6,
    unbiased_std: bool = True,
) -> Tensor:
    """Per-response advantages for rewards laid out group by group.

    Args:
        rewards: Shape ``[num_groups * group_size]``; the ``G`` responses of a
            prompt must be contiguous, which is how samplers emit them.
        group_size: Number of responses ``G`` per prompt.
        scale_by_std: Divide by the group standard deviation (GRPO). Set to
            ``False`` for Dr.GRPO.
        eps: Added to the standard deviation so that a group with identical
            rewards yields zero advantages instead of ``0 / 0``. Such groups
            carry no learning signal and must not produce NaNs.
        unbiased_std: Use Bessel's correction, matching ``torch.std`` defaults
            used by common open-source GRPO implementations.

    Returns:
        Advantages with the same shape as ``rewards``.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape {tuple(rewards.shape)}")
    if group_size < 1 or rewards.numel() % group_size != 0:
        raise ValueError(f"{rewards.numel()} rewards cannot be split into groups of {group_size}")
    if scale_by_std and unbiased_std and group_size < 2:
        raise ValueError("unbiased std needs at least two responses per group")

    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if scale_by_std:
        std = grouped.std(dim=1, keepdim=True, correction=1 if unbiased_std else 0)
        centered = centered / (std + eps)
    return centered.reshape(-1)
