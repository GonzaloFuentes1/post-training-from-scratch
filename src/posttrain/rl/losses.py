r"""Clipped policy-gradient losses: GRPO, DAPO and Dr.GRPO.

All three objectives share the PPO per-token surrogate

.. math::

    \ell_{i,t} = -\min\big(\rho_{i,t} A_i,\;
        \operatorname{clip}(\rho_{i,t}, 1 - \epsilon_l, 1 + \epsilon_h)\, A_i\big)
        + \beta\, \widehat{\mathrm{KL}}_{i,t},
    \qquad \rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}
                             {\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,<t})}

and differ mainly in how the per-token terms of ``G`` responses are reduced
to a scalar:

=============  ==========================================================  =================
normalization  loss                                                        used by
=============  ==========================================================  =================
``sequence``   :math:`\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \ell_{i,t}`  GRPO (original)
``token``      :math:`\frac{1}{\sum_i |o_i|}\sum_i\sum_t \ell_{i,t}`       DAPO
``constant``   :math:`\frac{1}{G \cdot L_{\max}}\sum_i\sum_t \ell_{i,t}`   Dr.GRPO
=============  ==========================================================  =================

The choice matters because it sets the gradient weight of every token. Under
``sequence`` normalization a token of response ``i`` is weighted by
:math:`A_i / (G |o_i|)`: for a correct answer (``A > 0``) short responses are
reinforced more per token, and for a wrong answer (``A < 0``) long responses
are penalised less per token, which pushes incorrect responses to grow
longer. ``token`` normalization removes the per-response dependence but its
denominator still varies with the batch; ``constant`` normalization makes the
weight :math:`A_i / (G L_{\max})` independent of any length.

Gradient accumulation
---------------------

All three denominators belong to the *whole* optimizer batch: ``G`` (in
general the number of responses ``B``) and :math:`\sum_i |o_i|` count every
response that contributes to one optimizer step. If that batch is split into
micro-batches and each one is reduced with its own counts, the result is a
mean of per-micro-batch means, the error that :mod:`posttrain.sft.loss`
describes for SFT: under ``token`` normalization, for example, a token in a
micro-batch with few response tokens is over-weighted. Pass the global counts
as ``num_sequences`` and ``num_tokens`` to every micro-batch and sum (do not
average) the micro-batch losses; the accumulated gradient then equals the
full-batch gradient. The per-response means of ``sequence`` normalization
are local to each response, so it only needs the global ``num_sequences``.
Under data-parallel training the counts are summed over ranks, and because
DDP averages gradients across ranks the same ``world_size`` correction as in
:mod:`posttrain.sft.loss` applies.

A response with no valid tokens adds 0 to the numerator of every
normalization, but under ``sequence`` and ``constant`` it still counts in the
default divisor ``B``, as in common open-source implementations. Pass
``num_sequences`` explicitly to count only non-empty responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from posttrain.ops import masked_mean, masked_sum
from posttrain.rl.kl import KLEstimator, kl_estimate

Normalization = Literal["sequence", "token", "constant"]


@dataclass(frozen=True)
class PolicyLossConfig:
    """Hyper-parameters that distinguish GRPO, DAPO and Dr.GRPO.

    Attributes:
        normalization: How per-token losses are reduced (see module docstring).
        eps_low: Lower clip range, the ratio is clipped at ``1 - eps_low``.
        eps_high: Upper clip range. DAPO's "clip-higher" raises it above
            ``eps_low`` so that low-probability tokens with positive advantage
            can grow faster, which counteracts entropy collapse.
        beta: Weight of the per-token KL penalty to the reference policy.
        kl_estimator: Which Schulman estimator implements the penalty. As a
            differentiated loss term, ``"k1"`` has zero expected gradient
            on-policy, ``"k2"`` gives the gradient of ``KL(pi || pi_ref)`` and
            ``"k3"`` (the GRPO choice) that of ``KL(pi_ref || pi)``; see
            :mod:`posttrain.rl.kl`.
        scale_advantages_by_std: Whether group advantages are divided by the
            group std. The loss does not use it; it is kept here so that one
            object fully describes a recipe.
    """

    normalization: Normalization = "sequence"
    eps_low: float = 0.2
    eps_high: float = 0.2
    beta: float = 0.0
    kl_estimator: KLEstimator = "k3"
    scale_advantages_by_std: bool = True

    @classmethod
    def grpo(cls) -> PolicyLossConfig:
        """GRPO as in DeepSeekMath: sequence-level mean, symmetric clip, k3 KL."""
        return cls(normalization="sequence", eps_low=0.2, eps_high=0.2, beta=0.04)

    @classmethod
    def dapo(cls) -> PolicyLossConfig:
        """DAPO: token-level mean, clip-higher (0.2 / 0.28), no KL penalty."""
        return cls(normalization="token", eps_low=0.2, eps_high=0.28, beta=0.0)

    @classmethod
    def dr_grpo(cls) -> PolicyLossConfig:
        """Dr.GRPO: constant normalizer, no std scaling of advantages, no KL."""
        return cls(normalization="constant", beta=0.0, scale_advantages_by_std=False)


@dataclass(frozen=True)
class PolicyLossOutput:
    """Scalar loss plus diagnostics that are cheap to compute alongside it."""

    loss: Tensor
    per_token_loss: Tensor
    clip_fraction: Tensor
    kl: Tensor | None


def clipped_surrogate(
    logp: Tensor,
    old_logp: Tensor,
    advantages: Tensor,
    mask: Tensor,
    eps_low: float,
    eps_high: float,
) -> tuple[Tensor, Tensor]:
    """PPO surrogate objective per token and the mask of tokens it clips.

    The returned ``clipped`` mask marks tokens whose gradient is zero because
    the pessimistic ``min`` selected the clipped branch, which is the quantity
    usually logged as the clip fraction.
    """
    # Compute the log-ratio only on valid tokens: padded log-probs may be -inf
    # in both policies and -inf - (-inf) is NaN.
    log_ratio = torch.where(mask, logp - old_logp, torch.zeros_like(logp))
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * advantages
    objective = torch.minimum(unclipped, clipped)
    is_clipped = (clipped < unclipped) & mask
    return objective, is_clipped


def _positive_count(name: str, value: int | None, default: int) -> int:
    if value is None:
        return default
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def aggregate_token_losses(
    per_token_loss: Tensor,
    mask: Tensor,
    normalization: Normalization,
    max_tokens: int | None = None,
    *,
    num_sequences: int | None = None,
    num_tokens: int | None = None,
) -> Tensor:
    """Reduce ``[num_sequences, seq_len]`` token losses to a scalar.

    Args:
        per_token_loss: ``[B, T]`` per-token losses.
        mask: ``[B, T]`` true on response tokens.
        normalization: ``"sequence"``, ``"token"`` or ``"constant"``.
        max_tokens: The generation budget :math:`L_{\\max}` for ``constant``
            normalization. It must be a true constant of the run (for example
            ``max_new_tokens``), not the padded length of the current batch,
            otherwise the length dependence that Dr.GRPO removes comes back
            through padding.
        num_sequences: Number of responses in the whole optimizer batch, used
            by ``sequence`` and ``constant``. Defaults to ``B``, which is only
            correct when this call sees the whole batch.
        num_tokens: Number of response tokens in the whole optimizer batch,
            used by ``token``. Defaults to ``mask.sum()``, which is only
            correct when this call sees the whole batch.

    With gradient accumulation, pass the global counts to every micro-batch
    and sum the returned losses (see the module docstring).
    """
    mask = mask.bool()
    if normalization == "sequence":
        sequences = _positive_count("num_sequences", num_sequences, per_token_loss.shape[0])
        return masked_mean(per_token_loss, mask, dim=-1).sum() / sequences
    if normalization == "token":
        tokens = _positive_count("num_tokens", num_tokens, max(int(mask.sum()), 1))
        return masked_sum(per_token_loss, mask) / tokens
    if normalization == "constant":
        if max_tokens is None or max_tokens <= 0:
            raise ValueError("constant normalization needs a positive max_tokens")
        sequences = _positive_count("num_sequences", num_sequences, per_token_loss.shape[0])
        return masked_sum(per_token_loss, mask) / (sequences * max_tokens)
    raise ValueError(f"unknown normalization {normalization!r}")


def policy_gradient_loss(
    logp: Tensor,
    old_logp: Tensor,
    advantages: Tensor,
    mask: Tensor,
    config: PolicyLossConfig,
    *,
    ref_logp: Tensor | None = None,
    max_tokens: int | None = None,
    num_sequences: int | None = None,
    num_tokens: int | None = None,
) -> PolicyLossOutput:
    """Clipped policy-gradient loss for a batch of sampled responses.

    Args:
        logp: ``[B, T]`` log-probs of the sampled tokens under the current policy.
        old_logp: ``[B, T]`` log-probs under the policy that generated the
            samples. It is detached here because the ratio must only
            differentiate through the current policy.
        advantages: ``[B]`` per-response (outcome) or ``[B, T]`` per-token advantages.
        mask: ``[B, T]`` true on response tokens. Prompt and padding positions
            must be false so they contribute neither loss nor gradient.
        config: Recipe hyper-parameters, see :class:`PolicyLossConfig`.
        ref_logp: ``[B, T]`` reference log-probs; required when ``config.beta > 0``.
        max_tokens: Generation budget for ``constant`` normalization.
        num_sequences: Responses in the whole optimizer batch; pass it when
            this call sees one micro-batch. See :func:`aggregate_token_losses`.
        num_tokens: Response tokens in the whole optimizer batch; pass it when
            this call sees one micro-batch. See :func:`aggregate_token_losses`.

    The diagnostics ``clip_fraction`` and ``kl`` are always local means over
    the tokens of this call.
    """
    if logp.shape != old_logp.shape or logp.shape != mask.shape:
        raise ValueError("logp, old_logp and mask must share a [B, T] shape")
    mask = mask.bool()
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    advantages = advantages.detach()

    objective, is_clipped = clipped_surrogate(
        logp, old_logp.detach(), advantages, mask, config.eps_low, config.eps_high
    )
    per_token_loss = -objective

    kl: Tensor | None = None
    if config.beta != 0.0:
        if ref_logp is None:
            raise ValueError("beta > 0 requires ref_logp")
        # Zero both inputs off-mask so that padded -inf values cannot produce
        # NaN gradients through exp() in the backward pass.
        zeros = torch.zeros_like(logp)
        safe_logp = torch.where(mask, logp, zeros)
        safe_ref = torch.where(mask, ref_logp.detach(), zeros)
        per_token_kl = kl_estimate(safe_logp, safe_ref, config.kl_estimator)
        per_token_loss = per_token_loss + config.beta * per_token_kl
        kl = masked_mean(per_token_kl.detach(), mask)

    loss = aggregate_token_losses(
        per_token_loss,
        mask,
        config.normalization,
        max_tokens,
        num_sequences=num_sequences,
        num_tokens=num_tokens,
    )
    per_token_loss = torch.where(mask, per_token_loss, torch.zeros_like(per_token_loss))
    clip_fraction = masked_mean(is_clipped.to(logp.dtype), mask)
    return PolicyLossOutput(
        loss=loss, per_token_loss=per_token_loss, clip_fraction=clip_fraction, kl=kl
    )
