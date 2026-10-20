r"""Offline preference objectives: DPO, conservative DPO, IPO and SimPO.

All losses take per-sequence log-likelihoods (see
:func:`posttrain.preference.logprobs.sequence_logprobs`) and return one loss
per preference pair, leaving the batch reduction to the caller.

Notation: :math:`y_w` is the chosen and :math:`y_l` the rejected response,
and the DPO *margin* is the difference of policy-to-reference log-ratios

.. math::

    m = \log\frac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}
      - \log\frac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}.

DPO (Rafailov et al., 2023) is the Bradley-Terry likelihood of the preference
under the implicit reward :math:`r(y) = \beta \log \pi_\theta(y) / \pi_{\mathrm{ref}}(y)`:

.. math::

    \mathcal{L}_{\mathrm{DPO}} = -\log \sigma(\beta m),
    \qquad \frac{\partial \mathcal{L}_{\mathrm{DPO}}}{\partial m} = -\beta\, \sigma(-\beta m).

The gradient never vanishes for finite :math:`m`, so on deterministic
preferences DPO keeps pushing the margin to infinity. Conservative DPO and IPO
are two fixes; SimPO removes the reference model altogether.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def dpo_margin(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
) -> Tensor:
    """Difference of policy/reference log-ratios between chosen and rejected."""
    return (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)


def implicit_rewards(policy_logps: Tensor, ref_logps: Tensor, beta: float) -> Tensor:
    """DPO's implicit reward ``beta * log(pi / pi_ref)``, useful for logging accuracy."""
    return beta * (policy_logps - ref_logps)


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tensor:
    r"""DPO loss, optionally with label smoothing (conservative DPO).

    With ``label_smoothing = eps`` the preference label is assumed to be flipped
    with probability ``eps``:

    .. math::

        \mathcal{L} = -(1 - \epsilon) \log\sigma(\beta m) - \epsilon \log\sigma(-\beta m).

    Its gradient is zero at the finite margin
    :math:`m^* = \beta^{-1}\log((1 - \epsilon) / \epsilon)`, which bounds how
    far the policy moves from the reference on noisy labels.

    ``F.logsigmoid`` is used instead of ``log(sigmoid(x))`` because the latter
    underflows to ``log(0) = -inf`` for margins of a few tens of nats.
    """
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    logits = beta * dpo_margin(
        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
    )
    return -(1.0 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)


def ipo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    *,
    tau: float = 0.1,
) -> Tensor:
    r"""IPO loss (Azar et al., 2023): squared regression of the margin onto ``1 / (2 tau)``.

    .. math::

        \mathcal{L}_{\mathrm{IPO}} = \Big(m - \frac{1}{2\tau}\Big)^2

    Replacing the log-sigmoid by a square gives the margin a finite target, so
    unlike DPO the objective penalises overshooting it. Smaller ``tau`` means a
    larger target margin, i.e. weaker regularisation towards the reference.
    The log-likelihoods are used as given; some implementations pass
    per-token averages instead of sums.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    margin = dpo_margin(
        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
    )
    return (margin - 1.0 / (2.0 * tau)).square()


def simpo_loss(
    policy_chosen_avg_logps: Tensor,
    policy_rejected_avg_logps: Tensor,
    *,
    beta: float = 2.0,
    gamma: float = 1.0,
) -> Tensor:
    r"""SimPO loss (Meng et al., 2024): reference-free, length-normalised DPO.

    .. math::

        \mathcal{L}_{\mathrm{SimPO}} = -\log\sigma\Big(
            \frac{\beta}{|y_w|}\log\pi_\theta(y_w) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l) - \gamma
        \Big)

    The inputs are *average* per-token log-probabilities (``|y|`` is already
    divided out). The implicit reward is the quantity used at generation time,
    so training and decoding rank responses the same way, and no reference
    model has to be kept in memory. Without the length normalisation the
    summed log-likelihood would favour short responses; ``gamma`` is a target
    reward margin that makes the loss keep separating pairs that are already
    ranked correctly.
    """
    logits = beta * (policy_chosen_avg_logps - policy_rejected_avg_logps) - gamma
    return -F.logsigmoid(logits)
