r"""Single-sample KL estimators (Schulman, "Approximating KL Divergence").

During RL fine-tuning we want :math:`\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})`
but only observe the log-probabilities of the tokens that were actually
sampled from :math:`\pi_\theta`. With :math:`x \sim \pi_\theta` and the ratio
:math:`r(x) = \pi_{\mathrm{ref}}(x) / \pi_\theta(x)` the three classic estimators are

.. math::

    k_1 = -\log r, \qquad
    k_2 = \tfrac{1}{2} (\log r)^2, \qquad
    k_3 = (r - 1) - \log r.

* ``k1`` is unbiased but has high variance and is negative for many samples,
  which is odd for a quantity that is non-negative in expectation.
* ``k2`` is always non-negative and has low variance, but it is biased; its
  expectation is an f-divergence that agrees with KL only to second order,
  i.e. when the two distributions are close.
* ``k3`` adds the control variate :math:`r - 1` (which has zero mean under
  :math:`\pi_\theta`) to ``k1``. It stays unbiased, is non-negative for every
  sample because :math:`\log r \le r - 1`, and usually has lower variance.
  GRPO uses ``k3`` as its per-token KL penalty.

The bias claims are verified by Monte Carlo against exact categorical KLs in
``tests/rl/test_kl.py``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

KLEstimator = Literal["k1", "k2", "k3"]


def kl_estimate(logp: Tensor, ref_logp: Tensor, estimator: KLEstimator = "k3") -> Tensor:
    """Per-sample estimate of ``KL(pi || pi_ref)`` from samples drawn from ``pi``.

    Args:
        logp: Log-probabilities of the sampled tokens under the sampling policy.
        ref_logp: Log-probabilities of the same tokens under the reference.
        estimator: ``"k1"``, ``"k2"`` or ``"k3"``.

    Returns:
        A tensor with the shape of ``logp``; average it over samples (with the
        appropriate mask) to obtain the KL estimate.
    """
    if logp.shape != ref_logp.shape:
        raise ValueError(f"shape mismatch: {tuple(logp.shape)} vs {tuple(ref_logp.shape)}")
    log_ratio = ref_logp - logp
    if estimator == "k1":
        return -log_ratio
    if estimator == "k2":
        return 0.5 * log_ratio.square()
    if estimator == "k3":
        # expm1 keeps precision when the policies are close and log_ratio ~ 0.
        return torch.expm1(log_ratio) - log_ratio
    raise ValueError(f"unknown KL estimator {estimator!r}")
