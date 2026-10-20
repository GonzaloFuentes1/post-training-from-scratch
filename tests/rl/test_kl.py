"""Monte Carlo checks of the bias of Schulman's KL estimators.

The exact KL between two categorical distributions is available in closed
form (``torch.distributions.kl_divergence`` is used as an independent
reference), so the estimators can be checked against the truth rather than
against each other.
"""

import pytest
import torch
from torch.distributions import Categorical, kl_divergence

from posttrain.rl import kl_estimate

NUM_SAMPLES = 400_000
# Five standard errors: a correct estimator fails with probability ~6e-7.
Z = 5.0


def _sample_estimates(q: torch.Tensor, p: torch.Tensor, estimator: str, seed: int):
    """Draw x ~ q and return per-sample estimates of KL(q || p)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.multinomial(q, NUM_SAMPLES, replacement=True, generator=gen)
    return kl_estimate(q.log()[x], p.log()[x], estimator)


def _far_pair():
    q = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    p = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float64)
    return q, p


def _close_pair():
    gen = torch.Generator().manual_seed(11)
    logits = torch.randn(10, generator=gen, dtype=torch.float64)
    q = torch.softmax(logits, -1)
    p = torch.softmax(logits + 0.1 * torch.randn(10, generator=gen, dtype=torch.float64), -1)
    return q, p


def _exact_kl(q: torch.Tensor, p: torch.Tensor) -> float:
    return kl_divergence(Categorical(probs=q), Categorical(probs=p)).item()


@pytest.mark.parametrize("estimator", ["k1", "k3"])
@pytest.mark.parametrize("pair", [_far_pair, _close_pair])
def test_k1_and_k3_are_unbiased(estimator, pair):
    q, p = pair()
    samples = _sample_estimates(q, p, estimator, seed=0)

    stderr = samples.std().item() / NUM_SAMPLES**0.5
    assert abs(samples.mean().item() - _exact_kl(q, p)) < Z * stderr


def test_k2_is_biased_for_distant_distributions():
    q, p = _far_pair()
    samples = _sample_estimates(q, p, "k2", seed=1)
    stderr = samples.std().item() / NUM_SAMPLES**0.5
    # E_q[k2] = 0.5 * E_q[(log p/q)^2] exactly; the MC mean must hit that value...
    expected_k2 = (0.5 * q * (p.log() - q.log()).square()).sum().item()
    assert abs(samples.mean().item() - expected_k2) < Z * stderr
    # ...which is far from the true KL (1.502 vs 1.102 for this pair).
    assert expected_k2 - _exact_kl(q, p) > 100 * stderr


def test_k2_bias_vanishes_for_close_distributions():
    q, p = _close_pair()
    expected_k2 = (0.5 * q * (p.log() - q.log()).square()).sum().item()
    exact = _exact_kl(q, p)

    # The bias is third order in the log-ratio, so it is tiny relative to KL.
    assert abs(expected_k2 - exact) / exact < 0.05


def test_k3_has_lower_variance_than_k1_for_close_distributions():
    q, p = _close_pair()

    var_k1 = _sample_estimates(q, p, "k1", seed=2).var()
    var_k3 = _sample_estimates(q, p, "k3", seed=2).var()

    assert var_k3 < 0.1 * var_k1


def test_k2_and_k3_are_non_negative_per_sample_while_k1_is_not():
    gen = torch.Generator().manual_seed(3)
    logp = torch.randn(10_000, generator=gen, dtype=torch.float64)
    ref_logp = torch.randn(10_000, generator=gen, dtype=torch.float64)

    assert (kl_estimate(logp, ref_logp, "k2") >= 0).all()
    assert (kl_estimate(logp, ref_logp, "k3") >= 0).all()
    assert (kl_estimate(logp, ref_logp, "k1") < 0).any()


@pytest.mark.parametrize("estimator", ["k1", "k2", "k3"])
def test_all_estimators_vanish_and_are_stationary_at_identical_policies(estimator):
    logp = torch.randn(32, dtype=torch.float64, requires_grad=True)

    value = kl_estimate(logp, logp.detach().clone(), estimator)
    (grad,) = torch.autograd.grad(value.sum(), logp)

    assert torch.equal(value.detach(), torch.zeros_like(value))
    if estimator != "k1":  # d k1 / d logp = 1: k1 is not minimised pointwise at equality.
        torch.testing.assert_close(grad, torch.zeros_like(grad))


def test_unknown_estimator_raises():
    with pytest.raises(ValueError, match="unknown KL estimator"):
        kl_estimate(torch.zeros(2), torch.zeros(2), "k4")  # type: ignore[arg-type]
