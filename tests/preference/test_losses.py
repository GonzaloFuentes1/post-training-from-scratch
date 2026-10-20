import math

import pytest
import torch

from posttrain.preference import (
    dpo_loss,
    dpo_margin,
    implicit_rewards,
    ipo_loss,
    sequence_logprobs,
    simpo_loss,
)

BETA = 0.5


def _logps(seed: int, n: int = 8, requires_grad: bool = False):
    gen = torch.Generator().manual_seed(seed)
    tensors = [-20 * torch.rand(n, generator=gen, dtype=torch.float64) for _ in range(4)]
    for t in tensors[:2]:
        t.requires_grad_(requires_grad)
    return tensors  # policy_chosen, policy_rejected, ref_chosen, ref_rejected


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def test_dpo_gradient_matches_analytic_form():
    """dL/dm = -beta * sigmoid(-beta * m), with m the chosen-minus-rejected log-ratio."""
    pc, pr, rc, rr = _logps(seed=0, requires_grad=True)
    rc.requires_grad_(True)

    losses = dpo_loss(pc, pr, rc, rr, beta=BETA)
    grad_pc, grad_pr, grad_rc = torch.autograd.grad(losses.sum(), (pc, pr, rc))

    margin = dpo_margin(pc, pr, rc, rr).detach()
    expected = -BETA * _sigmoid(-BETA * margin)
    # Sign check: the gradient on the chosen log-prob is negative, so gradient
    # descent increases pi(y_w); the rejected and reference terms enter with
    # the opposite sign.
    assert (expected < 0).all()
    torch.testing.assert_close(grad_pc, expected)
    torch.testing.assert_close(grad_pr, -expected)
    torch.testing.assert_close(grad_rc, -expected)


def test_dpo_is_bradley_terry_on_implicit_rewards():
    pc, pr, rc, rr = _logps(seed=1)

    reward_gap = implicit_rewards(pc, rc, BETA) - implicit_rewards(pr, rr, BETA)
    expected = -torch.log(_sigmoid(reward_gap))

    torch.testing.assert_close(dpo_loss(pc, pr, rc, rr, beta=BETA), expected)


def test_dpo_reference_values_and_symmetry():
    zero = torch.zeros(1, dtype=torch.float64)
    assert dpo_loss(zero, zero, zero, zero).item() == pytest.approx(math.log(2.0))

    pc, pr, rc, rr = _logps(seed=2)
    swapped = dpo_loss(pr, pc, rr, rc, beta=BETA)
    torch.testing.assert_close(
        dpo_loss(pc, pr, rc, rr, beta=BETA) - swapped,
        -BETA * dpo_margin(pc, pr, rc, rr),
    )


def test_dpo_is_finite_for_extreme_margins():
    big = torch.tensor([1e4], dtype=torch.float64)
    zero = torch.zeros(1, dtype=torch.float64)
    assert torch.isfinite(dpo_loss(zero, big, zero, zero, beta=1.0)).all()
    assert dpo_loss(big, zero, zero, zero, beta=1.0).item() == pytest.approx(0.0, abs=1e-12)


def test_conservative_dpo_reduces_to_dpo_and_has_finite_optimum():
    pc, pr, rc, rr = _logps(seed=3)
    torch.testing.assert_close(
        dpo_loss(pc, pr, rc, rr, beta=BETA, label_smoothing=0.0),
        dpo_loss(pc, pr, rc, rr, beta=BETA),
    )

    eps = 0.1
    optimum = math.log((1 - eps) / eps) / BETA
    margin = torch.tensor([optimum - 1.0, optimum, optimum + 1.0], dtype=torch.float64)
    margin.requires_grad_(True)
    zeros = torch.zeros(3, dtype=torch.float64)
    loss = dpo_loss(margin, zeros, zeros, zeros, beta=BETA, label_smoothing=eps)
    (grad,) = torch.autograd.grad(loss.sum(), margin)

    m = margin.detach()
    expected = -BETA * ((1 - eps) * _sigmoid(-BETA * m) - eps * _sigmoid(BETA * m))
    torch.testing.assert_close(grad, expected)
    assert grad[0] < 0
    assert grad[1].item() == pytest.approx(0.0, abs=1e-12)
    assert grad[2] > 0  # beyond m*, the loss pulls the margin back


def test_ipo_regresses_margin_to_target():
    tau = 0.25
    target = 1 / (2 * tau)
    margin = torch.tensor([0.0, target, 2 * target], dtype=torch.float64, requires_grad=True)
    zeros = torch.zeros(3, dtype=torch.float64)

    loss = ipo_loss(margin, zeros, zeros, zeros, tau=tau)
    (grad,) = torch.autograd.grad(loss.sum(), margin)

    torch.testing.assert_close(loss.detach(), (margin.detach() - target).square())
    torch.testing.assert_close(grad, 2 * (margin.detach() - target))
    assert loss[1].item() == 0.0
    # Overshooting the target is penalised; DPO would keep rewarding it.
    assert loss[2] > loss[1]
    dpo = dpo_loss(margin.detach(), zeros, zeros, zeros, beta=1.0)
    assert dpo[2] < dpo[1]


def test_simpo_gradient_and_margin():
    avg_c = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float64, requires_grad=True)
    avg_r = torch.tensor([-1.0, -1.0, -3.0], dtype=torch.float64)
    beta, gamma = 2.0, 1.0

    loss = simpo_loss(avg_c, avg_r, beta=beta, gamma=gamma)
    (grad,) = torch.autograd.grad(loss.sum(), avg_c)

    z = beta * (avg_c.detach() - avg_r) - gamma
    torch.testing.assert_close(grad, -beta * _sigmoid(-z))
    # Equal likelihoods still incur softplus(gamma): the margin demands separation.
    assert loss[0].item() == pytest.approx(math.log1p(math.exp(gamma)))


def test_simpo_is_dpo_on_average_logps_with_an_uninformative_reference():
    """SimPO needs no reference model: it equals DPO (shifted by gamma) for any
    reference that scores the chosen and rejected responses equally."""
    gen = torch.Generator().manual_seed(4)
    avg_c = -torch.rand(6, generator=gen, dtype=torch.float64)
    avg_r = -torch.rand(6, generator=gen, dtype=torch.float64)

    simpo = simpo_loss(avg_c, avg_r, beta=BETA, gamma=0.0)
    for ref_value in (-3.0, 0.0, 7.0):
        ref = torch.full_like(avg_c, ref_value)
        torch.testing.assert_close(simpo, dpo_loss(avg_c, avg_r, ref, ref, beta=BETA))


def test_simpo_is_length_invariant_where_summed_logps_are_not():
    """Repeating a response's per-token predictions leaves SimPO unchanged."""
    gen = torch.Generator().manual_seed(5)
    vocab = 7
    chosen_logits = torch.randn(1, 3, vocab, generator=gen, dtype=torch.float64)
    rejected_logits = torch.randn(1, 3, vocab, generator=gen, dtype=torch.float64)
    chosen_labels = torch.randint(0, vocab, (1, 3), generator=gen)
    rejected_labels = torch.randint(0, vocab, (1, 3), generator=gen)

    def logps(logits, labels, repeats, average):
        mask = torch.ones(1, 3 * repeats, dtype=torch.bool)
        return sequence_logprobs(
            logits.repeat(1, repeats, 1), labels.repeat(1, repeats), mask, average=average
        )

    def simpo(repeats, average=True):
        return simpo_loss(
            logps(chosen_logits, chosen_labels, repeats, average),
            logps(rejected_logits, rejected_labels, 1, average),
        )

    torch.testing.assert_close(simpo(repeats=1), simpo(repeats=3))
    assert not torch.allclose(simpo(repeats=1, average=False), simpo(repeats=3, average=False))


@pytest.mark.parametrize(
    "loss_fn",
    [
        lambda pc, pr, rc, rr: dpo_loss(pc, pr, rc, rr, beta=BETA),
        lambda pc, pr, rc, rr: dpo_loss(pc, pr, rc, rr, beta=BETA, label_smoothing=0.2),
        lambda pc, pr, rc, rr: ipo_loss(pc, pr, rc, rr, tau=0.3),
        lambda pc, pr, rc, rr: simpo_loss(pc / 10, pr / 10, beta=2.0, gamma=0.5),
    ],
    ids=["dpo", "cdpo", "ipo", "simpo"],
)
def test_gradcheck(loss_fn):
    pc, pr, rc, rr = _logps(seed=6, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: loss_fn(a, b, rc, rr), (pc, pr))


def test_invalid_hyperparameters_raise():
    x = torch.zeros(1)
    with pytest.raises(ValueError, match="label_smoothing"):
        dpo_loss(x, x, x, x, label_smoothing=0.5)
    with pytest.raises(ValueError, match="tau"):
        ipo_loss(x, x, x, x, tau=0.0)
