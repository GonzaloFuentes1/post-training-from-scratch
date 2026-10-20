import math

import pytest
import torch

from posttrain.ops import masked_mean, masked_sum, token_logprobs


def _reference_logprob(row: list[float], target: int) -> float:
    """Log-softmax written out by hand, independent of torch's kernels."""
    peak = max(row)
    log_norm = peak + math.log(sum(math.exp(x - peak) for x in row))
    return row[target] - log_norm


def test_token_logprobs_matches_scalar_reference():
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 5, 7, generator=gen, dtype=torch.float64) * 3
    targets = torch.randint(0, 7, (3, 5), generator=gen)

    got = token_logprobs(logits, targets)

    for b in range(3):
        for t in range(5):
            expected = _reference_logprob(logits[b, t].tolist(), int(targets[b, t]))
            assert got[b, t].item() == pytest.approx(expected, abs=1e-12)


def test_token_logprobs_masked_positions_are_zero_and_detached():
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 4, 6, generator=gen, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[1, 2, -100, -100], [3, -100, 0, 5]])
    mask = targets != -100

    logp = token_logprobs(logits, targets, mask)
    logp.sum().backward()

    assert torch.all(logp[~mask] == 0)
    assert logits.grad is not None
    assert torch.all(logits.grad[~mask] == 0)
    assert torch.all(logits.grad[mask].abs().sum(-1) > 0)


@pytest.mark.parametrize("junk", [float("-inf"), float("inf"), float("nan")])
def test_token_logprobs_non_finite_masked_logits_give_finite_gradients(junk):
    """A masked row of -inf, inf or NaN logits must not leak NaN into backward.

    Without selecting before ``logsumexp`` the forward value is still correct
    (the masked output is overwritten with 0), but the backward of
    ``logsumexp`` multiplies the zero upstream gradient by a non-finite local
    derivative and ``logits.grad`` becomes NaN on that row.
    """
    gen = torch.Generator().manual_seed(3)
    clean = torch.randn(2, 4, 6, generator=gen, dtype=torch.float64)
    targets = torch.tensor([[1, 2, -100, -100], [3, -100, 0, 5]])
    mask = targets != -100

    clean = clean.requires_grad_(True)
    expected = token_logprobs(clean, targets, mask)
    expected.sum().backward()

    dirty = torch.where(mask.unsqueeze(-1), clean.detach(), torch.full_like(clean, junk))
    dirty.requires_grad_(True)
    got = token_logprobs(dirty, targets, mask)
    got.sum().backward()

    assert torch.equal(got.detach(), expected.detach())
    assert dirty.grad is not None
    assert torch.isfinite(dirty.grad).all()
    assert torch.equal(dirty.grad, clean.grad)


def test_masked_sum_does_not_protect_the_op_that_produced_the_padding():
    """Documents the limit of masking: ``where`` guards ``values``, not its producer.

    ``log(0) = -inf`` at a masked slot gives the right value, but the backward
    of ``log`` evaluates ``0 * (1 / 0) = NaN``. Replacing the input before the
    op (the "double where") fixes it.
    """
    x = torch.tensor([0.5, 0.0], requires_grad=True)
    mask = torch.tensor([True, False])

    masked_sum(torch.log(x), mask).backward()
    assert x.grad is not None
    assert torch.isnan(x.grad[1])

    x.grad = None
    safe_x = torch.where(mask, x, torch.ones_like(x))
    masked_sum(torch.log(safe_x), mask).backward()
    assert torch.equal(x.grad, torch.tensor([2.0, 0.0]))


def test_token_logprobs_rejects_negative_targets_without_mask():
    with pytest.raises(ValueError, match="explicit mask"):
        token_logprobs(torch.zeros(1, 2, 3), torch.tensor([[0, -100]]))


def test_masked_mean_matches_python_mean_per_row():
    gen = torch.Generator().manual_seed(2)
    values = torch.randn(4, 6, generator=gen, dtype=torch.float64)
    mask = torch.rand(4, 6, generator=gen) > 0.4
    mask[:, 0] = True

    got = masked_mean(values, mask, dim=-1)

    for row in range(4):
        selected = [v for v, m in zip(values[row].tolist(), mask[row].tolist(), strict=True) if m]
        assert got[row].item() == pytest.approx(sum(selected) / len(selected), abs=1e-12)


def test_masked_mean_of_empty_row_is_zero_not_nan():
    values = torch.randn(2, 3, dtype=torch.float64)
    mask = torch.tensor([[True, False, True], [False, False, False]])

    got = masked_mean(values, mask, dim=-1)

    assert torch.isfinite(got).all()
    assert got[1].item() == 0.0


def test_masked_sum_ignores_non_finite_padding_in_value_and_gradient():
    values = torch.tensor([1.0, 2.0, float("-inf"), float("nan")], requires_grad=True)
    mask = torch.tensor([True, True, False, False])

    total = masked_sum(values, mask)
    total.backward()

    assert total.item() == 3.0
    assert values.grad is not None
    assert torch.equal(values.grad, torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="differ"):
        masked_sum(torch.zeros(2, 3), torch.ones(3, 2, dtype=torch.bool))
