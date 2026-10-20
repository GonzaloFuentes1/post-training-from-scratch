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
