import math

import pytest
import torch

from posttrain.preference import sequence_logprobs


def _batch(seed: int):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(3, 6, 5, generator=gen, dtype=torch.float64)
    labels = torch.randint(0, 5, (3, 6), generator=gen)
    mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 0],  # prompt, response, pad
            [0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    labels = torch.where(mask, labels, torch.full_like(labels, -100))
    return logits, labels, mask


def _reference(logits, labels, mask, average):
    out = []
    for b in range(logits.shape[0]):
        terms = []
        for t in range(logits.shape[1]):
            if mask[b, t]:
                row = logits[b, t].tolist()
                log_norm = math.log(sum(math.exp(x) for x in row))
                terms.append(row[int(labels[b, t])] - log_norm)
        out.append(sum(terms) / len(terms) if average else sum(terms))
    return out


@pytest.mark.parametrize("average", [False, True])
def test_matches_loop_reference(average):
    logits, labels, mask = _batch(seed=0)

    got = sequence_logprobs(logits, labels, mask, average=average)

    assert got.tolist() == pytest.approx(_reference(logits, labels, mask, average), abs=1e-12)


def test_prompt_and_padding_logits_do_not_matter():
    logits, labels, mask = _batch(seed=1)
    noisy = torch.where(mask.unsqueeze(-1), logits, 1e3 * torch.randn_like(logits))

    torch.testing.assert_close(
        sequence_logprobs(noisy, labels, mask), sequence_logprobs(logits, labels, mask)
    )


@pytest.mark.parametrize("junk", [float("-inf"), float("nan")])
@pytest.mark.parametrize("average", [False, True])
def test_non_finite_prompt_and_padding_logits_give_finite_gradients(junk, average):
    logits, labels, mask = _batch(seed=3)
    clean = logits.clone().requires_grad_(True)
    sequence_logprobs(clean, labels, mask, average=average).sum().backward()

    dirty = torch.where(mask.unsqueeze(-1), logits, torch.full_like(logits, junk))
    dirty.requires_grad_(True)
    value = sequence_logprobs(dirty, labels, mask, average=average)
    value.sum().backward()

    torch.testing.assert_close(
        value.detach(), sequence_logprobs(logits, labels, mask, average=average)
    )
    assert dirty.grad is not None
    assert torch.isfinite(dirty.grad).all()
    assert torch.equal(dirty.grad, clean.grad)


def test_gradcheck():
    logits, labels, mask = _batch(seed=2)
    logits.requires_grad_(True)

    assert torch.autograd.gradcheck(
        lambda x: sequence_logprobs(x, labels, mask, average=True), (logits,)
    )
