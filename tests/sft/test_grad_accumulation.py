"""Gradient accumulation with token weighting reproduces the full batch.

Micro-batches deliberately hold very different numbers of target tokens: that
is the regime in which averaging per-micro-batch means goes wrong.
"""

import torch

from posttrain.models import TinyGPT, TinyGPTConfig
from posttrain.sft import causal_lm_loss, count_target_tokens

VOCAB = 17
SEQ_LEN = 12


def _model() -> TinyGPT:
    torch.manual_seed(0)
    cfg = TinyGPTConfig(vocab_size=VOCAB, d_model=16, n_layers=2, n_heads=2, max_positions=16)
    return TinyGPT(cfg).double()


def _batch():
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB, (6, SEQ_LEN), generator=gen)
    labels = torch.roll(ids, -1, dims=1)
    # Number of target tokens per row: 11, 10, 1, 2, 9, 3 -> micro-batches of 21, 3, 12.
    for row, n in enumerate([11, 10, 1, 2, 9, 3]):
        labels[row, n:] = -100
    return ids, labels


MICRO_BATCHES = [slice(0, 2), slice(2, 4), slice(4, 6)]


def _grads(model: TinyGPT) -> list[torch.Tensor]:
    return [p.grad.clone() for p in model.parameters()]


def _full_batch_grads(model, ids, labels):
    model.zero_grad()
    causal_lm_loss(model(ids), labels).backward()
    return _grads(model)


def _token_weighted_accumulation(model, ids, labels):
    model.zero_grad()
    n_global = count_target_tokens(labels[s] for s in MICRO_BATCHES)
    for s in MICRO_BATCHES:
        causal_lm_loss(model(ids[s]), labels[s], num_target_tokens=n_global).backward()
    return _grads(model)


def _naive_accumulation(model, ids, labels):
    model.zero_grad()
    for s in MICRO_BATCHES:
        (causal_lm_loss(model(ids[s]), labels[s]) / len(MICRO_BATCHES)).backward()
    return _grads(model)


def test_token_weighted_accumulation_equals_full_batch():
    model = _model()
    ids, labels = _batch()

    full = _full_batch_grads(model, ids, labels)
    accumulated = _token_weighted_accumulation(model, ids, labels)

    for g, ref in zip(accumulated, full, strict=True):
        torch.testing.assert_close(g, ref, atol=1e-12, rtol=1e-10)


def test_naive_mean_of_micro_batch_means_does_not():
    model = _model()
    ids, labels = _batch()

    full = _full_batch_grads(model, ids, labels)
    naive = _naive_accumulation(model, ids, labels)

    rel_error = torch.cat([(g - f).flatten() for g, f in zip(naive, full, strict=True)]).norm()
    rel_error = rel_error / torch.cat([f.flatten() for f in full]).norm()
    assert rel_error > 0.1


def test_accumulation_is_deterministic():
    ids, labels = _batch()
    first = _token_weighted_accumulation(_model(), ids, labels)
    second = _token_weighted_accumulation(_model(), ids, labels)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)
