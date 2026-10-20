"""Packed training must be indistinguishable from unpacked training.

A tiny transformer in float64 is run (a) once per document and (b) on FFD-packed
rows with a block-diagonal causal mask and per-document position ids. The
summed loss and every parameter gradient must agree to rounding error. Two
negative controls show that dropping either ingredient breaks the equality.
"""

import random

import pytest
import torch

from posttrain.models import TinyGPT, TinyGPTConfig
from posttrain.sft import block_causal_mask, causal_lm_loss, pack_sequences

VOCAB = 23
CAPACITY = 16


def _model() -> TinyGPT:
    torch.manual_seed(0)
    cfg = TinyGPTConfig(vocab_size=VOCAB, d_model=16, n_layers=2, n_heads=2, max_positions=32)
    return TinyGPT(cfg).double()


def _documents():
    rng = random.Random(0)
    docs, masks = [], []
    for _ in range(9):
        n = rng.randint(2, 11)
        docs.append([rng.randrange(1, VOCAB) for _ in range(n)])
        masks.append([rng.random() < 0.7 for _ in range(n)])
    return docs, masks


def _unpacked_loss_sum(model: TinyGPT, docs, masks) -> torch.Tensor:
    """Independent reference: each document on its own, labels built by hand."""
    total = torch.zeros((), dtype=torch.float64)
    for doc, mask in zip(docs, masks, strict=True):
        ids = torch.tensor([doc])
        labels = torch.tensor([[doc[j + 1] if mask[j + 1] else -100 for j in range(len(doc) - 1)]])
        logits = model(ids)[:, :-1]
        total = total + causal_lm_loss(logits, labels, num_target_tokens=1)
    return total


def _packed_loss_sum(model: TinyGPT, docs, masks, *, block_mask=True, reset_positions=True):
    packed = pack_sequences(docs, CAPACITY, loss_masks=masks)
    attention_mask = block_causal_mask(packed.segment_ids) if block_mask else None
    positions = packed.position_ids if reset_positions else None
    logits = model(packed.input_ids, position_ids=positions, attention_mask=attention_mask)
    return causal_lm_loss(logits, packed.labels, num_target_tokens=1)


def _loss_and_grads(fn, model):
    model.zero_grad()
    loss = fn(model)
    loss.backward()
    return loss.detach(), [p.grad.clone() for p in model.parameters()]


def test_packed_loss_and_gradients_equal_unpacked_sum():
    model = _model()
    docs, masks = _documents()
    assert len(pack_sequences(docs, CAPACITY).input_ids) < len(docs)  # packing happened

    ref_loss, ref_grads = _loss_and_grads(lambda m: _unpacked_loss_sum(m, docs, masks), model)
    loss, grads = _loss_and_grads(lambda m: _packed_loss_sum(m, docs, masks), model)

    torch.testing.assert_close(loss, ref_loss, atol=1e-10, rtol=1e-10)
    for g, ref in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(g, ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    ("block_mask", "reset_positions"),
    [(False, True), (True, False)],
    ids=["plain-causal-mask", "no-position-reset"],
)
def test_packing_without_isolation_contaminates(block_mask, reset_positions):
    model = _model()
    docs, masks = _documents()

    with torch.no_grad():
        ref = _unpacked_loss_sum(model, docs, masks)
        loss = _packed_loss_sum(
            model, docs, masks, block_mask=block_mask, reset_positions=reset_positions
        )

    assert abs(loss.item() - ref.item()) > 1e-3
