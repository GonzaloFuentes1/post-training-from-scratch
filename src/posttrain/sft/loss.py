r"""Token-weighted SFT loss that is exact under gradient accumulation.

The SFT objective is the mean cross-entropy over *all* target tokens in the
global batch. When the batch is split into micro-batches the natural code,

.. code-block:: python

    for mb in micro_batches:
        (cross_entropy(mb, reduction="mean") / num_micro_batches).backward()

computes a mean of per-micro-batch means. That weights every micro-batch
equally regardless of how many target tokens it holds, so tokens in sparse
micro-batches (short answers, mostly-masked chats) are over-weighted and the
gradient depends on how the batch happened to be split.

The fix is to sum the per-token losses in each micro-batch and divide by the
number of target tokens in the *whole* batch, counted before the first
backward pass:

.. math::

    \mathcal{L} = \frac{1}{N_{\mathrm{global}}} \sum_{b} \sum_{t \in b} \ell_t .

In data-parallel training ``N_global`` is the token count summed over all
ranks (all-reduced before the first backward pass). DDP then *averages*
gradients across the ``world_size`` ranks, so dividing each rank's summed
loss by ``N_global`` alone shrinks the gradient by a factor ``world_size``.
Either multiply the loss by ``world_size`` or, equivalently, have each rank
pass ``num_target_tokens = N_global / world_size``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn.functional as F
from torch import Tensor


def count_target_tokens(labels: Iterable[Tensor], ignore_index: int = -100) -> int:
    """Number of positions that contribute to the loss across all micro-batches."""
    return sum(int((lab != ignore_index).sum()) for lab in labels)


def causal_lm_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    num_target_tokens: float | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Summed token cross-entropy divided by ``num_target_tokens``.

    Args:
        logits: ``[B, T, V]`` logits aligned with ``labels`` (already shifted).
        labels: ``[B, T]`` targets, ``ignore_index`` where there is no loss.
        num_target_tokens: Normalizer. Pass the global count from
            :func:`count_target_tokens` when accumulating gradients; when
            ``None`` the local count is used, which is only correct for a
            single micro-batch. Under DDP pass ``N_global / world_size``
            (see the module docstring), which need not be an integer.
    """
    loss_sum = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=ignore_index, reduction="sum"
    )
    if num_target_tokens is None:
        num_target_tokens = count_target_tokens([labels], ignore_index)
    return loss_sum / max(num_target_tokens, 1)
