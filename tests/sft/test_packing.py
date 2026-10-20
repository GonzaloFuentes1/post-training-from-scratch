import random

import pytest

from posttrain.sft import first_fit_decreasing, pack_sequences


def _optimal_num_bins(lengths: list[int], capacity: int) -> int:
    """Exact bin-packing optimum by exhaustive search (small inputs only)."""
    items = sorted(lengths, reverse=True)

    def fits(k: int) -> bool:
        loads = [0] * k

        def place(i: int) -> bool:
            if i == len(items):
                return True
            tried: set[int] = set()
            for b in range(k):
                if loads[b] in tried or loads[b] + items[i] > capacity:
                    continue
                tried.add(loads[b])
                loads[b] += items[i]
                if place(i + 1):
                    return True
                loads[b] -= items[i]
            return False

        return place(0)

    k = max(1, -(-sum(items) // capacity))
    while not fits(k):
        k += 1
    return k


@pytest.mark.parametrize("seed", range(20))
def test_ffd_is_a_valid_packing_within_the_known_bound(seed):
    rng = random.Random(seed)
    capacity = rng.randint(8, 20)
    lengths = [rng.randint(1, capacity) for _ in range(rng.randint(1, 9))]

    bins = first_fit_decreasing(lengths, capacity)

    assert sorted(i for b in bins for i in b) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in b) <= capacity for b in bins)
    # Dósa (2007): FFD <= 11/9 OPT + 6/9.
    assert len(bins) <= 11 / 9 * _optimal_num_bins(lengths, capacity) + 6 / 9


def test_ffd_places_longest_first_and_is_deterministic():
    lengths = [3, 7, 2, 7, 5, 1]

    bins = first_fit_decreasing(lengths, capacity=10)

    assert bins == [[1, 0], [3, 2, 5], [4]]  # [7, 3], [7, 2, 1], [5]
    assert first_fit_decreasing(lengths, capacity=10) == bins


@pytest.mark.parametrize("lengths", [[0], [11], [3, -1]])
def test_ffd_rejects_items_that_cannot_fit(lengths):
    with pytest.raises(ValueError, match="outside"):
        first_fit_decreasing(lengths, capacity=10)


def test_pack_sequences_layout():
    sequences = [[10, 11, 12], [20, 21, 22, 23, 24], [30, 31]]
    loss_masks = [[True, True, True], [False, False, True, True, True], [True, False]]

    packed = pack_sequences(sequences, capacity=6, loss_masks=loss_masks, pad_id=0)

    # FFD order: seq 1 (len 5) opens row 0; seq 0 (len 3) opens row 1; seq 2 joins row 1.
    assert packed.input_ids.tolist() == [[20, 21, 22, 23, 24, 0], [10, 11, 12, 30, 31, 0]]
    assert packed.segment_ids.tolist() == [[1, 1, 1, 1, 1, 0], [1, 1, 1, 2, 2, 0]]
    assert packed.position_ids.tolist() == [[0, 1, 2, 3, 4, 0], [0, 1, 2, 0, 1, 0]]
    ignore = -100
    assert packed.labels.tolist() == [
        # 21 is not a target (mask False); 22, 23, 24 are; last token has no target.
        [ignore, 22, 23, 24, ignore, ignore],
        # 12 -> 30 would cross a document boundary; 31 is masked out.
        [11, 12, ignore, ignore, ignore, ignore],
    ]


def test_pack_sequences_preserves_every_token_and_target():
    rng = random.Random(0)
    sequences = [[rng.randint(1, 50) for _ in range(rng.randint(2, 12))] for _ in range(25)]

    packed = pack_sequences(sequences, capacity=16)

    tokens = packed.input_ids[packed.segment_ids > 0]
    assert sorted(tokens.tolist()) == sorted(t for s in sequences for t in s)
    expected_targets = sum(len(s) - 1 for s in sequences)
    assert int((packed.labels != -100).sum()) == expected_targets
    assert (packed.labels[packed.segment_ids == 0] == -100).all()
