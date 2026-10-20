import torch

from posttrain.sft import block_causal_mask, position_ids_from_segments

SEGMENTS = torch.tensor(
    [
        [1, 1, 1, 2, 2, 3, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 3, 4, 0, 0, 0, 0],
    ]
)


def test_block_causal_mask_matches_definition():
    mask = block_causal_mask(SEGMENTS)

    for n, row in enumerate(SEGMENTS.tolist()):
        for i, seg_i in enumerate(row):
            for j, seg_j in enumerate(row):
                # Padding attends only to itself; documents attend causally within.
                expected = i == j if seg_i == 0 else (seg_i == seg_j and j <= i)
                assert bool(mask[n, i, j]) == expected, (n, i, j)


def test_every_query_has_at_least_one_key():
    assert block_causal_mask(SEGMENTS).any(dim=-1).all()


def test_position_ids_restart_per_document():
    expected = [
        [0, 1, 2, 0, 1, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert position_ids_from_segments(SEGMENTS).tolist() == expected
