import pytest
import torch

from posttrain.rl import group_relative_advantages


def _rewards(num_groups: int, group_size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(num_groups * group_size, generator=gen, dtype=torch.float64)


@pytest.mark.parametrize("scale_by_std", [True, False])
@pytest.mark.parametrize("group_size", [2, 4, 16])
def test_advantages_sum_to_zero_within_each_group(group_size, scale_by_std):
    rewards = _rewards(num_groups=5, group_size=group_size, seed=group_size)

    adv = group_relative_advantages(rewards, group_size, scale_by_std=scale_by_std)

    sums = adv.view(-1, group_size).sum(dim=1)
    torch.testing.assert_close(sums, torch.zeros_like(sums), atol=1e-12, rtol=0)


def test_std_scaled_advantages_have_unit_std():
    rewards = _rewards(num_groups=3, group_size=8, seed=0)

    adv = group_relative_advantages(rewards, 8, eps=0.0)

    std = adv.view(3, 8).std(dim=1)
    torch.testing.assert_close(std, torch.ones_like(std), atol=1e-12, rtol=0)


def test_matches_explicit_formula_for_binary_rewards():
    # One prompt, four samples, one correct answer: mean 0.25, unbiased std 0.5.
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    grpo = group_relative_advantages(rewards, 4, eps=0.0)
    dr_grpo = group_relative_advantages(rewards, 4, scale_by_std=False)

    torch.testing.assert_close(grpo, torch.tensor([1.5, -0.5, -0.5, -0.5], dtype=torch.float64))
    torch.testing.assert_close(
        dr_grpo, torch.tensor([0.75, -0.25, -0.25, -0.25], dtype=torch.float64)
    )


def test_constant_group_gives_zero_advantage_not_nan():
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float64)

    adv = group_relative_advantages(rewards, 3)

    assert torch.isfinite(adv).all()
    assert torch.equal(adv[:3], torch.zeros(3, dtype=torch.float64))


def test_invariant_to_per_group_reward_offset():
    rewards = _rewards(num_groups=4, group_size=6, seed=3)
    offsets = torch.tensor([10.0, -3.0, 0.5, 100.0], dtype=torch.float64).repeat_interleave(6)

    for scale in (True, False):
        base = group_relative_advantages(rewards, 6, scale_by_std=scale)
        shifted = group_relative_advantages(rewards + offsets, 6, scale_by_std=scale)
        torch.testing.assert_close(base, shifted, atol=1e-10, rtol=0)


def test_std_scaling_removes_reward_scale_but_dr_grpo_keeps_it():
    rewards = _rewards(num_groups=2, group_size=5, seed=4)

    scaled = group_relative_advantages(rewards * 7.0, 5, eps=0.0)
    torch.testing.assert_close(scaled, group_relative_advantages(rewards, 5, eps=0.0))

    unscaled = group_relative_advantages(rewards * 7.0, 5, scale_by_std=False)
    torch.testing.assert_close(
        unscaled, 7.0 * group_relative_advantages(rewards, 5, scale_by_std=False)
    )


def test_rejects_ragged_groups():
    with pytest.raises(ValueError, match="groups of 4"):
        group_relative_advantages(torch.zeros(6), 4)
