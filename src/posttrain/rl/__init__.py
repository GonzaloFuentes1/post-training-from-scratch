"""Group-relative policy optimization objectives (GRPO, DAPO, Dr.GRPO)."""

from posttrain.rl.advantages import group_relative_advantages
from posttrain.rl.kl import KLEstimator, kl_estimate
from posttrain.rl.losses import (
    Normalization,
    PolicyLossConfig,
    PolicyLossOutput,
    aggregate_token_losses,
    clipped_surrogate,
    policy_gradient_loss,
)

__all__ = [
    "KLEstimator",
    "Normalization",
    "PolicyLossConfig",
    "PolicyLossOutput",
    "aggregate_token_losses",
    "clipped_surrogate",
    "group_relative_advantages",
    "kl_estimate",
    "policy_gradient_loss",
]
