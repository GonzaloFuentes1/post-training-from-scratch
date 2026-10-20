"""Offline preference optimization objectives (DPO, cDPO, IPO, SimPO)."""

from posttrain.preference.logprobs import sequence_logprobs
from posttrain.preference.losses import (
    dpo_loss,
    dpo_margin,
    implicit_rewards,
    ipo_loss,
    simpo_loss,
)

__all__ = [
    "dpo_loss",
    "dpo_margin",
    "implicit_rewards",
    "ipo_loss",
    "sequence_logprobs",
    "simpo_loss",
]
