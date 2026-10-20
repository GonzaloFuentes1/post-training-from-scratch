"""Supervised fine-tuning: packing, attention masks, chat loss masks, loss normalization."""

from posttrain.sft.attention import block_causal_mask, position_ids_from_segments
from posttrain.sft.chat import ByteTokenizer, Message, render_chat
from posttrain.sft.loss import causal_lm_loss, count_target_tokens
from posttrain.sft.packing import PackedBatch, first_fit_decreasing, pack_sequences

__all__ = [
    "ByteTokenizer",
    "Message",
    "PackedBatch",
    "block_causal_mask",
    "causal_lm_loss",
    "count_target_tokens",
    "first_fit_decreasing",
    "pack_sequences",
    "position_ids_from_segments",
    "render_chat",
]
