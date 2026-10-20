"""A minimal chat template with assistant-only loss masks.

In chat SFT the model should learn to produce assistant turns, not to imitate
users or to predict the template's role headers. The standard approach is to
render the whole conversation, then train only on the tokens of assistant
messages plus the end-of-turn marker that follows each of them. Including the
end marker matters: it is how the model learns to stop.

The template is ChatML-like::

    <|im_start|>{role}\\n{content}<|im_end|>\\n

and is tokenized with a byte-level tokenizer plus a few special tokens, which is
enough to exercise every masking rule without depending on a real tokenizer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


class ByteTokenizer:
    """UTF-8 bytes as ids ``0..255`` plus special tokens.

    Special tokens get ids outside the byte range so that no user text can
    ever encode to them, which is the property that makes template injection
    impossible.
    """

    IM_START = 256
    IM_END = 257
    PAD = 258
    vocab_size = 259

    _special: ClassVar[dict[int, str]] = {
        IM_START: "<|im_start|>",
        IM_END: "<|im_end|>",
        PAD: "<|pad|>",
    }

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        out: list[str] = []
        buffer = bytearray()
        for i in ids:
            if i in self._special:
                out.append(buffer.decode("utf-8"))
                buffer.clear()
                out.append(self._special[i])
            else:
                buffer.append(i)
        out.append(buffer.decode("utf-8"))
        return "".join(out)


def render_chat(
    messages: Sequence[Message],
    tokenizer: ByteTokenizer | None = None,
    *,
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[bool]]:
    """Token ids of a conversation and a mask of the tokens to train on.

    The mask is true on the content and the closing ``<|im_end|>`` of every
    assistant message, and false on system and user messages, on role
    headers and on the newline between turns.

    With ``add_generation_prompt`` an open ``<|im_start|>assistant`` header
    is appended (unmasked) so that a model can continue from it.
    """
    tok = tokenizer or ByteTokenizer()
    ids: list[int] = []
    mask: list[bool] = []

    def emit(new_ids: list[int], trainable: bool) -> None:
        ids.extend(new_ids)
        mask.extend([trainable] * len(new_ids))

    for message in messages:
        is_assistant = message.role == "assistant"
        emit([tok.IM_START, *tok.encode(f"{message.role}\n")], False)
        emit([*tok.encode(message.content), tok.IM_END], is_assistant)
        emit(tok.encode("\n"), False)
    if add_generation_prompt:
        emit([tok.IM_START, *tok.encode("assistant\n")], False)
    return ids, mask
