"""Tokenizer — the 'dumb bookkeeping on both ends' from Phase 0, Bite 4.

We deliberately DON'T reimplement byte-pair encoding here. Tokenization is a
lookup, not intelligence — so we borrow GPT-2's official tokenizer and wrap it
in a tiny, explicit interface. All the *intelligence* we build ourselves (the
model). The tokenizer is just: text -> ints (encode) and ints -> text (decode).
"""

from __future__ import annotations

from transformers import GPT2Tokenizer


class Tokenizer:
    """Thin wrapper over GPT-2's tokenizer, exposing only encode/decode."""

    def __init__(self, model_name: str = "gpt2") -> None:
        self._tok = GPT2Tokenizer.from_pretrained(model_name)

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def encode(self, text: str) -> list[int]:
        """Text -> list of integer token IDs."""
        return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        """List of integer token IDs -> text."""
        return self._tok.decode(ids)

    def tokens(self, text: str) -> list[tuple[int, str]]:
        """Debug helper: show each token ID alongside the text chunk it maps to.

        This is exactly the (id, piece) pairing we'll visualize in the
        Transformer Explorer module later.
        """
        ids = self.encode(text)
        return [(i, self._tok.decode([i])) for i in ids]
