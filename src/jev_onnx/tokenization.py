"""Tokenisation that is bit-for-bit identical to what the ONNX graph does.

The exported model receives raw text and tokenises it *inside* the graph with

    StringNormalizer(case_change_action=LOWER, is_case_sensitive=0)
      -> StringSplit(delimiter="")

To train a PyTorch model that matches, this module reimplements exactly those
two steps in Python.  Any divergence here becomes a silent train/serve skew, so
``tests/test_tokenizer_parity.py`` checks this implementation against a real
onnxruntime graph built from those two operators.

Semantics verified against onnxruntime 1.31 (see tests):

* Lowercasing is ASCII/simple case folding.
* ``delimiter=""`` means "split on runs of the space character": consecutive
  spaces collapse and leading/trailing spaces are stripped.
* Only ``' '`` is a delimiter.  ``\\t``, ``\\n`` and ``\\r`` are NOT, so a word
  either side of a newline stays glued into one token.  This is a real
  limitation of the operator; see ``README.md`` for the workaround.
* An empty/whitespace-only input produces zero tokens.
"""

from __future__ import annotations

import re
from collections import Counter

# Reserved index. Never handed out to a vocabulary entry, so it doubles as
# "unknown word" and "padding". The graph can therefore derive its attention
# mask with a single `ids != 0`.
PAD_ID = 0

# How a decision candidate is assembled on the label side of the dual encoder.
# Training builds this string in Python (``data.label_text``); the exported
# graph builds it with StringConcat. They must agree exactly or the model is
# served text it was never trained on.
LABEL_SEPARATOR = " | "

_SPLIT_ON_SPACES = re.compile(r" +")

# The 13 codepoints in the BMP where ORT's StringNormalizer(LOWER) disagrees
# with Python's str.lower(). Found by probing every BMP codepoint through a real
# onnxruntime StringNormalizer (scripts/probe_case_folding.py): ORT uses Unicode
# *simple* lowercasing, which skips titlecase letters and has no multi-character
# expansions. ASCII, Latin-1, Cyrillic, CJK and punctuation agree exactly, so
# this table is the complete set of divergences and nothing else needs handling.
_ORT_LOWER_OVERRIDES = {
    0x0130: "i",  # LATIN CAPITAL LETTER I WITH DOT ABOVE -> i
    0x038D: "ύ",  # GREEK ... IOTA WITH VARIA AND TONOS (archaic, unassigned)
    0x038E: "ώ",  # GREEK CAPITAL LETTER UPSILON WITH VARIA AND TONOS
    0x038F: "Ϗ",  # GREEK CAPITAL LETTER IOTA WITH VARIA AND YPOGEGRAMMENI
    0x01C5: "ǅ",  # LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON (titlecase)
    0x01C8: "ǈ",  # LATIN CAPITAL LETTER L WITH SMALL LETTER J WITH CARON (titlecase)
    0x01CB: "ǋ",  # LATIN CAPITAL LETTER N WITH SMALL LETTER J WITH CARON (titlecase)
    0x01F2: "ǲ",  # LATIN CAPITAL LETTER D WITH SMALL LETTER Z WITH CARON (titlecase)
    0x1C89: "Ᲊ",  # CYRILLIC CAPITAL LETTER DZE (archaic)
    0xA7CB: "Ɤ",  # LATIN CAPITAL LETTER REVERSED R WITH FISHHOOK (archaic)
    0xA7CC: "Ꟍ",  # LATIN CAPITAL LETTER R WITH FISHHOOK (archaic)
    0xA7DA: "Ꟛ",  # LATIN CAPITAL LETTER VOLAPUK AE (archaic)
    0xA7DC: "Ƛ",  # LATIN CAPITAL LETTER VOLAPUK OE (archaic)
}


def _ort_lower_char(ch: str) -> str:
    cp = ord(ch)
    override = _ORT_LOWER_OVERRIDES.get(cp)
    if override is not None:
        return override
    low = ch.lower()
    # Simple lowercasing never expands; Python's full mapping can (e.g. U+0130).
    return low if len(low) == 1 else low[0]


def _ort_lower(text: str) -> str:
    """Mirror of StringNormalizer(case_change_action=LOWER, is_case_sensitive=0).

    Folding one character at a time is what makes the final-sigma rule (a
    *contextual* special case) never fire, matching ORT, which is context-free.
    """
    if text.isascii():
        return text.lower()
    return "".join(_ort_lower_char(ch) for ch in text)


def ort_split(text: str) -> list[str]:
    """Tokenise exactly like ``StringNormalizer(LOWER) -> StringSplit(delimiter="")``."""
    lowered = _ort_lower(text)
    # Leading/trailing spaces are removed and runs of spaces collapse to one.
    lowered = lowered.strip(" ")
    if not lowered:
        return []
    return _SPLIT_ON_SPACES.split(lowered)


class Vocab:
    """Word-level vocabulary backed by ``ai.onnx.ml.CategoryMapper`` in the graph."""

    def __init__(self, itos: list[str]):
        if not itos or itos[0] != "":
            raise ValueError("vocabulary index 0 is reserved for PAD/unknown")
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def size(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    @classmethod
    def build(
        cls,
        texts,
        min_freq: int = 5,
        max_size: int = 32_000,
    ) -> "Vocab":
        """Count tokens over ``texts`` and keep the ``max_size`` most frequent.

        Tokens below ``min_freq`` are deliberately dropped rather than mapped to
        ``<unk>``: at inference the graph sends anything it does not recognise to
        index 0, and a token so rare it carries no signal is cheaper to drop than
        to learn an embedding for.
        """
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(ort_split(t))
        kept = [tok for tok, n in counts.most_common(max_size - 1) if n >= min_freq]
        return cls([""] + kept)

    def encode(self, text: str, max_len: int) -> list[int]:
        """Return ``max_len`` ids, right-aligned-free (truncate on the right).

        Out-of-vocabulary words and padding both become ``PAD_ID`` (0), which the
        graph treats as "ignore" when it builds the attention mask.
        """
        stoi = self.stoi
        ids = [stoi.get(tok, PAD_ID) for tok in ort_split(text)]
        return ids[:max_len]

    def encode_batch(self, texts, max_len: int) -> list[list[int]]:
        return [self.encode(t, max_len) for t in texts]

    def save(self, path) -> None:
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.itos, fh, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "Vocab":
        import json

        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))