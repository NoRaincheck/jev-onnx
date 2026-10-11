"""Corpus helpers: what gets tokenised, and how much of it a vocabulary covers.

Both numbers are printed by ``train_potion`` and ``compare.py``, so a bug here
would quietly misreport the comparison rather than break anything.
"""

from __future__ import annotations

from jev_onnx.data import (
    Example,
    build_vocabulary,
    corpus_texts,
    label_text,
    token_coverage,
)
from jev_onnx.tokenization import Vocab


def _examples() -> list[Example]:
    return [
        Example("Refund the INVOICE", "What is wanted?", ["refund_request", "other"], [1.0, 0.0]),
        Example("the cat sat", "What is wanted?", ["refund_request", "other"], [0.0, 1.0]),
    ]


def test_corpus_texts_covers_content_and_every_candidate():
    ex = _examples()
    texts = corpus_texts(ex)
    # One content string plus one "<option> | <question>" per candidate, in the
    # same order the graph will see them.
    assert texts == [
        ex[0].content,
        label_text(ex[0].question, "refund_request"),
        label_text(ex[0].question, "other"),
        ex[1].content,
        label_text(ex[1].question, "refund_request"),
        label_text(ex[1].question, "other"),
    ]


def test_build_vocabulary_is_unchanged_by_the_refactor():
    """`build_vocabulary` must still produce exactly the words it always did."""
    vocab = build_vocabulary(_examples(), min_freq=1, max_size=1000)
    assert vocab.itos[0] == ""
    assert "refund" in vocab and "invoice" in vocab


def test_token_coverage_reports_mass_and_types():
    """Both numbers, worked out by hand from the two example rows.

    corpus_texts yields 3 + 5 x 4 + 3 = 26 tokens and 11 distinct types. Note
    ``wanted?`` is one type, not two: ``StringSplit`` breaks on spaces only, so
    trailing punctuation stays welded to the word -- which is exactly the kind of
    token a pretrained plain-word vocabulary cannot spell.
    """
    ex = _examples()
    vocab = Vocab(["", "refund", "invoice"])
    mass, types = token_coverage(ex, vocab)
    assert mass == 2 / 26
    assert types == 2 / 11


def test_token_coverage_counts_a_full_vocabulary_as_complete():
    vocab = Vocab.build(corpus_texts(_examples()), min_freq=1)
    mass, types = token_coverage(_examples(), vocab)
    assert mass == 1.0
    assert types == 1.0


def test_token_coverage_of_an_empty_corpus_is_zero_not_a_crash():
    assert token_coverage([], Vocab(["", "a"])) == (0.0, 0.0)
