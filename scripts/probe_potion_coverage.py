"""What the potion word list actually buys, once the graph has tokenised for us.

    python scripts/probe_potion_coverage.py --rows 20000

Establishes four numbers the design of ``jev_onnx.potion`` rests on:

1. how much of ``vocab.txt`` a *whitespace* tokenizer can ever hit,
2. how much token mass each vocabulary represents on Open-Jev,
3. what the uncovered mass is -- almost all of it is glued-together JSON, not
   English, and it is the part the baseline model memorises and this one drops,
4. that the vectors arrive semantically organised (the point of pretraining).
"""

from __future__ import annotations

import argparse
import collections

import numpy as np

from jev_onnx.data import build_vocabulary, load_examples, token_coverage
from jev_onnx.potion import (
    SPECIAL_TOKENS,
    build_potion_vocab,
    is_reachable,
    load_static_embeddings,
)
from jev_onnx.tokenization import ort_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--baseline-vocab", default="artifacts/vocab.json")
    args = ap.parse_args()

    print("=" * 74)
    print("1. how much of vocab.txt can StringSplit ever emit?")
    print("=" * 74)
    tokens, matrix = load_static_embeddings()
    dead_space = [t for t in tokens if " " in t]
    dead_frag = [t for t in tokens if t.startswith("##")]
    dead_special = [t for t in tokens if t in SPECIAL_TOKENS]
    reachable = [t for t in tokens if is_reachable(t)]
    print(f"  entries in vocab.txt        : {len(tokens):,}")
    print(f"  dropped, contains a space   : {len(dead_space):,}")
    print(f"  dropped, '##' wordpiece frag: {len(dead_frag):,}")
    print(f"  dropped, [CLS]-style special: {len(dead_special):,}  {dead_special}")
    print(f"  reachable by StringSplit    : {len(reachable):,}")

    print()
    print("=" * 74)
    print("2. token mass each vocabulary represents on Open-Jev")
    print("=" * 74)
    ex = load_examples("train", max_rows=args.rows, seed=0)
    print(f"  rows: {len(ex):,}")

    potion_full = build_potion_vocab(ex)
    baseline = build_vocabulary(ex, min_freq=5, max_size=32_000)

    for name, vocab in (
        ("baseline  (corpus-derived, 32k)", baseline),
        ("potion     (pretrained, full)", potion_full.vocab),
        ("potion     (min_freq=5)", build_potion_vocab(ex, min_freq=5).vocab),
        ("potion     (min_freq=5, 32k cap)", build_potion_vocab(ex, min_freq=5, max_size=32_000).vocab),
    ):
        mass, types = token_coverage(ex, vocab)
        print(f"  {name:<32s} {len(vocab):>7,} tokens   mass {mass:6.2%}   types {types:6.2%}")

    print()
    print("=" * 74)
    print("3. what the uncovered mass looks like")
    print("=" * 74)
    counts: collections.Counter[str] = collections.Counter()
    from jev_onnx.data import corpus_texts

    for text in corpus_texts(ex):
        counts.update(ort_split(text))
    total = sum(counts.values())
    covered = sum(n for t, n in counts.items() if t in potion_full.vocab)
    missed = sorted(
        ((t, n) for t, n in counts.items() if t not in potion_full.vocab),
        key=lambda kv: -kv[1],
    )
    glued = [(t, n) for t, n in missed if len(t) > 24]
    print(f"  token mass covered by potion : {covered / total:.2%}")
    print(f"  uncovered mass               : {(total - covered) / total:.2%}")
    print(f"  uncovered types longer than 24 chars: "
          f"{sum(n for _, n in glued) / max(total - covered, 1):.1%} of the uncovered mass")
    lens = [len(t) for t, _ in missed]
    lens.sort()
    print(f"  median uncovered token length: {lens[len(lens) // 2]} chars")
    print("  longest uncovered tokens (JSON structure with no spaces in it):")
    for t, n in missed[:5]:
        print(f"    {n:>7,}  {t[:96]!r}")

    print()
    print("=" * 74)
    print("4. are the vectors already semantically organised?")
    print("=" * 74)
    table = potion_full.table
    norms = np.linalg.norm(table, axis=1)
    print(f"  stored row norms  : min {np.linalg.norm(matrix, axis=1).min():.1f} "
          f"max {np.linalg.norm(matrix, axis=1).max():.1f}  (rescaled to "
          f"{norms[1:].mean():.1f})")
    unit = table / np.maximum(norms, 1e-8)[:, None]
    stoi = potion_full.vocab.stoi
    pairs = [
        ("angry", "furious"), ("urgent", "immediately"), ("bill", "billing"),
        ("cancel", "refund"), ("apple", "banana"), ("cat", "dog"), ("the", "of"),
    ]
    for a, b in pairs:
        ia, ib = stoi.get(a), stoi.get(b)
        if ia is None or ib is None:
            print(f"    {a:>12s} ~ {b:<12s}   oov ({a if ia is None else b})")
        else:
            print(f"    {a:>12s} ~ {b:<12s} {float(unit[ia] @ unit[ib]):6.3f}")


if __name__ == "__main__":
    main()
