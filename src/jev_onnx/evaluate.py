"""Measure the exported ONNX model on held-out Open-Jev rows.

    python -m jev_onnx.evaluate --model artifacts/jev.onnx --split validation

This scores the real artifact through onnxruntime -- no PyTorch, no tokenizer --
so the number it reports is the number a consumer would see. Rows are batched by
concatenating every candidate in the batch into one `[N]` score row, then each
row is softmaxed over its own candidates only, which is the same formulation the
model was trained with.
"""

from __future__ import annotations

import argparse
import collections
import json
import time

import numpy as np

from .data import Example, load_examples
from .decide import _softmax
from .export_onnx import load_vocab_for


def _batches(examples, size):
    for i in range(0, len(examples), size):
        yield examples[i : i + size]


def score_batch(decide, examples: list[Example]) -> list[dict]:
    """One ONNX call per batch; returns per-row label -> probability."""
    head_text: list[str] = []
    label_text: list[str] = []
    for ex in examples:
        # Candidates are laid out row-major: row i owns len(options_i) slots.
        head_text.extend([ex.question] * len(ex.options))
        label_text.extend(ex.options)

    logits = decide.session.run(
        decide.output_names,
        {
            "content": np.array([e.content for e in examples], dtype=object),
            "head_text": np.array(head_text, dtype=object),
            "label_text": np.array(label_text, dtype=object),
        },
    )[0]

    out = []
    cursor = 0
    for row, ex in enumerate(examples):
        n_opts = len(ex.options)
        cols = list(range(cursor, cursor + n_opts))
        cursor += n_opts
        probs = _softmax(logits[row, cols])
        out.append(
            {
                "labels": list(ex.options),
                "probs": probs.tolist(),
                "target": list(ex.target),
            }
        )
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="artifacts/jev.onnx")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--data",
        choices=("hub", "sample", "auto"),
        default="hub",
        help="'hub' streams the full Open-Jev corpus from the Hub; 'sample' uses "
        "the committed ~1k-row slice under vendor/open-jev-sample for an offline "
        "smoke run. Accuracy on the sample is NOT comparable to the Hub numbers",
    )
    ap.add_argument("--providers", default=None)
    ap.add_argument("--out", default=None, help="write per-row JSON here")
    args = ap.parse_args(argv)

    from .decide import JevDecide

    decide = JevDecide(args.model, providers=args.providers.split(",") if args.providers else None)
    print(f"model : {args.model}")
    print(f"inputs: {[i.name for i in decide.session.get_inputs()]}")
    vocab = load_vocab_for(args.model)
    print(f"vocab : {vocab.size:,} tokens (recovered from the graph)")

    examples = load_examples(args.split, max_rows=args.rows, seed=1234, source=args.data)
    print(f"rows  : {len(examples):,} from split={args.split} (source={args.data})")
    print()

    correct = total = 0
    baseline = 0.0
    gold_mass = 0.0
    per_source = collections.defaultdict(lambda: [0, 0])
    records = []
    t0 = time.time()
    for batch in _batches(examples, args.batch_size):
        for ex, scored in zip(batch, score_batch(decide, batch)):
            pred = int(np.argmax(scored["probs"]))
            gold = int(np.argmax(scored["target"]))
            hit = int(pred == gold)
            correct += hit
            total += 1
            # Chance level for this row's option count.
            baseline += 1.0 / len(scored["labels"])
            gold_mass += scored["probs"][gold]
            src = ex.question[:40]
            per_source[src][0] += hit
            per_source[src][1] += 1
            if args.out:
                records.append(
                    {
                        "pred": scored["labels"][pred],
                        "gold": scored["labels"][gold],
                        "probs": dict(zip(scored["labels"], scored["probs"])),
                        "hit": hit,
                    }
                )
    dt = time.time() - t0

    acc = correct / max(total, 1)
    print("=" * 74)
    print(f"  rows scored      : {total:,}  in {dt:.1f}s ({dt / max(total, 1) * 1000:.1f} ms/row)")
    print(f"  top-1 accuracy   : {acc:.4f}")
    print(f"  chance baseline  : {baseline / max(total, 1):.4f}  (1 / #options, averaged)")
    print(f"  prob on gold     : {gold_mass / max(total, 1):.4f}")
    print("=" * 74)

    print("\nper question head (top 12 by row count):")
    rows = sorted(per_source.items(), key=lambda kv: -kv[1][1])[:12]
    for src, (hit, n) in rows:
        print(f"  {hit / n:6.3f}  {n:6,}  {src}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False)
        print(f"\nwrote per-row predictions to {args.out}")

    return {"accuracy": acc, "n": total, "baseline": baseline / max(total, 1)}


if __name__ == "__main__":
    main()