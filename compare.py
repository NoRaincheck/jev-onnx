"""Score the two models on the same rows and put them side by side.

    python compare.py                       # artifacts/jev.onnx vs artifacts_potion/jev_potion.onnx
    python compare.py --split test --rows 2000

Both files are the same graph with the same three string inputs. The only
difference is the first layer: one learned from scratch on Open-Jev, one
initialised from `minishlab/potion-base-8M` and then fine-tuned. This script
pulls the embedding table straight out of each `.onnx` file -- it is the graph's
`tok_emb.weight` initializer -- so you can see the difference in the shipped
artefacts rather than taking it on trust.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from jev_onnx.data import load_examples
from jev_onnx.decide import JevDecide
from jev_onnx.evaluate import score_batch
from jev_onnx.export_onnx import load_vocab_for
from jev_onnx.tokenization import ort_split

BOLD, DIM, GREEN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[0m"

# Related pairs first, then deliberate controls: the last two should stay low in
# any embedding space that means something.
PROBE_PAIRS = [
    ("urgent", "immediately"),
    ("angry", "furious"),
    ("bill", "billing"),
    ("password", "login"),
    ("shipment", "delivery"),
    ("cat", "dog"),
    ("apple", "banana"),
]


def rule(title: str) -> None:
    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")


def embedding_table(path: Path) -> np.ndarray:
    """Pull ``tok_emb.weight`` out of the exported graph."""
    graph = onnx.load(str(path)).graph
    for init in graph.initializer:
        if init.name == "tok_emb.weight":
            return numpy_helper.to_array(init)
    raise ValueError(f"{path} has no tok_emb.weight initializer")


def cosine_probe(table: np.ndarray, vocab) -> list[tuple[str, str, float | None]]:
    unit = table / np.maximum(np.linalg.norm(table, axis=1, keepdims=True), 1e-8)
    rows = []
    for a, b in PROBE_PAIRS:
        ia, ib = vocab.stoi.get(a), vocab.stoi.get(b)
        rows.append(
            (a, b, float(unit[ia] @ unit[ib]) if ia is not None and ib is not None else None)
        )
    return rows


def label_words(examples) -> set[str]:
    """Distinct words that appear on the *label* side of the held-out rows.

    This is the zero-shot axis: the label set is supplied by the caller at call
    time, so how much of it a vocabulary can spell is a property of the model
    that no amount of training on Open-Jev fixes.
    """
    words: set[str] = set()
    for ex in examples:
        for option in ex.options:
            words.update(ort_split(option))
        words.update(ort_split(ex.question))
    return words


def score_model(path: Path, examples, batch_size: int) -> dict:
    decide = JevDecide(path)
    hits, total, baseline, gold_mass = 0, 0, 0.0, 0.0
    preds = []
    t0 = time.time()
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        for ex, scored in zip(batch, score_batch(decide, batch)):
            pred = int(np.argmax(scored["probs"]))
            gold = int(np.argmax(scored["target"]))
            hits += int(pred == gold)
            total += 1
            baseline += 1.0 / len(scored["labels"])
            gold_mass += scored["probs"][gold]
            preds.append(scored["labels"][pred])
    dt = time.time() - t0
    return {
        "path": path,
        "accuracy": hits / max(total, 1),
        "baseline": baseline / max(total, 1),
        "gold_mass": gold_mass / max(total, 1),
        "ms_per_row": dt / max(total, 1) * 1000,
        "preds": preds,
        "vocab": load_vocab_for(path),
        "table": embedding_table(path),
        "size_mb": path.stat().st_size / 1e6,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["artifacts/jev.onnx", "artifacts_potion/jev_potion.onnx"],
        help="one or more .onnx files to score on the same rows",
    )
    ap.add_argument("--split", default="validation")
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--data",
        choices=("hub", "sample", "auto"),
        default="hub",
        help="'hub' streams the full Open-Jev corpus; 'sample' uses the committed "
        "~1k-row slice for an offline smoke run",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in args.models]
    present = [p for p in paths if p.exists()]
    if not present:
        raise SystemExit(
            "no models found.\n"
            "  train them first:\n"
            "    python -m jev_onnx.train       --max-rows 120000 --epochs 3\n"
            "    python -m jev_onnx.train_potion --max-rows 120000 --epochs 3"
        )
    for p in paths:
        if not p.exists():
            print(f"{DIM}skipping {p} (not built yet){RESET}")

    rule("loading held-out rows (identical for every model)")
    examples = load_examples(args.split, max_rows=args.rows, seed=1234, source=args.data)
    print(f"  split {args.split}, {len(examples):,} rows (source={args.data})")

    results = [score_model(p, examples, args.batch_size) for p in present]

    # ------------------------------------------------------------------ table
    rule("the same graph, a different first layer")
    head = f"  {'':<22}" + "".join(f"{r['path'].stem:>22}" for r in results)
    print(head)
    print(f"  {'-' * (len(head) - 2)}")

    def row(label, fn):
        cells = [fn(r) for r in results]
        print(f"  {label:<22}" + "".join(f"{c:>22}" for c in cells))

    row("file", lambda r: f"{r['size_mb']:.1f} MB")
    row("vocabulary", lambda r: f"{len(r['vocab']):,} entries")  # incl. the PAD slot
    row("embedding table", lambda r: f"{r['table'].shape[0]:,} x {r['table'].shape[1]}")
    print()
    row("top-1 accuracy", lambda r: f"{r['accuracy']:.4f}")
    row("chance baseline", lambda r: f"{r['baseline']:.4f}")
    row("x chance", lambda r: f"{r['accuracy'] / max(r['baseline'], 1e-9):.2f}x")
    row("prob on gold", lambda r: f"{r['gold_mass']:.4f}")
    print()
    row("ms / row", lambda r: f"{r['ms_per_row']:.1f}")

    if len(results) > 1:
        print()
        first = results[0]
        for other in results[1:]:
            agree = sum(int(x == y) for x, y in zip(first["preds"], other["preds"]))
            print(f"  agrees with {first['path'].stem} on "
                  f"{agree / len(first['preds']):.1%} of rows".ljust(24)
                  + f"{'':<20}{other['path'].stem}")

    # ------------------------------------------------------------- the vocab
    rule("what each vocabulary can spell")
    words = label_words(examples)
    print(f"  distinct words on the label side of these rows: {len(words):,}")
    for r in results:
        hit = sum(1 for w in words if w in r["vocab"])
        print(f"  {r['path'].stem:<26} {hit / len(words):6.1%}")

    # ------------------------------------------------------- the first layer
    rule("first layer of the shipped .onnx file (cosine, before any projection)")
    probes = [cosine_probe(r["table"], r["vocab"]) for r in results]
    print(f"  {'':<26}" + "".join(f"{r['path'].stem:>22}" for r in results))
    for i, (a, b) in enumerate(PROBE_PAIRS):
        cells = []
        for r, p in zip(results, probes):
            cos = p[i][2]
            cells.append("oov" if cos is None else f"{cos:6.3f}")
        print(f"  {a + ' ~ ' + b:<26}" + "".join(f"{c:>22}" for c in cells))

    print()
    print(f"{GREEN}both files take the same three string inputs and return logits "
          f"[B, N]{RESET}")
    print(f"{DIM}so either one can be dropped into jev_onnx.decide.JevDecide unchanged{RESET}\n")


if __name__ == "__main__":
    main()
