"""Train the dual encoder with a *pretrained* first layer, then export it.

    python -m jev_onnx.train_potion --max-rows 120000 --epochs 3

Same model, same graph, same tokenizer as ``python -m jev_onnx.train``. The one
difference is where the embedding table starts:

    baseline   tok_emb.weight ~ N(0, 1)          learned from scratch on Open-Jev
    potion     tok_emb.weight = potion-base-8M    pretrained, then fine-tuned

The pretrained vectors come from ``minishlab/potion-base-8M``, a Model2Vec
distillation of ``baai/bge-base-en-v1.5``: one 256-d vector per word, no
transformer, no contextual passes. Lifting it into ``tok_emb`` is the whole
pretraining story -- the four encoder layers above it are still trained from
random, exactly as in the baseline.

**The tokenizer does not change.** The exported graph still runs
``StringNormalizer(LOWER) -> StringSplit -> CategoryMapper``; the potion word
list is only what goes *into* ``CategoryMapper``. Nothing from ``transformers``
or ``tokenizers`` is imported here or at serving time -- see ``potion.py``.

Output goes to ``artifacts_potion/jev_potion.onnx``. The baseline recipe writes
to ``artifacts/jev.onnx`` and is untouched; the two models are meant to sit side
by side and be compared (``python compare.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import JevDataset, load_examples, token_coverage
from .export_onnx import export_to_onnx
from .model import JevDualEncoder, soft_classification_loss
from .potion import POTION_REPO, build_potion_vocab, seed_token_embedding
from .train import evaluate

# Word pairs used to show what the first layer "knows" before and after training.
# The last two are controls: they should stay low if the geometry is real.
PROBE_PAIRS = [
    ("refund", "refunded"),
    ("invoice", "bill"),
    ("login", "log"),
    ("urgent", "immediately"),
    ("apple", "banana"),
    ("cat", "dog"),
]


def probe_first_layer(model, vocab) -> list[tuple[str, str, float | None]]:
    """Cosine similarity between related words in ``tok_emb.weight``."""
    import torch.nn.functional as F

    stoi = vocab.stoi
    out = []
    with torch.no_grad():
        weight = F.normalize(model.tok_emb.weight, dim=-1)
        for a, b in PROBE_PAIRS:
            ia, ib = stoi.get(a), stoi.get(b)
            cos = None
            if ia is not None and ib is not None:
                cos = float(weight[ia] @ weight[ib])
            out.append((a, b, cos))
    return out


def relative_drift(model, seeded: "np.ndarray") -> float:
    """Mean per-row ``||w - w_seed|| / ||w_seed||`` since the table was seeded.

    Row 0 is the reserved PAD/unknown slot: it is all zeros and never receives a
    gradient, so it would divide by zero.
    """
    with torch.no_grad():
        now = model.tok_emb.weight.detach()[1:].cpu().numpy()
    was = np.asarray(seeded, dtype=np.float32)[1:]
    denom = np.maximum(np.linalg.norm(was, axis=1, keepdims=True), 1e-8)
    return float((np.linalg.norm(now - was, axis=1, keepdims=True) / denom).mean())


def print_probe(model, vocab, when: str, seeded=None) -> None:
    print(f"  first layer, {when}:")
    for a, b, cos in probe_first_layer(model, vocab):
        mark = "oov" if cos is None else f"{cos:5.3f}"
        print(f"    {a:>12s} ~ {b:<12s} {mark}")
    if seeded is not None:
        print(f"    drift from the pretrained table: {relative_drift(model, seeded):.1%}")


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts_potion", help="output directory")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--emb-lr-mult",
        type=float,
        default=None,
        help="LR multiplier for the embedding table (1.0 == baseline). "
        "Default: 0.1 with a pretrained table, 1.0 with --init random",
    )
    ap.add_argument(
        "--freeze-embed",
        action="store_true",
        help="never update the pretrained table (pure Model2Vec-style first layer)",
    )
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=None, help="subsample for a quick run")
    ap.add_argument(
        "--min-freq",
        type=int,
        default=0,
        help="drop potion words rarer than this in the training corpus; "
        "0 ships every reachable word (baseline: 5, and its vocab is corpus-derived)",
    )
    ap.add_argument(
        "--max-vocab",
        type=int,
        default=None,
        help="cap the vocabulary (baseline: 32000). Default: keep every reachable word",
    )
    ap.add_argument(
        "--init-scale",
        type=float,
        default=1.0,
        help="scale on the pretrained rows; 1.0 == the norm nn.Embedding draws at init",
    )
    ap.add_argument("--repo", default=POTION_REPO)
    ap.add_argument(
        "--init",
        choices=("pretrained", "random"),
        default="pretrained",
        help="'random' keeps the potion *vocabulary* but seeds the table with "
        "N(0,1) -- the control that separates 'pretrained weights' from "
        "'a different word list'",
    )
    ap.add_argument(
        "--name",
        default="jev_potion",
        help="output file stem inside --out (default: jev_potion.onnx)",
    )
    ap.add_argument("--max-len", type=int, default=448)
    ap.add_argument("--max-label-len", type=int, default=96)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-rows", type=int, default=4000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The reduced LR exists only to protect vectors that already know something.
    # With `--init random` there is nothing to protect, and the control run
    # should train its table exactly the way the baseline trains its own.
    emb_lr_mult = args.emb_lr_mult
    if emb_lr_mult is None:
        emb_lr_mult = 1.0 if args.init == "random" else 0.1

    print("=" * 72)
    print("1/6  loading Open-Jev")
    print("=" * 72)
    train_ex = load_examples("train", max_rows=args.max_rows, seed=args.seed)
    eval_ex = load_examples("validation", max_rows=args.eval_rows, seed=args.seed)
    print(f"  train rows : {len(train_ex):,}")
    print(f"  eval rows  : {len(eval_ex):,}")

    print()
    print("=" * 72)
    print("2/6  pretrained first layer")
    print("=" * 72)
    potion = build_potion_vocab(
        train_ex,
        d_model=args.d_model,
        repo=args.repo,
        min_freq=args.min_freq,
        max_size=args.max_vocab,
        init_scale=args.init_scale,
    )
    vocab = potion.vocab
    vocab.save(out_dir / "vocab.json")
    print(f"  source     : {args.repo}  (static vectors, no tokenizer imported)")
    print(f"  vocabulary : {vocab.size:,} tokens (index 0 is reserved for PAD/unknown)")
    mass, types = token_coverage(train_ex, vocab)
    print(f"  coverage   : {mass:.1%} of token occurrences, {types:.1%} of types")
    print(f"  table      : {potion.table.shape[0]:,} x {potion.dim} float32")

    train_ds = JevDataset(train_ex, vocab, args.max_len, args.max_label_len)
    eval_ds = JevDataset(eval_ex, vocab, args.max_len, args.max_label_len)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_ds.collate,
        num_workers=0,
        drop_last=True,
    )
    eval_dl = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=eval_ds.collate
    )

    print()
    print("=" * 72)
    print("3/6  model")
    print("=" * 72)
    model = JevDualEncoder(
        vocab.size,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)
    if args.init == "pretrained":
        seed_token_embedding(model, potion.table)
        origin = f"seeded from {args.repo}"
    else:
        origin = "N(0,1) -- control run, potion vocabulary but no pretraining"
    print(f"  parameters : {model.n_parameters():,}")
    print(f"  tok_emb    : {origin}" + (", FROZEN" if args.freeze_embed else ", fine-tuned"))
    print_probe(model, vocab, "before fine-tuning")

    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))

    # Two parameter groups: the pretrained table moves at a fraction of the
    # baseline LR, because it already knows what a word means and a large step
    # would erase that faster than Open-Jev could replace it.
    other_params = [p for n, p in model.named_parameters() if n != "tok_emb.weight"]
    groups = [{"params": other_params, "lr": args.lr}]
    if not args.freeze_embed:
        groups.append({"params": [model.tok_emb.weight], "lr": args.lr * emb_lr_mult})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    print()
    print("=" * 72)
    print("4/6  fine-tuning")
    print("=" * 72)
    print(f"  steps/epoch : {steps_per_epoch:,}  x{args.epochs} epochs = {total_steps:,}")
    print(f"  device      : {device}")
    print(f"  lr          : {args.lr:g} (everything but tok_emb)"
          + ("" if args.freeze_embed else f", {args.lr * emb_lr_mult:g} (tok_emb)"))
    history = []
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        running = 0.0
        for i, batch in enumerate(train_dl):
            logits = model(
                batch["content_ids"].to(device),
                batch["label_ids"].to(device),
            )
            loss = soft_classification_loss(
                logits,
                batch["pair_row"].to(device),
                batch["pair_target"].to(device),
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], 1.0
            )
            opt.step()
            sched.step()

            running += float(loss.detach())
            if (i + 1) % 50 == 0 or i + 1 == steps_per_epoch:
                done = i + 1
                rate = done / max(time.time() - t0, 1e-6)
                eta = (steps_per_epoch - done) / max(rate, 1e-6)
                print(
                    f"  epoch {epoch + 1}/{args.epochs}  step {done:>6,}/{steps_per_epoch:,}"
                    f"  loss {running / done:.4f}  {rate:5.1f} step/s  eta {eta / 60:4.1f} min",
                    flush=True,
                )
        metrics = evaluate(model, eval_dl, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running / max(steps_per_epoch, 1),
                **{f"val_{k}": v for k, v in metrics.items()},
            }
        )
        print(
            f"  -> epoch {epoch + 1} train_loss {history[-1]['train_loss']:.4f}"
            f"  val_loss {metrics['loss']:.4f}  val_acc {metrics['accuracy']:.4f}"
            f"  ({metrics['n']} rows, {time.time() - t0:.0f}s)",
            flush=True,
        )

    print()
    print("=" * 72)
    print("5/6  history")
    print("=" * 72)
    for h in history:
        print("  " + json.dumps(h))
    print_probe(
        model, vocab, "after fine-tuning",
        seeded=potion.table if args.init == "pretrained" else None,
    )

    if args.no_export:
        print()
        print("  (--no-export: stopping before ONNX export)")
        return out_dir

    print()
    print("=" * 72)
    print(f"6/6  exporting to ONNX (same graph, first layer: {args.init})")
    print("=" * 72)
    onnx_path = out_dir / f"{args.name}.onnx"
    export_to_onnx(
        model.cpu(),
        vocab,
        onnx_path,
        max_len=args.max_len,
        max_label_len=args.max_label_len,
    )
    print(f"  wrote {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print("  inputs : content [B], head_text [N], label_text [N]  (all strings)")
    print("  output : logits [B, N]")
    print("  the .onnx file is self-contained: vocabulary + tokenizer are in the graph")
    print("  tokenizer is the same StringNormalizer -> StringSplit as the baseline")
    print()
    print("  compare against the baseline:  python compare.py")
    return onnx_path


if __name__ == "__main__":
    main()
