"""Train the dual encoder on Open-Jev and export it to a single ONNX file.

    python -m jev_onnx.train --max-rows 120000 --epochs 2

Deliberately plain: a manual training loop over a ``DataLoader``, AdamW, linear
warmup-then-decay. No trainer framework, so every step of what happens between
Open-Jev rows and ONNX logits is visible in one file.
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

from .data import JevDataset, build_vocabulary, load_examples
from .export_onnx import export_to_onnx
from .model import JevDualEncoder, soft_classification_loss


def evaluate(model, loader, device) -> dict:
    """Top-1 accuracy against the highest-mass target label.

    Open-Jev targets are soft distributions, so this is deliberately the simple
    view: how often the argmax of the prediction matches the argmax of the truth.
    """
    model.eval()
    correct = total = 0
    nll = 0.0
    with torch.no_grad():
        for batch in loader:
            content = batch["content_ids"].to(device)
            labels = batch["label_ids"].to(device)
            pair_row = batch["pair_row"].to(device)
            pair_target = batch["pair_target"].to(device)
            logits = model(content, labels)
            b = logits.shape[0]
            same = pair_row.unsqueeze(0) == torch.arange(b, device=device).unsqueeze(1)
            masked = logits.masked_fill(~same, torch.finfo(logits.dtype).min / 2)
            pred = masked.argmax(dim=-1)

            truth = pair_target.argmax(dim=0)
            correct += int((pred == truth).sum())
            total += b
            # soft_classification_loss already averages over the batch's rows;
            # recover the per-row sum so the running average matches train_loss.
            n_rows = int(same.any(dim=1).sum())
            nll += float(soft_classification_loss(logits, pair_row, pair_target)) * n_rows
    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "loss": nll / max(total, 1),
        "n": total,
    }


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts", help="output directory")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=None, help="subsample for a quick run")
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--max-vocab", type=int, default=32_000)
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

    print("=" * 72)
    print("1/5  loading Open-Jev")
    print("=" * 72)
    train_ex = load_examples("train", max_rows=args.max_rows, seed=args.seed)
    eval_ex = load_examples(
        "validation", max_rows=args.eval_rows, seed=args.seed
    )
    print(f"  train rows : {len(train_ex):,}")
    print(f"  eval rows  : {len(eval_ex):,}")

    print()
    print("=" * 72)
    print("2/5  building vocabulary (same tokenizer the graph will use)")
    print("=" * 72)
    vocab = build_vocabulary(train_ex, min_freq=args.min_freq, max_size=args.max_vocab)
    vocab.save(out_dir / "vocab.json")
    print(f"  vocabulary : {vocab.size:,} tokens (index 0 is reserved for PAD/unknown)")

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

    model = JevDualEncoder(
        vocab.size,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)
    print(f"  parameters : {model.n_parameters():,}")

    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    print()
    print("=" * 72)
    print("3/5  training")
    print("=" * 72)
    print(f"  steps/epoch : {steps_per_epoch:,}  x{args.epochs} epochs = {total_steps:,}")
    print(f"  device      : {device}")
    history = []
    step = 0
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        running = 0.0
        for i, batch in enumerate(train_dl):
            content = batch["content_ids"].to(device)
            labels = batch["label_ids"].to(device)
            pair_row = batch["pair_row"].to(device)
            pair_target = batch["pair_target"].to(device)

            logits = model(content, labels)
            loss = soft_classification_loss(logits, pair_row, pair_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

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
    print("4/5  history")
    print("=" * 72)
    for h in history:
        print("  " + json.dumps(h))

    if args.no_export:
        print()
        print("  (--no-export: stopping before ONNX export)")
        return out_dir

    print()
    print("=" * 72)
    print("5/5  exporting to ONNX (tokenizer baked into the graph)")
    print("=" * 72)
    onnx_path = out_dir / "jev.onnx"
    export_to_onnx(
        model.cpu(),
        vocab,
        onnx_path,
        max_len=args.max_len,
        max_label_len=args.max_label_len,
    )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  wrote {onnx_path}  ({size_mb:.1f} MB)")
    print("  inputs : content [B], head_text [N], label_text [N]  (all strings)")
    print("  output : logits [B, N]")
    print("  the .onnx file is self-contained: vocabulary + tokenizer are in the graph")
    return onnx_path


if __name__ == "__main__":
    main()