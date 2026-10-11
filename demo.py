"""Score a few real Open-Jev rows through the exported ONNX file.

    python demo.py --model artifacts/jev.onnx

Shows the API surface: a single decision, several decisions in one pass, a
multi-label head, and batch scoring -- all with raw text going into onnxruntime.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from jev_onnx.decide import JevDecide
from jev_onnx.data import render_content

GREEN, DIM, BOLD, RESET = "\033[32m", "\033[2m", "\033[1m", "\033[0m"


def rule(title: str) -> None:
    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")


def show(text: str, limit: int = 300) -> None:
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[:limit] + " ..."
    print(f"{DIM}{textwrap.indent(flat, '  | ', lambda s: s)[3:]}{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/jev.onnx")
    ap.add_argument("--rows", type=int, default=4, help="how many dataset rows to demo")
    ap.add_argument("--providers", default=None)
    args = ap.parse_args()

    path = Path(args.model)
    if not path.exists():
        raise SystemExit(f"model not found: {path}\nRun: python -m jev_onnx.train")

    decide = JevDecide(path)
    print(f"{BOLD}loaded{RESET} {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"{BOLD}inputs{RESET}  {[i.name for i in decide.session.get_inputs()]}")
    print(f"{BOLD}output{RESET}  {[o.name for o in decide.session.get_outputs()]}")

    from datasets import get_dataset_config_names, load_dataset

    rows = []
    for cfg in get_dataset_config_names("ZefanCai/Open-Jev"):
        try:
            d = load_dataset("ZefanCai/Open-Jev", cfg, split="train")
        except ValueError:
            continue
        for r in d:
            if len(r["options"]) >= 2:
                rows.append(
                    {
                        "content": render_content(r["state_json"]),
                        "question": r["question"],
                        "labels": list(r["options"]),
                        "target": list(r["target"]),
                    }
                )
        if len(rows) >= 4000:
            break

    sample = rows[: args.rows]

    # ---------------------------------------------------------------- one head
    rule("1. one decision, one head")
    first = sample[0]
    show(first["content"])
    out = decide.classify_text(first["content"], {"decision": first["labels"]})
    scores = decide.score(first["content"], {"decision": first["labels"]})["decision"]
    print(f"\n  {DIM}chosen:{RESET} {GREEN}{out['decision']}{RESET}")
    for lab, p in sorted(scores.items(), key=lambda kv: -kv[1]):
        mark = "*" if lab == out["decision"] else " "
        print(f"   {mark} {p:6.3f}  {lab[:66]}")
    print(f"  {DIM}gold (argmax):{RESET} {first['labels'][max(range(len(first['target'])), key=lambda i: first['target'][i])][:66]}")

    # ------------------------------------------------------------- many heads
    rule("2. several decisions in ONE forward pass")
    groups = [r for r in rows if 2 <= len(r["labels"]) <= 4][:3]
    if len(groups) >= 2:
        shared = groups[0]["content"]
        schema = {f"head{i}": g["labels"] for i, g in enumerate(groups)}
        out = decide.classify_text(shared, schema)
        show(shared)
        print(f"\n  {DIM}one call, {len(schema)} heads, {sum(len(v) for v in schema.values())} candidates{RESET}")
        for name, labels in schema.items():
            print(f"   {name:8s} -> {GREEN}{out[name]}{RESET}")

    # ------------------------------------------------------------- multi-label
    rule("3. multi-label head (every label above threshold)")
    text = (
        "Battery dies before lunch and the fan is loud, but the keyboard and "
        "the screen are the best I have used on a laptop."
    )
    schema = {
        "aspects": {
            "labels": ["battery", "keyboard", "screen", "fan", "price"],
            "multi_label": True,
            "cls_threshold": 0.15,
        }
    }
    probs = decide.score(text, schema)["aspects"]
    out = decide.classify_text(text, schema)
    show(text)
    for lab, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        mark = "*" if lab in out["aspects"] else " "
        print(f"   {mark} {p:6.3f}  {lab}")
    print(f"\n  {DIM}kept:{RESET} {GREEN}{out['aspects']}{RESET}")

    # -------------------------------------------------------------- batching
    rule("4. batch scoring")
    batch = [r["content"] for r in sample]
    schemas = [{"decision": r["labels"]} for r in sample]
    t0 = __import__("time").time()
    outputs = []
    for text, sch in zip(batch, schemas):
        outputs.append(decide.classify_text(text, sch))
    dt = __import__("time").time() - t0
    print(f"\n  {DIM}{len(batch)} texts scored in {dt * 1000:.0f} ms "
          f"({dt / len(batch) * 1000:.0f} ms/text){RESET}")
    for r, o in zip(sample, outputs):
        gold = r["labels"][max(range(len(r["target"])), key=lambda i: r["target"][i])]
        hit = "OK " if o["decision"] == gold else "-- "
        print(f"   {hit}pred={o['decision'][:26]:28s} gold={gold[:26]}")

    print(f"\n{GREEN}done{RESET}\n")


if __name__ == "__main__":
    main()