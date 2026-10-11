"""Entry point: train a model and score text with it.

    python main.py                      # full pipeline, if artifacts/jev.onnx exists
    python main.py --text "..."         # score a single string
    python main.py --train              # (re)train then export
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = "artifacts/jev.onnx"

EXAMPLE_SCHEMA = {
    "intent": [
        "order_status",
        "refund_request",
        "cancel_subscription",
        "login_problem",
        "billing_question",
        "other",
    ],
    "urgency": ["low", "normal", "high", "critical"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text", help="score this string and exit")
    ap.add_argument("--train", action="store_true", help="train and export first")
    ap.add_argument("--max-rows", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    if args.train:
        from jev_onnx.train import main as train_main

        train_main(
            [
                "--out",
                str(Path(args.model).parent),
                "--max-rows",
                str(args.max_rows),
                "--epochs",
                str(args.epochs),
            ]
        )

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"no model at {model_path}\n"
            f"train one first:  python -m jev_onnx.train --out {model_path.parent}"
        )

    from jev_onnx.decide import JevDecide

    decide = JevDecide(model_path)

    if args.text:
        scored = decide.score(args.text, EXAMPLE_SCHEMA)
        print(f"text: {args.text}\n")
        for head, probs in scored.items():
            print(f"{head}:")
            for label, p in sorted(probs.items(), key=lambda kv: -kv[1]):
                print(f"  {p:6.3f}  {label}")
        print()
        print("decision:", decide.classify_text(args.text, EXAMPLE_SCHEMA))
        return

    text = (
        "Hi, my subscription renewed on 3 May for $54.00 while the service was "
        "already down for two days. Please refund the duplicate charge and tell "
        "me when it is processed. This is urgent."
    )
    print(f"text: {text}\n")
    print("decision:", decide.classify_text(text, EXAMPLE_SCHEMA))
    print()
    print("run `python demo.py --model {model_path}` for the full tour".format(
        model_path=model_path
    ))


if __name__ == "__main__":
    main()