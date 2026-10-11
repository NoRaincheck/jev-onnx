"""Inference over the exported ONNX file.

Mirrors the ergonomics of ``fastino/GLiNER2.5-Decide``'s ``classify_text`` --
pass any label set at call time, score several decision heads in a single pass,
get back a plain dict -- but the heavy lifting is a string-in, logits-out ONNX
graph.  No tokenizer, no vocabulary file and no PyTorch at inference time: the
``.onnx`` file is the whole model.

    from jev_onnx.decide import JevDecide

    decide = JevDecide("artifacts/jev.onnx")
    decide.classify_text(
        "My subscription renewed after the service was down. Can I get a refund?",
        {
            "intent": ["order_status", "refund_request", "login_problem"],
            "urgency": ["low", "high"],
        },
    )
    # {'intent': 'order_status', 'urgency': 'low'}

The graph itself, with no SDK, takes ``content`` / ``head_text`` / ``label_text``
string tensors and returns raw ``logits [B, N]``. See the README for that
contract and for how it differs from this wrapper.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

DEFAULT_MULTI_LABEL_THRESHOLD = 0.4


@dataclass
class Head:
    """One decision being asked of the text."""

    name: str
    labels: list[str]
    prompt: str = ""
    multi_label: bool = False
    threshold: float = DEFAULT_MULTI_LABEL_THRESHOLD
    label_descriptions: dict[str, str] = field(default_factory=dict)

    def texts(self) -> list[str]:
        """The string scored for each candidate label.

        A description, when given, is appended so it participates in the match --
        this is what lets a private taxonomy stay precise without retraining.
        """
        out = []
        for lab in self.labels:
            if lab in self.label_descriptions:
                out.append(f"{lab}: {self.label_descriptions[lab]}")
            else:
                out.append(lab)
        return out

    def question(self) -> str:
        """Prompt first so it survives the label-side token budget."""
        if self.prompt:
            return f"{self.prompt} {self.head_text()}"
        return self.head_text()

    def head_text(self) -> str:
        """Head identity.

        Empty unless a prompt was supplied. The candidate label already carries
        the meaning, and emitting an unseen head name would add nothing but
        out-of-vocabulary tokens to the label-side sequence.
        """
        return ""


def _normalise_heads(schema: dict) -> list[Head]:
    """Accept the shapes ``classify_text`` is documented to take.

    Returns heads whose ``head_text`` is empty unless the caller supplied a
    ``prompt``: the candidate label already carries the meaning, and leaving the
    prompt free keeps unseen prompts from adding untrained tokens.
    """
    heads: list[Head] = []
    for name, spec in schema.items():
        if isinstance(spec, dict):
            labels = spec.get("labels")
            if isinstance(labels, dict):
                descriptions = dict(labels)
                labels = list(labels)
            else:
                descriptions = dict(spec.get("label_descriptions") or {})
            heads.append(
                Head(
                    name=name,
                    labels=list(labels),
                    prompt=spec.get("prompt", "") or "",
                    multi_label=bool(spec.get("multi_label", False)),
                    threshold=float(
                        spec.get("cls_threshold", DEFAULT_MULTI_LABEL_THRESHOLD)
                    ),
                    label_descriptions=descriptions,
                )
            )
        else:
            heads.append(Head(name=name, labels=list(spec)))
    return heads


class JevDecide:
    """Runs the exported graph. Thread-safe for reads after construction."""

    def __init__(
        self,
        model_path: str | Path,
        providers: Sequence[str] | None = None,
        intra_op_num_threads: int | None = None,
    ):
        self.model_path = Path(model_path)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if intra_op_num_threads:
            opts.intra_op_num_threads = intra_op_num_threads
        self.session = ort.InferenceSession(
            str(self.model_path),
            opts,
            providers=list(providers) if providers else None,
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.output_names = [o.name for o in self.session.get_outputs()]

    # ------------------------------------------------------------------ core
    def score(
        self,
        texts: str | Sequence[str],
        schema: dict,
    ) -> dict | list[dict]:
        """Raw per-head probabilities for one or more texts.

        For a single string returns ``{head: {label: prob}}``; for a batch
        returns a list of those. Probabilities are softmaxed *within* each head,
        so heads with different label counts stay comparable.
        """
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)

        heads = _normalise_heads(schema) if isinstance(schema, dict) else []
        if not batch:
            return {} if single else []
        if not heads:
            return [{} for _ in batch] if not single else {}

        # Flatten the schema into one candidate list. The graph scores every
        # candidate against every text, so the head structure is carried by the
        # caller's bookkeeping rather than by an index tensor.
        flat_labels: list[str] = []
        head_texts: list[str] = []
        for head in heads:
            flat_labels.extend(head.texts())
            head_texts.extend([head.question()] * len(head.labels))

        logits = self.session.run(
            self.output_names,
            {
                "content": np.array(batch, dtype=object),
                "head_text": np.array(head_texts, dtype=object),
                "label_text": np.array(flat_labels, dtype=object),
            },
        )[0]

        results = []
        for row in logits:
            per_head: dict[str, dict[str, float]] = {}
            offset = 0
            for head in heads:
                n = len(head.labels)
                chunk = row[offset : offset + n]
                offset += n
                per_head[head.name] = dict(zip(head.labels, _softmax(chunk)))
            results.append(per_head)
        return results[0] if single else results

    # ---------------------------------------------------------------- public
    def classify_text(
        self,
        texts: str | Sequence[str],
        schema: dict,
        threshold: float | None = None,
    ) -> dict | list[dict]:
        """Score text against a label set and return the decided labels.

        Single-label heads return the winning label; multi-label heads return
        every label above the threshold. Shape matches
        ``fastino/GLiNER2.5-Decide``'s ``classify_text``.
        """
        single = isinstance(texts, str)
        scored = self.score(texts, schema)
        heads = _normalise_heads(schema)
        out = []
        # `score` returns one dict per input text; a single string yields one.
        for per_head in (scored if not single else [scored]):
            decision: dict[str, str | list[str]] = {}
            for head in heads:
                probs = per_head.get(head.name, {})
                if head.multi_label:
                    cut = head.threshold if threshold is None else threshold
                    chosen = [lab for lab in head.labels if probs.get(lab, 0.0) >= cut]
                    decision[head.name] = chosen
                else:
                    best = max(head.labels, key=lambda lab: probs.get(lab, 0.0))
                    decision[head.name] = best
            out.append(decision)
        return out[0] if single else out

    def classify_jsonl(self, path: str | Path) -> list[dict]:
        """Score a JSONL file in the Fastino training format.

        Each line is ``{"input": ..., "output": {"classifications": [...]}}``
        where a classification is ``{"task", "labels", "true_label", optional
        "prompt", "multi_label", "label_descriptions"}``. Reuses this session.
        """
        rows: list[dict] = []
        schemas: list[dict] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                schema = {}
                for c in row["output"]["classifications"]:
                    spec = {
                        "labels": c["labels"],
                        "prompt": c.get("prompt", ""),
                        "multi_label": c.get("multi_label", False),
                    }
                    if c.get("label_descriptions"):
                        spec["label_descriptions"] = c["label_descriptions"]
                    schema[c["task"]] = spec
                rows.append(row)
                schemas.append(schema)

        results = []
        for row, schema in zip(rows, schemas):
            decision = self.classify_text(row["input"], schema)
            truth = {
                c["task"]: c.get("true_label")
                for c in row["output"]["classifications"]
            }
            results.append({"prediction": decision, "gold": truth})
        return results


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    shifted = x - x.max()
    exp = np.exp(shifted)
    total = exp.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full_like(exp, 1.0 / exp.size)
    return exp / total