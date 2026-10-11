"""Open-Jev -> (content, question, options, target) training examples.

Each Open-Jev row is one *decision*: a piece of content (``state_json``), the
question that frames it, a candidate option set and a target distribution.
Rows that share a ``group_id`` are different decisions about the same content,
which is exactly the multi-head shape the inference layer exposes.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .tokenization import LABEL_SEPARATOR, PAD_ID, Vocab

HF_DATASET = "ZefanCai/Open-Jev"

# Splits that exist only on the main config; the side configs are train-only.
EVAL_SPLITS = ("validation", "test")


def render_content(state_json: str) -> str:
    """Turn a row's ``state_json`` into the string the model will actually read.

    ``state_json`` is JSON-encoded. ~15k rows hold a plain string (support
    tickets, transcripts); the other ~331k hold an object. Re-serialising objects
    compactly turns them into ordinary text, so one ONNX graph handles both.

    A side benefit: ``json.dumps`` escapes newlines inside strings as a literal
    ``\\n``. ONNX ``StringSplit`` does not treat newlines as delimiters, so this
    keeps the token boundaries sane for the bulk of the corpus.
    """
    obj = json.loads(state_json)
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def label_text(question: str, option: str) -> str:
    """The text scored on the label side of the dual encoder.

    Option first so that, if the label budget truncates, the thing being asked
    about survives and only the tail of the (long) question is dropped. The
    separator must stay in step with ``export_onnx.LABEL_SEPARATOR``.
    """
    return f"{option}{LABEL_SEPARATOR}{question}"


@dataclass(slots=True)
class Example:
    content: str
    question: str
    options: list[str]
    target: list[float]


class JevDataset(Dataset):
    """Tokenises lazily and on demand, so the raw text can also be dumped."""

    def __init__(
        self,
        examples: list[Example],
        vocab: Vocab,
        max_len: int = 512,
        max_label_len: int = 192,
    ):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len
        self.max_label_len = max_label_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]

    def collate(self, batch: list[Example]) -> dict:
        """Flatten a batch of decisions into padded id tensors.

        Label rows from every example are concatenated into one ``[N, T]`` tensor
        and ``pair_row`` remembers which example each row belongs to, so the
        loss can softmax only over that example's own options.
        """
        content_ids = [self.vocab.encode(e.content, self.max_len) for e in batch]
        labels: list[list[int]] = []
        pair_row: list[int] = []
        pair_target: list[float] = []
        for row, ex in enumerate(batch):
            target = _normalise_target(ex.target)
            for option, t in zip(ex.options, target):
                labels.append(self.vocab.encode(label_text(ex.question, option), self.max_label_len))
                pair_row.append(row)
                pair_target.append(t)
        return {
            "content_ids": _pad(content_ids, self.max_len),
            "label_ids": _pad(labels, self.max_label_len),
            "pair_row": torch.tensor(pair_row, dtype=torch.long),
            "pair_target": torch.tensor(pair_target, dtype=torch.float32),
        }


def _normalise_target(target) -> list[float]:
    """Coerce to a probability distribution summing to 1."""
    vals = [float(t) for t in target]
    total = sum(vals)
    if total <= 0:
        # Degenerate row: fall back to a uniform distribution.
        return [1.0 / len(vals)] * len(vals)
    return [v / total for v in vals]


def _pad(rows: list[list[int]], width: int) -> torch.Tensor:
    out = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    for i, r in enumerate(rows):
        n = min(len(r), width)
        if n:
            out[i, :n] = torch.tensor(r[:n], dtype=torch.long)
    return out


# --------------------------------------------------------------------- loading
def load_examples(
    split: str = "train",
    configs: list[str] | None = None,
    max_rows: int | None = None,
    seed: int = 0,
) -> list[Example]:
    """Load Open-Jev into examples, optionally subsampling for a faster run.

    When ``max_rows`` is set the quota is applied *before* the rows are
    rendered, so a quick run parses 40k rows rather than all ~346k.
    """
    from datasets import get_dataset_config_names, load_dataset

    configs = configs or get_dataset_config_names(HF_DATASET)
    rng = random.Random(seed)
    quota = None if max_rows is None else math.ceil(max_rows / len(configs))

    out: list[Example] = []
    for n_cfg, cfg in enumerate(configs, 1):
        try:
            ds = load_dataset(HF_DATASET, cfg, split=split)
        except ValueError:
            continue  # this config does not have the requested split
        take = len(ds) if quota is None else min(quota, len(ds))
        indices = rng.sample(range(len(ds)), take) if take < len(ds) else range(len(ds))
        for i in indices:
            row = ds[int(i)]
            if not row["options"] or not row["target"]:
                continue
            out.append(
                Example(
                    content=render_content(row["state_json"]),
                    question=row["question"],
                    options=list(row["options"]),
                    target=list(row["target"]),
                )
            )
        print(f"    [{n_cfg}/{len(configs)}] {cfg:<42s} {len(out):>7,} rows", flush=True)
    rng.shuffle(out)
    if max_rows is not None and max_rows < len(out):
        out = out[:max_rows]
    return out


def build_vocabulary(
    examples: list[Example],
    min_freq: int = 5,
    max_size: int = 32_000,
) -> Vocab:
    """Count over everything the graph will ever tokenise: content and labels."""
    texts: list[str] = []
    for ex in examples:
        texts.append(ex.content)
        for option in ex.options:
            texts.append(label_text(ex.question, option))
    return Vocab.build(texts, min_freq=min_freq, max_size=max_size)