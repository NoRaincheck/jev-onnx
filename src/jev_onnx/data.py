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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenization import LABEL_SEPARATOR, PAD_ID, Vocab, ort_split

HF_DATASET = "ZefanCai/Open-Jev"
HF_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"

# Splits that exist only on the main config; the side configs are train-only.
EVAL_SPLITS = ("validation", "test")

# The committed ~1k-row slice of Open-Jev. Same `data/<config>/<split>-*.parquet`
# layout as upstream, so the only thing `--data sample` changes is where the
# bytes come from. `resolve_source` decides; see also
# `scripts/build_sample.py`, which regenerates it.
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "vendor" / "open-jev-sample"


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
def resolve_source(source: str = "auto") -> str:
    """Decide where Open-Jev rows come from: ``"hub"`` or ``"sample"``.

    ``"sample"`` is the committed slice under ``vendor/open-jev-sample``. It is
    opt-in rather than a silent fallback, because it is a different corpus and a
    number measured on it is not comparable to one measured on the Hub. Asking
    for it and not finding it is an error rather than a quiet upgrade, so a
    typo can never quietly downgrade a training run.

    ``"auto"`` prefers the vendored slice when it is present and falls back to
    the Hub when it is not. It is not the default anywhere -- every entry point
    defaults to ``"hub"``, so a real training run is never silently pointed at
    the sample.
    """
    if source not in ("auto", "hub", "sample"):
        raise ValueError(f"source must be auto/hub/sample, got {source!r}")
    if source == "auto":
        source = "sample" if SAMPLE_DIR.is_dir() else "hub"
    if source == "sample" and not SAMPLE_DIR.is_dir():
        raise FileNotFoundError(
            f"--data sample requested but {SAMPLE_DIR} is missing.\n"
            "  regenerate it with:  uv run python scripts/build_sample.py"
        )
    return source


def load_examples(
    split: str = "train",
    configs: list[str] | None = None,
    max_rows: int | None = None,
    seed: int = 0,
    source: str = "hub",
) -> list[Example]:
    """Load Open-Jev into examples, optionally subsampling for a faster run.

    When ``max_rows`` is set the quota is applied *before* the rows are
    rendered, so a quick run parses 40k rows rather than all ~346k.

    The quota is per config (``ceil(max_rows / n_configs)``), so small configs
    contribute all of their rows and large ones are sampled. A consequence worth
    knowing: ``--max-rows 120000`` yields ~89k rows, not 120k, because the seven
    configs smaller than the 10k quota are exhausted rather than padded.

    ``source`` selects the corpus; see `resolve_source`.
    """
    from datasets import load_dataset

    source = resolve_source(source)
    if configs is None:
        configs = _available_configs(source)

    rng = random.Random(seed)
    quota = None if max_rows is None else math.ceil(max_rows / len(configs))

    out: list[Example] = []
    for n_cfg, cfg in enumerate(configs, 1):
        try:
            ds = _load_split(source, cfg, split)
        except (ValueError, FileNotFoundError):
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


def _available_configs(source: str) -> list[str]:
    """Config names, in a stable, source-appropriate order.

    Order is load-bearing, not cosmetic: `load_examples` draws each config's
    sample from one `random.Random(seed)` in sequence, so enumerating the same
    configs in a different order yields a *different* set of rows for the same
    seed. The Hub order is therefore taken exactly as
    `get_dataset_config_names` returns it and is never sorted -- sorting it would
    silently change every held-out number this project has published.

    The sample cannot ask the Hub (that would defeat the point of vendoring it,
    and offline the call returns a single config rather than failing), so it
    reads the order recorded in its own manifest at build time.
    """
    if source == "sample":
        manifest = SAMPLE_DIR / "manifest.json"
        if manifest.is_file():
            recorded = json.loads(manifest.read_text())["configs"]
            present = {p.name for p in (SAMPLE_DIR / "data").iterdir() if p.is_dir()}
            return [c for c in recorded if c in present]
        return sorted(p.name for p in (SAMPLE_DIR / "data").iterdir() if p.is_dir())
    from datasets import get_dataset_config_names

    # Deliberately unsorted: see the docstring.
    return list(get_dataset_config_names(HF_DATASET))


def _load_split(source: str, config: str, split: str):
    """One (config, split) as a `datasets.Dataset`, from the Hub or the sample."""
    from datasets import load_dataset

    if source == "sample":
        # The sample keeps upstream's directory layout, so this is the same
        # data_files glob the Hub reader resolves internally.
        pattern = SAMPLE_DIR / "data" / config / f"{split}-*.parquet"
        return load_dataset("parquet", data_files=str(pattern), split="train")
    return load_dataset(HF_DATASET, config, split=split, revision=HF_REVISION)


def corpus_texts(examples: list[Example]) -> list[str]:
    """Every string the graph will tokenise: content, plus each candidate pair.

    Single definition of "what the vocabulary is built over", shared by
    ``build_vocabulary`` and by the potion-pretrained variant's frequency filter.
    """
    texts: list[str] = []
    for ex in examples:
        texts.append(ex.content)
        for option in ex.options:
            texts.append(label_text(ex.question, option))
    return texts


def token_coverage(examples: list[Example], vocab: Vocab) -> tuple[float, float]:
    """How much of the corpus this vocabulary can actually represent.

    Returns ``(token_mass, types)``: the share of *token occurrences* that land
    on a real index, and the share of distinct types. Token mass is the number
    that matters -- everything else falls to index 0, which the encoder masks
    out, so it is silently dropped context.
    """
    counts: Counter[str] = Counter()
    for text in corpus_texts(examples):
        counts.update(ort_split(text))
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    known = sum(n for tok, n in counts.items() if tok in vocab)
    return known / total, sum(1 for tok in counts if tok in vocab) / len(counts)


def build_vocabulary(
    examples: list[Example],
    min_freq: int = 5,
    max_size: int = 32_000,
) -> Vocab:
    """Count over everything the graph will ever tokenise: content and labels."""
    return Vocab.build(corpus_texts(examples), min_freq=min_freq, max_size=max_size)
