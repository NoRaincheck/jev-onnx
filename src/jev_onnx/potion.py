"""Pretrained static word vectors as the encoder's first layer.

`minishlab/potion-base-8M` is a Model2Vec model: a *distilled static embedding
table*, one 256-d vector per word, no transformer, no contextual passes. This
module lifts that table into ``JevDualEncoder.tok_emb`` so training starts from
real word semantics instead of noise. Everything above the embedding layer is
untouched -- the architecture, the loss and the export path are the ones in
``model.py`` / ``export_onnx.py``.

What is deliberately **not** here
--------------------------------
**No new tokenizer.** Nothing from ``transformers``, ``tokenizers`` or
``model2vec`` is imported, and no vocabulary side-car is needed at serving
time. The potion repo is read as what it plainly is -- a word list and a float
matrix -- and tokenisation stays exactly where it already was: inside the ONNX
graph, via

    StringNormalizer(LOWER) -> StringSplit -> CategoryMapper -> Slice -> Pad

So the exported potion model and the exported baseline model are the *same
graph*. The only two things that differ are the contents of the dictionaries the
graph already carried: ``CategoryMapper.cats_strings`` becomes the potion word
list, and ``tok_emb.weight`` starts from the matching vectors.

Why a *whitespace* word list and not the repo's WordPiece vocabulary
--------------------------------------------------------------------
``vocab.txt`` is inherited wholesale from ``baai/bge-base-en-v1.5``, so it still
contains the 5,828 wordpiece fragments (``##ing``, ``##s``) and the five
``[PAD] [UNK] [CLS] [SEP] [MASK]`` specials that a WordPiece tokenizer would
emit. A run of ``StringSplit`` can never produce any of them -- those strings
only exist because a *different* tokenizer was used to build the file. They are
dead weight in the embedding table, so `is_reachable` drops them. What is left
is exactly the set of strings this project's tokenizer can emit, which is also
the set the ``CategoryMapper`` can be asked about.

The practical consequence is a plain lowercased word vocabulary, looked up by
exact string match: that is the "exact string match" the pretrained table is
keyed by.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tokenization import Vocab, ort_split

POTION_REPO = "minishlab/potion-base-8M"

# A verbatim copy of the repo's two data files, committed under `vendor/` so the
# demo runs without reaching for the Hub. `resolve()` prefers it when present.
# Resolved relative to this file, which means it is found in a source checkout
# and simply not found in an installed wheel -- in which case the Hub is used.
VENDORED_DIR = Path(__file__).resolve().parents[2] / "vendor" / "potion-base-8M"

# Weight tensor inside `model.safetensors`. Model2Vec saves exactly one.
EMBEDDINGS_KEY = "embeddings"

# Artifacts of the WordPiece vocabulary potion inherited from its teacher.
CONTINUATION_PREFIX = "##"
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")

DOWNLOAD_FILES = ("vocab.txt", "model.safetensors")


# --------------------------------------------------------------------- loading
def resolve(repo: str = POTION_REPO) -> dict[str, Path]:
    """Locate ``vocab.txt`` and ``model.safetensors``, vendored copy first.

    `vendor/potion-base-8M/` holds a byte-identical copy of both files, so a
    clone of this repo trains and demos with no network access. Anything else --
    another model2vec repo, or a wheel installed outside the source tree -- falls
    through to ``huggingface_hub``, which caches, so it costs one download ever.

    ``huggingface_hub`` is already a transitive dependency of ``datasets``; it
    is used here for the download only and never at inference time.
    """
    vendored = {name: VENDORED_DIR / name for name in DOWNLOAD_FILES}
    if all(p.exists() for p in vendored.values()):
        return vendored

    from huggingface_hub import hf_hub_download

    return {name: Path(hf_hub_download(repo, name)) for name in DOWNLOAD_FILES}


def read_safetensors(path: str | Path, key: str = EMBEDDINGS_KEY) -> np.ndarray:
    """Read one tensor out of a ``.safetensors`` file without a new dependency.

    The format is deliberately trivial: a little-endian ``u64`` header length,
    a JSON header describing each tensor's dtype/shape/byte offsets, and then a
    flat buffer those offsets index into. Reading it here is ~20 lines and keeps
    the project free of a dependency that exists only to parse a download.

    Only float dtypes are accepted: this file is ``F32``, and an integer tensor
    would be a bug worth surfacing rather than silently reinterpreting.
    """
    dtype_of = {"F32": np.float32, "F64": np.float64, "F16": np.float16}
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        if key not in header:
            raise KeyError(f"{key!r} not in {Path(path).name}; has {sorted(header)}")
        info = header[key]
        if info["dtype"] not in dtype_of:
            raise ValueError(f"unsupported safetensors dtype {info['dtype']!r}")
        start, end = info["data_offsets"]
        fh.seek(header_len + 8 + start)
        raw = fh.read(end - start)
    arr = np.frombuffer(raw, dtype=dtype_of[info["dtype"]])
    return arr.reshape(info["shape"]).copy()


def load_static_embeddings(repo: str = POTION_REPO) -> tuple[list[str], np.ndarray]:
    """Return ``(vocab, matrix)`` exactly as stored in the repo.

    ``vocab[i]`` labels ``matrix[i]``; the two are parallel by line number, which
    is how Model2Vec writes them. The 5 bracketed specials and 5,828 wordpiece
    continuations are left in here -- filtering is a separate, deliberate step
    (`is_reachable`), so this function is a faithful reader and nothing else.
    """
    paths = resolve(repo)
    tokens = [line for line in paths["vocab.txt"].read_text(encoding="utf-8").split("\n") if line]
    matrix = read_safetensors(paths["model.safetensors"], EMBEDDINGS_KEY)
    if len(tokens) != matrix.shape[0]:
        raise ValueError(
            f"{repo}: {len(tokens)} tokens but {matrix.shape[0]} vectors -- misaligned"
        )
    if len(set(tokens)) != len(tokens):
        # A repeated word would give `CategoryMapper` two categories for one
        # string, and the int64 direction would disagree with the string one.
        raise ValueError(f"{repo}: vocab.txt contains duplicate entries")
    return tokens, matrix


# ------------------------------------------------------------------- filtering
def is_reachable(token: str) -> bool:
    """Could ``StringNormalizer(LOWER) -> StringSplit(delimiter="")`` ever emit this?

    The graph emits runs of non-space characters, lowercased. Three kinds of
    ``vocab.txt`` entry can never match that pattern, and none of them is a word:

    * anything containing a space -- a split token, by construction, cannot;
    * ``##`` continuations -- sub-word fragments belonging to a WordPiece
      vocabulary this project deliberately does not use;
    * ``[PAD] [UNK] [CLS] [SEP] [MASK]`` -- the same tokenizer's specials.
    """
    if not token or " " in token:
        return False
    if token.startswith(CONTINUATION_PREFIX):
        return False
    return token not in SPECIAL_TOKENS


# ---------------------------------------------------------------------- table
@dataclass(frozen=True)
class PotionVocab:
    """A graph-shaped vocabulary plus the pretrained vectors that go with it.

    ``vocab`` is an ordinary `Vocab`, so everything downstream
    (``JevDataset``, ``CategoryMapper``, ``load_vocab_for``) is unchanged: index
    0 is still the reserved PAD/unknown slot and still doubles as the attention
    mask. ``table`` is ``[vocab.size, d_model]`` and is copied straight into
    ``JevDualEncoder.tok_emb.weight``; row 0 is zeros, matching ``padding_idx``.
    """

    vocab: Vocab
    table: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.table.shape[1])


def build_potion_vocab(
    examples: list | None = None,
    *,
    d_model: int = 256,
    repo: str = POTION_REPO,
    min_freq: int = 0,
    max_size: int | None = None,
    init_scale: float = 1.0,
    tokens: list[str] | None = None,
    matrix: np.ndarray | None = None,
) -> PotionVocab:
    """Turn the potion table into a `Vocab` plus an aligned embedding matrix.

    ``min_freq`` (default 0) prunes the potion word list against the corpus:

    * ``0``   -- ship every reachable word. This is the recommended setting.
      A word Open-Jev never mentions still arrives with a trained vector instead
      of as index 0, and this model exists to score a label set the caller
      supplies at call time, so corpus frequency is the wrong thing to optimise
      for. Costs the full 23.7k x d_model table.
    * ``>0``  -- keep only words seen at least that often. Note that ``1``
      already means "seen at all": on a small ``--max-rows`` sample it can drop
      thousands of perfectly good words, so a sample-trained model is not a
      faithful stand-in for a full one here.
    * ``max_size`` additionally caps the table, most-frequent first. The
      baseline's ``--min-freq 5 --max-vocab 32000`` reproduces its *filtering
      rule* over a completely different word list, which is the fairest way to
      compare the two models on vocabulary policy alone.

    ``examples`` is the Open-Jev training set and is only read when a filter
    needs counts.
    """
    if tokens is None or matrix is None:
        tokens, matrix = load_static_embeddings(repo)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[1] != d_model:
        raise ValueError(
            f"{repo} vectors are {matrix.shape[1]}-d but d_model={d_model}; "
            "project them down or train a wider encoder"
        )

    kept = [tok for tok in tokens if is_reachable(tok)]

    counts: Counter[str] | None = None
    if examples is not None and (min_freq > 0 or max_size is not None):
        counts = Counter()
        from .data import corpus_texts

        for text in corpus_texts(examples):
            counts.update(ort_split(text))
    if counts is not None and min_freq > 0:
        kept = [tok for tok in kept if counts[tok] >= min_freq]

    if max_size is not None and len(kept) > max_size - 1:
        # Most frequent first, ties broken by potion's own order (stable sort),
        # so the resulting index assignment is deterministic across runs.
        if counts is None:
            kept = kept[: max_size - 1]
        else:
            kept = sorted(kept, key=lambda t: -counts[t])[: max_size - 1]

    index = {tok: i for i, tok in enumerate(tokens)}
    rows = matrix[[index[tok] for tok in kept]]

    # Renormalise every row to `sqrt(d_model)` before handing it to the encoder.
    #
    # The stored vectors are *not* unit length (potion's row norms run from ~0.5
    # to ~130), and that spread is an artefact of the distillation rather than a
    # signal: `emb_ln` is a LayerNorm, so it removes each position's mean and
    # scale anyway. Rescaling to the norm `nn.Embedding` draws at init
    # (sqrt(d_model) for N(0,1) entries) means the pretrained table starts in the
    # same numeric range as the random one it replaces.
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows / np.maximum(norms, 1e-8)
    rows *= np.sqrt(float(d_model)) * init_scale

    table = np.zeros((len(kept) + 1, matrix.shape[1]), dtype=np.float32)
    table[1:] = rows
    return PotionVocab(vocab=Vocab([""] + kept), table=table)


def seed_token_embedding(model, table: np.ndarray) -> None:
    """Copy a `PotionVocab` table into ``model.tok_emb.weight`` in place.

    The row count must match the vocabulary the graph will carry: the table *is*
    the ``CategoryMapper`` dictionary, so a mismatch would silently shift every
    id. Raise rather than pad or truncate.
    """
    import torch

    weight = model.tok_emb.weight
    if weight.shape != torch.from_numpy(table).shape:
        raise ValueError(
            f"table is {tuple(table.shape)} but tok_emb.weight is "
            f"{tuple(weight.shape)} -- build the vocabulary from the same table"
        )
    with torch.no_grad():
        weight.copy_(torch.from_numpy(table))
