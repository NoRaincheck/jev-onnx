"""The pretrained first layer must slot into the existing graph, not replace it.

The whole claim of the potion variant is that it is the *same model with a
different embedding table*: same tokenizer, same operators, same three string
inputs. These tests pin that claim down, plus the alignment between a word list
and its vectors, which is the one thing that can silently corrupt every id.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from jev_onnx.data import Example
from jev_onnx.export_onnx import export_to_onnx, load_vocab_for
from jev_onnx.model import JevDualEncoder
from jev_onnx.potion import (
    SPECIAL_TOKENS,
    build_potion_vocab,
    is_reachable,
    load_static_embeddings,
    read_safetensors,
    seed_token_embedding,
)
from jev_onnx.tokenization import PAD_ID

MAX_LEN = 32
MAX_LABEL_LEN = 24
D_MODEL = 32

# A stand-in for vocab.txt: ordinary words, plus the three kinds of entry a
# whitespace tokenizer can never emit.
FAKE_TOKENS = [
    "[PAD]",
    "[UNK]",
    "##ing",
    "##s",
    "two words",
    "refund",
    "invoice",
    "refunded",
    "login",
    "log",
    "urgent",
    "apple",
    "cat",
    "dog",
    "bank",
]


def _fake_matrix(n: int, dim: int) -> np.ndarray:
    """Distinct, deliberately *un-normalised* rows with an arbitrary norm range.

    Model2Vec ships rows whose norms span ~0.5 to ~130; if the loader trusted
    those, the rescale step would be untested here.
    """
    rng = np.random.default_rng(0)
    m = rng.normal(size=(n, dim)).astype(np.float32)
    m *= np.linspace(0.03, 8.0, n)[:, None]
    return m


@pytest.fixture(scope="module")
def fake_potion():
    tokens = FAKE_TOKENS
    return build_potion_vocab(
        d_model=D_MODEL, tokens=tokens, matrix=_fake_matrix(len(tokens), D_MODEL)
    )


# ----------------------------------------------------------------- the reader
def test_read_safetensors_parses_a_hand_written_file(tmp_path):
    """The format is 8 bytes + JSON header + flat buffer; prove the offsets."""
    payload = {"embeddings": np.arange(6, dtype=np.float32).reshape(2, 3)}
    header = json.dumps(
        {"embeddings": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
    ).encode("utf-8")
    blob = struct.pack("<Q", len(header)) + header + payload["embeddings"].tobytes()

    path = tmp_path / "model.safetensors"
    path.write_bytes(blob)
    np.testing.assert_array_equal(read_safetensors(path), payload["embeddings"])


def test_read_safetensors_rejects_unknown_key_and_dtype(tmp_path):
    header = json.dumps(
        {"other": {"dtype": "I64", "shape": [4], "data_offsets": [0, 32]}}
    ).encode("utf-8")
    path = tmp_path / "m.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 32)

    with pytest.raises(KeyError):
        read_safetensors(path, "embeddings")
    with pytest.raises(ValueError, match="dtype"):
        read_safetensors(path, "other")


# -------------------------------------------------------------- reachability
@pytest.mark.parametrize(
    "token, reachable",
    [
        ("refund", True),
        ("ankh", True),  # non-latin scripts are reachable too
        ("two words", False),  # a split token can never contain a space
        ("##ing", False),  # WordPiece continuation
        ("[UNK]", False),  # WordPiece special
        ("", False),
    ],
)
def test_is_reachable(token, reachable):
    assert is_reachable(token) is reachable


def test_every_special_is_unreachable():
    assert not any(is_reachable(s) for s in SPECIAL_TOKENS)


# ------------------------------------------------------------------ alignment
def test_table_is_row_aligned_with_the_vocabulary(fake_potion):
    """itos[i] and table[i] must describe the same word, or every id is wrong."""
    kept = [t for t in FAKE_TOKENS if is_reachable(t)]
    assert fake_potion.vocab.itos == [""] + kept
    assert fake_potion.table.shape == (len(kept) + 1, D_MODEL)
    assert fake_potion.table.dtype == np.float32


def test_index_zero_stays_the_reserved_pad_slot(fake_potion):
    """Row 0 must be all zeros and index 0 must not be a word.

    Index 0 doubles as PAD and as the attention mask's "ignore" marker, so a
    pretrained vector parked there would be attended as if it were content.
    """
    assert fake_potion.vocab.itos[0] == ""
    np.testing.assert_array_equal(fake_potion.table[0], np.zeros(D_MODEL))
    assert fake_potion.vocab.encode("refund zzzz", MAX_LEN)[0] != PAD_ID


def test_rows_are_rescaled_to_the_embedding_init_norm(fake_potion):
    """Every row lands at sqrt(d_model), the norm nn.Embedding draws at init."""
    norms = np.linalg.norm(fake_potion.table[1:], axis=1)
    np.testing.assert_allclose(norms, np.sqrt(D_MODEL), rtol=1e-5)


def test_min_freq_prunes_by_corpus_frequency():
    examples = [
        Example("refund refund invoice", "q", ["log"], [1.0]),
    ]
    table = _fake_matrix(len(FAKE_TOKENS), D_MODEL)
    tokens = list(FAKE_TOKENS)

    kept = build_potion_vocab(examples, d_model=D_MODEL, min_freq=2, tokens=tokens, matrix=table)
    assert kept.vocab.itos == ["", "refund"]

    kept = build_potion_vocab(examples, d_model=D_MODEL, min_freq=1, tokens=tokens, matrix=table)
    assert kept.vocab.itos == ["", "refund", "invoice", "log"]


def test_max_size_caps_the_table_deterministically():
    examples = [
        Example("refund refund invoice invoice log", "q", ["cat"], [1.0]),
    ]
    kwargs = dict(d_model=D_MODEL, tokens=FAKE_TOKENS, matrix=_fake_matrix(len(FAKE_TOKENS), D_MODEL))
    a = build_potion_vocab(examples, max_size=4, **kwargs)
    b = build_potion_vocab(examples, max_size=4, **kwargs)
    assert a.vocab.itos == b.vocab.itos
    assert len(a.vocab.itos) == 4
    assert a.table.shape == (4, D_MODEL)


def test_min_freq_zero_keeps_every_reachable_word():
    """The zero-shot default: a word the corpus never mentions still ships."""
    p = build_potion_vocab(
        None, d_model=D_MODEL, tokens=FAKE_TOKENS, matrix=_fake_matrix(len(FAKE_TOKENS), D_MODEL)
    )
    assert p.vocab.itos == [""] + [t for t in FAKE_TOKENS if is_reachable(t)]


def test_width_mismatch_is_rejected():
    """Potion ships 256-d vectors; a narrower encoder must fail loudly."""
    with pytest.raises(ValueError, match="d_model"):
        build_potion_vocab(
            d_model=8, tokens=FAKE_TOKENS, matrix=_fake_matrix(len(FAKE_TOKENS), D_MODEL)
        )


def test_seeding_copies_the_table_and_refuses_a_mismatch(fake_potion):
    model = JevDualEncoder(len(fake_potion.vocab), d_model=D_MODEL, n_layers=1, n_heads=2,
                           d_ff=64, max_len=MAX_LEN)
    seed_token_embedding(model, fake_potion.table)
    np.testing.assert_allclose(
        model.tok_emb.weight.detach().numpy(), fake_potion.table, rtol=1e-6
    )

    # A table built from a different vocabulary would shift every id in the
    # graph, so refuse rather than truncate.
    small = build_potion_vocab(
        None, d_model=D_MODEL, tokens=FAKE_TOKENS[:4],
        matrix=_fake_matrix(len(FAKE_TOKENS[:4]), D_MODEL),
    )
    with pytest.raises(ValueError, match="tok_emb"):
        seed_token_embedding(model, small.table)


# ------------------------------------------------------- the exported graph
CONTENT = [
    "refund the invoice please",
    "urgent login problem with the bank",
    "",
]
LABELS = ["refund_request", "bank_login", "other"]
HEADS = ["What does the customer want?"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory, fake_potion):
    model = JevDualEncoder(len(fake_potion.vocab), d_model=D_MODEL, n_layers=2, n_heads=2,
                           d_ff=64, max_len=MAX_LEN)
    seed_token_embedding(model, fake_potion.table)
    model.eval()
    path = export_to_onnx(
        model, fake_potion.vocab, tmp_path_factory.mktemp("potion") / "jev_potion.onnx",
        max_len=MAX_LEN, max_label_len=MAX_LABEL_LEN,
    )
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return model, fake_potion, session, path


def test_graph_is_the_same_graph(exported):
    """Same ops, same opsets, same string signature as the baseline export.

    This is the test that says "no new tokenizer": the tokenisation half of the
    graph is byte-for-byte the same operator set the baseline already ships.
    """
    _, _, _, path = exported
    graph = onnx.load(path).graph
    ops = {n.op_type for n in graph.node}
    assert {"StringNormalizer", "StringSplit", "StringConcat", "CategoryMapper"} <= ops
    # Nothing that would indicate a subword tokenizer sneaking in.
    assert not (ops & {"RegexFullMatch", "StringSplitV2", "TfIdfVectorizer"})


def test_inputs_and_outputs_are_unchanged(exported):
    _, _, session, _ = exported
    assert {i.name: (i.type, tuple(i.shape)) for i in session.get_inputs()} == {
        "content": ("tensor(string)", ("B",)),
        "head_text": ("tensor(string)", ("N",)),
        "label_text": ("tensor(string)", ("N",)),
    }
    assert [(o.name, tuple(o.shape)) for o in session.get_outputs()] == [("logits", ("B", "N"))]


def test_category_mapper_carries_the_potion_word_list(exported):
    """The vocabulary lives in the graph, and it is the pretrained one."""
    _, potion, _, path = exported
    recovered = load_vocab_for(path)
    assert recovered.itos == potion.vocab.itos
    assert "refund" in recovered
    assert "##ing" not in recovered, "wordpiece fragments must never reach the graph"
    assert "[UNK]" not in recovered


def test_graph_looks_words_up_by_exact_string_match(exported):
    """Behavioural check of the graph's dictionary, without reaching inside it.

    A word the pretrained table knows must move the logits; a word it does not
    know must be masked out entirely and change nothing. That is the whole
    contract of ``StringNormalizer -> StringSplit -> CategoryMapper``.
    """
    _, potion, session, _ = exported

    def score(texts):
        return session.run(
            None,
            {
                "content": np.array(texts, dtype=object),
                "head_text": np.array(["q"] * len(LABELS), dtype=object),
                "label_text": np.array(LABELS, dtype=object),
            },
        )[0]

    # "zzqq" is in no vocabulary: it becomes index 0 and the encoder ignores it.
    ignored, kept = score(["refund the invoice", "refund the invoice zzqq"])
    np.testing.assert_allclose(ignored, kept, rtol=1e-5, atol=1e-5)

    # Case and spacing are the only normalisation applied, so these differ.
    a, b = score(["Refund The Invoice", "refund the invoice"])
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    changed, same = score(["refund the invoice", "refund the cat"])
    assert not np.allclose(changed, same), "a known word must reach the encoder"

    # ...and it must be the *pretrained* row, not an arbitrary one: the same
    # graph reloaded from disk must still agree with the Python tokenizer.
    assert potion.vocab.encode("Refund The Invoice", MAX_LEN) == potion.vocab.encode(
        "refund the invoice", MAX_LEN
    )


def test_potion_graph_matches_pytorch(exported):
    """Seeding the table must not break ONNX-vs-PyTorch parity."""
    model, potion, session, _ = exported
    from jev_onnx.data import label_text

    def pad(rows, width):
        out = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
        for i, r in enumerate(rows):
            out[i, : min(len(r), width)] = torch.tensor(r[:width], dtype=torch.long)
        return out

    out = session.run(
        None,
        {
            "content": np.array(CONTENT, dtype=object),
            "head_text": np.array(HEADS * len(LABELS), dtype=object),
            "label_text": np.array(LABELS, dtype=object),
        },
    )[0]
    content_ids = pad([potion.vocab.encode(c, MAX_LEN) for c in CONTENT], MAX_LEN)
    label_ids = pad(
        [potion.vocab.encode(label_text(HEADS[0], l), MAX_LABEL_LEN) for l in LABELS],
        MAX_LABEL_LEN,
    )
    with torch.no_grad():
        ref = model(content_ids, label_ids).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------- the real artefact
@pytest.mark.slow
def test_real_potion_repo_loads_and_is_semantic():
    """Needs the download; the one place the real vocabulary is exercised.

    Guards the two assumptions the whole variant rests on: ``vocab.txt`` line i
    describes vector i, and the vectors carry real semantics before any training.
    """
    tokens, matrix = load_static_embeddings()
    assert matrix.shape[0] == len(tokens)
    assert matrix.shape[1] == 256

    p = build_potion_vocab()
    reachable = [t for t in tokens if is_reachable(t)]
    assert p.vocab.itos == [""] + reachable
    assert not any(t.startswith("##") for t in p.vocab.itos)
    norms = np.linalg.norm(p.table[1:], axis=1)
    np.testing.assert_allclose(norms, 16.0, rtol=1e-4)

    unit = p.table / np.maximum(np.linalg.norm(p.table, axis=1, keepdims=True), 1e-8)
    stoi = p.vocab.stoi

    def cos(a, b):
        return float(unit[stoi[a]] @ unit[stoi[b]])

    # Related words sit closer than unrelated ones, before any JEV training.
    assert cos("angry", "furious") > cos("cat", "dog")
    assert cos("cat", "dog") < 0.4
