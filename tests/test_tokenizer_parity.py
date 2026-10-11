"""The Python tokenizer must match the ONNX graph's tokenizer exactly.

If these drift, the model trains on one tokenisation and is served another,
which fails silently as lost accuracy.  This test builds the real operator
chain in onnxruntime and diffs it against ``jev_onnx.tokenization.ort_split``.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from jev_onnx.tokenization import Vocab, _ort_lower, ort_split

OPSET = 20


def _tokenizer_session() -> ort.InferenceSession:
    """The exact two-op chain the exported graph uses."""
    nodes = [
        helper.make_node(
            "StringNormalizer",
            ["x"],
            ["lo"],
            case_change_action="LOWER",
            is_case_sensitive=0,
        ),
        helper.make_node("StringSplit", ["lo"], ["y", "cnt"]),
        # ORT does not guarantee StringSplit's output rank; pin it to 2-D so the
        # test can index rows. The real graph does the same normalisation.
        helper.make_node("Shape", ["cnt"], ["cs"]),
        helper.make_node("ReduceMax", ["cnt"], ["mx"], keepdims=1),
        helper.make_node("Concat", ["cs", "mx"], ["tgt"], axis=0),
        helper.make_node("Reshape", ["y", "tgt"], ["y2"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tokenize",
        [helper.make_tensor_value_info("x", TensorProto.STRING, [None])],
        [
            helper.make_tensor_value_info("y2", TensorProto.STRING, [None, None]),
            helper.make_tensor_value_info("cnt", TensorProto.INT64, [None]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(
        model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )


@pytest.fixture(scope="module")
def session():
    return _tokenizer_session()


CASES = [
    "",
    " ",
    "   ",
    "hello world",
    "  leading and trailing  ",
    "double  space  between",
    "MiXeD CaSe",
    "punctuation, everywhere! (really) 42%",
    "tabs\tand\nnewlines\tmixed  end",
    "a b\nc d",
    "trailing newline\n",
    "  \n  ",
    "\n",
    "€500 and © symbol",
    "emoji \U0001f389 party time",
    "numbers 123 4567 and id like Abc-123def",
    "underscores_and.dots/hyphens-slashes",
    "ends with space ",
    " starts with space",
    "multiple   internal    runs     of spaces",
    "Grüße aus München",
    "ÄÖÜ äöü",
    "İstanbul",
    "ǅungla",
    "ΣΟΦΟΣ σοφος",
]


@pytest.mark.parametrize("text", CASES)
def test_ort_split_matches_graph(session, text):
    y, cnt = session.run(None, {"x": np.array([text], dtype=object)})
    assert y.ndim == 2, "Reshape normalisation failed to pin rank to 2"
    graph_tokens = y.tolist()[0]
    assert len(graph_tokens) == cnt.tolist()[0]
    assert ort_split(text) == graph_tokens, f"mismatch on {text!r}"


def test_batched_matches_graph(session):
    """Row-wise parity for a ragged batch (this is how the graph is really fed)."""
    batch = ["a b", "c d e f g", "h", "", "   ", "ONE two THREE"]
    y, cnt = session.run(None, {"x": np.array(batch, dtype=object)})
    assert y.shape[0] == len(batch)
    for i, text in enumerate(batch):
        row = y.tolist()[i][: cnt.tolist()[i]]
        assert ort_split(text) == row, f"mismatch on row {i} {text!r}"


def test_padded_slots_are_empty_strings(session):
    """Trailing padding from StringSplit is '', which must map to PAD_ID."""
    y, _ = session.run(None, {"x": np.array(["one two", "a b c"], dtype=object)})
    assert y.tolist() == [["one", "two", ""], ["a", "b", "c"]]


def test_vocab_roundtrip(tmp_path):
    texts = ["alpha beta gamma", "beta beta delta"] * 20
    v = Vocab.build(texts, min_freq=1)
    assert v.itos[0] == ""
    assert "beta" in v and "alpha" in v

    ids = v.encode("beta alpha unknownword", max_len=8)
    assert ids[0] == v.stoi["beta"]
    assert ids[1] == v.stoi["alpha"]
    assert ids[2] == 0, "out-of-vocabulary must fall back to index 0"

    p = tmp_path / "vocab.json"
    v.save(p)
    assert Vocab.load(p).itos == v.itos


def test_vocab_truncates_to_max_len():
    v = Vocab.build(["a b c d e"], min_freq=1)
    assert len(v.encode("a b c d e", max_len=3)) == 3


def test_min_freq_drops_rare_tokens():
    v = Vocab.build(["common common common rare"], min_freq=2)
    assert "common" in v
    assert "rare" not in v


def test_unicode_special_casing_matches_graph(session):
    """Titlecase letters and Turkish dotted capital I are the only divergences."""
    for text in [
        "İstanbul",
        "ǅUNGLA",
        "ǈUBROVIKA",
        "ǲ",
        "Ώ Ω ΣΣ",
        "ΑΣ Σ",
        "ǅ",
        "ß",
        "ﬁ",
        "ẞ",
        "Ǆ",
    ]:
        y, _ = session.run(None, {"x": np.array([text], dtype=object)})
        assert ort_split(text) == y.tolist()[0], f"mismatch on {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "ΑΣ Σ ΟΦΟΣ σοφος",  # final sigma is context-sensitive in Python, not in ORT
        "ΣΣΣ",
        "ΟΔΟΣ",
    ],
)
def test_final_sigma_parity(session, text):
    """Python's str.lower() applies the final-sigma rule; ORT does not."""
    y, _ = session.run(None, {"x": np.array([text], dtype=object)})
    assert ort_split(text) == y.tolist()[0]


def _normalizer_only_session() -> ort.InferenceSession:
    """StringNormalizer alone, for probing case folding without split interference."""
    graph = helper.make_graph(
        [
            helper.make_node(
                "StringNormalizer",
                ["x"],
                ["y"],
                case_change_action="LOWER",
                is_case_sensitive=0,
            )
        ],
        "fold",
        [helper.make_tensor_value_info("x", TensorProto.STRING, [None])],
        [helper.make_tensor_value_info("y", TensorProto.STRING, [None])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(
        model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )


def _bmp_fold_mismatches() -> list[tuple[int, str, str]]:
    """Fold every BMP codepoint through the real graph and through Python."""
    sess = _normalizer_only_session()
    sent = "@"  # no codepoint lowercases to a string containing '@'
    cps = [cp for cp in range(0x21, 0x10000) if not (0xD800 <= cp <= 0xDFFF)]
    mismatches = []
    for i in range(0, len(cps), 4096):
        chunk = cps[i : i + 4096]
        payload = np.array([sent + chr(cp) + sent for cp in chunk], dtype=object)
        folded = sess.run(None, {"x": payload})[0].tolist()
        for cp, txt in zip(chunk, folded):
            inner = txt[len(sent) : len(txt) - len(sent)]
            ours = _ort_lower(chr(cp))
            if inner != ours:
                mismatches.append((cp, inner, ours))
    return mismatches


@pytest.mark.slow
def test_exhaustive_bmp_case_folding():
    """Every BMP codepoint must fold identically in Python and in the graph.

    This is the guard that makes train/serve parity a guarantee rather than a
    hope: the only reason it passes is the 13-codepoint override table.
    """
    mismatches = _bmp_fold_mismatches()
    assert not mismatches, (
        f"{len(mismatches)} BMP codepoints diverge, first few: "
        + ", ".join(f"U+{cp:04X} (graph={g!r} ours={o!r})" for cp, g, o in mismatches[:10])
    )