"""The exported graph must agree with PyTorch, and must genuinely run on strings.

These are the tests that would catch a broken graft: a mis-ordered node, a
truncation/padding bug, or a label string assembled differently from training.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from jev_onnx.data import label_text
from jev_onnx.export_onnx import export_to_onnx, load_vocab_for
from jev_onnx.model import JevDualEncoder
from jev_onnx.tokenization import PAD_ID, Vocab

MAX_LEN = 32
MAX_LABEL_LEN = 24

CONTENT = [
    "Refund please, the invoice was charged twice",
    "I cannot log in to my account",
    "export fails again",
    "",
]
HEADS = [
    "Determine the broad category of this support ticket.",
    "How urgent is this?",
]
LABELS = {
    0: ["account: login and profile", "billing: charges and refunds", "bug_report: broken"],
    1: ["low", "high"],
}


def _tiny_model() -> tuple[JevDualEncoder, Vocab]:
    vocab = Vocab.build(CONTENT * 50 + [h for h in HEADS for _ in range(50)], min_freq=1)
    torch.manual_seed(0)
    model = JevDualEncoder(
        len(vocab), d_model=64, n_layers=2, n_heads=4, d_ff=128, max_len=MAX_LEN
    )
    model.eval()
    return model, vocab


def _pad(rows: list[list[int]], width: int) -> torch.Tensor:
    out = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : min(len(r), width)] = torch.tensor(r[:width], dtype=torch.long)
    return out


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    model, vocab = _tiny_model()
    path = export_to_onnx(
        model, vocab, tmp_path_factory.mktemp("onnx") / "model.onnx",
        max_len=MAX_LEN, max_label_len=MAX_LABEL_LEN,
    )
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return model, vocab, session, path


def _run(session, content, head_index, labels, prompts=None):
    """Run the graph exactly the way the inference wrapper does."""
    heads = [HEADS[head_index]] * len(labels) if prompts is None else [prompts] * len(labels)
    return session.run(
        None,
        {
            "content": np.array(content, dtype=object),
            "head_text": np.array(heads, dtype=object),
            "label_text": np.array(labels, dtype=object),
        },
    )[0]


def test_graph_accepts_strings_and_has_expected_signature(exported):
    _, _, session, path = exported
    inputs = {i.name: (i.type, tuple(i.shape)) for i in session.get_inputs()}
    assert inputs == {
        "content": ("tensor(string)", ("B",)),
        "head_text": ("tensor(string)", ("N",)),
        "label_text": ("tensor(string)", ("N",)),
    }
    assert [(o.name, tuple(o.shape)) for o in session.get_outputs()] == [
        ("logits", ("B", "N"))
    ]
    # The model must be self-describing: tokenizer baked in, no side-car file.
    assert path.exists() and path.stat().st_size > 0


def test_logits_match_pytorch(exported):
    model, vocab, session, _ = exported
    labels = LABELS[0]
    out = _run(session, CONTENT, 0, labels)
    assert out.shape == (len(CONTENT), len(labels))

    content_ids = _pad([vocab.encode(c, MAX_LEN) for c in CONTENT], MAX_LEN)
    label_ids = _pad(
        [vocab.encode(label_text(HEADS[0], l), MAX_LABEL_LEN) for l in labels],
        MAX_LABEL_LEN,
    )
    with torch.no_grad():
        ref = model(content_ids, label_ids).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_multiple_heads_in_one_pass(exported):
    """The point of the design: several decisions, one forward pass."""
    model, vocab, session, _ = exported
    content = CONTENT[:2]

    # Candidates from both heads go into a single call.
    flat_labels = LABELS[0] + LABELS[1]
    flat_heads = [HEADS[0]] * len(LABELS[0]) + [HEADS[1]] * len(LABELS[1])
    out = session.run(
        None,
        {
            "content": np.array(content, dtype=object),
            "head_text": np.array(flat_heads, dtype=object),
            "label_text": np.array(flat_labels, dtype=object),
        },
    )[0]
    assert out.shape == (2, 5)

    # Scoring head 1 alongside head 0 must not change head 0's answer.
    alone0 = _run(session, content, 0, LABELS[0])
    np.testing.assert_allclose(out[:, :3], alone0, rtol=1e-4, atol=1e-4)

    # ...and the concatenated form must equal the same thing scored separately.
    alone1 = _run(session, content, 1, LABELS[1])
    np.testing.assert_allclose(out[:, 3:], alone1, rtol=1e-4, atol=1e-4)


def test_label_order_does_not_change_scores(exported):
    """Candidates are scored independently, so permuting them permutes logits."""
    _, _, session, _ = exported
    labels = LABELS[0]
    straight = _run(session, CONTENT, 0, labels)
    shuffled = _run(session, CONTENT, 0, [labels[2], labels[0], labels[1]])
    np.testing.assert_allclose(straight[:, ::-1], shuffled, rtol=1e-4, atol=1e-4)


def test_out_of_vocabulary_text_is_survivable(exported):
    """Unknown words map to PAD (0) and are ignored rather than crashing."""
    _, _, session, _ = exported
    weird = "zzzz qqqq wwww"
    out = _run(session, [weird, ""], 0, LABELS[0])
    assert out.shape == (2, 3)
    assert np.isfinite(out).all(), "all-OOV input must not produce NaN"

    # An all-OOV row encodes identically to an empty one: both are all-padding.
    np.testing.assert_allclose(out[0], out[1], rtol=1e-4, atol=1e-4)


def test_long_input_is_truncated_not_crashed(exported):
    _, _, session, _ = exported
    long_text = " ".join(["word"] * 5000)
    out = _run(session, [long_text], 0, LABELS[0])
    assert out.shape == (1, 3)
    assert np.isfinite(out).all()


def test_vocabulary_roundtrips_from_the_graph(exported):
    _, vocab, _, path = exported
    recovered = load_vocab_for(path)
    assert recovered.itos == vocab.itos


def test_graph_is_valid_onnx(exported):
    _, _, _, path = exported
    model = onnx.load(path)
    onnx.checker.check_model(model)
    domains = {op.domain: op.version for op in model.opset_import}
    assert domains[""] == 20, "StringSplit needs ai.onnx opset 20"
    assert domains["ai.onnx.ml"] == 3, "CategoryMapper needs ai.onnx.ml opset 3"
    ops = {n.op_type for n in model.graph.node}
    assert {"StringSplit", "StringNormalizer", "StringConcat", "CategoryMapper"} <= ops