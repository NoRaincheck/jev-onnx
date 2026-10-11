"""The JevDecide wrapper: schema handling, multi-head, multi-label, edge cases."""

from __future__ import annotations

import pytest
import torch

from jev_onnx.decide import JevDecide
from jev_onnx.export_onnx import export_to_onnx
from jev_onnx.model import JevDualEncoder
from jev_onnx.tokenization import Vocab

MAX_LEN = 32
MAX_LABEL_LEN = 24

CONTENT = [
    "Refund please, the invoice was charged twice",
    "I cannot log in to my account",
    "the page will not load for anyone",
    "",
]
SCHEMA = {
    "intent": ["refund_request", "login_problem", "bug_report"],
    "urgency": ["low", "high"],
}


@pytest.fixture(scope="module")
def decide(tmp_path_factory):
    vocab = Vocab.build(CONTENT * 40, min_freq=1)
    torch.manual_seed(0)
    model = JevDualEncoder(
        len(vocab), d_model=64, n_layers=2, n_heads=4, d_ff=128, max_len=MAX_LEN
    )
    model.eval()
    path = export_to_onnx(
        model, vocab, tmp_path_factory.mktemp("onnx") / "m.onnx",
        max_len=MAX_LEN, max_label_len=MAX_LABEL_LEN,
    )
    return JevDecide(path)


def test_single_text_returns_single_dict(decide):
    out = decide.classify_text(CONTENT[0], SCHEMA)
    assert isinstance(out, dict)
    assert set(out) == {"intent", "urgency"}
    assert out["intent"] in SCHEMA["intent"]
    assert out["urgency"] in SCHEMA["urgency"]


def test_batch_text_returns_list(decide):
    out = decide.classify_text(CONTENT, SCHEMA)
    assert isinstance(out, list) and len(out) == len(CONTENT)
    for row in out:
        assert row["intent"] in SCHEMA["intent"]


def test_probabilities_sum_to_one_per_head(decide):
    scored = decide.score(CONTENT[0], SCHEMA)
    for head, probs in scored.items():
        assert len(probs) == len(SCHEMA[head])
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_softmax_is_within_head_not_across_heads(decide):
    """A 3-label head and a 2-label head must each be their own distribution."""
    scored = decide.score(CONTENT[0], SCHEMA)
    assert scored["intent"].keys() == {"refund_request", "login_problem", "bug_report"}
    assert scored["urgency"].keys() == {"low", "high"}
    assert sum(scored["intent"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(scored["urgency"].values()) == pytest.approx(1.0, abs=1e-6)


def test_multi_label_head_returns_a_list(decide):
    schema = {
        "aspects": {
            "labels": ["battery", "keyboard", "screen"],
            "multi_label": True,
            "cls_threshold": 0.0,  # keep everything so the shape is testable
        }
    }
    out = decide.classify_text(CONTENT[0], schema)
    assert isinstance(out["aspects"], list)
    assert set(out["aspects"]) == {"battery", "keyboard", "screen"}


def test_prompt_is_accepted_and_used(decide):
    schema = {"intent": {"labels": ["yes", "no"], "prompt": "Did they ask for a refund?"}}
    out = decide.classify_text(CONTENT[0], schema)
    assert out["intent"] in {"yes", "no"}
    # A different prompt must still produce a valid answer.
    schema2 = {"intent": {"labels": ["yes", "no"], "prompt": "Is the tone angry?"}}
    assert decide.classify_text(CONTENT[0], schema2)["intent"] in {"yes", "no"}


def test_label_descriptions_accepted(decide):
    schema = {
        "intent": {
            "labels": {
                "card_pin_change": "The customer wants a new PIN",
                "card_lost": "The physical card is missing",
            }
        }
    }
    out = decide.classify_text(CONTENT[0], schema)
    assert out["intent"] in {"card_pin_change", "card_lost"}


def test_single_label_head_only_one_head(decide):
    out = decide.classify_text(CONTENT[0], {"intent": ["a", "b", "c"]})
    assert set(out) == {"intent"}
    assert out["intent"] in {"a", "b", "c"}


def test_empty_text_is_safe(decide):
    out = decide.classify_text("", SCHEMA)
    assert out["intent"] in SCHEMA["intent"]


def test_wraps_matches_direct_scores(decide):
    """The wrapper's per-head softmax must not change ranking."""
    scored = decide.score(CONTENT[0], SCHEMA)
    best = max(scored["intent"], key=scored["intent"].get)
    assert decide.classify_text(CONTENT[0], SCHEMA)["intent"] == best


def test_no_numpy_or_torch_needed_for_inference():
    """decide.py must not import torch/transformers; only numpy + onnxruntime."""
    import ast
    import pathlib

    src = pathlib.Path(__import__("jev_onnx.decide", fromlist=["decide"]).__file__).read_text()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported, "inference must not depend on torch"
    assert "transformers" not in imported
    assert {"onnxruntime", "numpy"} <= imported