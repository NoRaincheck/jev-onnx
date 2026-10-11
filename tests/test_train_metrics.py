"""Training-loop metrics.

The accuracy here compares a *per-row* prediction against a *per-row* gold
column. An earlier version compared a `[B]` prediction against a `[N]` gold,
which silently produced a meaningless (near-zero) number.
"""

from __future__ import annotations

import torch

from jev_onnx.model import soft_classification_loss
from jev_onnx.train import evaluate


class _FakeLoader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


class _FakeModel(torch.nn.Module):
    """Returns fixed logits so the metric can be checked by hand."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, text_ids, label_ids):
        return self.logits


def test_accuracy_is_computed_per_row():
    # Two rows. Row 0 owns columns 0,1 and the answer is column 0.
    # Row 1 owns columns 2,3 and the answer is column 2.
    logits = torch.tensor(
        [
            [9.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
        ]
    )
    batch = {
        "content_ids": torch.zeros(2, 4, dtype=torch.long),
        "label_ids": torch.zeros(4, 4, dtype=torch.long),
        "pair_row": torch.tensor([0, 0, 1, 1]),
        "pair_target": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }
    model = _FakeModel(logits)
    out = evaluate(model, _FakeLoader([batch]), torch.device("cpu"))
    assert out["accuracy"] == 1.0, f"expected a perfect score, got {out}"
    assert out["n"] == 2
    assert out["loss"] > 0


def test_accuracy_penalises_wrong_picks():
    logits = torch.tensor(
        [
            [0.0, 9.0, 0.0, 0.0],  # row 0 wrongly prefers column 1
            [0.0, 0.0, 9.0, 0.0],  # row 1 correct
        ]
    )
    batch = {
        "content_ids": torch.zeros(2, 4, dtype=torch.long),
        "label_ids": torch.zeros(4, 4, dtype=torch.long),
        "pair_row": torch.tensor([0, 0, 1, 1]),
        "pair_target": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }
    out = evaluate(_FakeModel(logits), _FakeLoader([batch]), torch.device("cpu"))
    assert out["accuracy"] == 0.5


def test_ragged_candidate_counts_do_not_crash():
    """Rows with different numbers of options: 2, 3 and 2 columns."""
    logits = torch.zeros(3, 7)
    # Row 0 owns columns 0,1 (gold 1); row 1 owns 2,3,4 (gold 4); row 2 owns 5,6 (gold 6).
    logits[0, 1] = 5.0
    logits[1, 4] = 5.0
    logits[2, 6] = 5.0
    batch = {
        "content_ids": torch.zeros(3, 4, dtype=torch.long),
        "label_ids": torch.zeros(7, 4, dtype=torch.long),
        "pair_row": torch.tensor([0, 0, 1, 1, 1, 2, 2]),
        "pair_target": torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
    }
    out = evaluate(_FakeModel(logits), _FakeLoader([batch]), torch.device("cpu"))
    assert out["accuracy"] == 1.0
    assert out["n"] == 3


def test_loss_average_is_per_row_not_per_batch():
    """Two identical batches must not report half the loss of one batch."""
    logits = torch.zeros(2, 4)
    logits[0, 0] = 4.0
    logits[1, 2] = 4.0
    batch = {
        "content_ids": torch.zeros(2, 4, dtype=torch.long),
        "label_ids": torch.zeros(4, 4, dtype=torch.long),
        "pair_row": torch.tensor([0, 0, 1, 1]),
        "pair_target": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }
    model = _FakeModel(logits)
    one = evaluate(model, _FakeLoader([batch]), torch.device("cpu"))
    two = evaluate(model, _FakeLoader([batch, batch]), torch.device("cpu"))
    assert one["loss"] == pytest_approx(two["loss"]), (
        f"{one['loss']} vs {two['loss']}: averaging is not per-row"
    )
    assert two["n"] == 4


def pytest_approx(value, tol=1e-6):
    import pytest

    return pytest.approx(value, abs=tol)


def test_soft_classification_loss_is_the_training_objective():
    """Sanity: the loss evaluate() reports is the same one training uses."""
    logits = torch.randn(3, 6)
    pair_row = torch.tensor([0, 0, 1, 1, 1, 2])
    pair_target = torch.zeros(6)
    pair_target[0] = 1.0
    pair_target[2] = 0.5
    pair_target[3] = 0.5
    pair_target[4] = 1.0
    assert torch.isfinite(soft_classification_loss(logits, pair_row, pair_target))