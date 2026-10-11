"""Unit tests for the dual encoder and the loss.

These cover the numeric edges that produced a silent `inf` during a first
training run: masked-out columns multiplied by a soft target, single-candidate
rows, all-OOV inputs, and the logit-scale clamp.
"""

from __future__ import annotations

import torch

from jev_onnx.model import JevDualEncoder, soft_classification_loss
from jev_onnx.tokenization import PAD_ID


def test_loss_is_finite_with_masked_columns():
    """A soft target times a -1e38 masked log-prob must not overflow to -inf.

    This is the bug that made the first training run report `loss inf`: the
    foreign columns were masked with finfo.min and then multiplied by a target.
    """
    torch.manual_seed(0)
    b, n = 4, 10
    logits = torch.randn(b, n)
    pair_row = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
    pair_target = torch.zeros(n)
    pair_target[0] = 1.0
    pair_target[3] = 1.0
    pair_target[6] = 0.5
    pair_target[7] = 0.5
    pair_target[8] = 1.0
    loss = soft_classification_loss(logits, pair_row, pair_target)
    assert torch.isfinite(loss), f"loss went non-finite: {loss}"


def test_loss_is_finite_with_extreme_logits():
    torch.manual_seed(0)
    logits = torch.randn(3, 5) * 5000
    pair_row = torch.tensor([0, 1, 1, 2, 2])
    pair_target = torch.zeros(5)
    pair_target[[0, 1, 3]] = 1.0
    loss = soft_classification_loss(logits, pair_row, pair_target)
    assert torch.isfinite(loss)


def test_loss_rewards_correct_predictions():
    """A confident, correct assignment must score better than a wrong one."""
    pair_row = torch.tensor([0, 0, 1, 1])
    pair_target = torch.zeros(4)
    pair_target[0] = 1.0  # row 0 -> col 0
    pair_target[2] = 1.0  # row 1 -> col 2
    good = torch.tensor([[5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]])
    bad = torch.tensor([[0.0, 0.0, 5.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
    assert soft_classification_loss(good, pair_row, pair_target) < soft_classification_loss(
        bad, pair_row, pair_target
    )


def test_loss_matches_cross_entropy_for_hard_targets():
    """With one-hot targets this is exactly cross-entropy over each row's own subset."""
    torch.manual_seed(1)
    pair_row = torch.tensor([0, 0, 1, 1])
    # Two rows, two candidates each (cols 0,1 -> row 0; cols 2,3 -> row 1).
    logits = torch.randn(2, 4)
    pair_target = torch.zeros(4)
    pair_target[1] = 1.0  # row 0 -> col 1
    pair_target[2] = 1.0  # row 1 -> col 2
    loss = soft_classification_loss(logits, pair_row, pair_target)

    # Manual reference: softmax only within each row's candidate columns.
    ref_row0 = logits[0, 0:2].log_softmax(-1)[1]
    ref_row1 = logits[1, 2:4].log_softmax(-1)[0]
    ref = -(ref_row0 + ref_row1) / 2
    assert torch.allclose(loss, ref, atol=1e-5)


def _model(vocab=200, **kw):
    torch.manual_seed(0)
    return JevDualEncoder(vocab, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_len=16, **kw).eval()


def test_encode_handles_all_pad_without_nan():
    """An all-padding row must not produce NaN: index 0 is always attended."""
    model = _model()
    ids = torch.full((2, 16), PAD_ID, dtype=torch.long)
    with torch.no_grad():
        vec = model.encode(ids)
    assert torch.isfinite(vec).all(), "all-PAD input produced NaN"


def test_encode_is_normalised():
    model = _model()
    ids = torch.randint(1, 200, (3, 16))
    with torch.no_grad():
        vec = model.encode(ids)
    norms = vec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_forward_shape_and_finiteness():
    model = _model()
    content = torch.randint(1, 200, (3, 16))
    labels = torch.randint(1, 200, (5, 16))
    with torch.no_grad():
        logits = model(content, labels)
    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()


def test_logit_scale_is_clamped():
    model = _model()
    with torch.no_grad():
        model.logit_scale.fill_(10.0)  # exp() would be ~22026
    content = torch.randint(1, 200, (2, 16))
    labels = torch.randint(1, 200, (4, 16))
    with torch.no_grad():
        logits = model(content, labels)
    assert torch.isfinite(logits).all()
    # scale is capped at 100 and cosine is <=1, so logits stay within +-100.
    assert logits.abs().max().item() <= 100.0 + 1e-3


def test_candidate_order_does_not_change_a_row_scores():
    """Each (content, candidate) pair is scored independently."""
    model = _model()
    content = torch.randint(1, 200, (1, 16))
    labels = torch.randint(1, 200, (4, 16))
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        a = model(content, labels)
        b = model(content, labels[perm])
    assert torch.allclose(a[:, perm], b, atol=1e-5)