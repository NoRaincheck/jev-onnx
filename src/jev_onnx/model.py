"""A compact dual encoder that classifies text against an arbitrary label set.

Why a dual encoder rather than a fixed classification head: the decision label
set is chosen by the caller at inference time, exactly like
``fastino/GLiNER2.5-Decide``.  Encoding the content once and encoding each
candidate ``question + label`` with *shared* weights means any label set works
zero-shot, and several decision heads can be scored in one forward pass because
the content encoding is reused.

Scoring is a scaled cosine similarity, the same shape as CLIP:

    logits[b, n] = scale * <normalise(text_vec[b]), normalise(label_vec[n])>

The exported ONNX graph takes strings, not ids -- see ``export_onnx.py`` for how
the tokenisation half of the graph is grafted on.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenization import PAD_ID


class EncoderLayer(nn.Module):
    """Pre-norm transformer block written out explicitly.

    Hand-rolled rather than ``nn.TransformerEncoderLayer`` so the masking and
    the attention arithmetic are visible: the exported graph has to be a flat
    list of ONNX ops, and this keeps control over it.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        def split(z):
            return z.view(b, t, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # valid: [B, T] bool -> broadcast over heads and query positions
        attn = attn.masked_fill(~valid[:, None, None, :], torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)

        ctx = (attn @ v).permute(0, 2, 1, 3).reshape(b, t, d)
        x = x + self.drop(self.out(ctx))
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
        return x


class JevDualEncoder(nn.Module):
    """Shared-weight encoder for content and for label text."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_len: int = 256,
        dropout: float = 0.1,
        scale_init: float = 10.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.emb_ln = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(float(math.log(scale_init))))

    # ---------------------------------------------------------------- helpers
    def attention_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """Positions that carry information.

        Index 0 is always valid. Without this an all-OOV (or empty) input would
        mask every key, softmax would see an all ``-inf`` row and produce NaN.
        """
        valid = ids.ne(PAD_ID)
        valid = valid.clone()
        valid[:, 0] = True
        return valid

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, T] int64 -> L2-normalised vectors [B, d_model]."""
        valid = self.attention_mask(ids)
        x = self.tok_emb(ids) * math.sqrt(self.d_model)
        x = x + self.pos_emb.weight[: ids.shape[1]].unsqueeze(0)
        x = self.emb_drop(self.emb_ln(x))
        for layer in self.layers:
            x = layer(x, valid)
        x = self.final_ln(x)

        mask = valid.unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)

    # ---------------------------------------------------------------- forward
    def forward(self, text_ids: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        """text_ids: [B, T] content, label_ids: [N, T] candidate decisions.

        Returns [B, N] scaled cosine logits -- every candidate scored against
        every content row, which is what lets one call answer many heads.
        """
        text_vec = self.encode(text_ids)
        label_vec = self.encode(label_ids)
        # Clamp the scale: cosine similarity lives in [-1, 1], so an unbounded
        # exp() here would only ever push gradients into saturation.
        scale = self.logit_scale.clamp(max=4.6052).exp()  # <= 100
        return scale * (text_vec @ label_vec.t())

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def soft_classification_loss(
    logits: torch.Tensor,
    pair_row: torch.Tensor,
    pair_target: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over each example's own candidate subset.

    Open-Jev targets are distributions, not labels: ambiguity rows split mass
    evenly (e.g. ``[1/3, 1/3, 1/3]``). Treating those as a soft target with
    cross-entropy keeps that uncertainty instead of collapsing it to an argmax.

    * logits       [B, N] all candidate scores
    * pair_row     [N]    which example each candidate belongs to
    * pair_target  [N]    target mass for that candidate (0 for foreign ones)
    """
    b = logits.shape[0]
    same = pair_row.unsqueeze(0) == torch.arange(b, device=logits.device).unsqueeze(1)

    # Foreign candidates must not win their row's softmax. A large finite
    # negative (not finfo.min) keeps the masked log-probs in a safe numeric range
    # so that multiplying them by a soft target cannot overflow to -inf.
    mask_value = torch.finfo(logits.dtype).min / 2
    masked = logits.masked_fill(~same, mask_value)
    log_prob = masked.log_softmax(dim=-1)

    # Multiply each row's log-probs by that row's own targets, zero elsewhere.
    # gather pulls per-row target mass into [B, N] aligned with `same`.
    target = pair_target.unsqueeze(0).expand(b, logits.shape[1])
    target = target * same.to(target.dtype)

    n_rows = same.any(dim=1).sum().clamp(min=1)
    return -(target * log_prob).sum() / n_rows.to(logits.dtype)