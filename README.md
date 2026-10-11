# jev-onnx

A zero-shot decision classifier that ships as **one `.onnx` file** whose inputs
are **plain text**. No tokenizer, no vocabulary side-car, no PyTorch, no
`transformers` at inference time — onnxruntime runs the string pipeline itself.

Trained on [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev)
in vanilla PyTorch, and shaped after the ergonomics of
[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide):
pass any label set at call time and score several decision heads in one pass.

---

## The idea in one picture

```
        Open-Jev rows                        ONNX graph (the deliverable)
 ┌────────────────────────┐        ┌───────────────────────────────────────┐
 │ state_json ──render──▶  │        │  content   (string)                   │
 │ question               │        │     │                                  │
 │ options / target       │        │     ▼                                  │
 └────────────────────────┘        │  StringNormalizer(LOWER)               │
            │                      │     ▼                                  │
            ▼                      │  StringSplit                           │
  ┌───────────────────────┐        │     ▼                                  │
  │  PyTorch training     │        │  CategoryMapper  ← vocabulary lives    │
  │  dual encoder         │───────▶│     ▼             here, as an attr     │
  │  (ids in, logits out) │ export │  Slice → Pad → ids [B, L]            │
  └───────────────────────┘ graft  │     ▼                                  │
            │                      │  encoder ──▶ text_vec                  │
            ▼                      │  label_text, head_text ──▶ label_vec   │
   WordPiece-free tokenizer  ──────▶│     ▼                                  │
   that mirrors the graph exactly  │  logits [B, N]  (scaled cosine)         │
                                   └───────────────────────────────────────┘
```

The PyTorch model only ever sees integer ids, because that is all a transformer
can consume. `export_onnx.py` exports that tensor half with `torch.onnx.export`,
then **grafts a tokenisation subgraph** onto its inputs. The result is a graph
whose entry point is a string tensor.

---

## Quickstart

```bash
uv venv && uv pip install -e .
```

Train (subsample with `--max-rows` for a quick pass):

```bash
python -m jev_onnx.train --max-rows 120000 --epochs 2
```

Score text through the exported model:

```python
from jev_onnx.decide import JevDecide

decide = JevDecide("artifacts/jev.onnx")

decide.classify_text(
    "My subscription renewed after the service was already down. Can I get a refund?",
    {
        "intent": ["order_status", "refund_request", "login_problem", "other"],
        "urgency": ["low", "normal", "high", "critical"],
    },
)
# {'intent': 'refund_request', 'urgency': 'high'}
```

Run the tour:

```bash
python demo.py --model artifacts/jev.onnx
```

---

## What the ONNX graph actually does

### Inputs / outputs

| tensor | type | shape | meaning |
|---|---|---|---|
| `content` | string | `[B]` | the text to classify |
| `head_text` | string | `[N]` | the question framing each candidate |
| `label_text` | string | `[N]` | the candidate labels |
| `logits` | float | `[B, N]` | scaled cosine score per candidate |

There is no index tensor and no `head` name on the wire: the schema is flattened
into `[N]` candidate/question pairs before it reaches the graph, so any shape of
schema fits the same signature.

`N` is the **total** number of labels across all heads, so the schema is
flattened before it reaches the graph. Several decision heads therefore cost a
single forward pass: the content is encoded once and scored against every
candidate.

### The tokenisation subgraph

```
StringNormalizer(case_change_action=LOWER)
    ↓
StringSplit                        ← splits on runs of spaces
    ↓
Reshape(C, max(counts))            ← pins StringSplit's output rank
    ↓
CategoryMapper(vocab, default=0)   ← the dictionary; OOV → index 0
    ↓
Slice → Pad                        ← truncate to budget, pad to a fixed width
```

Three details that are load-bearing:

**`CategoryMapper` is the vocabulary.** It carries the whole word list as an
`ai.onnx.ml` attribute and maps anything unrecognised to `default_int64=0`.
Index 0 is reserved and never assigned to a real word, so OOV words and padding
share one representation and a single `ids != 0` recovers a correct attention
mask. That is the "skip out-of-dictionary words" behaviour you asked for, done
without a single conditional in the graph.

**`Reshape` pins `StringSplit`'s output rank.** onnxruntime does not
guarantee whether `StringSplit` returns `[C, M]` or a flattened `[C*M]` — the
same model was observed returning both. Rather than trusting the declared shape,
the graph reshapes using the per-row counts it already emits. This is why the
pipeline never trips over batch size.

**The label string is assembled in-graph.** `StringConcat` builds
`"<label> | <head>"`, byte-for-byte what `data.label_text` produces during
training. (`StringConcat` has no `sep` attribute at opset 20, so the separator
is broadcast in as a shape-`[1]` tensor.)

---

## Train/serve parity: the part that silently breaks things

The Python tokenizer must agree with the graph's tokenizer exactly, or the
model trains on one tokenisation and is served another — which shows up as
lost accuracy with no error anywhere. Three findings from probing the real
operators, all covered by `tests/test_tokenizer_parity.py`:

**1. `StringSplit` only splits on the literal space.** Tabs and newlines are
*not* delimiters, so `world\nfoo` stays one token. This is handled rather than
ignored: `json.dumps` escapes newlines inside strings as a literal `\n`, so
re-serialising structured state keeps token boundaries sane for the bulk of the
corpus. The residual cost is a few glued tokens in the ~15k plain-text rows.

**2. `StringNormalizer(LOWER)` is Unicode *simple* lowercasing, not Python's
`str.lower()`.** Probing every codepoint in the BMP found exactly **13**
divergences — Turkish `İ`, the titlecase digraphs (`ǅ ǈ ǋ ǲ`) and a handful of
archaic Greek/Cyrillic letters. ASCII, Latin-1, Cyrillic, CJK and punctuation
agree perfectly. Python's `str.lower()` also applies the **final-sigma** rule,
which is context-sensitive and which ORT does not implement; folding one
character at a time is what avoids firing it. All 13 are pinned in
`_ORT_LOWER_OVERRIDES`, and the test suite diffs the whole BMP on every run.

**3. `StringConcat` at opset 20 has no `sep` attribute** — despite what the
operator index implies.

---

## Architecture

A dual encoder, in the shape of CLIP:

```
logits[b, n] = scale · <normalize(text_vec[b]), normalize(label_vec[n])>
```

* Shared transformer encoder for content and for label text
* 4 layers, d_model 256, 4 heads, pre-norm — ~10M parameters (vocab dominates)
* Masked mean pooling; index 0 is always attended so an all-OOV input cannot
  produce a NaN row
* Trained with **soft** cross-entropy — Open-Jev targets are distributions, not
  labels (ambiguity rows split mass evenly, e.g. `[1/3, 1/3, 1/3]`), and
  collapsing them to an argmax throws away exactly the uncertainty the dataset
  is built to represent

Choosing a dual encoder over a fixed classification head is what makes the model
zero-shot: the label set is not baked in, so any taxonomy works at call time.

---

## The data

~95% of Open-Jev is structured, not prose. Each row's `state_json` is
JSON-encoded — about 15k rows hold a plain string (support tickets, email
bodies, transcripts) and ~331k hold an object. `data.render_content` passes
strings through and re-serialises objects compactly, so one graph handles both
and the model still receives one plain string.

Each row is one *decision*: content, question, candidate options, and a target
distribution. Rows sharing a `group_id` are different decisions about the same
content, which is exactly the multi-head shape the inference layer exposes.

---

## Layout

```
src/jev_onnx/
  tokenization.py   ORT-mirroring tokenizer + Vocab      ← the train/serve contract
  data.py           Open-Jev → examples, batching, soft targets
  model.py          the dual encoder and its loss
  train.py          training loop + export
  export_onnx.py    torch.onnx.export + subgraph grafting
  decide.py         JevDecide: the GLiNER-style scoring API
tests/
  test_tokenizer_parity.py   graph vs Python, incl. every BMP codepoint
  test_onnx_export.py        ONNX-vs-PyTorch parity, multi-head, edge cases
  test_decide_api.py         schema handling, multi-label, batch, no-torch check
```

```bash
python -m pytest tests/ -q
```

---

## Known limitations

* **Truncation.** Content is budgeted at 448 tokens and label text at 96.
  Open-Jev content runs to 464 tokens (p99 = 457), so a small tail is truncated;
  label options are p99 = 18, so the option is always preserved and only the
  question tail is dropped. Both are `--max-len` / `--max-label-len`.
* **Word-level, no subwords.** Morphology is split from its stem (`refund` /
  `refunded` are separate embeddings). A character n-gram table would help, but
  there is no in-graph operator that can slice characters out of a string, so
  it would mean moving tokenisation back to Python.
* **The head name is not encoded.** Candidates are scored as
  `"<label> | <prompt>"`. An unseen head *name* therefore carries no signal; the
  prompt and the label do. This is what makes unseen label sets work at all.
* **Padding is fixed.** Every sequence is padded to its budget even when short.
  The encoder is length-agnostic, so the graph could take a dynamic sequence
  length; it is fixed here to keep the exported shapes static and predictable.