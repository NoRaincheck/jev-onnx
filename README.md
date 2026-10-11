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
# {'intent': 'order_status', 'urgency': 'low'}
```

Or drive the `.onnx` graph directly, with no Python SDK at all — see
[The ONNX interface](#the-onnx-interface). The two sections below spell out
exactly what the graph exposes and what the SDK adds on top:

* [The ONNX interface](#the-onnx-interface) — tensor contract, opsets, edge cases
* [The Python SDK](#the-python-sdk-jev_onnxdecide) — the `jev` schema and how it
  differs from the raw graph

Run the tour:

```bash
python demo.py --model artifacts/jev.onnx
```

Measure it on held-out rows — this scores the **exported file** through
onnxruntime, so the number is the one a consumer would see:

```bash
python -m jev_onnx.evaluate --model artifacts/jev.onnx --split validation
```

```
  rows scored      : 3,655  in 70.8s (19.4 ms/row)
  top-1 accuracy   : 0.6774
  chance baseline  : 0.3779  (1 / #options, averaged)
  prob on gold     : 0.6085
```

The chance baseline matters: Open-Jev heads carry 2–16 options, so a headline
accuracy without it is not interpretable.

Measured on the shipped `artifacts/jev.onnx` (3 epochs over 120k rows, ~11.5M
parameters, on an M1 Max):

| split | rows | top-1 | chance |
|---|---|---|---|
| `validation` | 3,655 | **0.677** | 0.378 |
| `test` | 2,000 | **0.685** | 0.371 |
| `ood` | 2,000 | **0.553** | 0.375 |

~1.8x chance on in-distribution splits and ~1.5x on the out-of-distribution
split, from a word-level model with no subword tokenisation. It is a demo of the
pipeline, not a leaderboard entry — the gaps that would close fastest are in
the limitations below.

---

## The ONNX interface

The `.onnx` file is a self-contained, language-agnostic classifier. Any runtime
that speaks ONNX — onnxruntime, `onnxruntime-web`, Java, C#, Go, Rust — can drive
it directly. You never need Python to use the model.

### Contract

| tensor | direction | elem type | shape | meaning |
|---|---|---|---|---|
| `content` | input | `tensor(string)` | `[B]` | the text to classify |
| `head_text` | input | `tensor(string)` | `[N]` | the question framing each candidate |
| `label_text` | input | `tensor(string)` | `[N]` | the candidate label |
| `logits` | output | `tensor(float)` | `[B, N]` | scaled cosine score per candidate |

* **IR version 10**, opsets `ai.onnx=20` and `ai.onnx.ml=3`. The string
  operators (`StringNormalizer`, `StringSplit`, `StringConcat`) require
  `ai.onnx` ≥ 20 and `CategoryMapper` requires `ai.onnx.ml`, which is why both
  imports are present.
* `B` and `N` are **symbolic** — any batch size and any candidate count work
  from one compiled model. The token width is the only fixed dimension.
* `head_text[i]` and `label_text[i]` describe the *same* candidate. The graph
  concatenates them internally as `"<label_text> | <head_text>"`, so a candidate's
  meaning is the pair, never either alone.
* `N` is the **total** number of candidates across every head, so a multi-head
  schema is flattened into one `[N]` call. Several decisions therefore cost a
  single forward pass: content is encoded once and scored against every
  candidate.
* Strings may be passed as `np.array(..., dtype=object)` (recommended) or any
  fixed-width NumPy string dtype; both are accepted.

### Calling it directly (no Python SDK)

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("artifacts/jev.onnx", providers=["CPUExecutionProvider"])

content = ["My subscription renewed after the service was down. Can I refund it?"]
labels  = ["refund_request", "login_problem", "other"]
prompt  = ["Was the customer asking for money back?"] * len(labels)

logits = sess.run(
    ["logits"],
    {
        "content":    np.array(content,   dtype=object),
        "head_text":  np.array(prompt,    dtype=object),
        "label_text": np.array(labels,    dtype=object),
    },
)[0]
# logits.shape == (1, 3)
```

The graph returns **raw logits**, one row per input text and one column per
candidate, in the order you supplied the candidates. It does not softmax, does
not group columns back into heads, and does not know which of your labels is the
"winner" — turning a score matrix into a decision is the caller's job. That is
the seam the Python SDK fills.

### Edge cases worth knowing

* `N == 0` is invalid — `StringNormalizer` requires a non-empty dimension, so
  always submit at least one candidate.
* `content = ""` or all-OOV text is safe: every word becomes index 0, the row
  collapses to the always-attended position 0, and the result is finite (not
  `NaN`).
* Over-long input is silently truncated to the token budget rather than erroring.
* Duplicate candidates are fine; each is scored independently and returns its own
  column.

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

## The Python SDK (`jev_onnx.decide`)

The raw graph is deliberately minimal: text in, a score matrix out. `JevDecide`
is the convenience layer that turns that matrix into the *decision interface*
modelled on [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide).

```python
from jev_onnx.decide import JevDecide

decide = JevDecide("artifacts/jev.onnx")          # one session, reused across calls

decide.classify_text(
    "My subscription renewed after the service was down. Can I get a refund?",
    {
        "intent": ["refund_request", "login_problem", "other"],
        "urgency": ["low", "high"],
        "aspects": {
            "labels": ["billing", "account", "outage"],
            "multi_label": True,
            "cls_threshold": 0.4,
        },
    },
)
# {'intent': 'refund_request', 'urgency': 'low', 'aspects': ['billing']}
```

(That is the real output from the shipped model — treat it as an illustration of
the *shape* of the result, not a claim about accuracy.)

### The `jev` schema

The schema is a `dict` of **head name → spec**, mirroring `classify_text`:

```python
{
  "<head name>": ["label_a", "label_b", ...],           # plain single-label head
  "<head name>": {
      "labels": [...],                 # candidate labels for this decision
      "prompt": "optional question",   # framing text; becomes head_text
      "multi_label": True,             # keep every label above the threshold
      "cls_threshold": 0.4,            # multi-label cut (default 0.4)
      "label_descriptions": {...},     # {label: description} to disambiguate
      # or "labels": {"card_lost": "The physical card is missing", ...}
  }
}
```

### What the SDK does for you

| Feature | Behavior |
|---|---|
| `classify_text(text, schema)` | Single-label head → winning label (`str`); multi-label head → list of labels above the threshold. Single string in → single dict out; list in → list of dicts. |
| `score(text, schema)` | Same, but returns **probabilities** `{head: {label: prob}}` instead of decisions, with a softmax applied *within* each head. |
| Multi-head in one pass | Heads are flattened and scored in a single `run()`. Heads with different label counts stay comparable because each is softmaxed separately. |
| Multi-label | `multi_label: True` + `cls_threshold` returns every label above the cut, in schema order. |
| Label descriptions | `"labels": {name: description}` (or `label_descriptions`) is folded into the candidate string so a description participates in the match. |
| Prompt | `prompt` becomes each candidate's `head_text` (the graph concatenates it as `"<label> | <prompt>"`). With no prompt, `head_text` is empty and the bare label carries the meaning. |
| Session reuse | One `InferenceSession` is built at construction and reused; pass `providers=[...]` and `intra_op_num_threads`. |

### ONNX vs SDK — what differs

The SDK is a thin, honest layer over the graph. It adds no modelling; it only
does bookkeeping and decision-making that the graph intentionally leaves out.

| Concern | Raw ONNX | Python SDK (`JevDecide`) |
|---|---|---|
| Inputs | You assemble `content`, `head_text`, `label_text` arrays by hand and keep them index-aligned. | You pass a `dict` schema; the SDK flattens it into aligned arrays. |
| Output shape | `[B, N]` float logits, candidates in submission order. | `{head: {label: prob}}` (from `score`) or `{head: label}` / `{head: [labels]}` (from `classify_text`). |
| Normalisation | Raw scaled-cosine logits; softmax is your problem. | Per-head softmax inside the SDK. |
| Decision | None — you argmax / threshold yourself. | Argmax for single-label; threshold for multi-label. |
| Head bookkeeping | None; you must remember which `[N]` columns belong to which head. | Handled; heads are regrouped and named for you. |
| Batching | You build the arrays and loop `run()`. | Pass a list of texts; one `run()` per call, result as a list. |
| Multi-label | Not a concept. | `multi_label` + `cls_threshold`. |
| Prompt / descriptions | Must be concatenated into the strings yourself (the graph also concatenates `label | head`). | `prompt` and `label_descriptions` keys handled. |

Rule of thumb: **use the raw graph** when you are on a non-Python runtime or want
the score matrix; **use the SDK** when you want `classify_text`'s
return-a-decision ergonomics on the Python side. Both are backed by the exact
same `.onnx` file.

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
  evaluate.py       scores the exported .onnx on a held-out split
tests/
  test_tokenizer_parity.py   graph vs Python, incl. every BMP codepoint
  test_onnx_export.py        ONNX-vs-PyTorch parity, multi-head, edge cases
  test_decide_api.py         schema handling, multi-label, batch, no-torch check
  test_model.py              loss numerics, NaN safety, scale clamp
  test_train_metrics.py      accuracy/loss bookkeeping
scripts/                     probes that document why the code is shaped this way
```

68 tests, ~5 seconds:

```bash
python -m pytest tests/ -q
```

---

## Known limitations

* **Truncation.** Content is budgeted at 448 tokens and label text at 96.
  Open-Jev content runs to 464 tokens (p99 = 457), so a small tail is truncated;
  label options are p99 = 18, so the option is always preserved and only the
  question tail is dropped. Both are `--max-len` / `--max-label-len`.
* **The head name is not encoded.** Candidates are scored as
  `"<label> | <prompt>"`. An unseen head *name* therefore carries no signal; the
  prompt and the label do. This is what makes unseen label sets work at all —
  renaming a head cannot change the scores — but it also means accuracy is
  highest on the taxonomies the model trained on. On a genuinely novel
  taxonomy, expect noticeably weaker results than the 0.68 above.
* **Word-level, no subwords.** Morphology is split from its stem (`refund` /
  `refunded` are separate embeddings). A character n-gram table would help, but
  there is no in-graph operator that can slice characters out of a string, so
  it would mean moving tokenisation back to Python.
* **Padding is fixed.** Every sequence is padded to its budget even when short.
  The encoder is length-agnostic, so the graph could take a dynamic sequence
  length; it is fixed here to keep the exported shapes static and predictable.
  At `max_len=448` a short email costs the same as a long one, which is where
  most of the 19 ms/row goes.
* **The model file is 49 MB** in fp32. Nothing here is load-bearing —
  `--d-model` / `--layers` trade accuracy for size, and quantising the weights
  is the first thing to try if that matters.