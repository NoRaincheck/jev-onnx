# Training, data and evals

How a row of [Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) becomes
training data, how the model is trained on it, and how it is measured. The
[README](README.md) covers the exported graph, the inference API and the
vocabulary experiments; this covers everything on the training side.

```bash
uv venv && uv pip install -e .
```

---

## The corpus

12 configs, 5 splits each, published as Parquet. Only 4 splits are read here
(`train`, `validation`, `test`, `ood`); `calibration` is reserved upstream for a
procedure this project does not implement.

| | train | validation | test | ood |
|---|---:|---:|---:|---:|
| `release-v2-redistributable` | 79,116 | 3,723 | 10,356 | 15,701 |
| `browser-drone-expansion-v1-redistributable` | 108,624 | 5,493 | 14,726 | 25,160 |
| `mailroom-control-v1` | 73,472 | 6,027 | 8,323 | 22,960 |
| `amount-extraction-control-v1` | 32,984 | 992 | 3,472 | 9,920 |
| `phone-extraction-control-v1` | 12,350 | 380 | 1,615 | 3,800 |
| `ir-control-v1` | 7,366 | 464 | 986 | 2,320 |
| `entity-alignment-control-v1` | 6,944 | 280 | 1,008 | 2,240 |
| `silent-failure-control-v1` | 6,432 | 360 | 624 | 1,920 |
| `sponsor-segment-control-v1` | 6,345 | 540 | 999 | 2,160 |
| `context-retention-control-v1` | 6,138 | 558 | 522 | 2,160 |
| `email-selection-control-v1` | 3,618 | 189 | 432 | 1,080 |
| `citation-control-v1` | 2,520 | 180 | 300 | 800 |
| **sum over configs** | **345,909** | **19,186** | **43,363** | **90,221** |
| **unique `id`s** | **266,793** | **15,463** | **33,007** | **74,520** |

**That last row is not a correction, it is the corpus.** `release-v2-redistributable`
is a strict subset of `browser-drone-expansion-v1-redistributable` — every one of
its 79,116 train ids appears in the other — and the dataset card says so
explicitly ("do not add config totals and call them unique examples"). So ~23% of
train rows are seen twice per epoch. `load_examples` iterates configs
independently and does not de-duplicate, so this project trains on the weighted
corpus as published. It is not a bug in the loader; it is what the corpus is.

Data is pinned to revision `c67699e13d0ae25e35b77165a4b6b079bedc8aba`
(`data.HF_REVISION`) so a rerun reads the same bytes.

### What one row is

```python
{
  "id":           "customer-control-v1:42:9:category",
  "group_id":     "customer-control-v1:42:9",   # one content, several heads
  "kind":         "choice",                      # choice | noul | score
  "question":     "Determine the broad category of this support ticket. ...",
  "options":      ["billing: Charges, ...", "bug_report: ...", ...],
  "target":       [0.0, 0.0, 1.0, 0.0],          # a distribution, not a label
  "state_json":   "\"Customer Harbor-916e...: Exporting a report fails, ...\"",
}
```

One row is one **decision**: content, a question framing it, an ordered option
set, and a target distribution. Rows sharing a `group_id` are different decisions
about the *same* content — which is exactly the multi-head shape `JevDecide`
exposes at inference.

Three `kind`s, which behave differently and are all trained together:

| kind | options | target | share of train |
|---|---|---|---|
| `noul` | always `["no", "yes"]` | `[0,1]` / `[1,0]` | 210,790 (61%) |
| `choice` | 2–9, the real taxonomy | one-hot, sometimes split | 100,138 (29%) |
| `score` | 3–5 severity/ordinal bands | one-hot | 34,981 (10%) |

That `noul` rows are 61% of the corpus and binary is why the chance baseline is
what it is — see [Evals](#evals).

`metadata_json` carries privileged teacher/control labels and `record_json` is
the full original record. Neither is read by anything in this repo.

### ~95% of content is structured, not prose

`state_json` is JSON-encoded: ~15k rows hold a plain string (support tickets,
email bodies, transcripts) and ~331k hold an object.

`data.render_content` passes strings through and re-serialises objects compactly,
so **one graph handles both** and the model always receives one plain string:

```python
render_content('"Customer Harbor-916e: Exporting a report fails..."')
# -> unchanged

render_content('{"left": {"capacity": {"value": 500}, ...}}')
# -> '{"left":{"capacity":{"value":500},...}}'
```

Re-serialising also fixes a tokenisation problem for free: ONNX `StringSplit`
splits on **spaces only**, so a literal newline inside a string would glue the
words either side of it into one token. `json.dumps` escapes newlines as `\n`,
which keeps the boundaries sane for ~95% of the corpus. The residual cost is a
few glued tokens in the plain-text rows.

### Groups never cross a split

Verified over the whole corpus: **zero** `group_id` intersection between any pair
of splits. Whole conversations, documents and game families live in exactly one
split, so a held-out number is not contaminated by a group seen in training.

---

## Running with no network: the vendored sample

The full corpus is 48 MB. `vendor/open-jev-sample/` is a committed **944-row,
764 KB** slice — every one of the 12 configs × 4 splits — so the whole pipeline
runs offline:

```bash
python -m jev_onnx.train    --data sample --max-rows 400 --epochs 1 --out /tmp/smoke
python -m jev_onnx.evaluate --model artifacts/jev.onnx --data sample --rows 200
```

`--data` is `hub` (default), `sample`, or `auto`. **`sample` is opt-in and never
a silent fallback**: request it and it raises if missing, so a typo can never
quietly downgrade a real training run to a different corpus. Metrics from the
sample are **not comparable** to Hub numbers — the slice covers shapes, not
frequency. See [`vendor/open-jev-sample/README.md`](vendor/open-jev-sample/README.md);
regenerate with `python scripts/build_sample.py`.

---

## How a row becomes a training example

`src/jev_onnx/data.py`, in order:

**1. Render content.** `render_content(state_json) -> str`, above.

**2. Build the label string.** Every candidate is scored as
`"<option> | <question>"`:

```python
label_text(question, option) -> f"{option}{LABEL_SEPARATOR}{question}"
```

Option first, so if the 96-token label budget truncates, the thing being asked
*about* survives and only the tail of the (long) question is dropped.
`LABEL_SEPARATOR` must stay in step with `export_onnx.LABEL_SEPARATOR` — the
graph builds the same string with `StringConcat`.

**3. Tokenise.** With the ORT-mirroring tokenizer
(`tokenization.ort_split`), *not* a subword tokenizer — see
[Train/serve parity](README.md#trainserve-parity-the-part-that-silently-breaks-things).
Content is budgeted at `--max-len 448`, label text at `--max-label-len 96`.

**4. Build the vocabulary** (`--min-freq 5 --max-vocab 32000`) over
`corpus_texts(examples)` — content *and* every candidate pair, which is the
single definition of "what the model will ever tokenise". Index 0 is reserved for
PAD/unknown and doubles as the attention mask's ignore marker.

**5. Collate a batch** into four tensors:

| tensor | shape | meaning |
|---|---|---|
| `content_ids` | `[B, max_len]` | int64, 0 = ignore |
| `label_ids` | `[N, max_label_len]` | every candidate in the batch, stacked |
| `pair_row` | `[N]` | which example each candidate belongs to |
| `pair_target` | `[N]` | that candidate's target mass |

`pair_row` is what makes ragged option counts work: 32 examples with 2–9 options
each become one `[N]` score row, and the loss softmaxes **only over each row's
own candidates**. A candidate must not compete with another example's labels.

### The loss is soft on purpose

```python
logits[b,n] = scale · <normalize(text_vec[b]), normalize(label_vec[n])>
loss = -Σ_rows Σ_n  target[b,n] · log_softmax_over_row_n(logits)[b,n]
```

Open-Jev targets are distributions, not labels — ~1.2% of rows are genuinely
ambiguous and split mass evenly (e.g. `[1/3, 1/3, 1/3]`). Collapsing them to an
argmax at training time throws away exactly the uncertainty the dataset exists to
represent. This is a scaled-cosine dual encoder (CLIP-shaped) rather than a fixed
classification head, because a dual encoder is what makes the label set swappable
at call time.

---

## Training

Two entry points, one recipe, one difference:

| | command | first layer |
|---|---|---|
| **baseline** | `python -m jev_onnx.train` | `N(0,1)`, learned from scratch |
| **potion** | `python -m jev_onnx.train_potion` | `potion-base-8M` static vectors, then fine-tuned |

Deliberately plain: a manual loop over a `DataLoader`, AdamW, linear
warmup-then-cosine decay, `clip_grad_norm_(1.0)`. No trainer framework, so every
step between an Open-Jev row and an ONNX logit is visible in one file.

### Defaults

| flag | default | note |
|---|---|---|
| `--epochs` | 2 (3 in the README's runs) | |
| `--batch-size` | 32 | |
| `--lr` | 3e-4 | cosine-decayed, 5% linear warmup |
| `--weight-decay` | 0.01 | AdamW |
| `--max-rows` | `None` | all 345,909 |
| `--min-freq` / `--max-vocab` | 5 / 32000 | baseline only; potion defaults to 0 / all |
| `--max-len` / `--max-label-len` | 448 / 96 | |
| `--d-model` / `--layers` / `--heads` / `--d-ff` | 256 / 4 / 4 / 1024 | ~11.5M params |
| `--emb-lr-mult` | 0.1 (potion only) | 1.0 under `--init random` |
| `--device` | `mps` if available | |
| `--seed` | 0 | seeds data sampling *and* init |

### Reproducing the shipped models

```bash
python -m jev_onnx.train       --max-rows 120000 --epochs 3
python -m jev_onnx.train_potion --max-rows 120000 --epochs 3
python -m jev_onnx.train_potion --max-rows 120000 --epochs 3 \
       --init random --name jev_potion_randinit    # the control run
```

About 50 minutes each on an M1 Max. Both `.onnx` files are committed, so this is
only needed to rebuild them.

### `--max-rows` is a per-config quota, not a cap

This is the one thing about the flag that is easy to misread. The quota is
**`ceil(max_rows / n_configs)` applied to each config separately**, and small
configs are exhausted rather than padded:

```
--max-rows 120000  →  quota 10,000/config
  5 large configs × 10,000  =  50,000
  7 small configs  (2,520 … 7,366, all < 10,000) = 39,363
  total                        89,363 rows
```

So `--max-rows 120000` trains on **~89k rows, not 120k** — the README's "120k
rows" is the flag, not the count. The distribution is also not uniform: the small
configs are seen *in full* while `mailroom` and `browser-drone` are sampled. That
is fine (and arguably good — it keeps the rarer decision types in the mix) but it
is not proportional. At `--max-rows 20000` the quota is small enough that every
config is sampled and the total is truncated back to exactly 20,000.

---

## Evals

```bash
python -m jev_onnx.evaluate --model artifacts/jev.onnx --split validation
python compare.py --split test --rows 2000
```

`evaluate.py` scores **the exported `.onnx` through onnxruntime** — no PyTorch,
no tokenizer — so the number is the one a consumer sees. `compare.py` does the
same for several models on identical rows.

### Always report a chance baseline

Open-Jev heads carry **2–16 options**, so a bare accuracy is not interpretable:

| #options | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation rows | 12,344 | 2,201 | 1,075 | 1,070 | 1,099 | 731 | 410 | 128 | 128 |

A four-way head and a two-way head are not comparable, so `evaluate.py` averages
`1/#options` over the rows as the chance line. On the exact evaluation subsamples:

| split | rows | chance | baseline | potion |
|---|---:|---:|---:|---:|
| `validation` | 3,655 | 0.3779 | **0.6774** | 0.6019 |
| `test` | 2,000 | 0.3708 | **0.6850** | 0.5860 |
| `ood` | 2,000 | 0.3750 | **0.5530** | 0.5305 |

~1.8× chance in-distribution, ~1.5× OOD. A demo of the pipeline, not a
leaderboard entry — the gaps that would close fastest are in the README's
[Known limitations](README.md#known-limitations).

### Why the row counts are 3,655 and not 4,000

`--rows` is the same per-config quota as `--max-rows`, so the result is capped by
what the small configs can supply:

```
--rows 4000 → quota ceil(4000/12) = 334/config
  9 configs with ≥334 rows → 334 each  = 3,006
  citation 180 + entity-alignment 280 + email-selection 189 =   649
  total                                             3,655   (< the 4,000 asked for)
```

This is stable and reproducible: eval rows are drawn with `seed=1234` (training
uses `--seed 0`), so the numbers above are exactly re-derivable.

### Metrics

* **top-1 accuracy** — `argmax` of the prediction vs `argmax` of the target. The
  simplest view of a soft target.
* **chance baseline** — mean `1/#options`, per the table above.
* **prob on gold** — mean probability assigned to the gold option. More
  informative than accuracy when targets are soft, because it does not throw away
  how close the model was.
* **per question head** — accuracy broken down by the first 40 chars of
  `question`, so a headline number can be traced to the heads that drive it.

### Two eval paths, deliberately different

* `train.py:evaluate` runs **in-loop on the PyTorch model** every epoch, on
  `--eval-rows 4000` of `validation`. Cheap, used for model selection, never
  quoted as a result.
* `evaluate.py` / `compare.py` score **the exported file**, which is what a
  consumer actually gets. If these two ever disagree, the export is wrong — that
  is the point of keeping both. The per-row axis bug that once made in-loop
  accuracy meaningless is pinned by `tests/test_train_metrics.py`.

### Evaluating on your own labels

The model is zero-shot: the label set is supplied at call time, so you can score
your own taxonomy without retraining.

```python
decide = JevDecide("artifacts/jev.onnx")
decide.classify_text(text, {"intent": ["refund_request", "login_problem", "other"]})
```

Two honest caveats, both from the README's limitations: the **head name is not
encoded** (candidates are scored as `"<label> | <prompt>"`, so renaming a head
cannot change scores — good for unseen label sets, bad for taxonomies the model
never trained on), and the tokenizer is **word-level with no subwords**, so
`refund`/`refunded` are separate embeddings. For genuinely novel taxonomies
expect noticeably weaker results than the 0.68 above.

---

## Tests

```bash
python -m pytest tests/ -q          # 111 tests, ~20s
python -m pytest tests/ -q -m slow  # + the one test that needs the Hub
```

| file | what it protects |
|---|---|
| `test_open_jev_sample.py` | the vendored slice: byte-equal to upstream, no group crosses a split, loads with no network |
| `test_tokenizer_parity.py` | the Python tokenizer vs the real graph, incl. every BMP codepoint |
| `test_onnx_export.py` | ONNX-vs-PyTorch parity, multi-head, edge cases |
| `test_train_metrics.py` | per-row accuracy/loss bookkeeping |
| `test_data.py` | corpus text enumeration, vocabulary coverage |
| `test_potion.py` | table/vocabulary alignment, the same-graph guarantee |
| `test_model.py` | loss numerics, NaN safety, scale clamp |
| `test_decide_api.py` | schema handling, multi-label, batching |

`scripts/` holds probes that document *why* the code is shaped this way — see
[`scripts/README.md`](scripts/README.md).

---

## Licensing

Open-Jev is published **CC0-1.0** for its generated content
([card](https://huggingface.co/datasets/ZefanCai/Open-Jev)). The card also notes
that upstream wording, external assets and source data retain their own terms,
and that the Wikispeedia portion was excluded upstream for want of a verified
redistribution license. `vendor/open-jev-sample/` is drawn from the published
configs and inherits that boundary; `vendor/potion-base-8M/` is MIT, © Minish Lab.
