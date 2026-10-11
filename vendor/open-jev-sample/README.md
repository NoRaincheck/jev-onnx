# Open-Jev sample (vendored)

A small stratified slice of [`ZefanCai/Open-Jev`](https://huggingface.co/datasets/ZefanCai/Open-Jev),
committed so that every code path in this repo runs with **no network access**.

```
944 rows · 48 parquet files · 764 KB
source  revision c67699e13d0ae25e35b77165a4b6b079bedc8aba
license CC0-1.0 (dataset card); generated records only
```

## What this is for, and what it is not

The full corpus is **345,909 train rows across 48 files, ~48 MB**. That is the
right input for a real training run and the wrong thing to commit. This slice is
a *smoke* corpus: enough to exercise `load_examples`, the vocabulary builder,
`train.py`, `evaluate.py` and `compare.py` end to end, offline, in seconds.

**It is not a substitute for the real corpus and no metric here is comparable to
one measured on the Hub.** Accuracy on ~950 rows of a per-config sample is
meaningless as a benchmark — the slice is chosen to cover *shapes* (every config,
every split, every option-count), not to be representative in frequency. Use
`--data hub` for anything you intend to quote.

```bash
# smoke: no network, seconds
python -m jev_onnx.train --data sample --max-rows 400 --epochs 1 --out /tmp/smoke
python -m jev_onnx.evaluate --model artifacts/jev.onnx --data sample --rows 200

# the real thing: streams from the Hub
python -m jev_onnx.train --max-rows 120000 --epochs 3
```

`--data` accepts `hub` (default), `sample`, or `auto`. `sample` is deliberately
**opt-in and never a silent fallback**: if it is requested and missing it raises
rather than quietly upgrading to the Hub, so a typo can never downgrade a real
training run behind your back.

## Layout

Identical to upstream, so the loader's `data_files` glob is unchanged:

```
data/<config>/<split>-00000-of-00001.parquet
```

All **12 configs × 4 splits** (`train`, `validation`, `test`, `ood`) are present —
48 files, every cell non-empty — so a smoke run touches the same code paths a
full run does, including the unusual option-set shapes (16-way `citation`/`ood`
heads, 2-way `noul` yes/no heads).

`calibration` is upstream but deliberately absent: nothing in this repo reads it.

## How the slice was chosen

`scripts/build_sample.py`, with a fixed seed and fixed parquet writer options.
Budgets are 400/200/200/200 rows for train/validation/test/ood, spread evenly
over the 12 configs.

Groups (one piece of content, several decision heads) are taken **whole**,
smallest-first, so the budget packs as many distinct contents as possible. Where
no group fits the budget — the control configs have 24–287 rows per group, versus
a ~21-row cell budget — a single group is prefixed instead. 35 of the 48 cells do
this; the script prints which. The slice is therefore a *row* sample, not a
*group* sample, which matches what `load_examples` does anyway: it samples rows
without regard for groups.

Configs are enumerated in the **Hub's declaration order**, not sorted, and that
order is recorded in `manifest.json`. It is load-bearing: `load_examples` draws
each config's sample from one seeded RNG in sequence, so a different enumeration
order would select a different set of rows for the same seed — and would
silently change every held-out number this project has published.

The build is deterministic — same seed, same row selection, byte-identical
parquet. Re-running reproduces every checksum in `manifest.json`.

## Columns

Upstream ships 12 columns; 10 are kept:

| kept | dropped |
|---|---|
| `id`, `group_id`, `split`, `source`, `kind` | `metadata_json` — **privileged teacher/control labels the dataset card says must never reach a model** |
| `question`, `options`, `target`, `state_json` | `record_json` — the full original record, ~2 KB/row of duplicate |
| `original_line_number` | |

What remains is exactly what `data.load_examples` reads, plus the identity
columns needed to group rows and trace them back to the source split.
Dropping the two large columns is most of why the slice is 764 KB rather than
~3.5 MB.

## Known properties of the sample

Verified by `tests/test_open_jev_sample.py`:

* **Rows are byte-equal to upstream.** All 944 rows match the Hub on
  `question` / `options` / `target`.
* **No group crosses a split.** Upstream keeps whole families/conversations
  within one split, and slicing within a split cannot break that.
* **`id` is not unique across the sample — 38 ids appear twice.** This is
  inherited, not introduced: upstream `release-v2-redistributable` is a strict
  *subset* of `browser-drone-expansion-v1-redistributable` (the dataset card
  says so explicitly). All 38 sit in that pair. `load_examples` iterates
  configs independently, so those rows are counted twice in training, exactly as
  they are on the Hub — the slice preserves the corpus's real shape rather than
  papering over it.

## Verify

```bash
uv run python - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path("vendor/open-jev-sample")
m = json.loads((root / "manifest.json").read_text())
bad = [k for k, v in m["files"].items()
       if hashlib.sha256((root / "data" / f"{k}-00000-of-00001.parquet").read_bytes()).hexdigest()
       != v["sha256"]]
print("files:", len(m["files"]), "| rows:", m["totals"]["all"], "| mismatches:", bad)
PY
```

`manifest.json` records the source, pinned revision, seed, config order, per-file
row counts, byte sizes and sha256 checksums, plus which cells had to slice a
group.

## Licensing

The dataset card declares **CC0-1.0** for the generated content, which covers
these rows. The card also notes that upstream wording, external assets and
source data retain their own terms, and that the Wikispeedia portion was
excluded upstream for want of a verified redistribution license — this slice is
drawn from the published configs and so inherits that boundary. Upstream
repository: <https://github.com/Zefan-Cai/Open-Jev-Dev>.
