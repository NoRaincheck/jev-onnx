"""Build the vendored Open-Jev sample committed under ``vendor/open-jev-sample/``.

    uv run python scripts/build_sample.py                 # rewrite the sample
    uv run python scripts/build_sample.py --out /tmp/s    # somewhere else

The full corpus is 345,909 train rows across 48 parquet files, ~48 MB. That is
the right input for a training run and the wrong thing to commit, so this writes
a small stratified slice that is enough to exercise every code path offline:
``load_examples``, the vocabulary builder, ``train.py``, ``evaluate.py`` and
``compare.py`` all run against it with no network access.

What the slice guarantees
-------------------------
* **Every (config, split) cell is non-empty.** All 12 configs x 4 splits = 48
  files, so a smoke run touches the same code paths a full run does -- including
  the configs with unusual option-set shapes (16-way ``citation``/``ood`` heads,
  2-way ``noul`` yes/no heads).
* **Group integrity is preserved wherever the budget allows it.** A ``group_id``
  is one piece of content with several decision heads attached, and the
  multi-head inference story only makes sense on intact groups. Groups are taken
  whole until the budget is spent. They are *not* always taken whole: the
  control configs have 24-287 rows per group, so a ~21-row-per-cell budget
  cannot hold one. Where no group fits, a single group is prefixed instead, and
  the script reports every cell that did so. The slice is therefore a row sample,
  not a group sample -- which is the same thing ``load_examples`` does anyway,
  since it samples rows from the corpus without regard for groups.
* **Split separation is preserved.** Each cell is drawn from exactly one upstream
  split and no group crosses a split upstream, so a group is never torn across
  two files here either.
* **Deterministic.** Fixed seed, seeded group shuffle, output sorted by ``id``,
  parquet written with fixed options. Re-running produces byte-identical files,
  which is what makes the committed checksums meaningful.

Columns
-------
``metadata_json`` and ``record_json`` are dropped. ``metadata_json`` carries
privileged teacher/control labels that the dataset card says must never reach a
model, and ``record_json`` is the full original record -- 3 KB/row, mostly a
duplicate of the fields kept. What remains is exactly what ``data.load_examples``
reads: ``state_json``, ``question``, ``options``, ``target``, plus the identity
columns needed to group and trace rows back to the source split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import get_dataset_config_names, load_dataset

# Pinned so a rerun is reproducible against the same upstream bytes. See
# vendor/open-jev-sample/README.md.
HF_DATASET = "ZefanCai/Open-Jev"
HF_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"

# Splits this project actually loads. `calibration` exists upstream but nothing
# here reads it -- Open-Jev reserves it for a calibration procedure this project
# does not implement -- so it is left out of the slice.
SPLITS = ("train", "validation", "test", "ood")

# Rows per split, spread evenly over the 12 configs. 1,000 total.
BUDGET = {"train": 400, "validation": 200, "test": 200, "ood": 200}

SEED = 20240617

# Everything `data.load_examples` reads, plus identity for grouping/traceability.
COLUMNS = [
    "id",
    "group_id",
    "split",
    "source",
    "kind",
    "question",
    "options",
    "target",
    "state_json",
    "original_line_number",
]

# Fixed writer options. `store_schema` pins the pandas metadata block so an
# identical table produces identical bytes across pyarrow versions that would
# otherwise embed a differing version string.
WRITE_OPTS = dict(compression="zstd", compression_level=9, version="2.6", store_schema=True)


def select_rows(rows: list[dict], budget: int, rng: random.Random) -> tuple[list[dict], bool]:
    """Pick up to ``budget`` rows from one config/split, favouring whole groups.

    Returns ``(rows, sliced)``; ``sliced`` says no group fitted the budget and a
    single one had to be prefixed instead.

    Groups are visited smallest-first, because a small budget can only hold
    small groups -- taking the smallest ones packs the budget with the greatest
    number of distinct *contents* rather than the few contents that happen to
    come first. The shuffle runs before that sort and Python's sort is stable,
    so equally-sized groups stay in a seeded random order rather than an
    arbitrary one, and the choice is reproducible across runs.
    """
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group_id"], []).append(r)

    groups = list(by_group.values())
    rng.shuffle(groups)
    groups.sort(key=len)  # stable: ties keep the shuffled order

    chosen: list[dict] = []
    for g in groups:
        if len(chosen) + len(g) > budget:
            continue
        chosen.extend(g)
    if not chosen:
        # Every group is larger than the whole budget (the control configs have
        # 24-287 rows each). Prefix one rather than emit an empty file.
        return sorted(groups[0][:budget], key=lambda r: r["id"]), True
    return sorted(chosen, key=lambda r: r["id"]), False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="vendor/open-jev-sample")
    args = ap.parse_args()

    out_dir = Path(args.out)
    # Replace `data/` and `manifest.json` only. A blanket rmtree would delete the
    # hand-written README.md that documents provenance and licensing -- which is
    # the one file this script does not generate.
    if (out_dir / "data").exists():
        shutil.rmtree(out_dir / "data")
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True)

    # Declaration order, not sorted: `load_examples` draws each config's sample
    # from one seeded RNG in sequence, so the order decides *which* rows a given
    # seed selects. Recording it here lets the sample reproduce the Hub's
    # enumeration offline, so a `--data sample` run and a `--data hub` run walk
    # the configs in the same order.
    configs = list(get_dataset_config_names(HF_DATASET))
    n_cfg = len(configs)

    manifest: dict = {
        "source": HF_DATASET,
        "revision": HF_REVISION,
        "license": "cc0-1.0",
        "seed": SEED,
        "budget": BUDGET,
        "configs": configs,
        "files": {},
        "totals": {},
    }
    sliced_cells: list[str] = []
    grand = 0

    for split in SPLITS:
        budget = BUDGET[split]
        # Spread the split's budget evenly over the configs, remainder to the
        # first few in declaration order. Even rather than proportional, so a
        # 4,000-row config does not crowd out a 2,520-row one entirely.
        base, extra = divmod(budget, n_cfg)
        split_total = 0

        for i, cfg in enumerate(configs):
            quota = base + (1 if i < extra else 0)
            ds = load_dataset(HF_DATASET, cfg, split=split, revision=HF_REVISION)
            rows = ds.to_list()
            rng = random.Random(f"{SEED}/{cfg}/{split}")
            picked, sliced = select_rows(rows, quota, rng)
            if sliced:
                sliced_cells.append(f"{cfg}/{split}")

            if not picked:
                continue
            # Write in the upstream layout so `data_files` globs are unchanged.
            path = data_dir / cfg / f"{split}-00000-of-00001.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Build the arrow schema from the rows themselves, restricted to the
            # kept columns. `from_pylist` infers list<string> for `options` and
            # list<double> for `target` correctly from the Python values, which
            # is the same shape the upstream parquet carries -- verified by
            # tests/test_open_jev_sample.py against the committed files.
            table = pa.Table.from_pylist(picked)
            pq.write_table(table.select(COLUMNS), path, **WRITE_OPTS)

            manifest["files"][f"{cfg}/{split}"] = {
                "rows": len(picked),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            split_total += len(picked)
            grand += len(picked)

        manifest["totals"][split] = split_total
        print(f"  {split:12s} {split_total:6,d} rows over {n_cfg} configs")

    manifest["totals"]["all"] = grand
    manifest["sliced_cells"] = sliced_cells
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    n_files = len(manifest["files"])
    size = sum(f["bytes"] for f in manifest["files"].values()) / 1e6
    print(f"\n  {n_files} files, {grand:,} rows, {size:.2f} MB -> {out_dir}")
    if sliced_cells:
        print(f"  {len(sliced_cells)} cells had to slice a group: {', '.join(sliced_cells)}")
    print(f"  sha256 manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
