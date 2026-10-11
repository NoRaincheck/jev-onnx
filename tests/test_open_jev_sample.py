"""The vendored Open-Jev sample must stay a faithful, self-contained slice.

``vendor/open-jev-sample/`` exists so every code path runs offline. That only
helps if the slice is actually faithful, so these tests pin the three properties
that make it safe: the rows are the Hub's rows, no group crosses a split, and
the loader reads them with no network access.

The one property that is deliberately *not* true -- ids are not globally unique,
because upstream's ``release-v2`` config is a subset of ``browser-drone`` -- is
pinned too, so a future regeneration that silently stopped inheriting the
overlap would be noticed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_onnx.data import SAMPLE_DIR, load_examples, resolve_source

SPLITS = ("train", "validation", "test", "ood")

requires_sample = pytest.mark.skipif(
    not SAMPLE_DIR.is_dir(), reason="vendor/open-jev-sample is not in this checkout"
)


def _manifest() -> dict:
    return json.loads((SAMPLE_DIR / "manifest.json").read_text())


def _files() -> list[Path]:
    return sorted(SAMPLE_DIR.glob("data/*/*.parquet"))


# ------------------------------------------------------------------- provenance
@requires_sample
def test_every_config_and_split_is_present():
    """A gap here means a smoke run silently skips a code path."""
    m = _manifest()
    expected = {f"{c}/{s}" for c in m["configs"] for s in SPLITS}
    assert set(m["files"]) == expected
    assert all(v["rows"] > 0 for v in m["files"].values()), "a cell is empty"


@requires_sample
def test_checksums_match_the_manifest():
    import hashlib

    for key, entry in _manifest()["files"].items():
        path = SAMPLE_DIR / "data" / f"{key}-00000-of-00001.parquet"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"{key} does not match manifest.json"


@requires_sample
def test_privileged_and_bulky_columns_are_dropped():
    """`metadata_json` holds privileged control labels and must not be here.

    The dataset card is explicit that it must never reach a model, and dropping
    it (with the 2 KB/row `record_json`) is most of why the slice is 768 KB.
    """
    import pyarrow.parquet as pq

    for path in _files():
        cols = set(pq.ParquetFile(path).schema_arrow.names)
        assert "metadata_json" not in cols
        assert "record_json" not in cols
        assert {"question", "options", "target", "state_json"} <= cols


# ------------------------------------------------------------------ structure
@requires_sample
def test_no_group_crosses_a_split():
    """Split separation is what makes the eval numbers mean anything.

    Slicing happens *within* one upstream split, so this cannot break by
    construction -- which is exactly why it is worth asserting.
    """
    import pyarrow.parquet as pq

    groups = {}
    for split in SPLITS:
        ids = set()
        for path in SAMPLE_DIR.glob(f"data/*/{split}-*.parquet"):
            ids |= set(pq.read_table(path, columns=["group_id"]).column("group_id").to_pylist())
        groups[split] = ids

    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            assert not groups[a] & groups[b], f"a group appears in both {a} and {b}"


@requires_sample
def test_upstream_config_overlap_is_preserved_not_hidden():
    """Duplicate ids are expected, and they are all release-v2 vs browser-drone.

    Upstream states that `release-v2-redistributable` is contained in
    `browser-drone-expansion-v1-redistributable`. `load_examples` iterates
    configs independently, so those rows are counted twice in training on the Hub
    too. The sample inherits that rather than papering over it.
    """
    import collections

    import pyarrow.parquet as pq

    seen = collections.defaultdict(set)
    for path in _files():
        config = path.parent.name
        for i in pq.read_table(path, columns=["id"]).column("id").to_pylist():
            seen[i].add(config)

    dupes = {i: c for i, c in seen.items() if len(c) > 1}
    assert dupes, "expected the inherited upstream overlap to be present"
    assert all(
        c == {"release-v2-redistributable", "browser-drone-expansion-v1-redistributable"}
        for c in dupes.values()
    )


# --------------------------------------------------------------------- loading
@requires_sample
def test_sample_config_order_matches_the_hub_declaration_order(monkeypatch):
    """Config enumeration order is load-bearing and must not drift.

    `load_examples` draws each config's sample from one seeded RNG in
    *sequence*, so enumerating the same configs in a different order selects a
    different set of rows for the same seed. Sorting the Hub list would silently
    change every held-out number this project has published, so the order is
    pinned here against `get_dataset_config_names`.
    """
    import datasets

    real = datasets.get_dataset_config_names
    monkeypatch.setattr(datasets, "get_dataset_config_names", lambda *a, **k: list(real(*a, **k)))

    from jev_onnx.data import _available_configs

    hub = _available_configs("hub")
    sample = _available_configs("sample")
    assert hub == list(real("ZefanCai/Open-Jev"))
    assert hub != sorted(hub), "the Hub order is declaration order, not alphabetical"
    assert sample == hub, "the sample must enumerate the same configs in the same order"


@requires_sample
def test_sample_never_asks_the_hub_for_its_config_list(monkeypatch):
    """Resolving the sample's configs must not make a Hub call.

    Offline, `get_dataset_config_names` does not fail -- it silently returns a
    single config, which would quietly reduce a `--data sample` run to one
    config without any error. The sample therefore reads its order from the
    manifest written at build time.
    """
    import datasets

    def _no_network(*a, **k):
        raise AssertionError("the sample must not ask the Hub for its config list")

    monkeypatch.setattr(datasets, "get_dataset_config_names", _no_network)
    from jev_onnx.data import _available_configs

    assert len(_available_configs("sample")) == 12


@requires_sample
def test_resolve_source_rejects_anything_but_the_three_names():
    for good in ("auto", "hub", "sample"):
        assert resolve_source(good) in ("hub", "sample")
    with pytest.raises(ValueError):
        resolve_source("nope")


def test_sample_is_opt_in_not_a_silent_fallback():
    """`--data sample` must never quietly resolve to the Hub.

    A smoke run and a real run are different experiments; if a typo or a missing
    directory silently upgraded to the Hub, the smoke run would become a real
    one and the reported number would be silently incomparable.
    """
    assert resolve_source("hub") == "hub"
    if SAMPLE_DIR.is_dir():
        assert resolve_source("sample") == "sample"
    else:
        with pytest.raises(FileNotFoundError):
            resolve_source("sample")


@requires_sample
def test_auto_prefers_the_sample_when_present():
    """`auto` is a convenience for smoke runs, and is nobody's default."""
    assert resolve_source("auto") == "sample"


@pytest.mark.parametrize("module", ["train", "train_potion", "evaluate"])
def test_every_entry_point_defaults_to_hub(module):
    """No CLI may default to the sample.

    A real training run silently pointed at a 944-row slice of a 345,909-row
    corpus would still produce a plausible-looking model and a plausible-looking
    accuracy, which is the failure mode this default exists to prevent.
    """
    import importlib

    src = Path(importlib.import_module(f"jev_onnx.{module}").__file__).read_text()
    assert 'choices=("hub", "sample", "auto")' in src
    assert 'default="hub"' in src


@requires_sample
@pytest.mark.parametrize("split", SPLITS)
def test_load_examples_reads_every_split_offline(split, monkeypatch):
    """The whole point: these four splits load with no network at all."""
    import datasets

    real = datasets.load_dataset

    def _local_only(repo_id, *a, **k):
        # The sample path reads local parquet; anything addressed by repo id
        # would be a Hub fetch, which is exactly what must not happen.
        if repo_id != "parquet":
            raise AssertionError(f"the sample must not reach the Hub (asked for {repo_id!r})")
        return real(repo_id, *a, **k)

    monkeypatch.setattr(datasets, "load_dataset", _local_only)
    examples = load_examples(split, source="sample")

    assert examples, f"{split} produced nothing"
    for ex in examples:
        assert ex.content
        assert ex.question
        assert len(ex.options) == len(ex.target)
        assert sum(ex.target) > 0


@requires_sample
def test_sample_and_hub_render_identically():
    """The sample must go through `render_content` like any other row.

    Reading it as raw bytes would be fine for a smoke test and wrong for
    anything else: ~95% of Open-Jev content is JSON-encoded structured state,
    and `render_content` is what flattens it into the string the graph reads.
    """
    import json

    from jev_onnx.data import render_content

    import pyarrow.parquet as pq

    path = next(SAMPLE_DIR.glob("data/*/train-*.parquet"))
    rows = pq.read_table(path).to_pylist()
    assert rows

    structured = 0
    for row in rows:
        expected = render_content(row["state_json"])
        assert isinstance(expected, str) and expected
        parsed = json.loads(row["state_json"])
        if not isinstance(parsed, str):
            structured += 1
            # Objects are re-serialised compactly, which is what keeps `\n`
            # escaped so ONNX StringSplit does not glue words across newlines.
            assert "\\n" in expected or "\n" not in expected
    assert structured, "expected at least one structured state in the sample"


@requires_sample
def test_sample_is_small_enough_to_commit_and_big_enough_to_be_useful():
    m = _manifest()
    rows = m["totals"]["all"]
    assert 500 <= rows <= 1500, f"sample drifted to {rows} rows"
    size_mb = sum(v["bytes"] for v in m["files"].values()) / 1e6
    assert size_mb < 5, f"sample grew to {size_mb:.1f} MB"


@requires_sample
def test_manifest_records_how_to_regenerate():
    """Provenance has to travel with the data or the checksums mean nothing."""
    m = _manifest()
    assert m["source"] == "ZefanCai/Open-Jev"
    assert len(m["revision"]) == 40, "the upstream revision must be pinned"
    assert m["license"] == "cc0-1.0"
    assert isinstance(m["seed"], int)
    # `sliced_cells` is informational -- it records where a group had to be cut
    # to fit the budget -- but it must at least name real cells.
    for cell in m["sliced_cells"]:
        config, split = cell.split("/")
        assert config in m["configs"] and split in SPLITS
