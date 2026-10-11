"""Determine what state_json actually holds, and how to get the classifiable content."""

import collections
import json

from datasets import get_dataset_split_names, load_dataset

cfg = "release-v2-redistributable"
print("splits for", cfg, ":", get_dataset_split_names("ZefanCai/Open-Jev", cfg))
print()

ds = load_dataset("ZefanCai/Open-Jev", cfg, split="train")

print("=" * 70)
print("state_json: is it JSON? what keys?")
print("=" * 70)
for r in ds.select(range(3)):
    sj = r["state_json"]
    print("--- raw repr (first 200):", repr(sj[:200]))
    try:
        parsed = json.loads(sj)
        if isinstance(parsed, dict):
            print("    dict keys:", list(parsed.keys())[:20])
        else:
            print("    parsed type:", type(parsed).__name__, "->", repr(parsed)[:150])
    except Exception as e:
        print("    not JSON:", e)
    print()

print("=" * 70)
print("per-source: shape of question vs state_json")
print("=" * 70)
by_src = collections.defaultdict(list)
for i, r in enumerate(ds.select(range(6000))):
    by_src[r["source"]].append(r)

for src, rows in by_src.items():
    r = rows[0]
    print(f"--- {src}  (n={len(rows)}) kind={r['kind']}")
    print("    question[:180] :", repr(r["question"][:180]))
    print("    state_json[:240]:", repr(r["state_json"][:240]))
    print("    options[:3]     :", json.dumps(r["options"][:3], ensure_ascii=False)[:220])
    print()

print("=" * 70)
print("do all rows in a source share one question? (head-level grouping)")
print("=" * 70)
for src, rows in by_src.items():
    q = collections.Counter(r["question"] for r in rows)
    o = collections.Counter(json.dumps(r["options"], ensure_ascii=False) for r in rows)
    print(f"{src:52s} uniq_question={len(q):5d} uniq_options={len(o):5d} rows={len(rows)}")