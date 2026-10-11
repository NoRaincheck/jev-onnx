"""Inspect Open-Jev: schema, splits, label structure, fanout groups, text shape."""

import collections
import json

from datasets import load_dataset

ds = load_dataset("ZefanCai/Open-Jev", "release-v2-redistributable", split="train")
print("train rows:", len(ds))
print("columns:", ds.column_names)
print()

row = ds[0]
for k, v in row.items():
    s = str(v)
    print(f"  {k:24s} {s[:220]}")
print()

print("=" * 70)
print("splits (full dataset)")
print("=" * 70)
from datasets import get_dataset_config_names, get_dataset_split_names

print("configs:", get_dataset_config_names("ZefanCai/Open-Jev"))
print()

print("=" * 70)
print("group fanout: how many decisions share one text?")
print("=" * 70)
g = collections.Counter(ds["group_id"])
hist = collections.Counter(g.values())
print("group_id -> #rows histogram:", sorted(hist.items())[:20])
print("n unique groups:", len(g))

print()
print("=" * 70)
print("kind distribution")
print("=" * 70)
print(collections.Counter(ds["kind"]))

print()
print("=" * 70)
print("task names (id suffix after group)")
print("=" * 70)
tasks = collections.Counter(r["id"].split(":")[-1] for r in ds.select(range(min(20000, len(ds)))))
print(tasks.most_common(30))

print()
print("=" * 70)
print("option-set shapes: (#options, do they sum to 1?)")
print("=" * 70)
shapes = collections.Counter()
sums = collections.Counter()
for r in ds.select(range(min(5000, len(ds)))):
    shapes[len(r["options"])] += 1
    sums[round(sum(r["target"]), 2)] += 1
print("len(options):", sorted(shapes.items()))
print("sum(target):", sums.most_common(8))

print()
print("=" * 70)
print("text length stats (chars)")
print("=" * 70)
import numpy as np

L = np.array([len(r["question"]) for r in ds.select(range(min(20000, len(ds))))])
print("question chars: mean %.0f p50 %d p90 %d max %d" % (L.mean(), np.percentile(L, 50), np.percentile(L, 90), L.max()))
srcs = ds["source"]
print("n unique source:", len(set(srcs)))
print("source counts:", collections.Counter(srcs).most_common(15))

print()
print("=" * 70)
print("full example: one group's rows")
print("=" * 70)
gid = row["group_id"]
grp = ds.filter(lambda r, gid=gid: r["group_id"] == gid)
print("TEXT:\n", row["question"][:600])
print()
for r in grp:
    print(f"--- id={r['id']} kind={r['kind']}")
    print("    opts:", json.dumps(r["options"], ensure_ascii=False)[:300])
    print("    tgt :", json.dumps(r["target"], ensure_ascii=False)[:200])