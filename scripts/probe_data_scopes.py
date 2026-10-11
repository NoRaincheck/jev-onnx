"""Which sources are natural-language text vs structured state? Across all configs."""

import collections
import json

from datasets import get_dataset_config_names, load_dataset

CONFIGS = get_dataset_config_names("ZefanCai/Open-Jev")


def kind_of(sj):
    try:
        p = json.loads(sj)
    except Exception:
        return "unparsable"
    if isinstance(p, str):
        return "text"
    if isinstance(p, dict):
        return "dict"
    if isinstance(p, list):
        return "list"
    return type(p).__name__


print(f"{'config':45s} {'split':12s} {'rows':>7s}  {'text':>7s} {'dict':>7s}  sources")
print("-" * 130)

tot = collections.Counter()
for cfg in CONFIGS:
    try:
        d = load_dataset("ZefanCai/Open-Jev", cfg, split="train")
    except Exception as e:
        print(f"{cfg:45s} LOAD FAIL {type(e).__name__}: {str(e)[:60]}")
        continue
    kinds = collections.Counter()
    srcs = collections.Counter()
    for r in d:
        kinds[kind_of(r["state_json"])] += 1
        srcs[r["source"]] += 1
    tot.update(kinds)
    top = ", ".join(f"{k}={v}" for k, v in srcs.most_common(4))
    print(f"{cfg:45s} {'train':12s} {len(d):7d}  {kinds['text']:7d} {kinds['dict']:7d}  {top[:70]}")

print("-" * 130)
print("TOTAL over configs:", dict(tot))

print()
print("=" * 130)
print("DETAIL for main config: text-sources only")
print("=" * 130)
d = load_dataset("ZefanCai/Open-Jev", "release-v2-redistributable", split="train")
agg = collections.defaultdict(collections.Counter)
for r in d:
    agg[r["source"]][kind_of(r["state_json"])] += 1
print(f"{'source':50s} {'text':>7s} {'dict':>7s}")
for s, c in sorted(agg.items(), key=lambda kv: -sum(kv[1].values())):
    print(f"{s:50s} {c['text']:7d} {c['dict']:7d}")

print()
print("=" * 130)
print("All splits present in main config (row counts per split)")
print("=" * 130)
for sp in ["train", "calibration", "validation", "test", "ood"]:
    dd = load_dataset("ZefanCai/Open-Jev", "release-v2-redistributable", split=sp)
    k = collections.Counter(kind_of(r["state_json"]) for r in dd)
    print(f"  {sp:12s} rows={len(dd):7d} text={k['text']:7d} dict={k['dict']:7d}")