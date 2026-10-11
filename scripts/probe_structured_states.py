"""What is inside the structured (dict) state? Natural language or numeric telemetry?"""

import json

from datasets import load_dataset

d = load_dataset("ZefanCai/Open-Jev", "mailroom-control-v1", split="train")
print("=" * 100)
print("mailroom-control-v1  (73k rows - largest config)")
print("=" * 100)
r = d[0]
print("QUESTION:", r["question"][:300])
print()
st = json.loads(r["state_json"])
print("STATE (truncated 1200):")
print(json.dumps(st, indent=1, ensure_ascii=False)[:1200])
print()
print("OPTIONS:", json.dumps(r["options"], ensure_ascii=False)[:300])
print("TARGET:", r["target"])

print()
print("=" * 100)
print("email-selection-control-v1")
print("=" * 100)
d2 = load_dataset("ZefanCai/Open-Jev", "email-selection-control-v1", split="train")
r2 = d2[0]
print("QUESTION:", r2["question"][:300])
print("STATE:", json.dumps(json.loads(r2["state_json"]), indent=1, ensure_ascii=False)[:1400])
print("OPTIONS:", json.dumps(r2["options"], ensure_ascii=False)[:300])
print("TARGET:", r2["target"])

print()
print("=" * 100)
print("sponsor-segment-control-v1 + workflow-controls-v1/customer_service")
print("=" * 100)
d3 = load_dataset("ZefanCai/Open-Jev", "sponsor-segment-control-v1", split="train")
r3 = d3[0]
print("Q:", r3["question"][:250])
print("STATE:", json.dumps(json.loads(r3["state_json"]), indent=1, ensure_ascii=False)[:900])
print("OPTS:", json.dumps(r3["options"], ensure_ascii=False)[:250])

print()
print("=" * 100)
print("numeric-only control: vizdoom-basic-v1 (from main config)")
print("=" * 100)
d4 = load_dataset("ZefanCai/Open-Jev", "release-v2-redistributable", split="train")
viz = d4.filter(lambda r: r["source"] == "vizdoom-basic-v1", num_proc=4)
print("vizdoom rows:", len(viz))
r4 = viz[0]
print("Q:", r4["question"][:250])
print("STATE:", json.dumps(json.loads(r4["state_json"]), indent=1, ensure_ascii=False)[:900])
print("OPTS:", json.dumps(r4["options"], ensure_ascii=False)[:250])

print()
print("=" * 100)
print("GROUPING: (source, question) = head  |  uniq option-sets per head")
print("=" * 100)
import collections

heads = collections.defaultdict(lambda: collections.Counter())
for r in d4.select(range(20000)):
    heads[(r["source"], r["question"])][json.dumps(r["options"], ensure_ascii=False)] += 1
print("total heads (source,question):", len(heads))
for (src, q), c in list(heads.items())[:14]:
    print(f"  {src:46s} n_optsets={len(c):3d} rows={sum(c.values()):6d} q={q[:70]!r}")