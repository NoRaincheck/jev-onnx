"""Content length + vocabulary sizing for Open-Jev under the ONNX tokenizer."""

import collections
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from jev_onnx.tokenization import ort_split  # noqa: E402

from datasets import get_dataset_config_names, load_dataset  # noqa: E402

CONFIGS = get_dataset_config_names("ZefanCai/Open-Jev")


def render(state_json: str) -> str:
    """state_json is JSON-encoded. Plain strings pass through; objects are
    re-serialised compactly so they become ordinary text for the tokenizer."""
    obj = json.loads(state_json)
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


rng = np.random.default_rng(0)
content_lens, label_lens, tok_counts = [], [], collections.Counter()

rows_per_cfg = {}
for cfg in CONFIGS:
    d = load_dataset("ZefanCai/Open-Jev", cfg, split="train")
    idx = rng.choice(len(d), size=min(1500, len(d)), replace=False)
    for i in idx:
        r = d[int(i)]
        content_lens.append(len(ort_split(render(r["state_json"]))))
        label_lens.append(len(ort_split(r["question"])))
        for opt in r["options"]:
            label_lens.append(len(ort_split(opt)))
        tok_counts.update(ort_split(render(r["state_json"])))
    rows_per_cfg[cfg] = len(d)

c = np.array(content_lens)
l = np.array(label_lens)
print(f"sampled content rows: {len(c)}   total train rows: {sum(rows_per_cfg.values())}")
print()
print("CONTENT token counts:")
for p in [50, 75, 90, 95, 99, 99.9]:
    print(f"  p{p:<5} {np.percentile(c, p):8.0f}")
print(f"  max   {c.max():8.0f}   mean {c.mean():.1f}")
print()
print("LABEL-side token counts (question + each option):")
for p in [50, 90, 95, 99, 99.9]:
    print(f"  p{p:<5} {np.percentile(l, p):8.0f}")
print(f"  max   {l.max():8.0f}   mean {l.mean():.1f}")
print()
print(f"distinct content tokens in sample: {len(tok_counts)}")
print("top:", [w for w, _ in tok_counts.most_common(12)])
non_ascii = sum(1 for w in tok_counts if not w.isascii())
print(f"non-ascii distinct tokens: {non_ascii}")
print()
for ml in [128, 192, 256, 320, 384, 512]:
    trunc = (c > ml).mean() * 100
    print(f"  max_len={ml:4d} -> truncates {trunc:5.2f}% of content rows")
print()
for ml in [32, 48, 64, 96, 128, 256]:
    trunc = (l > ml).mean() * 100
    print(f"  label max_len={ml:4d} -> truncates {trunc:5.2f}% of label strings")