"""Exhaustively compare ORT's StringNormalizer lowercasing to Python's str.lower()
across the BMP, so the Python mirror can be made exact rather than guessed.
"""

import sys
import unicodedata

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

nodes = [helper.make_node("StringNormalizer", ["x"], ["y"], case_change_action="LOWER", is_case_sensitive=0)]
g = helper.make_graph(
    nodes,
    "g",
    [helper.make_tensor_value_info("x", TensorProto.STRING, [None])],
    [helper.make_tensor_value_info("y", TensorProto.STRING, [None])],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
m.ir_version = 10
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

SENT = "@"  # no codepoint lowercases to a string containing '@'


def probe(codepoints):
    """Feed '@<char>@' per codepoint; return ORT's folded inner text."""
    out = {}
    CH = 2048
    cps = list(codepoints)
    for i in range(0, len(cps), CH):
        chunk = cps[i : i + CH]
        payload = np.array([SENT + chr(cp) + SENT for cp in chunk], dtype=object)
        res = sess.run(None, {"x": payload})[0].tolist()
        for cp, txt in zip(chunk, res):
            out[cp] = txt[len(SENT) : len(txt) - len(SENT)] if txt.startswith(SENT) else txt
    return out


cps = [
    cp
    for cp in range(0x21, 0x10000)
    if not (0xD800 <= cp <= 0xDFFF)  # lone surrogates are not valid UTF-8
]
res = probe(cps)

divergent = []
for cp, ort_txt in res.items():
    py = chr(cp).lower()
    if ort_txt != py:
        divergent.append((cp, ort_txt, py, unicodedata.category(chr(cp))))

print(f"probed {len(cps)} codepoints; divergences from str.lower(): {len(divergent)}")
print()

# Classify the divergences
by_cat = {}
for cp, o, p, cat in divergent:
    by_cat.setdefault(cat, []).append((cp, o, p))
print("divergences by Unicode general category:")
for cat, items in sorted(by_cat.items()):
    print(f"  {cat}: {len(items)}")
print()

print("sample per category:")
for cat, items in sorted(by_cat.items()):
    print(f"  --- {cat} ({len(items)}) ---")
    for cp, o, p in items[:6]:
        ch = chr(cp)
        print(
            f"    U+{cp:04X} {ch!r} cat={cat}  ORT={o!r} ({len(o)} cp)  "
            f"py.lower={p!r} ({len(p)} cp)  title={unicodedata.combining(ch)!r}"
        )
    if len(items) > 6:
        print(f"    ... and {len(items)-6} more")

print()
print("=" * 70)
print("Are ALL divergences explained by (multi-char py expansion) or (Lt titlecase)?")
print("=" * 70)
unexplained = [
    (cp, o, p, cat)
    for cp, o, p, cat in divergent
    if not (len(p) > 1 or cat == "Lt")
]
print(f"unexplained: {len(unexplained)}")
for cp, o, p, cat in unexplained[:40]:
    print(f"  U+{cp:04X} {chr(cp)!r} cat={cat} ORT={o!r} py={p!r}")

print()
print("=" * 70)
print("ASCII / Latin-1 / common ranges clean?")
print("=" * 70)
for name, lo, hi in [
    ("ASCII", 0x00, 0x7F),
    ("Latin-1", 0xA0, 0x100),
    ("Latin Ext-A", 0x100, 0x180),
    ("Latin Ext-B", 0x180, 0x250),
    ("Greek", 0x370, 0x400),
    ("Cyrillic", 0x400, 0x460),
    ("General Punct", 0x2000, 0x2070),
    ("CJK", 0x4E00, 0x5000),
]:
    bad = [cp for cp in range(lo, hi) if cp in res and res[cp] != chr(cp).lower()]
    print(f"  {name:16s} divergences: {len(bad)}" + (f"  e.g. {[hex(c) for c in bad[:5]]}" if bad else ""))