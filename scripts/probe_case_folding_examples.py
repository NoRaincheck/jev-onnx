"""What exactly does ORT's StringNormalizer(case_change_action=LOWER) do?

Python's str.lower() is Unicode *full* lowercasing. ORT appears to use simple
case mapping. Probe per character so the Python mirror can match exactly.
"""

import sys
import unicodedata

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

CHARS = [
    "A", "Z", "a", "z",
    "É", "é", "À", "à", "Ç", "ç",
    "İ",  # U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE
    "ı",  # U+0131 dotless i
    "Ǆ", "ǅ", "ǆ",  # Dz digraph: uppercase / titlecase / lowercase
    "Σ", "σ", "ς",  # Sigma / final sigma
    "А", "а",  # Cyrillic
    "ß", "ẞ",  # sharp s
    "ﬁ",  # U+FB01 ligature fi
    "Ａ", "ａ",  # fullwidth
    "Ω", "ω", "Ώ",
    "Ⅰ",  # roman numeral one
    "ᾈ",  # Greek with ypogegrammeni
]

nodes = [
    helper.make_node(
        "StringNormalizer", ["x"], ["y"], case_change_action="LOWER", is_case_sensitive=0
    ),
    helper.make_node("StringSplit", ["y"], ["z", "c"]),
]
g = helper.make_graph(
    nodes,
    "g",
    [helper.make_tensor_value_info("x", TensorProto.STRING, [None])],
    [
        helper.make_tensor_value_info("z", TensorProto.STRING, [None, None]),
        helper.make_tensor_value_info("c", TensorProto.INT64, [None]),
    ],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
m.ir_version = 10
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

# Feed one string per char, separated by spaces, so each token is one char.
texts = ["x" + c + "x" for c in CHARS]
y, _ = sess.run(None, {"x": np.array(texts, dtype=object)})

print(f"{'char':>6} {'cp':>7} {'ORT':>8} {'py.lower()':>12} {'casefold':>10} {'simple?':>8}  match")
print("-" * 78)

simple_map = {}
mismatches = []
for c, row in zip(CHARS, y.tolist()):
    ort_out = row[0]
    got = ort_out[1:-1]  # strip the sentinels
    py_low = c.lower()
    py_cf = c.casefold()
    # simple lowercase: take the single codepoint result if the full mapping is 1 char
    simple = py_low if len(py_low) == 1 else py_low[0]
    ok_simple = got == simple
    ok_low = got == py_low
    flag = "lower" if ok_low else ("simple" if ok_simple else "NEITHER")
    if not (ok_low or ok_simple):
        mismatches.append((c, got, py_low, py_cf))
    print(
        f"{c:>6} U+{ord(c):05X} {got!r:>10} {py_low!r:>12} {py_cf!r:>10} {flag:>8}"
    )

print()
print("chars where neither str.lower() nor 'first char of lower' matches ORT:")
for c, got, py_low, py_cf in mismatches:
    print(f"  {c!r} U+{ord(c):05X} ORT={got!r} lower={py_low!r} casefold={py_cf!r}")
if not mismatches:
    print("  (none)")

print()
print("=" * 78)
print("Conclusion check: does ORT == 'str.lower()' for pure ASCII?")
print("=" * 78)
ascii_chars = [chr(c) for c in range(128)]
texts = ["x" + c + "x" for c in ascii_chars if c not in " "]
rows = sess.run(None, {"x": np.array(texts, dtype=object)})[0].tolist()
bad = [
    (c, r[0][1:-1])
    for c, r in zip([chr(c) for c in range(128) if c not in " "], rows)
    if r[0][1:-1] != c.lower()
]
print("ASCII mismatches:", bad if bad else "none")