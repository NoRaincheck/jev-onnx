"""Pin down EXACT ONNX StringSplit(delimiter='') semantics so the Python
tokenizer can mirror it bit-for-bit. Train/inference parity depends on this.
"""

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

OPSET = 20


def _sess(delim_attr):
    nodes = [
        helper.make_node(
            "StringNormalizer", ["x"], ["lo"], case_change_action="LOWER", is_case_sensitive=0
        ),
        helper.make_node("StringSplit", ["lo"], ["y", "cnt"], **delim_attr),
    ]
    g = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.STRING, [None])],
        [
            helper.make_tensor_value_info("y", TensorProto.STRING, [None, None]),
            helper.make_tensor_value_info("cnt", TensorProto.INT64, [None]),
        ],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 10
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])


empty = _sess({})
space = _sess({"delimiter": " "})

CASES = [
    "hello world",
    "  leading and trailing  ",
    "double  space  between",
    "a\tb",
    "a\nb",
    "a\rb",
    "a\n\nb",
    "\n",
    "",
    "   ",
    "single",
    "tabs\tand\nnewlines\tmixed  end",
    "MiXeD CaSe ÄÖÜ ß İstanbul",
    "punctuation, everywhere! (really) 42%",
    "trailing newline\n",
    "  \n  ",
    "a b\nc d",
    "€500 and © symbol",
    "zero_widthspace",
    "emoji 🎉 party time",
]


def norm_like_ort(s):
    """Hypothesis for StringNormalizer(case_change_action=LOWER, is_case_sensitive=0)."""
    return s.lower()


def ref_split(sess, s):
    out, cnt = sess.run(None, {"x": np.array([s], dtype=object)})
    toks = out[0].tolist()
    return toks[0] if toks else []


print("=" * 100)
print("StringSplit delimiter=''  (empty) -- ORT reference")
print("=" * 100)
for s in CASES:
    r = ref_split(empty, s)
    print(f"  {s!r:46s} -> {r!r:52s} n={len(r)}")

print()
print("=" * 100)
print("Python hypothesis:  re.split(r' +', s.strip(' '))  after s.lower()")
print("=" * 100)


def py_split(s):
    s = norm_like_ort(s).strip(" ")
    if s == "":
        return [""]
    return [t for t in s.split(" ")]


ok = True
for s in CASES:
    ref = ref_split(empty, s)
    got = py_split(s)
    same = ref == got
    ok &= same
    print(f"  {'OK ' if same else 'BAD'} {s!r:40s} ref={ref!r:44s} py={got!r}")

print()
print("PARITY:", "PASS" if ok else "FAIL")

print()
print("=" * 100)
print("Compare delimiter=' ' vs delimiter=''")
print("=" * 100)
for s in ["double  space  between", "  lead trail  "]:
    a, _ = empty.run(None, {"x": np.array([s], dtype=object)})
    b, _ = space.run(None, {"x": np.array([s], dtype=object)})
    print(f"  {s!r:30s} empty={a[0].tolist()[0]!r:44s} space={b[0].tolist()[0]!r}")

print()
print("=" * 100)
print("maxsplit behaviour (bounds output width)")
print("=" * 100)


def _sess_maxsplit(k):
    nodes = [
        helper.make_node(
            "StringNormalizer", ["x"], ["lo"], case_change_action="LOWER", is_case_sensitive=0
        ),
        helper.make_node("StringSplit", ["lo"], ["y", "cnt"], maxsplit=k),
    ]
    g = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.STRING, [None])],
        [
            helper.make_tensor_value_info("y", TensorProto.STRING, [None, None]),
            helper.make_tensor_value_info("cnt", TensorProto.INT64, [None]),
        ],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 10
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])


s3 = _sess_maxsplit(3)
for s in ["a b c d e f g", "a b", "one"]:
    y, c = s3.run(None, {"x": np.array([s], dtype=object)})
    print(f"  maxsplit=3 {s!r:22s} -> shape {y.shape} {y.tolist()[0]!r} n={c[0] if len(c) else 0}")

print()
print("=" * 100)
print("BATCH ragged behaviour: padding + counts per row")
print("=" * 100)
batch = ["a b", "c d e f g", "h"]
y, c = empty.run(None, {"x": np.array(batch, dtype=object)})
print("  in  :", batch)
print("  out :", y.tolist(), "shape", y.shape)
print("  cnt :", c.tolist())