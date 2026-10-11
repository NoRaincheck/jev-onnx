"""Probe which string ops the installed onnxruntime actually supports.

The whole design hinges on:
  1. StringSplit turning a string tensor into a token tensor.
  2. Some way to map arbitrary (incl. OOV) strings to integer ids INSIDE the graph.
"""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

print("ORT", ort.__version__)
print("onnx", onnx.__version__)
print("opset target:", 20)
print()


def run(ir, feeds, opset=20):
    ir.ir_version = 10
    onnx.checker.check_model(ir)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(ir.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


# ---------------------------------------------------------------- StringSplit
print("=" * 70)
print("1. StringSplit  [1-D input] -> [1-D or 2-D output]")
print("=" * 70)
for inp_dtype in (TensorProto.STRING,):
    node = helper.make_node("StringSplit", ["x"], ["y", "n"])
    g = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", inp_dtype, [2])],
        [
            helper.make_tensor_value_info("y", TensorProto.STRING, [2, None]),
            helper.make_tensor_value_info("n", TensorProto.INT64, [2]),
        ],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
    try:
        out = run(
            m,
            {"x": np.array(["hello world foo", "bar baz"], dtype=object)},
        )
        print("  OK -> tokens", out[0].shape, out[0].tolist(), "counts", out[1].tolist())
    except Exception as e:
        print("  FAIL:", type(e).__name__, str(e)[:220])

# multi-space / newlines / mixed case handled by empty delimiter?
print()
node = helper.make_node("StringNormalizer", ["x"], ["n"], case_change_action="LOWER", is_case_sensitive=0)
node2 = helper.make_node("StringSplit", ["n"], ["y", "_c"])
g = helper.make_graph(
    [node, node2],
    "g",
    [helper.make_tensor_value_info("x", TensorProto.STRING, [1])],
    [
        helper.make_tensor_value_info("y", TensorProto.STRING, [1, None]),
        helper.make_tensor_value_info("_c", TensorProto.INT64, [1]),
    ],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
try:
    out = run(m, {"x": np.array(["  HELLO   world\nfoo\tbar  "], dtype=object)})
    print("  messy whitespace OK ->", out[0].tolist(), out[1].tolist())
except Exception as e:
    print("  messy whitespace FAIL:", type(e).__name__, str(e)[:220])

print()

# ------------------------------------------------------ StringNormalizer (LC)
print("=" * 70)
print("2. StringNormalizer with case_change_action=LOWER (lowercasing in-graph)")
print("=" * 70)
node = helper.make_node(
    "StringNormalizer",
    ["x"],
    ["y"],
    case_change_action="LOWER",
    is_case_sensitive=0,
)
g = helper.make_graph(
    [node],
    "g",
    [helper.make_tensor_value_info("x", TensorProto.STRING, [3])],
    [helper.make_tensor_value_info("y", TensorProto.STRING, [3])],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
try:
    out = run(m, {"x": np.array(["HELLO World", "ABC def", "MiXeD"], dtype=object)})
    print("  OK ->", out[0].tolist())
except Exception as e:
    print("  FAIL:", type(e).__name__, str(e)[:220])

print()

# ---------------------------------------------- CategoryMapper (string -> int)
print("=" * 70)
print("3. ai.onnx.ml.CategoryMapper  string -> int64  (dictionary lookup!)")
print("=" * 70)
node = helper.make_node(
    "CategoryMapper",
    ["x"],
    ["y"],
    domain="ai.onnx.ml",
    cats_strings=["hello", "world", "foo", "bar"],
    cats_int64s=[1, 2, 3, 4],
    default_int64=0,
)
g = helper.make_graph(
    [node],
    "g",
    [helper.make_tensor_value_info("x", TensorProto.STRING, [5])],
    [helper.make_tensor_value_info("y", TensorProto.INT64, [5])],
)
m = helper.make_model(
    g, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("ai.onnx.ml", 3)]
)
m.ir_version = 10
try:
    out = run(m, {"x": np.array(["hello", "bar", "zzz", "foo", "world"], dtype=object)})
    print("  OK ->", out[0].tolist(), "(expect [1, 4, 0, 3, 2]; zzz=default 0)")
except Exception as e:
    print("  FAIL:", type(e).__name__, str(e)[:300])

print()

# ------------------------------------------------------- StringNormalizer+comb
print("=" * 70)
print("4. Full chain: lower -> split -> CategoryMapper")
print("=" * 70)
n1 = helper.make_node("StringNormalizer", ["x"], ["n"], case_change_action="LOWER", is_case_sensitive=0)
n2 = helper.make_node("StringSplit", ["n"], ["s", "_cnt"])
n3 = helper.make_node(
    "CategoryMapper",
    ["s"],
    ["y"],
    domain="ai.onnx.ml",
    cats_strings=["hello", "world", "foo"],
    cats_int64s=[10, 20, 30],
    default_int64=0,
)
g = helper.make_graph(
    [n1, n2, n3],
    "g",
    [helper.make_tensor_value_info("x", TensorProto.STRING, [1])],
    [
        helper.make_tensor_value_info("y", TensorProto.INT64, [1, None]),
        helper.make_tensor_value_info("_cnt", TensorProto.INT64, [1]),
    ],
)
m = helper.make_model(
    g, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("ai.onnx.ml", 3)]
)
m.ir_version = 10
try:
    out = run(m, {"x": np.array(["HELLO WORLD FOO zzz"], dtype=object)})
    print("  OK -> shape", out[0].shape, out[0].tolist())
except Exception as e:
    print("  FAIL:", type(e).__name__, str(e)[:300])