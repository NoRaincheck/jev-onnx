"""Characterize the ambiguity in ONNX StringSplit's output rank.

We saw y come back as (1,2) in one construction and (2,) in another. If the
rank is not guaranteed, the graph must normalise it with Reshape rather than
trusting the declared output shape.
"""

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper


def build(in_shape, out_rank, second_session=False, opt=None):
    nodes = [
        helper.make_node(
            "StringNormalizer", ["x"], ["lo"], case_change_action="LOWER", is_case_sensitive=0
        ),
        helper.make_node("StringSplit", ["lo"], ["y", "cnt"]),
    ]
    g = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.STRING, in_shape)],
        [
            helper.make_tensor_value_info("y", TensorProto.STRING, [None] * out_rank),
            helper.make_tensor_value_info("cnt", TensorProto.INT64, [None]),
        ],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
    m.ir_version = 10
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if opt is not None:
        so.graph_optimization_level = opt
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])


BATCHES = {
    "1 row": ["hello world"],
    "3 rows": ["a b", "c d e f", "g"],
}

print("=" * 96)
print("A. declared input [None], declared output rank 2, ONE session")
print("=" * 96)
s = build([None], 2)
for name, b in BATCHES.items():
    y, c = s.run(None, {"x": np.array(b, dtype=object)})
    print(f"  {name:8s} y.shape={str(y.shape):10s} cnt={c.tolist()}  y={y.tolist()}")

print()
print("=" * 96)
print("B. same, but a second session object created first (mimics _probe_split)")
print("=" * 96)
_ = build([None], 2, opt=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
s2 = build([None], 2)
for name, b in BATCHES.items():
    y, c = s2.run(None, {"x": np.array(b, dtype=object)})
    print(f"  {name:8s} y.shape={str(y.shape):10s} cnt={c.tolist()}  y={y.tolist()}")

print()
print("=" * 96)
print("C. repeat A five times -- deterministic or flaky?")
print("=" * 96)
for i in range(5):
    s3 = build([None], 2)
    y, c = s3.run(None, {"x": np.array(["hello world"], dtype=object)})
    print(f"  trial {i}: y.shape={y.shape}")

print()
print("=" * 96)
print("D. does Reshape(C, max(cnt)) normalise both shapes?  <-- the fix")
print("=" * 96)
nodes = [
    helper.make_node(
        "StringNormalizer", ["x"], ["lo"], case_change_action="LOWER", is_case_sensitive=0
    ),
    helper.make_node("StringSplit", ["lo"], ["y", "cnt"]),
    helper.make_node("Shape", ["cnt"], ["cs"]),
    helper.make_node("ReduceMax", ["cnt"], ["mx"], keepdims=1),
    helper.make_node("Concat", ["cs", "mx"], ["target"], axis=0),
    helper.make_node("Reshape", ["y", "target"], ["y2"]),
]
g = helper.make_graph(
    nodes,
    "g",
    [helper.make_tensor_value_info("x", TensorProto.STRING, [None])],
    [
        helper.make_tensor_value_info("y2", TensorProto.STRING, [None, None]),
        helper.make_tensor_value_info("cnt", TensorProto.INT64, [None]),
    ],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 20)])
m.ir_version = 10
so = ort.SessionOptions()
so.log_severity_level = 3
rs = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
ok = True
for name, b in BATCHES.items():
    y2, c = rs.run(None, {"x": np.array(b, dtype=object)})
    print(f"  {name:8s} y2.shape={str(y2.shape):10s} cnt={c.tolist()}")
    print(f"           y2={y2.tolist()}")
print()
print("  including nasty edge cases:")
for t in ["", "   ", "one", "a b\nc d", "  pad  "]:
    y2, c = rs.run(None, {"x": np.array([t], dtype=object)})
    print(f"    {t!r:16s} y2.shape={str(y2.shape):8s} cnt={c.tolist()} y2={y2.tolist()}")