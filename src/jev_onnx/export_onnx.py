"""Export the PyTorch model to a single ONNX file whose inputs are raw strings.

The PyTorch model only sees integer ids, because that is all a transformer can
consume.  To make the *shipped* graph accept text we export the tensor half with
``torch.onnx.export`` and then graft a tokenisation subgraph onto its inputs:

    StringNormalizer(LOWER) -> StringSplit -> CategoryMapper -> Slice -> Pad

The same three operators do in PyTorch's tokenizer, which is why
``jev_onnx.tokenization`` mirrors them exactly.  ``CategoryMapper`` is the
dictionary: it carries the whole vocabulary as an attribute and sends anything
unrecognised to ``default_int64=0``, and because index 0 is also the padding
index, a single ``ids != 0`` in the encoder recovers a correct attention mask.

Graph inputs::

    content      string [B]   the text to classify
    head_text    string [N]   the question/prompt framing each candidate
    label_text   string [N]   the candidate labels themselves

Graph output::

    logits       float  [B, N]

The label side is built *inside* the graph as ``"<label_text> | <head_text>"``,
matching ``data.label_text`` used in training. Several decision heads therefore
cost one forward pass: the content encoding is computed once and scored against
every candidate. The schema is flattened before it reaches the graph, so the
candidate count ``N`` is simply the total number of labels across all heads.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from torch.export import Dim

from .model import JevDualEncoder
from .tokenization import LABEL_SEPARATOR, PAD_ID, Vocab

ONNX_OPSET = 20  # StringSplit / StringConcat were introduced in opset 20.
ML_OPSET = 3  # ai.onnx.ml CategoryMapper
IR_VERSION = 10  # highest onnxruntime 1.31 accepts
INT64_MAX = (1 << 63) - 1


def _const(name: str, values: list[int], dtype=TensorProto.INT64) -> helper.make_node:
    """A Constant node; onnxruntime folds these away at load time."""
    arr = np.array(values, dtype=np.int64 if dtype == TensorProto.INT64 else np.float32)
    return helper.make_node(
        "Constant",
        [],
        [name],
        name=f"const/{name}",
        value=numpy_helper.from_array(arr, name),
    )


def _string_to_ids_nodes(prefix: str, src: str, budget: int, vocab: Vocab, out: str) -> list:
    """Nodes turning a ``[K]`` string tensor into a ``[K, budget]`` int64 tensor.

    ``budget`` fixes the sequence width so the encoder sees a static shape;
    longer inputs are truncated and shorter ones are padded with index 0, which
    the encoder treats as "ignore". The result is named ``out`` so it can be
    wired straight into an already-exported consumer.
    """
    lo = f"{prefix}/lower"
    tok = f"{prefix}/tokens"
    cnt = f"{prefix}/counts"
    ids = out
    nodes = [
        helper.make_node(
            "StringNormalizer",
            [src],
            [lo],
            name=f"{prefix}/normalize",
            case_change_action="LOWER",
            is_case_sensitive=0,
        ),
        helper.make_node("StringSplit", [lo], [tok, cnt], name=f"{prefix}/split"),
    ]

    # onnxruntime does not guarantee StringSplit's output rank (the same model
    # can hand back [K, M] or [K*M]). Pin it from the per-row counts.
    shape, maxcnt, reshape, tok2 = (
        f"{prefix}/tok_shape",
        f"{prefix}/tok_max",
        f"{prefix}/tok_reshape",
        f"{prefix}/tokens2d",
    )
    nodes += [
        helper.make_node("Shape", [cnt], [shape], name=f"{prefix}/shape"),
        helper.make_node("ReduceMax", [cnt], [maxcnt], name=f"{prefix}/max", keepdims=1),
        helper.make_node("Concat", [shape, maxcnt], [reshape], name=f"{prefix}/concat_shape", axis=0),
        helper.make_node("Reshape", [tok, reshape], [tok2], name=f"{prefix}/pin_rank"),
    ]

    raw = f"{prefix}/ids_raw"
    nodes.append(
        helper.make_node(
            "CategoryMapper",
            [tok2],
            [raw],
            name=f"{prefix}/lookup",
            domain="ai.onnx.ml",
            cats_strings=vocab.itos,
            cats_int64s=list(range(len(vocab))),
            default_int64=PAD_ID,
        )
    )

    # Truncate to the budget. Slice clamps `ends`, so a short row is unaffected.
    cut = f"{prefix}/ids_cut"
    starts, ends, axes = f"{prefix}/starts", f"{prefix}/ends", f"{prefix}/axes"
    nodes += [
        _const(starts, [0, 0]),
        _const(ends, [INT64_MAX, budget]),
        _const(axes, [0, 1]),
        helper.make_node(
            "Slice", [raw, starts, ends, axes], [cut], name=f"{prefix}/truncate"
        ),
    ]

    # Pad up to the budget: pads = [begin_0, begin_1, end_0, end_1] = [0,0,0,pad]
    cut_shape, idx, cur = f"{prefix}/cut_shape", f"{prefix}/dim1", f"{prefix}/cur"
    budget_c, pad_amt = f"{prefix}/budget", f"{prefix}/pad_amt"
    pads_prefix, pads = f"{prefix}/pads_prefix", f"{prefix}/pads"
    nodes += [
        helper.make_node("Shape", [cut], [cut_shape], name=f"{prefix}/cut_shape"),
        _const(idx, [1]),
        helper.make_node("Gather", [cut_shape, idx], [cur], name=f"{prefix}/cur_len", axis=0),
        _const(budget_c, [budget]),
        helper.make_node("Sub", [budget_c, cur], [pad_amt], name=f"{prefix}/pad_amount"),
        _const(pads_prefix, [0, 0, 0]),
        helper.make_node("Concat", [pads_prefix, pad_amt], [pads], name=f"{prefix}/pads", axis=0),
        helper.make_node("Pad", [cut, pads], [ids], name=f"{prefix}/pad"),
    ]
    return nodes


def _export_core(
    model: JevDualEncoder,
    max_len: int,
    max_label_len: int,
    dynamo: bool = True,
) -> onnx.ModelProto:
    """Export just the tensor half: (content_ids, label_ids) -> logits."""
    model.eval()
    n = max(2, model.vocab_size)
    dummy_content = torch.randint(0, n, (2, max_len), dtype=torch.long)
    dummy_label = torch.randint(0, n, (3, max_label_len), dtype=torch.long)

    # Batch and candidate count stay dynamic; the sequence width is fixed so the
    # encoder can use static shapes.
    batch_dim = Dim("B")
    pair_dim = Dim("N")

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "core.onnx"
        torch.onnx.export(
            model,
            (dummy_content, dummy_label),
            str(raw_path),
            input_names=["content_ids", "label_ids"],
            output_names=["logits"],
            # Keys are the forward() argument names, not the ONNX input names.
            dynamic_shapes={
                "text_ids": {0: batch_dim},
                "label_ids": {0: pair_dim},
            },
            opset_version=ONNX_OPSET,
            dynamo=dynamo,
        )
        core = onnx.load(raw_path)

    core.opset_import[0].version = ONNX_OPSET
    return core


def _graft(core: onnx.ModelProto, vocab: Vocab, max_len: int, max_label_len: int) -> onnx.ModelProto:
    """Replace the two id inputs with string inputs plus a tokenisation subgraph."""
    text_nodes: list = []
    value_infos = []

    # --- content ------------------------------------------------------------
    content_ids = "content_ids"
    label_ids = "label_ids"
    value_infos.append(
        helper.make_tensor_value_info("content", TensorProto.STRING, ["B"])
    )
    text_nodes += _string_to_ids_nodes("content", "content", max_len, vocab, content_ids)

    # --- label side ---------------------------------------------------------
    # Each candidate becomes "<label> | <its own question>", assembled in the
    # graph so the caller supplies labels and questions separately and keeps full
    # control over the separator. StringConcat has no `sep` attribute in opset
    # 20, so the separator is broadcast in explicitly as a shape-[1] tensor.
    sep = "label/sep"
    labelled = "label/with_sep"
    pair_text = "pair_text"
    text_nodes += [
        helper.make_node(
            "Constant",
            [],
            [sep],
            name="const/label_sep",
            value=numpy_helper.from_array(
                np.array([LABEL_SEPARATOR], dtype=object), "label_sep"
            ),
        ),
        helper.make_node(
            "StringConcat", ["label_text", sep], [labelled], name="concat/label_sep"
        ),
        helper.make_node(
            "StringConcat",
            [labelled, "head_text"],
            [pair_text],
            name="concat/label_and_head",
        ),
    ]
    value_infos.append(
        helper.make_tensor_value_info("head_text", TensorProto.STRING, ["N"])
    )
    value_infos.append(
        helper.make_tensor_value_info("label_text", TensorProto.STRING, ["N"])
    )
    text_nodes += _string_to_ids_nodes("label", pair_text, max_label_len, vocab, label_ids)

    # Swap the id inputs for string inputs. The subgraph is written to emit the
    # exact tensor names the core graph already consumes, so nothing downstream
    # needs renaming.
    keep = [
        vi
        for vi in core.graph.input
        if vi.name not in ("content_ids", "label_ids")
    ]
    del core.graph.input[:]
    core.graph.input.extend(keep + value_infos)

    # Prepend: a node's producer must precede its consumers in topological order.
    original_nodes = list(core.graph.node)
    del core.graph.node[:]
    core.graph.node.extend(text_nodes)
    core.graph.node.extend(original_nodes)

    # Declare the two StringConcat outputs so shape inference stays quiet.
    # (The int64 tensors produced by StringSplit/Constant already carry types.)
    core.graph.value_info.extend(
        [
            helper.make_tensor_value_info(labelled, TensorProto.STRING, ["N"]),
            helper.make_tensor_value_info(pair_text, TensorProto.STRING, ["N"]),
        ]
    )

    del core.opset_import[:]
    core.opset_import.extend(
        [
            helper.make_opsetid("", ONNX_OPSET),
            helper.make_opsetid("ai.onnx.ml", ML_OPSET),
        ]
    )
    core.ir_version = IR_VERSION
    core.producer_name = "jev-onnx"
    core.doc_string = (
        "Pure-string JEV decision classifier. Inputs: content, head_text, "
        "head_of_pair, label_text. Output: logits [B, N]."
    )
    return core


def export_to_onnx(
    model: JevDualEncoder,
    vocab: Vocab,
    path: str | Path,
    max_len: int = 448,
    max_label_len: int = 96,
    dynamo: bool = True,
    validate: bool = True,
) -> Path:
    """Write a self-contained ONNX model that takes text in and scores text out."""
    core = _export_core(model, max_len, max_label_len, dynamo=dynamo)
    full = _graft(core, vocab, max_len, max_label_len)

    if validate:
        onnx.checker.check_model(full)
        inferred = onnx.shape_inference.infer_shapes(full)
        if inferred is not None:
            full = inferred

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(full, path)
    return path


def load_vocab_for(onnx_path: str | Path) -> Vocab:
    """Recover the vocabulary from the graph's CategoryMapper attribute.

    Useful when the .onnx file is the only artefact you have: the tokeniser is
    fully described by the graph, so no side-car dictionary is required.
    """
    model = onnx.load(onnx_path)
    for node in model.graph.node:
        if node.op_type != "CategoryMapper":
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if "cats_strings" not in attrs:
            continue  # the int64->string direction; keep looking
        # Protobuf hands string attributes back as bytes.
        cats = [
            c.decode("utf-8") if isinstance(c, bytes) else c for c in attrs["cats_strings"]
        ]
        order = sorted(range(len(attrs["cats_int64s"])), key=lambda i: attrs["cats_int64s"][i])
        return Vocab([cats[i] for i in order])
    raise ValueError(f"no string->int CategoryMapper found in {onnx_path}")