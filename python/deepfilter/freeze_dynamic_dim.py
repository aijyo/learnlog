# -*- coding: utf-8 -*-
import argparse
import copy
from typing import Dict, List, Optional, Tuple

import onnx
from onnx import helper, shape_inference, TensorProto


def _is_value_info_tensor(vi: onnx.ValueInfoProto) -> bool:
    return vi.type.HasField("tensor_type")


def _get_dims(vi: onnx.ValueInfoProto) -> List[onnx.TensorShapeProto.Dimension]:
    # English: Return dimension protos for a ValueInfo (must be tensor).
    return list(vi.type.tensor_type.shape.dim)


def _dim_to_str(d: onnx.TensorShapeProto.Dimension) -> str:
    # English: Pretty-print a single dim.
    if d.HasField("dim_value"):
        return str(d.dim_value)
    if d.HasField("dim_param"):
        return d.dim_param
    return "?"


def _shape_to_str(vi: onnx.ValueInfoProto) -> str:
    if not _is_value_info_tensor(vi):
        return "<non-tensor>"
    dims = _get_dims(vi)
    return "[" + ",".join(_dim_to_str(d) for d in dims) + "]"


def _set_dim_value(dim: onnx.TensorShapeProto.Dimension, value: int):
    # English: Set a dimension to a concrete integer value.
    dim.ClearField("dim_param")
    dim.dim_value = int(value)


def freeze_input_shapes(
    model: onnx.ModelProto,
    *,
    dim_param_to_value: Optional[Dict[str, int]] = None,
    axis_to_value: Optional[Dict[int, int]] = None,
    freeze_all_unknown_to: Optional[int] = None,
    only_these_inputs: Optional[List[str]] = None,
) -> Tuple[onnx.ModelProto, List[str]]:
    """
    Freeze dynamic dims for graph inputs.

    Args:
        model: ONNX model.
        dim_param_to_value: map from symbolic dim name (e.g. "S") to value (e.g. 1).
        axis_to_value: map from axis index to value, applied to ALL selected inputs.
        freeze_all_unknown_to: if set, any unknown dim (no dim_value & no dim_param)
                               OR any dim_param not in dim_param_to_value will be set to this value.
        only_these_inputs: if provided, only modify these input names.

    Returns:
        (new_model, logs)
    """
    dim_param_to_value = dim_param_to_value or {}
    axis_to_value = axis_to_value or {}

    new_model = copy.deepcopy(model)
    g = new_model.graph

    # English: Build a set of initializer names (they can appear in graph.input, but are not real runtime inputs).
    initializer_names = {init.name for init in g.initializer}

    logs: List[str] = []

    for vi in g.input:
        name = vi.name

        # English: Skip initializers listed in graph.input (common in some exports).
        if name in initializer_names:
            continue

        if only_these_inputs and name not in set(only_these_inputs):
            continue

        if not _is_value_info_tensor(vi):
            continue

        dims = _get_dims(vi)
        before = _shape_to_str(vi)

        for ax, dim in enumerate(dims):
            # Case 1: axis override
            if ax in axis_to_value:
                _set_dim_value(dim, axis_to_value[ax])
                continue

            # Case 2: symbolic dim name match
            if dim.HasField("dim_param"):
                dp = dim.dim_param
                if dp in dim_param_to_value:
                    _set_dim_value(dim, dim_param_to_value[dp])
                elif freeze_all_unknown_to is not None:
                    # English: If requested, force any remaining dim_param to a value.
                    _set_dim_value(dim, freeze_all_unknown_to)
                continue

            # Case 3: completely unknown dim
            if (not dim.HasField("dim_value")) and (not dim.HasField("dim_param")):
                if freeze_all_unknown_to is not None:
                    _set_dim_value(dim, freeze_all_unknown_to)

            # Case 4: dim_value already concrete -> keep as is

        after = _shape_to_str(vi)
        if before != after:
            logs.append(f"input {name}: {before} -> {after}")

    return new_model, logs


src_onnx = r"D:\code\mycode\onnx_mlir_test\model_shim\3rd\deepfilter\lib\enc_e0.onnx"    # 原始onnx
dst_onnx = r"D:\code\mycode\onnx_mlir_test\model_shim\3rd\deepfilter\lib\enc_e0_fixed.onnx"    # 原始onnx
def main():

    ap = argparse.ArgumentParser()

    # Default paths
    ap.add_argument(
        "--in",
        dest="in_path",
        default=src_onnx,
        help="Input ONNX model path (default: enc.onnx)",
    )

    ap.add_argument(
        "--out",
        dest="out_path",
        default=dst_onnx,
        help="Output ONNX model path (default: enc_S1.onnx)",
    )

    # Freeze symbolic dims
    ap.add_argument(
        "--dim",
        action="append",
        default=[],
        help="Freeze symbolic dim, e.g. --dim S=1 (can be repeated)",
    )

    # Freeze by axis
    ap.add_argument(
        "--axis",
        action="append",
        default=[],
        help="Freeze by axis, e.g. --axis 2=1 (0-based)",
    )

    # Freeze remaining unknown dims
    ap.add_argument(
        "--freeze-all-unknown-to",
        type=int,
        default=None,
        help="Freeze any remaining unknown dims to this value",
    )

    # Limit to specific inputs
    ap.add_argument(
        "--only-input",
        action="append",
        default=[],
        help="Only modify this input name (can be repeated)",
    )

    # Default: ON
    ap.add_argument(
        "--infer-shapes",
        action="store_true",
        default=True,
        help="Run shape inference after freezing inputs (default: ON)",
    )

    args = ap.parse_args()

    model = onnx.load(args.in_path)

    dim_param_to_value: Dict[str, int] = {}
    for s in args.dim:
        # English: parse "S=1"
        k, v = s.split("=", 1)
        dim_param_to_value[k.strip()] = int(v.strip())

    # Default: S = 1
    if not dim_param_to_value:
        dim_param_to_value["S"] = 1

    axis_to_value: Dict[int, int] = {}
    for s in args.axis:
        # English: parse "2=1"
        k, v = s.split("=", 1)
        axis_to_value[int(k.strip())] = int(v.strip())

    only_inputs = args.only_input if args.only_input else None

    new_model, logs = freeze_input_shapes(
        model,
        dim_param_to_value=dim_param_to_value,
        axis_to_value=axis_to_value,
        freeze_all_unknown_to=args.freeze_all_unknown_to,
        only_these_inputs=only_inputs,
    )

    print("=== Input shapes (before) ===")
    init_names = {i.name for i in model.graph.initializer}
    for vi in model.graph.input:
        if vi.name in init_names:
            continue
        if not _is_value_info_tensor(vi):
            continue
        print(f"{vi.name}: {_shape_to_str(vi)}")

    print("\n=== Changes ===")
    if logs:
        for line in logs:
            print(line)
    else:
        print("(no changes)")

    if args.infer_shapes:
        # English: shape inference can fail on some models; keep it best-effort.
        try:
            new_model = shape_inference.infer_shapes(new_model)
            print("\nShape inference: OK")
        except Exception as e:
            print("\nShape inference: FAILED:", str(e))

    # English: keep external data if present
    onnx.save_model(new_model, args.out_path, save_as_external_data=True, all_tensors_to_one_file=True)

    print("\n=== Input shapes (after) ===")
    init_names2 = {i.name for i in new_model.graph.initializer}
    for vi in new_model.graph.input:
        if vi.name in init_names2:
            continue
        if not _is_value_info_tensor(vi):
            continue
        print(f"{vi.name}: {_shape_to_str(vi)}")

    print(f"\nSaved: {args.out_path}")


if __name__ == "__main__":
    main()
