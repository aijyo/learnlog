import os
import onnx
from onnx import shape_inference

# Model default dir
model_dir = r"D:\code\mycode\onnx_mlir_test\model_shim\3rd\deepfilter\lib"
models = ["enc.onnx", "df_dec.onnx", "erb_dec.onnx"]

def _replace_symbolic_dim_in_value_info(value_info, symbol_name: str, static_value: int) -> bool:
    """
    Replace dim_param == symbol_name with dim_value == static_value in a ValueInfoProto.
    Returns True if anything changed.
    """
    changed = False
    tt = value_info.type.tensor_type
    if not tt.HasField("shape"):
        return False

    for dim in tt.shape.dim:
        # Replace exact symbolic name
        if dim.HasField("dim_param") and dim.dim_param == symbol_name:
            dim.ClearField("dim_param")
            dim.dim_value = static_value
            changed = True

    return changed

def _replace_symbolic_dim_everywhere(model: onnx.ModelProto, symbol_name: str, static_value: int) -> int:
    """
    Replace dim_param == symbol_name with dim_value == static_value across:
      - graph.input
      - graph.output
      - graph.value_info (intermediate tensors)
    Returns number of ValueInfo entries that were changed (rough count).
    """
    g = model.graph
    changed_cnt = 0

    for vi in g.input:
        if _replace_symbolic_dim_in_value_info(vi, symbol_name, static_value):
            changed_cnt += 1

    for vi in g.output:
        if _replace_symbolic_dim_in_value_info(vi, symbol_name, static_value):
            changed_cnt += 1

    for vi in g.value_info:
        if _replace_symbolic_dim_in_value_info(vi, symbol_name, static_value):
            changed_cnt += 1

    return changed_cnt

def freeze_dynamic_shape(model_path: str, static_S: int = 5, symbol_name: str = "S") -> str:
    # Load model
    model = onnx.load(model_path)

    # 1) Replace symbol in all known places BEFORE inference
    _replace_symbolic_dim_everywhere(model, symbol_name, static_S)

    # 2) Run shape inference (may still keep symbolic for some ops)
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception as e:
        # If inference fails, still save the partially fixed model
        inferred = model
        print(f"[WARN] shape_inference failed for {os.path.basename(model_path)}: {e}")

    # 3) Replace again AFTER inference (important: inference may reintroduce/keep 'S')
    _replace_symbolic_dim_everywhere(inferred, symbol_name, static_S)

    # Build output path
    base = os.path.basename(model_path)
    out_name = base.replace(".onnx", f"_fixed.onnx")
    out_path = os.path.join(os.path.dirname(model_path), out_name)

    # Save
    onnx.save(inferred, out_path)
    print(f"Saved frozen model to: {out_path}")
    return out_path

def process_models(models, model_dir, static_S=5):
    for model_name in models:
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            print(f"Processing model: {model_name}")
            freeze_dynamic_shape(model_path, static_S=static_S, symbol_name="S")
        else:
            print(f"Model {model_name} not found in {model_dir}")

if __name__ == "__main__":
    process_models(models, model_dir, static_S=5)
