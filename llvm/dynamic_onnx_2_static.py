import onnx
import onnxsim
import json
import os
import sys
from onnx import shape_inference

# Helper functions
def write_info(msg):
    print(f"[INFO] {msg}")

def write_warn(msg):
    print(f"[WARN] {msg}")

def write_err(msg):
    print(f"[ERR ] {msg}")

def is_batch_like(name: str) -> bool:
    if not name: return False
    name_l = name.lower()
    return name_l in ("b", "batch", "n", "batch_size")

def analyze_model(model_path, batch, S, SName, otherDyn):
    model = onnx.load(model_path)
    report = {"model": model_path, "inputs": [], "overwrite_shapes": []}
    
    for inp in model.graph.input:
        dims_desc = []
        dims_num = []
        for i, d in enumerate(inp.type.tensor_type.shape.dim):
            if d.HasField("dim_value"):
                dims_desc.append(str(d.dim_value))
                dims_num.append(d.dim_value)
            elif d.HasField("dim_param"):
                name = d.dim_param
                dims_desc.append(f"{name}*")
                if i == 0 and is_batch_like(name):
                    dims_num.append(batch)
                elif name == SName:
                    dims_num.append(S)
                else:
                    dims_num.append(otherDyn)
            else:
                dims_desc.append("?*")
                dims_num.append(batch if i == 0 else otherDyn)
        
        report["inputs"].append({"name": inp.name, "desc": dims_desc})
        report["overwrite_shapes"].append(f"{inp.name}:{','.join(map(str, dims_num))}")
    
    return report

def freeze_model(model_path, out_path, batch, S, SName, otherDyn):
    model = onnx.load(model_path)
    for g in [model.graph]:
        for vi in list(g.input) + list(g.output) + list(g.value_info):
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                for i, d in enumerate(tt.shape.dim):
                    if d.HasField("dim_param"):
                        nm = d.dim_param
                        if i == 0 and is_batch_like(nm):
                            d.ClearField("dim_param"); d.dim_value = batch
                        elif nm == SName:
                            d.ClearField("dim_param"); d.dim_value = S
                        else:
                            d.ClearField("dim_param"); d.dim_value = otherDyn
    onnx.save(model, out_path)
    try:
        model = shape_inference.infer_shapes(model, check_type=False, strict_mode=False, data_prop=False)
    except Exception as e:
        write_warn(f"Shape inference failed: {e}")
    onnx.save(model, out_path)

def try_onnxsim(model_path, out_path, perform_optimization=True):
    try:
        # 进行简化，使用正确的参数
        model, _ = onnxsim.simplify(model_path, perform_optimization=perform_optimization)
        onnx.save(model, out_path)
        write_info(f"Model simplified and saved to {out_path}")
        return 0
    except Exception as e:
        write_warn(f"onnxsim failed: {e}")
        return 1

# Main function to process ONNX models
def process_onnx_models(models, default_dir, S=120, SName="S", batch=1, ReplaceOtherDynamicsWith=1, skip_sim=False, dry_run=False):
    for model_path in models:
        model_path = os.path.join(default_dir, model_path)  # Join the directory with model filename
        
        if not os.path.exists(model_path):
            write_warn(f"Model not found: {model_path}, skipping.")
            continue
        
        write_info(f"Analyzing model: {model_path}")
        report = analyze_model(model_path, batch, S, SName, ReplaceOtherDynamicsWith)
        
        for inp in report["inputs"]:
            dims = ", ".join(inp["desc"])
            write_info(f"Input {inp['name']} Shape: [{dims}]")
        
        model_dir = os.path.dirname(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        frozen_path = os.path.join(model_dir, f"{model_name}_frozen.onnx")
        final_path = os.path.join(model_dir, f"{model_name}_final.onnx")
        
        write_info(f"Freezing shapes and saving to: {frozen_path}")
        if not dry_run:
            freeze_model(model_path, frozen_path, batch, S, SName, ReplaceOtherDynamicsWith)
        else:
            write_warn(f"Dry run enabled. Skipping freezing for {model_path}")
        
        if skip_sim:
            write_warn("SkipSim enabled. Keeping frozen model as final.")
            os.rename(frozen_path, final_path)
            write_info(f"Final model saved to: {final_path}")
        else:
            write_info("Running onnxsim to simplify the model.")
            result = try_onnxsim(frozen_path, final_path)
            if result != 0:
                write_warn(f"Keeping frozen model due to onnxsim failure: {final_path}")
                os.rename(frozen_path, final_path)
            else:
                write_info(f"Final model saved to: {final_path}")
        
        if os.path.exists(frozen_path):
            os.remove(frozen_path)

# Example Usage
if __name__ == "__main__":
    default_dir = r"D:\code\mycode\python\learn\resource\deepfilter\export"  # Default directory
    models = ["enc.onnx", "df_dec.onnx", "erb_dec.onnx"]  # List of models
    process_onnx_models(models, default_dir, S=1, SName="S", batch=1, ReplaceOtherDynamicsWith=1, skip_sim=False, dry_run=False)
