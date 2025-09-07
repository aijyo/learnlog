# tvm_optimize_onnx_save.py
# -*- coding: utf-8 -*-
"""
Load an ONNX model, import it into TVM Relax, apply common optimization passes,
then save the optimized IR (as .py script and .json) and the detached params (.npz).

Notes:
- This script assumes your environment already handles any custom ops (e.g., GRU)
  via your own registered converters. We do NOT register any custom ops here.
- All comments are in English as requested.
"""

import os
import argparse
import onnx
import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.frontend import detach_params


def optimize_relax(mod: tvm.IRModule) -> tvm.IRModule:
    """
    Apply a reasonable Relax optimization pipeline.
    You can tweak the pass list based on your TVM version/needs.
    """
    passes = [
        relax.transform.NormalizeArgs(),
        relax.transform.FoldConstant(),
        relax.transform.SimplifyExpr(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),           # operator fusion
        relax.transform.ToNonDataflow(),     # finalize dataflow blocks
        relax.transform.DeadCodeElimination(),
        relax.transform.FoldConstant(),      # fold again after fusion/DCE
    ]
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        return seq(mod)


def save_text_ir(mod: tvm.IRModule, path: str):
    """
    Save Relax module as a .py-like textual script (mod.script()).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(mod.script())


def save_json_ir(mod: tvm.IRModule, path: str):
    """
    Save Relax module as JSON using tvm.ir.save_json (round-trippable).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tvm.ir.save_json(mod))


def save_params_npz(params: dict, path: str):
    """
    Save detached params as a .npz file. Keys are param names; values are numpy arrays.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ensure values are numpy arrays on CPU
    np_params = {}
    for k, v in params.items():
        # v is usually tvm.nd.NDArray; convert to numpy
        np_params[k] = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    np.savez(path, **np_params)


def main():
    parser = argparse.ArgumentParser(
        description="Load ONNX → Relax → Optimize → Save IR & Params"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=r"D:\learn\tvm\python\resource\DeepFilterNet3_onnx\export\erb_dec.onnx",
        help="Path to the ONNX model to import.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=r"./tvm_out",
        help="Output directory for saved IR and params.",
    )
    parser.add_argument(
        "--keep-params-in-input",
        action="store_true",
        help="Keep params as inputs during import (useful if your custom converters rely on it).",
    )
    parser.add_argument(
        "--save-unopt",
        action="store_true",
        help="Also save the unoptimized Relax module (before passes).",
    )
    args = parser.parse_args()

    onnx_path = args.onnx
    assert os.path.exists(onnx_path), f"ONNX file not found: {onnx_path}"
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load ONNX
    model = onnx.load(onnx_path)

    # 2) Import to Relax. If your custom op converters expect params as inputs,
    #    pass keep_params_in_input=True (you can detach later).
    mod = from_onnx(model, keep_params_in_input=args.keep_params_in_input)

    # 3) Detach parameters to get (IRModule without large constants inlined, params dict)
    mod, params = detach_params(mod)

    # 4) (Optional) Save the unoptimized IR
    if args.save_unopt:
        save_text_ir(mod, os.path.join(args.outdir, "model_unoptimized.py"))
        save_json_ir(mod, os.path.join(args.outdir, "model_unoptimized.json"))

    # 5) Optimize Relax
    mod_opt = optimize_relax(mod)

    # 6) Save optimized IR (text and JSON) and params (NPZ)
    save_text_ir(mod_opt, os.path.join(args.outdir, "model_optimized.py"))
    save_json_ir(mod_opt, os.path.join(args.outdir, "model_optimized.json"))
    save_params_npz(params, os.path.join(args.outdir, "params_optimized.npz"))

    # 7) Print a short summary
    print("=== TVM Relax Import/Optimize Done ===")
    print(f"ONNX:      {onnx_path}")
    print(f"Out dir:   {args.outdir}")
    if args.save_unopt:
        print("Saved:     model_unoptimized.py / model_unoptimized.json")
    print("Saved:     model_optimized.py / model_optimized.json / params_optimized.npz")


if __name__ == "__main__":
    main()
