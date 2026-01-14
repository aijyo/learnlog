# # End-to-End Optimize Model
# # =============================================================
# import os
# import numpy as np
# import torch
# from torch.export import export
# from torchvision.models.resnet import ResNet18_Weights, resnet18
# import tvm
# from tvm import relax
# from tvm.relax.frontend.torch import from_exported_program

# torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

# example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

# IS_IN_CI = os.environ.get("CI") == "true"

# if not IS_IN_CI:
#     # Convert the model to IRModule
#     with torch.no_grad():
#         exported_program = export(
#             torch_model,
#             example_args
#         )
#         mod = from_exported_program(exported_program, keep_params_as_input=True)

#     mod,params = relax.frontend.detach_params(mod)
#     mod.show()

# resnet18_onnx_to_tvm.py
# English comments only in code as requested.

import os
import sys
import time
import numpy as np

def ensure_import(pkg_name: str, install_hint: str):
    try:
        __import__(pkg_name)
        return True
    except Exception as e:
        print(f"[ERROR] Cannot import '{pkg_name}': {e}")
        print(f"[HINT] Please install dependencies: {install_hint}")
        return False

def main():
    # ---- Dependency checks ----
    ok = True
    ok &= ensure_import("torch", "pip install torch torchvision")
    ok &= ensure_import("torchvision", "pip install torchvision")
    ok &= ensure_import("onnx", "pip install onnx")
    ok &= ensure_import("tvm", "Please install TVM (your local build) and ensure 'import tvm' works.")
    if not ok:
        sys.exit(1)

    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    import onnx
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    # ---- Config ----
    onnx_path = os.path.abspath("resnet18.onnx")
    np.random.seed(0)
    torch.manual_seed(0)

    # ---- Build PyTorch model ----
    torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

    # Use a fixed input for reproducibility
    x_torch = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # ---- Export ONNX ----
    # English comments:
    # - opset_version 17 is generally good for ResNet18
    # - do_constant_folding=True enables basic folding at export time
    print(f"[INFO] Exporting ONNX to: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            torch_model,
            x_torch,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )

    # Optional: check ONNX
    print("[INFO] Loading and checking ONNX model...")
    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
        print("[INFO] ONNX checker: OK")
    except Exception as e:
        print(f"[WARN] ONNX checker failed (may still work): {e}")

    # ---- Convert ONNX -> Relax ----
    print("[INFO] Converting ONNX -> TVM Relax IRModule...")
    mod = from_onnx(onnx_model)
    # Show IR (optional)
    # mod.show()

    # ---- Prepare input ----
    x_np = x_torch.detach().cpu().numpy().astype("float32")

    # ---- Compile with TVM ----
    # English comments:
    # - Choose target="llvm" for CPU on Windows
    # - If your TVM supports CUDA, you can change target/device accordingly
    target = "llvm"
    dev = tvm.cpu()

    print(f"[INFO] Running Relax pipeline and compiling. target={target}")
    mod = relax.get_pipeline("zero")(mod)

    # TVM compile (build Relax VM)
    # Note: This step may fail if your Windows toolchain (clang/linker) isn't set up.
    try:
        ex = tvm.compile(mod, target=target)
    except Exception as e:
        print("\n[ERROR] tvm.compile failed.")
        print("Common Windows causes:")
        print("  - clang/linker not runnable or not found")
        print("  - MSVC/LLVM toolchain mismatch")
        print("  - TVM built with a different compiler toolchain than runtime expects")
        print("\nException:")
        raise

    vm = relax.VirtualMachine(ex, dev)

    # ---- Run TVM ----
    # English comments:
    # - Most ONNX->Relax imports name the main function "main"
    # - If it differs, you can inspect via: print(vm.module.get_global_vars())
    print("[INFO] Running TVM VM...")
    t0 = time.time()
    tvm_data = tvm.runtime.tensor(x_np, device=dev)
    tvm_out = vm["main"](tvm_data)
    t1 = time.time()

    # Convert output to numpy
    tvm_out_np = tvm_out.numpy()

    # ---- Run PyTorch for reference ----
    print("[INFO] Running PyTorch for reference...")
    with torch.no_grad():
        pt_out = torch_model(x_torch).detach().cpu().numpy()

    # ---- Compare ----
    max_abs_diff = np.max(np.abs(pt_out - tvm_out_np))
    mean_abs_diff = np.mean(np.abs(pt_out - tvm_out_np))

    # Top-1 compare
    pt_top1 = int(np.argmax(pt_out, axis=1)[0])
    tvm_top1 = int(np.argmax(tvm_out_np, axis=1)[0])

    print("\n========== RESULT ==========")
    print(f"TVM output shape: {tvm_out_np.shape}")
    print(f"PyTorch output shape: {pt_out.shape}")
    print(f"Top1(PyTorch) = {pt_top1}, Top1(TVM) = {tvm_top1}")
    print(f"Max abs diff  = {max_abs_diff:.6g}")
    print(f"Mean abs diff = {mean_abs_diff:.6g}")
    print(f"TVM VM time   = {(t1 - t0)*1000:.2f} ms")
    print("============================\n")

if __name__ == "__main__":
    main()
