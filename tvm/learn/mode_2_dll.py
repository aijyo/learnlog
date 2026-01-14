# import tvm
# import numpy as np
# from tvm.relax.frontend import nn
# from tvm import relax

# class RelaxModel(nn.Module):
#     def __init__(self):
#         super(RelaxModel, self).__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(256, 10)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         return x
    

# mode_from_relax, params_from_relax = RelaxModel().export_tvm({"forward": {"x":nn.spec.Tensor((1, 784), "float32")}})
# mode_from_relax.show()

# # print(mode_from_relax.get_global_vars())

# # print(mode_from_relax["forward"])

# # (gv,) = mode_from_relax.get_global_vars()
# # assert gv == mode_from_relax["forward"]

# # print(gv)

# mod = mode_from_relax
# mod = relax.transform.LegalizeOps()(mod)
# # mod = relax.transform.FuseOps(fuse_opt_level=2)(mod)
# mod.show()

# mod = relax.get_pipeline("zero")(mod)

# mod.show()

# exec = tvm.compile(mod, target="llvm")
# dev = tvm.cpu()
# vm = relax.VirtualMachine(exec, dev)

# raw_data = np.random.rand(1, 784).astype("float32")
# data = tvm.runtime.tensor(raw_data, device=dev)

# cpu_out = vm["forward"](data, *params_from_relax).numpy()
# print(cpu_out)
import os
import tvm
import numpy as np
from tvm.relax.frontend import nn
from tvm import relax
from tvm.contrib import cc, utils, tar

os.environ["PATH"] = r"D:\code\gitcode\install-llvm\bin;" + os.environ["PATH"]

class RelaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

mod_from_relax, params_from_relax = RelaxModel().export_tvm(
    {"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)

mod = mod_from_relax
mod = relax.transform.LegalizeOps()(mod)
mod = relax.get_pipeline("zero")(mod)

# Pick entry
gvars = [gv.name_hint for gv in mod.get_global_vars()]
entry = "forward" if "forward" in gvars else ("main" if "main" in gvars else gvars[0])

target = tvm.target.Target("llvm")
ex = relax.build(mod, target=target)

dev = tvm.cpu()
vm = relax.VirtualMachine(ex, dev)

raw_data = np.random.rand(1, 784).astype("float32")
tvm_data = tvm.runtime.tensor(raw_data, device=dev)

params = [np.random.rand(*param.shape).astype("float32") for _, param in params_from_relax]
params = [tvm.runtime.tensor(param, device=dev) for param in params]
cpu_out = vm["forward"](tvm_data, *params).numpy()
print(cpu_out)

# import shutil, subprocess, os

# def where(x):
#     p = shutil.which(x)
#     print(f"{x} =", p)
#     if p:
#         try:
#             out = subprocess.check_output([p, "--version"], stderr=subprocess.STDOUT, text=True)
#             print(out.splitlines()[0])
#         except Exception as e:
#             print("  version check failed:", e)

# print("PATH head =", os.environ["PATH"].split(";")[0:3])
# where("clang")
# where("gcc")
# where("ld.lld")
# where("lld-link")

def fcompile_to_dll(output, objects, **kwargs):
    # Force MSVC-style target + lld, avoid gcc/mingw entirely
    options = [
        "-shared",
        "--target=x86_64-pc-windows-msvc",
        "-fuse-ld=lld",
    ]
    cc.create_shared(output, objects, options=options, cc="clang")
ex.mod.export_library("my_relax_model.dll", fcompile=fcompile_to_dll)


def export_mod_to_staticlib(mod, lib_name="mymod.lib", work_dir_name="tvm_build"):
    # Get directory of current python file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a stable working directory under the script directory
    work_dir = os.path.join(script_dir, work_dir_name)
    os.makedirs(work_dir, exist_ok=True)

    # 1) Export object files as a tarball
    obj_tar = os.path.join(work_dir, "mymod_objs.tar")
    mod.export_library(obj_tar, fcompile=tar.tar)

    # 2) Untar into a subdirectory (must exist before calling tar.untar)
    obj_dir = os.path.join(work_dir, "objs")
    os.makedirs(obj_dir, exist_ok=True)
    tar.untar(obj_tar, obj_dir)

    # 3) Collect object files
    objs = []
    for root, _, files in os.walk(obj_dir):
        for fn in files:
            if fn.endswith((".o", ".obj")):
                objs.append(os.path.join(root, fn))

    if not objs:
        raise RuntimeError(f"No object files found under: {obj_dir}")

    # 4) Create static library (.lib) using MSVC lib.exe
    out_lib = os.path.join(work_dir, lib_name)
    cc.create_staticlib(out_lib, objs, ar="lib")
    return out_lib

# export_mod_to_staticlib(ex.mod, lib_name="my_relax_model.lib")

# main.obj: your compiled main file
# mymod.lib: static lib generated from mod
# tvm runtime libs: depends on your TVM build (often you link against tvm_runtime)

# objects = ["main.obj"]
# options = [
#     "my_relax_model.lib",
#     # "tvm_runtime.lib",  # if you have it
#     # Add system libs if needed
# ]

# cc.create_executable("app.exe", objects=objects, options=options, cc=cc.get_cc())
