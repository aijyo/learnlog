
import numpy as np
import tvm
from tvm import relax as rx
from tvm import tir
from tvm.script import tir as T

# 1) Create BlockBuilder and symbolic shapes (must be int64)
bb = rx.BlockBuilder()
m = tir.Var("m", "int64")
n = tir.Var("n", "int64")

# -----------------------------
# Function 1 (Relax): add_func(x, y) = x + y
# -----------------------------
x = rx.Var("x", rx.TensorStructInfo([m, n], "float32"))
y = rx.Var("y", rx.TensorStructInfo([m, n], "float32"))
with bb.function("add_func", [x, y]):
    with bb.dataflow():
        lv0 = bb.emit(rx.op.add(x, y))
        gv0 = bb.emit_output(lv0)
    bb.emit_func_output(gv0)

# -----------------------------
# TIR PrimFunc: tir_add(A, B, C) = A + B (elementwise)
# -----------------------------
@T.prim_func
def tir_add(A: T.Buffer[(m, n), "float32"],
            B: T.Buffer[(m, n), "float32"],
            C: T.Buffer[(m, n), "float32"]):
    T.func_attr({"global_symbol": "tir_add", "tir.noalias": True})
    for i, j in T.grid(m, n):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + B[vi, vj]

bb.add_func(tir_add, "tir_add")  # register PrimFunc into the module

# -----------------------------
# Function 2 (Relax): main(a, b) = (a+b)*b + (a+b)
#   - Call Relax function via rx.Call(bb.get().get_global_var("add_func"), ...)
#   - Call TIR function via rx.call_tir(GlobalVar("tir_add"), ...)
# -----------------------------
a = rx.Var("a", rx.TensorStructInfo([m, n], "float32"))
b = rx.Var("b", rx.TensorStructInfo([m, n], "float32"))

with bb.function("main", [a, b]):
    with bb.dataflow():
        # IMPORTANT: fetch GlobalVar with struct_info from the current IRModule
        add_gv = bb.get().get_global_var("add_func")
        # Some versions require sinfo_args; it is safe to pass it explicitly.
        lv0 = bb.emit(rx.Call(
            add_gv, [a, b],
            # attrs=None,
            sinfo_args=[rx.TensorStructInfo([m, n], "float32")]
        ))  # (a + b) via Relax function

        tir_gv = bb.get().get_global_var("tir_add")
        lv1 = bb.emit(
            rx.call_tir(
                tir_gv,
                (a, b),
                out_sinfo=rx.TensorStructInfo([m, n], "float32"),
            )
        )  # (a + b) via TIR

        lv2 = bb.emit(rx.op.multiply(lv0, b))
        lv3 = bb.emit(rx.op.add(lv2, lv1))
        gv0 = bb.emit_output(lv3)
    bb.emit_func_output(gv0)

# 2) Get IRModule
mod = bb.get()
print("=== IRModule ===")
print(mod)

# 3) Build & run with VM
target = "llvm"
exe = rx.build(mod, target=target)
dev = tvm.cpu()
vm = rx.VirtualMachine(exe, dev)

# Prepare input data
M, N = 2, 3
a_np = np.arange(M * N, dtype="float32").reshape(M, N)
b_np = np.ones((M, N), dtype="float32") * 2
a_nd = tvm.nd.array(a_np, dev)
b_nd = tvm.nd.array(b_np, dev)

# Run main
out_nd = vm["main"](a_nd, b_nd)
out_np = out_nd.numpy()

print("=== Input a ===")
print(a_np)
print("=== Input b ===")
print(b_np)
print("=== Output ( (a+b)*b + (a+b) ) ===")
print(out_np)

# Optional correctness check
ref = (a_np + b_np) * b_np + (a_np + b_np)
print("Allclose:", np.allclose(out_np, ref))
