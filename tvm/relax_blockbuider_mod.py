# # import tvm
# # from tvm.script import tir as T
# # import numpy as np

# # # --------------------------
# # # Config: fix B,I,H; dynamic S via function argument
# # # --------------------------
# # B, I, H = 2, 3, 4  # batch, in_dim, hidden_dim

# # @T.prim_func
# # def gru_dyn(x: T.handle, W: T.handle, U: T.handle, b: T.handle,
# #             h0: T.handle, y: T.handle, yh: T.handle, S: T.int32) -> None:
# #     # Buffers with dynamic S
# #     X  = T.match_buffer(x,  (S, B, I), "float32")
# #     Wg = T.match_buffer(W,  (3*H, I), "float32")      # [z, r, n] x input
# #     Ug = T.match_buffer(U,  (3*H, H), "float32")      # [z, r, n] x hidden
# #     bg = T.match_buffer(b,  (3*H,),   "float32")      # [z, r, n]
# #     H0 = T.match_buffer(h0, (B, H),   "float32")
# #     Y  = T.match_buffer(y,  (S, B, H), "float32")
# #     Yh = T.match_buffer(yh, (B, H),    "float32")

# #     # State and temporaries
# #     Ht     = T.alloc_buffer((B, H), "float32")
# #     gates  = T.alloc_buffer((B, 3*H), "float32")  # concat [z|r|n] pre-activations
# #     tmp    = T.alloc_buffer((B, H), "float32")    # for candidate n pre-tanh

# #     # Initialize hidden state
# #     for b0 in range(B):
# #         for h0i in range(H):
# #             Ht[b0, h0i] = H0[b0, h0i]

# #     # Helper inline ops as lambdas (used only in expressions, no local SSA vars)
# #     # NOTE: Don't bind these to python locals in a loop; just use inline.
# #     # sigmoid(x) = 1 / (1 + exp(-x))

# #     # Time steps: for s in [0..S-1]
# #     for s in range(S):
# #         # gates = X_s @ W^T + Ht @ U^T + b
# #         # Initialize with bias
# #         for b0 in range(B):
# #             for g in range(3*H):
# #                 gates[b0, g] = bg[g]

# #         # X_s @ W^T
# #         for b0 in range(B):
# #             for g in range(3*H):
# #                 for ii in range(I):
# #                     gates[b0, g] = gates[b0, g] + X[s, b0, ii] * Wg[g, ii]

# #         # Ht @ U^T
# #         for b0 in range(B):
# #             for g in range(3*H):
# #                 for hh in range(H):
# #                     gates[b0, g] = gates[b0, g] + Ht[b0, hh] * Ug[g, hh]

# #         # z = sigmoid(gates[0:H]); r = sigmoid(gates[H:2H]); in-place write-back
# #         for b0 in range(B):
# #             for h1 in range(H):
# #                 gates[b0, 0*H + h1] = T.float32(1.0) / (T.float32(1.0) + T.exp(-gates[b0, 0*H + h1]))
# #                 gates[b0, 1*H + h1] = T.float32(1.0) / (T.float32(1.0) + T.exp(-gates[b0, 1*H + h1]))

# #         # tmp = preact for n (start from XW_n + b_n)
# #         for b0 in range(B):
# #             for h1 in range(H):
# #                 tmp[b0, h1] = gates[b0, 2*H + h1]

# #         # add (r * H_{t-1}) * U_n   where U_n = Ug[2H:3H, :]
# #         for b0 in range(B):
# #             for h_out in range(H):
# #                 for h_in in range(H):
# #                     tmp[b0, h_out] = tmp[b0, h_out] + (gates[b0, 1*H + h_in] * Ht[b0, h_in]) * Ug[2*H + h_out, h_in]

# #         # n = tanh(tmp)   (reuse tmp as n)
# #         for b0 in range(B):
# #             for h1 in range(H):
# #                 tmp[b0, h1] = T.tanh(tmp[b0, h1])

# #         # H_t = (1 - z) * n + z * H_{t-1}    (no scalar locals)
# #         for b0 in range(B):
# #             for h1 in range(H):
# #                 Ht[b0, h1] = (T.float32(1.0) - gates[b0, 0*H + h1]) * tmp[b0, h1] + gates[b0, 0*H + h1] * Ht[b0, h1]

# #         # Y[s] = Ht
# #         for b0 in range(B):
# #             for h1 in range(H):
# #                 Y[s, b0, h1] = Ht[b0, h1]

# #     # output last hidden
# #     for b0 in range(B):
# #         for h1 in range(H):
# #             Yh[b0, h1] = Ht[b0, h1]


# # # --------------------------
# # # Build & test
# # # --------------------------
# # if __name__ == "__main__":
# #     mod = tvm.IRModule({"gru_dyn": gru_dyn})
# #     rt = tvm.build(mod, target="llvm")

# #     dev = tvm.cpu()
# #     rng = np.random.default_rng(0)

# #     # Try different dynamic S
# #     S_vals = [1, 3, 5]

# #     for S_val in S_vals:
# #         X  = rng.standard_normal((S_val, B, I), dtype=np.float32)
# #         W  = rng.standard_normal((3*H, I), dtype=np.float32)
# #         U  = rng.standard_normal((3*H, H), dtype=np.float32)
# #         b  = rng.standard_normal((3*H,),   dtype=np.float32)
# #         H0 = rng.standard_normal((B, H),   dtype=np.float32)

# #         Y  = np.empty((S_val, B, H), dtype=np.float32)
# #         Yh = np.empty((B, H),       dtype=np.float32)

# #         # Run TVM
# #         rt(tvm.nd.array(X,  dev),
# #            tvm.nd.array(W,  dev),
# #            tvm.nd.array(U,  dev),
# #            tvm.nd.array(b,  dev),
# #            tvm.nd.array(H0, dev),
# #            tvm.nd.array(Y,  dev),
# #            tvm.nd.array(Yh, dev),
# #            np.int32(S_val))

# #         # Numpy reference (same equations)
# #         def sigmoid(a): return 1.0 / (1.0 + np.exp(-a))
# #         Ht = H0.copy()
# #         Y_ref = np.empty_like(Y)
# #         for s in range(S_val):
# #             pre = (X[s] @ W.T) + (Ht @ U.T) + b  # [B, 3H]
# #             z = sigmoid(pre[:, 0*H:1*H])
# #             r = sigmoid(pre[:, 1*H:2*H])
# #             n_pre = pre[:, 2*H:3*H].copy()
# #             Un = U[2*H:3*H, :]                    # [H, H]
# #             n_pre += (r * Ht) @ Un.T
# #             n = np.tanh(n_pre)
# #             Ht = (1.0 - z) * n + z * Ht
# #             Y_ref[s] = Ht
# #         Yh_ref = Ht

# #         print(f"S={S_val}  max|Y-Y_ref|={np.max(np.abs(Y - Y_ref)):.3e}  "
# #               f"max|Yh-Yh_ref|={np.max(np.abs(Yh - Yh_ref)):.3e}")

# import tvm
# from tvm import relax, tir, te
# bb = relax.BlockBuilder()
# n = tir.Var("n", "int64")
# x = relax.Var("x", relax.TensorStructInfo([n], "float32"))
# y = relax.Var("y", relax.TensorStructInfo([n + 1], "float32"))

# def te_func(A):
#     C = te.compute((n + 1), lambda i: A[i])
#     return C

# with bb.function("rx_func", [x, y]):
#     x1 = bb.emit_te(te_func, y)
#     bb.emit_func_output(x1)

# mod = bb.get()
# mod.show()
# rt = tvm.build(mod, target="llvm")

# dev = tvm.cpu()

# All comments in this code are in English.

import numpy as np
import tvm
from tvm import relax as rx
from tvm import tir
from tvm.script import tir as T
from tvm import te, tir
from tvm.topi import tag


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
