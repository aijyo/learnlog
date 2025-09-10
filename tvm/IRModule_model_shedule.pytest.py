# -*- coding: utf-8 -*-
# All comments are in English.

import numpy as np
import tvm
from tvm import tir as T
from tvm.script import tir as T

# --------------------------------
# 1) Single-block MatMul with T.init (reduction block)
# --------------------------------
@T.prim_func
def matmul_tir(a: T.handle, b: T.handle, c: T.handle,
               M: T.int32, N: T.int32, K: T.int32):
    T.func_attr({"global_symbol": "matmul_tir", "tir.noalias": True})
    A = T.match_buffer(a, (M, K), dtype="float32")
    B = T.match_buffer(b, (K, N), dtype="float32")
    C = T.match_buffer(c, (M, N), dtype="float32")

    for i in T.serial(0, M):
        for j in T.serial(0, N):
            for k in T.serial(0, K):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] += A[vi, vk] * B[vk, vj]

# --------------------------------
# 2) Schedule (tile/reorder/vectorize/unroll/parallel with fallback)
# --------------------------------
def schedule_matmul(ir_mod, tile_i=32, tile_j=32, vec=8, unroll_k=4):
    sch = tvm.tir.Schedule(ir_mod)
    sch.work_on("matmul_tir")

    blk = sch.get_block("C", func_name="matmul_tir")
    i, j, k = sch.get_loops(blk)

    # Tile i, j
    io, ii = sch.split(i, factors=[None, tile_i])
    jo, ji = sch.split(j, factors=[None, tile_j])

    # Order: tiles first, then inner tiles, then reduction
    sch.reorder(io, jo, ii, ji, k)

    # Vector lane on ji
    v = vec if tile_j % vec == 0 else min(vec, tile_j)
    ji0, ji1 = sch.split(ji, factors=[None, v])
    sch.reorder(io, jo, ii, ji0, ji1, k)

    # Try to parallelize io; if the version still complains, fuse(io, jo) and parallel
    try:
        sch.parallel(io)
    except tvm.tir.ScheduleError:
        fused = sch.fuse(io, jo)
        sch.parallel(fused)

    # Vectorize innermost j
    sch.vectorize(ji1)

    # Light unroll on k
    if unroll_k > 1:
        ko, ki = sch.split(k, factors=[None, unroll_k])
        sch.unroll(ki)
        sch.reorder(io, jo, ii, ji0, ji1, ko, ki)

    return sch

# --------------------------------
# 3) Build & verify
# --------------------------------
def run_demo(M=256, N=256, K=256, seed=0):
    rng = np.random.default_rng(seed)
    a_np = rng.standard_normal((M, K), dtype=np.float32)
    b_np = rng.standard_normal((K, N), dtype=np.float32)
    c_ref = a_np @ b_np

    ir_mod = tvm.IRModule({"matmul_tir": matmul_tir})
    sch = schedule_matmul(ir_mod, tile_i=32, tile_j=32, vec=8, unroll_k=4)

    print("=== Scheduled TIR ===")
    print(sch.mod.script())

    rt_mod = tvm.build(sch.mod, target="llvm")
    dev = tvm.cpu()
    a_nd = tvm.nd.array(a_np, dev)
    b_nd = tvm.nd.array(b_np, dev)
    c_nd = tvm.nd.empty((M, N), dtype="float32", device=dev)

    rt_mod["matmul_tir"](a_nd, b_nd, c_nd, np.int32(M), np.int32(N), np.int32(K))

    err = float(np.max(np.abs(c_nd.numpy() - c_ref)))
    print(f"Max abs error: {err:.6g}")
    assert err < 1e-3
    print("✅ Passed")

if __name__ == "__main__":
    run_demo()
