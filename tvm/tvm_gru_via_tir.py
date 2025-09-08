# -*- coding: utf-8 -*-
# All comments are in English as requested.

import tvm
from tvm import tir as T
from tvm import relax as R
from tvm.relax import BlockBuilder
from tvm.script import tir as T

# Utilities you likely already have in your converter environment:
from .common import infer_shape, infer_type  # adjust the path to your project layout

# -------------------- TIR kernel (single-direction GRU, time loop inside) --------------------

@tvm.script.ir_module
class _GRUKernel:
    @T.prim_func
    def gru_time_loop(
        X: T.handle,     # float32[S,B,I]
        W: T.handle,     # float32[3H,I]
        Rm: T.handle,    # float32[3H,H]
        B: T.handle,     # float32[2,3H] (zeros if no bias)
        h0: T.handle,    # float32[B,H]
        Y: T.handle,     # float32[S,B,H] (out)
        Yh: T.handle,    # float32[B,H]   (out)
        S: T.int32, Bsz: T.int32, I: T.int32, H: T.int32,
        linear_before_reset: T.int32, has_bias: T.int32
    ) -> None:
        T.func_attr({"global_symbol": "gru_time_loop", "tir.noalias": True})
        X_buf  = T.match_buffer(X,  (S, Bsz, I), "float32")
        W_buf  = T.match_buffer(W,  (3*H, I),    "float32")
        R_buf  = T.match_buffer(Rm, (3*H, H),    "float32")
        B_buf  = T.match_buffer(B,  (2, 3*H),    "float32")
        h0_buf = T.match_buffer(h0, (Bsz, H),    "float32")
        Y_buf  = T.match_buffer(Y,  (S, Bsz, H), "float32")
        Yh_buf = T.match_buffer(Yh, (Bsz, H),    "float32")

        Ht   = T.alloc_buffer((Bsz, H), "float32")
        tmp3 = T.alloc_buffer((Bsz, 3*H), "float32")
        zv   = T.alloc_buffer((Bsz, H), "float32")
        rv   = T.alloc_buffer((Bsz, H), "float32")
        hv   = T.alloc_buffer((Bsz, H), "float32")
        rz   = T.alloc_buffer((H, H), "float32")
        rr   = T.alloc_buffer((H, H), "float32")
        rh   = T.alloc_buffer((H, H), "float32")

        # split R into rz, rr, rh
        for i in range(H):
            for j in range(H):
                rz[i, j] = R_buf[i, j]
                rr[i, j] = R_buf[i + H, j]
                rh[i, j] = R_buf[i + 2*H, j]

        # init Ht = h0
        for b in range(Bsz):
            for h in range(H):
                Ht[b, h] = h0_buf[b, h]

        def sigmoid(x: T.float32) -> T.float32:
            return T.float32(1.0) / (T.float32(1.0) + T.exp(-x))

        for s in range(S):
            # tmp3[b, :] = X[s,b,:] * W^T  -> [B,3H]
            for b in range(Bsz):
                for k in range(3*H):
                    acc = T.float32(0.0)
                    for ii in range(I):
                        acc += X_buf[s, b, ii] * W_buf[k, ii]
                    tmp3[b, k] = acc

            # split tmp3 into cz, cr, ch and add WB
            for b in range(Bsz):
                for h in range(H):
                    z_pre = tmp3[b, h]
                    r_pre = tmp3[b, h + H]
                    h_pre = tmp3[b, h + 2*H]
                    if has_bias == 1:
                        z_pre += B_buf[0, h]
                        r_pre += B_buf[0, h + H]
                        h_pre += B_buf[0, h + 2*H]
                    zv[b, h] = z_pre
                    rv[b, h] = r_pre
                    hv[b, h] = h_pre

            # add Ht @ Rz^T / Rr^T (+ RB)
            for b in range(Bsz):
                for h in range(H):
                    accz = zv[b, h]
                    accr = rv[b, h]
                    for hh in range(H):
                        accz += Ht[b, hh] * rz[hh, h]
                        accr += Ht[b, hh] * rr[hh, h]
                    if has_bias == 1:
                        accz += B_buf[1, h]
                        accr += B_buf[1, h + H]
                    zv[b, h] = sigmoid(accz)
                    rv[b, h] = sigmoid(accr)

            # h~ branch
            for b in range(Bsz):
                for h in range(H):
                    if linear_before_reset == 1:
                        acc = T.float32(0.0)
                        for hh in range(H):
                            acc += Ht[b, hh] * rh[hh, h]
                        if has_bias == 1:
                            acc += B_buf[1, h + 2*H]
                        pre = hv[b, h] + rv[b, h] * acc
                        if has_bias == 1:
                            pre += B_buf[0, h + 2*H]
                        hv[b, h] = T.tanh(pre)
                    else:
                        acc = T.float32(0.0)
                        for hh in range(H):
                            acc += (rv[b, hh] * Ht[b, hh]) * rh[hh, h]
                        if has_bias == 1:
                            acc += B_buf[0, h + 2*H] + B_buf[1, h + 2*H]
                        hv[b, h] = T.tanh(hv[b, h] + acc)

            # Ht = (1 - z)*h~ + z*Ht
            for b in range(Bsz):
                for h in range(H):
                    Ht[b, h] = (T.float32(1.0) - zv[b, h]) * hv[b, h] + zv[b, h] * Ht[b, h]

            # Y[s,:,:] = Ht
            for b in range(Bsz):
                for h in range(H):
                    Y_buf[s, b, h] = Ht[b, h]

        # Yh = last Ht
        for b in range(Bsz):
            for h in range(H):
                Yh_buf[b, h] = Ht[b, h]

# -------------------- Converter subclass: GRU (ai.onnx v7) --------------------

class GRU(OnnxOpConverter):
    """Operator converter for GRU (single-direction) with time loop in TIR (Relax)."""
    name = "GRU"

    # Cache the TIR PrimFunc so we don't re-add it for every node.
    _kernel_added = False

    @classmethod
    def _ensure_kernel(cls, bb: BlockBuilder):
        """Add the TIR PrimFunc into the current IRModule (idempotent)."""
        if not cls._kernel_added:
            bb.add_func(_GRUKernel["gru_time_loop"], "gru_time_loop")
            cls._kernel_added = True

    @classmethod
    def _impl_v7(cls, bb: BlockBuilder, inputs, attr, params):
        """
        Relax backend:
        - Moves the time loop into TIR 'gru_time_loop'.
        - Emits a call_tir once for the whole sequence.
        - Returns (Y, Y_h) with shapes [S,1,B,H] and [1,B,H].
        """
        # Unpack inputs (same semantics as your Relay version)
        X = inputs[0]        # [S,B,I]
        W = inputs[1]        # [1,3H,I] in ONNX -> we squeeze dir axis below
        Rm = inputs[2]       # [1,3H,H] in ONNX
        B  = inputs.get("B", None)         # [1,2,3H] -> squeeze to [2,3H] if present
        h0 = inputs.get("initial_h", None) # [1,B,H]  -> squeeze to [B,H] if present

        linear_before_reset = int(attr.get("linear_before_reset", 0))

        # Direction must be 1
        num_dirs = infer_shape(W)[0]
        if num_dirs != 1:
            raise NotImplementedError("Bidirectional GRU is not supported in this implementation.")

        # Squeeze direction axis
        W = bb.emit(R.squeeze(W, axis=[0]))    # [3H,I]
        Rm = bb.emit(R.squeeze(Rm, axis=[0]))  # [3H,H]
        if B is not None:
            B = bb.emit(R.squeeze(B, axis=[0]))  # [2,3H]

        # Infer shapes
        X_shape = infer_shape(X)            # [S,B,I]
        S = X_shape[0]
        Bsz = X_shape[1]
        I = X_shape[2]
        hidden_size = infer_shape(Rm)[1]    # H

        # Prepare h0
        if h0 is None:
            W_dtype = infer_type(W).type_annotation.dtype
            h0 = bb.emit(R.zeros((Bsz, hidden_size), W_dtype))  # [B,H]
        else:
            h0 = bb.emit(R.squeeze(h0, axis=[0]))               # [B,H]

        # Bias handling
        if B is None:
            has_bias = 0
            W_dtype = infer_type(W).type_annotation.dtype
            B = bb.emit(R.zeros((2, 3*hidden_size), W_dtype))    # passthrough zeros
        else:
            has_bias = 1

        # Output placeholders
        Y = bb.emit(R.alloc_tensor((S, Bsz, hidden_size), dtype="float32"))
        Yh = bb.emit(R.alloc_tensor((Bsz, hidden_size), dtype="float32"))

        # Ensure TIR kernel is in module
        cls._ensure_kernel(bb)

        # Call TIR
        _ = bb.emit(
            R.call_tir(
                tvm.ir.GlobalVar("gru_time_loop"),
                (X, W, Rm, B, h0, Y, Yh,
                 tvm.tir.IntImm("int32", S),
                 tvm.tir.IntImm("int32", Bsz),
                 tvm.tir.IntImm("int32", I),
                 tvm.tir.IntImm("int32", hidden_size),
                 tvm.tir.IntImm("int32", linear_before_reset),
                 tvm.tir.IntImm("int32", has_bias)),
                out=R.Tuple((Y, Yh))
            )
        )

        # Conform to ONNX output: add back direction axis
        Y_out  = bb.emit(R.expand_dims(Y, axis=1))   # [S,1,B,H]
        Yh_out = bb.emit(R.expand_dims(Yh, axis=0))  # [1,B,H]
        return (Y_out, Yh_out)

    # --- If your framework still uses the (inputs, attr, params) signature without bb:
    # @classmethod
    # def _impl_v7(cls, inputs, attr, params):
    #     bb = cls.bb  # assuming your base class stashes bb on self/cls
    #     return cls._impl_v7(bb, inputs, attr, params)

