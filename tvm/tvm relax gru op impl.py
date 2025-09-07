from tvm import relax, tir
from tvm.script import tir as T


def _intimm(v: int):
    # Helper to build int64 immediate PrimExpr
    return tir.IntImm("int64", v)

def _shape(sinfo):
    # Helper to read TensorStructInfo.shape (may be ShapeExpr or None)
    return getattr(sinfo, "shape", None)

def _is_intimm(x):
    return isinstance(x, tir.IntImm)

class GRU(OnnxOpConverter):
    name = "GRU"

    @classmethod
    def _impl_v7(cls, bb, inputs, attrs, params):
        """
        GRU via call_tir: Run time-step loop inside a TIR PrimFunc (single direction).
        Returns true GRU outputs with dynamic S/B/I/H supported.

        ONNX inputs:
          X:[S,B,I]
          W:[3H,I] or [D,3H,I]  (this impl uses direction d=0)
          R:[D,3H,H]
          B:[6H] or [D,6H] (optional)
          sequence_lens (optional, ignored in this stub)
          initial_h:[B,H] or [D,B,H] (optional)

        Outputs:
          Y:[S,1,B,H]     (num_directions=1)
          Y_h:[1,B,H]
        """

        # -------- Unpack ONNX inputs --------
        X = inputs[0]                           # [S,B,I]
        W = inputs[1]                           # [3H,I] or [D,3H,I]
        R = inputs[2]                           # [D,3H,H]
        B = inputs[3] if len(inputs) > 3 else None
        # seq_lens = inputs[4] if len(inputs) > 4 else None
        initial_h = inputs[5] if len(inputs) > 5 else None

        linear_before_reset = int(attrs.get("linear_before_reset", 0)) if attrs else 0

        # -------- Infer S,B,I and H (prefer from R) --------
        SX, BX, IX = _shape(X.struct_info)  # [S,B,I]
        if SX is None or BX is None or IX is None:
            raise ValueError("X must have rank-3 shape [S,B,I].")

        r_shape = _shape(R.struct_info)     # [D,3H,H]
        if r_shape is None or len(r_shape) != 3:
            raise ValueError("R must be rank-3 [D,3H,H].")
        D, threeH, H = r_shape
        Hsz = tir.floordiv(threeH, _intimm(3))

        # -------- Slice direction d=0 weights/bias to per-gate tensors --------
        # Helper: slice first direction from rank-3 [D,*,*] -> rank-2 [*,*]
        def dir0_slice_rank3(t):
            shp = _shape(t.struct_info)
            if shp is not None and len(shp) == 3:
                # strided_slice with axes specified explicitly
                s = bb.emit(
                    relax.op.strided_slice(
                        t,
                        axes=[_intimm(0), _intimm(1), _intimm(2)],
                        begin=[_intimm(0), _intimm(0), _intimm(0)],
                        end=[_intimm(1), shp[1], shp[2]],
                        strides=[_intimm(1), _intimm(1), _intimm(1)],
                    )
                )
                return bb.emit(relax.op.squeeze(s, axis=[0]))
            return t

        # W_d0: [3H,I]
        W_d0 = W
        if _shape(W.struct_info) is not None and len(_shape(W.struct_info)) == 3:
            W_d0 = dir0_slice_rank3(W)

        # Split W into Wz, Wr, Wn : [H,I] each (split on axis=0)
        Wparts = relax.op.split(W_d0, 3, axis=0)
        Wz = bb.emit(relax.TupleGetItem(Wparts, 0))
        Wr = bb.emit(relax.TupleGetItem(Wparts, 1))
        Wn = bb.emit(relax.TupleGetItem(Wparts, 2))

        # R_d0: [3H,H]  -> Rz, Rr, Rn : [H,H]
        R_d0 = dir0_slice_rank3(R)
        Rparts = relax.op.split(R_d0, 3, axis=0)
        Rz = bb.emit(relax.TupleGetItem(Rparts, 0))
        Rr = bb.emit(relax.TupleGetItem(Rparts, 1))
        Rn = bb.emit(relax.TupleGetItem(Rparts, 2))

        # Bias handling: B can be [6H] or [D,6H] or None
        # We need wbz,wbr,wbn, rbz,rbr,rbn : each [H]
        def zeros_1d_H():
            return bb.emit(relax.op.zeros(relax.ShapeExpr([Hsz]), "float32"))

        if B is None:
            wbz, wbr, wbn = zeros_1d_H(), zeros_1d_H(), zeros_1d_H()
            rbz, rbr, rbn = zeros_1d_H(), zeros_1d_H(), zeros_1d_H()
        else:
            B_d0 = B
            if _shape(B.struct_info) is not None and len(_shape(B.struct_info)) == 2:
                # [D,6H] -> pick direction 0 -> [6H]
                shpB = _shape(B.struct_info)
                sB = bb.emit(
                    relax.op.strided_slice(
                        B,
                        axes=[_intimm(0), _intimm(1)],
                        begin=[_intimm(0), _intimm(0)],
                        end=[_intimm(1), shpB[1]],
                        strides=[_intimm(1), _intimm(1)],
                    )
                )
                B_d0 = bb.emit(relax.op.squeeze(sB, axis=[0]))  # [6H]

            # First half [0:3H) -> W biases, second half [3H:6H) -> R biases
            first_half = bb.emit(
                relax.op.strided_slice(
                    B_d0,
                    axes=[_intimm(0)],
                    begin=[_intimm(0)],
                    end=[tir.Mul(_intimm(3), Hsz)],
                    strides=[_intimm(1)],
                )
            )  # [3H]
            second_half = bb.emit(
                relax.op.strided_slice(
                    B_d0,
                    axes=[_intimm(0)],
                    begin=[tir.Mul(_intimm(3), Hsz)],
                    end=[tir.Mul(_intimm(6), Hsz)],
                    strides=[_intimm(1)],
                )
            )  # [3H]

            Wb_parts = relax.op.split(first_half, 3, axis=0)
            Rb_parts = relax.op.split(second_half, 3, axis=0)
            wbz = bb.emit(relax.TupleGetItem(Wb_parts, 0))
            wbr = bb.emit(relax.TupleGetItem(Wb_parts, 1))
            wbn = bb.emit(relax.TupleGetItem(Wb_parts, 2))
            rbz = bb.emit(relax.TupleGetItem(Rb_parts, 0))
            rbr = bb.emit(relax.TupleGetItem(Rb_parts, 1))
            rbn = bb.emit(relax.TupleGetItem(Rb_parts, 2))

        # Normalize initial_h to [B,H]
        if initial_h is None:
            h0 = bb.emit(relax.op.zeros(relax.ShapeExpr([BX, Hsz]), "float32"))
        else:
            shp_h = _shape(initial_h.struct_info)
            if shp_h is not None and len(shp_h) == 3:
                # [D,B,H] -> take direction 0 -> [B,H]
                s = bb.emit(
                    relax.op.strided_slice(
                        initial_h,
                        axes=[_intimm(0), _intimm(1), _intimm(2)],
                        begin=[_intimm(0), _intimm(0), _intimm(0)],
                        end=[_intimm(1), shp_h[1], shp_h[2]],
                        strides=[_intimm(1), _intimm(1), _intimm(1)],
                    )
                )
                h0 = bb.emit(relax.op.squeeze(s, axis=[0]))
            else:
                # [B,H] already
                h0 = initial_h

        # -------- Define TIR PrimFunc (time loop runs at runtime) --------
        @T.prim_func
        def gru_time_loop(
            X: T.Buffer((T.int64(), T.int64(), T.int64()), "float32"),   # [S,B,I]
            Wz: T.Buffer((T.int64(), T.int64()), "float32"),             # [H,I]
            Wr: T.Buffer((T.int64(), T.int64()), "float32"),             # [H,I]
            Wn: T.Buffer((T.int64(), T.int64()), "float32"),             # [H,I]
            Rz: T.Buffer((T.int64(), T.int64()), "float32"),             # [H,H]
            Rr: T.Buffer((T.int64(), T.int64()), "float32"),             # [H,H]
            Rn: T.Buffer((T.int64(), T.int64()), "float32"),             # [H,H]
            wbz: T.Buffer((T.int64(),), "float32"),                      # [H]
            wbr: T.Buffer((T.int64(),), "float32"),                      # [H]
            wbn: T.Buffer((T.int64(),), "float32"),                      # [H]
            rbz: T.Buffer((T.int64(),), "float32"),                      # [H]
            rbr: T.Buffer((T.int64(),), "float32"),                      # [H]
            rbn: T.Buffer((T.int64(),), "float32"),                      # [H]
            h0:  T.Buffer((T.int64(), T.int64()), "float32"),            # [B,H]
            Y:   T.Buffer((T.int64(), T.int64(), T.int64()), "float32"), # [S,B,H]
            Yh:  T.Buffer((T.int64(), T.int64()), "float32"),            # [B,H]
            linear_before_reset: T.int32                                  # flag
        ):
            # Read dynamic extents
            S = X.shape[0]
            B = X.shape[1]
            I = X.shape[2]
            H = Wz.shape[0]

            # Temporary buffers
            z_pre = T.alloc_buffer((B, H), "float32")
            r_pre = T.alloc_buffer((B, H), "float32")
            n_pre = T.alloc_buffer((B, H), "float32")
            z_gate = T.alloc_buffer((B, H), "float32")
            r_gate = T.alloc_buffer((B, H), "float32")
            n_cand = T.alloc_buffer((B, H), "float32")
            H_prev = T.alloc_buffer((B, H), "float32")
            H_new  = T.alloc_buffer((B, H), "float32")

            # Extra temporaries for sums to avoid scalar locals
            tmp_sum = T.alloc_buffer((B, H), "float32")
            tmp_sum2 = T.alloc_buffer((B, H), "float32")

            # Initialize H_prev = h0
            for b in range(B):
                for h in range(H):
                    H_prev[b, h] = h0[b, h]

            # Time loop
            for t in range(S):
                # z_pre = X[t] @ Wz^T + H_prev @ Rz^T + wbz + rbz
                for b in range(B):
                    for h in range(H):
                        z_pre[b, h] = T.float32(0)
                for b in range(B):
                    for h in range(H):
                        for i in range(I):
                            z_pre[b, h] = z_pre[b, h] + X[t, b, i] * Wz[h, i]
                for b in range(B):
                    for h in range(H):
                        for k in range(H):
                            z_pre[b, h] = z_pre[b, h] + H_prev[b, k] * Rz[h, k]
                for b in range(B):
                    for h in range(H):
                        z_pre[b, h] = z_pre[b, h] + wbz[h] + rbz[h]

                # r_pre = X[t] @ Wr^T + H_prev @ Rr^T + wbr + rbr
                for b in range(B):
                    for h in range(H):
                        r_pre[b, h] = T.float32(0)
                for b in range(B):
                    for h in range(H):
                        for i in range(I):
                            r_pre[b, h] = r_pre[b, h] + X[t, b, i] * Wr[h, i]
                for b in range(B):
                    for h in range(H):
                        for k in range(H):
                            r_pre[b, h] = r_pre[b, h] + H_prev[b, k] * Rr[h, k]
                for b in range(B):
                    for h in range(H):
                        r_pre[b, h] = r_pre[b, h] + wbr[h] + rbr[h]

                # z = sigmoid(z_pre), r = sigmoid(r_pre)
                for b in range(B):
                    for h in range(H):
                        z_gate[b, h] = T.float32(1) / (T.float32(1) + T.exp(-z_pre[b, h]))
                        r_gate[b, h] = T.float32(1) / (T.float32(1) + T.exp(-r_pre[b, h]))

                if linear_before_reset == 1:
                    # n_pre = X[t] @ Wn^T + r * (H_prev @ Rn^T + rbn) + wbn
                    for b in range(B):
                        for h in range(H):
                            n_pre[b, h] = T.float32(0)
                    # X[t] @ Wn^T
                    for b in range(B):
                        for h in range(H):
                            for i in range(I):
                                n_pre[b, h] = n_pre[b, h] + X[t, b, i] * Wn[h, i]
                    # tmp_sum = H_prev @ Rn^T
                    for b in range(B):
                        for h in range(H):
                            tmp_sum[b, h] = T.float32(0)
                    for b in range(B):
                        for h in range(H):
                            for k in range(H):
                                tmp_sum[b, h] = tmp_sum[b, h] + H_prev[b, k] * Rn[h, k]
                    # add r * (tmp_sum + rbn) + wbn
                    for b in range(B):
                        for h in range(H):
                            n_pre[b, h] = n_pre[b, h] + r_gate[b, h] * (tmp_sum[b, h] + rbn[h]) + wbn[h]
                else:
                    # n_pre = X[t] @ Wn^T + (r * H_prev) @ Rn^T + wbn + rbn
                    for b in range(B):
                        for h in range(H):
                            n_pre[b, h] = T.float32(0)
                    # X[t] @ Wn^T
                    for b in range(B):
                        for h in range(H):
                            for i in range(I):
                                n_pre[b, h] = n_pre[b, h] + X[t, b, i] * Wn[h, i]
                    # tmp_sum2 = (r * H_prev) @ Rn^T
                    for b in range(B):
                        for h in range(H):
                            tmp_sum2[b, h] = T.float32(0)
                    for b in range(B):
                        for h in range(H):
                            for k in range(H):
                                tmp_sum2[b, h] = tmp_sum2[b, h] + (r_gate[b, k] * H_prev[b, k]) * Rn[h, k]
                    # add tmp_sum2 + wbn + rbn
                    for b in range(B):
                        for h in range(H):
                            n_pre[b, h] = n_pre[b, h] + tmp_sum2[b, h] + wbn[h] + rbn[h]

                # n = tanh(n_pre)
                for b in range(B):
                    for h in range(H):
                        n_cand[b, h] = T.tanh(n_pre[b, h])

                # H_new = (1 - z) * n + z * H_prev, write Y[t]
                for b in range(B):
                    for h in range(H):
                        H_new[b, h] = (T.float32(1) - z_gate[b, h]) * n_cand[b, h] + z_gate[b, h] * H_prev[b, h]
                        Y[t, b, h] = H_new[b, h]

                # carry state
                for b in range(B):
                    for h in range(H):
                        H_prev[b, h] = H_new[b, h]

            # Output final state
            for b in range(B):
                for h in range(H):
                    Yh[b, h] = H_prev[b, h]
        # -------- Add PrimFunc to module --------
        bb.add_func(gru_time_loop, "gru_time_loop")

        # -------- call_tir once: run the whole time loop at runtime --------
        Y_sinfo  = relax.TensorStructInfo(shape=[SX, BX, Hsz], dtype="float32")  # [S,B,H]
        Yh_sinfo = relax.TensorStructInfo(shape=[BX, Hsz],     dtype="float32")  # [B,H]
        out_sinfo = relax.TupleStructInfo([Y_sinfo, Yh_sinfo])

        # Pack linear_before_reset as scalar PrimValue
        lbr_scalar = relax.PrimValue(tir.IntImm("int32", linear_before_reset))

        gvar = bb.get().get_global_var("gru_time_loop")
        call = bb.emit(
            relax.call_tir(
                gvar,
                [X, Wz, Wr, Wn, Rz, Rr, Rn, wbz, wbr, wbn, rbz, rbr, rbn, h0, lbr_scalar],
                out_sinfo=[Y_sinfo, Yh_sinfo],
            )
        )

        Y_SBH = bb.emit(relax.TupleGetItem(call, 0))  # [S,B,H]
        Yh_BH = bb.emit(relax.TupleGetItem(call, 1))  # [B,H]

        # ONNX expects Y:[S,1,B,H], Y_h:[1,B,H] for single direction
        Y   = bb.emit(relax.op.expand_dims(Y_SBH, axis=1))
        Y_h = bb.emit(relax.op.expand_dims(Yh_BH, axis=0))
        return [Y, Y_h]
