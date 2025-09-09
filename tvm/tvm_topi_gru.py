# -*- coding: utf-8 -*-
# TE/scan-based GRU implementation with a topi.nn.lstm-like interface.
# All comments in English as requested.

from typing import Tuple
import tvm
from tvm import te, tir, topi

# Optional: export under topi.nn namespace (adjust to your project layout).
# e.g., from tvm.topi.nn import gru  (after you place this in proper module)

def _check_dims_3H(mat, H, name):
    """Check that mat has shape [3H, K]."""
    assert len(mat.shape) == 2, f"{name} must be rank-2"
    threeH = mat.shape[0]
    if isinstance(threeH, tir.IntImm) and isinstance(H, tir.IntImm):
        assert threeH.value == 3 * H.value, f"{name}.shape[0] != 3*H"
    # If dynamic, we do not hard assert here; rely on schedule-time checks.

def _slice_3H(mat, H, which, name):
    """Slice one gate chunk from [3H, K] -> [H, K].
    which in {0,1,2}.
    """
    K = mat.shape[1]
    i = te.reduce_axis((0, 1), name="__dummy")  # just to avoid flake
    del i
    return te.compute(
        (H, K),
        lambda h, k: mat[which * H + h, k],
        name=f"{name}_slice{which}"
    )

def _reorder_3H(mat, H, layout_from, layout_to, name):
    """Reorder gates along first dim from layout_from to layout_to.
    layout strings are permutations of 'Z', 'R', 'N', e.g., 'ZRN', 'RZN', etc.
    """
    layout_from = layout_from.upper()
    layout_to = layout_to.upper()
    assert set(layout_from) == set("ZRN") == set(layout_to), "Invalid gate layout"
    # Map gate char to index
    idx_from = {ch: i for i, ch in enumerate(layout_from)}
    order = [idx_from[ch] for ch in layout_to]  # e.g., [0,1,2] for identity
    parts = [_slice_3H(mat, H, which=o, name=name) for o in order]
    K = mat.shape[1]
    return topi.concatenate(parts, axis=0)  # (3H, K)

def _dense_bn(xBK, wHK, bH=None, name="dense"):
    """Compute: x[B,K] * w[H,K]^T + (b[H] if provided).
       Returns [B, H].
    """
    B, K = xBK.shape
    H, K2 = wHK.shape
    assert (isinstance(K, tvm.tir.PrimExpr) and isinstance(K2, tvm.tir.PrimExpr)) or K == K2, "K mismatch"
    r = te.reduce_axis((0, K), name=f"{name}_k")
    out = te.compute(
        (B, H),
        lambda b, h: te.sum(xBK[b, r] * wHK[h, r], axis=r),
        name=f"{name}_out"
    )
    if bH is not None:
        out = topi.add(out, topi.expand_dims(bH, axis=0))
    return out  # [B,H]

def gru(
    Xs: te.Tensor,                 # [S, B, I]
    Wi: te.Tensor,                 # [3H, I] packed by weight_layout (default "ZRN")
    Wh: te.Tensor,                 # [3H, H or proj_dim] packed by weight_layout
    Bi: te.Tensor = None,          # [3H], optional; pack matches Wi
    Bh: te.Tensor = None,          # [3H], optional; pack matches Wh
    h_init: te.Tensor = None,      # [B, H], optional zero if None
    proj: te.Tensor = None,        # [P, H], optional projection (output dim P)
    f_act=topi.sigmoid,            # Gate activation (for z, r), default sigmoid
    h_act=topi.tanh,               # Candidate activation (for n), default tanh
    reverse: bool = False,         # Reverse time
    weight_layout: str = "ZRN",    # Gate packing order along 3H: "ZRN" by default
    linear_before_reset: bool = False,  # If True, candidate uses r ⊙ (h @ Rn^T); else (r ⊙ h) @ Rn^T
) -> Tuple[te.Tensor, te.Tensor]:
    """
    General GRU implemented using TE scan.

    Parameters
    ----------
    Xs : te.Tensor
        Input sequence with shape (S, B, I).
    Wi : te.Tensor
        Input weight matrix [3H, I], gates packed by weight_layout.
    Wh : te.Tensor
        Hidden weight matrix [3H, H], packed by weight_layout.
    Bi : te.Tensor, optional
        Input bias [3H], packed by weight_layout.
    Bh : te.Tensor, optional
        Hidden bias [3H], packed by weight_layout.
    h_init : te.Tensor, optional
        Initial hidden state [B, H], zero if None.
    proj : te.Tensor, optional
        Optional projection [P, H]. If provided, Y_seq is projected to P.
    f_act : callable
        Gate activation for z,r (default sigmoid).
    h_act : callable
        Candidate activation for n (default tanh).
    reverse : bool
        Whether to process time in reverse order.
    weight_layout : str
        Gate packing order string among permutations of 'Z','R','N'.
    linear_before_reset : bool
        If True: n = h_act(x*Wn^T + r ⊙ (h*Rn^T) + bn + bhn)
        Else:    n = h_act(x*Wn^T + (r ⊙ h)*Rn^T + bn + bhn)

    Returns
    -------
    Y_seq : te.Tensor
        Output sequence [S, B, P] if proj given else [S, B, H].
    H_seq : te.Tensor
        Hidden state sequence [S, B, H] (pre-projection).
    """
    # -------- Shapes and dtypes --------
    S, B, I = Xs.shape
    threeH, I2 = Wi.shape
    threeH2, H_or_P = Wh.shape
    dtype = Xs.dtype

    # Infer H from Wi first dim (3H)
    # Note: Use tvm.tir.floordiv for general case; here we assume 3H is divisible by 3.
    if isinstance(threeH, tir.IntImm):
        H = tir.IntImm("int32", threeH.value // 3)
    else:
        H = tir.floordiv(threeH, tir.const(3, "int32"))

    # We assume Wh's second dim equals H (no GRU projection on recurrent path).
    # Optional "proj" is applied only to output Y_seq, not to recurrent state.
    _check_dims_3H(Wi, H, "Wi")
    _check_dims_3H(Wh, H, "Wh")

    # -------- Reorder Wi/Wh/Bi/Bh to canonical "ZRN" internal order --------
    # Internally we compute in "ZRN". If user passes another layout, reorder to "ZRN".
    if weight_layout.upper() != "ZRN":
        Wi_int = _reorder_3H(Wi, H, layout_from=weight_layout, layout_to="ZRN", name="Wi")
        Wh_int = _reorder_3H(Wh, H, layout_from=weight_layout, layout_to="ZRN", name="Wh")
        Bi_int = _reorder_3H(topi.expand_dims(Bi, 1), H, layout_from=weight_layout, layout_to="ZRN", name="Bi")[:, 0] if Bi is not None else None
        Bh_int = _reorder_3H(topi.expand_dims(Bh, 1), H, layout_from=weight_layout, layout_to="ZRN", name="Bh")[:, 0] if Bh is not None else None
    else:
        Wi_int, Wh_int, Bi_int, Bh_int = Wi, Wh, Bi, Bh

    # Split into Z,R,N blocks
    Wi_z = _slice_3H(Wi_int, H, 0, "Wi_z")  # [H, I]
    Wi_r = _slice_3H(Wi_int, H, 1, "Wi_r")
    Wi_n = _slice_3H(Wi_int, H, 2, "Wi_n")

    Wh_z = _slice_3H(Wh_int, H, 0, "Wh_z")  # [H, H]
    Wh_r = _slice_3H(Wh_int, H, 1, "Wh_r")
    Wh_n = _slice_3H(Wh_int, H, 2, "Wh_n")

    Bi_z = None if Bi_int is None else te.compute((H,), lambda h: Bi_int[h], name="Bi_z")
    Bi_r = None if Bi_int is None else te.compute((H,), lambda h: Bi_int[H + h], name="Bi_r")
    Bi_n = None if Bi_int is None else te.compute((H,), lambda h: Bi_int[2 * H + h], name="Bi_n")

    Bh_z = None if Bh_int is None else te.compute((H,), lambda h: Bh_int[h], name="Bh_z")
    Bh_r = None if Bh_int is None else te.compute((H,), lambda h: Bh_int[H + h], name="Bh_r")
    Bh_n = None if Bh_int is None else te.compute((H,), lambda h: Bh_int[2 * H + h], name="Bh_n")

    # -------- Time access (support reverse) --------
    def xt_t(t, b, i):
        # If reverse, map t -> (S-1-t); else identity.
        return Xs[(S - 1 - t) if reverse else t, b, i]

    Xs_view = te.compute((S, B, I), lambda t, b, i: xt_t(t, b, i), name="Xs_view")

    # -------- Scan state placeholder & init --------
    H_state = te.placeholder((S, B, H), name="H_state")
    if h_init is None:
        h0 = te.compute((B, H), lambda b, h: tir.const(0, dtype), name="h0")
    else:
        # If provided, shape must be [B, H]
        h0 = h_init

    # -------- One-step update (define via H_state[t-1]) --------
    # We write a te.compute over (S,B,H) using H_state to access previous timestep.
    def step(t, b, h):
        # Slice x_t and h_prev as rank-1/2 views for ease of reuse
        # x_t: [B, I] at time t -> we use dense with shape [B,I] x [H,I]^T => [B,H]
        # However in scalar lambda we compute gate element [b,h] directly.

        # r,gates: compute affine terms
        # Affine_x_gate[b,h] = sum_i Xs_view[t,b,i]*Wi_gate[h,i] + Bi_gate[h]
        # Affine_h_gate[b,h] = sum_j H_prev[b,j]*Wh_gate[h,j] + Bh_gate[h]
        # For scalar compute, we define reductions inline.

        # Helpers for affine terms
        rI = te.reduce_axis((0, I), name="kI")
        rH = te.reduce_axis((0, H), name="kH")

        # previous hidden: if t==0 -> h0; else H_state[t-1,b,h]
        h_prev = tir.if_then_else(t == 0, h0[b, h], H_state[t - 1, b, h])

        # z gate pre-activation
        z_x = te.sum(Xs_view[t, b, rI] * Wi_z[h, rI], axis=rI)
        z_h = te.sum((tir.if_then_else(t == 0, h0[b, rH], H_state[t - 1, b, rH])) * Wh_z[h, rH], axis=rH)
        if Bi_z is not None:
            z_x = z_x + Bi_z[h]
        if Bh_z is not None:
            z_h = z_h + Bh_z[h]
        z_pre = z_x + z_h
        z = f_act(z_pre)  # [scalar]

        # r gate pre-activation
        r_x = te.sum(Xs_view[t, b, rI] * Wi_r[h, rI], axis=rI)
        r_h = te.sum((tir.if_then_else(t == 0, h0[b, rH], H_state[t - 1, b, rH])) * Wh_r[h, rH], axis=rH)
        if Bi_r is not None:
            r_x = r_x + Bi_r[h]
        if Bh_r is not None:
            r_h = r_h + Bh_r[h]
        r_pre = r_x + r_h
        r_gate = f_act(r_pre)

        # candidate n
        n_x = te.sum(Xs_view[t, b, rI] * Wi_n[h, rI], axis=rI)
        if Bi_n is not None:
            n_x = n_x + Bi_n[h]

        if not linear_before_reset:
            # n = h_act( n_x + ((r ⊙ h_prev) @ Rn^T) + Bh_n )
            n_h = te.sum((r_gate * (tir.if_then_else(t == 0, h0[b, rH], H_state[t - 1, b, rH]))) * Wh_n[h, rH], axis=rH)
        else:
            # n = h_act( n_x + (r ⊙ (h_prev @ Rn^T)) + Bh_n )
            pre_h = te.sum((tir.if_then_else(t == 0, h0[b, rH], H_state[t - 1, b, rH])) * Wh_n[h, rH], axis=rH)
            n_h = r_gate * pre_h

        if Bh_n is not None:
            n_pre = n_x + n_h + Bh_n[h]
        else:
            n_pre = n_x + n_h

        n_val = h_act(n_pre)

        # h_t = (1 - z) * n + z * h_prev
        one = tir.const(1.0, dtype)
        h_t = (one - z) * n_val + z * h_prev
        return h_t

    H_update = te.compute((S, B, H), step, name="H_update")

    # -------- Build scan: H_seq --------
    H_seq = te.scan(h0, H_update, H_state, inputs=[Xs_view], name="GRU_scan")

    # -------- Y_seq (projection optional) --------
    if proj is not None:
        # proj: [P, H], Y = H_seq @ proj^T -> [S,B,P]
        P = proj.shape[0]
        rPH = te.reduce_axis((0, H), name="kPH")
        Y_seq = te.compute(
            (S, B, P),
            lambda t, b, p: te.sum(H_seq[t, b, rPH] * proj[p, rPH], axis=rPH),
            name="Y_seq"
        )
    else:
        Y_seq = H_seq

    return Y_seq, H_seq
