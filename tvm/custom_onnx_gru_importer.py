# custom_onnx_gru_importer.py
# -*- coding: utf-8 -*-

import numpy as np
import tvm
from tvm import te, topi
from tvm import relax
from tvm.relax.block_builder import BlockBuilder

# ---------------------------
# TE kernel: single GRU step
# ---------------------------
def gru_step_te(x_t, h_prev, W, R, bW, bR, hidden_size, linear_before_reset):
    """
    Compute one GRU step for a single direction, single layer.

    Shapes:
      x_t:        [B, K]      (float32)
      h_prev:     [B, H]      (float32)
      W:          [3H, K]     (float32)  gate order: [z, r, n] per ONNX
      R:          [3H, H]     (float32)
      bW:         [3H]        (float32)  bias for W-part (may be zero)
      bR:         [3H]        (float32)  bias for R-part (may be zero)
    Returns:
      h_new:      [B, H]
      gates_pre:  [B, 3H]     (pre-activation for potential debugging)
    """

    B, K = x_t.shape
    H = hidden_size

    # Dense: x_t * W^T  -> [B, 3H]
    xW = topi.nn.dense(x_t, W)  # W is [3H, K]; dense will do x * W^T
    # Dense: h_prev * R^T -> [B, 3H]
    hR = topi.nn.dense(h_prev, R)

    # Add biases
    xW = topi.add(xW, bW)
    hR = topi.add(hR, bR)

    # Split gates: z, r, n  (ONNX gate order is [z, r, h] where h is n)
    def split_gates(t):
        z = topi.strided_slice(t, begin=(0, 0), end=(B, H))
        r = topi.strided_slice(t, begin=(0, H), end=(B, 2*H))
        n = topi.strided_slice(t, begin=(0, 2*H), end=(B, 3*H))
        return z, r, n

    xW_z, xW_r, xW_n = split_gates(xW)
    hR_z, hR_r, hR_n = split_gates(hR)

    # z = sigmoid(xW_z + hR_z)
    # r = sigmoid(xW_r + hR_r)
    z_pre = topi.add(xW_z, hR_z)
    r_pre = topi.add(xW_r, hR_r)
    z = topi.sigmoid(z_pre)
    r = topi.sigmoid(r_pre)

    if linear_before_reset:
        # ONNX: if linear_before_reset == 1:
        # n = tanh(xW_n + r * (h_prev * Rn + bRn))
        # We already have hR_n = h_prev * Rn + bRn
        rh = topi.multiply(r, hR_n)
        n_pre = topi.add(xW_n, rh)
        n = topi.tanh(n_pre)
    else:
        # Standard GRU:
        # n = tanh(xW_n + (r * (h_prev)) * Rn + bRn)
        # But we already computed hR_n = h_prev * Rn + bRn
        # We need (r * h_prev) * Rn, so recompute with gating applied first.
        # To avoid another dense, do: (r * h_prev) @ Rn^T + (bRn)
        # Extract Rn: [H, H] from R[2H:3H, :]
        Rn = topi.strided_slice(R, begin=(2*H, 0), end=(3*H, H))
        # (r * h_prev): elementwise gate on hidden state
        rh = topi.multiply(r, h_prev)
        rhRn = topi.nn.dense(rh, Rn)  # -> [B, H]
        # bias for Rn is already in bR[2H:3H]; add it:
        bRn = topi.strided_slice(bR, begin=(2*H,), end=(3*H,))
        rhRn = topi.add(rhRn, bRn)
        n_pre = topi.add(xW_n, rhRn)
        n = topi.tanh(n_pre)

    # h_new = (1 - z) * n + z * h_prev
    one_minus_z = topi.subtract(topi.full_like(z, 1.0), z)
    part1 = topi.multiply(one_minus_z, n)
    part2 = topi.multiply(z, h_prev)
    h_new = topi.add(part1, part2)

    # Optional: concatenated pre-activations for debug (z_pre, r_pre, n_pre)
    gates_pre = topi.concatenate([z_pre, r_pre, n_pre], axis=1)
    return h_new, gates_pre


# --------------------------------------------
# Relax helper: unroll GRU along time dimension
# --------------------------------------------
def build_gru_relax_unrolled(bb: BlockBuilder,
                             X: relax.Expr,
                             W: np.ndarray,
                             R: np.ndarray,
                             B: np.ndarray,
                             h0: relax.Expr | None,
                             linear_before_reset: bool):
    """
    Unroll GRU over time inside Relax using emit_te on top of the TE gru_step.

    Inputs:
      X:  Relax Expr with shape [T, B, K] (float32)
      W:  numpy array [1, 3H, K]
      R:  numpy array [1, 3H, H]
      B:  numpy array [1, 6H] or None
      h0: Relax Expr with shape [1, B, H] or None (if None, use zeros)
      linear_before_reset: bool

    Returns (Y, Y_h):
      Y:   [T, 1, B, H]
      Y_h: [1, B, H]
    """
    # Infer shapes from numpy weights
    assert W.ndim == 3 and R.ndim == 3
    num_dir, threeH, K = W.shape
    assert num_dir == 1, "MVP only supports single direction"
    H = threeH // 3
    T, B, Kx = [int(d) for d in X.struct_info.shape]
    assert Kx == K, "Input size mismatch with W"

    # Split B into bW and bR if provided
    if B is not None:
        assert B.shape == (1, 6 * H)
        bW_np = B[0, :3*H]
        bR_np = B[0, 3*H:]
    else:
        bW_np = np.zeros((3 * H,), dtype="float32")
        bR_np = np.zeros((3 * H,), dtype="float32")

    # Const parameters in Relax
    W_c = relax.const(W[0].astype("float32"))    # [3H, K]
    R_c = relax.const(R[0].astype("float32"))    # [3H, H]
    bW_c = relax.const(bW_np.astype("float32"))  # [3H]
    bR_c = relax.const(bR_np.astype("float32"))  # [3H]

    # Initial hidden state h0: [1, B, H] -> [B, H]
    if h0 is not None:
        h0_sinfo = h0.struct_info
        # squeeze first dim
        h_prev = relax.op.squeeze(h0, axis=[0])
    else:
        h_prev = relax.op.zeros((B, H), "float32")

    # Prepare output Y: accumulate each time step
    ys = []

    # Unroll over time steps: X[t] is [B, K]
    for t in range(T):
        x_t = relax.op.take(X, relax.const(t, "int64"), axis=0)  # [B, K]

        # Emit TE for gru_step
        h_prev, gates_pre = bb.emit_te(
            gru_step_te,
            x_t, h_prev, W_c, R_c, bW_c, bR_c,
            H, linear_before_reset,
        )

        # Save output of this step (Y requires shape [1, B, H] then will stack)
        y_t = h_prev
        ys.append(y_t)

    # Stack along time: list of [B, H] -> [T, B, H]
    Y_tb = relax.op.stack(ys, axis=0)

    # Add direction axis: [T, 1, B, H]
    Y = relax.op.expand_dims(Y_tb, axis=1)

    # Final hidden: [1, B, H]
    Y_h = relax.op.expand_dims(h_prev, axis=0)

    return Y, Y_h


# ---------------------------------------------------
# ONNX frontend hook: custom handler for "GRU" node
# ---------------------------------------------------
def register_custom_onnx_gru_handler():
    """
    Monkey-patch TVM Relax ONNX frontend to handle GRU nodes by unrolling to base ops.
    This avoids OpNotImplemented errors for GRU.

    Usage:
      register_custom_onnx_gru_handler()
      mod = tvm.relax.frontend.onnx.from_onnx(model_onnx)
    """
    from tvm.relax.frontend.onnx.onnx_frontend import Frontend
    from tvm.relax.frontend.onnx import op_impl as _op_impl

    # Keep original dispatch so we can fallback if needed
    original_get_op = Frontend.get_op

    def _my_get_op(self, op_name):
        if op_name == "GRU":
            return _onnx_gru_impl
        return original_get_op(self, op_name)

    # Patch it
    Frontend.get_op = _my_get_op


def _onnx_gru_impl(g, node):
    """
    Implementation for ONNX GRU via unrolled TE in Relax.

    Inputs (by ONNX spec, opset 7+):
      0: X   [T, B, K]
      1: W   [D, 3H, K] (D is num_directions)
      2: R   [D, 3H, H]
      3: B   [D, 6H] or omitted
      4: sequence_lens (ignored in MVP)
      5: initial_h    [D, B, H] or omitted

    Attrs of interest:
      direction (default: "forward")
      linear_before_reset (0 or 1)
      layout (we assume default)
      clip (ignored in MVP)

    Returns:
      Y:   [T, D, B, H]
      Y_h: [D, B, H]
    """
    from tvm.relax.frontend.onnx.onnx_frontend import AttrCvt

    # Parse attributes
    attrs = g.get_attrs(node)
    direction = attrs.get("direction", "forward")
    linear_before_reset = int(attrs.get("linear_before_reset", 0)) == 1

    assert direction in ("forward",), "MVP only supports direction=forward"

    # Fetch inputs (some may be optional)
    X = g.get_input(node, 0, must_exist=True)
    W = g.get_initializer(node, 1)  # numpy or None
    R = g.get_initializer(node, 2)
    B = g.get_initializer(node, 3)  # may be None
    initial_h = g.get_input(node, 5, must_exist=False)  # Relax Expr or None

    # Basic checks
    assert W is not None and R is not None, "GRU requires W, R initializers in MVP"
    assert isinstance(W, np.ndarray) and isinstance(R, np.ndarray)

    # Create Relax using a dataflow block
    bb: BlockBuilder = g.bb
    with bb.function_scope():
        with bb.dataflow():
            Y, Y_h = build_gru_relax_unrolled(
                bb,
                X=X,
                W=W,
                R=R,
                B=B,
                h0=initial_h,
                linear_before_reset=linear_before_reset,
            )
            Y = bb.emit_output(Y)
            Y_h = bb.emit_output(Y_h)
        g.add_output(Y)
        g.add_output(Y_h)

    # Return names for graph bookkeeping
    return [Y, Y_h]


# ----------------
# Example usage
# ----------------
if __name__ == "__main__":
    import onnx
    from tvm.relax.frontend.onnx import from_onnx

    # 1) Register the custom handler
    register_custom_onnx_gru_handler()

    # 2) Load an ONNX model that contains a single GRU (or larger graph with GRU inside)
    #    Replace with your actual path:
    onnx_path = r"D:\learn\tvm\python\resource\DeepFilterNet3_onnx\export\erb_dec.onnx"
    model = onnx.load(onnx_path)

    # 3) Import to Relax (this should no longer throw OpNotImplemented for GRU)
    mod = from_onnx(model)

    # 4) (Optional) Build it
    target = "llvm"
    ex = relax.build(mod, target=target)
    dev = tvm.device(target, 0)
    vm = relax.VirtualMachine(ex, dev)

    print("Imported and built model with custom GRU successfully.")
