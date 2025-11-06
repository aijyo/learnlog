# dfn_onnx_rewriter_v4_fix.py
# Fix: In replace_einsum_btgi_gih_to_matmul(), append the producer Transpose BEFORE any Slice nodes that consume it.
# Other features are identical to v4.

import argparse
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

def convert_unsqueeze_squeeze_axes_attr_to_input(model):
    converted = 0
    new_nodes = []
    for n in model.graph.node:
        if n.op_type in ("Unsqueeze", "Squeeze"):
            has_axes_attr = False
            axes_vals = None
            kept_attrs = []
            for a in n.attribute:
                if a.name == "axes":
                    has_axes_attr = True
                    axes_vals = list(a.ints)
                else:
                    kept_attrs.append(a)
            if has_axes_attr and len(n.input) == 1:
                axes_arr = np.array(axes_vals, dtype=np.int64)
                axes_name = f"{n.name or n.op_type}_axes_const"
                axes_init = numpy_helper.from_array(axes_arr, name=axes_name)
                model.graph.initializer.append(axes_init)
                n.input.append(axes_name)
                del n.attribute[:]
                n.attribute.extend(kept_attrs)
                converted += 1
        new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return converted

def upgrade_opset(model, target_opset=17):
    model.opset_import.clear()
    model.opset_import.extend([helper.make_operatorsetid("", target_opset)])
    return model

def _make_const(name, np_array, dtype=None):
    if dtype is None:
        return numpy_helper.from_array(np_array, name=name)
    return helper.make_tensor(name, dtype, np_array.shape, np_array.flatten().tolist())

def _add_initializer(model, tensor):
    names = {i.name for i in model.graph.initializer}
    if tensor.name not in names:
        model.graph.initializer.append(tensor)

def _get_vi_map(model):
    m = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        m[vi.name] = vi
    return m

def _get_value_info_shape(model, name):
    vi = _get_vi_map(model).get(name)
    if not vi or not vi.type.HasField("tensor_type"):
        return None
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return None
    shp = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            shp.append(d.dim_value)
        elif d.HasField("dim_param"):
            shp.append(None)
        else:
            shp.append(None)
    return shp

def fill_optional_inputs_for_slice(model):
    name2shape = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            shp = []
            for d in vi.type.tensor_type.shape.dim:
                shp.append(d.dim_value if d.HasField("dim_value") else None)
            name2shape[vi.name] = shp
    new_nodes = []
    const_counter = 0
    changed = 0
    for n in model.graph.node:
        if n.op_type == "Slice" and len(n.input) == 3:
            data_name = n.input[0]
            shp = name2shape.get(data_name) or _get_value_info_shape(model, data_name)
            rank = len(shp) if shp is not None else None
            if rank is not None:
                axes = np.arange(rank, dtype=np.int64)
                axes_name = f"{n.name or 'Slice'}_auto_axes_{const_counter}"
                const_counter += 1
                _add_initializer(model, _make_const(axes_name, axes, TensorProto.INT64))
                n.input.append(axes_name)
                steps = np.ones((rank,), dtype=np.int64)
            else:
                steps = np.array([1], dtype=np.int64)
            steps_name = f"{n.name or 'Slice'}_auto_steps_{const_counter}"
            const_counter += 1
            _add_initializer(model, _make_const(steps_name, steps, TensorProto.INT64))
            n.input.append(steps_name)
            changed += 1
        new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return changed

def conv_fill_zero_bias(model):
    new_nodes, counter, changed = [], 0, 0
    for n in model.graph.node:
        if n.op_type == "Conv" and len(n.input) == 2:
            W_name = n.input[1]
            W_init = next((i for i in model.graph.initializer if i.name == W_name), None)
            if W_init is not None:
                W = numpy_helper.to_array(W_init)
                outC = W.shape[0]
                bias_name = f"{n.name or 'Conv'}_auto_bias_{counter}"
                counter += 1
                bias = np.zeros((outC,), dtype=np.float32)
                _add_initializer(model, _make_const(bias_name, bias))
                n.input.append(bias_name)
                changed += 1
        new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return changed

def gemm_fill_zero_bias(model):
    new_nodes, counter, changed = [], 0, 0
    for n in model.graph.node:
        if n.op_type == "Gemm" and len(n.input) == 2:
            B_name = n.input[1]
            B_init = next((i for i in model.graph.initializer if i.name == B_name), None)
            if B_init is not None:
                B = numpy_helper.to_array(B_init)
                N = B.shape[1] if B.ndim == 2 else B.shape[-1]
                bias_name = f"{n.name or 'Gemm'}_auto_bias_{counter}"
                counter += 1
                bias = np.zeros((N,), dtype=np.float32)
                _add_initializer(model, _make_const(bias_name, bias))
                n.input.append(bias_name)
                changed += 1
        new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return changed

def _value_info_shape_lookup(model):
    lookup = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            shp = []
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    shp.append(d.dim_value)
                elif d.HasField("dim_param"):
                    shp.append(None)
                else:
                    shp.append(None)
            lookup[vi.name] = shp
    return lookup

def _can_broadcast(src, dst):
    if src is None or dst is None:
        return False
    a = list(src); b = list(dst)
    if len(a) < len(b):
        a = [1] * (len(b) - len(a)) + a
    if len(b) < len(a):
        return False
    for sa, sb in zip(a, b):
        if sa == sb:
            continue
        if sa == 1 or sa is None:
            continue
        if sb is None:
            continue
        return False
    return True

def make_broadcast_explicit(model, ops=("Add","Sub","Mul","Div")):
    name2shape = _value_info_shape_lookup(model)
    new_nodes = []
    rewritten = 0
    auto = 0
    for n in model.graph.node:
        if n.op_type not in ops or len(n.input) < 2:
            new_nodes.append(n); continue
        A, B = n.input[0], n.input[1]
        sA = name2shape.get(A) or _get_value_info_shape(model, A)
        sB = name2shape.get(B) or _get_value_info_shape(model, B)
        if sA and sB and len(sA) == len(sB) and all((a == b) or (a is None) or (b is None) for a, b in zip(sA, sB)):
            new_nodes.append(n); continue
        target = None; expand_idx = None
        if _can_broadcast(sA, sB):
            target, expand_idx = B, 0
        elif _can_broadcast(sB, sA):
            target, expand_idx = A, 1
        else:
            rankA = len(sA) if sA else None
            rankB = len(sB) if sB else None
            if rankA is not None and rankB is not None:
                if rankA < rankB:
                    target, expand_idx = B, 0
                elif rankB < rankA:
                    target, expand_idx = A, 1
                else:
                    new_nodes.append(n); continue
            else:
                new_nodes.append(n); continue
        shape_out = f"{n.name or n.op_type}_shape_{auto}"; auto += 1
        shape_node = helper.make_node("Shape", [target], [shape_out], name=f"{n.name}_shape" if n.name else None)
        if expand_idx == 0:
            src = A
            exp_out = f"{n.name or n.op_type}_A_expanded_{auto}"; auto += 1
            expand_node = helper.make_node("Expand", [src, shape_out], [exp_out], name=f"{n.name}_expandA" if n.name else None)
            n.input[0] = exp_out
        else:
            src = B
            exp_out = f"{n.name or n.op_type}_B_expanded_{auto}"; auto += 1
            expand_node = helper.make_node("Expand", [src, shape_out], [exp_out], name=f"{n.name}_expandB" if n.name else None)
            n.input[1] = exp_out
        new_nodes.extend([shape_node, expand_node, n])
        rewritten += 1
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return rewritten

def replace_einsum_btgi_gih_to_matmul(model):
    nodes = model.graph.node
    inits = {i.name: i for i in model.graph.initializer}
    value_shapes_cache = {}
    def get_shape(name):
        if name in value_shapes_cache:
            return value_shapes_cache[name]
        s = _get_value_info_shape(model, name)
        value_shapes_cache[name] = s
        return s
    new_nodes = []
    replaced = 0
    auto_id = 0
    for n in nodes:
        if n.op_type == "Einsum":
            eq = None
            for a in n.attribute:
                if a.name == "equation":
                    eq = a.s.decode("utf-8") if isinstance(a.s, (bytes, bytearray)) else a.s
                    break
            if eq == "btgi,gih->btgh":
                X, W = n.input
                if W not in inits:
                    new_nodes.append(n); continue
                W_arr = numpy_helper.to_array(inits[W])
                if W_arr.ndim != 3:
                    new_nodes.append(n); continue
                G, I, H = W_arr.shape
                x_shape = get_shape(X)
                if not x_shape or len(x_shape) != 4:
                    new_nodes.append(n); continue
                B, T, Gx, Ix = x_shape
                if (Gx is not None and Gx != G) or (Ix is not None and Ix != I):
                    new_nodes.append(n); continue

                # Producer first: Transpose X -> [B,G,T,I] so slices can consume it immediately after
                x_t_out = f"{n.name}_X_TBTGI_BGTI_{auto_id}"; auto_id += 1
                x_t = helper.make_node("Transpose", [X], [x_t_out],
                                       perm=[0, 2, 1, 3], name=f"{n.name}_transpose_x")
                new_nodes.append(x_t)

                parts = []
                for g in range(G):
                    starts = np.array([0, g, 0, 0], dtype=np.int64)
                    ends = np.array([B if B is not None else 2**31-1,
                                     g+1,
                                     T if T is not None else 2**31-1,
                                     I], dtype=np.int64)
                    s_name = f"{n.name}_g{g}_starts_{auto_id}"; auto_id += 1
                    e_name = f"{n.name}_g{g}_ends_{auto_id}"; auto_id += 1
                    ax_name = f"{n.name}_g{g}_axes_{auto_id}"; auto_id += 1
                    st_name = f"{n.name}_g{g}_steps_{auto_id}"; auto_id += 1
                    _add_initializer(model, _make_const(s_name, starts, TensorProto.INT64))
                    _add_initializer(model, _make_const(e_name, ends,   TensorProto.INT64))
                    _add_initializer(model, _make_const(ax_name, np.array([0,1,2,3], dtype=np.int64), TensorProto.INT64))
                    _add_initializer(model, _make_const(st_name, np.array([1,1,1,1], dtype=np.int64), TensorProto.INT64))
                    slice_out = f"{n.name}_Xg_{g}_{auto_id}"; auto_id += 1
                    sl = helper.make_node("Slice", [x_t_out, s_name, e_name, ax_name, st_name], [slice_out],
                                          name=f"{n.name}_slice_g{g}")
                    shp2 = np.array([ (B if B is not None else -1)*(T if T is not None else -1), I ], dtype=np.int64)
                    shp2_name = f"{n.name}_g{g}_rshape_{auto_id}"; auto_id += 1
                    _add_initializer(model, _make_const(shp2_name, shp2, TensorProto.INT64))
                    xg_2d = f"{n.name}_Xg2d_{g}_{auto_id}"; auto_id += 1
                    r2 = helper.make_node("Reshape", [slice_out, shp2_name], [xg_2d], name=f"{n.name}_reshape_g{g}")
                    Wg = W_arr[g].reshape(I, H).astype(np.float32)
                    Wg_name = f"{n.name}_Wg_{g}_{auto_id}"; auto_id += 1
                    _add_initializer(model, _make_const(Wg_name, Wg))
                    yg_2d = f"{n.name}_Yg2d_{g}_{auto_id}"; auto_id += 1
                    mm = helper.make_node("MatMul", [xg_2d, Wg_name], [yg_2d], name=f"{n.name}_matmul_g{g}")
                    shp3 = np.array([B if B is not None else -1,
                                     T if T is not None else -1,
                                     1, H], dtype=np.int64)
                    shp3_name = f"{n.name}_g{g}_rshape_back_{auto_id}"; auto_id += 1
                    _add_initializer(model, _make_const(shp3_name, shp3, TensorProto.INT64))
                    yg_4d = f"{n.name}_Yg4d_{g}_{auto_id}"; auto_id += 1
                    r3 = helper.make_node("Reshape", [yg_2d, shp3_name], [yg_4d], name=f"{n.name}_reshape_back_g{g}")
                    new_nodes.extend([sl, r2, mm, r3])
                    parts.append(yg_4d)

                y_cat = f"{n.name}_Ycat_{auto_id}"; auto_id += 1
                cat = helper.make_node("Concat", parts, [y_cat], name=f"{n.name}_concat_g", axis=2)
                final_out = n.output[0]
                id_node = helper.make_node("Identity", [y_cat], [final_out], name=f"{n.name}_identity_out")
                new_nodes.extend([cat, id_node])
                replaced += 1
            else:
                new_nodes.append(n)
        else:
            new_nodes.append(n)
    if replaced > 0:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return replaced

def freeze_dim_params(model, mapping):
    changed = 0
    def _apply(vi):
        nonlocal changed
        if not vi.type.HasField("tensor_type"):
            return
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return
        for d in tt.shape.dim:
            if d.HasField("dim_param"):
                name = d.dim_param
                if name in mapping:
                    d.ClearField("dim_param")
                    d.dim_value = int(mapping[name])
                    changed += 1
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        _apply(vi)
    return changed

def align_conv_input_channels(model):
    init = {i.name: i for i in model.graph.initializer}
    vi_map = _get_vi_map(model)
    def get_shape(name):
        vi = vi_map.get(name)
        if not vi or not vi.type.HasField("tensor_type"):
            return None
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            return None
        shp = []
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                shp.append(d.dim_value)
            else:
                shp.append(None)
        return shp
    new_nodes = []
    fixed = 0
    auto = 0
    for n in model.graph.node:
        if n.op_type != "Conv":
            new_nodes.append(n); continue
        if len(n.input) < 2:
            new_nodes.append(n); continue
        X, W = n.input[0], n.input[1]
        if W not in init:
            new_nodes.append(n); continue
        W_arr = numpy_helper.to_array(init[W])
        if W_arr.ndim != 4:
            new_nodes.append(n); continue
        M, Cpg, kH, kW = W_arr.shape
        g = 1
        for a in n.attribute:
            if a.name == "group" and a.type == onnx.AttributeProto.INT:
                g = a.i
                break
        C_expected = Cpg * g
        x_shape = get_shape(X)
        if not x_shape or len(x_shape) != 4:
            new_nodes.append(n); continue
        if isinstance(x_shape[1], int) and x_shape[1] == C_expected:
            new_nodes.append(n); continue
        candidate_axis = None
        for ax in range(4):
            if ax == 1:
                continue
            if isinstance(x_shape[ax], int) and x_shape[ax] == C_expected:
                candidate_axis = ax
                break
        if candidate_axis is None:
            new_nodes.append(n); continue
        if candidate_axis == 3:
            perm = [0, 3, 1, 2]
        elif candidate_axis == 2:
            perm = [0, 2, 1, 3]
        else:
            new_nodes.append(n); continue
        t_out = f"{n.name or 'Conv'}_align_in_{auto}"; auto += 1
        t = helper.make_node("Transpose", [X], [t_out], perm=perm, name=f"{n.name}_align_channels" if n.name else None)
        n.input[0] = t_out
        new_nodes.extend([t, n])
        fixed += 1
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return fixed

def polish_and_save(model, out_path):
    try:
        ts = getattr(helper, "topological_sort", None)
        if callable(ts):
            model = ts(model)
    except Exception as e:
        print("[WARN] topological_sort failed:", e)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print("[WARN] shape inference failed:", e)
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"[OK] Saved: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--freeze-dim", action="append", default=[], help="Freeze symbolic dim param, format NAME=VALUE. Repeatable.")
    args = ap.parse_args()

    model = onnx.load(args.in_path)
    print("Loaded:", args.in_path)
    print("Original opset imports:", [f"{op.domain or ''}:{op.version}" for op in model.opset_import])

    mapping = {}
    for item in args.freeze_dim:
        if "=" in item:
            k, v = item.split("=", 1)
            k = k.strip()
            v = int(v.strip())
            mapping[k] = v
    if mapping:
        print("Requested freeze dims:", mapping)

    a2i = convert_unsqueeze_squeeze_axes_attr_to_input(model)
    model = upgrade_opset(model, args.opset)

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print("[WARN] pre-align shape inference failed:", e)

    sfix = fill_optional_inputs_for_slice(model)
    cfix = conv_fill_zero_bias(model)
    gfix = gemm_fill_zero_bias(model)
    # bfix = make_broadcast_explicit(model)
    # erep = replace_einsum_btgi_gih_to_matmul(model)
    dfix = freeze_dim_params(model, mapping) if mapping else 0
    afix = align_conv_input_channels(model)

    # print(f"[Pass stats] axes-attr→input: {a2i}, Slice fixed: {sfix}, Conv bias added: {cfix}, Gemm bias added: {gfix}, Broadcast explicit: {bfix}, Einsum replaced: {erep}, Dim frozen: {dfix}, Conv aligned: {afix}")
    polish_and_save(model, args.out_path)

if __name__ == "__main__":
    main()
