# -*- coding: utf-8 -*-
import numpy as np
import onnxruntime as ort


# ----------------------- C++ vector formatter -----------------------
def format_cpp_vector(name: str, arr: np.ndarray, per_line: int = 6, fmt: str = "%.9e") -> str:
    """
    Format a numpy array into a C++ std::vector<float> initializer.

    Args:
        name: C++ variable name.
        arr: numpy array, will be flattened.
        per_line: how many elements per line.
        fmt: numeric format, e.g. "%.9e".

    Returns:
        A string like: std::vector<float> name = { ... };
    """
    flat = arr.astype(np.float32).ravel()
    lines = []
    lines.append(f"std::vector<float> {name} = {{")
    for i, v in enumerate(flat):
        s = (fmt % float(v)) + "f"
        if i % per_line == 0:
            lines.append("    " + s + ("," if i != flat.size - 1 else ""))
        else:
            lines[-1] += " " + s + ("," if i != flat.size - 1 else "")
    lines.append("};")
    return "\n".join(lines)


def find_erb_input_name(sess: ort.InferenceSession) -> str:
    """
    Try to locate the ERB input tensor name:
    1) Prefer input name containing 'erb'
    2) If only one input, use it
    3) Otherwise, pick the input whose last dim is 32 (common for [1,1,1,32])
    """
    inputs = sess.get_inputs()
    # 1) name contains erb
    for i in inputs:
        if "erb" in i.name.lower():
            return i.name
    # 2) single input
    if len(inputs) == 1:
        return inputs[0].name
    # 3) shape heuristic
    for i in inputs:
        shp = i.shape  # may contain None / 'unk__'
        if shp is not None and len(shp) >= 1:
            try:
                # check last dim equals 32
                if int(shp[-1]) == 32:
                    return i.name
            except Exception:
                pass
    # fallback: first input
    return inputs[0].name


def pick_output(sess: ort.InferenceSession) -> str:
    """
    Prefer output named 'e0', otherwise first output.
    """
    outs = sess.get_outputs()
    for o in outs:
        if o.name == "e0":
            return o.name
    return outs[0].name


# ----------------------- Your two test inputs -----------------------
enc_in_erb_buf_flat_0 = [
    -7.791345716e-1, -6.797742844e-1, -8.138147593e-1, -8.895276189e-1, -8.661772013e-1, -8.536968231e-1,
    -8.418461084e-1, -8.153778315e-1, -7.955732346e-1, -7.720203400e-1, -7.486737967e-1, -7.246176004e-1,
    -7.012136579e-1, -6.757476926e-1, -6.431888342e-1, -6.294124722e-1, -6.044980884e-1, -5.810844302e-1,
    -5.526832342e-1, -2.673951983e-1, -9.493903816e-2, -1.003126130e-1, -2.873689532e-1, -3.927566409e-1,
    -3.836507797e-1, -3.908355832e-1, -3.669029176e-1, -3.430414200e-1, -3.191837370e-1, -2.951537967e-1,
    -2.713060379e-1, -2.473762482e-1
]

enc_in_erb_buf_flat_1 = [
    1.708396971e-1, 3.488453031e-1, 1.762540787e-1, -2.114420012e-2, -3.559608385e-2, -2.049369812e-1,
    -2.041631639e-1, -1.752944887e-1, -2.880403399e-1, -3.005224168e-1, -1.327016801e-1, -1.762481630e-1,
    -1.335769594e-1, 1.082403213e-1, 1.800828874e-1, -1.204706207e-1, -1.080387086e-1, -1.316509210e-2,
    3.766994551e-2, 2.996458113e-1, 6.671785116e-1, 5.717741251e-1, 3.531543612e-1, 3.104232848e-1,
    2.743015289e-1, 2.175287306e-1, 1.293848008e-1, 7.199458778e-2, 3.911991045e-2, 3.165607527e-2,
    -2.154808119e-2, -1.612033881e-2
]


def run_one(sess: ort.InferenceSession, erb_input_name: str, out_name: str, vec: list[float], tag: str):
    x = np.array(vec, dtype=np.float32).reshape(1, 1, 1, 32)

    # If the model has extra inputs, feed them with zeros (best-effort).
    feeds = {}
    for inp in sess.get_inputs():
        if inp.name == erb_input_name:
            feeds[inp.name] = x
        else:
            # best-effort: create zeros with same shape if fully known
            shp = inp.shape
            try:
                concrete = [int(d) for d in shp]
                feeds[inp.name] = np.zeros(concrete, dtype=np.float32)
            except Exception:
                # If dynamic/unknown, you must provide it manually.
                raise RuntimeError(
                    f"模型有额外输入 '{inp.name}'，但它的 shape 是 {shp}（动态/未知），脚本无法自动构造。"
                    f"建议用 extract_model 把模型裁到只剩 erb->e0，或手动给这个输入赋值。"
                )

    y = sess.run([out_name], feeds)[0]

    print("=" * 80)
    print(f"[{tag}] output_name={out_name}  shape={list(y.shape)}  dtype={y.dtype}")
    print(f"[{tag}] stats: min={float(np.min(y)):.9e}, max={float(np.max(y)):.9e}, mean={float(np.mean(y)):.9e}")

    # Print as C++ vector<float>
    print(format_cpp_vector(f"{tag}_e0_out_flat", y, per_line=6, fmt="%.9e"))


def main():
    # TODO: 改成你的模型路径（建议用你裁出来只输出 e0 的 enc_e0.onnx）
    MODEL_PATH = r"D:\code\mycode\onnx_mlir_test\model_shim\3rd\deepfilter\lib\enc_e0_fixed.onnx"

    so = ort.SessionOptions()
    # If you want deterministic-ish runs, keep it single-threaded.
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])

    erb_input_name = find_erb_input_name(sess)
    out_name = pick_output(sess)

    print("Model inputs:")
    for i in sess.get_inputs():
        print(f"  - name={i.name}, type={i.type}, shape={i.shape}")
    print("Model outputs:")
    for o in sess.get_outputs():
        print(f"  - name={o.name}, type={o.type}, shape={o.shape}")

    print(f"\nPicked erb_input_name = {erb_input_name}")
    print(f"Picked output_name    = {out_name}\n")

    run_one(sess, erb_input_name, out_name, enc_in_erb_buf_flat_0, tag="case0")
    run_one(sess, erb_input_name, out_name, enc_in_erb_buf_flat_1, tag="case1")


if __name__ == "__main__":
    main()
