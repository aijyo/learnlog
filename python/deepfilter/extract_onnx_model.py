import os
import onnx
from onnx import utils

src_onnx = "D:\\code\\mycode\\onnx_mlir_test\\model_shim\\3rd\\deepfilter\\lib\\enc.onnx"    # 原始onnx
dst_onnx = r"D:\code\mycode\onnx_mlir_test\model_shim\3rd\deepfilter\lib\enc_e0.onnx"        # 输出：只保留到e0的子图

# 1) 读取原模型，拿到“原模型的真实输入名”
m = onnx.load(src_onnx)
input_names = [i.name for i in m.graph.input]

# 2) 你从上一步打印出来的 e0 tensor 名，填到这里
e0_output_name = "e0"   # TODO: 改成你的实际名字

# 3) 裁剪：从原始输入 -> e0
utils.extract_model(
    src_onnx,
    dst_onnx,
    input_names=input_names,
    output_names=[e0_output_name],
)

print("saved:", dst_onnx)

# 4) 如果你的原模型是 external data（超大权重分离存储），extract_model 有时会生成不带外部数据的壳
#    保险起见：重新保存一次，并允许 external data（目录里会生成 enc_e0.onnx.data）
mm = onnx.load(dst_onnx)
onnx.save_model(
    mm,
    dst_onnx,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=os.path.basename(dst_onnx) + ".data",
    size_threshold=1024,  # 小于1KB的张量仍内嵌
)

print("re-saved with external data support:", dst_onnx)
