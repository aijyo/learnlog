import onnx
from collections import Counter
from onnx import TensorProto

model = onnx.load("D:\\code\\mycode\\python\\learn\\libdf\\enc.onnx")

init_types = Counter(t.data_type for t in model.graph.initializer)
print("Initializer dtypes:")
for k, v in init_types.most_common():
    print(TensorProto.DataType.Name(k), v)
