#include "df_ffi.h"
#include <vector>
#include <iostream>

static DFTensor MakeF32TensorView(const std::vector<int64_t>& shape, float* data) {
    size_t elem = 1;
    for (auto d : shape) elem *= (size_t)d;
    DFTensor t{};
    t.dtype = DF_DTYPE_F32;
    t.rank = (int32_t)shape.size();
    t.shape = shape.data();
    t.data = data;
    t.byte_len = elem * sizeof(float);
    t.strides = nullptr; // contiguous
    return t;
}

int main() {
    DFHandle h = nullptr;
    if (df_plan_create("D:\\code\\mycode\\onnx_mlir_test\\model_shim\\3rd\\deepfilter\\lib\\enc.onnx", &h) != 0) {
        std::cerr << df_last_error_message() << "\n";
        return 1;
    }

    // Build inputs (pulse=1 => S=1)
    std::vector<float> erb(32, 0.0f);
    std::vector<float> spec(192, 0.0f);

    std::vector<int64_t> erb_shape = { 1,1,1,32 };
    std::vector<int64_t> spec_shape = { 1,2,1,96 };

    DFTensor in[2];
    in[0] = MakeF32TensorView(erb_shape, erb.data());
    in[1] = MakeF32TensorView(spec_shape, spec.data());

    // Request first 1 or more outputs
    DFTensor out[1];
    int rc = df_plan_run(h, in, 2, out, 1);
    if (rc != 0) {
        std::cerr << df_last_error_message() << "\n";
        df_plan_destroy(h);
        return 2;
    }

    std::cout << "out0 rank=" << out[0].rank << " bytes=" << out[0].byte_len << "\n";
    auto* out0 = (float*)out[0].data;

    // use out0...

    df_tensor_free(&out[0]);
    df_plan_destroy(h);
    return 0;
}
