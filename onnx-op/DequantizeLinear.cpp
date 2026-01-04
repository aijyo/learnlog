#include <cstdint>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <limits>

// --------------------------- DequantizeLinear (ONNX-like) ---------------------------
//
// Returns y (float tensor).
//
// Definition (most common form):
//   y = (x - x_zero_point) * x_scale
//
// Inputs:
//   - x: quantized tensor of integer type (uint8/int8/int32 commonly)
//   - x_scale: scale factor
//       * scalar  -> per-tensor dequant
//       * 1D      -> per-axis dequant, length must equal x.shape[axis]
//   - x_zero_point (optional): zero point
//       * if omitted -> treated as 0
//       * shape must match x_scale (scalar or 1D)
//
// Attribute:
//   - axis: only used when x_scale is 1D (per-axis). Can be negative.
//
// Broadcasting rule:
//   - If x_scale is scalar: use same scale for every element.
//   - If x_scale is 1D: for each element, pick scale/zp by the index along `axis`.
//
// Notes:
//   - In ONNX, output type can be float/float16/bfloat16. Here we output float.
//   - Computation uses float for scale and output; (x - zp) is done in int64 to avoid overflow.
//
// ------------------------------------------------------------------------------------

static int64_t NormalizeAxis(int64_t axis, int64_t rank) {
    if (rank <= 0) throw std::invalid_argument("rank must be > 0");
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) throw std::invalid_argument("axis out of range");
    return axis;
}

static int64_t Numel(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1;
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dim not supported");
        n *= d;
    }
    return n;
}

static std::vector<int64_t> ComputeStridesRowMajor(const std::vector<int64_t>& shape) {
    // Row-major contiguous strides.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static int64_t AxisIndexFromFlatIndex(
    int64_t flatIndex,
    const std::vector<int64_t>& shape,
    int64_t axis)
{
    // Given a flat index (row-major), extract the coordinate along `axis`.
    // coord_axis = (flatIndex / stride[axis]) % shape[axis]
    auto strides = ComputeStridesRowMajor(shape);
    int64_t strideA = strides[axis];
    int64_t dimA = shape[axis];
    return (flatIndex / strideA) % dimA;
}

template <typename QType, typename ZPType>
std::vector<float> DequantizeLinear(
    const std::vector<QType>& x,
    const std::vector<int64_t>& x_shape,
    const std::vector<float>& x_scale,
    const std::vector<ZPType>* x_zero_point,  // nullptr means default 0
    int64_t axis = 1)
{
    const int64_t rank = (int64_t)x_shape.size();
    const int64_t total = Numel(x_shape);
    if ((int64_t)x.size() != total) {
        throw std::invalid_argument("x.size() must equal numel(x_shape)");
    }

    // Validate scale shape: scalar or 1D
    const bool scale_is_scalar = (x_scale.size() == 1);
    const bool scale_is_1d = (x_scale.size() > 1);

    if (!scale_is_scalar && !scale_is_1d) {
        throw std::invalid_argument("x_scale must have at least 1 element");
    }

    int64_t ax = axis;
    if (scale_is_1d) {
        if (rank == 0) throw std::invalid_argument("per-axis requires rank > 0");
        ax = NormalizeAxis(axis, rank);
        if (x_scale.size() != (size_t)x_shape[ax]) {
            throw std::invalid_argument("x_scale length must equal x_shape[axis] for per-axis mode");
        }
    }

    // Validate zero point shape if provided
    if (x_zero_point) {
        if (x_zero_point->size() != x_scale.size()) {
            throw std::invalid_argument("x_zero_point shape/length must match x_scale");
        }
    }

    std::vector<float> y(total);

    for (int64_t i = 0; i < total; ++i) {
        float scale = 0.0f;
        int64_t zp = 0;

        if (scale_is_scalar) {
            scale = x_scale[0];
            zp = x_zero_point ? (int64_t)(*x_zero_point)[0] : 0;
        }
        else {
            int64_t aidx = AxisIndexFromFlatIndex(i, x_shape, ax);
            scale = x_scale[(size_t)aidx];
            zp = x_zero_point ? (int64_t)(*x_zero_point)[(size_t)aidx] : 0;
        }

        // Use int64 for subtraction to avoid overflow issues
        int64_t xv = (int64_t)x[(size_t)i];
        y[(size_t)i] = (float)(xv - zp) * scale;
    }

    return y;
}

static void PrintTensor(const std::vector<float>& data, const std::vector<int64_t>& shape, const std::string& name) {
    std::cout << name << " shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i + 1 < shape.size() ? "," : "");
    }
    std::cout << "]\n";
    std::cout << name << " data: ";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << (i + 1 < data.size() ? ", " : "");
    }
    std::cout << "\n";
}

int main() {
    try {
        // ---------------------------
        // Example A: per-tensor (scalar) dequant
        // x: uint8 tensor, shape [2, 3]
        // scale = 0.1
        // zero_point = 128
        // y = (x - 128) * 0.1
        // ---------------------------
        std::vector<int64_t> shapeA = { 2, 3 };
        std::vector<uint8_t> xA = { 128, 129, 127, 140, 100, 128 };

        std::vector<float> scaleA = { 0.1f };
        std::vector<uint8_t> zpA = { 128 };

        auto yA = DequantizeLinear<uint8_t, uint8_t>(xA, shapeA, scaleA, &zpA, /*axis*/1);
        PrintTensor(yA, shapeA, "yA(per-tensor)");

        // ---------------------------
        // Example B: per-axis dequant (typical per-channel)
        // x: int8 tensor, shape [1, 3, 2]  (N=1, C=3, W=2)
        // axis = 1 (C dimension)
        // scale = [0.1, 0.2, 0.4] per channel
        // zero_point = [0, 0, 0] here omitted (default 0)
        //
        // y[n,c,w] = x[n,c,w] * scale[c]
        // ---------------------------
        std::vector<int64_t> shapeB = { 1, 3, 2 };
        std::vector<int8_t> xB = {
            // c0
            10, -10,
            // c1
            10, -10,
            // c2
            10, -10
        };
        std::vector<float> scaleB = { 0.1f, 0.2f, 0.4f };

        auto yB = DequantizeLinear<int8_t, int8_t>(xB, shapeB, scaleB, /*x_zero_point*/nullptr, /*axis*/1);
        PrintTensor(yB, shapeB, "yB(per-axis, axis=1)");

        // ---------------------------
        // Example C: per-axis with non-zero zero_point
        // x: uint8 shape [1, 2, 2] axis=1 (C=2)
        // scale=[0.5, 2.0], zp=[10, 100]
        // ---------------------------
        std::vector<int64_t> shapeC = { 1, 2, 2 };
        std::vector<uint8_t> xC = {
            // c0
            10, 12,
            // c1
            100, 104
        };
        std::vector<float> scaleC = { 0.5f, 2.0f };
        std::vector<uint8_t> zpC = { 10, 100 };

        auto yC = DequantizeLinear<uint8_t, uint8_t>(xC, shapeC, scaleC, &zpC, /*axis*/1);
        PrintTensor(yC, shapeC, "yC(per-axis with zp)");

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
