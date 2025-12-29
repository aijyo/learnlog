#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iomanip>

// --------------------------- Tensor (Minimal) ---------------------------
//
// A minimal dense tensor in row-major order.
// - shape: [d0, d1, ...]
// - strides: row-major strides
// - data: contiguous buffer
//
template <typename T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<T> data;

    static int64_t numel_from_shape(const std::vector<int64_t>& s) {
        if (s.empty()) return 1;
        int64_t n = 1;
        for (int64_t v : s) {
            if (v < 0) throw std::invalid_argument("Negative dimension is invalid.");
            n *= v;
        }
        return n;
    }

    static std::vector<int64_t> make_row_major_strides(const std::vector<int64_t>& s) {
        std::vector<int64_t> st(s.size(), 1);
        for (int i = (int)s.size() - 2; i >= 0; --i) {
            st[i] = st[i + 1] * s[i + 1];
        }
        return st;
    }

    Tensor() = default;

    explicit Tensor(std::vector<int64_t> shp)
        : shape(std::move(shp)) {
        strides = make_row_major_strides(shape);
        data.resize((size_t)numel_from_shape(shape));
    }

    int64_t rank() const { return (int64_t)shape.size(); }
    int64_t numel() const { return numel_from_shape(shape); }
};

enum class UpsampleMode {
    Nearest,
    LinearLast2D // bilinear on last two dims, nearest on others
};

// --------------------------- Upsample (ONNX-like) ---------------------------
//
// Shape rule (matches onnx-mlir snippet when dims/scales are constant):
//   outDim[i] = int64(inDim[i] * scales[i])  // trunc toward zero, for positive = floor
//
// Verify rule (matches onnx-mlir verify()):
// - scales must be rank-1
// - scales length must equal input rank
//
// Runtime mapping (common practical behavior):
// - Nearest:
//     in_i = floor(out_i / scale_i), clamped
// - LinearLast2D:
//     For last two dims (H,W): bilinear interpolation
//     For other dims: nearest mapping
//
static std::vector<int64_t> UpsampleInferShape(
    const std::vector<int64_t>& in_shape,
    const std::vector<double>& scales) {

    if (scales.size() != in_shape.size())
        throw std::invalid_argument("Upsample: scales length must equal input rank.");

    std::vector<int64_t> out_shape(in_shape.size(), 0);
    for (size_t i = 0; i < in_shape.size(); ++i) {
        if (in_shape[i] < 0) throw std::invalid_argument("Upsample: negative dimension is invalid.");
        if (!(scales[i] > 0.0)) throw std::invalid_argument("Upsample: scale must be > 0.");
        double dim = (double)in_shape[i];
        double s = scales[i];

        // Match onnx-mlir: static_cast<int64_t>(dim * scale)
        // For positive numbers, this truncates like floor.
        out_shape[i] = (int64_t)(dim * s);
    }
    return out_shape;
}

static int64_t clamp_i64(int64_t v, int64_t lo, int64_t hi) {
    return std::max(lo, std::min(v, hi));
}

template <typename T>
static Tensor<T> UpsampleNearest(const Tensor<T>& X, const std::vector<double>& scales) {
    const int64_t r = X.rank();
    if ((int64_t)scales.size() != r)
        throw std::invalid_argument("UpsampleNearest: scales length mismatch.");

    std::vector<int64_t> out_shape = UpsampleInferShape(X.shape, scales);
    Tensor<T> Y(out_shape);
    if (Y.numel() == 0) return Y;

    const auto& in_shape = X.shape;
    const auto& in_strides = X.strides;
    const auto& out_strides = Y.strides;

    // Iterate over output linear index and compute multi-index.
    const int64_t out_numel = Y.numel();
    std::vector<int64_t> out_idx((size_t)r, 0);
    std::vector<int64_t> in_idx((size_t)r, 0);

    for (int64_t out_lin = 0; out_lin < out_numel; ++out_lin) {
        int64_t tmp = out_lin;
        for (int64_t i = 0; i < r; ++i) {
            int64_t s = out_strides[(size_t)i];
            out_idx[(size_t)i] = (s == 0) ? 0 : (tmp / s);
            tmp = (s == 0) ? 0 : (tmp % s);
        }

        // Map output idx -> input idx via nearest (floor(out/scale)).
        for (int64_t i = 0; i < r; ++i) {
            double inv = (double)out_idx[(size_t)i] / scales[(size_t)i];
            int64_t ii = (int64_t)std::floor(inv);
            ii = clamp_i64(ii, 0, in_shape[(size_t)i] - 1);
            in_idx[(size_t)i] = ii;
        }

        // Compute input linear index.
        int64_t in_lin = 0;
        for (int64_t i = 0; i < r; ++i) {
            in_lin += in_idx[(size_t)i] * in_strides[(size_t)i];
        }

        Y.data[(size_t)out_lin] = X.data[(size_t)in_lin];
    }

    return Y;
}

template <typename T>
static Tensor<T> UpsampleLinearLast2D(const Tensor<T>& X, const std::vector<double>& scales) {
    const int64_t r = X.rank();
    if (r < 2) throw std::invalid_argument("UpsampleLinearLast2D: rank must be >= 2.");
    if ((int64_t)scales.size() != r)
        throw std::invalid_argument("UpsampleLinearLast2D: scales length mismatch.");

    std::vector<int64_t> out_shape = UpsampleInferShape(X.shape, scales);
    Tensor<T> Y(out_shape);
    if (Y.numel() == 0) return Y;

    const int64_t H_in = X.shape[(size_t)r - 2];
    const int64_t W_in = X.shape[(size_t)r - 1];
    const int64_t H_out = Y.shape[(size_t)r - 2];
    const int64_t W_out = Y.shape[(size_t)r - 1];

    const double sH = scales[(size_t)r - 2];
    const double sW = scales[(size_t)r - 1];

    const auto& in_strides = X.strides;
    const auto& out_strides = Y.strides;

    // We treat all dims except last2 as "batch dims".
    // We'll iterate over all output positions, but compute bilinear only on last2.
    const int64_t out_numel = Y.numel();
    std::vector<int64_t> out_idx((size_t)r, 0);
    std::vector<int64_t> in_base((size_t)r, 0);

    for (int64_t out_lin = 0; out_lin < out_numel; ++out_lin) {
        int64_t tmp = out_lin;
        for (int64_t i = 0; i < r; ++i) {
            int64_t s = out_strides[(size_t)i];
            out_idx[(size_t)i] = (s == 0) ? 0 : (tmp / s);
            tmp = (s == 0) ? 0 : (tmp % s);
        }

        // Map non-last2 dims by nearest.
        for (int64_t i = 0; i < r - 2; ++i) {
            double inv = (double)out_idx[(size_t)i] / scales[(size_t)i];
            int64_t ii = (int64_t)std::floor(inv);
            ii = clamp_i64(ii, 0, X.shape[(size_t)i] - 1);
            in_base[(size_t)i] = ii;
        }

        // Bilinear on last2 dims.
        const int64_t oh = out_idx[(size_t)r - 2];
        const int64_t ow = out_idx[(size_t)r - 1];

        // Common practical mapping: in = out / scale
        const double ih_f = (double)oh / sH;
        const double iw_f = (double)ow / sW;

        int64_t ih0 = (int64_t)std::floor(ih_f);
        int64_t iw0 = (int64_t)std::floor(iw_f);
        int64_t ih1 = ih0 + 1;
        int64_t iw1 = iw0 + 1;

        ih0 = clamp_i64(ih0, 0, H_in - 1);
        iw0 = clamp_i64(iw0, 0, W_in - 1);
        ih1 = clamp_i64(ih1, 0, H_in - 1);
        iw1 = clamp_i64(iw1, 0, W_in - 1);

        const double dh = ih_f - (double)ih0;
        const double dw = iw_f - (double)iw0;

        // Compute 4 neighbors' linear indices.
        auto in_offset = [&](int64_t h, int64_t w) -> int64_t {
            int64_t lin = 0;
            for (int64_t i = 0; i < r - 2; ++i) lin += in_base[(size_t)i] * in_strides[(size_t)i];
            lin += h * in_strides[(size_t)r - 2];
            lin += w * in_strides[(size_t)r - 1];
            return lin;
        };

        const T v00 = X.data[(size_t)in_offset(ih0, iw0)];
        const T v01 = X.data[(size_t)in_offset(ih0, iw1)];
        const T v10 = X.data[(size_t)in_offset(ih1, iw0)];
        const T v11 = X.data[(size_t)in_offset(ih1, iw1)];

        // Bilinear interpolation in double, then cast back.
        const double a0 = (1.0 - dw) * (double)v00 + dw * (double)v01;
        const double a1 = (1.0 - dw) * (double)v10 + dw * (double)v11;
        const double v  = (1.0 - dh) * a0 + dh * a1;

        Y.data[(size_t)out_lin] = (T)v;
    }

    return Y;
}

template <typename T>
static Tensor<T> Upsample(const Tensor<T>& X, const std::vector<double>& scales, UpsampleMode mode) {
    switch (mode) {
    case UpsampleMode::Nearest:
        return UpsampleNearest(X, scales);
    case UpsampleMode::LinearLast2D:
        return UpsampleLinearLast2D(X, scales);
    default:
        throw std::invalid_argument("Upsample: unknown mode.");
    }
}

// --------------------------- Demo printing helpers ---------------------------
template <typename T>
static void Print2D(const Tensor<T>& t) {
    if (t.rank() != 2) { std::cout << "Print2D: rank != 2\n"; return; }
    int64_t H = t.shape[0], W = t.shape[1];
    for (int64_t i = 0; i < H; ++i) {
        std::cout << "[ ";
        for (int64_t j = 0; j < W; ++j) {
            std::cout << std::setw(6) << t.data[(size_t)(i * W + j)] << " ";
        }
        std::cout << "]\n";
    }
}

int main() {
    // Example: 2D upsample (like a single-channel image)
    Tensor<float> X({2, 3});
    // X =
    // [ [1, 2, 3],
    //   [4, 5, 6] ]
    X.data = {1,2,3,4,5,6};

    std::vector<double> scales = {2.0, 2.0}; // Hx2, Wx2

    std::cout << "X shape=[2,3]\n";
    Print2D(X);

    auto Y_nearest = Upsample(X, scales, UpsampleMode::Nearest);
    std::cout << "\nNearest Upsample scales=[2,2], Y shape=["
              << Y_nearest.shape[0] << "," << Y_nearest.shape[1] << "]\n";
    Print2D(Y_nearest);

    auto Y_linear = Upsample(X, scales, UpsampleMode::LinearLast2D);
    std::cout << "\nLinear(bilinear last2D) Upsample scales=[2,2], Y shape=["
              << Y_linear.shape[0] << "," << Y_linear.shape[1] << "]\n";
    Print2D(Y_linear);

    return 0;
}
