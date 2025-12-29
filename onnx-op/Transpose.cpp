#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <numeric>
#include <algorithm>
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
        if (s.empty()) return 1; // scalar
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

// --------------------------- Transpose (ONNX-like) ---------------------------
//
// Transpose permutes dimensions of the input tensor.
//
// Shape rule:
//   if perm is not provided:
//     perm = [rank-1, ..., 0]  (reverse order, numpy default transpose)
//   out_shape[i] = in_shape[perm[i]]
//
// Value rule:
//   out_idx[i] corresponds to in_idx[perm[i]].
//   Equivalently:
//     in_idx[perm[i]] = out_idx[i]
//
// Validations (stronger than the onnx-mlir snippet):
// - perm.size() == rank
// - after normalization (p<0 => p+=rank), p in [0, rank-1]
// - perm is a permutation (no duplicates)
//
static std::vector<int64_t> NormalizePermOrDefault(int64_t rank, const std::vector<int64_t>* permOpt) {
    std::vector<int64_t> perm;
    if (!permOpt || permOpt->empty()) {
        // Default reverse permutation.
        perm.resize((size_t)rank);
        for (int64_t i = 0; i < rank; ++i) perm[(size_t)i] = (rank - 1 - i);
        return perm;
    }

    perm = *permOpt;
    if ((int64_t)perm.size() != rank)
        throw std::invalid_argument("Transpose: perm length must equal input rank.");

    // Normalize negative indices and range-check.
    for (int64_t i = 0; i < rank; ++i) {
        int64_t p = perm[(size_t)i];
        if (p < 0) p += rank;
        if (p < 0 || p >= rank)
            throw std::invalid_argument("Transpose: perm value out of range after normalization.");
        perm[(size_t)i] = p;
    }

    // Check permutation uniqueness.
    std::vector<char> seen((size_t)rank, 0);
    for (int64_t i = 0; i < rank; ++i) {
        int64_t p = perm[(size_t)i];
        if (seen[(size_t)p])
            throw std::invalid_argument("Transpose: perm must be a permutation (duplicate axis found).");
        seen[(size_t)p] = 1;
    }

    return perm;
}

static std::vector<int64_t> TransposeInferShape(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& perm) {

    const int64_t rank = (int64_t)in_shape.size();
    if ((int64_t)perm.size() != rank)
        throw std::invalid_argument("TransposeInferShape: perm length mismatch.");

    std::vector<int64_t> out_shape((size_t)rank);
    for (int64_t i = 0; i < rank; ++i) {
        out_shape[(size_t)i] = in_shape[(size_t)perm[(size_t)i]];
    }
    return out_shape;
}

template <typename T>
static Tensor<T> Transpose(const Tensor<T>& input, const std::vector<int64_t>* permOpt = nullptr) {
    const int64_t rank = input.rank();
    if (rank <= 0) {
        // ONNX allows transpose on scalar? Usually no-op, but rarely used.
        // We'll treat scalar transpose as identity.
        return input;
    }

    std::vector<int64_t> perm = NormalizePermOrDefault(rank, permOpt);
    std::vector<int64_t> out_shape = TransposeInferShape(input.shape, perm);
    Tensor<T> output(out_shape);

    // If any dimension is 0, output has 0 elements.
    if (output.numel() == 0) return output;

    const auto& in_strides = input.strides;
    const auto& out_strides = output.strides;

    // For each output linear index, compute its multi-index, map to input multi-index, compute input linear index.
    const int64_t out_numel = output.numel();
    std::vector<int64_t> out_idx((size_t)rank, 0);
    std::vector<int64_t> in_idx((size_t)rank, 0);

    for (int64_t out_lin = 0; out_lin < out_numel; ++out_lin) {
        // Convert output linear -> out_idx
        int64_t tmp = out_lin;
        for (int64_t i = 0; i < rank; ++i) {
            const int64_t stride = out_strides[(size_t)i];
            out_idx[(size_t)i] = (stride == 0) ? 0 : (tmp / stride);
            tmp = (stride == 0) ? 0 : (tmp % stride);
        }

        // Map out_idx -> in_idx using perm: in_idx[perm[i]] = out_idx[i]
        std::fill(in_idx.begin(), in_idx.end(), 0);
        for (int64_t i = 0; i < rank; ++i) {
            in_idx[(size_t)perm[(size_t)i]] = out_idx[(size_t)i];
        }

        // Compute input linear index.
        int64_t in_lin = 0;
        for (int64_t i = 0; i < rank; ++i) {
            in_lin += in_idx[(size_t)i] * in_strides[(size_t)i];
        }

        output.data[(size_t)out_lin] = input.data[(size_t)in_lin];
    }

    return output;
}

// --------------------------- Demo printing helpers ---------------------------
template <typename T>
static void Print2D(const Tensor<T>& t) {
    if (t.rank() != 2) { std::cout << "Print2D: rank != 2\n"; return; }
    int64_t H = t.shape[0], W = t.shape[1];
    for (int64_t i = 0; i < H; ++i) {
        std::cout << "[ ";
        for (int64_t j = 0; j < W; ++j) {
            std::cout << std::setw(4) << t.data[(size_t)(i * W + j)] << " ";
        }
        std::cout << "]\n";
    }
}

int main() {
    // ---------------- Example 1: default perm (reverse) on 2D ----------------
    Tensor<int> A({ 2, 3 });
    // A =
    // [ [1, 2, 3],
    //   [4, 5, 6] ]
    A.data = { 1,2,3,4,5,6 };

    std::cout << "A (2x3):\n";
    Print2D(A);

    auto AT = Transpose(A, nullptr); // default reverse perm => [1,0]
    std::cout << "\nTranspose(A) with default perm (reverse): shape=["
        << AT.shape[0] << "," << AT.shape[1] << "]\n";
    Print2D(AT);

    // ---------------- Example 2: explicit perm on 3D ----------------
    // Shape [2,3,4], perm [0,2,1] => out shape [2,4,3]
    Tensor<float> X({ 2,3,4 });
    for (int i = 0; i < (int)X.data.size(); ++i) X.data[(size_t)i] = (float)i;

    std::vector<int64_t> perm = { 0, 2, 1 };
    auto Y = Transpose(X, &perm);

    std::cout << "\n3D example: X shape=[2,3,4], perm=[0,2,1] => Y shape=["
        << Y.shape[0] << "," << Y.shape[1] << "," << Y.shape[2] << "]\n";

    return 0;
}
