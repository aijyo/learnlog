#include <iostream>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <cstdint>
#include <iomanip>

// --------------------------- Tensor (Minimal) ---------------------------
//
// A minimal dense tensor in row-major order.
// - shape: [d0, d1, ... , d_{r-1}]
// - data: contiguous buffer, size == product(shape)
// - strides: computed for row-major indexing
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

    Tensor(std::vector<int64_t> shp)
        : shape(std::move(shp)) {
        strides = make_row_major_strides(shape);
        data.resize((size_t)numel_from_shape(shape));
    }

    int64_t rank() const { return (int64_t)shape.size(); }
    int64_t numel() const { return numel_from_shape(shape); }

    int64_t offset_of(const std::vector<int64_t>& idx) const {
        if ((int64_t)idx.size() != rank())
            throw std::invalid_argument("Index rank mismatch.");
        int64_t off = 0;
        for (int64_t i = 0; i < rank(); ++i) {
            if (idx[i] < 0 || idx[i] >= shape[i])
                throw std::out_of_range("Index out of bounds.");
            off += idx[i] * strides[i];
        }
        return off;
    }

    T& at(const std::vector<int64_t>& idx) {
        return data[(size_t)offset_of(idx)];
    }
    const T& at(const std::vector<int64_t>& idx) const {
        return data[(size_t)offset_of(idx)];
    }
};

// --------------------------- Tile (ONNX-like) ---------------------------
//
// Tile repeats the input tensor along each axis.
// Shape rule:
//   output_shape[i] = input_shape[i] * repeats[i]
//
// Index mapping rule (row-major conceptual):
//   in_idx[i] = out_idx[i] % input_shape[i]
//
// Notes:
// - repeats must be 1D and length == input_rank.
// - repeats[i] >= 0. If repeats[i] == 0, output dim becomes 0.
//
static std::vector<int64_t> TileInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& repeats) {

    if (repeats.size() != input_shape.size()) {
        throw std::invalid_argument("Tile: repeats length must equal input rank.");
    }

    std::vector<int64_t> out_shape(input_shape.size(), 0);
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (input_shape[i] < 0) {
            throw std::invalid_argument("Tile: negative input dimension is invalid.");
        }
        if (repeats[i] < 0) {
            throw std::invalid_argument("Tile: negative repeats is invalid.");
        }
        out_shape[i] = input_shape[i] * repeats[i];
    }
    return out_shape;
}

template <typename T>
static Tensor<T> Tile(const Tensor<T>& input, const std::vector<int64_t>& repeats) {
    const int64_t r = input.rank();
    if ((int64_t)repeats.size() != r) {
        throw std::invalid_argument("Tile: repeats length must equal input rank.");
    }

    // Infer output shape.
    std::vector<int64_t> out_shape = TileInferShape(input.shape, repeats);
    Tensor<T> output(out_shape);

    // Early exit: if any dim is 0, output has 0 elements.
    if (output.numel() == 0) return output;

    // For scalar: rank == 0, repeats must be empty, output is identical.
    if (r == 0) {
        output.data[0] = input.data[0];
        return output;
    }

    // Compute output strides once.
    const auto& in_shape = input.shape;
    const auto& in_strides = input.strides;
    const auto& out_strides = output.strides;

    // Iterate over all output linear indices, map to input linear index.
    // Complexity: O(out_numel * rank)
    const int64_t out_numel = output.numel();
    for (int64_t out_lin = 0; out_lin < out_numel; ++out_lin) {
        int64_t tmp = out_lin;
        int64_t in_lin = 0;

        for (int64_t axis = 0; axis < r; ++axis) {
            // Convert linear index to multi-dim index along this axis.
            const int64_t out_idx_axis = tmp / out_strides[axis];
            tmp = tmp % out_strides[axis];

            // Map to input index by modulo.
            // If in_shape[axis] == 0, output dim should also be 0 (handled above).
            const int64_t in_idx_axis = (in_shape[axis] == 0) ? 0 : (out_idx_axis % in_shape[axis]);
            in_lin += in_idx_axis * in_strides[axis];
        }

        output.data[(size_t)out_lin] = input.data[(size_t)in_lin];
    }

    return output;
}

// --------------------------- Demo helpers ---------------------------
template <typename T>
static void Print2D(const Tensor<T>& t) {
    if (t.rank() != 2) {
        std::cout << "Print2D: rank != 2\n";
        return;
    }
    int64_t H = t.shape[0], W = t.shape[1];
    for (int64_t i = 0; i < H; ++i) {
        for (int64_t j = 0; j < W; ++j) {
            std::cout << std::setw(4) << t.at({ i, j }) << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Example 1: 2D tile
    Tensor<int> a({ 2, 2 });
    // Fill:
    // [ [1, 2],
    //   [3, 4] ]
    a.at({ 0,0 }) = 1; a.at({ 0,1 }) = 2;
    a.at({ 1,0 }) = 3; a.at({ 1,1 }) = 4;

    std::cout << "Input A (2x2):\n";
    Print2D(a);

    std::vector<int64_t> repeats = { 2, 3 };
    auto b = Tile(a, repeats);

    std::cout << "\nRepeats = [2, 3]\n";
    std::cout << "Output B shape = [" << b.shape[0] << ", " << b.shape[1] << "]\n";
    std::cout << "Output B:\n";
    Print2D(b);

    // Example 2: 1D tile
    Tensor<float> v({ 3 });
    v.data = { 0.5f, 1.5f, 2.5f };
    auto v2 = Tile(v, { 4 });
    std::cout << "\nInput v (len=3), repeats=[4], output len=" << v2.shape[0] << "\n";
    for (auto x : v2.data) std::cout << x << " ";
    std::cout << "\n";

    return 0;
}
