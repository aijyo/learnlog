#include <cstdint>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <iostream>

template <typename T>
struct TensorND {
    std::vector<T> data;
    std::vector<int64_t> shape; // Row-major contiguous
};

static inline int64_t NormalizeAxis(int64_t axis, int64_t rank) {
    if (rank <= 0) throw std::invalid_argument("rank must be > 0");
    int64_t a = axis;
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) throw std::out_of_range("axis out of range");
    return a;
}

static inline int64_t NumElements(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1; // scalar
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dim not supported");
        n *= d;
    }
    return n;
}

static inline int64_t ProductRange(const std::vector<int64_t>& shape, int64_t begin, int64_t end) {
    // Product of shape[begin..end-1]
    int64_t p = 1;
    for (int64_t i = begin; i < end; ++i) p *= shape[(size_t)i];
    return p;
}

template <typename T>
TensorND<T> Concat(const std::vector<TensorND<T>>& inputs, int64_t axis_raw) {
    static_assert(std::is_trivially_copyable<T>::value,
        "This simple Concat assumes trivially copyable element type");

    if (inputs.empty()) throw std::invalid_argument("Concat requires at least one input");

    const int64_t rank = (int64_t)inputs[0].shape.size();
    if (rank <= 0) throw std::invalid_argument("Concat requires rank >= 1 in this impl");

    const int64_t axis = NormalizeAxis(axis_raw, rank);

    // Validate shapes and compute output shape
    std::vector<int64_t> out_shape = inputs[0].shape;
    int64_t out_axis_dim = 0;

    for (size_t k = 0; k < inputs.size(); ++k) {
        const auto& s = inputs[k].shape;
        if ((int64_t)s.size() != rank) throw std::invalid_argument("All inputs must have same rank");

        for (int64_t i = 0; i < rank; ++i) {
            if (i == axis) continue;
            if (s[(size_t)i] != out_shape[(size_t)i]) {
                throw std::invalid_argument("All inputs must match on non-axis dimensions");
            }
        }

        // Validate data size matches shape
        const int64_t n = NumElements(s);
        if ((int64_t)inputs[k].data.size() != n) {
            throw std::invalid_argument("Input data size does not match its shape element count");
        }

        out_axis_dim += s[(size_t)axis];
    }

    out_shape[(size_t)axis] = out_axis_dim;

    const int64_t outer = ProductRange(out_shape, 0, axis);
    const int64_t inner = ProductRange(out_shape, axis + 1, rank);
    const int64_t out_elems = NumElements(out_shape);

    TensorND<T> out;
    out.shape = std::move(out_shape);
    out.data.resize((size_t)out_elems);

    // Copy blocks
    // For each outer index, we append each input's axis blocks (size = axis_dim_k * inner).
    for (int64_t o = 0; o < outer; ++o) {
        int64_t out_offset = o * (out_axis_dim * inner);
        for (size_t k = 0; k < inputs.size(); ++k) {
            const auto& in = inputs[k];
            const int64_t in_axis_dim = in.shape[(size_t)axis];

            const int64_t in_offset = o * (in_axis_dim * inner);
            const int64_t copy_count = in_axis_dim * inner;

            // Copy contiguous region
            for (int64_t t = 0; t < copy_count; ++t) {
                out.data[(size_t)(out_offset + t)] = in.data[(size_t)(in_offset + t)];
            }

            out_offset += copy_count;
        }
    }

    return out;
}

static void PrintTensor2D(const TensorND<int>& t, const char* name) {
    if (t.shape.size() != 2) throw std::invalid_argument("PrintTensor2D expects rank=2");
    const int64_t H = t.shape[0];
    const int64_t W = t.shape[1];

    std::cout << name << " shape=[" << H << "," << W << "]\n";
    for (int64_t i = 0; i < H; ++i) {
        for (int64_t j = 0; j < W; ++j) {
            std::cout << t.data[(size_t)(i * W + j)] << (j + 1 == W ? "" : " ");
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

static void DemoConcat() {
    // Example 1: axis = 0 (concat rows)
    TensorND<int> A;
    A.shape = { 2, 3 };
    A.data = { 1, 2, 3,
               4, 5, 6 };

    TensorND<int> B;
    B.shape = { 1, 3 };
    B.data = { 7, 8, 9 };

    auto Y0 = Concat<int>({ A, B }, /*axis=*/0);
    PrintTensor2D(A, "A");
    PrintTensor2D(B, "B");
    PrintTensor2D(Y0, "Concat axis=0");

    // Example 2: axis = 1 (concat cols)
    TensorND<int> C;
    C.shape = { 2, 2 };
    C.data = { 10, 11,
               12, 13 };

    TensorND<int> D;
    D.shape = { 2, 1 };
    D.data = { 99,
               88 };

    auto Y1 = Concat<int>({ C, D }, /*axis=*/1);
    PrintTensor2D(C, "C");
    PrintTensor2D(D, "D");
    PrintTensor2D(Y1, "Concat axis=1");
}

int main() {
    DemoConcat();
    return 0;
}
