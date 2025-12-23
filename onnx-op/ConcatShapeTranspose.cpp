#include <cstdint>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <optional>

// -------------------- Tensor --------------------
template <typename T>
struct TensorND {
    std::vector<T> data;
    std::vector<int64_t> shape; // row-major contiguous
};

static inline int64_t NumElements(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1; // scalar
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dim not supported");
        n *= d;
    }
    return n;
}

static inline int64_t NormalizeAxis(int64_t axis, int64_t rank) {
    if (rank <= 0) throw std::invalid_argument("rank must be > 0");
    int64_t a = axis;
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) throw std::out_of_range("axis out of range");
    return a;
}

static inline int64_t ProductRange(const std::vector<int64_t>& shape, int64_t begin, int64_t end) {
    // Product of shape[begin..end-1]
    int64_t p = 1;
    for (int64_t i = begin; i < end; ++i) p *= shape[(size_t)i];
    return p;
}

static inline std::vector<int64_t> ComputeStridesRowMajor(const std::vector<int64_t>& shape) {
    // stride[i] = product(shape[i+1..])
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[(size_t)i] = strides[(size_t)i + 1] * shape[(size_t)i + 1];
    }
    return strides;
}

// -------------------- Concat --------------------
template <typename T>
TensorND<T> Concat(const std::vector<TensorND<T>>& inputs, int64_t axis_raw) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "This Concat assumes trivially copyable T");

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

        const int64_t n = NumElements(s);
        if ((int64_t)inputs[k].data.size() != n) {
            throw std::invalid_argument("Input data size does not match shape");
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
    for (int64_t o = 0; o < outer; ++o) {
        int64_t out_offset = o * (out_axis_dim * inner);
        for (size_t k = 0; k < inputs.size(); ++k) {
            const auto& in = inputs[k];
            const int64_t in_axis_dim = in.shape[(size_t)axis];
            const int64_t in_offset = o * (in_axis_dim * inner);
            const int64_t copy_count = in_axis_dim * inner;

            for (int64_t t = 0; t < copy_count; ++t) {
                out.data[(size_t)(out_offset + t)] = in.data[(size_t)(in_offset + t)];
            }
            out_offset += copy_count;
        }
    }

    return out;
}

// -------------------- Transpose --------------------
template <typename T>
TensorND<T> Transpose(const TensorND<T>& input, const std::vector<int64_t>& perm) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "This Transpose assumes trivially copyable T");

    const int64_t rank = (int64_t)input.shape.size();
    if ((int64_t)perm.size() != rank) throw std::invalid_argument("perm size must equal rank");

    // Validate perm is a permutation of [0..rank-1]
    std::vector<bool> seen((size_t)rank, false);
    for (int64_t i = 0; i < rank; ++i) {
        int64_t p = perm[(size_t)i];
        if (p < 0 || p >= rank) throw std::invalid_argument("perm element out of range");
        if (seen[(size_t)p]) throw std::invalid_argument("perm has duplicates");
        seen[(size_t)p] = true;
    }

    TensorND<T> out;
    out.shape.resize((size_t)rank);
    for (int64_t i = 0; i < rank; ++i) {
        out.shape[(size_t)i] = input.shape[(size_t)perm[(size_t)i]];
    }

    const int64_t n = NumElements(input.shape);
    if ((int64_t)input.data.size() != n) throw std::invalid_argument("input data size mismatch");
    out.data.resize((size_t)n);

    const auto in_strides = ComputeStridesRowMajor(input.shape);
    const auto out_strides = ComputeStridesRowMajor(out.shape);

    // For each output linear index, compute its ND index, map to input ND index, then copy.
    // This is simple and correct; performance can be improved but fine for reference.
    std::vector<int64_t> out_idx((size_t)rank, 0);
    std::vector<int64_t> in_idx((size_t)rank, 0);

    for (int64_t linear = 0; linear < n; ++linear) {
        // Unravel linear index to out_idx
        int64_t tmp = linear;
        for (int64_t i = 0; i < rank; ++i) {
            const int64_t stride = out_strides[(size_t)i];
            out_idx[(size_t)i] = (stride == 0) ? 0 : (tmp / stride);
            tmp = (stride == 0) ? 0 : (tmp % stride);
        }

        // Map out_idx to in_idx using perm:
        // out_dim i comes from in_dim perm[i]
        for (int64_t i = 0; i < rank; ++i) {
            in_idx[(size_t)perm[(size_t)i]] = out_idx[(size_t)i];
        }

        // Ravel in_idx to input linear
        int64_t in_linear = 0;
        for (int64_t i = 0; i < rank; ++i) {
            in_linear += in_idx[(size_t)i] * in_strides[(size_t)i];
        }

        out.data[(size_t)linear] = input.data[(size_t)in_linear];
    }

    return out;
}

// -------------------- Shape slice --------------------
static inline TensorND<int64_t>
ShapeSlice(const std::vector<int64_t>& full_shape, int64_t start, std::optional<int64_t> end_opt) {
    const int64_t rank = (int64_t)full_shape.size();
    int64_t s = start;
    if (s < 0) s += rank;
    if (s < 0 || s > rank) throw std::out_of_range("start out of range");

    int64_t e = end_opt.has_value() ? *end_opt : rank;
    if (e < 0) e += rank;
    if (e < 0 || e > rank) throw std::out_of_range("end out of range");
    if (e < s) throw std::invalid_argument("end must be >= start");

    TensorND<int64_t> out;
    const int64_t len = e - s;
    out.shape = {len};
    out.data.resize((size_t)len);
    for (int64_t i = 0; i < len; ++i) {
        out.data[(size_t)i] = full_shape[(size_t)(s + i)];
    }
    return out;
}

// -------------------- ConcatShapeTranspose fused op --------------------
template <typename T>
struct ConcatShapeTransposeResult {
    TensorND<int64_t> shape; // 1D int64 tensor
    TensorND<T> transposed;  // tensor after concat (+ optional transpose)
};

template <typename T>
ConcatShapeTransposeResult<T>
ConcatShapeTranspose(const std::vector<TensorND<T>>& inputs,
                     int64_t axis,
                     int64_t start,
                     std::optional<int64_t> end,
                     std::optional<std::vector<int64_t>> perm_opt) {
    // 1) Concat
    TensorND<T> concat = Concat<T>(inputs, axis);

    // 2) Optional transpose
    TensorND<T> trans;
    if (perm_opt.has_value()) {
        trans = Transpose<T>(concat, *perm_opt);
    } else {
        trans = std::move(concat);
    }

    // 3) Shape slice on transposed tensor's shape
    TensorND<int64_t> shape = ShapeSlice(trans.shape, start, end);

    return {shape, trans};
}

// -------------------- Demo helpers --------------------
static void PrintShape(const std::vector<int64_t>& s) {
    std::cout << "[";
    for (size_t i = 0; i < s.size(); ++i) {
        std::cout << s[i] << (i + 1 == s.size() ? "" : ",");
    }
    std::cout << "]";
}

template <typename T>
static void PrintFlat(const TensorND<T>& t, const char* name) {
    std::cout << name << " shape=";
    PrintShape(t.shape);
    std::cout << " data={";
    for (size_t i = 0; i < t.data.size(); ++i) {
        std::cout << t.data[i] << (i + 1 == t.data.size() ? "" : ", ");
    }
    std::cout << "}\n";
}

// -------------------- Demo --------------------
static void DemoConcatShapeTranspose() {
    // Two inputs of shape [2,2]
    TensorND<int> A;
    A.shape = {2, 2};
    A.data  = {1, 2,
               3, 4};

    TensorND<int> B;
    B.shape = {2, 2};
    B.data  = {10, 20,
               30, 40};

    std::vector<TensorND<int>> inputs = {A, B};

    // axis=1: concat columns -> shape [2,4]
    // perm=[1,0]: transpose -> shape [4,2]
    // start=0, end=2: take full shape vector [4,2]
    auto r = ConcatShapeTranspose<int>(
        inputs,
        /*axis=*/1,
        /*start=*/0,
        /*end=*/2,
        /*perm_opt=*/std::vector<int64_t>{1, 0}
    );

    std::cout << "=== Inputs ===\n";
    PrintFlat(A, "A");
    PrintFlat(B, "B");

    std::cout << "\n=== Result ===\n";
    PrintFlat(r.transposed, "transposed");
    PrintFlat(r.shape, "shape(slice)");

    // Another example: no perm, only take last dim of shape using start=-1
    auto r2 = ConcatShapeTranspose<int>(
        inputs,
        /*axis=*/1,
        /*start=*/-1,
        /*end=*/std::nullopt,
        /*perm_opt=*/std::nullopt
    );

    std::cout << "\n=== Result 2 (no transpose, shape last dim only) ===\n";
    PrintFlat(r2.transposed, "transposed2");
    PrintFlat(r2.shape, "shape2(slice)");
}

int main() {
    DemoConcatShapeTranspose();
    return 0;
}
