#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <functional>

// --------------------------- Tensor (Minimal) ---------------------------
//
// A minimal dense tensor in row-major order.
// - shape: [d0, d1, ... , d_{r-1}]
// - data: contiguous buffer, size == product(shape)
// - strides: row-major strides
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

// --------------------------- TopK (ONNX-like) ---------------------------
//
// Returns (values, indices) along a given axis.
//
// Shape rule (matches onnx-mlir snippet):
//   out_shape[i] = (i == axis) ? K : X.shape[i]
// and both outputs share the same out_shape.
//
// Core idea:
// - For each "slice" obtained by fixing all dims except 'axis',
//   select top-K elements from the axis vector.
// - indices are positions along the 'axis' dimension (int64).
//
// Common behavior flags:
// - largest=true: pick largest K. largest=false: pick smallest K.
// - sorted=true: sort selected K by value (desc for largest, asc for smallest).
//   sorted=false: unspecified order (we still return a deterministic order here).
//
// Tie-break (deterministic):
// - If values are equal, smaller original index comes first.
//
static std::vector<int64_t> TopKInferShape(
    const std::vector<int64_t>& x_shape,
    int64_t K,
    int64_t axis_raw) {

    const int64_t r = (int64_t)x_shape.size();
    if (r == 0) {
        // Scalar: axis must be 0 or -0 effectively, but ONNX TopK on scalar is unusual.
        // We'll allow rank==0 only if axis resolves to 0 and treat axis dim as 1.
        if (!(axis_raw == 0 || axis_raw == -0))
            throw std::invalid_argument("TopK: invalid axis for scalar input.");
        if (K < 0) throw std::invalid_argument("TopK: K must be >= 0.");
        // Output would be [K]? but scalar has no axis. For simplicity, disallow.
        throw std::invalid_argument("TopK: scalar input is not supported in this demo.");
    }

    int64_t axis = axis_raw;
    axis = (axis < 0) ? (axis + r) : axis;
    if (axis < 0 || axis >= r)
        throw std::invalid_argument("TopK: axis out of range after normalization.");
    if (K < 0)
        throw std::invalid_argument("TopK: K must be >= 0.");

    const int64_t axisDim = x_shape[axis];
    if (axisDim < 0) throw std::invalid_argument("TopK: negative dimension is invalid.");
    if (K > axisDim)
        throw std::invalid_argument("TopK: K is out of bound for the given axis dimension.");

    std::vector<int64_t> out_shape = x_shape;
    out_shape[axis] = K;
    return out_shape;
}

template <typename T>
static void TopK(
    const Tensor<T>& X,
    int64_t K,
    int64_t axis_raw,
    Tensor<T>& values_out,
    Tensor<int64_t>& indices_out,
    bool largest = true,
    bool sorted = true) {

    const int64_t r = X.rank();
    if (r <= 0) throw std::invalid_argument("TopK: rank must be >= 1.");
    int64_t axis = axis_raw;
    axis = (axis < 0) ? (axis + r) : axis;
    if (axis < 0 || axis >= r)
        throw std::invalid_argument("TopK: axis out of range.");
    if (K < 0)
        throw std::invalid_argument("TopK: K must be >= 0.");

    const int64_t axisDim = X.shape[axis];
    if (axisDim < 0) throw std::invalid_argument("TopK: negative dimension is invalid.");
    if (K > axisDim)
        throw std::invalid_argument("TopK: K is out of bound.");

    // Infer output shape and allocate.
    std::vector<int64_t> out_shape = X.shape;
    out_shape[axis] = K;
    values_out = Tensor<T>(out_shape);
    indices_out = Tensor<int64_t>(out_shape);

    // If K == 0, output has 0 elements; done.
    if (values_out.numel() == 0) return;

    // Precompute helpers:
    // outer = product of dims before axis
    // inner = product of dims after axis
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= X.shape[i];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < r; ++i) inner *= X.shape[i];

    // Each slice corresponds to a fixed (outer_idx, inner_idx):
    // slice length = axisDim, element access:
    //   x_index = (outer_idx * axisDim + a) * inner + inner_idx
    // where a in [0, axisDim)
    //
    // Output element access:
    //   out_index = (outer_idx * K + k) * inner + inner_idx
    // where k in [0, K)
    //
    const int64_t numSlices = outer * inner;

    // Temporary buffer per slice: pairs of (value, axis_index)
    std::vector<std::pair<T, int64_t>> buf;
    buf.reserve((size_t)axisDim);

    for (int64_t s = 0; s < numSlices; ++s) {
        const int64_t outer_idx = (inner == 0) ? 0 : (s / inner);
        const int64_t inner_idx = (inner == 0) ? 0 : (s % inner);

        buf.clear();
        for (int64_t a = 0; a < axisDim; ++a) {
            const int64_t x_lin = (outer_idx * axisDim + a) * inner + inner_idx;
            buf.emplace_back(X.data[(size_t)x_lin], a);
        }

        // Comparator with deterministic tie-break.
        //auto cmpLargest = [](const auto& p1, const auto& p2) {
        //    if (p1.first != p2.first) return p1.first > p2.first; // larger value first
        //    return p1.second < p2.second; // smaller index first
        //    };
        //auto cmpSmallest = [](const auto& p1, const auto& p2) {
        //    if (p1.first != p2.first) return p1.first < p2.first; // smaller value first
        //    return p1.second < p2.second; // smaller index first
        //    };

        std::function<bool(const std::pair<T, int64_t>&, const std::pair<T, int64_t>&)> cmp;

        if (largest) {
            cmp = [](const auto& p1, const auto& p2) {
                if (p1.first != p2.first) return p1.first > p2.first;
                return p1.second < p2.second;
                };
        }
        else {
            cmp = [](const auto& p1, const auto& p2) {
                if (p1.first != p2.first) return p1.first < p2.first;
                return p1.second < p2.second;
                };
        }

        // Select top-K using partial_sort for O(D log K).
        // If K == axisDim, this becomes full sort.
        if (K > 0) {
            std::partial_sort(buf.begin(), buf.begin() + K, buf.end(), cmp);

            // If sorted == false, ONNX allows unspecified order; we keep the partial_sort order (deterministic).
            // If sorted == true, partial_sort already provides sorted first K under 'cmp'.
            (void)sorted;
        }

        for (int64_t k = 0; k < K; ++k) {
            const int64_t out_lin = (outer_idx * K + k) * inner + inner_idx;
            values_out.data[(size_t)out_lin] = buf[(size_t)k].first;
            indices_out.data[(size_t)out_lin] = (int64_t)buf[(size_t)k].second;
        }
    }
}

// --------------------------- Demo printing helpers ---------------------------
template <typename T>
static void Print1D(const Tensor<T>& t) {
    if (t.rank() != 1) { std::cout << "Print1D: rank != 1\n"; return; }
    std::cout << "[ ";
    for (int64_t i = 0; i < t.shape[0]; ++i) {
        std::cout << t.data[(size_t)i] << " ";
    }
    std::cout << "]\n";
}

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
    // ---------------- Example A: 2D, axis=1 ----------------
    // X shape: [2, 5]
    // Each row take TopK along columns.
    Tensor<float> X({ 2, 5 });
    // Row 0: [1, 9, 3, 9, 2]
    // Row 1: [4, 0, 7, 7, 6]
    X.data = {
        1, 9, 3, 9, 2,
        4, 0, 7, 7, 6
    };

    std::cout << "X (2x5):\n";
    Print2D(X);

    int64_t K = 3;
    int64_t axis = 1;

    Tensor<float> V;
    Tensor<int64_t> I;
    TopK(X, K, axis, V, I, /*largest=*/true, /*sorted=*/true);

    std::cout << "\nTopK largest=true sorted=true, K=3, axis=1\n";
    std::cout << "Values V shape=[" << V.shape[0] << "," << V.shape[1] << "]\n";
    Print2D(V);
    std::cout << "Indices I shape=[" << I.shape[0] << "," << I.shape[1] << "]\n";
    Print2D(I);

    // ---------------- Example B: 1D, axis=0 ----------------
    Tensor<int> v({ 6 });
    v.data = { 10, 3, 7, 7, 2, 9 };

    Tensor<int> vV;
    Tensor<int64_t> vI;
    TopK(v, /*K=*/2, /*axis=*/0, vV, vI, /*largest=*/true, /*sorted=*/true);

    std::cout << "\n1D TopK K=2 axis=0\n";
    std::cout << "values: "; Print1D(vV);
    std::cout << "indices:"; Print1D(vI);

    return 0;
}
