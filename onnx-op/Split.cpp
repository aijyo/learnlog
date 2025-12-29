#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// ------------------------------ Tensor (row-major) ------------------------------
//
// A minimal N-D tensor container:
// - shape: N-D dimensions
// - data : flat storage in row-major order
//
template <typename T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<T> data;

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) {
            if (d < 0) throw std::runtime_error("Negative dimension is not supported.");
            n *= d;
        }
        return n;
    }
};

// Compute row-major strides: strides[i] = product(shape[i+1:])
static inline std::vector<int64_t> compute_strides_row_major(const std::vector<int64_t>& shape) {
    const int64_t r = static_cast<int64_t>(shape.size());
    std::vector<int64_t> strides(r, 1);
    for (int64_t i = r - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// Normalize negative axis
static inline int64_t normalize_axis(int64_t axis, int64_t rank) {
    if (axis < -rank || axis >= rank) throw std::runtime_error("Split: axis out of bound.");
    if (axis < 0) axis += rank;
    return axis;
}

// Compute split sizes following onnx-mlir equal-split behavior:
// - If D % K != 0, the first (D%K) chunks are bigger by 1.
static inline std::vector<int64_t> compute_equal_split_sizes(int64_t D, int64_t K) {
    if (K <= 0) throw std::runtime_error("Split: num_outputs must be positive.");
    if (D < 0) throw std::runtime_error("Split: negative dimension is not supported.");

    std::vector<int64_t> sizes;
    sizes.reserve(static_cast<size_t>(K));

    const int64_t q = D / K;   // floor
    const int64_t r = D % K;   // remainder

    for (int64_t i = 0; i < K; ++i) {
        sizes.push_back(i < r ? (q + 1) : q);
    }
    return sizes;
}

// ------------------------------ Split (runtime) ------------------------------
//
// Implements ONNX Split op in runtime form.
// - input: N-D tensor
// - axis: split axis (can be negative)
// - num_outputs: number of outputs (equals numOfResults in onnx-mlir snippet)
// - split_opt: optional explicit split sizes; if null => equal-split
//
// Returns a vector of output tensors.
//
template <typename T>
std::vector<Tensor<T>> onnx_split(const Tensor<T>& input,
    int64_t axis,
    int64_t num_outputs,
    const std::vector<int64_t>* split_opt = nullptr) {
    const int64_t rank = input.rank();
    if (rank <= 0) throw std::runtime_error("Split: input must be ranked.");
    if (num_outputs <= 0) throw std::runtime_error("Split: num_outputs must be > 0.");

    const int64_t ax = normalize_axis(axis, rank);
    const int64_t D = input.shape[ax];

    // Determine split sizes
    std::vector<int64_t> splitSizes;
    if (split_opt) {
        if (static_cast<int64_t>(split_opt->size()) != num_outputs)
            throw std::runtime_error("Split: split size not equal to number of outputs.");
        splitSizes = *split_opt;

        int64_t sum = 0;
        for (int64_t s : splitSizes) {
            if (s < 0) throw std::runtime_error("Split: split sizes must be non-negative.");
            sum += s;
        }
        if (sum != D) throw std::runtime_error("Split: sum(split) must equal input dim at axis.");
    }
    else {
        // Equal split, onnx-mlir style: first remainder chunks are larger
        splitSizes = compute_equal_split_sizes(D, num_outputs);

        // In onnx-mlir, there is an extra check for older opsets when D is literal
        // and D % num_outputs != 0. Here we follow the snippet behavior: it allows
        // uneven splits by distributing remainder to early chunks (ceil/floor mix).
    }

    // Prepare outputs (shapes and allocations)
    std::vector<Tensor<T>> outputs(static_cast<size_t>(num_outputs));
    for (int64_t i = 0; i < num_outputs; ++i) {
        outputs[i].shape = input.shape;
        outputs[i].shape[ax] = splitSizes[i];
        outputs[i].data.resize(outputs[i].numel());
    }

    // If any output is empty, still fine (can happen if D < num_outputs)
    // Now perform data copy by slicing contiguous blocks along the axis in row-major layout.

    const auto inStrides = compute_strides_row_major(input.shape);
    const int64_t inner = inStrides[ax];              // product of dims after axis
    const int64_t outer = input.numel() / (D * inner); // product of dims before axis

    // For each outer index, we copy axis segments as contiguous blocks of length (split * inner)
    int64_t axisOffset = 0;
    for (int64_t outIdx = 0; outIdx < num_outputs; ++outIdx) {
        const int64_t cur = splitSizes[outIdx];
        const int64_t block = cur * inner;

        for (int64_t o = 0; o < outer; ++o) {
            const int64_t inBase = o * (D * inner) + axisOffset * inner;
            const int64_t outBase = o * block;

            // Copy contiguous block
            std::copy_n(input.data.begin() + inBase, block, outputs[outIdx].data.begin() + outBase);
        }

        axisOffset += cur;
    }

    return outputs;
}

// ------------------------------ Demo Helpers ------------------------------
template <typename T>
static void print_shape(const Tensor<T>& t, const std::string& name) {
    std::cout << name << " shape = [";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        std::cout << t.shape[i] << (i + 1 == t.shape.size() ? "" : ", ");
    }
    std::cout << "]\n";
}

static Tensor<int64_t> make_tensor_i64(std::vector<int64_t> shape) {
    Tensor<int64_t> t;
    t.shape = std::move(shape);
    t.data.resize(t.numel());
    std::iota(t.data.begin(), t.data.end(), 0);
    return t;
}

// Print a 2D tensor for demo convenience (rank==2)
static void print_2d_i64(const Tensor<int64_t>& t, const std::string& name) {
    print_shape(t, name);
    if (t.shape.size() != 2) {
        std::cout << "(print_2d only supports rank=2)\n";
        return;
    }
    const int64_t H = t.shape[0], W = t.shape[1];
    for (int64_t i = 0; i < H; ++i) {
        std::cout << "  ";
        for (int64_t j = 0; j < W; ++j) {
            std::cout << t.data[i * W + j] << (j + 1 == W ? "" : ", ");
        }
        std::cout << "\n";
    }
}

int main() {
    // Example 1: 2D input [4, 6], split on axis=1 into 3 outputs (equal split)
    // D=6, K=3 => [2,2,2]
    auto x1 = make_tensor_i64({ 4, 6 });
    print_2d_i64(x1, "x1");

    auto outs1 = onnx_split(x1, /*axis=*/1, /*num_outputs=*/3, /*split_opt=*/nullptr);
    for (size_t i = 0; i < outs1.size(); ++i) {
        print_2d_i64(outs1[i], "outs1[" + std::to_string(i) + "]");
    }

    // Example 2: 2D input [4, 7], split on axis=1 into 3 outputs (no split provided)
    // D=7, K=3 => [3,2,2] (first remainder chunk bigger)
    auto x2 = make_tensor_i64({ 4, 7 });
    print_2d_i64(x2, "x2");

    auto outs2 = onnx_split(x2, /*axis=*/1, /*num_outputs=*/3, /*split_opt=*/nullptr);
    for (size_t i = 0; i < outs2.size(); ++i) {
        print_2d_i64(outs2[i], "outs2[" + std::to_string(i) + "]");
    }

    // Example 3: explicit split sizes on negative axis
    // input [2, 3, 10], axis=-1 => axis=2
    // split = [4, 6] => outputs shapes [2,3,4] and [2,3,6]
    Tensor<int64_t> x3 = make_tensor_i64({ 2, 3, 10 });
    print_shape(x3, "x3");

    std::vector<int64_t> split3 = { 4, 6 };
    auto outs3 = onnx_split(x3, /*axis=*/-1, /*num_outputs=*/2, &split3);
    for (size_t i = 0; i < outs3.size(); ++i) {
        print_shape(outs3[i], "outs3[" + std::to_string(i) + "]");
    }

    return 0;
}
