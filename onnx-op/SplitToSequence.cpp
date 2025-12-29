#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

// ------------------------------ Tensor (row-major) ------------------------------
//
// Minimal N-D tensor container:
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
    if (axis < -rank || axis >= rank) throw std::runtime_error("SplitToSequence: axis out of range.");
    if (axis < 0) axis += rank;
    return axis;
}

// Split specification for runtime:
// - std::monostate: None
// - int64_t: scalar split (>0)
// - std::vector<int64_t>: explicit sizes (>=0)
using SplitSpec = std::variant<std::monostate, int64_t, std::vector<int64_t>>;

// Validate split spec similar to onnx-mlir verify.
static inline void verify_split_to_sequence(const std::vector<int64_t>& inputShape,
    int64_t axis,
    const SplitSpec& split,
    int64_t keepdims) {
    const int64_t rank = static_cast<int64_t>(inputShape.size());
    const int64_t ax = normalize_axis(axis, rank);
    const int64_t dimSize = inputShape[ax];

    if (std::holds_alternative<std::monostate>(split)) {
        if (keepdims < 0 || keepdims > 1)
            throw std::runtime_error("SplitToSequence: keepdims must be 0 or 1 when split is None.");
        (void)dimSize;
        return;
    }

    // split is specified: keepdims is ignored by spec/onnx-mlir
    if (std::holds_alternative<int64_t>(split)) {
        const int64_t scalar = std::get<int64_t>(split);
        if (scalar <= 0) throw std::runtime_error("SplitToSequence: split scalar must be > 0.");
        return;
    }

    const auto& sizes = std::get<std::vector<int64_t>>(split);
    int64_t sum = 0;
    for (int64_t s : sizes) {
        if (s < 0) throw std::runtime_error("SplitToSequence: split sizes must be >= 0.");
        sum += s;
    }
    // If dimSize is known (runtime always knows), enforce sum == dimSize.
    if (sum != dimSize) {
        throw std::runtime_error("SplitToSequence: sum(split) must equal axis dimension size.");
    }
}

// Copy a slice block along axis into an output tensor.
// This function assumes the output keeps the axis dimension (axis present).
// It copies for each "outer" prefix a contiguous block of (len * inner).
template <typename T>
static void copy_axis_segment_keep_axis(const Tensor<T>& input,
    Tensor<T>& output,
    int64_t axis,
    int64_t axisStart,
    int64_t len) {
    const int64_t rank = input.rank();
    const int64_t D = input.shape[axis];

    const auto inStrides = compute_strides_row_major(input.shape);
    const auto outStrides = compute_strides_row_major(output.shape);

    const int64_t inner = inStrides[axis];                 // product dims after axis
    const int64_t outer = input.numel() / (D * inner);     // product dims before axis
    const int64_t block = len * inner;

    for (int64_t o = 0; o < outer; ++o) {
        const int64_t inBase = o * (D * inner) + axisStart * inner;
        const int64_t outBase = o * (output.shape[axis] * inner); // axis dim in output is len
        std::copy_n(input.data.begin() + inBase, block, output.data.begin() + outBase);
    }

    (void)rank;
    (void)outStrides;
}

// Copy a single-index slice (len==1) along axis into an output tensor that REMOVES the axis.
// Output shape = input.shape with axis removed.
// For each outer prefix, we copy exactly `inner` values (since axis removed).
template <typename T>
static void copy_axis_index_remove_axis(const Tensor<T>& input,
    Tensor<T>& output,
    int64_t axis,
    int64_t axisIndex) {
    const int64_t D = input.shape[axis];
    const auto inStrides = compute_strides_row_major(input.shape);

    const int64_t inner = inStrides[axis];             // product dims after axis
    const int64_t outer = input.numel() / (D * inner); // product dims before axis

    for (int64_t o = 0; o < outer; ++o) {
        const int64_t inBase = o * (D * inner) + axisIndex * inner;
        const int64_t outBase = o * inner;
        std::copy_n(input.data.begin() + inBase, inner, output.data.begin() + outBase);
    }
}

// ------------------------------ SplitToSequence (runtime) ------------------------------
//
// Returns a sequence (vector) of tensors.
//
// Semantics aligned with onnx-mlir snippet:
// - If split is None:
//   length = dimSize
//   keepdims=1: each element keeps axis with size=1
//   keepdims=0: each element removes axis
// - If split is scalar (>0):
//   split axis into chunks of size=scalar, last chunk may be smaller
//   axis is kept in each element
// - If split is vector sizes:
//   split axis into given segment sizes, axis kept in each element
//
template <typename T>
std::vector<Tensor<T>> onnx_split_to_sequence(const Tensor<T>& input,
    int64_t axis,
    const SplitSpec& split,
    int64_t keepdims) {
    if (input.rank() <= 0) throw std::runtime_error("SplitToSequence: input must be ranked.");
    verify_split_to_sequence(input.shape, axis, split, keepdims);

    const int64_t rank = input.rank();
    const int64_t ax = normalize_axis(axis, rank);
    const int64_t dimSize = input.shape[ax];

    std::vector<Tensor<T>> seq;

    // Case 1: split is None
    if (std::holds_alternative<std::monostate>(split)) {
        seq.reserve(static_cast<size_t>(dimSize));

        for (int64_t i = 0; i < dimSize; ++i) {
            Tensor<T> out;
            if (keepdims == 1) {
                out.shape = input.shape;
                out.shape[ax] = 1;
                out.data.resize(out.numel());
                copy_axis_segment_keep_axis(input, out, ax, /*axisStart=*/i, /*len=*/1);
            }
            else {
                out.shape = input.shape;
                out.shape.erase(out.shape.begin() + ax);
                out.data.resize(out.numel());
                copy_axis_index_remove_axis(input, out, ax, /*axisIndex=*/i);
            }
            seq.push_back(std::move(out));
        }
        return seq;
    }

    // Helper: produce segment sizes
    std::vector<int64_t> segSizes;

    // Case 2: split is scalar
    if (std::holds_alternative<int64_t>(split)) {
        const int64_t scalar = std::get<int64_t>(split);
        // Cut into chunks of `scalar`, last chunk may be smaller.
        for (int64_t start = 0; start < dimSize; start += scalar) {
            const int64_t len = std::min<int64_t>(scalar, dimSize - start);
            segSizes.push_back(len);
        }
    }
    else {
        // Case 3: split is explicit vector
        segSizes = std::get<std::vector<int64_t>>(split);
    }

    // Build outputs: axis is kept for split specified cases
    seq.reserve(segSizes.size());
    int64_t axisStart = 0;
    for (size_t k = 0; k < segSizes.size(); ++k) {
        const int64_t len = segSizes[k];

        Tensor<T> out;
        out.shape = input.shape;
        out.shape[ax] = len;
        out.data.resize(out.numel());

        copy_axis_segment_keep_axis(input, out, ax, axisStart, len);

        seq.push_back(std::move(out));
        axisStart += len;
    }

    return seq;
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

int main() {
    // Input: [2, 3, 4] filled with 0..23
    auto x = make_tensor_i64({ 2, 3, 4 });
    print_shape(x, "x");

    // ------------------- Case A: split=None, keepdims=1 -------------------
    // axis=1 => dimSize=3 => sequence length = 3
    // each element shape: [2, 1, 4]
    {
        auto seq = onnx_split_to_sequence<int64_t>(x, /*axis=*/1, /*split=*/std::monostate{}, /*keepdims=*/1);
        std::cout << "\nCase A: split=None, keepdims=1, seq length = " << seq.size() << "\n";
        for (size_t i = 0; i < seq.size(); ++i) {
            print_shape(seq[i], "  seqA[" + std::to_string(i) + "]");
        }
    }

    // ------------------- Case B: split=None, keepdims=0 -------------------
    // each element shape: [2, 4] (axis removed)
    {
        auto seq = onnx_split_to_sequence<int64_t>(x, /*axis=*/1, /*split=*/std::monostate{}, /*keepdims=*/0);
        std::cout << "\nCase B: split=None, keepdims=0, seq length = " << seq.size() << "\n";
        for (size_t i = 0; i < seq.size(); ++i) {
            print_shape(seq[i], "  seqB[" + std::to_string(i) + "]");
        }
    }

    // ------------------- Case C: split=scalar -------------------
    // axis=2 => dimSize=4, scalar=3 => chunks: [3,1], seq length=2
    // each element keeps axis, shapes: [2,3,3] and [2,3,1]
    {
        SplitSpec splitScalar = int64_t(3);
        auto seq = onnx_split_to_sequence<int64_t>(x, /*axis=*/2, splitScalar, /*keepdims=*/0 /*ignored*/);
        std::cout << "\nCase C: split=scalar(3) on axis=2, seq length = " << seq.size() << "\n";
        for (size_t i = 0; i < seq.size(); ++i) {
            print_shape(seq[i], "  seqC[" + std::to_string(i) + "]");
        }
    }

    // ------------------- Case D: split=vector sizes -------------------
    // axis=-1 => axis=2, dimSize=4, split=[1,0,3] => seq length=3
    // shapes: [2,3,1], [2,3,0], [2,3,3]
    {
        SplitSpec splitVec = std::vector<int64_t>{ 1, 0, 3 };
        auto seq = onnx_split_to_sequence<int64_t>(x, /*axis=*/-1, splitVec, /*keepdims=*/1 /*ignored*/);
        std::cout << "\nCase D: split=vector [1,0,3] on axis=-1, seq length = " << seq.size() << "\n";
        for (size_t i = 0; i < seq.size(); ++i) {
            print_shape(seq[i], "  seqD[" + std::to_string(i) + "]");
        }
    }

    return 0;
}
