#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// --------------------------- Minimal Tensor ---------------------------
//
// A simple row-major tensor container.
// - shape: int64 dims
// - data: flat buffer, size == product(shape)
//
// NOTE: This is a standalone helper for demonstrating ONNX ReverseSequence.
// ---------------------------------------------------------------------
template <typename T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<T> data;

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t numel() const {
        if (shape.empty()) return 1;
        int64_t n = 1;
        for (int64_t d : shape) {
            if (d < 0) throw std::runtime_error("Tensor: negative dimension.");
            if (d == 0) return 0;
            if (n > INT64_MAX / d) throw std::runtime_error("Tensor: numel overflow.");
            n *= d;
        }
        return n;
    }
};

static std::string ShapeToString(const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        out += std::to_string(s[i]);
        if (i + 1 < s.size()) out += ", ";
    }
    out += "]";
    return out;
}

static void ValidateTensorBuffer(const Tensor<float>& t, const std::string& name) {
    if (t.numel() != static_cast<int64_t>(t.data.size())) {
        throw std::runtime_error(name + ": buffer size != numel. shape=" +
            ShapeToString(t.shape) + " numel=" +
            std::to_string(t.numel()) + " data_size=" +
            std::to_string(t.data.size()));
    }
}

static std::vector<int64_t> ComputeStridesRowMajor(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static int64_t OffsetFromIndices(const std::vector<int64_t>& idx,
    const std::vector<int64_t>& strides) {
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
    return off;
}

// --------------------------- ONNX ReverseSequence (Standalone) ---------------------------
//
// ReverseSequence reverses variable-length slices along `time_axis` for each batch.
//
// Inputs:
// - input: tensor with rank >= 2
// - sequence_lens: 1D tensor/array, length == input.shape[batch_axis]
//
// Attributes:
// - batch_axis: which axis represents the batch dimension
// - time_axis: which axis represents the time/sequence dimension
//
// Semantics:
// For each batch index b:
//   L = sequence_lens[b]
//   for t in [0, time_dim):
//     if t < L:
//        output[t,b,...] = input[L-1-t, b, ...]   (reverse the valid prefix)
//     else:
//        output[t,b,...] = input[t, b, ...]       (keep the padded suffix)
//
// Output shape is identical to input shape.
//
// This matches onnx-mlir shape helper behavior (output dims from input),
// and extends it with the runtime data transformation.
// ---------------------------------------------------------------------

template <typename T>
static Tensor<T> OnnxReverseSequence(const Tensor<T>& input,
    const std::vector<int64_t>& sequence_lens,
    int64_t batch_axis = 1,
    int64_t time_axis = 0) {
    // Basic shape checks (similar spirit to onnx-mlir verify)
    if (input.rank() < 2)
        throw std::runtime_error("ReverseSequence: input must have rank >= 2.");

    const int64_t rank = input.rank();

    auto norm_axis = [&](int64_t a) -> int64_t {
        if (a < 0) a += rank;
        if (a < 0 || a >= rank)
            throw std::runtime_error("ReverseSequence: axis out of range.");
        return a;
        };

    batch_axis = norm_axis(batch_axis);
    time_axis = norm_axis(time_axis);

    if (sequence_lens.empty())
        throw std::runtime_error("ReverseSequence: sequence_lens must not be empty.");
    if (static_cast<int64_t>(sequence_lens.size()) != input.shape[batch_axis]) {
        throw std::runtime_error(
            "ReverseSequence: sequence_lens length must match input batch dimension. "
            "sequence_lens=" + std::to_string(sequence_lens.size()) +
            ", input.batch=" + std::to_string(input.shape[batch_axis]));
    }

    // Validate lens values
    const int64_t time_dim = input.shape[time_axis];
    for (int64_t b = 0; b < static_cast<int64_t>(sequence_lens.size()); ++b) {
        int64_t L = sequence_lens[b];
        if (L < 0)
            throw std::runtime_error("ReverseSequence: sequence_lens contains negative value.");
        // ONNX expects 0 <= L <= time_dim
        if (L > time_dim) {
            throw std::runtime_error(
                "ReverseSequence: sequence_lens[" + std::to_string(b) +
                "] exceeds time dimension. L=" + std::to_string(L) +
                ", time_dim=" + std::to_string(time_dim));
        }
    }

    // Create output (same shape)
    Tensor<T> output;
    output.shape = input.shape;
    output.data.resize(static_cast<size_t>(input.numel()));

    const auto strides = ComputeStridesRowMajor(input.shape);

    // Iterate over all output elements using linear index -> N-D index decoding.
    // Then map (t,b,...) to source time index accordingly.
    const int64_t total = static_cast<int64_t>(output.data.size());
    for (int64_t outLinear = 0; outLinear < total; ++outLinear) {
        // Decode linear index to multi-dimensional index
        std::vector<int64_t> idx(rank, 0);
        int64_t tmp = outLinear;
        for (int64_t i = 0; i < rank; ++i) {
            idx[i] = tmp / strides[i];
            tmp %= strides[i];
        }

        const int64_t b = idx[batch_axis];
        const int64_t t = idx[time_axis];
        const int64_t L = sequence_lens[b];

        std::vector<int64_t> srcIdx = idx;
        if (t < L) {
            srcIdx[time_axis] = (L - 1 - t);
        } // else keep t unchanged

        const int64_t inOff = OffsetFromIndices(srcIdx, strides);
        output.data[outLinear] = input.data[inOff];
    }

    return output;
}

// --------------------------- Demo Helpers ---------------------------
static void PrintTBC(const Tensor<float>& t, const std::string& name) {
    // Expect shape [T, B, C]
    std::cout << name << " shape=" << ShapeToString(t.shape) << "\n";
    if (t.shape.size() != 3) {
        std::cout << "(skip print: not rank-3)\n";
        return;
    }
    int64_t T = t.shape[0], B = t.shape[1], C = t.shape[2];
    for (int64_t b = 0; b < B; ++b) {
        std::cout << "b=" << b << ":\n";
        for (int64_t tt = 0; tt < T; ++tt) {
            std::cout << "  t=" << tt << ": ";
            for (int64_t c = 0; c < C; ++c) {
                // offset = ((t*B + b)*C + c)
                int64_t off = ((tt * B + b) * C + c);
                std::cout << t.data[off] << (c + 1 < C ? " " : "");
            }
            std::cout << "\n";
        }
    }
}

int main() {
    try {
        // Build an input tensor with shape [T=5, B=2, C=1]
        // Fill values as X[t,b,0] = 10*b + t for easy viewing.
        Tensor<float> X;
        X.shape = { 5, 2, 1 };
        X.data.resize(static_cast<size_t>(X.numel()));
        for (int64_t t = 0; t < 5; ++t) {
            for (int64_t b = 0; b < 2; ++b) {
                int64_t off = ((t * 2 + b) * 1 + 0);
                X.data[off] = static_cast<float>(10 * b + t);
            }
        }

        PrintTBC(X, "X");

        // sequence_lens length must match B (=2)
        // For b=0: L=3 => reverse t in [0,1,2]
        // For b=1: L=5 => reverse the whole time axis
        std::vector<int64_t> lens = { 3, 5 };

        auto Y = OnnxReverseSequence<float>(X, lens, /*batch_axis=*/1, /*time_axis=*/0);
        PrintTBC(Y, "Y (ReverseSequence)");

        std::cout << "Done.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
