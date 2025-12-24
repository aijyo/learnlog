// gather_elements_demo.cpp
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// -------------------------- GatherElements (ONNX) ---------------------------
//
// Output shape (matches onnx-mlir shape helper):
//   output.shape = indices.shape
//
// Verification rules (matches onnx-mlir verify intent):
//   - rank(data) > 0
//   - rank(indices) == rank(data)
//   - axis in [-r, r-1], normalize axis if negative
//   - indices values must be in [-s, s-1] where s = data.shape[axis]
//     (negative indices are allowed, Python-style)
//
// Semantics (element-wise gather):
//   For each output element at multi-index idx:
//     j = indices[idx]
//     if j < 0: j += s
//     out[idx] = data[idx with idx[axis] replaced by j]
//
// Key difference vs Gather:
//   - Gather replaces the axis dimension by indices.shape and changes rank.
//   - GatherElements keeps rank and gathers element-by-element using a full
//     indices tensor with the same rank as data.
//
// ---------------------------------------------------------------------------


// --------------------------- Principle Explanation Block ---------------------------
//
// GatherElements is a "per-element indexing" operator along a given axis.
// It is typically used when you have an indices tensor that provides, for every
// output position, which element to select along `axis`.
//
// Common patterns:
//   - Selecting per-position best candidate after ArgMax/TopK where indices has
//     the same shape as the desired output.
//   - Implementing advanced indexing behavior similar to NumPy take_along_axis.
//   - Reordering elements differently at each position (position-dependent gather),
//     which Gather (slice-based) cannot express.
//
// Why it exists:
//   - Provides a precise and efficient IR-level primitive for per-element index
//     selection. This avoids building complicated loop logic from lower-level ops.
// -----------------------------------------------------------------------------

template <typename T>
struct Tensor {
    std::vector<int64_t> shape; // row-major
    std::vector<T> data;

    Tensor() = default;

    explicit Tensor(std::vector<int64_t> s) : shape(std::move(s)) {
        int64_t n = 1;
        for (int64_t d : shape) {
            if (d < 0) throw std::runtime_error("Tensor: negative dim is not allowed at runtime.");
            n *= d;
        }
        data.resize(static_cast<size_t>(n));
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

static std::vector<int64_t> computeStridesRowMajor(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static void unravelIndexRowMajor(int64_t linear,
    const std::vector<int64_t>& shape,
    std::vector<int64_t>& outIdx) {
    outIdx.resize(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        int64_t dim = shape[static_cast<size_t>(i)];
        outIdx[i] = (dim == 0) ? 0 : (linear % dim);
        linear = (dim == 0) ? 0 : (linear / dim);
    }
}

static int64_t ravelIndexRowMajor(const std::vector<int64_t>& idx,
    const std::vector<int64_t>& strides) {
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
    return off;
}

static int64_t normalizeAxis(int64_t axis, int64_t rank) {
    // Must be in [-rank, rank-1]
    if (axis < -rank || axis >= rank) {
        throw std::runtime_error("GatherElements: axis out of range. axis=" + std::to_string(axis) +
            ", rank=" + std::to_string(rank));
    }
    if (axis < 0) axis += rank;
    return axis;
}

static int64_t normalizeIndexInAxis(int64_t j, int64_t axisDim) {
    // Indices expected in [-axisDim, axisDim-1]
    // Negative means counting from the end.
    if (j < 0) j += axisDim;
    if (j < 0 || j >= axisDim) {
        throw std::runtime_error("GatherElements: index out of bounds after normalization. j=" +
            std::to_string(j) + ", axisDim=" + std::to_string(axisDim));
    }
    return j;
}

template <typename T>
Tensor<T> onnxGatherElements(const Tensor<T>& data,
    const Tensor<int64_t>& indices,
    int64_t axis) {
    // Verify rank constraints (like onnx-mlir).
    const int64_t r = data.rank();
    if (r < 1) throw std::runtime_error("GatherElements: data rank must be > 0.");
    if (indices.rank() != r) throw std::runtime_error("GatherElements: indices rank must equal data rank.");

    axis = normalizeAxis(axis, r);

    const int64_t axisDim = data.shape[static_cast<size_t>(axis)];
    if (axisDim <= 0) throw std::runtime_error("GatherElements: data dim at axis must be > 0 at runtime.");

    // Output shape equals indices shape (onnx-mlir shape helper).
    Tensor<T> out(indices.shape);

    // Strides for data and indices/out (same shape for indices and out).
    auto dataStrides = computeStridesRowMajor(data.shape);
    auto idxStrides = computeStridesRowMajor(indices.shape);
    auto outStrides = idxStrides;

    // Optional: you can enforce indices.shape == data.shape if desired,
    // but ONNX spec allows indices dims <= data dims in non-axis dimensions.
    // onnx-mlir snippet mainly checks rank; we keep runtime general.
    // We DO require that all non-axis indices coordinates fit data dims:
    for (int64_t d = 0; d < r; ++d) {
        if (d == axis) continue;
        if (indices.shape[static_cast<size_t>(d)] > data.shape[static_cast<size_t>(d)]) {
            throw std::runtime_error("GatherElements: indices dim exceeds data dim at non-axis dim " +
                std::to_string(d));
        }
    }

    std::vector<int64_t> outIdx;
    std::vector<int64_t> dataIdx;
    dataIdx.resize(static_cast<size_t>(r));

    const int64_t outNumel = out.numel();
    for (int64_t linear = 0; linear < outNumel; ++linear) {
        unravelIndexRowMajor(linear, out.shape, outIdx);

        // Read gather index for this element.
        const int64_t idxOff = ravelIndexRowMajor(outIdx, idxStrides);
        int64_t j = indices.data[static_cast<size_t>(idxOff)];
        j = normalizeIndexInAxis(j, axisDim);

        // Build dataIdx = outIdx, but replace axis with j.
        for (int64_t d = 0; d < r; ++d) dataIdx[static_cast<size_t>(d)] = outIdx[static_cast<size_t>(d)];
        dataIdx[static_cast<size_t>(axis)] = j;

        // Load from data.
        const int64_t dataOff = ravelIndexRowMajor(dataIdx, dataStrides);
        out.data[static_cast<size_t>(linear)] = data.data[static_cast<size_t>(dataOff)];
    }

    return out;
}

template <typename T>
static void printTensor(const Tensor<T>& t, const std::string& name, int64_t maxPrint = 64) {
    std::cout << name << " shape=(";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        std::cout << t.shape[i] << (i + 1 == t.shape.size() ? "" : ",");
    }
    std::cout << "), numel=" << t.numel() << "\n  data: ";
    int64_t n = std::min<int64_t>(t.numel(), maxPrint);
    for (int64_t i = 0; i < n; ++i) {
        std::cout << t.data[static_cast<size_t>(i)] << (i + 1 == n ? "" : " ");
    }
    if (t.numel() > maxPrint) std::cout << "...";
    std::cout << "\n";
}

int main() {
    try {
        // ------------------------- Example 1 -------------------------
        // data shape: (2, 4)
        // indices shape: (2, 4) (same rank, same shape)
        // axis = 1: for each position (i, j), indices[i,j] selects a column in data row i.
        //
        // data:
        //   [0 1 2 3]
        //   [4 5 6 7]
        //
        // indices:
        //   [0 0 1 3]
        //   [3 2 1 0]
        //
        // out:
        //   row0: take columns [0,0,1,3] => [0,0,1,3]
        //   row1: take columns [3,2,1,0] => [7,6,5,4]
        Tensor<int32_t> data1({ 2, 4 });
        for (int64_t i = 0; i < data1.numel(); ++i) data1.data[static_cast<size_t>(i)] = (int32_t)i;

        Tensor<int64_t> idx1({ 2, 4 });
        idx1.data = { 0, 0, 1, 3,
                     3, 2, 1, 0 };

        auto out1 = onnxGatherElements<int32_t>(data1, idx1, /*axis=*/1);

        std::cout << "Example1: data(2,4), indices(2,4), axis=1 => out(2,4)\n";
        printTensor(data1, "data1");
        printTensor(idx1, "idx1");
        printTensor(out1, "out1");

        // ------------------------- Example 2 -------------------------
        // Negative indices and negative axis:
        // axis = -1 means last dimension (axis=1 here).
        // indices in [-4,3], e.g. -1 means last element.
        Tensor<int64_t> idx2({ 2, 4 });
        idx2.data = { -1, -2, 0, 1,
                     1,  0, -3, -4 }; // -4 means first element when axisDim=4

        auto out2 = onnxGatherElements<int32_t>(data1, idx2, /*axis=*/-1);

        std::cout << "\nExample2: data(2,4), indices(2,4), axis=-1 with negatives => out(2,4)\n";
        printTensor(idx2, "idx2");
        printTensor(out2, "out2");

        // ------------------------- Example 3 -------------------------
        // Higher-rank example:
        // data shape: (2, 3, 4)
        // indices shape: (2, 3, 4)
        // axis = 2 (last dim)
        Tensor<float> data3({ 2, 3, 4 });
        for (int64_t i = 0; i < data3.numel(); ++i) data3.data[static_cast<size_t>(i)] = (float)i;

        Tensor<int64_t> idx3({ 2, 3, 4 });
        // For each position (b,m,n), pick along last dim by idx3[b,m,n].
        // We'll just alternate 0 and 3 for demonstration.
        for (int64_t i = 0; i < idx3.numel(); ++i) idx3.data[static_cast<size_t>(i)] = (i % 2 == 0) ? 0 : 3;

        auto out3 = onnxGatherElements<float>(data3, idx3, /*axis=*/2);

        std::cout << "\nExample3: data(2,3,4), indices(2,3,4), axis=2 => out(2,3,4)\n";
        printTensor(idx3, "idx3", 48);
        printTensor(out3, "out3", 48);

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
