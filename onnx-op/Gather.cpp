// gather_demo.cpp
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// ------------------------------- Gather (ONNX) -------------------------------
//
// Returns out = Gather(data, indices, axis).
//
// Shape rule (matches onnx-mlir shape helper):
//   data shape:    [D0, D1, ..., D(axis), ..., D(n-1)]  (rank = n)
//   indices shape: [I0, I1, ..., I(m-1)]                (rank = m)
//   out shape:     [D0, ..., D(axis-1), I0, ..., I(m-1), D(axis+1), ..., D(n-1)]
//
// Index mapping:
//   Let outIdx be split into three parts:
//     outPrefix: dims before axis (length = axis)
//     outMid:    dims corresponding to indices (length = m)
//     outSuffix: dims after axis (length = n-axis-1)
//
//   g = indices[outMid]    // gather index
//   if g < 0: g += D(axis) // ONNX supports negative indices (Python-like)
//   Require 0 <= g < D(axis)
//
//   dataIdx = [outPrefix, g, outSuffix]
//   out[outIdx] = data[dataIdx]
//
// Notes:
// - This is a straightforward CPU reference implementation.
// - Row-major (C-contiguous) storage.
// - indices are int64_t.
// -----------------------------------------------------------------------------


// --------------------------- Principle Explanation Block ---------------------------
//
// Gather is an "index-based select" operator.
// It selects elements/slices from `data` along a chosen `axis` according to `indices`,
// and expands the output rank by "inserting" the shape of `indices` at that axis.
// Unlike Slice (range-based), Gather is sparse / arbitrary indexing.
// Unlike Take (flattened indexing), Gather indexes along a specific dimension.
//
// This operator is widely used for:
//   - Embedding lookup (word id -> embedding vector).
//   - Reordering / permuting by an index table.
//   - Selecting top-k elements after TopK/ArgMax.
//   - Dynamic routing / mixture-of-experts token dispatch (index-driven).
//   - Building masks or extracting diagonal / banded structures (with crafted indices).
//
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
            n *= (d == 0 ? 0 : d);
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
    // Row-major strides: stride[i] = product(shape[i+1..end-1])
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static void unravelIndexRowMajor(int64_t linear,
    const std::vector<int64_t>& shape,
    std::vector<int64_t>& outIdx) {
    // Convert linear index -> multi-d index (row-major)
    outIdx.resize(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        int64_t dim = shape[i];
        if (dim == 0) { outIdx[i] = 0; continue; }
        outIdx[i] = linear % dim;
        linear /= dim;
    }
}

static int64_t ravelIndexRowMajor(const std::vector<int64_t>& idx,
    const std::vector<int64_t>& strides) {
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
    return off;
}

static int64_t normalizeAxis(int64_t axis, int64_t rank) {
    // onnx-mlir: axis in [-rank, rank-1]
    if (axis < -rank || axis >= rank) {
        throw std::runtime_error("Gather: axis out of range. axis=" + std::to_string(axis) +
            ", rank=" + std::to_string(rank));
    }
    if (axis < 0) axis += rank;
    return axis;
}

static int64_t normalizeIndex(int64_t idx, int64_t dimSize) {
    // ONNX: negative indices count from the end
    if (idx < 0) idx += dimSize;
    if (idx < 0 || idx >= dimSize) {
        throw std::runtime_error("Gather: index out of bounds after normalization. idx=" +
            std::to_string(idx) + ", dimSize=" + std::to_string(dimSize));
    }
    return idx;
}

static std::vector<int64_t> computeGatherOutputShape(const std::vector<int64_t>& dataShape,
    const std::vector<int64_t>& indicesShape,
    int64_t axis) {
    // Output shape = dataShape[:axis] + indicesShape + dataShape[axis+1:]
    std::vector<int64_t> outShape;
    outShape.reserve(dataShape.size() - 1 + indicesShape.size());
    for (int64_t i = 0; i < axis; ++i) outShape.push_back(dataShape[static_cast<size_t>(i)]);
    for (auto d : indicesShape) outShape.push_back(d);
    for (int64_t i = axis + 1; i < static_cast<int64_t>(dataShape.size()); ++i)
        outShape.push_back(dataShape[static_cast<size_t>(i)]);
    return outShape;
}

template <typename T>
Tensor<T> onnxGather(const Tensor<T>& data,
    const Tensor<int64_t>& indices,
    int64_t axis) {
    const int64_t dataRank = data.rank();
    const int64_t idxRank = indices.rank();
    if (dataRank == 0) throw std::runtime_error("Gather: data rank must be >= 1.");

    axis = normalizeAxis(axis, dataRank);

    // Compute output shape (matches onnx-mlir computeShape()).
    Tensor<T> out(computeGatherOutputShape(data.shape, indices.shape, axis));

    // Precompute strides for data, indices, output.
    auto dataStrides = computeStridesRowMajor(data.shape);
    auto idxStrides = computeStridesRowMajor(indices.shape);
    auto outStrides = computeStridesRowMajor(out.shape);

    const int64_t axisDim = data.shape[static_cast<size_t>(axis)];

    // For each output element, map to indices -> gather index -> data offset.
    std::vector<int64_t> outIdx;     // out multi-index
    std::vector<int64_t> idxMid;     // indices multi-index (length idxRank)
    std::vector<int64_t> dataIdx;    // data multi-index (length dataRank)
    dataIdx.resize(static_cast<size_t>(dataRank));

    const int64_t outNumel = out.numel();
    for (int64_t outLinear = 0; outLinear < outNumel; ++outLinear) {
        unravelIndexRowMajor(outLinear, out.shape, outIdx);

        // Split outIdx into prefix / mid / suffix.
        // prefix: [0 .. axis-1]
        // mid:    [axis .. axis+idxRank-1] in out index space
        // suffix: remaining dims
        idxMid.assign(static_cast<size_t>(idxRank), 0);

        // Build dataIdx prefix
        for (int64_t i = 0; i < axis; ++i) {
            dataIdx[static_cast<size_t>(i)] = outIdx[static_cast<size_t>(i)];
        }

        // Build idxMid from outIdx
        for (int64_t j = 0; j < idxRank; ++j) {
            idxMid[static_cast<size_t>(j)] = outIdx[static_cast<size_t>(axis + j)];
        }

        // Load gather index g = indices[idxMid]
        int64_t idxOff = 0;
        if (idxRank > 0) {
            idxOff = ravelIndexRowMajor(idxMid, idxStrides);
        }
        else {
            // Scalar indices: rank 0, only one value.
            idxOff = 0;
        }
        int64_t g = indices.data[static_cast<size_t>(idxOff)];
        g = normalizeIndex(g, axisDim);
        dataIdx[static_cast<size_t>(axis)] = g;

        // Build dataIdx suffix from outIdx
        // out suffix starts at (axis + idxRank) in out shape
        for (int64_t i = axis + 1; i < dataRank; ++i) {
            int64_t outPos = i - 1 + idxRank; // mapping from data dim -> out dim
            dataIdx[static_cast<size_t>(i)] = outIdx[static_cast<size_t>(outPos)];
        }

        const int64_t dataOff = ravelIndexRowMajor(dataIdx, dataStrides);
        out.data[static_cast<size_t>(outLinear)] = data.data[static_cast<size_t>(dataOff)];
    }

    return out;
}

// Simple pretty printer (prints flat data and shape).
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
        // indices shape: (3)
        // axis = 1
        // out shape: (2, 3)
        Tensor<int32_t> data1({ 2, 4 });
        // Fill data1 with 0..7:
        for (int64_t i = 0; i < data1.numel(); ++i) data1.data[static_cast<size_t>(i)] = (int32_t)i;
        // indices = [3, 1, 1]
        Tensor<int64_t> idx1({ 3 });
        idx1.data = { 3, 1, 1 };

        auto out1 = onnxGather<int32_t>(data1, idx1, /*axis=*/1);
        std::cout << "Example1: data(2,4), indices(3), axis=1 => out(2,3)\n";
        printTensor(data1, "data1");
        printTensor(idx1, "idx1");
        printTensor(out1, "out1");
        // Expected out1 rows:
        // row0 picks data1[0,3], data1[0,1], data1[0,1] => [3,1,1]
        // row1 picks data1[1,3], data1[1,1], data1[1,1] => [7,5,5]

        // ------------------------- Example 2 -------------------------
        // data shape: (2, 3, 4)
        // indices shape: (2, 2)
        // axis = 1
        // out shape: (2, 2, 2, 4)   (replace dim=3 with (2,2))
        Tensor<float> data2({ 2, 3, 4 });
        for (int64_t i = 0; i < data2.numel(); ++i) data2.data[static_cast<size_t>(i)] = (float)i;

        Tensor<int64_t> idx2({ 2, 2 });
        // indices:
        // [[ 2, 0],
        //  [ 1,-1]]   (-1 means last element along axis dim)
        idx2.data = { 2, 0, 1, -1 };

        auto out2 = onnxGather<float>(data2, idx2, /*axis=*/1);
        std::cout << "\nExample2: data(2,3,4), indices(2,2), axis=1 => out(2,2,2,4)\n";
        printTensor(idx2, "idx2");
        printTensor(out2, "out2", /*maxPrint=*/48);

        // ------------------------- Example 3 -------------------------
        // Negative axis: axis = -1 (last dimension)
        // data shape: (2, 3, 4)
        // indices shape: (2)
        // out shape: (2, 3, 2)  (replace last dim=4 with indices(2))
        Tensor<int64_t> idx3({ 2 });
        idx3.data = { 0, 3 };
        auto out3 = onnxGather<float>(data2, idx3, /*axis=*/-1);
        std::cout << "\nExample3: data(2,3,4), indices(2), axis=-1 => out(2,3,2)\n";
        printTensor(idx3, "idx3");
        printTensor(out3, "out3", /*maxPrint=*/48);

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
