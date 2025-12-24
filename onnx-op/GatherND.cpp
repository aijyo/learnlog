// gathernd_demo.cpp
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// ------------------------------ GatherND (ONNX) ------------------------------
//
// Let:
//   data shape    = [D0, D1, ..., D(R-1)]          (rank R)
//   indices shape = [I0, I1, ..., I(Q-2), K]       (rank Q, K = indices.shape[-1])
//   batch_dims    = b
//
// Constraints (aligned with onnx-mlir verify):
//   - R >= 1, Q >= 1
//   - 0 <= b < min(R, Q)
//   - 1 <= K <= R - b
//   - For i in [0, b): data.shape[i] == indices.shape[i] if both static
//   - For each index component along K:
//       indices[..., i] is in [-data.shape[b+i], data.shape[b+i]-1]
//
// Output shape (aligned with onnx-mlir shape helper):
//   out.shape = indices.shape[0:b] + indices.shape[b:Q-1] + data.shape[b+K:R]
//
// Semantics:
//   For each position p in indices.shape[0:Q-1] (all dims except last):
//     - Read index tuple t[0..K-1] = indices[p, 0..K-1]
//     - Normalize negatives: if t[i] < 0, t[i] += data.shape[b+i]
//     - This selects a slice from data at prefix:
//         dataPrefix = [ batchCoords (first b coords from p), t[0..K-1] ]
//       and copies the trailing slice data.shape[b+K:R] to output.
//
// Notes:
// - This is a CPU reference implementation (row-major).
// - We assume row-major contiguous storage for easy slice copy.
// -----------------------------------------------------------------------------


// --------------------------- Principle Explanation Block ---------------------------
//
// GatherND performs "multi-dimensional advanced indexing".
// Each index tuple (length K) selects a location in the first (b+K) dimensions
// of `data` (with the first b dims treated as batch-aligned), and returns the
// remaining trailing slice data.shape[b+K:].
//
// This operator is used when you need to index data with *tuples* of coordinates
// rather than a single axis index (Gather) or per-element axis replacement
// (GatherElements).
//
// Typical use cases:
//   - Advanced indexing like NumPy/TF gather_nd (select arbitrary points/patches).
//   - Extracting values at (x,y) coordinates from feature maps.
//   - Selecting boxes/anchors/points by multi-d coordinates in detection pipelines.
//   - Dynamic routing / sparse selection where indices are coordinate tuples.
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

static int64_t product(const std::vector<int64_t>& v, int64_t begin, int64_t end) {
    int64_t p = 1;
    for (int64_t i = begin; i < end; ++i) p *= v[static_cast<size_t>(i)];
    return p;
}

static int64_t normalizeIndex(int64_t idx, int64_t dimSize) {
    // ONNX allows negative indices.
    if (idx < 0) idx += dimSize;
    if (idx < 0 || idx >= dimSize) {
        throw std::runtime_error("GatherND: index out of bounds after normalization. idx=" +
            std::to_string(idx) + ", dimSize=" + std::to_string(dimSize));
    }
    return idx;
}

static void verifyGatherNDConstraints(const std::vector<int64_t>& dataShape,
    const std::vector<int64_t>& indicesShape,
    int64_t b) {
    const int64_t R = static_cast<int64_t>(dataShape.size());
    const int64_t Q = static_cast<int64_t>(indicesShape.size());
    if (R < 1) throw std::runtime_error("GatherND: data rank must be > 0.");
    if (Q < 1) throw std::runtime_error("GatherND: indices rank must be > 0.");

    const int64_t minRQ = std::min(R, Q);
    if (b < 0 || b >= minRQ) {
        throw std::runtime_error("GatherND: batch_dims out of range.");
    }

    const int64_t K = indicesShape[static_cast<size_t>(Q - 1)];
    if (K < 1) throw std::runtime_error("GatherND: indices.shape[-1] must be >= 1.");
    if (K > R - b) throw std::runtime_error("GatherND: indices.shape[-1] must be <= dataRank - batch_dims.");

    // Check first b dims match if known (runtime they are known).
    for (int64_t i = 0; i < b; ++i) {
        if (dataShape[static_cast<size_t>(i)] != indicesShape[static_cast<size_t>(i)]) {
            throw std::runtime_error("GatherND: batch dim mismatch at dim " + std::to_string(i));
        }
    }
}

static std::vector<int64_t> computeGatherNDOutputShape(const std::vector<int64_t>& dataShape,
    const std::vector<int64_t>& indicesShape,
    int64_t b) {
    const int64_t R = static_cast<int64_t>(dataShape.size());
    const int64_t Q = static_cast<int64_t>(indicesShape.size());
    const int64_t K = indicesShape[static_cast<size_t>(Q - 1)];

    // out = indices[:b] + indices[b:-1] + data[b+K:]
    std::vector<int64_t> outShape;
    outShape.reserve(static_cast<size_t>(R + Q - K - 1 - b));

    for (int64_t i = 0; i < b; ++i) outShape.push_back(indicesShape[static_cast<size_t>(i)]);
    for (int64_t i = b; i < Q - 1; ++i) outShape.push_back(indicesShape[static_cast<size_t>(i)]);
    for (int64_t i = b + K; i < R; ++i) outShape.push_back(dataShape[static_cast<size_t>(i)]);

    return outShape;
}

template <typename T>
Tensor<T> onnxGatherND(const Tensor<T>& data,
    const Tensor<int64_t>& indices,
    int64_t batch_dims) {
    // Verify constraints (runtime version of onnx-mlir verify).
    verifyGatherNDConstraints(data.shape, indices.shape, batch_dims);

    const int64_t R = data.rank();
    const int64_t Q = indices.rank();
    const int64_t b = batch_dims;
    const int64_t K = indices.shape[static_cast<size_t>(Q - 1)];

    // Compute output shape (aligned with onnx-mlir shape helper).
    Tensor<T> out(computeGatherNDOutputShape(data.shape, indices.shape, b));

    // Precompute strides.
    auto dataStrides = computeStridesRowMajor(data.shape);
    auto idxStrides = computeStridesRowMajor(indices.shape);

    // The trailing slice size = product(data.shape[b+K:])
    const int64_t sliceStartDim = b + K;
    const int64_t sliceSize = (sliceStartDim < R) ? product(data.shape, sliceStartDim, R) : 1;

    // Iterate over all positions in indices excluding the last dim (K).
    // prefixRank = Q - 1, prefixShape = indices.shape[0:Q-1]
    std::vector<int64_t> prefixShape(indices.shape.begin(), indices.shape.end() - 1);
    const int64_t prefixNumel = (Q > 1) ? product(prefixShape, 0, static_cast<int64_t>(prefixShape.size())) : 1;

    std::vector<int64_t> pIdx;                 // multi-index for prefix
    std::vector<int64_t> dataPrefixIdx;        // length sliceStartDim (b+K)
    dataPrefixIdx.resize(static_cast<size_t>(sliceStartDim));

    // Output is arranged as: outPrefixShape (= indices[:b] + indices[b:-1]) then trailing slice dims.
    // The number of "prefix items" in output equals prefixNumel.
    // Each prefix item corresponds to one selected slice of length sliceSize.
    for (int64_t pLinear = 0; pLinear < prefixNumel; ++pLinear) {
        unravelIndexRowMajor(pLinear, prefixShape, pIdx);

        // 1) Copy batch coords (first b dims) from pIdx to dataPrefixIdx[0:b]
        for (int64_t i = 0; i < b; ++i) {
            dataPrefixIdx[static_cast<size_t>(i)] = pIdx[static_cast<size_t>(i)];
        }

        // 2) Read index tuple of length K from indices at position pIdx
        // indices offset for tuple element i is: indices[pIdx..., i]
        // We'll fetch K values and map them to data dims [b .. b+K-1]
        for (int64_t i = 0; i < K; ++i) {
            std::vector<int64_t> fullIdx = pIdx;
            fullIdx.push_back(i); // last dim selects which component in the tuple
            const int64_t off = ravelIndexRowMajor(fullIdx, idxStrides);

            int64_t t = indices.data[static_cast<size_t>(off)];
            const int64_t axisDim = data.shape[static_cast<size_t>(b + i)];
            t = normalizeIndex(t, axisDim);
            dataPrefixIdx[static_cast<size_t>(b + i)] = t;
        }

        // 3) Compute base offset in data for this slice (suffix indices are all zero)
        // dataBase = sum_{d<sliceStartDim} dataPrefixIdx[d] * dataStrides[d]
        int64_t dataBase = 0;
        for (int64_t d = 0; d < sliceStartDim; ++d) {
            dataBase += dataPrefixIdx[static_cast<size_t>(d)] * dataStrides[static_cast<size_t>(d)];
        }

        // 4) Compute base offset in output for this prefix item.
        // In row-major, we can treat output as [prefixNumel, sliceSize] (flattened view).
        const int64_t outBase = pLinear * sliceSize;

        // 5) Copy the trailing slice (contiguous in row-major).
        for (int64_t t = 0; t < sliceSize; ++t) {
            out.data[static_cast<size_t>(outBase + t)] =
                data.data[static_cast<size_t>(dataBase + t)];
        }
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
        // ------------------------- Example 1 (b=0, returns slices) -------------------------
        // data shape: (2, 3, 4)
        // indices shape: (2, 2)  => K=2 (last dim=2), prefixShape=(2)
        // b=0
        //
        // Each index tuple selects data[a, b, :] (a from dim0, b from dim1),
        // and returns trailing slice dim2 (size 4).
        //
        // Output shape:
        //   indices[:b]=[] + indices[b:-1]=indices[:1]=(2) + data[b+K:]=data[2:]=(4)
        // => out shape = (2, 4)
        Tensor<float> data1({ 2, 3, 4 });
        for (int64_t i = 0; i < data1.numel(); ++i) data1.data[static_cast<size_t>(i)] = (float)i;

        Tensor<int64_t> indices1({ 2, 2 });
        // Two tuples:
        // p=0 -> (0, 2) selects data[0,2,:] = [8,9,10,11]
        // p=1 -> (1, 1) selects data[1,1,:] = [16,17,18,19]
        indices1.data = { 0, 2,
                         1, 1 };

        auto out1 = onnxGatherND<float>(data1, indices1, /*batch_dims=*/0);

        std::cout << "Example1: data(2,3,4), indices(2,2), b=0 => out(2,4)\n";
        printTensor(indices1, "indices1");
        printTensor(out1, "out1", 32);

        // ------------------------- Example 2 (b=1, batch-aligned, returns scalars) -------------------------
        // data shape: (2, 3, 4)
        // indices shape: (2, 2, 2) => Q=3, K=2, b=1
        // First dim is batch: indices.shape[0]==data.shape[0]==2
        //
        // For each batch, each prefix position selects (dim1, dim2) and returns scalar
        // because b+K = 3 == dataRank => no trailing dims.
        //
        // Output shape:
        //   indices[:b]=indices[:1]=(2)
        // + indices[b:-1]=indices[1:2]=(2)
        // + data[b+K:]=data[3:]=()
        // => out shape = (2,2)
        Tensor<int64_t> indices2({ 2, 2, 2 });
        // batch 0: tuples [(0,3), (2,1)] => data[0,0,3]=3, data[0,2,1]=9
        // batch 1: tuples [(1,0), (0,2)] => data[1,1,0]=16, data[1,0,2]=14
        indices2.data = {
          0, 3,   2, 1,
          1, 0,   0, 2
        };

        auto out2 = onnxGatherND<float>(data1, indices2, /*batch_dims=*/1);

        std::cout << "\nExample2: data(2,3,4), indices(2,2,2), b=1 => out(2,2)\n";
        printTensor(indices2, "indices2");
        printTensor(out2, "out2", 16);

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
