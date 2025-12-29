#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

// ------------------------------ TinyTensor ------------------------------
//
// Minimal dense tensor for runtime demo.
// - Row-major contiguous storage.
// - Supports int64 indices tensor and float data tensor in a simple way.
// - Focus: correctness & clarity rather than performance.
//
template <typename T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<T> data;

    Tensor() = default;

    explicit Tensor(std::vector<int64_t> s)
        : shape(std::move(s)), data(static_cast<size_t>(numelFromShape(shape))) {
    }

    Tensor(std::vector<int64_t> s, std::vector<T> d)
        : shape(std::move(s)), data(std::move(d)) {
        if (static_cast<int64_t>(data.size()) != numel()) {
            throw std::runtime_error("Tensor: data size mismatch with shape.");
        }
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t dim(int64_t i) const {
        if (i < 0 || i >= rank()) throw std::runtime_error("dim: out of range");
        return shape[static_cast<size_t>(i)];
    }

    int64_t numel() const { return numelFromShape(shape); }

    static int64_t numelFromShape(const std::vector<int64_t>& s) {
        if (s.empty()) return 1;
        int64_t n = 1;
        for (int64_t v : s) {
            if (v < 0) throw std::runtime_error("Negative dims not supported in this demo.");
            n *= v;
        }
        return n;
    }

    // Compute row-major linear offset.
    int64_t offset(const std::vector<int64_t>& idx) const {
        if (static_cast<int64_t>(idx.size()) != rank())
            throw std::runtime_error("offset: rank mismatch");
        int64_t off = 0;
        int64_t stride = 1;
        for (int64_t i = rank() - 1; i >= 0; --i) {
            int64_t d = dim(i);
            int64_t v = idx[static_cast<size_t>(i)];
            if (v < 0 || v >= d) throw std::runtime_error("offset: index out of bounds");
            off += v * stride;
            stride *= d;
        }
        return off;
    }
};

static inline int64_t normalizeAxis(int64_t axis, int64_t rank) {
    if (axis < -rank || axis >= rank) throw std::runtime_error("axis out of range");
    if (axis < 0) axis += rank;
    return axis;
}

static inline int64_t normalizeIndex(int64_t idx, int64_t dimSize) {
    // Allow [-dimSize, dimSize-1]
    if (idx < -dimSize || idx >= dimSize) throw std::runtime_error("index out of range");
    if (idx < 0) idx += dimSize;
    return idx;
}

static inline void requireSameShape(const std::vector<int64_t>& a,
    const std::vector<int64_t>& b,
    const std::string& msg) {
    if (a != b) throw std::runtime_error(msg);
}

// --------------------------- ScatterElements (Verify) ---------------------------
//
// Checks based on onnx-mlir verify():
// - data rank >= 1
// - indices rank == data rank
// - updates rank == data rank
// - axis in [-r, r-1]
// - if data dim at axis is known, indices values in [-s, s-1]
//
static void VerifyScatterElements(const Tensor<float>& data,
    const Tensor<int64_t>& indices,
    const Tensor<float>& updates,
    int64_t axis) {
    int64_t r = data.rank();
    if (r < 1) throw std::runtime_error("ScatterElements: data rank must be > 0");
    if (indices.rank() != r) throw std::runtime_error("ScatterElements: indices rank mismatch");
    if (updates.rank() != r) throw std::runtime_error("ScatterElements: updates rank mismatch");

    axis = normalizeAxis(axis, r);

    requireSameShape(indices.shape, updates.shape,
        "ScatterElements: indices shape must equal updates shape");

    // The spec requires indices shape = updates shape, and both rank = data rank.
    // Note: indices/upd dims are allowed to be <= data dims, but onnx-mlir verify
    // doesn't enforce that here; it only validates indices values range if possible.

    // Validate indices values are in [-s, s-1] for data dimension at axis.
    int64_t s = data.dim(axis);
    // s must be known positive in this demo tensor.
    for (int64_t i = 0; i < indices.numel(); ++i) {
        (void)normalizeIndex(indices.data[static_cast<size_t>(i)], s);
    }
}

// --------------------------- ScatterND (Verify) ---------------------------
//
// Checks based on onnx-mlir verify():
// - data rank >= 1, indices rank >= 1
// - let q = rank(indices), k = indices.shape[q-1]
//   * updates rank must be: r + q - k - 1
//   * k <= r
// - if shapes are known:
//   * updates.shape[0:q-1] == indices.shape[0:q-1]
//   * updates.shape[q-1:] == data.shape[k:]
//
static void VerifyScatterND(const Tensor<float>& data,
    const Tensor<int64_t>& indices,
    const Tensor<float>& updates) {
    int64_t r = data.rank();
    int64_t q = indices.rank();
    int64_t u = updates.rank();
    if (r < 1) throw std::runtime_error("ScatterND: data rank must be > 0");
    if (q < 1) throw std::runtime_error("ScatterND: indices rank must be > 0");

    int64_t k = indices.dim(q - 1);
    if (k <= 0) throw std::runtime_error("ScatterND: indices.shape[-1] must be > 0");
    if (k > r) throw std::runtime_error("ScatterND: indices.shape[-1] must be <= rank(data)");

    int64_t expectedUpdatesRank = r + q - k - 1;
    if (u != expectedUpdatesRank)
        throw std::runtime_error("ScatterND: updates rank mismatch with formula");

    // updates.shape[0:q-1] == indices.shape[0:q-1]
    for (int64_t i = 0; i < q - 1; ++i) {
        if (updates.dim(i) != indices.dim(i))
            throw std::runtime_error("ScatterND: updates.shape[0:q-1] must match indices.shape[0:q-1]");
    }

    // updates.shape[q-1:] == data.shape[k:]
    // updates dims from (q-1) maps to data dims from k
    for (int64_t di = k, ui = (q - 1); di < r; ++di, ++ui) {
        if (updates.dim(ui) != data.dim(di))
            throw std::runtime_error("ScatterND: updates.shape[q-1:] must match data.shape[k:]");
    }

    // Note: indices values bounds check requires data dims known; we do it at runtime in exec.
}

// --------------------------- ScatterElements (Exec) ---------------------------
//
// Runtime semantics (no reduction):
// out = data (copy)
// For each position p in indices/updates:
//   idx = indices[p] (along axis)
//   q = p; q[axis] = normalize(idx)
//   out[q] = updates[p]
//
static Tensor<float> ScatterElements(const Tensor<float>& data,
    const Tensor<int64_t>& indices,
    const Tensor<float>& updates,
    int64_t axis) {
    VerifyScatterElements(data, indices, updates, axis);

    Tensor<float> out = data;
    int64_t r = data.rank();
    axis = normalizeAxis(axis, r);

    // Precompute strides for indices/upd traversal to multi-index.
    // We do a simple linear-to-multi conversion.
    std::vector<int64_t> dims = indices.shape;

    auto linearToMulti = [&](int64_t lin) -> std::vector<int64_t> {
        std::vector<int64_t> mi(static_cast<size_t>(r), 0);
        for (int64_t i = r - 1; i >= 0; --i) {
            int64_t d = dims[static_cast<size_t>(i)];
            mi[static_cast<size_t>(i)] = lin % d;
            lin /= d;
        }
        return mi;
        };

    int64_t axisDim = data.dim(axis);

    for (int64_t lin = 0; lin < indices.numel(); ++lin) {
        std::vector<int64_t> p = linearToMulti(lin);
        int64_t rawIdx = indices.data[static_cast<size_t>(lin)];
        int64_t w = normalizeIndex(rawIdx, axisDim);

        std::vector<int64_t> qidx = p;
        qidx[static_cast<size_t>(axis)] = w;

        int64_t outOff = out.offset(qidx);
        out.data[static_cast<size_t>(outOff)] = updates.data[static_cast<size_t>(lin)];
    }

    return out;
}

// --------------------------- ScatterND (Exec) ---------------------------
//
// Runtime semantics:
// out = data (copy)
// indices shape [..., k], where k <= rank(data).
// For each "row" i over indices[0:q-1]:
//   coord = indices[i, 0:k] (allow negative -> normalize)
//   Write a block of shape data.shape[k:] from updates at corresponding position.
//
static Tensor<float> ScatterND(const Tensor<float>& data,
    const Tensor<int64_t>& indices,
    const Tensor<float>& updates) {
    VerifyScatterND(data, indices, updates);

    Tensor<float> out = data;

    int64_t r = data.rank();
    int64_t q = indices.rank();
    int64_t k = indices.dim(q - 1);

    // Number of index rows = product(indices.shape[0:q-1])
    int64_t rows = 1;
    for (int64_t i = 0; i < q - 1; ++i) rows *= indices.dim(i);

    // Block shape = data.shape[k:]
    std::vector<int64_t> blockShape;
    for (int64_t i = k; i < r; ++i) blockShape.push_back(data.dim(i));
    int64_t blockSize = Tensor<float>::numelFromShape(blockShape);

    // Helper: convert rowId to multi-index over indices[0:q-1]
    std::vector<int64_t> rowDims;
    for (int64_t i = 0; i < q - 1; ++i) rowDims.push_back(indices.dim(i));

    auto rowLinearToMulti = [&](int64_t lin) -> std::vector<int64_t> {
        std::vector<int64_t> mi(static_cast<size_t>(q - 1), 0);
        for (int64_t i = (q - 2); i >= 0; --i) {
            int64_t d = rowDims[static_cast<size_t>(i)];
            mi[static_cast<size_t>(i)] = lin % d;
            lin /= d;
        }
        return mi;
        };

    // Compute indices strides to locate indices[i0,i1,...,i_{q-2}, j] in linear memory.
    // Row-major: last dim is k.
    auto indicesOffset = [&](const std::vector<int64_t>& prefix, int64_t j) -> int64_t {
        // prefix has length (q-1)
        std::vector<int64_t> full(prefix);
        full.push_back(j);
        // manual offset to avoid allocating a Tensor<int64_t> view
        int64_t off = 0;
        int64_t stride = 1;
        for (int64_t i = q - 1; i >= 0; --i) {
            int64_t d = indices.dim(i);
            int64_t v = full[static_cast<size_t>(i)];
            off += v * stride;
            stride *= d;
        }
        return off;
        };

    // Compute updates base offset for a given row prefix.
    // updates shape: [indices.shape[0:q-1]] + [data.shape[k:]]
    auto updatesBaseOffset = [&](const std::vector<int64_t>& prefix) -> int64_t {
        // prefix length q-1, followed by block coords.
        // Base offset means block coords all zero.
        int64_t off = 0;
        int64_t stride = 1;
        // Build full index: [prefix..., 0,0,...]
        int64_t ur = updates.rank();
        // iterate from last dim backwards
        for (int64_t i = ur - 1; i >= 0; --i) {
            int64_t d = updates.dim(i);
            int64_t v = 0;
            if (i < (q - 1)) v = prefix[static_cast<size_t>(i)];
            off += v * stride;
            stride *= d;
        }
        return off;
        };

    // For each row
    for (int64_t row = 0; row < rows; ++row) {
        std::vector<int64_t> prefix = rowLinearToMulti(row);

        // Read k coordinates
        std::vector<int64_t> coord(static_cast<size_t>(k), 0);
        for (int64_t j = 0; j < k; ++j) {
            int64_t off = indicesOffset(prefix, j);
            int64_t raw = indices.data[static_cast<size_t>(off)];
            coord[static_cast<size_t>(j)] = normalizeIndex(raw, data.dim(j));
        }

        // Destination base offset in out for this coord (block coords are 0)
        // out index = [coord..., 0,0,...]
        std::vector<int64_t> outBaseIndex;
        outBaseIndex.reserve(static_cast<size_t>(r));
        for (int64_t j = 0; j < k; ++j) outBaseIndex.push_back(coord[static_cast<size_t>(j)]);
        for (int64_t j = k; j < r; ++j) outBaseIndex.push_back(0);

        int64_t outBaseOff = out.offset(outBaseIndex);
        int64_t updBaseOff = updatesBaseOffset(prefix);

        // Copy block
        for (int64_t t = 0; t < blockSize; ++t) {
            out.data[static_cast<size_t>(outBaseOff + t)] =
                updates.data[static_cast<size_t>(updBaseOff + t)];
        }
    }

    return out;
}

// --------------------------- Scatter (Compat Wrapper) ---------------------------
//
// onnx-mlir only infers shape for Scatter. Historically Scatter existed in older opsets.
// In practice many engines map it to ScatterElements with a default axis.
//
// Here we implement Scatter as ScatterElements(axis=0) for a practical runtime.
//
static Tensor<float> Scatter(const Tensor<float>& data,
    const Tensor<int64_t>& indices,
    const Tensor<float>& updates) {
    return ScatterElements(data, indices, updates, /*axis=*/0);
}

// ------------------------------ Demo Helpers ------------------------------
static void PrintTensor(const Tensor<float>& t, const std::string& name) {
    std::cout << name << " shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        std::cout << t.shape[i] << (i + 1 == t.shape.size() ? "" : ",");
    }
    std::cout << "] data=[";
    for (size_t i = 0; i < t.data.size(); ++i) {
        std::cout << t.data[i] << (i + 1 == t.data.size() ? "" : ", ");
    }
    std::cout << "]\n";
}

int main() {
    // -------- ScatterElements demo --------
    // data: 2x3
    Tensor<float> data({ 2, 3 }, {
      0, 1, 2,
      3, 4, 5
        });

    // indices & updates must have same shape and rank as data for ScatterElements
    // axis = 1 means update along columns
    Tensor<int64_t> indicesSE({ 2, 3 }, {
      2, 1, 0,
      0, 2, 1
        });
    Tensor<float> updatesSE({ 2, 3 }, {
      10, 11, 12,
      13, 14, 15
        });

    auto outSE = ScatterElements(data, indicesSE, updatesSE, /*axis=*/1);
    PrintTensor(data, "data");
    PrintTensor(outSE, "ScatterElements(axis=1) out");

    // -------- ScatterND demo --------
    // dataND: 3x3
    Tensor<float> dataND({ 3, 3 }, {
      0, 1, 2,
      3, 4, 5,
      6, 7, 8
        });

    // indices: shape [N, k] = [2, 2], so q=2, k=2
    // each row is a 2D coordinate into dataND
    Tensor<int64_t> indicesND({ 2, 2 }, {
      0, 0,
      2, 1
        });

    // updates rank = r + q - k - 1 = 2 + 2 - 2 - 1 = 1, shape [N] = [2]
    Tensor<float> updatesND({ 2 }, { 100, 200 });

    auto outND = ScatterND(dataND, indicesND, updatesND);
    PrintTensor(dataND, "dataND");
    PrintTensor(outND, "ScatterND out");

    // -------- Scatter (compat) demo --------
    // This calls ScatterElements(axis=0) internally.
    Tensor<float> dataS({ 2, 2 }, { 1, 2, 3, 4 });
    Tensor<int64_t> idxS({ 2, 2 }, { 1, 0, 0, 1 });
    Tensor<float> updS({ 2, 2 }, { 9, 8, 7, 6 });
    auto outS = Scatter(dataS, idxS, updS);
    PrintTensor(dataS, "dataS");
    PrintTensor(outS, "Scatter(axis=0 compat) out");

    return 0;
}
