//同一块内存：
//buffer = [0, 1, 2, 3, 4, 5, 6, 7...]
//
//NCHW 解释： idx = ((n * C + c) * H + h) * W + w
//NHWC 解释： idx = ((n * H + h) * W + w) * C + c
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// --------------------------- LayoutTransform (Runtime Kernel) ---------------------------
//
// This file provides a minimal, self-contained C++ implementation that mirrors the
// common semantics of onnx-mlir's ONNXLayoutTransformOp:
//
// - Input/Output logical shape is the same.
// - Layout change is represented by a permutation of axes.
// - If the target layout differs, we perform a physical reorder (transpose/permute).
// - In compiler IR, this op may be "metadata-only" (encoding change) and become a
//   no-op at runtime if the backend can reinterpret the same buffer.
//
// Notes:
// - This runtime kernel performs a real reorder (copy) because raw pointers
//   typically cannot be reinterpreted safely across different layouts without
//   changing indexing semantics.
// - For a compiler, you might lower this op into either:
//     (1) a no-op (if layout is only encoding / consumers agree), or
//     (2) a transpose kernel (like implemented below).
//

template <typename T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<T> data;

    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

static std::vector<int64_t> computeStridesRowMajor(const std::vector<int64_t>& shape) {
    // Row-major (C-order) strides.
    // Example shape [N,C,H,W] => strides [C*H*W, H*W, W, 1]
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static int64_t offsetFromIndices(const std::vector<int64_t>& idx,
    const std::vector<int64_t>& strides) {
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
    return off;
}

static std::vector<int64_t> unflattenIndex(int64_t linear,
    const std::vector<int64_t>& shape) {
    // Converts a linear index to multi-d index under row-major order.
    std::vector<int64_t> idx(shape.size(), 0);
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        int64_t dim = shape[i];
        idx[i] = linear % dim;
        linear /= dim;
    }
    return idx;
}

template <typename T>
Tensor<T> permuteAxes(const Tensor<T>& in, const std::vector<int64_t>& perm) {
    assert(in.shape.size() == perm.size() && "perm rank must match input rank");

    // Output shape is permuted version of input shape.
    Tensor<T> out;
    out.shape.resize(in.shape.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        int64_t srcAxis = perm[i];
        assert(srcAxis >= 0 && (size_t)srcAxis < perm.size());
        out.shape[i] = in.shape[srcAxis];
    }
    out.data.resize((size_t)in.numel());

    // Strides for row-major memory.
    auto inStrides = computeStridesRowMajor(in.shape);
    auto outStrides = computeStridesRowMajor(out.shape);

    // For each output element, compute corresponding input index.
    const int64_t total = out.numel();
    for (int64_t o = 0; o < total; ++o) {
        auto outIdx = unflattenIndex(o, out.shape);

        // Map outIdx -> inIdx by inverse permutation:
        // out axis i corresponds to in axis perm[i].
        std::vector<int64_t> inIdx(in.shape.size(), 0);
        for (size_t i = 0; i < perm.size(); ++i) {
            inIdx[perm[i]] = outIdx[i];
        }

        int64_t inOff = offsetFromIndices(inIdx, inStrides);
        int64_t outOff = offsetFromIndices(outIdx, outStrides);
        out.data[(size_t)outOff] = in.data[(size_t)inOff];
    }
    return out;
}

// Map common layout strings to permutation.
// We interpret the input "logical axes order" as the layout string itself.
// For example, for 4D:
// - "NCHW" means axes [0,1,2,3] correspond to [N,C,H,W].
// - "NHWC" means axes [0,1,2,3] correspond to [N,H,W,C].
//
// To transform from srcLayout to dstLayout, we compute a perm such that:
// outAxes follow dstLayout, picking indices from srcLayout.
static std::vector<int64_t> layoutPerm4D(const std::string& src, const std::string& dst) {
    assert(src.size() == 4 && dst.size() == 4);
    // For each char in dst, find its position in src.
    std::vector<int64_t> perm(4, 0);
    for (int i = 0; i < 4; ++i) {
        char c = dst[i];
        size_t pos = src.find(c);
        assert(pos != std::string::npos && "dst axis must exist in src layout");
        perm[i] = (int64_t)pos;
    }
    return perm;
}

template <typename T>
Tensor<T> layoutTransform4D(const Tensor<T>& in,
    const std::string& srcLayout,
    const std::string& dstLayout) {
    assert(in.shape.size() == 4 && "This helper assumes 4D tensor");
    if (srcLayout == dstLayout) {
        // No transform needed (conceptually metadata-only).
        return in;
    }
    auto perm = layoutPerm4D(srcLayout, dstLayout);
    return permuteAxes(in, perm);
}

// Pretty print a small 4D tensor as nested loops.
template <typename T>
void print4D(const Tensor<T>& t, const std::string& layout, const std::string& name) {
    auto s = t.shape;
    std::cout << name << " shape=[";
    for (size_t i = 0; i < s.size(); ++i) {
        std::cout << s[i] << (i + 1 < s.size() ? "," : "");
    }
    std::cout << "], layout=" << layout << "\n";

    // Print in row-major memory order with indices interpretation of given layout.
    // For demo simplicity, only show a few elements.
    const int64_t total = t.numel();
    std::cout << "  data (linear): ";
    for (int64_t i = 0; i < total; ++i) {
        std::cout << t.data[(size_t)i] << (i + 1 < total ? " " : "");
    }
    std::cout << "\n";
}

int main() {
    // Example: N=1, C=2, H=2, W=3 (NCHW)
    Tensor<float> x;
    x.shape = { 1, 2, 2, 3 };
    x.data.resize((size_t)x.numel());

    // Fill with 0..numel-1
    std::iota(x.data.begin(), x.data.end(), 0.0f);

    print4D(x, "NCHW", "x");

    // Transform NCHW -> NHWC
    Tensor<float> y = layoutTransform4D(x, "NCHW", "NHWC");
    print4D(y, "NHWC", "y");

    // Transform back NHWC -> NCHW
    Tensor<float> z = layoutTransform4D(y, "NHWC", "NCHW");
    print4D(z, "NCHW", "z");

    // z should equal x
    assert(z.shape == x.shape);
    assert(z.data == x.data);
    std::cout << "Round-trip OK.\n";
    return 0;
}
