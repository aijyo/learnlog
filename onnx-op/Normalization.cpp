#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// --------------------------- Tensor (Simple Row-Major) ---------------------------
//
// A minimal N-D tensor helper for CPU reference implementation.
// - data stored in row-major contiguous buffer
// - shape: vector<int64_t>
// - indexing: flatten offset computed by strides
//
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<float> data;

    Tensor() = default;

    explicit Tensor(std::vector<int64_t> shp)
        : shape(std::move(shp)), strides(computeStrides(shape)) {
        int64_t n = numel(shape);
        data.assign(static_cast<size_t>(n), 0.0f);
    }

    static int64_t numel(const std::vector<int64_t>& shp) {
        if (shp.empty()) return 0;
        int64_t n = 1;
        for (int64_t d : shp) {
            if (d <= 0) throw std::runtime_error("Invalid shape dimension.");
            n *= d;
        }
        return n;
    }

    static std::vector<int64_t> computeStrides(const std::vector<int64_t>& shp) {
        std::vector<int64_t> st(shp.size(), 1);
        for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i) {
            st[i] = st[i + 1] * shp[i + 1];
        }
        return st;
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t numel() const { return numel(shape); }

    int64_t offset(const std::vector<int64_t>& idx) const {
        assert(idx.size() == shape.size());
        int64_t off = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            assert(idx[i] >= 0 && idx[i] < shape[i]);
            off += idx[i] * strides[i];
        }
        return off;
    }

    float& at(const std::vector<int64_t>& idx) { return data[offset(idx)]; }
    const float& at(const std::vector<int64_t>& idx) const { return data[offset(idx)]; }
};

// --------------------------- Utilities ---------------------------
//
// Broadcast helpers for ONNX-style elementwise ops.
// We support "unidirectional broadcast" by aligning trailing dims.
//
// Example:
//   X shape: [N, C, H, W]
//   scale shape: [C] -> treated as [1, C, 1, 1]
//   bias shape:  [C] -> treated as [1, C, 1, 1]
//
static int64_t normalizeAxis(int64_t axis, int64_t rank) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) throw std::runtime_error("Axis out of range.");
    return axis;
}

static std::vector<int64_t> alignTrailingShape(const std::vector<int64_t>& src,
    int64_t targetRank) {
    // Pad leading ones to match target rank.
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(targetRank));
    int64_t pad = targetRank - static_cast<int64_t>(src.size());
    if (pad < 0) throw std::runtime_error("Cannot align shape: src rank > target rank.");
    for (int64_t i = 0; i < pad; ++i) out.push_back(1);
    for (auto d : src) out.push_back(d);
    return out;
}

static std::vector<int64_t> computeStridesRowMajor(const std::vector<int64_t>& shp) {
    return Tensor::computeStrides(shp);
}

// Map an output index to a broadcasted input offset.
static int64_t broadcastOffset(const std::vector<int64_t>& outIdx,
    const std::vector<int64_t>& inShapeAligned,
    const std::vector<int64_t>& inStridesAligned) {
    // For dims where inShapeAligned[d] == 1, use index 0.
    assert(outIdx.size() == inShapeAligned.size());
    int64_t off = 0;
    for (size_t d = 0; d < outIdx.size(); ++d) {
        int64_t i = (inShapeAligned[d] == 1) ? 0 : outIdx[d];
        off += i * inStridesAligned[d];
    }
    return off;
}

// Convert a flat index to N-D index using shape/strides.
static std::vector<int64_t> unravelIndex(int64_t flat,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides) {
    std::vector<int64_t> idx(shape.size(), 0);
    int64_t rem = flat;
    for (size_t i = 0; i < shape.size(); ++i) {
        idx[i] = rem / strides[i];
        rem = rem % strides[i];
    }
    return idx;
}

// --------------------------- BatchNormalization (Inference Mode) ---------------------------
//
// Semantics (per channel C):
//   y = (x - mean[c]) / sqrt(var[c] + eps) * scale[c] + b[c]
//
// Shape expectations (common ONNX):
//   x:     [N, C, D1, D2, ...]  or [N] (treated as C=1)
//   scale: [C]
//   b:     [C]
//   mean:  [C]
//   var:   [C]
//
Tensor batchNormInference(const Tensor& x,
    const Tensor& scale,
    const Tensor& b,
    const Tensor& mean,
    const Tensor& var,
    float eps) {
    if (x.rank() < 1) throw std::runtime_error("BatchNorm: x rank must be >= 1.");

    int64_t C = 1;
    if (x.rank() >= 2) C = x.shape[1];

    auto checkVecC = [&](const Tensor& t, const std::string& name) {
        if (t.rank() != 1) throw std::runtime_error("BatchNorm: " + name + " must be rank-1.");
        if (t.shape[0] != C) throw std::runtime_error("BatchNorm: " + name + " dim must equal C.");
        };

    // If x is rank-1, treat as C=1 and require stats length 1.
    if (x.rank() == 1) C = 1;

    checkVecC(scale, "scale");
    checkVecC(b, "b");
    checkVecC(mean, "mean");
    checkVecC(var, "var");

    Tensor y(x.shape);
    const auto& xStr = x.strides;

    for (int64_t i = 0; i < x.numel(); ++i) {
        auto idx = unravelIndex(i, x.shape, xStr);
        int64_t c = (x.rank() >= 2) ? idx[1] : 0;

        float xv = x.data[static_cast<size_t>(i)];
        float s = scale.data[static_cast<size_t>(c)];
        float bb = b.data[static_cast<size_t>(c)];
        float m = mean.data[static_cast<size_t>(c)];
        float v = var.data[static_cast<size_t>(c)];

        float invStd = 1.0f / std::sqrt(v + eps);
        y.data[static_cast<size_t>(i)] = (xv - m) * invStd * s + bb;
    }
    return y;
}

// --------------------------- InstanceNormalization ---------------------------
//
// Semantics (per instance n and channel c):
//   mean[n,c] = avg over spatial dims
//   var[n,c]  = avg (x-mean)^2 over spatial dims
//   y = (x - mean) / sqrt(var + eps) * scale[c] + b[c]
//
// Typical x shape: [N, C, D1, D2, ...] with spatialRank >= 1
// scale/b shape: [C]
//
Tensor instanceNorm(const Tensor& x,
    const Tensor& scale,
    const Tensor& b,
    float eps) {
    if (x.rank() < 3) throw std::runtime_error("InstanceNorm: x rank must be >= 3 (N,C,spatial...).");

    int64_t N = x.shape[0];
    int64_t C = x.shape[1];

    if (scale.rank() != 1 || scale.shape[0] != C)
        throw std::runtime_error("InstanceNorm: scale must be [C].");
    if (b.rank() != 1 || b.shape[0] != C)
        throw std::runtime_error("InstanceNorm: b must be [C].");

    // spatial size = product of dims from 2..end
    int64_t spatialSize = 1;
    for (int64_t r = 2; r < x.rank(); ++r) spatialSize *= x.shape[static_cast<size_t>(r)];

    Tensor y(x.shape);

    // Compute per-(n,c) mean/var in a straightforward way.
    // Layout: x is row-major. For fixed n,c, the spatial block is contiguous if dims >=2? Not always,
    // but with row-major and fixed first two indices, the remaining dims are contiguous.
    // Offset base = n*stride0 + c*stride1.
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t base = n * x.strides[0] + c * x.strides[1];

            // Mean
            double sum = 0.0;
            for (int64_t k = 0; k < spatialSize; ++k) {
                sum += x.data[static_cast<size_t>(base + k)];
            }
            float meanNC = static_cast<float>(sum / static_cast<double>(spatialSize));

            // Var
            double sq = 0.0;
            for (int64_t k = 0; k < spatialSize; ++k) {
                double d = static_cast<double>(x.data[static_cast<size_t>(base + k)] - meanNC);
                sq += d * d;
            }
            float varNC = static_cast<float>(sq / static_cast<double>(spatialSize));
            float invStd = 1.0f / std::sqrt(varNC + eps);

            float s = scale.data[static_cast<size_t>(c)];
            float bb = b.data[static_cast<size_t>(c)];

            for (int64_t k = 0; k < spatialSize; ++k) {
                float xv = x.data[static_cast<size_t>(base + k)];
                y.data[static_cast<size_t>(base + k)] = (xv - meanNC) * invStd * s + bb;
            }
        }
    }
    return y;
}

// --------------------------- LayerNormalization ---------------------------
//
// Semantics:
//   Normalize over the last K dims, where K = rank - axis.
// For each outer index (dims < axis):
//   mean = avg(X over dims axis..end)
//   var  = avg((X-mean)^2 over dims axis..end)
//   Y = (X-mean)/sqrt(var+eps) * scale + B
//
// scale/B are broadcastable to X (trailing alignment).
//
struct LayerNormOutput {
    Tensor Y;
    Tensor Mean;      // Optional
    Tensor InvStdDev; // Optional
};

LayerNormOutput layerNorm(const Tensor& X,
    const Tensor& scale,
    const Tensor* B, // nullable
    int64_t axis,
    float eps,
    bool outputMean,
    bool outputInvStdDev) {
    if (X.rank() < 1) throw std::runtime_error("LayerNorm: X rank must be >= 1.");
    int64_t r = X.rank();
    axis = normalizeAxis(axis, r);

    // Prepare broadcast shapes for scale and bias (align trailing dims).
    auto scaleAlignedShape = alignTrailingShape(scale.shape, r);
    auto scaleAlignedStrides = computeStridesRowMajor(scaleAlignedShape);

    std::vector<int64_t> bAlignedShape;
    std::vector<int64_t> bAlignedStrides;
    if (B) {
        bAlignedShape = alignTrailingShape(B->shape, r);
        bAlignedStrides = computeStridesRowMajor(bAlignedShape);
    }

    // Validate broadcast compatibility: each aligned dim must be 1 or equal to X dim.
    auto checkBroadcast = [&](const std::vector<int64_t>& inShpAligned, const std::string& name) {
        for (int64_t i = 0; i < r; ++i) {
            int64_t xd = X.shape[static_cast<size_t>(i)];
            int64_t id = inShpAligned[static_cast<size_t>(i)];
            if (id != 1 && id != xd)
                throw std::runtime_error("LayerNorm: " + name + " not broadcastable to X.");
        }
        };
    checkBroadcast(scaleAlignedShape, "scale");
    if (B) checkBroadcast(bAlignedShape, "B");

    Tensor Y(X.shape);

    // Mean/InvStdDev shapes follow onnx-mlir logic: same as X but dims [axis..] become 1.
    Tensor MeanT;
    Tensor InvStdT;
    std::vector<int64_t> statShape = X.shape;
    for (int64_t i = axis; i < r; ++i) statShape[static_cast<size_t>(i)] = 1;
    if (outputMean) MeanT = Tensor(statShape);
    if (outputInvStdDev) InvStdT = Tensor(statShape);

    // inner size = product of dims axis..end
    int64_t inner = 1;
    for (int64_t i = axis; i < r; ++i) inner *= X.shape[static_cast<size_t>(i)];
    // outer size = product of dims 0..axis-1
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= X.shape[static_cast<size_t>(i)];

    // For each outer position, normalize a contiguous block of length inner.
    // Because row-major, fixing first axis dims produces contiguous tail.
    for (int64_t o = 0; o < outer; ++o) {
        int64_t base = o * inner;

        // mean
        double sum = 0.0;
        for (int64_t k = 0; k < inner; ++k) sum += X.data[static_cast<size_t>(base + k)];
        float mean = static_cast<float>(sum / static_cast<double>(inner));

        // var
        double sq = 0.0;
        for (int64_t k = 0; k < inner; ++k) {
            double d = static_cast<double>(X.data[static_cast<size_t>(base + k)] - mean);
            sq += d * d;
        }
        float var = static_cast<float>(sq / static_cast<double>(inner));
        float invStd = 1.0f / std::sqrt(var + eps);

        // Write optional stats.
        if (outputMean || outputInvStdDev) {
            // stat tensor has same outer indexing, but inner collapsed to 1s.
            // Its contiguous layout means one value per outer.
            if (outputMean) MeanT.data[static_cast<size_t>(o)] = mean;
            if (outputInvStdDev) InvStdT.data[static_cast<size_t>(o)] = invStd;
        }

        // Apply affine with broadcast scale/B.
        for (int64_t k = 0; k < inner; ++k) {
            int64_t flat = base + k;
            auto outIdx = unravelIndex(flat, X.shape, X.strides);

            // scale
            int64_t so = broadcastOffset(outIdx, scaleAlignedShape, scaleAlignedStrides);
            float s = scale.data[static_cast<size_t>(so)];

            float bb = 0.0f;
            if (B) {
                int64_t bo = broadcastOffset(outIdx, bAlignedShape, bAlignedStrides);
                bb = B->data[static_cast<size_t>(bo)];
            }

            float xv = X.data[static_cast<size_t>(flat)];
            Y.data[static_cast<size_t>(flat)] = (xv - mean) * invStd * s + bb;
        }
    }

    LayerNormOutput out{ Y, MeanT, InvStdT };
    return out;
}

// --------------------------- RMSLayerNormalization ---------------------------
//
// Semantics:
//   rms = sqrt(mean(x^2) + eps) over dims axis..end
//   y = x / rms * scale + B
//
// Optional output: invStdDev (often used as 1/rms).
//
struct RMSLayerNormOutput {
    Tensor Y;
    Tensor InvStdDev; // Optional
};

RMSLayerNormOutput rmsLayerNorm(const Tensor& X,
    const Tensor& scale,
    const Tensor* B, // nullable
    int64_t axis,
    float eps,
    bool outputInvStdDev) {
    if (X.rank() < 1) throw std::runtime_error("RMSLayerNorm: X rank must be >= 1.");
    int64_t r = X.rank();
    axis = normalizeAxis(axis, r);

    auto scaleAlignedShape = alignTrailingShape(scale.shape, r);
    auto scaleAlignedStrides = computeStridesRowMajor(scaleAlignedShape);

    std::vector<int64_t> bAlignedShape;
    std::vector<int64_t> bAlignedStrides;
    if (B) {
        bAlignedShape = alignTrailingShape(B->shape, r);
        bAlignedStrides = computeStridesRowMajor(bAlignedShape);
    }

    auto checkBroadcast = [&](const std::vector<int64_t>& inShpAligned, const std::string& name) {
        for (int64_t i = 0; i < r; ++i) {
            int64_t xd = X.shape[static_cast<size_t>(i)];
            int64_t id = inShpAligned[static_cast<size_t>(i)];
            if (id != 1 && id != xd)
                throw std::runtime_error("RMSLayerNorm: " + name + " not broadcastable to X.");
        }
        };
    checkBroadcast(scaleAlignedShape, "scale");
    if (B) checkBroadcast(bAlignedShape, "B");

    Tensor Y(X.shape);

    // invStdDev shape: X with dims [axis..] set to 1 (onnx-mlir style).
    Tensor InvStdT;
    if (outputInvStdDev) {
        std::vector<int64_t> statShape = X.shape;
        for (int64_t i = axis; i < r; ++i) statShape[static_cast<size_t>(i)] = 1;
        InvStdT = Tensor(statShape);
    }

    int64_t inner = 1;
    for (int64_t i = axis; i < r; ++i) inner *= X.shape[static_cast<size_t>(i)];
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= X.shape[static_cast<size_t>(i)];

    for (int64_t o = 0; o < outer; ++o) {
        int64_t base = o * inner;

        double sq = 0.0;
        for (int64_t k = 0; k < inner; ++k) {
            double v = static_cast<double>(X.data[static_cast<size_t>(base + k)]);
            sq += v * v;
        }
        float meanSq = static_cast<float>(sq / static_cast<double>(inner));
        float invRms = 1.0f / std::sqrt(meanSq + eps);

        if (outputInvStdDev) InvStdT.data[static_cast<size_t>(o)] = invRms;

        for (int64_t k = 0; k < inner; ++k) {
            int64_t flat = base + k;
            auto outIdx = unravelIndex(flat, X.shape, X.strides);

            int64_t so = broadcastOffset(outIdx, scaleAlignedShape, scaleAlignedStrides);
            float s = scale.data[static_cast<size_t>(so)];

            float bb = 0.0f;
            if (B) {
                int64_t bo = broadcastOffset(outIdx, bAlignedShape, bAlignedStrides);
                bb = B->data[static_cast<size_t>(bo)];
            }

            float xv = X.data[static_cast<size_t>(flat)];
            Y.data[static_cast<size_t>(flat)] = (xv * invRms) * s + bb;
        }
    }

    RMSLayerNormOutput out{ Y, InvStdT };
    return out;
}

// --------------------------- Demo ---------------------------
//
// Build and run:
//   g++ -O2 -std=c++17 norm_demo.cpp -o norm_demo && ./norm_demo
//
static void fillSequential(Tensor& t, float start = 0.0f, float step = 1.0f) {
    for (int64_t i = 0; i < t.numel(); ++i) t.data[static_cast<size_t>(i)] = start + step * i;
}

static void printTensor(const Tensor& t, const std::string& name, int maxPrint = 16) {
    std::cout << name << " shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        std::cout << t.shape[i] << (i + 1 == t.shape.size() ? "" : ",");
    }
    std::cout << "] data=(";
    int64_t n = std::min<int64_t>(t.numel(), maxPrint);
    for (int64_t i = 0; i < n; ++i) {
        std::cout << t.data[static_cast<size_t>(i)] << (i + 1 == n ? "" : ", ");
    }
    if (t.numel() > n) std::cout << ", ...";
    std::cout << ")\n";
}

int main() {
    try {
        // Example X: [N,C,H,W] = [1,2,2,3]
        Tensor X({ 1, 2, 2, 3 });
        fillSequential(X, 1.0f, 1.0f);

        // BatchNorm stats: [C]
        Tensor scale({ 2 }); scale.data = { 1.0f, 1.5f };
        Tensor bias({ 2 });  bias.data = { 0.1f, -0.2f };
        Tensor mean({ 2 });  mean.data = { 3.0f, 9.0f };
        Tensor var({ 2 });   var.data = { 4.0f, 1.0f };

        auto Ybn = batchNormInference(X, scale, bias, mean, var, 1e-5f);

        // InstanceNorm uses per-instance spatial stats + affine [C]
        auto Yin = instanceNorm(X, scale, bias, 1e-5f);

        // LayerNorm on last dims (axis = -1 means normalize last dim only)
        // Use scale broadcastable to X: e.g. [W] = [3]
        Tensor lnScale({ 3 }); lnScale.data = { 1.0f, 1.0f, 1.0f };
        Tensor lnBias({ 3 });  lnBias.data = { 0.0f, 0.0f, 0.0f };

        auto LN = layerNorm(X, lnScale, &lnBias, -1, 1e-5f, /*mean*/true, /*invstd*/true);

        // RMSLayerNorm normalize over last 2 dims (axis = 2 => (H,W))
        Tensor rmsScale({ 1, 1, 2, 3 }); // same shape for simplicity (no broadcast issues)
        fillSequential(rmsScale, 0.5f, 0.01f);
        auto RMS = rmsLayerNorm(X, rmsScale, nullptr, 2, 1e-5f, /*invstd*/true);

        printTensor(X, "X");
        printTensor(Ybn, "BatchNormInference(Y)");
        printTensor(Yin, "InstanceNorm(Y)");
        printTensor(LN.Y, "LayerNorm(Y)");
        if (LN.Mean.numel() > 0) printTensor(LN.Mean, "LayerNorm(Mean)");
        if (LN.InvStdDev.numel() > 0) printTensor(LN.InvStdDev, "LayerNorm(InvStdDev)");
        printTensor(RMS.Y, "RMSLayerNorm(Y)");
        if (RMS.InvStdDev.numel() > 0) printTensor(RMS.InvStdDev, "RMSLayerNorm(InvStdDev)");

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
