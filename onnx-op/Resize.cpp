#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// --------------------------- Minimal Tensor ---------------------------
//
// A simple row-major tensor container.
// - shape: int64 dims
// - data: flat buffer, size == product(shape)
// ---------------------------------------------------------------------
template <typename T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<T> data;

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t numel() const {
        if (shape.empty()) return 1; // scalar-like
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

// --------------------------- ONNX Resize (Standalone) ---------------------------
//
// This implementation follows the onnx-mlir snippet logic for shape + scales:
//
// - axes unsupported (reject).
// - scales and sizes are mutually exclusive.
// - "absent" inputs are represented by:
//     * std::nullopt
//     * OR empty vectors (simulating an empty tensor with numel==0)
//
// Output shape rules:
// - If scales is present:
//     out_dim[i] = int(input_dim[i] * scales[i])
//   (for positive values, trunc-toward-zero equals floor).
// - If sizes is present:
//     out_dim = sizes
//   and scales[i] = float(out_dim[i]) / float(input_dim[i])
//
// Interpolation (runtime):
// - mode="nearest": supports arbitrary rank.
// - mode="linear": supports rank==4 (NCHW) bilinear over H,W.
//
// Supported coordinate_transformation_mode:
// - "asymmetric" (default)
// - "half_pixel"
// - "align_corners"
//
// Supported nearest_mode:
// - "round_prefer_floor" (default)
// - "floor"
// - "ceil"
// - "round_prefer_ceil"
//
// Notes:
// - roi/cubic/antialias/exclude_outside/extrapolation_value are not implemented
//   because they are not present in the provided onnx-mlir code snippet.
// -------------------------------------------------------------------------------

struct ResizeAttrs {
    std::string mode = "nearest";                       // "nearest" or "linear"
    std::string coordinate_transformation_mode = "asymmetric";
    std::string nearest_mode = "round_prefer_floor";
    // Keep placeholders for completeness
    float extrapolation_value = 0.0f;
};

static bool IsAbsent(const std::optional<std::vector<int64_t>>& sizesOpt,
    const std::optional<std::vector<float>>& scalesOpt,
    bool checkSizes) {
    // This helper is only used conceptually; we check absent directly in main code.
    (void)sizesOpt; (void)scalesOpt; (void)checkSizes;
    return false;
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

static float TransformCoordinate(float outCoord,
    int64_t inSize,
    int64_t outSize,
    float scale,
    const std::string& ctm) {
    // outCoord is integer index in output, but passed as float.
    if (ctm == "asymmetric") {
        // x = out / scale
        return outCoord / scale;
    }
    else if (ctm == "half_pixel") {
        // x = (out + 0.5)/scale - 0.5
        return (outCoord + 0.5f) / scale - 0.5f;
    }
    else if (ctm == "align_corners") {
        // x = out * (in-1)/(out-1)
        if (outSize <= 1) return 0.0f;
        return outCoord * (static_cast<float>(inSize - 1) / static_cast<float>(outSize - 1));
    }
    else {
        throw std::runtime_error("Unsupported coordinate_transformation_mode: " + ctm);
    }
}

static int64_t NearestIndex(float x, int64_t inSize, const std::string& nearestMode) {
    // Clamp helper
    auto clamp = [&](int64_t v) -> int64_t {
        if (v < 0) return 0;
        if (v > inSize - 1) return inSize - 1;
        return v;
        };

    if (nearestMode == "floor") {
        return clamp(static_cast<int64_t>(std::floor(x)));
    }
    else if (nearestMode == "ceil") {
        return clamp(static_cast<int64_t>(std::ceil(x)));
    }
    else if (nearestMode == "round_prefer_ceil") {
        // half goes to ceil
        float f = std::floor(x);
        float frac = x - f;
        int64_t base = static_cast<int64_t>(f);
        if (frac > 0.5f) return clamp(base + 1);
        if (frac < 0.5f) return clamp(base);
        return clamp(base + 1);
    }
    else {
        // "round_prefer_floor" default: half goes to floor
        float f = std::floor(x);
        float frac = x - f;
        int64_t base = static_cast<int64_t>(f);
        if (frac > 0.5f) return clamp(base + 1);
        if (frac < 0.5f) return clamp(base);
        return clamp(base);
    }
}

static void ResolveOutputShapeAndScales(
    const std::vector<int64_t>& inShape,
    const std::optional<std::vector<float>>& scalesOpt,
    const std::optional<std::vector<int64_t>>& sizesOpt,
    std::vector<int64_t>& outShape,
    std::vector<float>& scalesOut) {

    const int64_t rank = static_cast<int64_t>(inShape.size());

    const bool scalesAbsent = (!scalesOpt.has_value() || scalesOpt->empty());
    const bool sizesAbsent = (!sizesOpt.has_value() || sizesOpt->empty());

    if (scalesAbsent && sizesAbsent)
        throw std::runtime_error("Resize: scales and sizes cannot both be absent.");
    if (!scalesAbsent && !sizesAbsent)
        throw std::runtime_error("Resize: scales and sizes cannot both be defined.");

    if (!scalesAbsent) {
        const auto& scales = *scalesOpt;
        if (static_cast<int64_t>(scales.size()) != rank)
            throw std::runtime_error("Resize: expected scales size == input rank.");

        outShape.resize(rank);
        scalesOut = scales;

        for (int64_t i = 0; i < rank; ++i) {
            if (inShape[i] < 0) throw std::runtime_error("Resize: negative input dim.");
            // int(trunc) for positive equals floor
            float prod = static_cast<float>(inShape[i]) * scales[i];
            int64_t od = static_cast<int64_t>(prod); // trunc toward 0
            if (od < 0) throw std::runtime_error("Resize: computed negative output dim.");
            outShape[i] = od;
        }
    }
    else {
        const auto& sizes = *sizesOpt;
        if (static_cast<int64_t>(sizes.size()) != rank)
            throw std::runtime_error("Resize: expected sizes size == input rank.");

        outShape = sizes;
        scalesOut.resize(rank);

        for (int64_t i = 0; i < rank; ++i) {
            if (inShape[i] <= 0) {
                // If input dim is 0, scale is ill-defined. In real ONNX, this is tricky.
                // We choose a conservative behavior: allow out dim only if it is also 0.
                if (inShape[i] == 0 && outShape[i] == 0) {
                    scalesOut[i] = 1.0f;
                    continue;
                }
                throw std::runtime_error("Resize: input dim must be > 0 to compute scales from sizes.");
            }
            if (outShape[i] < 0) throw std::runtime_error("Resize: negative sizes dim.");
            scalesOut[i] = static_cast<float>(outShape[i]) / static_cast<float>(inShape[i]);
        }
    }
}

static Tensor<float> ResizeNearestND(const Tensor<float>& X,
    const std::vector<int64_t>& outShape,
    const std::vector<float>& scales,
    const ResizeAttrs& attrs) {
    Tensor<float> Y;
    Y.shape = outShape;
    Y.data.resize(Y.numel());

    const auto inStrides = ComputeStridesRowMajor(X.shape);
    const auto outStrides = ComputeStridesRowMajor(Y.shape);

    const int64_t rank = X.rank();
    if (static_cast<int64_t>(scales.size()) != rank)
        throw std::runtime_error("ResizeNearestND: scales rank mismatch.");

    // Iterate all output elements by linear index, then decode to N-D indices.
    for (int64_t outLinear = 0; outLinear < static_cast<int64_t>(Y.data.size()); ++outLinear) {
        // Decode output linear index -> out indices
        std::vector<int64_t> outIdx(rank, 0);
        int64_t tmp = outLinear;
        for (int64_t i = 0; i < rank; ++i) {
            outIdx[i] = tmp / outStrides[i];
            tmp %= outStrides[i];
        }

        // Map output indices -> input indices
        std::vector<int64_t> inIdx(rank, 0);
        for (int64_t i = 0; i < rank; ++i) {
            const int64_t inSize = X.shape[i];
            const int64_t outSize = Y.shape[i];
            if (inSize == 0 || outSize == 0) {
                inIdx[i] = 0;
                continue;
            }
            float x = TransformCoordinate(static_cast<float>(outIdx[i]),
                inSize, outSize, scales[i],
                attrs.coordinate_transformation_mode);
            inIdx[i] = NearestIndex(x, inSize, attrs.nearest_mode);
        }

        const int64_t inOff = OffsetFromIndices(inIdx, inStrides);
        Y.data[outLinear] = X.data[inOff];
    }

    return Y;
}

static Tensor<float> ResizeLinearNCHW2D(const Tensor<float>& X,
    const std::vector<int64_t>& outShape,
    const std::vector<float>& scales,
    const ResizeAttrs& attrs) {
    // Only implement NCHW bilinear (rank == 4)
    if (X.rank() != 4)
        throw std::runtime_error("ResizeLinearNCHW2D: only rank==4 NCHW is supported.");

    const int64_t N = X.shape[0];
    const int64_t C = X.shape[1];
    const int64_t inH = X.shape[2];
    const int64_t inW = X.shape[3];

    const int64_t outN = outShape[0];
    const int64_t outC = outShape[1];
    const int64_t outH = outShape[2];
    const int64_t outW = outShape[3];

    if (N != outN || C != outC) {
        // In many models, N/C are kept same with scale=1. Here we enforce consistency.
        throw std::runtime_error("ResizeLinearNCHW2D: this demo expects N and C unchanged.");
    }

    Tensor<float> Y;
    Y.shape = outShape;
    Y.data.resize(Y.numel());

    auto idx4 = [&](int64_t n, int64_t c, int64_t h, int64_t w,
        const std::vector<int64_t>& shape) -> int64_t {
            // shape is [N,C,H,W]
            return ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
        };

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < outH; ++oh) {
                float inY = TransformCoordinate(static_cast<float>(oh), inH, outH, scales[2],
                    attrs.coordinate_transformation_mode);
                // Clamp for safety
                inY = std::max(0.0f, std::min(inY, static_cast<float>(inH - 1)));
                int64_t y0 = static_cast<int64_t>(std::floor(inY));
                int64_t y1 = std::min<int64_t>(y0 + 1, inH - 1);
                float wy = inY - static_cast<float>(y0);

                for (int64_t ow = 0; ow < outW; ++ow) {
                    float inX = TransformCoordinate(static_cast<float>(ow), inW, outW, scales[3],
                        attrs.coordinate_transformation_mode);
                    inX = std::max(0.0f, std::min(inX, static_cast<float>(inW - 1)));
                    int64_t x0 = static_cast<int64_t>(std::floor(inX));
                    int64_t x1 = std::min<int64_t>(x0 + 1, inW - 1);
                    float wx = inX - static_cast<float>(x0);

                    float v00 = X.data[idx4(n, c, y0, x0, X.shape)];
                    float v01 = X.data[idx4(n, c, y0, x1, X.shape)];
                    float v10 = X.data[idx4(n, c, y1, x0, X.shape)];
                    float v11 = X.data[idx4(n, c, y1, x1, X.shape)];

                    float v0 = v00 * (1.0f - wx) + v01 * wx;
                    float v1 = v10 * (1.0f - wx) + v11 * wx;
                    float v = v0 * (1.0f - wy) + v1 * wy;

                    Y.data[idx4(n, c, oh, ow, Y.shape)] = v;
                }
            }
        }
    }

    return Y;
}

static Tensor<float> OnnxResize(const Tensor<float>& X,
    const std::optional<std::vector<float>>& scalesOpt,
    const std::optional<std::vector<int64_t>>& sizesOpt,
    const ResizeAttrs& attrs) {
    ValidateTensorBuffer(X, "X");

    // Resolve output shape & runtime scales (same as onnx-mlir shape helper idea)
    std::vector<int64_t> outShape;
    std::vector<float> scales;
    ResolveOutputShapeAndScales(X.shape, scalesOpt, sizesOpt, outShape, scales);

    // Basic validation
    if (outShape.size() != X.shape.size())
        throw std::runtime_error("Resize: output rank mismatch.");

    // Dispatch by mode
    if (attrs.mode == "nearest") {
        return ResizeNearestND(X, outShape, scales, attrs);
    }
    else if (attrs.mode == "linear") {
        // Most common path: bilinear on NCHW H,W
        return ResizeLinearNCHW2D(X, outShape, scales, attrs);
    }
    else {
        throw std::runtime_error("Unsupported Resize mode: " + attrs.mode);
    }
}

// --------------------------- Demo ---------------------------
static void PrintTensor2D_NCHW_1x1(const Tensor<float>& t, const std::string& name) {
    std::cout << name << " shape=" << ShapeToString(t.shape) << "\n";
    if (t.shape.size() != 4 || t.shape[0] != 1 || t.shape[1] != 1) {
        std::cout << "(skip print: not 1x1xHxW)\n";
        return;
    }
    int64_t H = t.shape[2], W = t.shape[3];
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
            std::cout << t.data[h * W + w] << (w + 1 < W ? " " : "");
        }
        std::cout << "\n";
    }
}

int main() {
    try {
        // Input: 1x1x2x2
        // [ [1, 2],
        //   [3, 4] ]
        Tensor<float> X;
        X.shape = { 1, 1, 2, 2 };
        X.data = { 1, 2, 3, 4 };

        PrintTensor2D_NCHW_1x1(X, "X");

        // Example A: scales provided -> output shape computed by int(in_dim * scale)
        // Scale H,W by 2 -> 1x1x4x4
        ResizeAttrs attrsA;
        attrsA.mode = "nearest";
        attrsA.coordinate_transformation_mode = "asymmetric";
        attrsA.nearest_mode = "round_prefer_floor";

        auto Y_nearest = OnnxResize(
            X,
            std::optional<std::vector<float>>({ 1.0f, 1.0f, 2.0f, 2.0f }),
            std::nullopt,
            attrsA);

        PrintTensor2D_NCHW_1x1(Y_nearest, "Y_nearest (scales, nearest)");

        // Example B: sizes provided -> output shape is sizes, scales computed as out/in
        ResizeAttrs attrsB;
        attrsB.mode = "linear";
        attrsB.coordinate_transformation_mode = "half_pixel";

        auto Y_linear = OnnxResize(
            X,
            std::nullopt,
            std::optional<std::vector<int64_t>>({ 1, 1, 4, 4 }),
            attrsB);

        PrintTensor2D_NCHW_1x1(Y_linear, "Y_linear (sizes, bilinear)");

        // Example C: verify constraint error (both absent)
        // Uncomment to see expected failure:
        // auto Y_err = OnnxResize(X, std::nullopt, std::nullopt, attrsA);

        std::cout << "Done.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
