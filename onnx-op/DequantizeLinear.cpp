#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// -------------------------- Utilities --------------------------

// English comments per your preference.

static int64_t NumElements(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::runtime_error("Shape contains negative dim.");
        n *= d;
    }
    return n;
}

static std::vector<int64_t> StridesRowMajor(const std::vector<int64_t>& shape) {
    std::vector<int64_t> s(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        s[(size_t)i] = s[(size_t)i + 1] * shape[(size_t)i + 1];
    }
    return s;
}

static int64_t NormalizeAxis(int64_t axis, int64_t rank) {
    if (rank <= 0) throw std::runtime_error("Rank must be > 0.");
    if (axis < -rank || axis >= rank) {
        throw std::runtime_error("axis out of range: axis=" + std::to_string(axis) +
            " rank=" + std::to_string(rank));
    }
    if (axis < 0) axis += rank;
    return axis;
}

// Returns true if this 1-D tensor should be treated as "non-scalar" (len > 1).
static bool IsNonScalar1D(int64_t len) {
    // Matches onnx-mlir's idea: len == 1 is considered scalar-equivalent.
    return len > 1;
}

// -------------------------- DequantizeLinear --------------------------
//
// Implements ONNX DequantizeLinear semantics:
//   y = (x - zero_point) * scale
// scale: scalar or 1-D (len d>1) for per-axis
// zero_point: optional, scalar or 1-D (must match scale shape kind)
// axis: used only when per-axis
//
// Output is float for simplicity.

template <typename TX, typename TZP>
static void DequantizeLinearImpl(
    const TX* x, const std::vector<int64_t>& xShape,
    const float* scale, int64_t scaleLen,              // scaleLen = 1 (scalar) or d (per-axis)
    const TZP* zeroPoint, int64_t zpLen,               // zpLen = 0 (none), 1 (scalar), or d (per-axis)
    int64_t axis,                                      // meaningful only if per-axis
    float* y) {

    const int64_t rank = (int64_t)xShape.size();
    const int64_t N = NumElements(xShape);
    if (N == 0) return;

    const bool scalePerAxis = IsNonScalar1D(scaleLen);
    const bool zpProvided = (zeroPoint != nullptr) && (zpLen != 0);

    // Validate scale shape.
    if (!(scaleLen == 1 || scalePerAxis)) {
        throw std::runtime_error("scaleLen must be 1 (scalar) or >1 (per-axis).");
    }
    // Validate zero point shape kind if provided.
    if (zpProvided) {
        const bool zpPerAxis = IsNonScalar1D(zpLen);
        const bool zpScalar = (zpLen == 1);
        // Must match shape kind of scale: scalar with scalar, per-axis with per-axis.
        if (scalePerAxis) {
            if (!zpPerAxis || zpLen != scaleLen) {
                throw std::runtime_error("x_scale and x_zero_point must have same 1-D length for per-axis.");
            }
        }
        else { // scale scalar
            if (!zpScalar) {
                throw std::runtime_error("x_scale and x_zero_point must have same shape kind (both scalar).");
            }
        }
    }

    if (!scalePerAxis) {
        // Per-tensor: axis ignored.
        const float s = scale[0];
        if (zpProvided) {
            const TZP zp = zeroPoint[0];
            for (int64_t i = 0; i < N; ++i) {
                // Cast to int64_t to avoid overflow on subtraction.
                const int64_t xi = static_cast<int64_t>(x[i]);
                const int64_t zpi = static_cast<int64_t>(zp);
                y[i] = static_cast<float>((xi - zpi) * (double)s);
            }
        }
        else {
            for (int64_t i = 0; i < N; ++i) {
                y[i] = static_cast<float>(static_cast<double>(x[i]) * (double)s);
            }
        }
        return;
    }

    // Per-axis:
    const int64_t a = NormalizeAxis(axis, rank);
    const int64_t d = scaleLen;

    // Verify axis dim matches d when static.
    if (xShape[a] != d) {
        throw std::runtime_error("Per-axis: xShape[axis] must equal scaleLen (d).");
    }

    // Row-major strides.
    const auto strides = StridesRowMajor(xShape);
    const int64_t axisStride = strides[a];

    // We can compute the axis index for each flat index:
    // axisIndex = (flatIndex / axisStride) % d
    if (zpProvided) {
        for (int64_t i = 0; i < N; ++i) {
            const int64_t t = (i / axisStride) % d;
            const float s = scale[t];
            const TZP zp = zeroPoint[t];
            const int64_t xi = static_cast<int64_t>(x[i]);
            const int64_t zpi = static_cast<int64_t>(zp);
            y[i] = static_cast<float>((xi - zpi) * (double)s);
        }
    }
    else {
        for (int64_t i = 0; i < N; ++i) {
            const int64_t t = (i / axisStride) % d;
            const float s = scale[t];
            y[i] = static_cast<float>(static_cast<double>(x[i]) * (double)s);
        }
    }
}

template <typename TX>
std::vector<float> DequantizeLinear(
    const std::vector<TX>& x,
    const std::vector<int64_t>& xShape,
    const std::vector<float>& xScale,                    // scalar: {s} or per-axis: len d>1
    const std::optional<std::vector<int64_t>>& xZeroPoint,// none or scalar/per-axis; stored as int64 for convenience
    int64_t axis = 1) {

    const int64_t N = NumElements(xShape);
    if ((int64_t)x.size() != N) {
        throw std::runtime_error("x size does not match shape product.");
    }
    if (xScale.empty()) throw std::runtime_error("xScale must not be empty.");

    const int64_t scaleLen = (int64_t)xScale.size();

    // Handle zero point: we keep it in int64 but must clamp/cast to the intended type.
    // For a reference kernel, we treat zero_point type as int64_t and cast inside.
    const int64_t zpLen = xZeroPoint ? (int64_t)xZeroPoint->size() : 0;

    std::vector<float> y((size_t)N, 0.0f);

    // Prepare pointers.
    const TX* xPtr = x.data();
    const float* sPtr = xScale.data();

    // If zero point is absent, pass nullptr.
    if (!xZeroPoint.has_value()) {
        // Use int64_t template parameter just as placeholder; pointer is nullptr so it won't be read.
        DequantizeLinearImpl<TX, int64_t>(xPtr, xShape, sPtr, scaleLen,
            nullptr, 0, axis, y.data());
        return y;
    }

    // onnx-mlir has a special rule: if x_zero_point is int32, it must be 0.
    // Here we provide a simple check when user gives zero point values.
    // (We cannot know the "declared dtype" here, so we do not enforce by dtype, only provide a helper check path.)

    // Copy zp values.
    const std::vector<int64_t>& zpVec = *xZeroPoint;
    if ((int64_t)zpVec.size() != zpLen) {
        throw std::runtime_error("Internal error: zp length mismatch.");
    }

    // We'll pass int64 zero point values to the impl and cast per element.
    DequantizeLinearImpl<TX, int64_t>(xPtr, xShape, sPtr, scaleLen,
        zpVec.data(), zpLen, axis, y.data());
    return y;
}

// -------------------------- Demo --------------------------

static void PrintTensorFlat(const std::vector<float>& y, const std::string& name, int maxN = 32) {
    std::cout << name << " = [";
    int n = (int)y.size();
    int k = std::min(n, maxN);
    for (int i = 0; i < k; ++i) {
        std::cout << y[i];
        if (i + 1 < k) std::cout << ", ";
    }
    if (n > k) std::cout << ", ...";
    std::cout << "]\n";
}

int main() {
    try {
        // -------- Example 1: per-tensor --------
        // x: uint8, shape [2,3]
        std::vector<int64_t> shape1 = { 2, 3 };
        std::vector<uint8_t> x1 = {
          0,  10, 20,
          30, 40, 50
        };
        std::vector<float> scale1 = { 0.1f }; // scalar scale
        std::optional<std::vector<int64_t>> zp1 = std::vector<int64_t>{ 128 }; // scalar zero point

        auto y1 = DequantizeLinear<uint8_t>(x1, shape1, scale1, zp1, /*axis ignored*/ 1);
        PrintTensorFlat(y1, "y1(per-tensor)");

        // -------- Example 2: per-axis (per-channel) --------
        // NCHW = [1,3,2,2], axis=1 (C)
        std::vector<int64_t> shape2 = { 1, 3, 2, 2 };
        // 1*3*2*2 = 12 elements.
        // Lay out in row-major: N, C, H, W. The C blocks are contiguous chunks of size H*W=4.
        std::vector<int8_t> x2 = {
            // C0 (4)
            -10, -5, 0, 5,
            // C1 (4)
            10, 20, 30, 40,
            // C2 (4)
            -128, -64, 64, 127
        };

        std::vector<float> scale2 = { 0.01f, 0.02f, 0.04f }; // per-channel scales (d=3)
        std::optional<std::vector<int64_t>> zp2 = std::vector<int64_t>{ 0, 1, -2 }; // per-channel zero points
        auto y2 = DequantizeLinear<int8_t>(x2, shape2, scale2, zp2, /*axis*/ 1);
        PrintTensorFlat(y2, "y2(per-axis axis=1)");

        // -------- Example 3: per-axis without zero_point --------
        std::optional<std::vector<int64_t>> zp3 = std::nullopt;
        auto y3 = DequantizeLinear<int8_t>(x2, shape2, scale2, zp3, /*axis*/ 1);
        PrintTensorFlat(y3, "y3(per-axis no zp)");

        std::cout << "Done.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
