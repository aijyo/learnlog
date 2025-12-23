#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// --------------------------- Utilities ---------------------------

// English comments as requested.

static int64_t NumElements(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::runtime_error("Negative dim in shape.");
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
    if (axis < -rank || axis >= rank)
        throw std::runtime_error("axis out of range.");
    if (axis < 0) axis += rank;
    return axis;
}

// Read complex (real, imag) from flat buffer where last dim is 2.
struct ComplexF {
    float re;
    float im;
};

static inline ComplexF LoadComplex(const float* data, int64_t baseIndex) {
    return { data[baseIndex + 0], data[baseIndex + 1] };
}

static inline void StoreComplex(float* data, int64_t baseIndex, ComplexF v) {
    data[baseIndex + 0] = v.re;
    data[baseIndex + 1] = v.im;
}

static inline ComplexF Add(ComplexF a, ComplexF b) {
    return { a.re + b.re, a.im + b.im };
}

static inline ComplexF Mul(ComplexF a, ComplexF b) {
    // (a.re + j a.im) * (b.re + j b.im)
    return { a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}

// exp( +/- j * theta )
static inline ComplexF ExpJ(float theta, bool inverse) {
    // Forward: e^{-j theta} = cos(theta) - j sin(theta)
    // Inverse: e^{+j theta} = cos(theta) + j sin(theta)
    float c = std::cos(theta);
    float s = std::sin(theta);
    return inverse ? ComplexF{ c, s } : ComplexF{ c, -s };
}

// --------------------------- DFT Core ---------------------------
//
// Assumptions:
// - Input shape: [..., N, 2] (last dim=2 stores complex components)
// - DFT axis refers to one of dims except the last complex dim.
//   Default axis = -2 (the dim just before the complex dim).
// - onesided=1 reduces output length on axis to floor(N/2)+1 for forward transform.
// - inverse=1 computes inverse DFT and applies 1/N scaling.
//   For simplicity, inverse with onesided=1 is rejected here.

struct DFTAttrs {
    int64_t axis = -2;
    bool inverse = false;
    bool onesided = false;
};

static std::vector<float> DFT(const std::vector<float>& input,
    const std::vector<int64_t>& inShape,
    const DFTAttrs& attrs,
    std::vector<int64_t>* outShape /*optional*/) {
    if (inShape.size() < 2)
        throw std::runtime_error("DFT: input rank must be >= 2 (need complex dim).");
    if (inShape.back() != 2)
        throw std::runtime_error("DFT: last dim must be 2 for (real, imag).");

    const int64_t rank = (int64_t)inShape.size();
    const int64_t complexDim = rank - 1;

    if (NumElements(inShape) != (int64_t)input.size())
        throw std::runtime_error("DFT: input size mismatch with shape.");

    if (attrs.inverse && attrs.onesided)
        throw std::runtime_error("DFT: inverse with onesided=1 is not supported in this demo.");

    int64_t axis = NormalizeAxis(attrs.axis, rank);
    if (axis == complexDim)
        throw std::runtime_error("DFT: axis cannot be the last complex dim.");

    const int64_t N = inShape[axis];

    // Output shape: same as input, except axis length may change for onesided forward.
    std::vector<int64_t> oShape = inShape;
    int64_t K = N;
    if (!attrs.inverse && attrs.onesided) {
        K = (N / 2) + 1; // floor(N/2)+1
        oShape[axis] = K;
    }

    if (outShape) *outShape = oShape;

    // Allocate output buffer.
    std::vector<float> output((size_t)NumElements(oShape), 0.0f);

    // Strides for input and output.
    const auto inStrides = StridesRowMajor(inShape);
    const auto outStrides = StridesRowMajor(oShape);

    // We iterate over all "outer" indices excluding the DFT axis and excluding the complex dim.
    // For each outer index, we run a 1-D DFT of length N -> K along the axis.
    //
    // Flattening logic:
    // - Treat the tensor as blocks where the axis index changes and complex dim=2 is contiguous.

    // Prepare a list of dims to iterate (outer dims).
    std::vector<int64_t> outerDims;
    outerDims.reserve((size_t)rank - 2);
    for (int64_t d = 0; d < rank; ++d) {
        if (d == axis || d == complexDim) continue;
        outerDims.push_back(d);
    }

    // Number of outer positions.
    int64_t outerCount = 1;
    for (int64_t d : outerDims) outerCount *= inShape[d];

    // Helper: decode an outer index (0..outerCount-1) into multi-dim coordinates for outer dims.
    // We'll compute base offsets for input and output for each outer coordinate with axis=0 and complex=0.
    auto DecodeOuter = [&](int64_t idx, const std::vector<int64_t>& shape,
        const std::vector<int64_t>& dims,
        std::vector<int64_t>& coords) {
            coords.resize(dims.size());
            for (int i = (int)dims.size() - 1; i >= 0; --i) {
                int64_t dim = dims[(size_t)i];
                int64_t size = shape[(size_t)dim];
                coords[(size_t)i] = idx % size;
                idx /= size;
            }
        };

    std::vector<int64_t> coords;
    coords.reserve(outerDims.size());

    const double twoPi = 2.0 * M_PI;

    for (int64_t oc = 0; oc < outerCount; ++oc) {
        DecodeOuter(oc, inShape, outerDims, coords);

        // Compute base offsets for input and output where axisIndex=0 and complexIndex=0.
        int64_t inBase = 0;
        int64_t outBase = 0;

        for (size_t i = 0; i < outerDims.size(); ++i) {
            int64_t dim = outerDims[i];
            int64_t c = coords[i];
            inBase += c * inStrides[(size_t)dim];
            outBase += c * outStrides[(size_t)dim];
        }

        // Now run DFT for this slice.
        // Input element at axis=n has base index: inBase + n*inStrides[axis]
        // But remember complex dim contributes 2 contiguous floats, i.e. index +0/+1.
        // Since complex dim is last, inStrides[complexDim] == 2? Actually in row-major,
        // stride of complexDim is 1 and complexDim size is 2, so the pair is contiguous.
        // We'll compute the flat base for the complex pair as:
        //   (inBase + n*inStrides[axis]) * 2? No: because last dim is part of shape.
        // Here inBase already points to the flat index including all dims; we must include axis.
        //
        // Because the last dim exists in shape, the "baseIndex" for the complex pair is:
        //   inOffset = inBase + n*inStrides[axis]
        // and then we read [inOffset + 0], [inOffset + 1].
        //
        // Same for output with axis=k.

        for (int64_t k = 0; k < K; ++k) {
            ComplexF acc{ 0.0f, 0.0f };

            for (int64_t n = 0; n < N; ++n) {
                const int64_t inOff = inBase + n * inStrides[(size_t)axis];
                ComplexF xn = LoadComplex(input.data(), inOff);

                double angle = twoPi * (double)k * (double)n / (double)N;
                ComplexF w = ExpJ((float)angle, attrs.inverse);

                acc = Add(acc, Mul(xn, w));
            }

            if (attrs.inverse) {
                acc.re = (float)(acc.re / (double)N);
                acc.im = (float)(acc.im / (double)N);
            }

            const int64_t outOff = outBase + k * outStrides[(size_t)axis];
            StoreComplex(output.data(), outOff, acc);
        }
    }

    return output;
}

// --------------------------- Demo ---------------------------

static void PrintComplex1D(const std::vector<float>& data,
    const std::vector<int64_t>& shape,
    const std::string& name,
    int maxPrint = 16) {
    // Assumes shape = [N,2] or [*, N, 2] with * collapsed for demo simplicity.
    if (shape.size() != 2 || shape[1] != 2) {
        std::cout << name << ": (skip print, not [N,2])\n";
        return;
    }
    int64_t N = shape[0];
    std::cout << name << " (N=" << N << "):\n";
    int64_t k = std::min<int64_t>(N, maxPrint);
    for (int64_t i = 0; i < k; ++i) {
        int64_t off = i * 2;
        std::cout << "  [" << i << "] " << data[off] << " + j" << data[off + 1] << "\n";
    }
    if (N > k) std::cout << "  ...\n";
}

int main() {
    try {
        // Example A: 1D complex signal, forward DFT.
        // Shape: [8, 2]
        std::vector<int64_t> xShape = { 8, 2 };
        std::vector<float> x(8 * 2, 0.0f);

        // Put a simple real impulse at n=1: x[1]=1
        x[1 * 2 + 0] = 1.0f; // real
        x[1 * 2 + 1] = 0.0f; // imag

        DFTAttrs a;
        a.axis = -2;       // transform along the dim before complex dim
        a.inverse = false;
        a.onesided = false;

        std::vector<int64_t> yShape;
        auto Y = DFT(x, xShape, a, &yShape);

        PrintComplex1D(x, xShape, "x");
        PrintComplex1D(Y, yShape, "DFT(x)");

        // Example B: inverse DFT should recover the impulse (approximately).
        DFTAttrs b = a;
        b.inverse = true;
        auto xRec = DFT(Y, yShape, b, nullptr);
        PrintComplex1D(xRec, xShape, "IDFT(DFT(x))");

        // Example C: onesided forward for real input (imag=0).
        DFTAttrs c = a;
        c.onesided = true;
        std::vector<int64_t> y1Shape;
        auto Y1 = DFT(x, xShape, c, &y1Shape);
        PrintComplex1D(Y1, y1Shape, "DFT(x) onesided");

        std::cout << "Done.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
