#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// ------------------------------- Expand (ONNX) -------------------------------
//
// ONNX Expand(input, shape):
//   - `shape` is a 1-D int64 tensor that specifies the target output shape.
//   - Broadcasting follows NumPy rules, aligned from the right.
//   - If inputRank < targetRank, we pad LEADING 1s to input shape.
//
// Broadcast check (after padding):
//   for each dim d:
//     - if inDim == outDim => ok
//     - else if inDim == 1 => ok (broadcast)
//     - else => incompatible
//
// Index mapping:
//   outIdx[d] maps to inIdx[d] = 0 if inDim==1 else outIdx[d]
//   Using strides, broadcast dims can be treated as stride = 0.
//
// This demo materializes the output buffer for simplicity.
// -----------------------------------------------------------------------------

template <typename T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<T> data;

    Tensor() = default;
    Tensor(std::vector<int64_t> s) : shape(std::move(s)) {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
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
        int64_t dim = shape[i];
        outIdx[i] = linear % dim;
        linear /= dim;
    }
}

static std::vector<int64_t> padLeadingOnes(const std::vector<int64_t>& inShape, int64_t targetRank) {
    std::vector<int64_t> padded;
    padded.reserve(static_cast<size_t>(targetRank));
    int64_t pad = targetRank - static_cast<int64_t>(inShape.size());
    for (int64_t i = 0; i < pad; ++i) padded.push_back(1);
    padded.insert(padded.end(), inShape.begin(), inShape.end());
    return padded;
}

static void verifyBroadcastable(const std::vector<int64_t>& inPadded,
    const std::vector<int64_t>& outShape) {
    for (size_t i = 0; i < outShape.size(); ++i) {
        int64_t inDim = inPadded[i];
        int64_t outDim = outShape[i];
        if (outDim < 0) {
            throw std::runtime_error("Expand: target shape must be non-negative.");
        }
        if (!(inDim == outDim || inDim == 1)) {
            throw std::runtime_error("Expand: incompatible broadcast at dim " + std::to_string(i) +
                " (in=" + std::to_string(inDim) +
                ", out=" + std::to_string(outDim) + ")");
        }
    }
}

template <typename T>
Tensor<T> onnxExpand(const Tensor<T>& input, const std::vector<int64_t>& targetShape) {
    if (targetShape.empty())
        throw std::runtime_error("Expand: targetShape must be 1-D with length >= 1.");

    int64_t outRank = static_cast<int64_t>(targetShape.size());
    std::vector<int64_t> inPadded = padLeadingOnes(input.shape, outRank);
    verifyBroadcastable(inPadded, targetShape);

    // Compute original strides then map into padded strides (right aligned).
    std::vector<int64_t> inStridesOrig = computeStridesRowMajor(input.shape);
    std::vector<int64_t> inStridesPadded(outRank, 0);

    int64_t pad = outRank - input.rank();
    for (int64_t i = 0; i < input.rank(); ++i) {
        inStridesPadded[pad + i] = inStridesOrig[i];
    }

    // Broadcast dims: set stride = 0.
    for (int64_t d = 0; d < outRank; ++d) {
        if (inPadded[d] == 1 && targetShape[d] > 1) {
            inStridesPadded[d] = 0;
        }
    }

    Tensor<T> output(targetShape);
    std::vector<int64_t> outIdx;
    int64_t outNumel = output.numel();

    for (int64_t linear = 0; linear < outNumel; ++linear) {
        unravelIndexRowMajor(linear, targetShape, outIdx);

        int64_t inOffset = 0;
        for (int64_t d = 0; d < outRank; ++d) {
            inOffset += outIdx[d] * inStridesPadded[d];
        }
        output.data[static_cast<size_t>(linear)] = input.data[static_cast<size_t>(inOffset)];
    }

    return output;
}

// Print a (3,4) slice for the first demo.
static void print2D(const std::vector<float>& buf, int64_t H, int64_t W, const std::string& name) {
    std::cout << name << " (" << H << "x" << W << ")\n";
    for (int64_t i = 0; i < H; ++i) {
        std::cout << "  ";
        for (int64_t j = 0; j < W; ++j) {
            std::cout << buf[static_cast<size_t>(i * W + j)] << (j + 1 == W ? "" : " ");
        }
        std::cout << "\n";
    }
}

int main() {
    try {
        // --------------------- Example 1 (Correct) ---------------------
        // We want: (2,3,1) -> Expand -> (2,3,4)
        // This matches the common pattern: Unsqueeze then Expand.
        Tensor<float> A({ 2, 3, 1 });
        // Fill A so that each (i,k,0) has unique values:
        // i=0: 1,2,3 ; i=1: 4,5,6
        A.data = {
          1, 2, 3,   // i=0, k=0..2
          4, 5, 6    // i=1, k=0..2
        };

        std::vector<int64_t> target1 = { 2, 3, 4 };
        auto Y1 = onnxExpand(A, target1);

        std::cout << "Example1: A(2,3,1) -> Expand to (2,3,4)\n";
        std::cout << "Y1 shape=(" << Y1.shape[0] << "," << Y1.shape[1] << "," << Y1.shape[2] << ")\n";

        // Print Y1[0,:,:] and Y1[1,:,:], each is (3,4).
        // Row-major layout for (2,3,4): each i slice has 3*4 = 12 elements.
        std::vector<float> slice0(Y1.data.begin(), Y1.data.begin() + 12);
        std::vector<float> slice1(Y1.data.begin() + 12, Y1.data.begin() + 24);

        print2D(slice0, 3, 4, "Y1[0,:,:]");
        print2D(slice1, 3, 4, "Y1[1,:,:]");

        // --------------------- Example 2 (Correct) ---------------------
        // (1,3) -> Expand -> (2,3)
        Tensor<float> B({ 1, 3 });
        B.data = { 10, 20, 30 };
        std::vector<int64_t> target2 = { 2, 3 };
        auto Y2 = onnxExpand(B, target2);

        std::cout << "\nExample2: B(1,3) -> Expand to (2,3)\n";
        std::cout << "Y2 shape=(" << Y2.shape[0] << "," << Y2.shape[1] << ")\n";
        // Print Y2 as 2x3
        print2D(Y2.data, 2, 3, "Y2");

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
