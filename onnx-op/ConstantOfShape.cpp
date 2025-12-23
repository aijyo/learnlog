#include <cstdint>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <type_traits>

// -------------------- TensorND --------------------
template <typename T>
struct TensorND {
    std::vector<int64_t> shape; // rank-0 scalar uses empty shape {}
    std::vector<T> data;        // row-major contiguous

    int64_t numElements() const {
        if (shape.empty()) return 1; // scalar has 1 element
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

// -------------------- Helpers --------------------
static inline int64_t SafeProduct(const std::vector<int64_t>& dims) {
    // Compute product with basic overflow/validity checks.
    if (dims.empty()) return 1; // scalar
    int64_t prod = 1;
    for (int64_t d : dims) {
        if (d < 0) throw std::invalid_argument("ConstantOfShape: negative dimension is invalid");
        // If any dim is 0, total elements is 0 (empty tensor).
        if (d == 0) return 0;
        // Basic overflow guard for int64 product (optional but safer).
        if (prod > (INT64_MAX / d)) {
            throw std::overflow_error("ConstantOfShape: element count overflow");
        }
        prod *= d;
    }
    return prod;
}

// -------------------- ConstantOfShape --------------------
// This is a runtime-style implementation:
// - inputShapeValues: the CONTENTS of the 1D input tensor (int64), describing output dims.
// - fillValue: scalar value to fill output.
// Returns: output tensor with shape=inputShapeValues and all elements=fillValue.
template <typename T>
TensorND<T> ConstantOfShape(const std::vector<int64_t>& inputShapeValues, const T& fillValue) {
    static_assert(std::is_arithmetic<T>::value || std::is_same<T, bool>::value,
        "ConstantOfShape: T must be arithmetic or bool");

    TensorND<T> out;

    // If input length is 0, output is scalar (rank 0).
    // This matches the common interpretation in ONNX-MLIR shape helper:
    // outputDims.clear() => scalar.
    if (inputShapeValues.empty()) {
        out.shape = {};        // scalar
        out.data.resize(1);
        out.data[0] = fillValue;
        return out;
    }

    // Otherwise output shape is exactly the input values.
    out.shape = inputShapeValues;

    // Validate dims and allocate data.
    const int64_t n = SafeProduct(out.shape);
    out.data.resize((size_t)n);

    // Fill with fillValue.
    for (int64_t i = 0; i < n; ++i) {
        out.data[(size_t)i] = fillValue;
    }

    return out;
}

// -------------------- Pretty Print --------------------
static void PrintShape(const std::vector<int64_t>& s) {
    std::cout << "[";
    for (size_t i = 0; i < s.size(); ++i) {
        std::cout << s[i] << (i + 1 == s.size() ? "" : ", ");
    }
    std::cout << "]";
}

template <typename T>
static void PrintTensorFlat(const TensorND<T>& t, const char* name, int maxPrint = 64) {
    std::cout << name << " shape=";
    PrintShape(t.shape);
    std::cout << " numel=" << t.numElements() << " data={";

    const int64_t n = t.numElements();
    const int64_t m = (n < maxPrint) ? n : maxPrint;
    for (int64_t i = 0; i < m; ++i) {
        std::cout << t.data[(size_t)i];
        if (i + 1 != m) std::cout << ", ";
    }
    if (n > m) std::cout << ", ...";
    std::cout << "}\n";
}

// -------------------- Demo --------------------
static void DemoConstantOfShape() {
    std::cout << "=== Demo 1: normal shape [2,3], fill=7 (int32) ===\n";
    {
        std::vector<int64_t> shapeVals = { 2, 3 };
        auto y = ConstantOfShape<int32_t>(shapeVals, 7);
        PrintTensorFlat(y, "Y");
        // Expected: shape=[2,3], data=7 repeated 6 times
    }

    std::cout << "\n=== Demo 2: empty shape vector => scalar, fill=0.5 (float) ===\n";
    {
        std::vector<int64_t> shapeVals = {}; // input length = 0
        auto y = ConstantOfShape<float>(shapeVals, 0.5f);
        PrintTensorFlat(y, "Y");
        // Expected: shape=[], numel=1, data={0.5}
    }

    std::cout << "\n=== Demo 3: shape contains zero [2,0,4], fill=1 (int64) => empty tensor ===\n";
    {
        std::vector<int64_t> shapeVals = { 2, 0, 4 };
        auto y = ConstantOfShape<int64_t>(shapeVals, 1);
        PrintTensorFlat(y, "Y");
        // Expected: shape=[2,0,4], numel=0, data={}
    }

    std::cout << "\n=== Demo 4: invalid negative dim [2,-1] should throw ===\n";
    {
        try {
            std::vector<int64_t> shapeVals = { 2, -1 };
            auto y = ConstantOfShape<int>(shapeVals, 3);
            (void)y;
        }
        catch (const std::exception& e) {
            std::cout << "Caught exception as expected: " << e.what() << "\n";
        }
    }
}

int main() {
    DemoConstantOfShape();
    return 0;
}
