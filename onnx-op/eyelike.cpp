#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <algorithm>

// ------------------------------ EyeLike (ONNX) ------------------------------
//
// What EyeLike does:
//   Given a 2-D input tensor X with shape [N, M], EyeLike returns Y with the
//   same shape [N, M]. Y is an identity-like matrix:
//
//     Y[i,j] = 1 if j == i + k
//              0 otherwise
//
// Attributes:
//   - k (int): diagonal offset (default 0)
//       k = 0  -> main diagonal
//       k > 0  -> upper diagonal
//       k < 0  -> lower diagonal
//   - dtype: output element type (optional). If not provided, use input type.
//
// Shape inference (as in onnx-mlir snippet):
//   output shape == input shape.
//
// This demo materializes output buffer and ignores input values.
// ---------------------------------------------------------------------------

// A tiny tensor helper for demo.
template <typename T>
struct Tensor2D {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<T> data;

    Tensor2D() = default;
    Tensor2D(int64_t r, int64_t c) : rows(r), cols(c), data(static_cast<size_t>(r* c)) {}

    T& operator()(int64_t i, int64_t j) { return data[static_cast<size_t>(i * cols + j)]; }
    const T& operator()(int64_t i, int64_t j) const { return data[static_cast<size_t>(i * cols + j)]; }
};

template <typename T>
static void printTensor(const Tensor2D<T>& t, const std::string& name) {
    std::cout << name << " shape=(" << t.rows << "," << t.cols << ")\n";
    for (int64_t i = 0; i < t.rows; ++i) {
        std::cout << "  ";
        for (int64_t j = 0; j < t.cols; ++j) {
            std::cout << t(i, j) << (j + 1 == t.cols ? "" : " ");
        }
        std::cout << "\n";
    }
}

// Core EyeLike kernel.
template <typename T>
Tensor2D<T> onnxEyeLike(const Tensor2D<T>& input, int64_t k = 0) {
    if (input.rows <= 0 || input.cols <= 0) {
        throw std::runtime_error("EyeLike: input shape must be positive.");
    }

    Tensor2D<T> output(input.rows, input.cols);

    // Fill zeros.
    std::fill(output.data.begin(), output.data.end(), static_cast<T>(0));

    // Set ones on the (offset) diagonal.
    // Condition: j == i + k  <=>  i == j - k
    // Iterate over i and compute j, check bounds.
    for (int64_t i = 0; i < input.rows; ++i) {
        int64_t j = i + k;
        if (j >= 0 && j < input.cols) {
            output(i, j) = static_cast<T>(1);
        }
    }

    return output;
}

int main() {
    try {
        // Input tensor values do not matter for EyeLike; only shape matters.
        Tensor2D<float> X(3, 5);
        // Fill input with some numbers (just for showing it exists).
        for (int64_t i = 0; i < X.rows; ++i) {
            for (int64_t j = 0; j < X.cols; ++j) {
                X(i, j) = static_cast<float>(i * 10 + j);
            }
        }

        std::cout << "Input (values ignored by EyeLike):\n";
        printTensor(X, "X");

        // Example 1: k = 0 (main diagonal)
        auto Y0 = onnxEyeLike(X, /*k=*/0);
        std::cout << "\nEyeLike k=0:\n";
        printTensor(Y0, "Y0");

        // Example 2: k = 1 (upper diagonal)
        auto Y1 = onnxEyeLike(X, /*k=*/1);
        std::cout << "\nEyeLike k=1:\n";
        printTensor(Y1, "Y1");

        // Example 3: k = -2 (lower diagonal)
        auto Ym2 = onnxEyeLike(X, /*k=*/-2);
        std::cout << "\nEyeLike k=-2:\n";
        printTensor(Ym2, "Ym2");

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
