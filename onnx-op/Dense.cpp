// dense_demo.cpp
// A minimal, self-contained Dense (Fully Connected) operator implementation
// with a runnable example.
//
// Dense math:
//   Y = X * W + b
// Shapes:
//   X: (N, K)
//   W: (K, M)
//   b: (M) optional
//   Y: (N, M)
//
// Notes:
// - Row-major storage.
// - Naive CPU implementation (triple loop).
// - Bias is optional.
// - No activation (ReLU/GELU) applied here.

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ------------------------------- Tensor2D ------------------------------------
// Simple 2D tensor container with row-major storage.
// -----------------------------------------------------------------------------
template <typename T>
struct Tensor2D {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<T> data;

    Tensor2D() = default;

    Tensor2D(int64_t r, int64_t c) : rows(r), cols(c) {
        if (r <= 0 || c <= 0) {
            throw std::runtime_error("Tensor2D: invalid shape (rows/cols must be > 0).");
        }
        data.resize(static_cast<size_t>(r * c));
    }

    T& operator()(int64_t i, int64_t j) {
        return data[static_cast<size_t>(i * cols + j)];
    }

    const T& operator()(int64_t i, int64_t j) const {
        return data[static_cast<size_t>(i * cols + j)];
    }
};

// ------------------------------- Dense Op ------------------------------------
//
// Dense(X, W, b):
//   Y[i,j] = sum_k X[i,k] * W[k,j] + b[j]
//
// -----------------------------------------------------------------------------
template <typename T>
Tensor2D<T> DenseOp(const Tensor2D<T>& X,
    const Tensor2D<T>& W,
    const std::vector<T>* bias /*nullable*/) {
    // Shape checks
    if (X.cols != W.rows) {
        throw std::runtime_error("DenseOp: shape mismatch (X.cols must equal W.rows).");
    }
    if (bias && static_cast<int64_t>(bias->size()) != W.cols) {
        throw std::runtime_error("DenseOp: bias size mismatch (bias.size must equal W.cols).");
    }

    const int64_t N = X.rows;
    const int64_t K = X.cols;
    const int64_t M = W.cols;

    Tensor2D<T> Y(N, M);

    // Naive GEMM + bias
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            T acc = static_cast<T>(0);
            for (int64_t k = 0; k < K; ++k) {
                acc += X(i, k) * W(k, j);
            }
            if (bias) {
                acc += (*bias)[static_cast<size_t>(j)];
            }
            Y(i, j) = acc;
        }
    }

    return Y;
}

// ------------------------------- Utilities -----------------------------------
template <typename T>
void PrintTensor(const Tensor2D<T>& t, const std::string& name) {
    std::cout << name << " shape=(" << t.rows << "," << t.cols << ")\n";
    for (int64_t i = 0; i < t.rows; ++i) {
        std::cout << "  ";
        for (int64_t j = 0; j < t.cols; ++j) {
            std::cout << t(i, j) << (j + 1 == t.cols ? "" : " ");
        }
        std::cout << "\n";
    }
}

// ---------------------------------- main -------------------------------------
// Runnable example:
//   X: (2,3)
//   W: (3,4)
//   b: (4)
// Expected:
//   X*W = [[1,2,3,6], [4,5,6,15]]
//   +b  = [[2,4,6,10], [5,7,9,19]]
// -----------------------------------------------------------------------------
int main() {
    try {
        Tensor2D<float> X(2, 3);
        Tensor2D<float> W(3, 4);
        std::vector<float> b = { 1.0f, 2.0f, 3.0f, 4.0f };

        // Fill X:
        // [1 2 3]
        // [4 5 6]
        X(0, 0) = 1; X(0, 1) = 2; X(0, 2) = 3;
        X(1, 0) = 4; X(1, 1) = 5; X(1, 2) = 6;

        // Fill W:
        // [1 0 0 1]
        // [0 1 0 1]
        // [0 0 1 1]
        for (int64_t i = 0; i < W.rows; ++i) {
            for (int64_t j = 0; j < W.cols; ++j) {
                W(i, j) = 0.0f;
            }
        }
        W(0, 0) = 1; W(0, 3) = 1;
        W(1, 1) = 1; W(1, 3) = 1;
        W(2, 2) = 1; W(2, 3) = 1;

        PrintTensor(X, "X");
        PrintTensor(W, "W");

        // Dense with bias
        auto Y = DenseOp<float>(X, W, &b);
        PrintTensor(Y, "Y = DenseOp(X, W, b)");

        // Dense without bias (optional demo)
        auto Y_no_bias = DenseOp<float>(X, W, nullptr);
        PrintTensor(Y_no_bias, "Y_no_bias = DenseOp(X, W, nullptr)");

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
