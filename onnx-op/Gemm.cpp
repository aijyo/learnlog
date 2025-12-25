#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// --------------------------- Gemm (ONNX-style) ---------------------------
//
// Computes Y = alpha * (A' * B') + beta * C
//
// - A and B must be 2D matrices.
// - A' is A if transA==0, otherwise A^T.
// - B' is B if transB==0, otherwise B^T.
// - Output Y has shape (M x N), where A' is (M x K) and B' is (K x N).
//
// Bias C is optional and supports "unidirectional broadcast" (as in onnx-mlir helper):
//   * scalar (rank=0) treated as (1 x 1)
//   * 1D (rank=1) treated as (1 x N)  (first dim padded to 1)
//   * 2D (rank=2) treated as (M x N)
// Allowed broadcast per dim: cDim == 1 or cDim == outDim.
//
// This implementation is a CPU reference kernel (row-major).
// -------------------------------------------------------------------------

struct Tensor2D {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> data; // row-major

  Tensor2D() = default;
  Tensor2D(std::size_t r, std::size_t c, float init = 0.0f)
      : rows(r), cols(c), data(r * c, init) {}

  float &operator()(std::size_t r, std::size_t c) {
    return data[r * cols + c];
  }
  float operator()(std::size_t r, std::size_t c) const {
    return data[r * cols + c];
  }
};

struct BiasTensor {
  // rank = 0: scalar stored in scalar
  // rank = 1: vec length = n  (treated as 1 x n)
  // rank = 2: mat (r x c)
  int rank = 0;
  float scalar = 0.0f;
  std::vector<float> vec; // rank=1
  Tensor2D mat;           // rank=2

  static BiasTensor Scalar(float v) {
    BiasTensor b;
    b.rank = 0;
    b.scalar = v;
    return b;
  }

  static BiasTensor Vector(std::vector<float> v) {
    BiasTensor b;
    b.rank = 1;
    b.vec = std::move(v);
    return b;
  }

  static BiasTensor Matrix(Tensor2D m) {
    BiasTensor b;
    b.rank = 2;
    b.mat = std::move(m);
    return b;
  }
};

static inline std::pair<std::size_t, std::size_t>
effectiveShapeA(const Tensor2D &A, int transA) {
  // English comment: return shape of A' (after transpose decision)
  if (transA == 0)
    return {A.rows, A.cols};
  return {A.cols, A.rows};
}

static inline std::pair<std::size_t, std::size_t>
effectiveShapeB(const Tensor2D &B, int transB) {
  // English comment: return shape of B' (after transpose decision)
  if (transB == 0)
    return {B.rows, B.cols};
  return {B.cols, B.rows};
}

static inline float readA(const Tensor2D &A, int transA,
                          std::size_t i, std::size_t k) {
  // English comment: read A'(i,k)
  if (transA == 0)
    return A(i, k);
  return A(k, i);
}

static inline float readB(const Tensor2D &B, int transB,
                          std::size_t k, std::size_t j) {
  // English comment: read B'(k,j)
  if (transB == 0)
    return B(k, j);
  return B(j, k);
}

static inline std::pair<std::size_t, std::size_t>
biasAs2DShape(const BiasTensor &C) {
  // English comment: convert rank 0/1/2 bias into a 2D "virtual" shape
  if (C.rank == 0)
    return {1, 1};
  if (C.rank == 1)
    return {1, C.vec.size()};
  if (C.rank == 2)
    return {C.mat.rows, C.mat.cols};
  throw std::invalid_argument("Unsupported bias rank");
}

static inline float readBiasBroadcast(const BiasTensor &C,
                                      std::size_t outM, std::size_t outN,
                                      std::size_t i, std::size_t j) {
  // English comment: read bias value with broadcast semantics for output (outM x outN)
  if (C.rank == 0) {
    return C.scalar;
  } else if (C.rank == 1) {
    // treated as (1 x N)
    if (C.vec.empty())
      throw std::invalid_argument("Bias vector is empty");
    // broadcast in first dim always; second dim either matches N or is 1
    std::size_t n = C.vec.size();
    if (n == 1)
      return C.vec[0];
    if (j >= n)
      throw std::invalid_argument("Bias vector length < output N");
    return C.vec[j];
  } else if (C.rank == 2) {
    std::size_t r = C.mat.rows;
    std::size_t c = C.mat.cols;
    std::size_t bi = (r == 1) ? 0 : i;
    std::size_t bj = (c == 1) ? 0 : j;
    if (bi >= r || bj >= c)
      throw std::invalid_argument("Bias matrix shape incompatible with output");
    return C.mat(bi, bj);
  }
  throw std::invalid_argument("Unsupported bias rank");
}

static inline void verifyBiasCompatible(const BiasTensor &C,
                                        std::size_t outM, std::size_t outN) {
  // English comment: verify unidirectional broadcast: each dim is 1 or equals output dim
  auto [cM, cN] = biasAs2DShape(C);

  auto okDim = [](std::size_t cd, std::size_t od) {
    return (cd == 1) || (cd == od);
  };

  if (!okDim(cM, outM) || !okDim(cN, outN)) {
    throw std::invalid_argument("Bias add has bad dimension (broadcast mismatch)");
  }
  // Extra check for rank=1: treated as (1 x N), so N must be 1 or outN.
  if (C.rank == 1) {
    if (!(C.vec.size() == 1 || C.vec.size() == outN)) {
      throw std::invalid_argument("Bias vector length must be 1 or output N");
    }
  }
}

Tensor2D Gemm(const Tensor2D &A, const Tensor2D &B,
              const std::optional<BiasTensor> &C,
              float alpha = 1.0f, float beta = 1.0f,
              int transA = 0, int transB = 0) {
  // English comment: validate trans flags
  if (!((transA == 0) || (transA == 1)))
    throw std::invalid_argument("transA must be 0 or 1");
  if (!((transB == 0) || (transB == 1)))
    throw std::invalid_argument("transB must be 0 or 1");

  // English comment: compute effective shapes
  auto [m, kA] = effectiveShapeA(A, transA);
  auto [kB, n] = effectiveShapeB(B, transB);

  // English comment: K dimension must match
  if (kA != kB) {
    throw std::invalid_argument("Gemm: inner dimension mismatch (A' cols != B' rows)");
  }
  std::size_t k = kA;

  // English comment: verify bias broadcast compatibility if present
  if (C.has_value()) {
    verifyBiasCompatible(*C, m, n);
  }

  Tensor2D Y(m, n, 0.0f);

  // English comment: naive triple loop GEMM (reference)
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (std::size_t kk = 0; kk < k; ++kk) {
        acc += readA(A, transA, i, kk) * readB(B, transB, kk, j);
      }
      float y = alpha * acc;
      if (C.has_value()) {
        float cval = readBiasBroadcast(*C, m, n, i, j);
        y += beta * cval;
      }
      Y(i, j) = y;
    }
  }
  return Y;
}

static void printMat(const Tensor2D &T, const std::string &name) {
  std::cout << name << " (" << T.rows << "x" << T.cols << ")\n";
  for (std::size_t i = 0; i < T.rows; ++i) {
    for (std::size_t j = 0; j < T.cols; ++j) {
      std::cout << std::setw(8) << T(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main() {
  try {
    // Example 1: Y = A * B + C(1D)  where C is length N (broadcast to 1xN)
    Tensor2D A(2, 3);
    Tensor2D B(3, 4);

    // Fill A
    // [1 2 3
    //  4 5 6]
    float aVal = 1.0f;
    for (std::size_t i = 0; i < A.rows; ++i)
      for (std::size_t j = 0; j < A.cols; ++j)
        A(i, j) = aVal++;

    // Fill B
    // 3x4
    float bVal = 1.0f;
    for (std::size_t i = 0; i < B.rows; ++i)
      for (std::size_t j = 0; j < B.cols; ++j)
        B(i, j) = bVal++;

    // Bias vector length N=4
    BiasTensor C1 = BiasTensor::Vector({10.f, 20.f, 30.f, 40.f});

    Tensor2D Y1 = Gemm(A, B, C1, /*alpha=*/1.0f, /*beta=*/1.0f,
                       /*transA=*/0, /*transB=*/0);

    printMat(A, "A");
    printMat(B, "B");
    printMat(Y1, "Y1 = A*B + C(1D)");

    // Example 2: Typical FC pattern: Y = X * W^T + b
    // X: (batch=2, in=3), W: (out=4, in=3) => W^T: (3,4)
    Tensor2D X = A; // reuse A as X(2x3)
    Tensor2D W(4, 3);
    float wVal = 0.5f;
    for (std::size_t i = 0; i < W.rows; ++i)
      for (std::size_t j = 0; j < W.cols; ++j)
        W(i, j) = wVal += 0.5f;

    BiasTensor b = BiasTensor::Vector({1.f, 1.f, 1.f, 1.f});

    // transB=1 means B' = W^T (because B is W here)
    Tensor2D Y2 = Gemm(X, W, b, /*alpha=*/1.0f, /*beta=*/1.0f,
                       /*transA=*/0, /*transB=*/1);

    printMat(W, "W (out x in)");
    printMat(Y2, "Y2 = X * W^T + b");

    // Example 3: scalar bias and scaling: Y = 0.1*(A*B) + 2.0*Cscalar
    BiasTensor Cscalar = BiasTensor::Scalar(3.0f);
    Tensor2D Y3 = Gemm(A, B, Cscalar, /*alpha=*/0.1f, /*beta=*/2.0f,
                       /*transA=*/0, /*transB=*/0);
    printMat(Y3, "Y3 = 0.1*(A*B) + 2.0*3.0");

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
