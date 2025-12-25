#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

// --------------------------- LRN (Local Response Normalization) ---------------------------
//
// ONNX-style LRN normalizes each element by a local neighborhood across the channel dimension.
//
// Given input X of shape [N, C, D1, D2, ...] (rank >= 2):
//   spatialSize = D1 * D2 * ...
//
// For each n in [0..N), each spatial position s in [0..spatialSize),
// and each channel c in [0..C):
//
//   sumsq = sum_{j in window(c)} X[n, j, s]^2
//   denom = (bias + (alpha / size) * sumsq) ^ beta
//   Y[n, c, s] = X[n, c, s] / denom
//
// where window(c) = [c - floor(size/2), c + floor(size/2)] clipped to [0, C-1].
//
// This implementation uses a sliding-window sum of squares along channel axis for each
// (n, s) to achieve O(N * spatialSize * C) time.
//

template <typename T>
struct Tensor {
  std::vector<int64_t> shape; // [N, C, ...]
  std::vector<T> data;        // row-major contiguous
};

static int64_t numel(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// Convert (n, c, s) into linear offset for a tensor logically viewed as [N, C, spatialSize].
static inline int64_t offsetNCS(int64_t n, int64_t c, int64_t s,
                                int64_t C, int64_t spatialSize) {
  return (n * C + c) * spatialSize + s;
}

template <typename T>
Tensor<T> lrn(const Tensor<T> &x,
              int64_t size,
              float alpha,
              float beta,
              float bias) {
  assert(x.shape.size() >= 2 && "LRN expects rank >= 2");
  assert(size > 0 && "size must be > 0");

  const int64_t N = x.shape[0];
  const int64_t C = x.shape[1];

  int64_t spatialSize = 1;
  for (size_t i = 2; i < x.shape.size(); ++i) spatialSize *= x.shape[i];

  assert((int64_t)x.data.size() == N * C * spatialSize && "data size mismatch");

  Tensor<T> y;
  y.shape = x.shape;
  y.data.resize((size_t)(N * C * spatialSize));

  const int64_t half = size / 2;
  const float alphaOverSize = alpha / (float)size;

  // For each (n, spatial position), compute sliding sum of squares over channels.
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t s = 0; s < spatialSize; ++s) {

      // Initialize window sum for c=0: channels [0 .. min(C-1, half)]
      float windowSumSq = 0.0f;
      const int64_t initRight = std::min<int64_t>(C - 1, half);
      for (int64_t j = 0; j <= initRight; ++j) {
        const float v = (float)x.data[(size_t)offsetNCS(n, j, s, C, spatialSize)];
        windowSumSq += v * v;
      }

      for (int64_t c = 0; c < C; ++c) {
        // Compute output for current channel c.
        const float xc = (float)x.data[(size_t)offsetNCS(n, c, s, C, spatialSize)];
        const float denom = std::pow(bias + alphaOverSize * windowSumSq, beta);
        y.data[(size_t)offsetNCS(n, c, s, C, spatialSize)] = (T)(xc / denom);

        // Slide window from center c to c+1:
        // Old window: [c-half .. c+half]
        // New window: [c+1-half .. c+1+half]
        const int64_t outLeft = c - half;
        const int64_t inRight = c + half + 1;

        if (outLeft >= 0) {
          const float v = (float)x.data[(size_t)offsetNCS(n, outLeft, s, C, spatialSize)];
          windowSumSq -= v * v;
        }
        if (inRight < C) {
          const float v = (float)x.data[(size_t)offsetNCS(n, inRight, s, C, spatialSize)];
          windowSumSq += v * v;
        }
      }
    }
  }
  return y;
}

// Pretty print a [N,C,H,W] tensor for demo.
template <typename T>
void printNCHW(const Tensor<T> &t, const std::string &name) {
  assert(t.shape.size() == 4);
  const int64_t N = t.shape[0], C = t.shape[1], H = t.shape[2], W = t.shape[3];
  std::cout << name << " shape=[" << N << "," << C << "," << H << "," << W << "]\n";
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      std::cout << "  n=" << n << " c=" << c << ":\n";
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t w = 0; w < W; ++w) {
          const int64_t s = h * W + w;
          std::cout << t.data[(size_t)offsetNCS(n, c, s, C, H * W)] << " ";
        }
        std::cout << "\n";
      }
    }
  }
}

int main() {
  // Demo input: N=1, C=3, H=2, W=2
  Tensor<float> x;
  x.shape = {1, 3, 2, 2};
  x.data.resize((size_t)numel(x.shape));

  // Fill with simple values 1..12 in NCHW order:
  // [n=0,c=0] 1 2 3 4
  // [n=0,c=1] 5 6 7 8
  // [n=0,c=2] 9 10 11 12
  std::iota(x.data.begin(), x.data.end(), 1.0f);

  printNCHW(x, "X");

  // Typical AlexNet-like params: size=5, alpha=1e-4, beta=0.75, bias=1.0
  // For this tiny C=3, the window will be clipped by channel boundaries.
  auto y = lrn(x, /*size=*/5, /*alpha=*/1e-4f, /*beta=*/0.75f, /*bias=*/1.0f);

  printNCHW(y, "Y (LRN)");

  return 0;
}
