#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// --------------------------- GridSample (ONNX-style) ---------------------------
//
// 2D specialization (most common):
//   X    : (N, C, H, W)
//   Grid : (N, H_out, W_out, 2) where last dim = (x_norm, y_norm), typically in [-1, 1]
//   Y    : (N, C, H_out, W_out)
//
// Coordinate transform (normalized -> input continuous index):
//   if align_corners == 1:
//       x = ((x_norm + 1) / 2) * (W - 1)
//       y = ((y_norm + 1) / 2) * (H - 1)
//   else:
//       x = ((x_norm + 1) * W - 1) / 2
//       y = ((y_norm + 1) * H - 1) / 2
//
// mode:
//   - nearest : nearest neighbor sampling
//   - linear  : bilinear sampling
//   - cubic   : bicubic sampling (cubic convolution kernel, a = -0.75)
//
// padding_mode:
//   - zeros      : out-of-bound reads as 0
//   - border     : clamp coordinates to border
//   - reflection : reflect coordinates at borders
// -----------------------------------------------------------------------------


struct Tensor4D {
  // Layout: NCHW contiguous
  int64_t N = 0, C = 0, H = 0, W = 0;
  std::vector<float> data;

  Tensor4D() = default;
  Tensor4D(int64_t n, int64_t c, int64_t h, int64_t w, float init = 0.0f)
      : N(n), C(c), H(h), W(w), data(static_cast<size_t>(n * c * h * w), init) {}

  inline float &at(int64_t n, int64_t c, int64_t y, int64_t x) {
    size_t idx = static_cast<size_t>(((n * C + c) * H + y) * W + x);
    return data[idx];
  }
  inline float at(int64_t n, int64_t c, int64_t y, int64_t x) const {
    size_t idx = static_cast<size_t>(((n * C + c) * H + y) * W + x);
    return data[idx];
  }
};

struct Grid4D {
  // Layout: NHW2 contiguous: (N, H_out, W_out, 2)
  int64_t N = 0, Hout = 0, Wout = 0;
  std::vector<float> data; // last dim = 2

  Grid4D() = default;
  Grid4D(int64_t n, int64_t hout, int64_t wout)
      : N(n), Hout(hout), Wout(wout),
        data(static_cast<size_t>(n * hout * wout * 2), 0.0f) {}

  inline float &at(int64_t n, int64_t y, int64_t x, int64_t k) {
    // k: 0 -> x_norm, 1 -> y_norm
    size_t idx = static_cast<size_t>(((n * Hout + y) * Wout + x) * 2 + k);
    return data[idx];
  }
  inline float at(int64_t n, int64_t y, int64_t x, int64_t k) const {
    size_t idx = static_cast<size_t>(((n * Hout + y) * Wout + x) * 2 + k);
    return data[idx];
  }
};

enum class Mode { Linear, Nearest, Cubic };
enum class PaddingMode { Zeros, Border, Reflection };

static inline float unnormalize(float v_norm, int64_t size, bool align_corners) {
  // English comment: map normalized [-1,1] to continuous input coordinate.
  if (align_corners) {
    // size==1 corner case: all coords map to 0
    if (size <= 1) return 0.0f;
    return (v_norm + 1.0f) * 0.5f * static_cast<float>(size - 1);
  } else {
    return ((v_norm + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
  }
}

static inline int64_t reflect_index(int64_t x, int64_t low, int64_t high) {
  // English comment:
  // Reflect integer index into [low, high] by mirroring at the boundaries.
  // Example for [0, W-1]:
  //   -1 -> 0, -2 -> 1, W -> W-1, W+1 -> W-2, ...
  if (high < low) return low;
  int64_t range = high - low;
  if (range == 0) return low;

  // Shift to [0, range]
  int64_t t = x - low;
  int64_t period = 2 * range;

  // Proper modulo for negatives
  int64_t m = t % period;
  if (m < 0) m += period;

  if (m <= range) {
    return low + m;
  } else {
    return high - (m - range);
  }
}

static inline float get_value_with_padding(
    const Tensor4D &X, int64_t n, int64_t c, int64_t y, int64_t x,
    PaddingMode padding) {
  // English comment: handle out-of-bound reads according to padding_mode.
  if (x >= 0 && x < X.W && y >= 0 && y < X.H) {
    return X.at(n, c, y, x);
  }

  if (padding == PaddingMode::Zeros) {
    return 0.0f;
  } else if (padding == PaddingMode::Border) {
    int64_t xx = std::min<int64_t>(std::max<int64_t>(x, 0), X.W - 1);
    int64_t yy = std::min<int64_t>(std::max<int64_t>(y, 0), X.H - 1);
    return X.at(n, c, yy, xx);
  } else { // Reflection
    int64_t xx = reflect_index(x, 0, X.W - 1);
    int64_t yy = reflect_index(y, 0, X.H - 1);
    return X.at(n, c, yy, xx);
  }
}

static inline float cubic_kernel(float x, float a) {
  // English comment:
  // Cubic convolution kernel (Keys). Common choice in grid_sample: a = -0.75.
  float ax = std::fabs(x);
  float ax2 = ax * ax;
  float ax3 = ax2 * ax;

  if (ax <= 1.0f) {
    return (a + 2.0f) * ax3 - (a + 3.0f) * ax2 + 1.0f;
  } else if (ax < 2.0f) {
    return a * ax3 - 5.0f * a * ax2 + 8.0f * a * ax - 4.0f * a;
  }
  return 0.0f;
}

static inline float sample_nearest(
    const Tensor4D &X, int64_t n, int64_t c, float y, float x,
    PaddingMode padding) {
  // English comment: nearest neighbor sampling.
  int64_t xi = static_cast<int64_t>(std::nearbyint(x));
  int64_t yi = static_cast<int64_t>(std::nearbyint(y));
  return get_value_with_padding(X, n, c, yi, xi, padding);
}

static inline float sample_bilinear(
    const Tensor4D &X, int64_t n, int64_t c, float y, float x,
    PaddingMode padding) {
  // English comment: bilinear interpolation.
  int64_t x0 = static_cast<int64_t>(std::floor(x));
  int64_t y0 = static_cast<int64_t>(std::floor(y));
  int64_t x1 = x0 + 1;
  int64_t y1 = y0 + 1;

  float wx = x - static_cast<float>(x0);
  float wy = y - static_cast<float>(y0);

  float v00 = get_value_with_padding(X, n, c, y0, x0, padding);
  float v01 = get_value_with_padding(X, n, c, y0, x1, padding);
  float v10 = get_value_with_padding(X, n, c, y1, x0, padding);
  float v11 = get_value_with_padding(X, n, c, y1, x1, padding);

  float v0 = v00 * (1.0f - wx) + v01 * wx;
  float v1 = v10 * (1.0f - wx) + v11 * wx;
  return v0 * (1.0f - wy) + v1 * wy;
}

static inline float sample_bicubic(
    const Tensor4D &X, int64_t n, int64_t c, float y, float x,
    PaddingMode padding) {
  // English comment:
  // Bicubic interpolation: 4x4 neighborhood with cubic convolution weights.
  // Common a used by PyTorch grid_sample: a = -0.75.
  constexpr float a = -0.75f;

  int64_t x_int = static_cast<int64_t>(std::floor(x));
  int64_t y_int = static_cast<int64_t>(std::floor(y));
  float dx = x - static_cast<float>(x_int);
  float dy = y - static_cast<float>(y_int);

  float sum = 0.0f;
  float wsum = 0.0f;

  for (int m = -1; m <= 2; ++m) {
    float wy = cubic_kernel(static_cast<float>(m) - dy, a);
    int64_t yy = y_int + m;
    for (int k = -1; k <= 2; ++k) {
      float wx = cubic_kernel(static_cast<float>(k) - dx, a);
      int64_t xx = x_int + k;
      float w = wx * wy;
      float v = get_value_with_padding(X, n, c, yy, xx, padding);
      sum += w * v;
      wsum += w;
    }
  }

  // English comment: wsum is usually ~1 for in-bound; keep safe for extreme cases.
  if (wsum != 0.0f) sum /= wsum;
  return sum;
}

Tensor4D GridSample2D(
    const Tensor4D &X,
    const Grid4D &G,
    Mode mode,
    PaddingMode padding_mode,
    bool align_corners) {

  if (X.N != G.N) {
    throw std::invalid_argument("GridSample2D: X and Grid batch (N) must match.");
  }
  Tensor4D Y(X.N, X.C, G.Hout, G.Wout, 0.0f);

  for (int64_t n = 0; n < X.N; ++n) {
    for (int64_t ho = 0; ho < G.Hout; ++ho) {
      for (int64_t wo = 0; wo < G.Wout; ++wo) {
        float x_norm = G.at(n, ho, wo, 0);
        float y_norm = G.at(n, ho, wo, 1);

        float x = unnormalize(x_norm, X.W, align_corners);
        float y = unnormalize(y_norm, X.H, align_corners);

        for (int64_t c = 0; c < X.C; ++c) {
          float out = 0.0f;
          if (mode == Mode::Nearest) {
            out = sample_nearest(X, n, c, y, x, padding_mode);
          } else if (mode == Mode::Linear) {
            out = sample_bilinear(X, n, c, y, x, padding_mode);
          } else { // Cubic
            out = sample_bicubic(X, n, c, y, x, padding_mode);
          }
          Y.at(n, c, ho, wo) = out;
        }
      }
    }
  }
  return Y;
}

static void print_nchw(const Tensor4D &T, const std::string &name) {
  std::cout << name << " (N,C,H,W)=(" << T.N << "," << T.C << "," << T.H << "," << T.W << ")\n";
  for (int64_t n = 0; n < T.N; ++n) {
    for (int64_t c = 0; c < T.C; ++c) {
      std::cout << "n=" << n << ", c=" << c << ":\n";
      for (int64_t y = 0; y < T.H; ++y) {
        for (int64_t x = 0; x < T.W; ++x) {
          std::cout << std::setw(7) << T.at(n, c, y, x) << " ";
        }
        std::cout << "\n";
      }
    }
  }
  std::cout << "\n";
}

int main() {
  try {
    // ---------------- Demo 1: Identity sampling (linear) ----------------
    // X: (1,1,3,3)
    //  1 2 3
    //  4 5 6
    //  7 8 9
    Tensor4D X(1, 1, 3, 3);
    float v = 1.0f;
    for (int64_t y = 0; y < 3; ++y)
      for (int64_t x = 0; x < 3; ++x)
        X.at(0, 0, y, x) = v++;

    // Grid for output 3x3, normalized coordinates
    // We build a grid that maps each output pixel back to same input pixel.
    Grid4D G(1, 3, 3);
    // For align_corners=1, normalized coordinate of integer pixel x is: x_norm = 2*x/(W-1) - 1
    // Same for y.
    for (int64_t ho = 0; ho < 3; ++ho) {
      for (int64_t wo = 0; wo < 3; ++wo) {
        float x_norm = (3 <= 1) ? 0.0f : (2.0f * wo / (3.0f - 1.0f) - 1.0f);
        float y_norm = (3 <= 1) ? 0.0f : (2.0f * ho / (3.0f - 1.0f) - 1.0f);
        G.at(0, ho, wo, 0) = x_norm;
        G.at(0, ho, wo, 1) = y_norm;
      }
    }

    auto Y_id = GridSample2D(X, G, Mode::Linear, PaddingMode::Zeros, /*align_corners=*/true);

    print_nchw(X, "X");
    print_nchw(Y_id, "Y_identity (linear, zeros, align_corners=1)");

    // ---------------- Demo 2: Shift sampling (nearest) ----------------
    // Shift right by 1 pixel: output(ho,wo) samples input at (wo-1)
    // In normalized space (align_corners=1): x_norm = 2*x/(W-1)-1
    Grid4D G_shift(1, 3, 3);
    for (int64_t ho = 0; ho < 3; ++ho) {
      for (int64_t wo = 0; wo < 3; ++wo) {
        int64_t src_x = wo - 1;
        int64_t src_y = ho;
        float x_norm = (2.0f * static_cast<float>(src_x) / (3.0f - 1.0f) - 1.0f);
        float y_norm = (2.0f * static_cast<float>(src_y) / (3.0f - 1.0f) - 1.0f);
        G_shift.at(0, ho, wo, 0) = x_norm;
        G_shift.at(0, ho, wo, 1) = y_norm;
      }
    }
    auto Y_shift = GridSample2D(X, G_shift, Mode::Nearest, PaddingMode::Zeros, /*align_corners=*/true);
    print_nchw(Y_shift, "Y_shift_right_by1 (nearest, zeros, align_corners=1)");

    // ---------------- Demo 3: Out-of-bound behavior (border vs reflection) ----------------
    // Sample a grid that goes beyond [-1,1] to force padding behavior.
    Grid4D G_oob(1, 1, 5);
    // y fixed at center (0), x sweeps: -1.5, -1, 0, 1, 1.5
    float xs[5] = {-1.5f, -1.0f, 0.0f, 1.0f, 1.5f};
    for (int i = 0; i < 5; ++i) {
      G_oob.at(0, 0, i, 0) = xs[i];
      G_oob.at(0, 0, i, 1) = 0.0f;
    }
    auto Y_border = GridSample2D(X, G_oob, Mode::Linear, PaddingMode::Border, /*align_corners=*/true);
    auto Y_reflec = GridSample2D(X, G_oob, Mode::Linear, PaddingMode::Reflection, /*align_corners=*/true);

    print_nchw(Y_border, "Y_oob (linear, border, align_corners=1)");
    print_nchw(Y_reflec, "Y_oob (linear, reflection, align_corners=1)");

    // ---------------- Demo 4: Bicubic sampling ----------------
    auto Y_cubic = GridSample2D(X, G, Mode::Cubic, PaddingMode::Border, /*align_corners=*/true);
    print_nchw(Y_cubic, "Y_identity (cubic, border, align_corners=1)");

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
