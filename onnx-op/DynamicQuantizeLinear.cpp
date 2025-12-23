#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

// --------------------------- Utilities ---------------------------

// English comments per your preference.

static int64_t NumElements(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d < 0) throw std::runtime_error("Negative dim in shape.");
    n *= d;
  }
  return n;
}

static inline uint8_t ClampU8(int64_t v) {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return static_cast<uint8_t>(v);
}

static inline int64_t RoundToInt64(double x) {
  // std::nearbyint rounds to nearest integer according to current rounding mode;
  // ONNX typically uses round-to-nearest-even or nearest; in practice this is OK for demo.
  return static_cast<int64_t>(std::llround(x));
}

// --------------------------- DynamicQuantizeLinear ---------------------------
//
// Returns (y, y_scale, y_zero_point).
// - y: uint8 tensor with same shape as x
// - y_scale: scalar float
// - y_zero_point: scalar uint8
//
// Common dynamic quantization recipe:
//   rmin = min(xmin, 0)
//   rmax = max(xmax, 0)
//   scale = (rmax - rmin) / 255
//   zero_point = round(0 - rmin/scale), clamp to [0,255]
//   y = clamp(round(x/scale) + zero_point)
//
// If rmax == rmin, use scale=1 and zp=0 to avoid division by zero.

struct DynamicQuantizeResult {
  std::vector<uint8_t> y;
  float y_scale;
  uint8_t y_zero_point;
  float x_min;
  float x_max;
  float r_min;
  float r_max;
};

static DynamicQuantizeResult DynamicQuantizeLinear(
    const std::vector<float>& x,
    const std::vector<int64_t>& xShape) {

  const int64_t N = NumElements(xShape);
  if ((int64_t)x.size() != N)
    throw std::runtime_error("x size does not match shape product.");

  if (N == 0) {
    return {{}, 1.0f, 0, 0.0f, 0.0f, 0.0f, 0.0f};
  }

  // Compute min/max.
  float xmin = std::numeric_limits<float>::infinity();
  float xmax = -std::numeric_limits<float>::infinity();
  for (float v : x) {
    xmin = std::min(xmin, v);
    xmax = std::max(xmax, v);
  }

  // Include 0 in range.
  float rmin = std::min(xmin, 0.0f);
  float rmax = std::max(xmax, 0.0f);

  float scale = 1.0f;
  uint8_t zp = 0;

  if (rmax > rmin) {
    scale = (rmax - rmin) / 255.0f;

    // Compute zero point so that real 0 maps close to an integer.
    // zp = round(-rmin / scale)
    double zp_fp = -static_cast<double>(rmin) / static_cast<double>(scale);
    int64_t zp_i64 = RoundToInt64(zp_fp);
    zp = ClampU8(zp_i64);

    // Optional: adjust rmin/rmax mapping boundaries (not necessary for this demo).
  } else {
    // Degenerate case: all values equal (or range zero).
    scale = 1.0f;
    zp = 0;
  }

  std::vector<uint8_t> y((size_t)N);

  // Quantize.
  for (int64_t i = 0; i < N; ++i) {
    double q = static_cast<double>(x[(size_t)i]) / static_cast<double>(scale);
    int64_t qi = RoundToInt64(q) + (int64_t)zp;
    y[(size_t)i] = ClampU8(qi);
  }

  return {std::move(y), scale, zp, xmin, xmax, rmin, rmax};
}

// Simple dequantize for demo verification.
static std::vector<float> DequantizeLinearU8(
    const std::vector<uint8_t>& y, float scale, uint8_t zp) {
  std::vector<float> xrec(y.size());
  for (size_t i = 0; i < y.size(); ++i) {
    int64_t yi = (int64_t)y[i];
    int64_t zpi = (int64_t)zp;
    xrec[i] = static_cast<float>((yi - zpi) * (double)scale);
  }
  return xrec;
}

static void PrintVec(const std::vector<float>& v, const std::string& name, int maxN = 16) {
  std::cout << name << " = [";
  int n = (int)v.size();
  int k = std::min(n, maxN);
  for (int i = 0; i < k; ++i) {
    std::cout << v[i];
    if (i + 1 < k) std::cout << ", ";
  }
  if (n > k) std::cout << ", ...";
  std::cout << "]\n";
}

static void PrintVecU8(const std::vector<uint8_t>& v, const std::string& name, int maxN = 32) {
  std::cout << name << " = [";
  int n = (int)v.size();
  int k = std::min(n, maxN);
  for (int i = 0; i < k; ++i) {
    std::cout << (int)v[i];
    if (i + 1 < k) std::cout << ", ";
  }
  if (n > k) std::cout << ", ...";
  std::cout << "]\n";
}

int main() {
  try {
    // Demo input: shape [2, 6]
    std::vector<int64_t> shape = {2, 6};
    std::vector<float> x = {
      -3.2f, -1.0f, -0.1f, 0.0f, 0.2f, 1.7f,
       2.5f,  3.9f,  7.0f, -8.0f, 0.01f, 0.5f
    };

    PrintVec(x, "x");

    auto res = DynamicQuantizeLinear(x, shape);

    std::cout << "x_min=" << res.x_min << " x_max=" << res.x_max << "\n";
    std::cout << "r_min(min(x,0))=" << res.r_min << " r_max(max(x,0))=" << res.r_max << "\n";
    std::cout << "y_scale=" << res.y_scale << " y_zero_point=" << (int)res.y_zero_point << "\n";

    PrintVecU8(res.y, "y(uint8)");

    auto xrec = DequantizeLinearU8(res.y, res.y_scale, res.y_zero_point);
    PrintVec(xrec, "dequant(y)");

    // Report max abs error.
    double maxAbsErr = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
      maxAbsErr = std::max(maxAbsErr, std::abs((double)x[i] - (double)xrec[i]));
    }
    std::cout << "max_abs_error=" << maxAbsErr << "\n";

    std::cout << "Done.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
