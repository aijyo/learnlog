#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// ------------------------------ Tensor (row-major) ------------------------------
//
// Minimal N-D tensor container for Squeeze demo.
// Squeeze is a pure reshape/view operation: data content is unchanged.
//
template <typename T>
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<T> data;

  int64_t rank() const { return static_cast<int64_t>(shape.size()); }

  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) {
      if (d < 0) throw std::runtime_error("Negative dimension is not supported in this demo.");
      n *= d;
    }
    return n;
  }
};

static inline int64_t normalize_axis(int64_t axis, int64_t rank) {
  if (axis < -rank || axis >= rank) throw std::runtime_error("Squeeze: axis out of range.");
  if (axis < 0) axis += rank;
  return axis;
}

// Result helper to expose normalized axes (like onnx-mlir saveAxes side-effect).
struct SqueezeResult {
  std::vector<int64_t> normalized_axes; // sorted unique
};

// ------------------------------ ONNX Squeeze (runtime) ------------------------------
//
// Semantics aligned to the onnx-mlir snippet:
//
// If axes is not provided (axesFromShape == true):
//   squeeze all dimensions whose size == 1.
//   NOTE: onnx-mlir requires compile-time literal dims for this path.
//         runtime always has dims, so we can do it directly.
//
// If axes is provided:
//   normalize negative axes, validate in range,
//   then squeeze only those axes.
//   Per ONNX behavior, attempting to squeeze a dim not equal to 1 is an error.
//
template <typename T>
Tensor<T> onnx_squeeze(const Tensor<T>& data,
                       const std::vector<int64_t>* axes_opt,
                       SqueezeResult* info = nullptr) {
  const int64_t rank = data.rank();
  if (rank < 0) throw std::runtime_error("Squeeze: invalid rank.");

  std::vector<int64_t> squeezedAxes;

  if (!axes_opt) {
    // axesFromShape == true: squeeze all dims with size 1.
    for (int64_t i = 0; i < rank; ++i) {
      if (data.shape[i] == 1) squeezedAxes.push_back(i);
    }
  } else {
    // axesFromShape == false: normalize given axes.
    std::unordered_set<int64_t> seen;
    for (int64_t a : *axes_opt) {
      const int64_t ax = normalize_axis(a, rank);
      if (!seen.insert(ax).second) continue; // de-dup
      squeezedAxes.push_back(ax);
    }
    std::sort(squeezedAxes.begin(), squeezedAxes.end());

    // Validate that specified axes are squeezable (dim must be 1).
    for (int64_t ax : squeezedAxes) {
      if (data.shape[ax] != 1) {
        throw std::runtime_error("Squeeze: cannot squeeze an axis whose dimension != 1.");
      }
    }
  }

  // Build output shape by keeping dims not in squeezedAxes.
  std::vector<int64_t> outShape;
  outShape.reserve(static_cast<size_t>(rank));

  size_t p = 0;
  for (int64_t i = 0; i < rank; ++i) {
    if (p < squeezedAxes.size() && squeezedAxes[p] == i) {
      ++p; // skip this dim
    } else {
      outShape.push_back(data.shape[i]);
    }
  }

  // Corner case: if all dims are squeezed, result is a scalar tensor with empty shape.
  // That is consistent with common tensor conventions.
  Tensor<T> out;
  out.shape = std::move(outShape);
  out.data = data.data; // no data change (view/reshape semantics)

  // Sanity check: numel must match
  if (out.numel() != data.numel()) {
    throw std::runtime_error("Squeeze: internal error, numel mismatch.");
  }

  if (info) {
    info->normalized_axes = squeezedAxes;
  }
  return out;
}

// ------------------------------ Demo Helpers ------------------------------
template <typename T>
static void print_shape(const Tensor<T>& t, const std::string& name) {
  std::cout << name << " shape = [";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    std::cout << t.shape[i] << (i + 1 == t.shape.size() ? "" : ", ");
  }
  std::cout << "], numel=" << t.numel() << "\n";
}

static Tensor<int64_t> make_tensor_i64(std::vector<int64_t> shape) {
  Tensor<int64_t> t;
  t.shape = std::move(shape);
  t.data.resize(t.numel());
  std::iota(t.data.begin(), t.data.end(), 0);
  return t;
}

static void print_axes(const std::vector<int64_t>& axes, const std::string& name) {
  std::cout << name << " = [";
  for (size_t i = 0; i < axes.size(); ++i) {
    std::cout << axes[i] << (i + 1 == axes.size() ? "" : ", ");
  }
  std::cout << "]\n";
}

int main() {
  // Input: shape [2, 1, 3, 1]
  auto x = make_tensor_i64({2, 1, 3, 1});
  print_shape(x, "x");

  // Case A: axes = None -> squeeze all dims with size 1 (axesFromShape=true)
  {
    SqueezeResult info;
    auto y = onnx_squeeze<int64_t>(x, /*axes_opt=*/nullptr, &info);
    print_shape(y, "y (axes=None)");
    print_axes(info.normalized_axes, "  squeezed_axes");
    // expected: squeezed_axes [1,3], y shape [2,3]
  }

  // Case B: axes = [1] -> only squeeze axis 1
  {
    std::vector<int64_t> axes = {1};
    SqueezeResult info;
    auto y = onnx_squeeze<int64_t>(x, &axes, &info);
    print_shape(y, "y (axes=[1])");
    print_axes(info.normalized_axes, "  normalized_axes");
    // expected: y shape [2,3,1]
  }

  // Case C: axes = [-1] -> normalize to axis 3, squeeze last dim (which is 1)
  {
    std::vector<int64_t> axes = {-1};
    SqueezeResult info;
    auto y = onnx_squeeze<int64_t>(x, &axes, &info);
    print_shape(y, "y (axes=[-1])");
    print_axes(info.normalized_axes, "  normalized_axes");
    // expected: normalized_axes [3], y shape [2,1,3]
  }

  // Case D: error example: try to squeeze a dim not equal to 1
  {
    try {
      auto x2 = make_tensor_i64({2, 2, 3});
      std::vector<int64_t> axes = {1}; // dim is 2, not squeezable
      auto y2 = onnx_squeeze<int64_t>(x2, &axes, nullptr);
      (void)y2;
    } catch (const std::exception& e) {
      std::cout << "Expected error: " << e.what() << "\n";
    }
  }

  return 0;
}
