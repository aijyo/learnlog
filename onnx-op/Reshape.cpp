#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// --------------------------- ONNX Reshape (Standalone) ---------------------------
//
// ONNX Reshape takes:
//   - data: input tensor
//   - shape: int64 array (1-D) describing the output shape
//   - allowzero (attribute, default 0):
//       * allowzero == 0: shape[i] == 0 means "copy input dimension at i"
//       * allowzero == 1: shape[i] == 0 means literal 0
//
// shape[i] semantics:
//   - > 0 : fixed output dimension
//   - == 0: see allowzero rules
//   - == -1: inferred dimension (at most one -1 allowed)
//
// Inference rule for -1:
//   inferred = num_input_elements / product(other_output_dims)
//   must be divisible with no remainder.
//
// This standalone implementation assumes input tensor has known static shape.
// It returns a new Tensor view that shares data (copied here for simplicity).
// You can optimize to share storage using shared_ptr if needed.
//
// -------------------------------------------------------------------------------

template <typename T>
struct Tensor {
    std::vector<int64_t> shape; // e.g., [N, C, H, W]
    std::vector<T> data;        // flat buffer in row-major

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t numel() const {
        // Note: if any dim is 0, numel is 0
        if (shape.empty()) return 1; // scalar-like
        int64_t n = 1;
        for (int64_t d : shape) {
            if (d < 0) throw std::runtime_error("Tensor has negative dimension.");
            if (d == 0) return 0;
            // Basic overflow guard (optional)
            if (n > INT64_MAX / d) throw std::runtime_error("numel overflow.");
            n *= d;
        }
        return n;
    }
};

static std::string ShapeToString(const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        out += std::to_string(s[i]);
        if (i + 1 < s.size()) out += ", ";
    }
    out += "]";
    return out;
}

static std::vector<int64_t> ResolveOnnxReshapeOutputShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& target_shape,
    int64_t allowzero /*0 or 1*/) {
    if (!(allowzero == 0 || allowzero == 1))
        throw std::runtime_error("allowzero must be 0 or 1.");

    const int64_t in_rank = static_cast<int64_t>(input_shape.size());
    const int64_t out_rank = static_cast<int64_t>(target_shape.size());

    // Compute input numel
    auto compute_numel = [](const std::vector<int64_t>& s) -> int64_t {
        if (s.empty()) return 1;
        int64_t n = 1;
        for (int64_t d : s) {
            if (d < 0) throw std::runtime_error("Negative dimension in shape.");
            if (d == 0) return 0;
            if (n > INT64_MAX / d) throw std::runtime_error("numel overflow.");
            n *= d;
        }
        return n;
        };

    const int64_t input_numel = compute_numel(input_shape);

    std::vector<int64_t> out_dims(out_rank, 0);

    int64_t minus_one_index = -1;
    int64_t known_product = 1;
    bool known_product_is_zero = false;

    for (int64_t i = 0; i < out_rank; ++i) {
        int64_t v = target_shape[i];

        if (v == -1) {
            if (minus_one_index != -1)
                throw std::runtime_error("Reshape: multiple -1 in target shape.");
            minus_one_index = i;
            out_dims[i] = -1; // placeholder
            // For product, treat -1 as 1
            continue;
        }

        if (v == 0) {
            if (allowzero == 0) {
                if (i >= in_rank)
                    throw std::runtime_error("Reshape: 0 copies input dim, but i >= input rank.");
                out_dims[i] = input_shape[i];
            }
            else {
                out_dims[i] = 0; // literal zero dimension
            }
        }
        else if (v > 0) {
            out_dims[i] = v;
        }
        else {
            throw std::runtime_error("Reshape: target shape values must be >0, 0, or -1.");
        }

        // Update known_product (excluding -1)
        if (out_dims[i] == 0) {
            known_product_is_zero = true;
        }
        else {
            if (!known_product_is_zero) {
                if (known_product > INT64_MAX / out_dims[i])
                    throw std::runtime_error("Reshape: product overflow.");
                known_product *= out_dims[i];
            }
        }
    }

    // Infer -1 if present
    if (minus_one_index != -1) {
        if (known_product_is_zero) {
            // If other dims product is 0, inference is ambiguous unless input_numel is 0.
            if (input_numel != 0) {
                throw std::runtime_error(
                    "Reshape: cannot infer -1 when other dims contain 0 but input_numel != 0.");
            }
            // If input_numel == 0, set inferred dim to 0 (common practical choice).
            out_dims[minus_one_index] = 0;
        }
        else {
            if (known_product == 0) {
                // Should be covered by known_product_is_zero, but keep safe.
                if (input_numel != 0) throw std::runtime_error("Reshape: invalid 0 product.");
                out_dims[minus_one_index] = 0;
            }
            else {
                if (input_numel % known_product != 0) {
                    throw std::runtime_error(
                        "Reshape: input_numel not divisible by product of known output dims.");
                }
                out_dims[minus_one_index] = input_numel / known_product;
            }
        }
    }

    // Final consistency check: output numel must match input numel
    const int64_t output_numel = compute_numel(out_dims);
    if (output_numel != input_numel) {
        throw std::runtime_error(
            "Reshape: output numel != input numel. input=" + std::to_string(input_numel) +
            ", output=" + std::to_string(output_numel) +
            ", out_shape=" + ShapeToString(out_dims));
    }

    return out_dims;
}

template <typename T>
static Tensor<T> OnnxReshape(const Tensor<T>& input,
    const std::vector<int64_t>& target_shape,
    int64_t allowzero = 0) {
    // Resolve output shape under ONNX rules
    std::vector<int64_t> out_shape =
        ResolveOnnxReshapeOutputShape(input.shape, target_shape, allowzero);

    Tensor<T> out;
    out.shape = std::move(out_shape);
    out.data = input.data; // copy for simplicity; you can share storage if desired

    // Validate buffer size matches
    if (out.numel() != static_cast<int64_t>(out.data.size())) {
        throw std::runtime_error(
            "Reshape: data buffer size does not match numel. buffer=" +
            std::to_string(out.data.size()) + ", numel=" + std::to_string(out.numel()));
    }

    return out;
}

// --------------------------- Demo ---------------------------
static void PrintTensorInfo(const Tensor<float>& t, const std::string& name) {
    std::cout << name << " shape=" << ShapeToString(t.shape)
        << " numel=" << t.numel()
        << " data_size=" << t.data.size() << "\n";
}

int main() {
    try {
        // Example input: shape [2, 3, 4] => numel=24
        Tensor<float> x;
        x.shape = { 2, 3, 4 };
        x.data.resize(24);
        for (int i = 0; i < 24; ++i) x.data[i] = static_cast<float>(i);

        PrintTensorInfo(x, "x");

        // Case A: target [6, 4] (no -1, no 0)
        auto yA = OnnxReshape(x, { 6, 4 }, /*allowzero=*/0);
        PrintTensorInfo(yA, "yA");

        // Case B: target [0, -1] with allowzero=0 => [copy input dim0, infer dim1] => [2, 12]
        auto yB = OnnxReshape(x, { 0, -1 }, /*allowzero=*/0);
        PrintTensorInfo(yB, "yB");

        // Case C: target [1, 0, -1] with allowzero=0 => [1, copy dim1=3, infer] => [1,3,8]
        auto yC = OnnxReshape(x, { 1, 0, -1 }, /*allowzero=*/0);
        PrintTensorInfo(yC, "yC");

        // Case D: allowzero=1, target [0, 24] => literal 0 dimension => output numel=0 (will fail because input_numel=24)
        // Uncomment to see expected failure:
        // auto yD = OnnxReshape(x, {0, 24}, /*allowzero=*/1);

        // Case E: invalid: multiple -1
        // Uncomment to see expected failure:
        // auto yE = OnnxReshape(x, {-1, -1}, 0);

        std::cout << "All done.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
