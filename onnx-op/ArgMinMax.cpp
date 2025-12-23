#include <cstdint>
#include <vector>
#include <stdexcept>
#include <limits>
#include <type_traits>

static inline int64_t NormalizeAxis(int64_t axis, int64_t rank) {
    if (rank <= 0) throw std::invalid_argument("rank must be > 0");
    int64_t a = axis;
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) throw std::out_of_range("axis out of range");
    return a;
}

static inline int64_t NumElements(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1; // scalar
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dim not supported in this runtime impl");
        n *= d;
    }
    return n;
}

template <typename T>
struct TensorND {
    std::vector<T> data;
    std::vector<int64_t> shape;
};

// Build output shape according to keepdims
static inline std::vector<int64_t> MakeReducedShape(const std::vector<int64_t>& in_shape,
                                                    int64_t axis,
                                                    bool keepdims) {
    std::vector<int64_t> out_shape;
    out_shape.reserve(in_shape.size());
    if (keepdims) {
        out_shape = in_shape;
        out_shape[(size_t)axis] = 1;
    } else {
        for (int64_t i = 0; i < (int64_t)in_shape.size(); ++i) {
            if (i == axis) continue;
            out_shape.push_back(in_shape[(size_t)i]);
        }
        if (out_shape.empty()) {
            // When reducing a 1D tensor with keepdims=0, output can be scalar
            // We'll represent scalar as shape = {}
        }
    }
    return out_shape;
}

// Core arg-reduce implementation
// is_max = true  -> ArgMax
// is_max = false -> ArgMin
template <typename T>
TensorND<int64_t> ArgMinMax(const T* input,
                            const std::vector<int64_t>& in_shape,
                            int64_t axis_raw,
                            bool keepdims,
                            bool select_last_index,
                            bool is_max) {
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");

    const int64_t rank = (int64_t)in_shape.size();
    if (rank <= 0) {
        throw std::invalid_argument("ArgMin/ArgMax requires rank >= 1 in this impl");
    }

    const int64_t axis = NormalizeAxis(axis_raw, rank);
    const int64_t axis_dim = in_shape[(size_t)axis];
    if (axis_dim <= 0) {
        throw std::invalid_argument("axis dimension must be > 0");
    }

    // Compute outer and inner sizes:
    // outer = product of dims before axis
    // inner = product of dims after axis
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= in_shape[(size_t)i];

    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) inner *= in_shape[(size_t)i];

    // Output shape and size
    TensorND<int64_t> out;
    out.shape = MakeReducedShape(in_shape, axis, keepdims);
    const int64_t out_elems = outer * inner;
    out.data.resize((size_t)out_elems);

    // For each (outer, inner) position, scan over axis_dim
    // Index mapping for row-major contiguous:
    // base = o * axis_dim * inner + i
    // element at axis index a: input[base + a*inner]
    int64_t out_idx = 0;
    for (int64_t o = 0; o < outer; ++o) {
        const int64_t base0 = o * axis_dim * inner;
        for (int64_t i = 0; i < inner; ++i) {
            int64_t best_index = 0;
            T best_value = input[(size_t)(base0 + 0 * inner + i)];

            for (int64_t a = 1; a < axis_dim; ++a) {
                const T v = input[(size_t)(base0 + a * inner + i)];

                bool better = false;
                if (is_max) {
                    if (v > best_value) better = true;
                    else if (v == best_value && select_last_index) better = true;
                } else {
                    if (v < best_value) better = true;
                    else if (v == best_value && select_last_index) better = true;
                }

                if (better) {
                    best_value = v;
                    best_index = a;
                }
            }

            out.data[(size_t)out_idx++] = best_index;
        }
    }

    return out;
}

// Convenience wrappers
template <typename T>
TensorND<int64_t> ArgMax(const T* input,
                         const std::vector<int64_t>& in_shape,
                         int64_t axis = 0,
                         bool keepdims = true,
                         bool select_last_index = false) {
    return ArgMinMax(input, in_shape, axis, keepdims, select_last_index, /*is_max=*/true);
}

template <typename T>
TensorND<int64_t> ArgMin(const T* input,
                         const std::vector<int64_t>& in_shape,
                         int64_t axis = 0,
                         bool keepdims = true,
                         bool select_last_index = false) {
    return ArgMinMax(input, in_shape, axis, keepdims, select_last_index, /*is_max=*/false);
}
