template <typename T>
struct TensorND {
    std::vector<T> data;
    std::vector<int64_t> shape;
};

template <typename T>
TensorND<T> CompressAxis(const T* input,
                         const std::vector<int64_t>& in_shape,
                         const bool* condition,
                         int64_t cond_len,
                         int64_t axis_raw) {
    const int64_t rank = (int64_t)in_shape.size();
    if (rank == 0) {
        throw std::invalid_argument("axis mode not supported for scalar in this simple impl");
    }

    const int64_t axis = NormalizeAxis(axis_raw, rank);
    const int64_t axis_dim = in_shape[(size_t)axis];

    if (cond_len != axis_dim) {
        throw std::invalid_argument("condition length must equal data.shape[axis]");
    }

    // Compute outer/inner sizes
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= in_shape[(size_t)i];

    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) inner *= in_shape[(size_t)i];

    // Count trues
    int64_t kept = 0;
    for (int64_t i = 0; i < axis_dim; ++i) if (condition[i]) ++kept;

    // Output shape: same rank, axis dim replaced by kept
    TensorND<T> out;
    out.shape = in_shape;
    out.shape[(size_t)axis] = kept;
    out.data.resize((size_t)(outer * kept * inner));

    // Copy blocks
    // Layout concept:
    // input is grouped as: outer blocks, each has axis_dim blocks, each block size = inner
    // output is grouped as: outer blocks, each has kept blocks, each block size = inner
    int64_t out_index = 0;
    for (int64_t o = 0; o < outer; ++o) {
        const int64_t base_in = o * axis_dim * inner;
        for (int64_t a = 0; a < axis_dim; ++a) {
            if (!condition[a]) continue;
            const int64_t in_index = base_in + a * inner;

            // Copy a contiguous block of 'inner' elements
            for (int64_t k = 0; k < inner; ++k) {
                out.data[(size_t)out_index++] = input[(size_t)(in_index + k)];
            }
        }
    }

    return out;
}
