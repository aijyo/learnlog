#include <cstdint>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include <iostream>

template <typename T>
struct TensorND {
    std::vector<T> data;
    std::vector<int64_t> shape; // Row-major contiguous
};

static inline int64_t NumElements(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1; // scalar
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dim not supported");
        n *= d;
    }
    return n;
}

static inline int64_t ProductRange(const std::vector<int64_t>& shape, int64_t begin, int64_t end) {
    // Product of shape[begin..end-1]
    int64_t p = 1;
    for (int64_t i = begin; i < end; ++i) p *= shape[(size_t)i];
    return p;
}

static inline int64_t NormalizeAxisForRank(int64_t axis, int64_t rank) {
    // Normalize axis into [0, rank-1]
    if (rank <= 0) throw std::invalid_argument("rank must be > 0");
    int64_t a = axis;
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) throw std::out_of_range("axis out of range");
    return a;
}

static inline int64_t NormalizeAxisForInsert(int64_t axis, int64_t out_rank) {
    // Normalize insert position into [0, out_rank-1]
    // For insertion, valid positions are [0, out_rank-1], but we allow axis==out_rank-1 as "insert at last dim".
    // Here out_rank = R+1, valid positions [0, R].
    if (out_rank <= 0) throw std::invalid_argument("out_rank must be > 0");
    int64_t a = axis;
    if (a < 0) a += out_rank;
    if (a < 0 || a >= out_rank) throw std::out_of_range("axis out of range for insertion");
    return a;
}


template <typename T>
TensorND<T> ConcatFromSequence_NewAxis0(const std::vector<TensorND<T>>& seq, int64_t axis_raw) {
    static_assert(std::is_trivially_copyable<T>::value,
        "This simple impl assumes trivially copyable element type");

    if (seq.empty()) throw std::invalid_argument("sequence must not be empty");

    const int64_t rank = (int64_t)seq[0].shape.size();
    if (rank <= 0) throw std::invalid_argument("rank must be >= 1 in this impl");

    const int64_t axis = NormalizeAxisForRank(axis_raw, rank);

    // Validate shapes and compute output shape
    std::vector<int64_t> out_shape = seq[0].shape;
    int64_t out_axis_dim = 0;

    for (size_t k = 0; k < seq.size(); ++k) {
        const auto& s = seq[k].shape;
        if ((int64_t)s.size() != rank) throw std::invalid_argument("all sequence tensors must have same rank");

        for (int64_t i = 0; i < rank; ++i) {
            if (i == axis) continue;
            if (s[(size_t)i] != out_shape[(size_t)i]) {
                throw std::invalid_argument("non-axis dimensions must match for new_axis=0");
            }
        }

        const int64_t n = NumElements(s);
        if ((int64_t)seq[k].data.size() != n) {
            throw std::invalid_argument("input data size does not match shape");
        }

        out_axis_dim += s[(size_t)axis];
    }

    out_shape[(size_t)axis] = out_axis_dim;

    const int64_t outer = ProductRange(out_shape, 0, axis);
    const int64_t inner = ProductRange(out_shape, axis + 1, rank);
    const int64_t out_elems = NumElements(out_shape);

    TensorND<T> out;
    out.shape = std::move(out_shape);
    out.data.resize((size_t)out_elems);

    // Copy blocks: same as Concat
    for (int64_t o = 0; o < outer; ++o) {
        int64_t out_offset = o * (out_axis_dim * inner);
        for (size_t k = 0; k < seq.size(); ++k) {
            const auto& in = seq[k];
            const int64_t in_axis_dim = in.shape[(size_t)axis];
            const int64_t in_offset = o * (in_axis_dim * inner);
            const int64_t copy_count = in_axis_dim * inner;

            for (int64_t t = 0; t < copy_count; ++t) {
                out.data[(size_t)(out_offset + t)] = in.data[(size_t)(in_offset + t)];
            }
            out_offset += copy_count;
        }
    }

    return out;
}


template <typename T>
TensorND<T> ConcatFromSequence_NewAxis1(const std::vector<TensorND<T>>& seq, int64_t axis_raw) {
    static_assert(std::is_trivially_copyable<T>::value,
        "This simple impl assumes trivially copyable element type");

    if (seq.empty()) throw std::invalid_argument("sequence must not be empty");

    const std::vector<int64_t>& in_shape0 = seq[0].shape;
    const int64_t in_rank = (int64_t)in_shape0.size();
    const int64_t out_rank = in_rank + 1;

    // axis is insertion position in output rank
    const int64_t axis = NormalizeAxisForInsert(axis_raw, out_rank);

    // Validate all shapes are identical
    const int64_t in_elems = NumElements(in_shape0);
    for (size_t k = 0; k < seq.size(); ++k) {
        if (seq[k].shape != in_shape0) {
            throw std::invalid_argument("all sequence tensors must have identical shape for new_axis=1");
        }
        if ((int64_t)seq[k].data.size() != in_elems) {
            throw std::invalid_argument("input data size does not match shape");
        }
    }

    // Build output shape: insert N at axis
    std::vector<int64_t> out_shape;
    out_shape.reserve((size_t)out_rank);
    for (int64_t i = 0; i < axis; ++i) out_shape.push_back(in_shape0[(size_t)i]);
    out_shape.push_back((int64_t)seq.size());
    for (int64_t i = axis; i < in_rank; ++i) out_shape.push_back(in_shape0[(size_t)i]);

    TensorND<T> out;
    out.shape = std::move(out_shape);
    out.data.resize((size_t)NumElements(out.shape));

    // Compute outer/inner w.r.t insertion axis:
    // outer = product of dims before insertion axis in input (same indices)
    // inner = product of dims from insertion position to end in input
    const int64_t outer = ProductRange(in_shape0, 0, axis);
    const int64_t inner = ProductRange(in_shape0, axis, in_rank);
    const int64_t N = (int64_t)seq.size();

    // Write layout:
    // output viewed as [outer, N, inner] contiguous in row-major
    // For each outer o, for each k in [0..N-1], copy inner elements from seq[k] at offset o*inner.
    for (int64_t o = 0; o < outer; ++o) {
        const int64_t in_offset = o * inner;
        int64_t out_base = o * (N * inner);
        for (int64_t k = 0; k < N; ++k) {
            const auto& in = seq[(size_t)k];
            const int64_t out_offset = out_base + k * inner;
            for (int64_t t = 0; t < inner; ++t) {
                out.data[(size_t)(out_offset + t)] = in.data[(size_t)(in_offset + t)];
            }
        }
    }

    return out;
}

template <typename T>
TensorND<T> ConcatFromSequence(const std::vector<TensorND<T>>& seq,
    int64_t axis,
    bool new_axis) {
    if (!new_axis) {
        return ConcatFromSequence_NewAxis0<T>(seq, axis);
    }
    else {
        return ConcatFromSequence_NewAxis1<T>(seq, axis);
    }
}

static void PrintShape(const std::vector<int64_t>& s) {
    std::cout << "[";
    for (size_t i = 0; i < s.size(); ++i) {
        std::cout << s[i] << (i + 1 == s.size() ? "" : ",");
    }
    std::cout << "]";
}

static void PrintFlat(const TensorND<int>& t, const char* name) {
    std::cout << name << " shape=";
    PrintShape(t.shape);
    std::cout << " data={";
    for (size_t i = 0; i < t.data.size(); ++i) {
        std::cout << t.data[i] << (i + 1 == t.data.size() ? "" : ", ");
    }
    std::cout << "}\n";
}

static void DemoConcatFromSequence() {
    // Build a sequence of 2 tensors, each shape [2, 2]
    TensorND<int> A;
    A.shape = { 2, 2 };
    A.data = { 1, 2,
               3, 4 };

    TensorND<int> B;
    B.shape = { 2, 2 };
    B.data = { 10, 20,
               30, 40 };

    std::vector<TensorND<int>> seq = { A, B };

    // Case 1: new_axis = 0, axis = 0 -> Concat along rows: [2,2] + [2,2] => [4,2]
    auto Y0 = ConcatFromSequence<int>(seq, /*axis=*/0, /*new_axis=*/false);
    PrintFlat(A, "A");
    PrintFlat(B, "B");
    PrintFlat(Y0, "ConcatFromSequence new_axis=0 axis=0");

    // Case 2: new_axis = 0, axis = 1 -> Concat along cols: [2,2] + [2,2] => [2,4]
    auto Y1 = ConcatFromSequence<int>(seq, /*axis=*/1, /*new_axis=*/false);
    PrintFlat(Y1, "ConcatFromSequence new_axis=0 axis=1");

    // Case 3: new_axis = 1, axis = 0 -> Stack at front: => [N,2,2] where N=2 => [2,2,2]
    auto S0 = ConcatFromSequence<int>(seq, /*axis=*/0, /*new_axis=*/true);
    PrintFlat(S0, "ConcatFromSequence new_axis=1 axis=0 (stack)");

    // Case 4: new_axis = 1, axis = 2 -> Stack at last: => [2,2,N] => [2,2,2]
    auto S1 = ConcatFromSequence<int>(seq, /*axis=*/2, /*new_axis=*/true);
    PrintFlat(S1, "ConcatFromSequence new_axis=1 axis=2 (stack at last)");
}

int main() {
    DemoConcatFromSequence();
    return 0;
}
