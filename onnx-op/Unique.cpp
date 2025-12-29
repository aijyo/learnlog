#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cstring>

// --------------------------- Tensor (Minimal) ---------------------------
//
// A minimal dense tensor in row-major order.
// - shape: [d0, d1, ...]
// - strides: row-major strides
// - data: contiguous buffer
//
template <typename T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<T> data;

    static int64_t numel_from_shape(const std::vector<int64_t>& s) {
        if (s.empty()) return 1;
        int64_t n = 1;
        for (int64_t v : s) {
            if (v < 0) throw std::invalid_argument("Negative dimension is invalid.");
            n *= v;
        }
        return n;
    }

    static std::vector<int64_t> make_row_major_strides(const std::vector<int64_t>& s) {
        std::vector<int64_t> st(s.size(), 1);
        for (int i = (int)s.size() - 2; i >= 0; --i) {
            st[i] = st[i + 1] * s[i + 1];
        }
        return st;
    }

    Tensor() = default;

    explicit Tensor(std::vector<int64_t> shp)
        : shape(std::move(shp)) {
        strides = make_row_major_strides(shape);
        data.resize((size_t)numel_from_shape(shape));
    }

    int64_t rank() const { return (int64_t)shape.size(); }
    int64_t numel() const { return numel_from_shape(shape); }
};

// --------------------------- Unique (ONNX-like) ---------------------------
//
// Returns (Y, indices, inverse_indices, counts).
//
// Behavior:
// - If axis is NOT provided:
//   * Flatten X into 1D, compute unique over elements.
//   * Y is 1D [num_unique].
//   * indices is 1D [num_unique], first occurrence position in flattened X.
//   * inverse_indices is 1D [X.numel()], mapping each element to Y index.
//   * counts is 1D [num_unique].
//
// - If axis IS provided:
//   * Treat each slice along 'axis' as a unit (a sub-tensor).
//   * Unique is computed over these slices.
//   * Y has same rank as X, but axis dimension becomes num_unique.
//   * indices is 1D [num_unique], first occurrence axis index.
//   * inverse_indices is 1D [X.shape[axis]], mapping each slice to Y index.
//   * counts is 1D [num_unique].
//
// sorted:
// - If sorted == 1, sort Y (and reorder indices/counts accordingly),
//   then remap inverse_indices to the new order.
// - If sorted is not provided, treat as 0 (keep first-seen order).
//
// Notes:
// - This is a reference implementation focusing on correctness and clarity.
// - For floating point, hashing is based on raw bits (NaNs with different payloads may differ).
//

// Hash helper for scalar T by raw bytes (works for ints/floats).
template <typename T>
struct ScalarBitHash {
    size_t operator()(const T& v) const noexcept {
        // FNV-1a on bytes
        const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
        size_t h = 1469598103934665603ull;
        for (size_t i = 0; i < sizeof(T); ++i) {
            h ^= (size_t)p[i];
            h *= 1099511628211ull;
        }
        return h;
    }
};

template <typename T>
struct ScalarBitEq {
    bool operator()(const T& a, const T& b) const noexcept {
        return std::memcmp(&a, &b, sizeof(T)) == 0;
    }
};

// A slice key: stores the slice data to allow hashing + comparison.
template <typename T>
struct SliceKey {
    std::vector<T> v;
};

template <typename T>
struct SliceHash {
    size_t operator()(const SliceKey<T>& k) const noexcept {
        // FNV-1a over concatenated element bytes
        size_t h = 1469598103934665603ull;
        for (const T& x : k.v) {
            const unsigned char* p = reinterpret_cast<const unsigned char*>(&x);
            for (size_t i = 0; i < sizeof(T); ++i) {
                h ^= (size_t)p[i];
                h *= 1099511628211ull;
            }
        }
        return h;
    }
};

template <typename T>
struct SliceEq {
    bool operator()(const SliceKey<T>& a, const SliceKey<T>& b) const noexcept {
        if (a.v.size() != b.v.size()) return false;
        return std::memcmp(a.v.data(), b.v.data(), a.v.size() * sizeof(T)) == 0;
    }
};

static int64_t NormalizeAxis(int64_t axis_raw, int64_t rank) {
    int64_t axis = axis_raw;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) throw std::invalid_argument("Unique: axis out of range.");
    return axis;
}

template <typename T>
struct UniqueOutputs {
    Tensor<T> Y;
    Tensor<int64_t> indices;
    Tensor<int64_t> inverse_indices;
    Tensor<int64_t> counts;
};

template <typename T>
static UniqueOutputs<T> Unique(
    const Tensor<T>& X,
    std::optional<int64_t> axisOpt,
    std::optional<int64_t> sortedOpt) {

    // Verify sorted attribute (0 or 1) if provided.
    if (sortedOpt.has_value()) {
        int64_t s = sortedOpt.value();
        if (s < 0 || s > 1) throw std::invalid_argument("Unique: sorted must be 0 or 1.");
    }
    const bool sorted = sortedOpt.has_value() ? (sortedOpt.value() != 0) : false;

    const int64_t rank = X.rank();
    if (rank < 1) throw std::invalid_argument("Unique: rank must be >= 1 for this demo.");
    for (int64_t d : X.shape) {
        if (d < 0) throw std::invalid_argument("Unique: negative dimension is invalid.");
    }

    UniqueOutputs<T> out;

    if (!axisOpt.has_value()) {
        // ---------------- Case A: no axis (flatten) ----------------
        const int64_t N = X.numel();

        // Map value -> unique_id (in first-seen order)
        std::unordered_map<T, int64_t, ScalarBitHash<T>, ScalarBitEq<T>> map;
        map.reserve((size_t)std::min<int64_t>(N, 1024));

        std::vector<T> uniq_vals;
        std::vector<int64_t> uniq_first_pos;
        std::vector<int64_t> uniq_counts;

        out.inverse_indices = Tensor<int64_t>({ N });

        for (int64_t i = 0; i < N; ++i) {
            const T& v = X.data[(size_t)i];
            auto it = map.find(v);
            if (it == map.end()) {
                int64_t id = (int64_t)uniq_vals.size();
                map.emplace(v, id);
                uniq_vals.push_back(v);
                uniq_first_pos.push_back(i);
                uniq_counts.push_back(1);
                out.inverse_indices.data[(size_t)i] = id;
            }
            else {
                int64_t id = it->second;
                uniq_counts[(size_t)id] += 1;
                out.inverse_indices.data[(size_t)i] = id;
            }
        }

        // If sorted, reorder uniques and remap inverse_indices.
        std::vector<int64_t> old_to_new;
        if (sorted) {
            const int64_t U = (int64_t)uniq_vals.size();
            std::vector<int64_t> order(U);
            std::iota(order.begin(), order.end(), 0);

            std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                // Sort by value ascending; tie-break by first position.
                // (ONNX doesn't strictly define tie-break; this is deterministic.)
                if (!ScalarBitEq<T>()(uniq_vals[(size_t)a], uniq_vals[(size_t)b])) {
                    return uniq_vals[(size_t)a] < uniq_vals[(size_t)b];
                }
                return uniq_first_pos[(size_t)a] < uniq_first_pos[(size_t)b];
                });

            old_to_new.assign((size_t)U, 0);
            for (int64_t newId = 0; newId < U; ++newId) {
                old_to_new[(size_t)order[(size_t)newId]] = newId;
            }

            std::vector<T> new_vals(U);
            std::vector<int64_t> new_first(U), new_counts(U);
            for (int64_t newId = 0; newId < U; ++newId) {
                int64_t oldId = order[(size_t)newId];
                new_vals[(size_t)newId] = uniq_vals[(size_t)oldId];
                new_first[(size_t)newId] = uniq_first_pos[(size_t)oldId];
                new_counts[(size_t)newId] = uniq_counts[(size_t)oldId];
            }
            uniq_vals.swap(new_vals);
            uniq_first_pos.swap(new_first);
            uniq_counts.swap(new_counts);

            // Remap inverse_indices
            for (int64_t i = 0; i < N; ++i) {
                int64_t oldId = out.inverse_indices.data[(size_t)i];
                out.inverse_indices.data[(size_t)i] = old_to_new[(size_t)oldId];
            }
        }

        const int64_t U = (int64_t)uniq_vals.size();
        out.Y = Tensor<T>({ U });
        out.indices = Tensor<int64_t>({ U });
        out.counts = Tensor<int64_t>({ U });

        for (int64_t u = 0; u < U; ++u) {
            out.Y.data[(size_t)u] = uniq_vals[(size_t)u];
            out.indices.data[(size_t)u] = uniq_first_pos[(size_t)u];
            out.counts.data[(size_t)u] = uniq_counts[(size_t)u];
        }

        return out;
    }

    // ---------------- Case B: axis provided (unique slices) ----------------
    const int64_t axis = NormalizeAxis(axisOpt.value(), rank);

    // Compute outer/inner sizes for slicing along axis:
    // X shape: [d0 ... d_{axis-1}, d_axis, d_{axis+1} ...]
    // outer = product(d0..d_{axis-1})
    // inner = product(d_{axis+1}..d_{r-1})
    int64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= X.shape[(size_t)i];
    int64_t axisDim = X.shape[(size_t)axis];
    int64_t inner = 1;
    for (int64_t i = axis + 1; i < rank; ++i) inner *= X.shape[(size_t)i];

    // Each slice is of length: outer? Actually for a fixed axis index 'a',
    // slice contains all elements where that axis equals 'a':
    // total slice size = outer * inner, but arranged by outer blocks.
    // We'll encode slice key as a vector<T> of length (outer*inner) in a consistent order.
    const int64_t sliceLen = outer * inner;

    std::unordered_map<SliceKey<T>, int64_t, SliceHash<T>, SliceEq<T>> map;
    map.reserve((size_t)std::min<int64_t>(axisDim, 1024));

    std::vector<SliceKey<T>> uniq_slices;
    std::vector<int64_t> uniq_first_axis;
    std::vector<int64_t> uniq_counts;

    out.inverse_indices = Tensor<int64_t>({ axisDim });

    // Helper to extract slice at axis index 'a' into SliceKey<T>.
    auto extract_slice = [&](int64_t a) -> SliceKey<T> {
        SliceKey<T> key;
        key.v.resize((size_t)sliceLen);

        // For each outer_idx in [0, outer), inner_idx in [0, inner):
        // X linear index = (outer_idx * axisDim + a) * inner + inner_idx
        // We store in key.v in [outer_idx * inner + inner_idx] order.
        for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
            for (int64_t inner_idx = 0; inner_idx < inner; ++inner_idx) {
                int64_t x_lin = (outer_idx * axisDim + a) * inner + inner_idx;
                int64_t k_lin = outer_idx * inner + inner_idx;
                key.v[(size_t)k_lin] = X.data[(size_t)x_lin];
            }
        }
        return key;
        };

    for (int64_t a = 0; a < axisDim; ++a) {
        SliceKey<T> key = extract_slice(a);
        auto it = map.find(key);
        if (it == map.end()) {
            int64_t id = (int64_t)uniq_slices.size();
            map.emplace(key, id);
            uniq_slices.push_back(std::move(key));
            uniq_first_axis.push_back(a);
            uniq_counts.push_back(1);
            out.inverse_indices.data[(size_t)a] = id;
        }
        else {
            int64_t id = it->second;
            uniq_counts[(size_t)id] += 1;
            out.inverse_indices.data[(size_t)a] = id;
        }
    }

    // Optionally sort unique slices lexicographically by their content.
    if (sorted) {
        const int64_t U = (int64_t)uniq_slices.size();
        std::vector<int64_t> order(U);
        std::iota(order.begin(), order.end(), 0);

        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            const auto& va = uniq_slices[(size_t)a].v;
            const auto& vb = uniq_slices[(size_t)b].v;
            // Lexicographic compare on slice content; tie-break by first axis.
            size_t n = std::min(va.size(), vb.size());
            for (size_t i = 0; i < n; ++i) {
                if (!ScalarBitEq<T>()(va[i], vb[i])) return va[i] < vb[i];
            }
            return uniq_first_axis[(size_t)a] < uniq_first_axis[(size_t)b];
            });

        std::vector<int64_t> old_to_new((size_t)U, 0);
        for (int64_t newId = 0; newId < U; ++newId) {
            old_to_new[(size_t)order[(size_t)newId]] = newId;
        }

        std::vector<SliceKey<T>> new_slices(U);
        std::vector<int64_t> new_first(U), new_counts(U);
        for (int64_t newId = 0; newId < U; ++newId) {
            int64_t oldId = order[(size_t)newId];
            new_slices[(size_t)newId] = std::move(uniq_slices[(size_t)oldId]);
            new_first[(size_t)newId] = uniq_first_axis[(size_t)oldId];
            new_counts[(size_t)newId] = uniq_counts[(size_t)oldId];
        }
        uniq_slices.swap(new_slices);
        uniq_first_axis.swap(new_first);
        uniq_counts.swap(new_counts);

        // Remap inverse_indices (length axisDim)
        for (int64_t a = 0; a < axisDim; ++a) {
            int64_t oldId = out.inverse_indices.data[(size_t)a];
            out.inverse_indices.data[(size_t)a] = old_to_new[(size_t)oldId];
        }
    }

    const int64_t U = (int64_t)uniq_slices.size();

    // Build output Y shape: same as X but axis dimension becomes U.
    std::vector<int64_t> y_shape = X.shape;
    y_shape[(size_t)axis] = U;
    out.Y = Tensor<T>(y_shape);

    out.indices = Tensor<int64_t>({ U });
    out.counts = Tensor<int64_t>({ U });

    // Write indices and counts.
    for (int64_t u = 0; u < U; ++u) {
        out.indices.data[(size_t)u] = uniq_first_axis[(size_t)u];
        out.counts.data[(size_t)u] = uniq_counts[(size_t)u];
    }

    // Materialize Y by copying each unique slice back into output at axis=u.
    // out linear index: (outer_idx * U + u) * inner + inner_idx
    for (int64_t u = 0; u < U; ++u) {
        const auto& sv = uniq_slices[(size_t)u].v; // length sliceLen = outer*inner
        for (int64_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
            for (int64_t inner_idx = 0; inner_idx < inner; ++inner_idx) {
                int64_t k_lin = outer_idx * inner + inner_idx;
                int64_t y_lin = (outer_idx * U + u) * inner + inner_idx;
                out.Y.data[(size_t)y_lin] = sv[(size_t)k_lin];
            }
        }
    }

    return out;
}

// --------------------------- Demo helpers ---------------------------
template <typename T>
static void Print1D(const Tensor<T>& t) {
    if (t.rank() != 1) { std::cout << "Print1D: rank != 1\n"; return; }
    std::cout << "[ ";
    for (int64_t i = 0; i < t.shape[0]; ++i) std::cout << t.data[(size_t)i] << " ";
    std::cout << "]\n";
}

template <typename T>
static void Print2D(const Tensor<T>& t) {
    if (t.rank() != 2) { std::cout << "Print2D: rank != 2\n"; return; }
    int64_t H = t.shape[0], W = t.shape[1];
    for (int64_t i = 0; i < H; ++i) {
        std::cout << "[ ";
        for (int64_t j = 0; j < W; ++j) {
            std::cout << std::setw(3) << t.data[(size_t)(i * W + j)] << " ";
        }
        std::cout << "]\n";
    }
}

int main() {
    // ---------------- Example A: no axis (flatten) ----------------
    Tensor<int> X({ 2, 3 });
    // X =
    // [ [1, 2, 1],
    //   [3, 2, 3] ]
    X.data = { 1,2,1, 3,2,3 };

    std::cout << "X (2x3):\n";
    Print2D(X);

    auto outA = Unique<int>(X, std::nullopt, /*sorted=*/0);
    std::cout << "\nUnique(no axis, sorted=0)\n";
    std::cout << "Y: "; Print1D(outA.Y);
    std::cout << "indices: "; Print1D(outA.indices);
    std::cout << "inverse_indices: "; Print1D(outA.inverse_indices);
    std::cout << "counts: "; Print1D(outA.counts);

    // ---------------- Example B: axis=0 (unique rows) ----------------
    // Make rows with duplication:
    // row0 = [5,6]
    // row1 = [7,8]
    // row2 = [5,6]  (duplicate of row0)
    Tensor<int> R({ 3, 2 });
    R.data = { 5,6, 7,8, 5,6 };

    std::cout << "\nR (3x2):\n";
    Print2D(R);

    auto outB = Unique<int>(R, /*axis=*/0, /*sorted=*/0);
    std::cout << "\nUnique(axis=0, sorted=0)\n";
    std::cout << "Y shape=[" << outB.Y.shape[0] << "," << outB.Y.shape[1] << "]\n";
    Print2D(outB.Y);
    std::cout << "indices: "; Print1D(outB.indices);
    std::cout << "inverse_indices: "; Print1D(outB.inverse_indices);
    std::cout << "counts: "; Print1D(outB.counts);

    // ---------------- Example C: axis=1 (unique columns) with sorted=1 ----------------
    // C =
    // [ [1, 9, 1],
    //   [2, 8, 2] ]
    // column0=[1,2], column1=[9,8], column2=[1,2] duplicate
    Tensor<int> C({ 2, 3 });
    C.data = { 1,9,1, 2,8,2 };

    std::cout << "\nC (2x3):\n";
    Print2D(C);

    auto outC = Unique<int>(C, /*axis=*/1, /*sorted=*/1);
    std::cout << "\nUnique(axis=1, sorted=1)\n";
    std::cout << "Y shape=[" << outC.Y.shape[0] << "," << outC.Y.shape[1] << "]\n";
    Print2D(outC.Y);
    std::cout << "indices: "; Print1D(outC.indices);
    std::cout << "inverse_indices: "; Print1D(outC.inverse_indices);
    std::cout << "counts: "; Print1D(outC.counts);

    return 0;
}
