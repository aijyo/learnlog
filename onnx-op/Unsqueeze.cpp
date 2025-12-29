#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <memory>
#include <iomanip>

// --------------------------- Tensor (Minimal) ---------------------------
//
// This tensor supports:
// - owning storage via shared_ptr<vector<T>> (so we can create views cheaply)
// - row-major strides
// - reshape-like view: same buffer, different shape/strides
//
template <typename T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::shared_ptr<std::vector<T>> storage; // shared data
    int64_t offset = 0; // element offset (for views)

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
        storage = std::make_shared<std::vector<T>>();
        storage->resize((size_t)numel_from_shape(shape));
        offset = 0;
    }

    int64_t rank() const { return (int64_t)shape.size(); }
    int64_t numel() const { return numel_from_shape(shape); }

    T* data() { return storage->data() + offset; }
    const T* data() const { return storage->data() + offset; }

    // Create a view with new shape and row-major strides; requires same numel.
    Tensor<T> reshape_view(const std::vector<int64_t>& new_shape) const {
        if (numel_from_shape(new_shape) != numel())
            throw std::invalid_argument("reshape_view: numel mismatch.");
        Tensor<T> v;
        v.shape = new_shape;
        v.strides = make_row_major_strides(new_shape);
        v.storage = storage;
        v.offset = offset;
        return v;
    }
};

// --------------------------- Unsqueeze (ONNX-like) ---------------------------
//
// Adds axes of length 1 at the specified positions.
//
// Shape rule (matches onnx-mlir snippet):
// - outRank = dataRank + axes.size()
// - normalize each axis:
//     * axis must be in [-outRank, outRank-1]
//     * if axis < 0: axis += outRank
//     * no duplicates allowed
// - build output shape by iterating i=0..outRank-1:
//     if i in axes: outputDims.push_back(1)
//     else: outputDims.push_back(inputDims[j++])
//
// Execution rule:
// - Unsqueeze does NOT change element order.
// - It is essentially a reshape, so we can return a view (zero-copy).
//
static std::vector<int64_t> UnsqueezeInferShape(
    const std::vector<int64_t>& in_shape,
    const std::vector<int64_t>& axes_raw) {

    const int64_t dataRank = (int64_t)in_shape.size();
    const int64_t outRank = dataRank + (int64_t)axes_raw.size();

    // Normalize axes and check duplicates.
    std::vector<int64_t> axes;
    axes.reserve(axes_raw.size());

    for (size_t i = 0; i < axes_raw.size(); ++i) {
        int64_t a = axes_raw[i];

        // Range check uses outRank (same as onnx-mlir).
        if (a < -outRank || a >= outRank)
            throw std::invalid_argument("Unsqueeze: invalid axis value (out of range).");

        if (a < 0) a += outRank;

        // Duplicate check.
        if (std::find(axes.begin(), axes.end(), a) != axes.end())
            throw std::invalid_argument("Unsqueeze: duplicated axes.");

        axes.push_back(a);
    }

    // Build output shape.
    std::vector<int64_t> out_shape;
    out_shape.reserve((size_t)outRank);

    int64_t j = 0;
    for (int64_t i = 0; i < outRank; ++i) {
        if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
            out_shape.push_back(1);
        }
        else {
            out_shape.push_back(in_shape[(size_t)j]);
            j++;
        }
    }

    return out_shape;
}

template <typename T>
static Tensor<T> Unsqueeze(const Tensor<T>& data, const std::vector<int64_t>& axes_raw) {
    // Infer output shape.
    std::vector<int64_t> out_shape = UnsqueezeInferShape(data.shape, axes_raw);

    // Unsqueeze is reshape-like: data order unchanged, just a view.
    return data.reshape_view(out_shape);
}

// --------------------------- Demo helpers ---------------------------
static void PrintShape(const std::vector<int64_t>& s) {
    std::cout << "[";
    for (size_t i = 0; i < s.size(); ++i) {
        std::cout << s[i];
        if (i + 1 < s.size()) std::cout << ", ";
    }
    std::cout << "]";
}

template <typename T>
static void Print1D(const Tensor<T>& t) {
    if (t.rank() != 1) { std::cout << "Print1D: rank != 1\n"; return; }
    std::cout << "[ ";
    for (int64_t i = 0; i < t.shape[0]; ++i) std::cout << t.data()[i] << " ";
    std::cout << "]\n";
}

int main() {
    // Input: shape [3,4]
    Tensor<float> X({ 3, 4 });
    for (int i = 0; i < (int)X.storage->size(); ++i) {
        (*X.storage)[(size_t)i] = (float)(i + 1); // 1..12
    }

    std::cout << "X shape="; PrintShape(X.shape); std::cout << "\n";
    std::cout << "X numel=" << X.numel() << "\n";

    // Example 1: axes = [0] => [1,3,4]
    std::vector<int64_t> axes1 = { 0 };
    auto Y1 = Unsqueeze(X, axes1);
    std::cout << "\naxes="; PrintShape(axes1); std::cout << " => Y1 shape=";
    PrintShape(Y1.shape); std::cout << "\n";

    // Example 2: axes = [1] => [3,1,4]
    std::vector<int64_t> axes2 = { 1 };
    auto Y2 = Unsqueeze(X, axes2);
    std::cout << "axes="; PrintShape(axes2); std::cout << " => Y2 shape=";
    PrintShape(Y2.shape); std::cout << "\n";

    // Example 3: axes = [-1] => outRank=3, -1 -> 2 => [3,4,1]
    std::vector<int64_t> axes3 = { -1 };
    auto Y3 = Unsqueeze(X, axes3);
    std::cout << "axes="; PrintShape(axes3); std::cout << " => Y3 shape=";
    PrintShape(Y3.shape); std::cout << "\n";

    // Example 4: axes = [0, 2] on [3,4] => outRank=4 => [1,3,1,4]
    std::vector<int64_t> axes4 = { 0, 2 };
    auto Y4 = Unsqueeze(X, axes4);
    std::cout << "axes="; PrintShape(axes4); std::cout << " => Y4 shape=";
    PrintShape(Y4.shape); std::cout << "\n";

    // Verify data unchanged (same buffer)
    std::cout << "\nCheck: Y4 shares storage with X? "
        << (Y4.storage.get() == X.storage.get() ? "YES" : "NO") << "\n";

    // Optional: show flatten view equals original order
    auto Xflat = X.reshape_view({ X.numel() });
    auto Y4flat = Y4.reshape_view({ Y4.numel() });
    std::cout << "First 8 elements in X flat: ";
    for (int i = 0; i < 8; ++i) std::cout << Xflat.data()[i] << " ";
    std::cout << "\nFirst 8 elements in Y4 flat: ";
    for (int i = 0; i < 8; ++i) std::cout << Y4flat.data()[i] << " ";
    std::cout << "\n";

    return 0;
}
