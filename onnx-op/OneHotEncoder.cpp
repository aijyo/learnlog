#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// --------------------------- Tensor (Row-Major, Minimal) ---------------------------
//
// A tiny N-D tensor helper used for CPU reference implementations.
// - shape: vector<int64_t>
// - row-major contiguous storage
//
template <typename T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<T> data;

    Tensor() = default;

    explicit Tensor(std::vector<int64_t> shp) : shape(std::move(shp)) {
        strides = computeStrides(shape);
        data.resize(static_cast<size_t>(numel(shape)));
    }

    static int64_t numel(const std::vector<int64_t>& shp) {
        int64_t n = 1;
        for (int64_t d : shp) {
            if (d <= 0) throw std::runtime_error("Invalid dimension <= 0.");
            n *= d;
        }
        return n;
    }

    static std::vector<int64_t> computeStrides(const std::vector<int64_t>& shp) {
        std::vector<int64_t> st(shp.size(), 1);
        for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i)
            st[i] = st[i + 1] * shp[i + 1];
        return st;
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }
    int64_t numel() const { return numel(shape); }
};

// Convert flat index -> multi-index.
static std::vector<int64_t> unravelIndex(int64_t flat,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides) {
    std::vector<int64_t> idx(shape.size(), 0);
    int64_t rem = flat;
    for (size_t i = 0; i < shape.size(); ++i) {
        idx[i] = rem / strides[i];
        rem = rem % strides[i];
    }
    return idx;
}

// Convert multi-index -> flat index.
static int64_t ravelIndex(const std::vector<int64_t>& idx,
    const std::vector<int64_t>& strides) {
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
    return off;
}

template <typename T>
static void printTensor(const Tensor<T>& t, const std::string& name, int64_t maxPrint = 64) {
    std::cout << name << " shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i)
        std::cout << t.shape[i] << (i + 1 == t.shape.size() ? "" : ",");
    std::cout << "] data=(";
    int64_t n = std::min<int64_t>(t.numel(), maxPrint);
    for (int64_t i = 0; i < n; ++i) {
        std::cout << t.data[static_cast<size_t>(i)] << (i + 1 == n ? "" : ", ");
    }
    if (t.numel() > n) std::cout << ", ...";
    std::cout << ")\n";
}

// --------------------------- OneHotEncoder (ONNX) ---------------------------
//
// This op encodes each element of X into a one-hot vector over a category list.
// Output Y has one more dimension than X, and the last dimension equals #categories.
//
// Category source:
//   - If X is numeric (int/float), cast X elements to int64 and look up in cats_int64s.
//   - Else (e.g. string tensor), look up in cats_strings.
//
// Semantics for each element x at position p in X:
//   Let j be the index where cats[j] == x. If found:
//     Y[p, j] = 1.0, and Y[p, k!=j] = 0.0
//   If not found:
//     Y[p, :] = all zeros.
//
// Notes:
//   - Output type is float (float32), following onnx-mlir inferShapes.
//   - This reference implementation uses exact equality for matching.
//
Tensor<float> oneHotEncoderNumeric(const Tensor<double>& X,
    const std::vector<int64_t>& cats_int64s) {
    if (cats_int64s.empty()) throw std::runtime_error("cats_int64s must not be empty.");

    // Build output shape = X.shape + [numCats].
    std::vector<int64_t> outShape = X.shape;
    outShape.push_back(static_cast<int64_t>(cats_int64s.size()));
    Tensor<float> Y(outShape);

    // Initialize to 0.
    for (auto& v : Y.data) v = 0.0f;

    // Build a map cat_value -> cat_index for O(1) lookup.
    std::unordered_map<int64_t, int64_t> cat2idx;
    cat2idx.reserve(cats_int64s.size());
    for (int64_t j = 0; j < static_cast<int64_t>(cats_int64s.size()); ++j)
        cat2idx[cats_int64s[static_cast<size_t>(j)]] = j;

    int64_t numCats = static_cast<int64_t>(cats_int64s.size());
    int64_t xNumel = X.numel();

    for (int64_t i = 0; i < xNumel; ++i) {
        // Cast numeric input to int64 for lookup (per onnx-mlir comment).
        // Here we use truncation toward zero like C++ static_cast.
        int64_t key = static_cast<int64_t>(X.data[static_cast<size_t>(i)]);

        auto it = cat2idx.find(key);
        if (it == cat2idx.end()) continue;

        int64_t j = it->second;

        // Compute output index: outIdx = idx(X) + [j]
        std::vector<int64_t> idx = unravelIndex(i, X.shape, X.strides);
        idx.push_back(j);

        int64_t outFlat = ravelIndex(idx, Y.strides);
        Y.data[static_cast<size_t>(outFlat)] = 1.0f;
    }

    return Y;
}

Tensor<float> oneHotEncoderString(const Tensor<std::string>& X,
    const std::vector<std::string>& cats_strings) {
    if (cats_strings.empty()) throw std::runtime_error("cats_strings must not be empty.");

    std::vector<int64_t> outShape = X.shape;
    outShape.push_back(static_cast<int64_t>(cats_strings.size()));
    Tensor<float> Y(outShape);

    for (auto& v : Y.data) v = 0.0f;

    std::unordered_map<std::string, int64_t> cat2idx;
    cat2idx.reserve(cats_strings.size());
    for (int64_t j = 0; j < static_cast<int64_t>(cats_strings.size()); ++j)
        cat2idx[cats_strings[static_cast<size_t>(j)]] = j;

    int64_t xNumel = X.numel();

    for (int64_t i = 0; i < xNumel; ++i) {
        const std::string& key = X.data[static_cast<size_t>(i)];
        auto it = cat2idx.find(key);
        if (it == cat2idx.end()) continue;

        int64_t j = it->second;

        std::vector<int64_t> idx = unravelIndex(i, X.shape, X.strides);
        idx.push_back(j);

        int64_t outFlat = ravelIndex(idx, Y.strides);
        Y.data[static_cast<size_t>(outFlat)] = 1.0f;
    }

    return Y;
}

// --------------------------- Demo ---------------------------
//
// Build & run:
//   g++ -O2 -std=c++17 onehot_encoder_demo.cpp -o onehot_encoder_demo && ./onehot_encoder_demo
//
int main() {
    try {
        // ===== Demo 1: Numeric X -> cats_int64s =====
        // X shape [2,3]
        Tensor<double> Xnum({ 2, 3 });
        Xnum.data = {
          10.0, 20.0, 30.0,
          20.0, 99.0, -1.0
        };

        // Categories dictionary (attribute): cats_int64s
        // outDim = 4, output shape = [2,3,4]
        std::vector<int64_t> cats_int64s = { 10, 20, 30, 40 };

        auto Ynum = oneHotEncoderNumeric(Xnum, cats_int64s);

        printTensor(Xnum, "Xnum");
        printTensor(Ynum, "Ynum (OneHotEncoder)");

        // Interpretation:
        // - Xnum[0,0]=10 -> [1,0,0,0]
        // - Xnum[0,1]=20 -> [0,1,0,0]
        // - Xnum[0,2]=30 -> [0,0,1,0]
        // - Xnum[1,0]=20 -> [0,1,0,0]
        // - Xnum[1,1]=99 -> not found -> [0,0,0,0]
        // - Xnum[1,2]=-1 -> not found -> [0,0,0,0]

        // ===== Demo 2: String X -> cats_strings =====
        Tensor<std::string> Xstr({ 2, 2 });
        Xstr.data = {
          "red", "blue",
          "green", "unknown"
        };

        std::vector<std::string> cats_strings = { "blue", "green", "red" };

        auto Ystr = oneHotEncoderString(Xstr, cats_strings);

        // Print string tensor manually (simple)
        std::cout << "Xstr shape=[2,2] data=("
            << Xstr.data[0] << ", " << Xstr.data[1] << ", "
            << Xstr.data[2] << ", " << Xstr.data[3] << ")\n";
        printTensor(Ystr, "Ystr (OneHotEncoder)");

        // Interpretation for cats_strings = [blue, green, red]:
        // - "red"     -> [0,0,1]
        // - "blue"    -> [1,0,0]
        // - "green"   -> [0,1,0]
        // - "unknown" -> [0,0,0]

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
