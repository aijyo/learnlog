#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// --------------------------- Hardmax (ONNX-style) ---------------------------
//
// Hardmax produces a one-hot tensor along a given axis.
// For each slice along 'axis', it finds the index of the maximum value and sets:
//   output[max_index] = 1, others = 0
//
// - Input and output have the same shape.
// - axis must be in [-rank, rank-1]. Negative axis means axis += rank.
// - Tie-breaking: if multiple max values exist, pick the first max (smallest index).
// ---------------------------------------------------------------------------

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data; // row-major contiguous

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t p = 1;
        for (int64_t d : shape) p *= d;
        return p;
    }
};

static inline int64_t prod(const std::vector<int64_t>& v, int64_t begin, int64_t end) {
    // English comment: product of v[begin..end-1]
    int64_t p = 1;
    for (int64_t i = begin; i < end; ++i) p *= v[static_cast<size_t>(i)];
    return p;
}

Tensor Hardmax(const Tensor& X, int64_t axis) {
    if (X.rank() == 0) {
        throw std::invalid_argument("Hardmax: input rank must be >= 1.");
    }
    int64_t r = X.rank();

    // English comment: validate axis range [-r, r-1]
    if (axis < -r || axis >= r) {
        throw std::invalid_argument("Hardmax: axis out of range.");
    }

    // English comment: normalize negative axis
    if (axis < 0) axis += r;

    // English comment: compute outer/axisDim/inner for flattening
    int64_t outer = prod(X.shape, 0, axis);
    int64_t axisDim = X.shape[static_cast<size_t>(axis)];
    int64_t inner = prod(X.shape, axis + 1, r);

    if (axisDim <= 0) {
        throw std::invalid_argument("Hardmax: axis dimension must be > 0.");
    }
    if (X.numel() != static_cast<int64_t>(X.data.size())) {
        throw std::invalid_argument("Hardmax: data size does not match shape.");
    }

    Tensor Y;
    Y.shape = X.shape;
    Y.data.assign(X.data.size(), 0.0f);

    // English comment:
    // Layout index mapping:
    // For a fixed (o, k, i):
    //   linear_index = (o * axisDim + k) * inner + i
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < inner; ++i) {
            // Find argmax along k in [0, axisDim)
            int64_t bestK = 0;
            float bestV = X.data[static_cast<size_t>(((o * axisDim + 0) * inner + i))];

            for (int64_t k = 1; k < axisDim; ++k) {
                float v = X.data[static_cast<size_t>(((o * axisDim + k) * inner + i))];
                // English comment: pick first max on ties by using strict '>'
                if (v > bestV) {
                    bestV = v;
                    bestK = k;
                }
            }

            // Write one-hot
            Y.data[static_cast<size_t>(((o * axisDim + bestK) * inner + i))] = 1.0f;
        }
    }

    return Y;
}

static void printTensorFlat(const Tensor& T, const std::string& name) {
    std::cout << name << " shape=[";
    for (size_t i = 0; i < T.shape.size(); ++i) {
        std::cout << T.shape[i] << (i + 1 < T.shape.size() ? "," : "");
    }
    std::cout << "]\n";
    std::cout << "data: ";
    for (size_t i = 0; i < T.data.size(); ++i) {
        std::cout << T.data[i] << (i + 1 < T.data.size() ? ", " : "");
    }
    std::cout << "\n\n";
}

int main() {
    try {
        // ---------------- Demo 1: 2D input, axis=1 (row-wise hardmax) ----------------
        // X shape: (2, 4)
        // row0: [ 1, 3, 2, 3 ] -> max is 3, tie at index 1 and 3 -> pick first => index 1
        // row1: [ 0, -1, 5, 4 ] -> max at index 2
        Tensor X1;
        X1.shape = { 2, 4 };
        X1.data = { 1, 3, 2, 3,
                    0, -1, 5, 4 };

        Tensor Y1 = Hardmax(X1, /*axis=*/1);
        printTensorFlat(X1, "X1");
        printTensorFlat(Y1, "Y1 = Hardmax(X1, axis=1)");

        // ---------------- Demo 2: 3D input, axis=-1 (last dim hardmax) ----------------
        // Shape: (1, 2, 3). For each (n,c), hardmax over length-3 vector.
        Tensor X2;
        X2.shape = { 1, 2, 3 };
        X2.data = {
            // n=0,c=0: [0.1, 0.2, 0.0] -> index 1
            0.1f, 0.2f, 0.0f,
            // n=0,c=1: [5.0, 5.0, 4.9] -> tie at 0 and 1 -> pick 0
            5.0f, 5.0f, 4.9f
        };

        Tensor Y2 = Hardmax(X2, /*axis=*/-1);
        printTensorFlat(X2, "X2");
        printTensorFlat(Y2, "Y2 = Hardmax(X2, axis=-1)");

        // ---------------- Demo 3: axis=0 (column-wise hardmax for 2D) ----------------
        // Shape (3,2), axis=0 means for each column, pick max across rows.
        Tensor X3;
        X3.shape = { 3, 2 };
        X3.data = {
          1, 10,
          2,  9,
          3,  8
        };
        Tensor Y3 = Hardmax(X3, /*axis=*/0);
        printTensorFlat(X3, "X3");
        printTensorFlat(Y3, "Y3 = Hardmax(X3, axis=0)");

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
