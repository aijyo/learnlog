#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

// A simple float tensor in row-major layout.
struct TensorF {
    std::vector<int64_t> shape;   // Tensor shape, e.g. {3, 3}
    std::vector<float> data;      // Row-major contiguous data

    TensorF(std::vector<int64_t> s, std::vector<float> d)
        : shape(std::move(s)), data(std::move(d)) {
        int64_t n = 1;
        for (int64_t dim : shape) n *= dim;
        if ((int64_t)data.size() != n) {
            throw std::runtime_error("TensorF: data size does not match shape.");
        }
    }

    int64_t rank() const { return (int64_t)shape.size(); }
    int64_t numel() const { return (int64_t)data.size(); }
};

// Compute row-major strides.
// For shape [D0,D1,...,D{r-1}], strides are:
// stride[i] = product_{j=i+1..r-1} Dj
static std::vector<int64_t> computeRowMajorStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// Convert a linear index to a multi-dimensional index for row-major tensor.
static std::vector<int64_t> linearToCoord(int64_t linear,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides) {
    const int64_t rank = (int64_t)shape.size();
    std::vector<int64_t> coord(rank, 0);
    int64_t remaining = linear;
    for (int64_t axis = 0; axis < rank; ++axis) {
        int64_t s = strides[(size_t)axis];
        coord[(size_t)axis] = (s == 0) ? 0 : (remaining / s);
        remaining = (s == 0) ? 0 : (remaining % s);
    }
    return coord;
}

// ONNX NonZero:
// Output is int64 tensor of shape [rank, nnz].
static std::vector<int64_t> NonZeroONNX(const TensorF& x,
    std::vector<int64_t>& outShape /* [rank, nnz] */) {
    if (x.rank() < 1) {
        throw std::runtime_error("NonZeroONNX: input rank must be >= 1.");
    }

    const int64_t rank = x.rank();
    const int64_t N = x.numel();
    auto strides = computeRowMajorStrides(x.shape);

    // Collect coordinates of non-zero elements.
    std::vector<std::vector<int64_t>> coords;
    coords.reserve((size_t)N);

    for (int64_t i = 0; i < N; ++i) {
        // Note: ONNX treats "non-zero" as x != 0. For floats, exact compare is used here.
        // If you want epsilon behavior, adjust it explicitly.
        if (x.data[(size_t)i] != 0.0f) {
            coords.push_back(linearToCoord(i, x.shape, strides));
        }
    }

    const int64_t nnz = (int64_t)coords.size();
    outShape = { rank, nnz };

    // Layout output as row-major [rank, nnz].
    // Y[axis, k] = coords[k][axis]
    std::vector<int64_t> y((size_t)(rank * nnz), 0);
    for (int64_t k = 0; k < nnz; ++k) {
        for (int64_t axis = 0; axis < rank; ++axis) {
            y[(size_t)(axis * nnz + k)] = coords[(size_t)k][(size_t)axis];
        }
    }

    return y;
}

static void printNonZeroOutput(const std::vector<int64_t>& y,
    const std::vector<int64_t>& outShape) {
    if (outShape.size() != 2) throw std::runtime_error("Output shape must be [2].");
    int64_t rank = outShape[0];
    int64_t nnz = outShape[1];
    std::cout << "Output shape = [" << rank << ", " << nnz << "]\n";

    // Print columns (each column is a coordinate)
    for (int64_t k = 0; k < nnz; ++k) {
        std::cout << "[";
        for (int64_t axis = 0; axis < rank; ++axis) {
            std::cout << y[(size_t)(axis * nnz + k)] << (axis + 1 == rank ? "" : ", ");
        }
        std::cout << "]\n";
    }
}

int main() {
    try {
        // Example tensor:
        // [[0, 1, 0],
        //  [0, 0, 2],
        //  [3, 0, 0]]
        TensorF x({ 3, 3 }, {
          0.0f, 1.0f, 0.0f,
          0.0f, 0.0f, 2.0f,
          3.0f, 0.0f, 0.0f
            });

        std::vector<int64_t> outShape;
        auto y = NonZeroONNX(x, outShape);

        std::cout << "NonZero indices (each row is one coordinate):\n";
        printNonZeroOutput(y, outShape);

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
