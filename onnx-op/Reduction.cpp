#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <stdexcept>

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data; // row-major
    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t dim : shape) n *= dim;
        return n;
    }

    const float* data_ptr() const { return data.data(); }
    float* data_ptr() { return data.data(); }
};

enum class ReduceOpType {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    L1,
    L2,
    LogSum,
    LogSumExp,
    SumSquare
};

// Initialize the accumulator based on the operation type
static float InitValue(ReduceOpType op) {
    switch (op) {
    case ReduceOpType::Sum: return 0.0f;
    case ReduceOpType::Prod: return 1.0f;
    case ReduceOpType::Max: return -std::numeric_limits<float>::infinity();
    case ReduceOpType::Min: return std::numeric_limits<float>::infinity();
    default: return 0.0f;
    }
}

// Update the accumulator based on the operation type
static float ReduceUpdate(float acc, float x, ReduceOpType op) {
    switch (op) {
    case ReduceOpType::Sum: return acc + x;
    case ReduceOpType::Prod: return acc * x;
    case ReduceOpType::Max: return std::max(acc, x);
    case ReduceOpType::Min: return std::min(acc, x);
    default: return acc;
    }
}

// Finalize the accumulator, performing operations like mean or square root
static float ReduceFinalize(float acc, int64_t reduce_count, ReduceOpType op) {
    if (op == ReduceOpType::Mean) {
        return acc / static_cast<float>(reduce_count);
    }
    return acc;
}

static int64_t getLinearIndex(const std::vector<int64_t>& out_multi, const std::vector<int64_t>& strides) {
    int64_t out_index = 0;
    int64_t rank = out_multi.size();

    // Calculate the linear index by summing the product of each dimension's index and its stride
    for (int64_t i = 0; i < rank; ++i) {
        out_index += out_multi[i] * strides[i];
    }

    return out_index;
}

// Main reduction function that performs the specified reduction operation
static Tensor ReduceONNX(const Tensor& input, std::vector<int64_t> axes, bool keepdims, ReduceOpType op) {
    const int64_t rank = input.rank();
    std::vector<char> reduce_flag(rank, 0);

    // Mark the reduced axes
    for (int64_t ax : axes) reduce_flag[ax] = 1;

    // Calculate the output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < rank; ++i) {
        if (reduce_flag[i] && keepdims) {
            out_shape.push_back(1);  // Keep the dimension with size 1 if keepdims is true
        }
        else if (!reduce_flag[i]) {
            out_shape.push_back(input.shape[i]);  // Retain non-reduced dimensions
        }
    }

    // Initialize the output tensor
    Tensor output;
    output.shape = out_shape;
    output.data.resize(output.numel(), 0.0f);

    // Calculate reduce_count (number of elements in the reduced dimensions)
    int64_t reduce_count = 1;
    for (int64_t ax : axes) {
        reduce_count *= input.shape[ax];
    }

    // Calculate strides for each dimension in reverse order
    std::vector<int64_t> strides(rank, 1);  // Initialize strides
    for (int64_t i = rank - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * input.shape[i + 1];  // Calculate stride based on the next dimension
    }

    std::vector<int64_t> out_strides(out_shape.size(), 1);  // Initialize strides
    for (int64_t i = out_shape.size() - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * output.shape[i + 1];  // Calculate stride based on the next dimension
    }

    // Iterate through the input tensor and perform the reduction operation
    for (int64_t in_linear = 0; in_linear < input.numel(); ++in_linear) {
        // Convert the input linear index to input multi-dimensional coordinates
        std::vector<int64_t> in_multi(rank, 0);
        int64_t tmp = in_linear;
        for (int64_t i = 0; i < rank; ++i) {
            in_multi[i] = tmp / strides[i];  // Calculate the multi-dimensional index
            tmp -= in_multi[i] * strides[i];  // Update tmp for the next dimension
        }

        // For each output element, determine its corresponding multi-index in the input
        std::vector<int64_t> out_multi(in_multi);
        if (keepdims) {
            // If keepdims is true, set the reduced dimensions' coordinates to 0
            for (int64_t ax : axes) {
                out_multi[ax] = 0;
            }
        }
        else {
            // If keepdims is false, remove the reduced dimensions
            for (auto i = 0; i < axes.size(); ++i) {
                out_multi.erase(out_multi.begin() + axes[i] - i);
            }
        }

        auto out_linear = getLinearIndex(out_multi, out_strides);
        output.data[out_linear] = ReduceUpdate(output.data[out_linear], input.data[out_linear], op);

    }

    return output;
}

// Print the tensor's shape and data
static void PrintTensor(const Tensor& t, const std::string& name) {
    std::cout << name << " shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        std::cout << t.shape[i] << (i + 1 == t.shape.size() ? "" : ", ");
    }
    std::cout << "] data={";
    for (size_t i = 0; i < t.data.size(); ++i) {
        std::cout << t.data[i] << (i + 1 == t.data.size() ? "" : ", ");
    }
    std::cout << "}\n";
}

int main() {
    // Example: Input tensor shape = [2, 3, 4] and output shape = [1, 3, 4]
    Tensor x;
    x.shape = { 2, 3, 4 };
    x.data = { 1, 2, 3, 4,   5, 6, 7, 8,   9, 10, 11, 12,
              13, 14, 15, 16,  17, 18, 19, 20,  21, 22, 23, 24 };

    // Test ReduceSum over axis=0, keepdims=1 => shape (1, 3, 4)
    {
        Tensor y = ReduceONNX(x, { 0 }, true, ReduceOpType::Sum);
        PrintTensor(y, "ReduceSum(axis=0, keepdims=1)");
    }

    // Test ReduceSum over axis=1, keepdims=0 => shape (2, 1, 4)
    {
        Tensor y = ReduceONNX(x, { 1 }, false, ReduceOpType::Sum);
        PrintTensor(y, "ReduceSum(axis=1, keepdims=0)");
    }

    // Test ReduceSum over axis=2, keepdims=0 => shape (2, 3, 1)
    {
        Tensor y = ReduceONNX(x, { 2 }, false, ReduceOpType::Sum);
        PrintTensor(y, "ReduceSum(axis=2, keepdims=0)");
    }

    return 0;
}
