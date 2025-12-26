#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

class Tensor {
public:
    // Constructor for Tensor class
    Tensor(std::vector<int64_t> shape, std::vector<float> values)
        : shape_(shape), values_(values) {}

    const std::vector<int64_t>& getShape() const { return shape_; }
    const std::vector<float>& getValues() const { return values_; }
    size_t getSize() const { return values_.size(); }

private:
    std::vector<int64_t> shape_;
    std::vector<float> values_;
};

class ONNXRangeOp {
public:
    // Constructor for ONNXRangeOp class
    ONNXRangeOp(Tensor start, Tensor limit, Tensor delta)
        : start_(start), limit_(limit), delta_(delta) {}

    // Compute the range based on start, limit, and delta
    Tensor computeShape() {
        // Verify input tensor shapes and types
        if (!verify()) {
            std::cerr << "Error: Input tensors are invalid!" << std::endl;
            exit(1);
        }

        // Compute the range (size) of the tensor
        bool isFloat = isFloatType(start_);
        float num = 0.0;
        if (isFloat) {
            float startVal = start_.getValues()[0];
            float limitVal = limit_.getValues()[0];
            float deltaVal = delta_.getValues()[0];

            // Compute number of elements in range
            num = std::ceil((limitVal - startVal) / deltaVal);
        } else {
            std::cerr << "Error: Only float type is supported!" << std::endl;
            exit(1);
        }

        // Return the result as a tensor with 1 dimension
        return Tensor({1}, {num});
    }

    // Verify that input tensors are valid
    bool verify() {
        // Validate tensor rank and types
        if (start_.getShape().size() > 1 || limit_.getShape().size() > 1 || delta_.getShape().size() > 1) {
            std::cerr << "Error: Tensors must have rank 1 or less." << std::endl;
            return false;
        }

        // Check if tensor element types match
        if (start_.getValues().size() != limit_.getValues().size() || start_.getValues().size() != delta_.getValues().size()) {
            std::cerr << "Error: Input tensors must have the same size." << std::endl;
            return false;
        }

        return true;
    }

private:
    Tensor start_;
    Tensor limit_;
    Tensor delta_;

    // Check if the tensor elements are of float type
    bool isFloatType(const Tensor& tensor) {
        return tensor.getValues()[0] == static_cast<float>(tensor.getValues()[0]);
    }
};

int main() {
    // Example usage
    std::vector<float> startVals = {0.0};   // Start value
    std::vector<float> limitVals = {10.0};  // Limit value
    std::vector<float> deltaVals = {1.0};   // Delta value

    // Create tensors for start, limit, and delta
    Tensor start({1}, startVals);
    Tensor limit({1}, limitVals);
    Tensor delta({1}, deltaVals);

    // Instantiate the ONNXRangeOp
    ONNXRangeOp rangeOp(start, limit, delta);

    // Compute the shape
    Tensor result = rangeOp.computeShape();

    // Print the result
    std::cout << "Computed shape size: " << result.getValues()[0] << std::endl;

    return 0;
}
