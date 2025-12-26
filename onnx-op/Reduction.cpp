#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>

// Tensor class to represent the tensor structure
class Tensor {
public:
    Tensor(std::vector<int64_t> shape, std::vector<float> values)
        : shape_(shape), values_(values) {
    }

    const std::vector<int64_t>& getShape() const { return shape_; }
    const std::vector<float>& getValues() const { return values_; }
    size_t getSize() const { return values_.size(); }

    // Print tensor details
    void print() const {
        std::cout << "Tensor Shape: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i != shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]\nValues: [";
        for (size_t i = 0; i < values_.size(); ++i) {
            std::cout << values_[i];
            if (i != values_.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

private:
    std::vector<int64_t> shape_;
    std::vector<float> values_;
};

// Helper to handle reduction
template <typename OP_TYPE>
class ONNXReductionOpShapeHelper {
public:
    ONNXReductionOpShapeHelper(Tensor& data, std::vector<int64_t>& axes, bool keepDims)
        : data_(data), axes_(axes), keepDims_(keepDims) {
    }

    // Custom shape computation
    void customComputeShape() {
        int64_t rank = data_.getShape().size();
        std::vector<int64_t> uniqueAxes;

        // Normalize axes (to handle negative indices)
        for (int64_t axis : axes_) {
            if (axis < -rank || axis >= rank) {
                std::cerr << "Error: reduction axis is out of bound" << std::endl;
                exit(1);
            }
            axis = axis >= 0 ? axis : (rank + axis);
            if (std::find(uniqueAxes.begin(), uniqueAxes.end(), axis) == uniqueAxes.end()) {
                uniqueAxes.push_back(axis);
            }
        }

        // Mark reduction axes
        isReductionAxis_.resize(rank, false);
        for (int64_t axis : uniqueAxes) {
            isReductionAxis_[axis] = true;
        }

        // Generate output dimensions
        std::vector<int64_t> outputDims;
        for (int64_t i = 0; i < rank; ++i) {
            if (isReductionAxis_[i]) {
                outputDims.push_back(keepDims_ ? 1 : 0);  // Keep the dimension or remove it
            }
            else {
                outputDims.push_back(data_.getShape()[i]);
            }
        }

        // Output result
        outputDims_ = outputDims;
    }

    // Get the output dimensions
    const std::vector<int64_t>& getOutputDims() const {
        return outputDims_;
    }

private:
    Tensor& data_;
    std::vector<int64_t>& axes_;
    bool keepDims_;
    std::vector<bool> isReductionAxis_;
    std::vector<int64_t> outputDims_;
};

// Generic Reduction Operation (e.g., ReduceSum)
class ONNXReduceOp {
public:
    ONNXReduceOp(Tensor& data, std::vector<int64_t>& axes, bool keepDims, std::string opType)
        : data_(data), axes_(axes), keepDims_(keepDims), opType_(opType) {
    }

    // Compute the shape for Reduce operation and perform the actual reduction
    void computeShape() {
        std::cout << "Input Tensor:\n";
        data_.print();

        ONNXReductionOpShapeHelper<ONNXReduceOp> shapeHelper(data_, axes_, keepDims_);
        shapeHelper.customComputeShape();

        // Perform the actual reduction operation
        performReduction();

        outputDims_ = shapeHelper.getOutputDims();
    }

    // Get the output dimensions of the operation
    const std::vector<int64_t>& getOutputDims() const {
        return outputDims_;
    }

    // Print the output dimensions
    void printOutputDims() const {
        std::cout << "Output Tensor Shape: [";
        for (size_t i = 0; i < outputDims_.size(); ++i) {
            std::cout << outputDims_[i];
            if (i != outputDims_.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // Print the output tensor values (after reduction)
    void printOutputValues() const {
        std::cout << "Output Tensor Values: [";
        for (size_t i = 0; i < outputValues_.size(); ++i) {
            std::cout << outputValues_[i];
            if (i != outputValues_.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

private:
    Tensor& data_;
    std::vector<int64_t>& axes_;
    bool keepDims_;
    std::vector<int64_t> outputDims_;
    std::vector<float> outputValues_;
    std::string opType_; // "sum", "mean", "max", etc.

    // Perform actual reduction based on operation type
    void performReduction() {
        std::vector<float> inputValues = data_.getValues();
        std::vector<int64_t> shape = data_.getShape();
        int64_t rows = shape[0];
        int64_t cols = shape[1];

        if (opType_ == "sum") {
            // Perform sum along axis 0 (reduce by rows)
            if (axes_[0] == 0) {
                for (int64_t col = 0; col < cols; ++col) {
                    float sum = 0.0;
                    for (int64_t row = 0; row < rows; ++row) {
                        sum += inputValues[row * cols + col];
                    }
                    outputValues_.push_back(sum);
                }
            }
            // Perform sum along axis 1 (reduce by columns)
            else if (axes_[0] == 1) {
                for (int64_t row = 0; row < rows; ++row) {
                    float sum = 0.0;
                    for (int64_t col = 0; col < cols; ++col) {
                        sum += inputValues[row * cols + col];
                    }
                    outputValues_.push_back(sum);
                }
            }
        }
        else if (opType_ == "mean") {
            // Perform mean along axis 0 (reduce by rows)
            if (axes_[0] == 0) {
                for (int64_t col = 0; col < cols; ++col) {
                    float sum = 0.0;
                    for (int64_t row = 0; row < rows; ++row) {
                        sum += inputValues[row * cols + col];
                    }
                    outputValues_.push_back(sum / rows);  // Mean
                }
            }
            // Perform mean along axis 1 (reduce by columns)
            else if (axes_[0] == 1) {
                for (int64_t row = 0; row < rows; ++row) {
                    float sum = 0.0;
                    for (int64_t col = 0; col < cols; ++col) {
                        sum += inputValues[row * cols + col];
                    }
                    outputValues_.push_back(sum / cols);  // Mean
                }
            }
        }
        // Other operations (like max, min) can be added similarly
    }
};

int main() {
    // Define a tensor with shape (3, 4) and some values
    std::vector<int64_t> shape = { 3, 4 };
    std::vector<float> values = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
    Tensor data(shape, values);

    // Define axes to reduce along (e.g., reduce along axis 1)
    std::vector<int64_t> axes = { 0 };

    // Set keepDims to true (keep the reduced dimensions)
    bool keepDims = true;

    // Create and compute shape for ReduceSum operation
    ONNXReduceOp reduceSumOp(data, axes, keepDims, "sum");
    reduceSumOp.computeShape();

    // Print the output dimensions and values
    reduceSumOp.printOutputDims();
    reduceSumOp.printOutputValues();

    return 0;
}
