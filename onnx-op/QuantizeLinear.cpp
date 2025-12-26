#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>

using namespace std;

// A helper function to display the tensor
void printTensor(const vector<int>& tensor) {
    for (int elem : tensor) {
        cout << elem << " ";
    }
    cout << endl;
}

// Reshape or replicate scale/zero-point to be broadcastable to shape
vector<float> reshapeInput(const vector<float>& value, const vector<int>& shape, int axis = 1, int block_size = 1) {
    if (value.size() == 1) {
        // If value size is 1, we can just return it, as it can be broadcasted
        vector<float> reshaped(shape[axis], value[0]); // Repeat the value to match the axis size
        return reshaped;
    }

    vector<int> dims(shape.size(), 1);
    dims[axis] = value.size();

    // Repeat the scale values to match the shape
    vector<float> reshaped;
    for (size_t i = 0; i < shape[axis]; ++i) {  // Use size_t for indexing
        reshaped.push_back(value[i % value.size()]);
    }

    return reshaped;
}

// QuantizeLinear operation class
class QuantizeLinearOp {
public:
    // Constructor to initialize the scale and zero_point
    QuantizeLinearOp(float scale, int zero_point)
        : scale(scale), zero_point(zero_point) {
    }

    // Perform quantization operation
    vector<int> quantize(const vector<float>& input, const vector<float>& y_scale, const vector<float>& zero_point) {
        vector<int> output;
        vector<float> reshaped_scale = reshapeInput(y_scale, { (int)input.size() }, 0);
        vector<float> reshaped_zero_point = reshapeInput(zero_point, { (int)input.size() }, 0);

        // Quantize using the formula: round((x - zero_point) / scale)
        for (size_t i = 0; i < input.size(); ++i) {  // Use size_t for indexing
            int quantized_value = round((input[i] - reshaped_zero_point[i]) / reshaped_scale[i]);
            output.push_back(quantized_value);
        }

        return output;
    }

    // Return the scale and zero_point as output
    float getScale() const {
        return scale;
    }

    int getZeroPoint() const {
        return zero_point;
    }

private:
    float scale;       // Quantization scale
    int zero_point;    // Quantization zero point
};

int main() {
    // Example input tensor (floating point values)
    vector<float> input = { 0.5, 1.5, 2.5, 3.5, 4.5 };

    // Quantization parameters
    float scale = 0.1f;    // Example scale
    int zero_point = 128;  // Example zero point

    // Scale and zero point for quantization (example)
    vector<float> y_scale = { 0.1f };
    vector<float> zero_point_values = { 128.0f };

    // Create QuantizeLinearOp instance
    QuantizeLinearOp quantizeOp(scale, zero_point);

    // Perform the quantization operation
    vector<int> output = quantizeOp.quantize(input, y_scale, zero_point_values);

    // Print the quantized output tensor
    cout << "Quantized Tensor:" << endl;
    printTensor(output);

    // Print the scale and zero_point
    cout << "Scale: " << quantizeOp.getScale() << endl;
    cout << "Zero Point: " << quantizeOp.getZeroPoint() << endl;

    return 0;
}
