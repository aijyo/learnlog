#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>  // Add this header for accumulate

using namespace std;

// A helper function to display the tensor
void printTensor(const vector<vector<int>>& tensor) {
    for (const auto& row : tensor) {
        for (int elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }
}

// Pooling operation class
class PoolingOp {
public:
    // Constructor to initialize the pooling parameters
    // kernel_size: size of the pooling window (e.g., 2, 2 for 2x2)
    // stride: step size for pooling operation (e.g., 2)
    // padding: padding mode ("valid" or "same")
    // pool_type: pooling type ("max" or "average")
    PoolingOp(int kernel_size, int stride, const string& padding = "valid", const string& pool_type = "max")
        : kernel_size(kernel_size), stride(stride), padding(padding), pool_type(pool_type) {
    }

    // Perform pooling operation
    vector<vector<int>> pool(const vector<vector<int>>& input) {
        int inputHeight = input.size();
        int inputWidth = input[0].size();

        // Calculate output dimensions
        int outputHeight, outputWidth;

        if (padding == "valid") {
            outputHeight = (inputHeight - kernel_size) / stride + 1;
            outputWidth = (inputWidth - kernel_size) / stride + 1;
        }
        else if (padding == "same") {
            outputHeight = (inputHeight + stride - 1) / stride;
            outputWidth = (inputWidth + stride - 1) / stride;
        }
        else {
            throw invalid_argument("Unsupported padding mode");
        }

        // Create the output tensor
        vector<vector<int>> output(outputHeight, vector<int>(outputWidth, 0));

        // Apply pooling operation (max or average)
        for (int i = 0; i < outputHeight; ++i) {
            for (int j = 0; j < outputWidth; ++j) {
                int startX = i * stride;
                int startY = j * stride;

                // Adjust window based on padding
                int endX = min(startX + kernel_size, inputHeight);
                int endY = min(startY + kernel_size, inputWidth);

                vector<int> window;

                for (int x = startX; x < endX; ++x) {
                    for (int y = startY; y < endY; ++y) {
                        window.push_back(input[x][y]);
                    }
                }

                if (pool_type == "max") {
                    output[i][j] = *max_element(window.begin(), window.end());
                }
                else if (pool_type == "average") {
                    output[i][j] = accumulate(window.begin(), window.end(), 0) / window.size();
                }
                else {
                    throw invalid_argument("Unsupported pooling type");
                }
            }
        }

        return output;
    }

private:
    int kernel_size;   // Pooling window size (e.g., 2, 2 for 2x2)
    int stride;        // Step size for pooling operation
    string padding;    // Padding mode ("valid" or "same")
    string pool_type;  // Pooling type ("max" or "average")
};

int main() {
    // Example input tensor (e.g., an image with size 4x4)
    vector<vector<int>> input = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    // Pooling parameters
    int kernel_size = 2;
    int stride = 2;
    string padding = "valid";  // Can also be "same"
    string pool_type = "max";  // Can also be "average"

    // Create PoolingOp instance
    PoolingOp poolingOp(kernel_size, stride, padding, pool_type);

    // Perform the pooling operation
    vector<vector<int>> output = poolingOp.pool(input);

    // Print the pooled output tensor
    cout << "Pooled Tensor:" << endl;
    printTensor(output);

    return 0;
}
