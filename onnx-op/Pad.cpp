#include <iostream>
#include <vector>
#include <cassert>

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

// Pad operation class that supports multiple modes: constant, edge, reflect, symmetric
class PadOp {
public:
    // Constructor to initialize the padding values
    // pads: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    // value: the value to pad with for constant mode
    // mode: padding mode: "constant", "edge", "reflect", "symmetric"
    PadOp(const vector<int>& pads, int value = 0, const string& mode = "constant")
        : pads(pads), value(value), mode(mode) {
        assert(pads.size() % 2 == 0 && "Pads vector must have an even number of elements");
    }

    // Perform padding operation
    vector<vector<int>> pad(const vector<vector<int>>& input) {
        int inputHeight = input.size();
        int inputWidth = input[0].size();

        // Calculate the output dimensions based on pads
        int top = pads[0], bottom = pads[1];
        int left = pads[2], right = pads[3];

        int outputHeight = inputHeight + top + bottom;
        int outputWidth = inputWidth + left + right;

        // Create the output tensor, initialized with the constant value
        vector<vector<int>> output(outputHeight, vector<int>(outputWidth, value));

        // Copy the input tensor into the center of the output tensor
        for (int i = 0; i < inputHeight; ++i) {
            for (int j = 0; j < inputWidth; ++j) {
                output[i + top][j + left] = input[i][j];
            }
        }

        // Apply the selected padding mode (edge, reflect, symmetric)
        if (mode == "edge") {
            applyEdgePadding(output, inputHeight, inputWidth, top, bottom, left, right);
        }
        else if (mode == "reflect") {
            applyReflectPadding(output, inputHeight, inputWidth, top, bottom, left, right);
        }
        else if (mode == "symmetric") {
            applySymmetricPadding(output, inputHeight, inputWidth, top, bottom, left, right);
        }

        return output;
    }

private:
    vector<int> pads;   // The padding values: [x1_begin, x2_begin, ..., x1_end, x2_end]
    int value;          // Padding value for constant mode
    string mode;        // Padding mode: "constant", "edge", "reflect", "symmetric"

    // Apply edge padding mode: pad with the edge values of the input tensor
    void applyEdgePadding(vector<vector<int>>& output, int inputHeight, int inputWidth,
        int top, int bottom, int left, int right) {
        // Top and bottom padding
        for (int i = 0; i < top; ++i) {
            for (int j = 0; j < output[0].size(); ++j) {
                output[i][j] = output[top][j];  // Top padding using first row
            }
        }
        for (int i = inputHeight + top; i < output.size(); ++i) {
            for (int j = 0; j < output[0].size(); ++j) {
                output[i][j] = output[inputHeight + top - 1][j];  // Bottom padding using last row
            }
        }
        // Left and right padding
        for (int i = 0; i < output.size(); ++i) {
            for (int j = 0; j < left; ++j) {
                output[i][j] = output[i][left];  // Left padding using first column
            }
            for (int j = inputWidth + left; j < output[0].size(); ++j) {
                output[i][j] = output[i][inputWidth + left - 1];  // Right padding using last column
            }
        }
    }

    // Apply reflect padding mode: pad with reflected values from the input tensor
    void applyReflectPadding(vector<vector<int>>& output, int inputHeight, int inputWidth,
        int top, int bottom, int left, int right) {
        // Top and bottom padding
        for (int i = 0; i < top; ++i) {
            for (int j = 0; j < output[0].size(); ++j) {
                output[i][j] = output[top - 1 - i][j];  // Reflecting top rows
            }
        }
        for (int i = inputHeight + top; i < output.size(); ++i) {
            for (int j = 0; j < output[0].size(); ++j) {
                output[i][j] = output[inputHeight + top - 1 - (i - (inputHeight + top))][j];  // Reflecting bottom rows
            }
        }
        // Left and right padding
        for (int i = 0; i < output.size(); ++i) {
            for (int j = 0; j < left; ++j) {
                output[i][j] = output[i][left - 1 - j];  // Reflecting left columns
            }
            for (int j = inputWidth + left; j < output[0].size(); ++j) {
                output[i][j] = output[i][inputWidth + left - 1 - (j - (inputWidth + left))];  // Reflecting right columns
            }
        }
    }

    // Apply symmetric padding mode: pad symmetrically from the input tensor
    void applySymmetricPadding(vector<vector<int>>& output, int inputHeight, int inputWidth,
        int top, int bottom, int left, int right) {
        // Top and bottom padding
        for (int i = 0; i < top; ++i) {
            for (int j = 0; j < output[0].size(); ++j) {
                output[i][j] = output[top - 1 - i][j];  // Symmetric top rows
            }
        }
        for (int i = inputHeight + top; i < output.size(); ++i) {
            for (int j = 0; j < output[0].size(); ++j) {
                output[i][j] = output[inputHeight + top - 1 - (i - (inputHeight + top))][j];  // Symmetric bottom rows
            }
        }
        // Left and right padding
        for (int i = 0; i < output.size(); ++i) {
            for (int j = 0; j < left; ++j) {
                output[i][j] = output[i][left - 1 - j];  // Symmetric left columns
            }
            for (int j = inputWidth + left; j < output[0].size(); ++j) {
                output[i][j] = output[i][inputWidth + left - 1 - (j - (inputWidth + left))];  // Symmetric right columns
            }
        }
    }
};

int main() {
    // Input tensor of shape [2, 3]
    vector<vector<int>> input = {
        {1, 2, 3},
        {4, 5, 6}
    };

    // Pad values: [top, bottom, left, right]
    // For example: [1, 2, 1, 2] means top=1, bottom=2, left=1, right=2
    vector<int> pads = { 1, 1, 2, 2 };  // Pad top=1, bottom=1, left=2, right=2

    // Specific padding values for each side: [top pad value, bottom pad value, left pad value, right pad value]
    int value = 0;  // Padding value for constant mode

    // Create PadOp instance with 'constant' mode
    PadOp padOp(pads, value, "constant");

    // Perform the padding operation
    vector<vector<int>> output = padOp.pad(input);

    // Print the padded output tensor
    cout << "Padded Tensor (Constant):" << endl;
    printTensor(output);

    // Change the mode to 'edge' and perform padding again
    padOp = PadOp(pads, value, "edge");
    output = padOp.pad(input);
    cout << "Padded Tensor (Edge):" << endl;
    printTensor(output);

    // Change the mode to 'reflect' and perform padding again
    padOp = PadOp(pads, value, "reflect");
    output = padOp.pad(input);
    cout << "Padded Tensor (Reflect):" << endl;
    printTensor(output);

    // Change the mode to 'symmetric' and perform padding again
    padOp = PadOp(pads, value, "symmetric");
    output = padOp.pad(input);
    cout << "Padded Tensor (Symmetric):" << endl;
    printTensor(output);

    return 0;
}
