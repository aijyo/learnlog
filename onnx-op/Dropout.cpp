#include <iostream>
#include <vector>
#include <random>
#include <ctime>

// Function to apply dropout to a tensor (2D for simplicity)
void ApplyDropout(std::vector<float>& data, const std::vector<int64_t>& shape, float p,
    std::vector<float>& mask, bool& isMaskRequired) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int64_t numElements = 1;
    for (int64_t dim : shape) {
        numElements *= dim;
    }

    // Generate dropout mask and apply dropout
    isMaskRequired = true;
    mask.resize(numElements);
    for (int64_t i = 0; i < numElements; ++i) {
        float randValue = dis(gen);  // Random value between [0, 1)
        if (randValue < p) {
            mask[i] = 0;  // "Drop" this element
            data[i] = 0;  // Set this element to 0
        }
        else {
            mask[i] = 1;  // Keep this element
            data[i] *= (1.0 / (1.0 - p));  // Scale the value (to maintain expected sum)
        }
    }
}

// Function to print a tensor (2D for simplicity)
void PrintTensor(const std::vector<float>& tensor, const std::vector<int64_t>& shape) {
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            std::cout << tensor[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Example tensor shape [2, 3] (2 rows, 3 columns)
    std::vector<int64_t> shape = { 2, 3 };
    std::vector<float> data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };  // Example data tensor

    float p = 0.5;  // Dropout probability

    std::vector<float> mask;  // Mask to track which elements were dropped
    bool isMaskRequired = false;

    std::cout << "Original Tensor:" << std::endl;
    PrintTensor(data, shape);

    // Apply dropout
    ApplyDropout(data, shape, p, mask, isMaskRequired);

    std::cout << "Tensor after Dropout (p = " << p << "):" << std::endl;
    PrintTensor(data, shape);

    // Output mask if needed
    if (isMaskRequired) {
        std::cout << "Mask Tensor:" << std::endl;
        for (size_t i = 0; i < mask.size(); ++i) {
            std::cout << mask[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
