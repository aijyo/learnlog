#include <iostream>
#include <vector>
#include <random>
#include <cassert>

// Class to implement the RandomNormalLike operation
class RandomNormalLikeOp {
public:
    // Constructor for the RandomNormalLikeOp class.
    // Parameters:
    // - templateShape: the shape of the input tensor to mimic.
    // - mean: the mean value for the normal distribution.
    // - stddev: the standard deviation for the normal distribution.
    // - dtype: the data type of the output tensor (e.g., float, double).
    RandomNormalLikeOp(const std::vector<int64_t>& templateShape, float mean, float stddev, std::string dtype)
        : templateShape_(templateShape), mean_(mean), stddev_(stddev), dtype_(dtype) {
    }

    // This method computes the shape of the output tensor by inheriting from the template shape.
    std::vector<int64_t> computeShape() {
        return templateShape_;
    }

    // This method generates a tensor filled with random values sampled from a normal distribution.
    std::vector<float> generate() {
        std::random_device rd;
        std::mt19937 gen(rd());  // Standard Mersenne Twister random number generator.
        std::normal_distribution<float> dist(mean_, stddev_);  // Normal distribution with the given mean and stddev.

        std::vector<float> result;
        int total_elements = 1;
        for (auto dim : templateShape_) {
            total_elements *= dim;  // Calculate the total number of elements based on the template shape.
        }

        result.reserve(total_elements);  // Reserve space for the tensor elements.
        for (int i = 0; i < total_elements; ++i) {
            result.push_back(dist(gen));  // Generate a random value and add it to the result.
        }
        return result;
    }

    // This method returns the element type of the tensor based on the dtype (data type).
    std::string getRandomNormalLikeElementType() {
        return dtype_;
    }

    // This method performs type inference and returns the expected data type of the result.
    std::vector<std::string> resultTypeInference() {
        return { getRandomNormalLikeElementType() };  // Return the data type for the output tensor.
    }

    // This method prints the generated random tensor (RandomNormalLike).
    void printRandomNormalLikeTensor() {
        auto result = generate();
        std::cout << "Generated Random Normal Like Tensor: ";
        for (size_t i = 0; i < result.size(); ++i) {
            std::cout << result[i] << " ";  // Print each element of the tensor.
        }
        std::cout << std::endl;
    }

private:
    std::vector<int64_t> templateShape_;  // The shape of the input tensor (template).
    float mean_;  // Mean of the normal distribution.
    float stddev_;  // Standard deviation of the normal distribution.
    std::string dtype_;  // Data type of the output tensor (e.g., "float32").
};

// Main function to test the RandomNormalLikeOp class.
int main() {
    std::vector<int64_t> templateShape = { 2, 3 };  // Define a 2x3 matrix as the template tensor shape.
    float mean = 0.0f;  // Mean of the normal distribution.
    float stddev = 1.0f;  // Standard deviation of the normal distribution.
    std::string dtype = "float32";  // Data type of the tensor.

    // Create an instance of RandomNormalLikeOp with the above parameters.
    RandomNormalLikeOp randomOp(templateShape, mean, stddev, dtype);

    // Compute and print the shape of the output tensor (it should match the input shape).
    std::vector<int64_t> computedShape = randomOp.computeShape();
    std::cout << "Computed Shape: ";
    for (int64_t dim : computedShape) {
        std::cout << dim << " ";  // Print each dimension.
    }
    std::cout << std::endl;

    // Perform type inference and print the result type.
    std::vector<std::string> resultTypes = randomOp.resultTypeInference();
    std::cout << "Result Type: " << resultTypes[0] << std::endl;

    // Print the generated random normal tensor.
    randomOp.printRandomNormalLikeTensor();

    return 0;
}
