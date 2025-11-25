#include <iostream>
#include <vector>
#include <cassert>

class Tensor {
public:
    // 构造函数：初始化张量的维度和数据
    Tensor(std::vector<int> shape) : shape_(shape) {
        // 计算张量的元素总数
        int total_elements = 1;
        for (int dim : shape_) {
            total_elements *= dim;
        }

        // 分配底层数据
        data_.resize(total_elements);

        // 计算步幅
        strides_.resize(shape_.size());
        strides_[shape_.size() - 1] = 1;
        for (int i = shape_.size() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    // 新的构造函数，支持传递步幅
    Tensor(std::vector<int> shape, std::vector<int> strides) : shape_(shape), strides_(strides) {
        // 分配底层数据
        int total_elements = 1;
        for (int dim : shape_) {
            total_elements *= dim;
        }
        data_.resize(total_elements);
    }

    // 通过指定的索引来访问数据
    float& operator[](std::vector<int> indices) {
        int flat_index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            flat_index += indices[i] * strides_[i];
        }
        return data_[flat_index];
    }

    // 获取张量的形状
    const std::vector<int>& shape() const {
        return shape_;
    }

    // 获取张量的步幅
    const std::vector<int>& strides() const {
        return strides_;
    }

    // 设置底层数据（用于测试）
    void set_data(const std::vector<float>& new_data) {
        assert(new_data.size() == data_.size());
        data_ = new_data;
    }

    // 打印张量的数据
    void print_data() const {
        for (float val : data_) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // permute_axes 方法：根据给定的新轴顺序来调整步幅和形状
    Tensor permute_axes(const std::vector<int>& new_order) {
        // 创建新的张量
        std::vector<int> new_shape(new_order.size());
        std::vector<int> new_strides(new_order.size());

        // 重新计算形状
        for (size_t i = 0; i < new_order.size(); ++i) {
            new_shape[i] = shape_[new_order[i]];
            new_strides[i] = strides_[new_order[i]];
        }

        // 创建新的张量，并传递新的步幅
        Tensor permuted_tensor(new_shape, new_strides); // 传递新的步幅
        permuted_tensor.set_data(data_);  // 共享底层数据

        return permuted_tensor;
    }

private:
    std::vector<int> shape_;       // 张量的形状
    std::vector<int> strides_;     // 步幅
    std::vector<float> data_;      // 张量的底层数据
};

int main() {
    // 创建一个形状为 [2, 3, 4] 的张量
    Tensor A({ 2, 3, 4 });

    // 设置数据（24个元素，符合A的shape [2, 3, 4]）
    A.set_data({
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        });

    std::cout << "Original tensor A data (shape: [2, 3, 4]): " << std::endl;
    A.print_data();

    // permute_axes 操作，重新排列轴顺序 [0, 2, 1] -> [2, 4, 3]
    // 例如：原始张量 A 的形状为 [2, 3, 4]，现在我们将其轴重新排列为 [0, 2, 1]
    Tensor B = A.permute_axes({ 0, 2, 1 });

    std::cout << "Permuted tensor B data (shape: [2, 4, 3]): " << std::endl;
    B.print_data();

    // 访问 permuted_tensor 中的元素验证
    std::cout << "Accessing B[0, 1, 1]: " << B[{0, 1, 1}] << std::endl; // 应该与 A[0, 1, 1] 对应
    std::cout << "Accessing A[0, 1, 1]: " << A[{0, 1, 1}] << std::endl; // 应该与 A[0, 1, 1] 对应
    std::cout << "Accessing B[1, 0, 1]: " << B[{1, 0, 1}] << std::endl; // 应该与 A[1, 1, 0] 对应
    std::cout << "Accessing A[1, 1, 0]: " << A[{1, 1, 0}] << std::endl; // 应该与 A[1, 1, 0] 对应

    return 0;
}
