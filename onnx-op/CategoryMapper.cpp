#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

struct CategoryMapperConfig {
    std::vector<std::string> cats_strings;
    std::vector<int64_t> cats_int64s;
    std::string default_string;
    int64_t default_int64 = 0;

    std::unordered_map<std::string, int64_t> s2i;
    std::unordered_map<int64_t, std::string> i2s;

    // Build lookup tables. Last duplicate wins.
    void Build() {
        if (cats_strings.size() != cats_int64s.size()) {
            throw std::invalid_argument("cats_strings and cats_int64s must have the same length");
        }
        s2i.clear();
        i2s.clear();
        s2i.reserve(cats_strings.size());
        i2s.reserve(cats_int64s.size());

        for (size_t i = 0; i < cats_strings.size(); ++i) {
            s2i[cats_strings[i]] = cats_int64s[i];
            i2s[cats_int64s[i]] = cats_strings[i];
        }
    }
};

template <typename T>
struct TensorND {
    std::vector<T> data;
    std::vector<int64_t> shape;
};

// Compute number of elements from shape (row-major assumption not needed here).
static inline int64_t NumElements(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1; // scalar
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dim not supported");
        n *= d;
    }
    return n;
}

// Map string tensor to int64 tensor.
static inline TensorND<int64_t>
MapStringsToInts(const TensorND<std::string>& input, const CategoryMapperConfig& cfg) {
    const int64_t n = NumElements(input.shape);
    if ((int64_t)input.data.size() != n) {
        throw std::invalid_argument("input.data size does not match shape element count");
    }

    TensorND<int64_t> out;
    out.shape = input.shape;
    out.data.resize((size_t)n);

    for (int64_t i = 0; i < n; ++i) {
        const auto& key = input.data[(size_t)i];
        auto it = cfg.s2i.find(key);
        out.data[(size_t)i] = (it == cfg.s2i.end()) ? cfg.default_int64 : it->second;
    }
    return out;
}

// Map int64 tensor to string tensor.
static inline TensorND<std::string>
MapIntsToStrings(const TensorND<int64_t>& input, const CategoryMapperConfig& cfg) {
    const int64_t n = NumElements(input.shape);
    if ((int64_t)input.data.size() != n) {
        throw std::invalid_argument("input.data size does not match shape element count");
    }

    TensorND<std::string> out;
    out.shape = input.shape;
    out.data.resize((size_t)n);

    for (int64_t i = 0; i < n; ++i) {
        const int64_t key = input.data[(size_t)i];
        auto it = cfg.i2s.find(key);
        out.data[(size_t)i] = (it == cfg.i2s.end()) ? cfg.default_string : it->second;
    }
    return out;
}


static void DemoCategoryMapper() {
    CategoryMapperConfig cfg;
    cfg.cats_strings = { "cat", "dog", "bird" };
    cfg.cats_int64s = { 1, 2, 5 };
    cfg.default_int64 = -1;
    cfg.default_string = "UNKNOWN";
    cfg.Build();

    // Example 1: string -> int64
    TensorND<std::string> in_str;
    in_str.shape = { 2, 3 };
    in_str.data = { "cat", "dog", "fish",
                   "bird", "cat", "lion" };

    auto out_i64 = MapStringsToInts(in_str, cfg);

    std::cout << "string -> int64:\n";
    for (size_t i = 0; i < out_i64.data.size(); ++i) {
        std::cout << out_i64.data[i] << (i + 1 == out_i64.data.size() ? "\n" : ", ");
    }

    // Example 2: int64 -> string
    TensorND<int64_t> in_i64;
    in_i64.shape = { 1, 6 };
    in_i64.data = { 2, 5, 7, 1, 2, -3 };

    auto out_str = MapIntsToStrings(in_i64, cfg);

    std::cout << "int64 -> string:\n";
    for (size_t i = 0; i < out_str.data.size(); ++i) {
        std::cout << out_str.data[i] << (i + 1 == out_str.data.size() ? "\n" : ", ");
    }
}

int main() {
    DemoCategoryMapper();
    return 0;
}
