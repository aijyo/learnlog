#include <cstdint>
#include <vector>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <cmath>

static inline int64_t NumElements(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 1; // scalar
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dim not supported in this runtime impl");
        n *= d;
    }
    return n;
}

template <typename T>
static inline T Clamp01(T p) {
    if (p < (T)0) return (T)0;
    if (p > (T)1) return (T)1;
    return p;
}

template <typename InT, typename OutT>
void Bernoulli(const InT* input_prob,
               OutT* output,
               const std::vector<int64_t>& shape,
               bool clamp_prob = true,
               bool nan_to_zero = true,
               // If seed is not provided, use random_device.
               std::optional<uint64_t> seed = std::nullopt) {
    static_assert(std::is_floating_point<InT>::value,
                  "Bernoulli input probability type should be floating point");
    static_assert(std::is_arithmetic<OutT>::value || std::is_same<OutT, bool>::value,
                  "Bernoulli output type must be arithmetic or bool");

    const int64_t n = NumElements(shape);
    if (n < 0) throw std::invalid_argument("invalid element count");

    // RNG setup
    std::mt19937_64 rng;
    if (seed.has_value()) {
        rng.seed(*seed);
    } else {
        std::random_device rd;
        // Seed with multiple entropy values to reduce correlation
        std::seed_seq seq{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
        rng.seed(seq);
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0); // [0,1)

    for (int64_t i = 0; i < n; ++i) {
        InT p = input_prob[i];

        // Handle NaN
        if (std::isnan((double)p)) {
            if (nan_to_zero) {
                output[i] = (OutT)0;
                continue;
            } else {
                throw std::runtime_error("NaN probability encountered");
            }
        }

        if (clamp_prob) {
            p = Clamp01(p);
        } else {
            // Optionally, you could enforce strict range check here.
            // if (p < 0 || p > 1) throw std::runtime_error("probability out of [0,1]");
        }

        const double u = dist(rng);
        const bool one = (u < (double)p);

        if constexpr (std::is_same<OutT, bool>::value) {
            output[i] = one;
        } else {
            output[i] = one ? (OutT)1 : (OutT)0;
        }
    }
}

#include <iostream>

int main() {
    std::vector<int64_t> shape = {2, 3};
    float probs[] = {0.1f, 0.5f, 0.9f,
                     1.2f, -0.2f, 0.7f};

    std::vector<int64_t> out(6);

    // Deterministic run with seed
    Bernoulli<float, int64_t>(probs, out.data(), shape,
                              /*clamp_prob=*/true,
                              /*nan_to_zero=*/true,
                              /*seed=*/12345ULL);

    for (auto v : out) std::cout << v << " ";
    std::cout << "\n";
    return 0;
}
