#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <iomanip>

// --------------------------- Window Ops (ONNX-like) ---------------------------
//
// Output shape:
//   Y is 1D tensor with length = size (N)
//
// Parameters:
// - size (N): window length (must be >= 0)
// - periodic:
//     false -> symmetric window uses M = N-1
//     true  -> periodic  window uses M = N
// - output datatype:
//     In this standalone implementation, the caller selects template T.
//
// Window formulas:
// - Hann:
//     w[n] = 0.5 - 0.5*cos(2*pi*n/M)
// - Hamming:
//     w[n] = 0.54 - 0.46*cos(2*pi*n/M)
// - Blackman:
//     w[n] = 0.42 - 0.5*cos(2*pi*n/M) + 0.08*cos(4*pi*n/M)
//
// Notes:
// - For N == 0: return empty.
// - For N == 1: return [1] (common practical convention), since M would be 0.
//   Different libs may define it slightly differently; this is stable and useful.
//
enum class WindowType {
    Hann,
    Hamming,
    Blackman
};

static double two_pi() {
    return 2.0 * std::acos(-1.0);
}

template <typename T>
static std::vector<T> GenerateWindow(WindowType type, int64_t N, bool periodic) {
    if (N < 0) throw std::invalid_argument("Window: size must be >= 0.");
    if (N == 0) return {};
    if (N == 1) return { (T)1 };

    // M determines symmetry vs periodic behavior.
    // symmetric: M = N-1
    // periodic : M = N
    const double M = periodic ? (double)N : (double)(N - 1);

    // Coefficients for generalized cosine window.
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    switch (type) {
    case WindowType::Hann:
        a0 = 0.5;  a1 = 0.5;  a2 = 0.0;
        break;
    case WindowType::Hamming:
        a0 = 0.54; a1 = 0.46; a2 = 0.0;
        break;
    case WindowType::Blackman:
        a0 = 0.42; a1 = 0.5;  a2 = 0.08;
        break;
    default:
        throw std::invalid_argument("Window: unknown type.");
    }

    std::vector<T> w((size_t)N);
    const double tp = two_pi();
    for (int64_t n = 0; n < N; ++n) {
        const double x = tp * (double)n / M;
        double v = a0 - a1 * std::cos(x);
        if (a2 != 0.0) v += a2 * std::cos(2.0 * x); // cos(4*pi*n/M)
        w[(size_t)n] = (T)v;
    }
    return w;
}

// --------------------------- Demo ---------------------------
static void PrintVec(const std::vector<double>& v, int maxN = 16) {
    int n = (int)v.size();
    int m = std::min(n, maxN);
    std::cout << "[ ";
    for (int i = 0; i < m; ++i) {
        std::cout << std::fixed << std::setprecision(6) << v[(size_t)i] << " ";
    }
    if (n > m) std::cout << "... ";
    std::cout << "]\n";
}

int main() {
    int64_t N = 8;

    auto hann_sym = GenerateWindow<double>(WindowType::Hann, N, /*periodic=*/false);
    auto hann_per = GenerateWindow<double>(WindowType::Hann, N, /*periodic=*/true);

    auto hamm_sym = GenerateWindow<double>(WindowType::Hamming, N, /*periodic=*/false);
    auto black_sym = GenerateWindow<double>(WindowType::Blackman, N, /*periodic=*/false);

    std::cout << "N=" << N << "\n";

    std::cout << "\nHann (symmetric, M=N-1):\n";
    PrintVec(hann_sym);

    std::cout << "Hann (periodic, M=N):\n";
    PrintVec(hann_per);

    std::cout << "\nHamming (symmetric):\n";
    PrintVec(hamm_sym);

    std::cout << "\nBlackman (symmetric):\n";
    PrintVec(black_sym);

    return 0;
}
