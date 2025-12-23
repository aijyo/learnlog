#include <cstdint>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
#include <type_traits>

// -------------------- TensorND --------------------
template <typename T>
struct TensorND {
    std::vector<int64_t> shape; // For DepthToSpace: [N,C,H,W]
    std::vector<T> data;        // Row-major contiguous

    int64_t rank() const { return (int64_t)shape.size(); }

    int64_t numel() const {
        if (shape.empty()) return 1;
        int64_t n = 1;
        for (int64_t d : shape) {
            if (d < 0) throw std::invalid_argument("negative dim");
            n *= d;
        }
        return n;
    }
};

static inline void Require(bool cond, const char* msg) {
    if (!cond) throw std::invalid_argument(msg);
}

static inline std::vector<int64_t> StridesRowMajor(const std::vector<int64_t>& shape) {
    std::vector<int64_t> s(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        s[(size_t)i] = s[(size_t)i + 1] * shape[(size_t)i + 1];
    }
    return s;
}

static inline int64_t Offset4D_NCHW(const std::vector<int64_t>& strides,
    int64_t n, int64_t c, int64_t h, int64_t w) {
    // Assumes strides corresponds to shape [N,C,H,W]
    return n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3];
}

// -------------------- DepthToSpace --------------------
enum class DepthToSpaceMode {
    DCR,
    CRD
};

static inline DepthToSpaceMode ParseMode(const std::string& mode) {
    if (mode == "DCR") return DepthToSpaceMode::DCR;
    if (mode == "CRD") return DepthToSpaceMode::CRD;
    throw std::invalid_argument("DepthToSpace: mode must be \"DCR\" or \"CRD\"");
}

template <typename T>
TensorND<T> DepthToSpace_NCHW(const TensorND<T>& X, int64_t blocksize, const std::string& modeStr) {
    static_assert(std::is_arithmetic<T>::value || std::is_same<T, bool>::value,
        "DepthToSpace: T must be arithmetic or bool");

    Require(X.rank() == 4, "DepthToSpace: input must be 4D [N,C,H,W]");
    Require(blocksize > 0, "DepthToSpace: blocksize must be > 0");

    const auto mode = ParseMode(modeStr);

    const int64_t N = X.shape[0];
    const int64_t C = X.shape[1];
    const int64_t H = X.shape[2];
    const int64_t W = X.shape[3];

    const int64_t bs = blocksize;
    const int64_t bs2 = bs * bs;

    Require(C % bs2 == 0, "DepthToSpace: C must be divisible by blocksize^2");

    const int64_t C_out = C / bs2;
    const int64_t H_out = H * bs;
    const int64_t W_out = W * bs;

    Require((int64_t)X.data.size() == X.numel(), "DepthToSpace: input data size mismatch");

    TensorND<T> Y;
    Y.shape = { N, C_out, H_out, W_out };
    Y.data.resize((size_t)(N * C_out * H_out * W_out));

    const auto Xstr = StridesRowMajor(X.shape);
    const auto Ystr = StridesRowMajor(Y.shape);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C_out; ++c) {
            for (int64_t h = 0; h < H_out; ++h) {
                const int64_t h_in = h / bs;
                const int64_t off_h = h % bs;

                for (int64_t w = 0; w < W_out; ++w) {
                    const int64_t w_in = w / bs;
                    const int64_t off_w = w % bs;

                    int64_t c_in = 0;
                    if (mode == DepthToSpaceMode::DCR) {
                        // c_in = c * (bs*bs) + off_h*bs + off_w
                        c_in = c * bs2 + off_h * bs + off_w;
                    }
                    else {
                        // CRD:
                        // c_in = (off_h*bs + off_w) * C_out + c
                        c_in = (off_h * bs + off_w) * C_out + c;
                    }

                    const int64_t xoff = Offset4D_NCHW(Xstr, n, c_in, h_in, w_in);
                    const int64_t yoff = Offset4D_NCHW(Ystr, n, c, h, w);

                    Y.data[(size_t)yoff] = X.data[(size_t)xoff];
                }
            }
        }
    }

    return Y;
}

// -------------------- Demo printing --------------------
static void PrintShape(const std::vector<int64_t>& s) {
    std::cout << "[";
    for (size_t i = 0; i < s.size(); ++i) {
        std::cout << s[i] << (i + 1 == s.size() ? "" : ", ");
    }
    std::cout << "]";
}

template <typename T>
static void PrintTensorFlat(const TensorND<T>& t, const char* name) {
    std::cout << name << " shape=";
    PrintShape(t.shape);
    std::cout << " data={";
    for (size_t i = 0; i < t.data.size(); ++i) {
        std::cout << t.data[i] << (i + 1 == t.data.size() ? "" : ", ");
    }
    std::cout << "}\n";
}

// Pretty print as NCHW when N=1, C small
template <typename T>
static void PrintNCHW(const TensorND<T>& t, const char* name) {
    Require(t.rank() == 4, "PrintNCHW expects rank=4");
    const int64_t N = t.shape[0], C = t.shape[1], H = t.shape[2], W = t.shape[3];
    std::cout << name << " shape="; PrintShape(t.shape); std::cout << "\n";
    auto s = StridesRowMajor(t.shape);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            std::cout << "  n=" << n << " c=" << c << ":\n";
            for (int64_t h = 0; h < H; ++h) {
                std::cout << "    ";
                for (int64_t w = 0; w < W; ++w) {
                    auto off = Offset4D_NCHW(s, n, c, h, w);
                    std::cout << t.data[(size_t)off] << (w + 1 == W ? "" : " ");
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << "\n";
}

// -------------------- Demo --------------------
static void DemoDepthToSpace() {
    // Example: N=1, C=4, H=1, W=1, blocksize=2
    // Output should be N=1, C_out=1, H_out=2, W_out=2
    TensorND<int> X;
    X.shape = { 1, 4, 1, 1 };
    X.data = { 0, 1, 2, 3 }; // channel values

    std::cout << "=== Input ===\n";
    PrintNCHW(X, "X");

    auto Y_dcr = DepthToSpace_NCHW<int>(X, /*blocksize=*/2, "DCR");
    std::cout << "=== Output (mode=DCR, blocksize=2) ===\n";
    PrintNCHW(Y_dcr, "Y_dcr");

    auto Y_crd = DepthToSpace_NCHW<int>(X, /*blocksize=*/2, "CRD");
    std::cout << "=== Output (mode=CRD, blocksize=2) ===\n";
    PrintNCHW(Y_crd, "Y_crd");

    // Another example: a slightly larger spatial input
    // N=1, C=4, H=2, W=2, blocksize=2 -> output [1,1,4,4]
    TensorND<int> X2;
    X2.shape = { 1, 4, 2, 2 };
    X2.data.resize((size_t)(1 * 4 * 2 * 2));
    for (int i = 0; i < (int)X2.data.size(); ++i) X2.data[(size_t)i] = i;

    std::cout << "=== Input2 ===\n";
    PrintShape(X2.shape); std::cout << " (flat shown)\n";
    PrintTensorFlat(X2, "X2");

    auto Y2 = DepthToSpace_NCHW<int>(X2, 2, "DCR");
    std::cout << "=== Output2 (DCR) ===\n";
    PrintShape(Y2.shape); std::cout << " (flat shown)\n";
    PrintTensorFlat(Y2, "Y2");
}

int main() {
    DemoDepthToSpace();
    return 0;
}
