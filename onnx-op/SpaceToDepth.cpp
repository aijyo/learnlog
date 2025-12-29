#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

// ------------------------------ Tensor (NCHW) ------------------------------
//
// Minimal 4D tensor container in row-major order.
// For NCHW layout, the linear offset is:
//   (((n * C + c) * H + h) * W + w)
//
template <typename T>
struct Tensor4D {
    int64_t N = 0, C = 0, H = 0, W = 0;
    std::vector<T> data;

    int64_t numel() const { return N * C * H * W; }

    T& at(int64_t n, int64_t c, int64_t h, int64_t w) {
        const int64_t idx = (((n * C + c) * H + h) * W + w);
        return data[idx];
    }
    const T& at(int64_t n, int64_t c, int64_t h, int64_t w) const {
        const int64_t idx = (((n * C + c) * H + h) * W + w);
        return data[idx];
    }
};

// ------------------------------ Verify ------------------------------
//
// Mirrors the onnx-mlir verify intent:
// - input rank is 4 (we encode as Tensor4D)
// - blocksize must be > 0
// - H and W must be divisible by blocksize
//
static inline void verify_space_to_depth(const int64_t H, const int64_t W, int64_t blocksize) {
    if (blocksize <= 0) {
        throw std::runtime_error("SpaceToDepth: blocksize must be strictly positive.");
    }
    if (H % blocksize != 0) {
        throw std::runtime_error("SpaceToDepth: input height must be divisible by blocksize.");
    }
    if (W % blocksize != 0) {
        throw std::runtime_error("SpaceToDepth: input width must be divisible by blocksize.");
    }
}

// ------------------------------ Shape Inference ------------------------------
//
// Equivalent to onnx-mlir shape helper:
// output = [N, C*bs*bs, H/bs, W/bs]
//
static inline void infer_space_to_depth_shape(int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t bs,
    int64_t& oN, int64_t& oC, int64_t& oH, int64_t& oW) {
    verify_space_to_depth(H, W, bs);
    oN = N;
    oC = C * bs * bs;
    oH = H / bs;
    oW = W / bs;
}

// ------------------------------ SpaceToDepth ------------------------------
//
// Implements SpaceToDepth for NCHW.
// Mapping:
//   out[n, c*bs*bs + bh*bs + bw, h2, w2] = in[n, c, h2*bs + bh, w2*bs + bw]
//
template <typename T>
Tensor4D<T> space_to_depth_nchw(const Tensor4D<T>& input, int64_t blocksize) {
    verify_space_to_depth(input.H, input.W, blocksize);

    int64_t oN, oC, oH, oW;
    infer_space_to_depth_shape(input.N, input.C, input.H, input.W,
        blocksize, oN, oC, oH, oW);

    Tensor4D<T> output;
    output.N = oN;
    output.C = oC;
    output.H = oH;
    output.W = oW;
    output.data.resize(output.numel());

    const int64_t bs = blocksize;

    for (int64_t n = 0; n < oN; ++n) {
        for (int64_t c = 0; c < input.C; ++c) {
            for (int64_t h2 = 0; h2 < oH; ++h2) {
                for (int64_t w2 = 0; w2 < oW; ++w2) {
                    for (int64_t bh = 0; bh < bs; ++bh) {
                        for (int64_t bw = 0; bw < bs; ++bw) {
                            const int64_t inH = h2 * bs + bh;
                            const int64_t inW = w2 * bs + bw;

                            const int64_t outC = c * (bs * bs) + (bh * bs + bw);

                            output.at(n, outC, h2, w2) = input.at(n, c, inH, inW);
                        }
                    }
                }
            }
        }
    }

    return output;
}

// ------------------------------ Pretty Print (for demo) ------------------------------
template <typename T>
static void print_tensor_nchw(const Tensor4D<T>& t, const std::string& name) {
    std::cout << name << " shape = [N=" << t.N << ", C=" << t.C << ", H=" << t.H << ", W=" << t.W << "]\n";
    for (int64_t n = 0; n < t.N; ++n) {
        for (int64_t c = 0; c < t.C; ++c) {
            std::cout << "  n=" << n << ", c=" << c << ":\n";
            for (int64_t h = 0; h < t.H; ++h) {
                std::cout << "    ";
                for (int64_t w = 0; w < t.W; ++w) {
                    std::cout << t.at(n, c, h, w) << (w + 1 == t.W ? "" : ", ");
                }
                std::cout << "\n";
            }
        }
    }
}

int main() {
    // Demo input: N=1, C=1, H=4, W=4, filled with 0..15
    Tensor4D<int64_t> x;
    x.N = 1; x.C = 1; x.H = 4; x.W = 4;
    x.data.resize(x.numel());
    std::iota(x.data.begin(), x.data.end(), 0);

    print_tensor_nchw(x, "x");

    // SpaceToDepth with blocksize=2:
    // output shape = [1, 1*2*2=4, 4/2=2, 4/2=2]
    const int64_t bs = 2;
    auto y = space_to_depth_nchw(x, bs);

    print_tensor_nchw(y, "y (SpaceToDepth bs=2)");

    // Intuition check:
    // For input (single channel):
    // [[ 0,  1,  2,  3],
    //  [ 4,  5,  6,  7],
    //  [ 8,  9, 10, 11],
    //  [12, 13, 14, 15]]
    //
    // output has 4 channels each 2x2:
    // outC=0 (bh=0,bw=0): [[0,2],[8,10]]
    // outC=1 (bh=0,bw=1): [[1,3],[9,11]]
    // outC=2 (bh=1,bw=0): [[4,6],[12,14]]
    // outC=3 (bh=1,bw=1): [[5,7],[13,15]]
    return 0;
}
