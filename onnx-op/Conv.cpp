#include <cstdint>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <optional>
#include <type_traits>

// -------------------- TensorND --------------------
template <typename T>
struct TensorND {
    std::vector<int64_t> shape; // N, C, spatial...
    std::vector<T> data;        // row-major contiguous

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

// -------------------- Helpers --------------------
static inline std::vector<int64_t> ComputeStridesRowMajor(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[(size_t)i] = strides[(size_t)i + 1] * shape[(size_t)i + 1];
    }
    return strides;
}

static inline int64_t Offset(const std::vector<int64_t>& idx, const std::vector<int64_t>& strides) {
    int64_t off = 0;
    for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
    return off;
}

static inline void Require(bool cond, const char* msg) {
    if (!cond) throw std::invalid_argument(msg);
}

// -------------------- Conv attributes --------------------
struct ConvAttrs {
    int64_t group = 1;
    std::vector<int64_t> strides;   // k
    std::vector<int64_t> dilations; // k
    std::vector<int64_t> pads;      // 2k: [begin..., end...]

    // Fill defaults for k spatial dims if missing.
    void normalize(int64_t k) {
        if (strides.empty()) strides.assign((size_t)k, 1);
        if (dilations.empty()) dilations.assign((size_t)k, 1);
        if (pads.empty()) pads.assign((size_t)(2 * k), 0);

        Require((int64_t)strides.size() == k, "strides size must equal #spatial dims");
        Require((int64_t)dilations.size() == k, "dilations size must equal #spatial dims");
        Require((int64_t)pads.size() == 2 * k, "pads size must be 2 * #spatial dims");

        Require(group >= 1, "group must be >= 1");
        for (int64_t v : strides) Require(v >= 1, "stride must be >= 1");
        for (int64_t v : dilations) Require(v >= 1, "dilation must be >= 1");
        for (int64_t v : pads) Require(v >= 0, "pad must be >= 0");
    }
};

// Compute output spatial size for one dim.
static inline int64_t ConvOutDim(int64_t in, int64_t k, int64_t pad_b, int64_t pad_e, int64_t stride, int64_t dilation) {
    // effective kernel = dilation*(k-1)+1
    const int64_t eff = dilation * (k - 1) + 1;
    // floor((in + pad_b + pad_e - eff)/stride) + 1
    return (in + pad_b + pad_e - eff) / stride + 1;
}

// -------------------- Conv (N-D) --------------------
// Layout: X [N, Cin, S1..Sk], W [Cout, Cin/group, K1..Kk], B [Cout] optional
template <typename T>
TensorND<T> ConvND_NCHW(const TensorND<T>& X,
    const TensorND<T>& W,
    const std::optional<TensorND<T>>& B,
    ConvAttrs attrs) {
    static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
        "T must be numeric");

    Require(X.rank() >= 3, "X rank must be >= 3 (N,C,spatial...)");
    Require(W.rank() == X.rank(), "W rank must equal X rank");
    const int64_t rank = X.rank();
    const int64_t k = rank - 2; // spatial dims count

    attrs.normalize(k);

    const int64_t N = X.shape[0];
    const int64_t Cin = X.shape[1];
    const int64_t Cout = W.shape[0];
    const int64_t CinPerGroup = W.shape[1];

    Require(Cin % attrs.group == 0, "Cin must be divisible by group");
    Require(Cout % attrs.group == 0, "Cout must be divisible by group");
    Require(CinPerGroup * attrs.group == Cin, "W.shape[1] must be Cin/group");

    if (B.has_value()) {
        Require(B->rank() == 1, "Bias must be rank-1 [Cout]");
        Require(B->shape[0] == Cout, "Bias size must equal Cout");
    }

    // Validate kernel dims match
    for (int64_t i = 0; i < k; ++i) {
        Require(W.shape[(size_t)(2 + i)] >= 1, "Kernel dim must be >= 1");
        Require(X.shape[(size_t)(2 + i)] >= 0, "Input spatial dim must be >= 0");
    }

    // Compute output shape
    TensorND<T> Y;
    Y.shape.resize((size_t)rank);
    Y.shape[0] = N;
    Y.shape[1] = Cout;

    for (int64_t i = 0; i < k; ++i) {
        const int64_t in = X.shape[(size_t)(2 + i)];
        const int64_t kd = W.shape[(size_t)(2 + i)];
        const int64_t pb = attrs.pads[(size_t)i];
        const int64_t pe = attrs.pads[(size_t)(k + i)];
        const int64_t st = attrs.strides[(size_t)i];
        const int64_t dl = attrs.dilations[(size_t)i];

        const int64_t out = ConvOutDim(in, kd, pb, pe, st, dl);
        Require(out >= 0, "Computed output dim is negative (check pads/strides/dilations/kernel)");
        Y.shape[(size_t)(2 + i)] = out;
    }

    const int64_t Ynumel = Y.numel();
    Y.data.assign((size_t)Ynumel, (T)0);

    const auto Xstr = ComputeStridesRowMajor(X.shape);
    const auto Wstr = ComputeStridesRowMajor(W.shape);
    const auto Ystr = ComputeStridesRowMajor(Y.shape);

    const int64_t CoutPerGroup = Cout / attrs.group;

    // Iterate over output indices
    std::vector<int64_t> y_idx((size_t)rank, 0);
    std::vector<int64_t> x_idx((size_t)rank, 0);
    std::vector<int64_t> w_idx((size_t)rank, 0);

    // Pre-extract spatial sizes
    std::vector<int64_t> out_spatial((size_t)k);
    std::vector<int64_t> in_spatial((size_t)k);
    std::vector<int64_t> ker_spatial((size_t)k);
    for (int64_t i = 0; i < k; ++i) {
        out_spatial[(size_t)i] = Y.shape[(size_t)(2 + i)];
        in_spatial[(size_t)i] = X.shape[(size_t)(2 + i)];
        ker_spatial[(size_t)i] = W.shape[(size_t)(2 + i)];
    }

    // Multi-index iteration for output spatial dims (naive recursion via counters)
    for (int64_t n = 0; n < N; ++n) {
        y_idx[0] = n;
        x_idx[0] = n;

        for (int64_t oc = 0; oc < Cout; ++oc) {
            y_idx[1] = oc;

            // Determine group id from output channel
            const int64_t g = oc / CoutPerGroup;
            const int64_t ic_base = g * CinPerGroup;

            // Iterate output spatial positions
            std::vector<int64_t> o((size_t)k, 0);
            while (true) {
                // Set y spatial indices
                for (int64_t i = 0; i < k; ++i) y_idx[(size_t)(2 + i)] = o[(size_t)i];

                // Initialize accumulator with bias if present
                T acc = (T)0;
                if (B.has_value()) acc = B->data[(size_t)oc];

                // Sum over input channels in this group
                for (int64_t icg = 0; icg < CinPerGroup; ++icg) {
                    const int64_t ic = ic_base + icg;
                    x_idx[1] = ic;

                    w_idx[0] = oc;
                    w_idx[1] = icg;

                    // Iterate kernel spatial positions
                    std::vector<int64_t> kk_idx((size_t)k, 0);
                    while (true) {
                        bool inside = true;

                        // Compute corresponding input spatial index:
                        // xi = o*stride - pad_begin + k*dilation
                        for (int64_t d = 0; d < k; ++d) {
                            const int64_t pb = attrs.pads[(size_t)d];
                            const int64_t st = attrs.strides[(size_t)d];
                            const int64_t dl = attrs.dilations[(size_t)d];

                            const int64_t xi = o[(size_t)d] * st - pb + kk_idx[(size_t)d] * dl;
                            x_idx[(size_t)(2 + d)] = xi;
                            w_idx[(size_t)(2 + d)] = kk_idx[(size_t)d];

                            if (xi < 0 || xi >= in_spatial[(size_t)d]) {
                                inside = false;
                            }
                        }

                        if (inside) {
                            const int64_t xo = Offset(x_idx, Xstr);
                            const int64_t wo = Offset(w_idx, Wstr);
                            acc = (T)(acc + X.data[(size_t)xo] * W.data[(size_t)wo]);
                        }

                        // Increment kernel counters
                        int64_t carry = (int64_t)k - 1;
                        while (carry >= 0) {
                            kk_idx[(size_t)carry] += 1;
                            if (kk_idx[(size_t)carry] < ker_spatial[(size_t)carry]) break;
                            kk_idx[(size_t)carry] = 0;
                            carry--;
                        }
                        if (carry < 0) break; // done kernel loop
                    }
                }

                // Store output
                const int64_t yo = Offset(y_idx, Ystr);
                Y.data[(size_t)yo] = acc;

                // Increment output spatial counters
                int64_t carry = (int64_t)k - 1;
                while (carry >= 0) {
                    o[(size_t)carry] += 1;
                    if (o[(size_t)carry] < out_spatial[(size_t)carry]) break;
                    o[(size_t)carry] = 0;
                    carry--;
                }
                if (carry < 0) break; // done output spatial loop
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


static void DemoConv2D() {
    TensorND<float> X;
    X.shape = { 1, 1, 3, 3 };
    X.data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    TensorND<float> W;
    W.shape = { 1, 1, 2, 2 };
    W.data = {
        1, 0,
        0, 1
    };

    TensorND<float> B;
    B.shape = { 1 };
    B.data = { 0.0f };

    ConvAttrs attrs;
    attrs.group = 1;
    attrs.strides = { 1, 1 };
    attrs.dilations = { 1, 1 };
    attrs.pads = { 0, 0, 0, 0 }; // [pad_top, pad_left, pad_bottom, pad_right] for 2D

    auto Y = ConvND_NCHW<float>(X, W, B, attrs);

    std::cout << "=== Conv2D Demo ===\n";
    PrintTensorFlat(X, "X");
    PrintTensorFlat(W, "W");
    PrintTensorFlat(Y, "Y");
    // Expected Y (2x2):
    // [1*1 + 5*1]=6, [2 + 6]=8
    // [4 + 8]=12, [5 + 9]=14
}

int main() {
    DemoConv2D();
    return 0;
}
