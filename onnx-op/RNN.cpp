#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// --------------------------- Minimal Tensor ---------------------------
//
// A simple row-major tensor container for float.
// shape: int64 dims
// data : flat row-major buffer, size == product(shape)
// ---------------------------------------------------------------------
struct TensorF {
    std::vector<int64_t> shape;
    std::vector<float> data;

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t numel() const {
        if (shape.empty()) return 1;
        int64_t n = 1;
        for (int64_t d : shape) {
            if (d < 0) throw std::runtime_error("Tensor: negative dimension.");
            if (d == 0) return 0;
            if (n > INT64_MAX / d) throw std::runtime_error("Tensor: numel overflow.");
            n *= d;
        }
        return n;
    }
};

static std::string ShapeToString(const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        out += std::to_string(s[i]);
        if (i + 1 < s.size()) out += ", ";
    }
    out += "]";
    return out;
}

static void CheckBuffer(const TensorF& t, const char* name) {
    if (t.numel() != static_cast<int64_t>(t.data.size())) {
        throw std::runtime_error(std::string(name) + ": data size mismatch. shape=" +
            ShapeToString(t.shape) + " numel=" +
            std::to_string(t.numel()) + " data_size=" +
            std::to_string(t.data.size()));
    }
}

static std::vector<int64_t> StridesRowMajor(const std::vector<int64_t>& shape) {
    std::vector<int64_t> st(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        st[i] = st[i + 1] * shape[i + 1];
    }
    return st;
}

static int64_t Offset(const std::vector<int64_t>& idx, const std::vector<int64_t>& st) {
    int64_t o = 0;
    for (size_t i = 0; i < idx.size(); ++i) o += idx[i] * st[i];
    return o;
}

// --------------------------- RNN Attributes ---------------------------
//
// direction: "forward" | "reverse" | "bidirectional"
// layout   : 0 -> X [S,B,I], Y [S,D,B,H]
//            1 -> X [B,S,I], Y [B,S,D,H]
//
// hidden_size: if not set, infer from R/W shapes.
// activations: for RNN default is ["Tanh"] (one per direction in spec).
// clip: optional, applied before activation.
// ---------------------------------------------------------------------
struct RNNAttrs {
    std::string direction = "forward";
    int64_t layout = 0; // 0 or 1
    std::optional<int64_t> hidden_size;
    std::vector<std::string> activations = { "Tanh" };
    std::optional<float> clip;
};

// --------------------------- Activation Helpers ---------------------------
static float ApplyActivation(float x, const std::string& act) {
    if (act == "Tanh" || act == "tanh") {
        return std::tanh(x);
    }
    else if (act == "Relu" || act == "relu") {
        return std::max(0.0f, x);
    }
    else if (act == "Sigmoid" || act == "sigmoid") {
        return 1.0f / (1.0f + std::exp(-x));
    }
    throw std::runtime_error("Unsupported activation: " + act);
}

// --------------------------- ONNX RNN (Standalone) ---------------------------
//
// Implements vanilla RNN:
//   h_t = f( W * x_t + R * h_{t-1} + b )
//
// Inputs:
// - X: [S,B,I] if layout=0 else [B,S,I]
// - W: [D,H,I]
// - R: [D,H,H]
// - B (optional): [D, 2H]   (Wb and Rb concatenated)
// - initial_h (optional): [D,B,H]
// - sequence_lens (optional): [B]  (common behavior: stop updating when t >= L[b])
//
// Outputs (matching the onnx-mlir shape helper snippet):
// - Y: [S,D,B,H] if layout=0 else [B,S,D,H]
// - Y_h: [D,B,H] if layout=0 else [B,D,H]   (onnx-mlir uses layout-dependent order)
//
// Notes:
// - This code focuses on correctness and clarity.
// - sequence_lens padding behavior can vary across runtimes; here we "freeze" h
//   once t exceeds the length (i.e., keep h unchanged).
// ---------------------------------------------------------------------
struct RNNOutputs {
    TensorF Y;   // optional in ONNX; always produced here for demo
    TensorF Y_h; // final hidden
};

static int64_t NumDirectionsFrom(const std::string& dir) {
    if (dir == "forward" || dir == "reverse") return 1;
    if (dir == "bidirectional") return 2;
    throw std::runtime_error("direction must be forward, reverse, or bidirectional.");
}

static int64_t InferHiddenSize(const TensorF& W, const TensorF& R, const RNNAttrs& attrs) {
    if (attrs.hidden_size.has_value()) return attrs.hidden_size.value();

    // W: [D,H,I], R: [D,H,H]
    if (R.shape.size() == 3 && R.shape[2] > 0) return R.shape[2];
    if (R.shape.size() == 3 && R.shape[1] > 0) return R.shape[1]; // gates=1
    if (W.shape.size() == 3 && W.shape[1] > 0) return W.shape[1];
    throw std::runtime_error("Cannot infer hidden_size.");
}

static void ValidateRNNShapes(const TensorF& X, const TensorF& W, const TensorF& R,
    const std::optional<TensorF>& B,
    const std::optional<TensorF>& initial_h,
    const std::optional<std::vector<int64_t>>& sequence_lens,
    const RNNAttrs& attrs) {
    CheckBuffer(X, "X");
    CheckBuffer(W, "W");
    CheckBuffer(R, "R");
    if (B.has_value()) CheckBuffer(*B, "B");
    if (initial_h.has_value()) CheckBuffer(*initial_h, "initial_h");

    if (X.rank() != 3) throw std::runtime_error("RNN: X must be rank 3.");
    if (W.rank() != 3) throw std::runtime_error("RNN: W must be rank 3.");
    if (R.rank() != 3) throw std::runtime_error("RNN: R must be rank 3.");

    const int64_t D = NumDirectionsFrom(attrs.direction);
    if (W.shape[0] != D || R.shape[0] != D)
        throw std::runtime_error("RNN: W/R num_directions mismatch.");

    const int64_t H = InferHiddenSize(W, R, attrs);
    const int64_t I = (attrs.layout == 1) ? X.shape[2] : X.shape[2]; // input_size always last
    if (W.shape[1] != H || W.shape[2] != I)
        throw std::runtime_error("RNN: W shape must be [D,H,I].");
    if (R.shape[1] != H || R.shape[2] != H)
        throw std::runtime_error("RNN: R shape must be [D,H,H].");

    // Layout dependent checks for X
    const int64_t S = (attrs.layout == 1) ? X.shape[1] : X.shape[0];
    const int64_t Bsz = (attrs.layout == 1) ? X.shape[0] : X.shape[1];
    (void)S;

    if (B.has_value()) {
        if (B->rank() != 2) throw std::runtime_error("RNN: B must be rank 2.");
        if (B->shape[0] != D || B->shape[1] != 2 * H)
            throw std::runtime_error("RNN: B shape must be [D, 2H].");
    }

    if (initial_h.has_value()) {
        // ONNX spec uses [D,B,H]. We keep that.
        if (initial_h->rank() != 3) throw std::runtime_error("RNN: initial_h must be rank 3.");
        if (initial_h->shape[0] != D || initial_h->shape[1] != Bsz || initial_h->shape[2] != H)
            throw std::runtime_error("RNN: initial_h shape must be [D,B,H].");
    }

    if (sequence_lens.has_value()) {
        if (static_cast<int64_t>(sequence_lens->size()) != Bsz)
            throw std::runtime_error("RNN: sequence_lens length must equal batch size.");
        if (S >= 0) {
            for (int64_t b = 0; b < Bsz; ++b) {
                int64_t L = (*sequence_lens)[b];
                if (L < 0 || L > S)
                    throw std::runtime_error("RNN: sequence_lens values must be in [0, seq_length].");
            }
        }
    }

    if (!(attrs.layout == 0 || attrs.layout == 1))
        throw std::runtime_error("RNN: layout must be 0 or 1.");
}

static RNNOutputs OnnxRNN(const TensorF& X,
    const TensorF& W,
    const TensorF& R,
    const std::optional<TensorF>& B,
    const std::optional<TensorF>& initial_h,
    const std::optional<std::vector<int64_t>>& sequence_lens,
    const RNNAttrs& attrs) {
    ValidateRNNShapes(X, W, R, B, initial_h, sequence_lens, attrs);

    const int64_t D = NumDirectionsFrom(attrs.direction);
    const int64_t H = InferHiddenSize(W, R, attrs);

    const int64_t S = (attrs.layout == 1) ? X.shape[1] : X.shape[0];
    const int64_t Bsz = (attrs.layout == 1) ? X.shape[0] : X.shape[1];
    const int64_t I = X.shape[2];

    // Select activation per direction (fallback to first)
    auto actOfDir = [&](int64_t d) -> std::string {
        if (attrs.activations.empty()) return "Tanh";
        if (static_cast<int64_t>(attrs.activations.size()) == 1) return attrs.activations[0];
        // Spec allows multiple activations; for simplicity map d -> activations[d]
        if (d < static_cast<int64_t>(attrs.activations.size())) return attrs.activations[d];
        return attrs.activations[0];
        };

    // Prepare outputs with shapes matching the onnx-mlir helper snippet
    RNNOutputs out;

    if (attrs.layout == 0) {
        out.Y.shape = { S, D, Bsz, H };
        out.Y_h.shape = { D, Bsz, H };
    }
    else {
        out.Y.shape = { Bsz, S, D, H };
        out.Y_h.shape = { Bsz, D, H }; // layout-dependent as in onnx-mlir snippet
    }
    out.Y.data.assign(static_cast<size_t>(out.Y.numel()), 0.0f);
    out.Y_h.data.assign(static_cast<size_t>(out.Y_h.numel()), 0.0f);

    // Strides for input X access
    const auto xStrides = StridesRowMajor(X.shape);
    const auto wStrides = StridesRowMajor(W.shape); // [D,H,I]
    const auto rStrides = StridesRowMajor(R.shape); // [D,H,H]

    // Bias pointers: B[d, 0:H] is Wb, B[d, H:2H] is Rb
    auto getBias = [&](int64_t d, int64_t h) -> float {
        if (!B.has_value()) return 0.0f;
        const TensorF& Bb = *B;
        // Sum of Wb and Rb (common ONNX interpretation)
        float wb = Bb.data[static_cast<size_t>(d * Bb.shape[1] + h)];
        float rb = Bb.data[static_cast<size_t>(d * Bb.shape[1] + (H + h))];
        return wb + rb;
        };

    // Hidden state buffer per direction: h_prev[b][h]
    std::vector<float> h_prev(static_cast<size_t>(Bsz * H), 0.0f);
    std::vector<float> h_cur(static_cast<size_t>(Bsz * H), 0.0f);

    auto loadInitialH = [&](int64_t d) {
        std::fill(h_prev.begin(), h_prev.end(), 0.0f);
        if (initial_h.has_value()) {
            const TensorF& H0 = *initial_h; // [D,B,H]
            for (int64_t b = 0; b < Bsz; ++b) {
                for (int64_t h = 0; h < H; ++h) {
                    int64_t off = (d * H0.shape[1] + b) * H0.shape[2] + h;
                    h_prev[static_cast<size_t>(b * H + h)] = H0.data[static_cast<size_t>(off)];
                }
            }
        }
        };

    auto xAt = [&](int64_t s, int64_t b, int64_t i) -> float {
        // X is [S,B,I] when layout=0; [B,S,I] when layout=1
        std::vector<int64_t> idx(3, 0);
        if (attrs.layout == 0) { idx = { s, b, i }; }
        else { idx = { b, s, i }; }
        return X.data[static_cast<size_t>(Offset(idx, xStrides))];
        };

    auto storeY = [&](int64_t s, int64_t d, int64_t b, int64_t h, float v) {
        // Y: layout=0 -> [S,D,B,H], layout=1 -> [B,S,D,H]
        if (attrs.layout == 0) {
            int64_t off = ((s * D + d) * Bsz + b) * H + h;
            out.Y.data[static_cast<size_t>(off)] = v;
        }
        else {
            int64_t off = ((b * S + s) * D + d) * H + h;
            out.Y.data[static_cast<size_t>(off)] = v;
        }
        };

    auto storeYh = [&](int64_t d, int64_t b, int64_t h, float v) {
        // Y_h: layout=0 -> [D,B,H], layout=1 -> [B,D,H] (per onnx-mlir snippet)
        if (attrs.layout == 0) {
            int64_t off = (d * Bsz + b) * H + h;
            out.Y_h.data[static_cast<size_t>(off)] = v;
        }
        else {
            int64_t off = (b * D + d) * H + h;
            out.Y_h.data[static_cast<size_t>(off)] = v;
        }
        };

    auto computeOneStep = [&](int64_t d, int64_t s, int64_t b, int64_t h) -> float {
        // Compute dot(W[d,h,:], x) + dot(R[d,h,:], h_prev[b,:]) + bias
        float sum = 0.0f;

        // W part
        for (int64_t i = 0; i < I; ++i) {
            // W index: [d,h,i]
            int64_t wOff = (d * W.shape[1] + h) * W.shape[2] + i;
            sum += W.data[static_cast<size_t>(wOff)] * xAt(s, b, i);
        }

        // R part
        for (int64_t hh = 0; hh < H; ++hh) {
            // R index: [d,h,hh]
            int64_t rOff = (d * R.shape[1] + h) * R.shape[2] + hh;
            sum += R.data[static_cast<size_t>(rOff)] * h_prev[static_cast<size_t>(b * H + hh)];
        }

        // bias
        sum += getBias(d, h);

        // clip
        if (attrs.clip.has_value()) {
            float c = attrs.clip.value();
            sum = std::max(-c, std::min(c, sum));
        }

        // activation
        return ApplyActivation(sum, actOfDir(d));
        };

    // Direction loop: dir0 forward, dir1 reverse (if bidirectional)
    auto runDirection = [&](int64_t d, bool reverseTime) {
        loadInitialH(d);

        // Iterate time
        for (int64_t ti = 0; ti < S; ++ti) {
            int64_t s = reverseTime ? (S - 1 - ti) : ti;

            // Optionally stop updating per batch by sequence_lens
            for (int64_t b = 0; b < Bsz; ++b) {
                bool active = true;
                if (sequence_lens.has_value()) {
                    int64_t L = (*sequence_lens)[b];
                    // For reverse direction, ONNX defines lens relative to original order.
                    // Here we use a common "freeze" policy:
                    // - forward: active when s < L
                    // - reverse: active when s >= S - L
                    if (!reverseTime) {
                        active = (s < L);
                    }
                    else {
                        active = (s >= (S - L));
                    }
                }

                for (int64_t h = 0; h < H; ++h) {
                    float v;
                    if (active) {
                        v = computeOneStep(d, s, b, h);
                        h_cur[static_cast<size_t>(b * H + h)] = v;
                    }
                    else {
                        // Freeze hidden state (common practical behavior)
                        v = h_prev[static_cast<size_t>(b * H + h)];
                        h_cur[static_cast<size_t>(b * H + h)] = v;
                    }

                    // Store Y at "original time index" position:
                    // ONNX outputs are aligned with input time order even for reverse direction.
                    // So we store using ti (0..S-1) for the output time index in that direction.
                    int64_t out_s = ti;
                    storeY(out_s, d, b, h, v);
                }
            }

            // Move h_cur -> h_prev
            std::swap(h_prev, h_cur);
        }

        // Final hidden state is last computed h_prev
        for (int64_t b = 0; b < Bsz; ++b) {
            for (int64_t h = 0; h < H; ++h) {
                storeYh(d, b, h, h_prev[static_cast<size_t>(b * H + h)]);
            }
        }
        };

    if (attrs.direction == "forward") {
        runDirection(0, /*reverseTime=*/false);
    }
    else if (attrs.direction == "reverse") {
        runDirection(0, /*reverseTime=*/true);
    }
    else {
        // bidirectional
        runDirection(0, /*reverseTime=*/false);
        runDirection(1, /*reverseTime=*/true);
    }

    return out;
}

// --------------------------- Demo ---------------------------
static void PrintY_SDBH(const TensorF& Y, int64_t S, int64_t D, int64_t B, int64_t H) {
    // Expect layout=0: [S,D,B,H]
    std::cout << "Y shape=" << ShapeToString(Y.shape) << "\n";
    for (int64_t s = 0; s < S; ++s) {
        for (int64_t d = 0; d < D; ++d) {
            std::cout << "s=" << s << ", d=" << d << ":\n";
            for (int64_t b = 0; b < B; ++b) {
                std::cout << "  b=" << b << ": ";
                for (int64_t h = 0; h < H; ++h) {
                    int64_t off = ((s * D + d) * B + b) * H + h;
                    std::cout << Y.data[static_cast<size_t>(off)] << (h + 1 < H ? " " : "");
                }
                std::cout << "\n";
            }
        }
    }
}

int main() {
    try {
        // Example: layout=0, forward, S=3, B=2, I=2, H=2
        // X shape [S,B,I]
        TensorF X;
        X.shape = { 3, 2, 2 };
        X.data = {
            // s=0
            1, 2,   // b=0, i=0..1
            3, 4,   // b=1
            // s=1
            5, 6,
            7, 8,
            // s=2
            9, 10,
            11, 12
        };

        // W shape [D=1,H=2,I=2]
        TensorF W;
        W.shape = { 1, 2, 2 };
        // h0: [0.1, 0.2], h1: [0.3, 0.4]
        W.data = { 0.1f, 0.2f, 0.3f, 0.4f };

        // R shape [D=1,H=2,H=2]
        TensorF R;
        R.shape = { 1, 2, 2 };
        // Simple recurrent weights
        // h0 <- 0.5*h0 + 0.0*h1
        // h1 <- 0.0*h0 + 0.5*h1
        R.data = { 0.5f, 0.0f,
                  0.0f, 0.5f };

        // Bias B optional: [D,2H] (Wb + Rb)
        TensorF B;
        B.shape = { 1, 4 };
        // Wb=[0,0], Rb=[0,0] => all zeros
        B.data = { 0,0,0,0 };

        RNNAttrs attrs;
        attrs.direction = "forward";
        attrs.layout = 0;
        attrs.activations = { "Tanh" };
        // attrs.clip = 1.0f; // optional

        auto out = OnnxRNN(X, W, R, B, std::nullopt, std::nullopt, attrs);

        int64_t S = 3, D = 1, Bsz = 2, H = 2;
        PrintY_SDBH(out.Y, S, D, Bsz, H);

        std::cout << "Y_h shape=" << ShapeToString(out.Y_h.shape) << "\n";
        std::cout << "Done.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
