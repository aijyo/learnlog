#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// --------------------------- Tiny Tensor ---------------------------
//
// Minimal float tensor to demonstrate ShapeTransform.
// Row-major contiguous storage, static shape only.
//
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;

    Tensor() = default;

    Tensor(std::vector<int64_t> s, std::vector<float> d)
        : shape(std::move(s)), data(std::move(d)) {
        if (numel() != static_cast<int64_t>(data.size()))
            throw std::runtime_error("Tensor: data size mismatch.");
    }

    int64_t rank() const { return static_cast<int64_t>(shape.size()); }

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) {
            if (d <= 0) throw std::runtime_error("Tensor: only positive static dims supported.");
            n *= d;
        }
        return n;
    }
};

// --------------------------- Affine Expr AST ---------------------------
//
// A minimal affine expression model:
// - dim(i)
// - const(c)
// - add(a,b)
// - mul_const(c, a)
//
// This is enough to represent common index_map patterns used in shape transforms.
//
struct AffineExpr {
    enum class Kind { Dim, Const, Add, MulConst };

    Kind kind;
    int64_t value; // for Dim: dim index, for Const: constant, for MulConst: multiplier
    std::shared_ptr<AffineExpr> lhs;
    std::shared_ptr<AffineExpr> rhs;

    static std::shared_ptr<AffineExpr> Dim(int64_t i) {
        auto e = std::make_shared<AffineExpr>();
        e->kind = Kind::Dim;
        e->value = i;
        return e;
    }

    static std::shared_ptr<AffineExpr> Const(int64_t c) {
        auto e = std::make_shared<AffineExpr>();
        e->kind = Kind::Const;
        e->value = c;
        return e;
    }

    static std::shared_ptr<AffineExpr> Add(std::shared_ptr<AffineExpr> a,
        std::shared_ptr<AffineExpr> b) {
        auto e = std::make_shared<AffineExpr>();
        e->kind = Kind::Add;
        e->lhs = std::move(a);
        e->rhs = std::move(b);
        return e;
    }

    static std::shared_ptr<AffineExpr> MulConst(int64_t c, std::shared_ptr<AffineExpr> a) {
        auto e = std::make_shared<AffineExpr>();
        e->kind = Kind::MulConst;
        e->value = c;
        e->lhs = std::move(a);
        return e;
    }
};

struct AffineMap {
    // Number of input dims (no symbols supported, matching onnx-mlir verify).
    int64_t numDims = 0;
    // Each result is an affine expr computing one output index dimension.
    std::vector<std::shared_ptr<AffineExpr>> results;

    int64_t getNumResults() const { return static_cast<int64_t>(results.size()); }
};

// --------------------------- ShapeTransform ---------------------------
//
// This implementation follows the spirit of onnx-mlir:
// - No symbols.
// - Input must be static shape.
// - Infer output shape as the upper bound of each affine result expr.
// - Ensure numel(input) == numel(output) for a valid reinterpretation.
//
// Note: onnx-mlir borrows MLIR memref normalization to get the upper bound.
// Here we compute upper bounds analytically for a restricted affine expr set.
//
class ShapeTransformOp {
public:
    // Verify constraints similar to onnx-mlir:
    // - No symbols => our AffineMap has none by design.
    // - Input must be static => Tensor always static here.
    // - Optionally check provided output shape numel matches input.
    static void Verify(const Tensor& input,
        const AffineMap& map,
        const std::vector<int64_t>* outputShapeOpt = nullptr) {
        if (map.numDims != input.rank())
            throw std::runtime_error("ShapeTransform: affine_map numDims must match input rank.");

        // If output shape is provided, check numel equality (onnx-mlir verify does this).
        if (outputShapeOpt) {
            int64_t inN = input.numel();
            int64_t outN = numelOfShape(*outputShapeOpt);
            if (inN != outN)
                throw std::runtime_error("ShapeTransform: input/output numel mismatch.");
        }
    }

    // Infer output shape: compute upper bound for each result expr.
    static std::vector<int64_t> InferOutputShape(const Tensor& input, const AffineMap& map) {
        Verify(input, map, nullptr);

        std::vector<int64_t> out;
        out.reserve(static_cast<size_t>(map.getNumResults()));
        for (const auto& expr : map.results) {
            int64_t maxVal = maxValueOverInputDomain(expr, input.shape);
            // Dimension size is max index + 1 (indices are 0-based).
            out.push_back(maxVal + 1);
        }

        // Enforce same number of elements for a valid shape transform.
        // This matches the typical intent: reshape-like reinterpretation of the same buffer.
        int64_t inN = input.numel();
        int64_t outN = numelOfShape(out);
        if (inN != outN) {
            throw std::runtime_error("ShapeTransform: inferred output numel != input numel.");
        }
        return out;
    }

    // Execute: since ShapeTransform is about layout/shape reinterpretation,
    // a simple runtime model is: output shares the same underlying buffer
    // but with a different shape. Here we return a new Tensor object that
    // copies data for simplicity of demo, but conceptually it's a view.
    static Tensor Run(const Tensor& input, const AffineMap& map) {
        auto outShape = InferOutputShape(input, map);
        // In a real engine, you'd return a view without copying.
        Tensor out(outShape, input.data);
        return out;
    }

private:
    static int64_t numelOfShape(const std::vector<int64_t>& s) {
        int64_t n = 1;
        for (int64_t d : s) {
            if (d <= 0) throw std::runtime_error("numelOfShape: invalid dim.");
            n *= d;
        }
        return n;
    }

    // Compute max value of expr over input index domain:
    //   0 <= di <= shape[i]-1
    // For this restricted affine expr set, we require:
    // - Multipliers are non-negative to make max computation straightforward.
    static int64_t maxValueOverInputDomain(const std::shared_ptr<AffineExpr>& e,
        const std::vector<int64_t>& inputShape) {
        using K = AffineExpr::Kind;
        switch (e->kind) {
        case K::Const:
            return e->value;

        case K::Dim: {
            int64_t i = e->value;
            if (i < 0 || i >= static_cast<int64_t>(inputShape.size()))
                throw std::runtime_error("AffineExpr dim out of range.");
            return inputShape[static_cast<size_t>(i)] - 1;
        }

        case K::Add: {
            int64_t a = maxValueOverInputDomain(e->lhs, inputShape);
            int64_t b = maxValueOverInputDomain(e->rhs, inputShape);
            return a + b;
        }

        case K::MulConst: {
            int64_t c = e->value;
            if (c < 0) throw std::runtime_error("MulConst: negative multiplier not supported.");
            int64_t a = maxValueOverInputDomain(e->lhs, inputShape);
            return c * a;
        }

        default:
            throw std::runtime_error("Unknown affine expr kind.");
        }
    }
};

// --------------------------- Demo Helpers ---------------------------
static void PrintShape(const std::vector<int64_t>& s, const std::string& name) {
    std::cout << name << "=[";
    for (size_t i = 0; i < s.size(); ++i) {
        std::cout << s[i] << (i + 1 == s.size() ? "" : ", ");
    }
    std::cout << "]\n";
}

int main() {
    // Example input: shape [2,3], data 0..5
    Tensor input({ 2, 3 }, { 0,1,2,3,4,5 });

    // ---------------- Example 1: Transpose-like map (d0,d1)->(d1,d0) ----------------
    {
        AffineMap map;
        map.numDims = 2;
        map.results = {
          AffineExpr::Dim(1), // out0 = d1
          AffineExpr::Dim(0)  // out1 = d0
        };

        auto outShape = ShapeTransformOp::InferOutputShape(input, map);
        Tensor out = ShapeTransformOp::Run(input, map);

        std::cout << "Example 1: transpose-like\n";
        PrintShape(input.shape, "input.shape");
        PrintShape(outShape, "inferred out.shape");
        std::cout << "out.numel = " << out.numel() << " (data copied for demo)\n\n";
    }

    // ---------------- Example 2: Flatten 2D -> 1D using out0 = d0*W + d1 ----------------
    // input shape [H,W] = [2,3] => out shape should be [6]
    {
        int64_t W = input.shape[1];

        AffineMap map;
        map.numDims = 2;
        // out0 = d0*(W) + d1
        map.results = {
          AffineExpr::Add(
            AffineExpr::MulConst(W, AffineExpr::Dim(0)),
            AffineExpr::Dim(1))
        };

        auto outShape = ShapeTransformOp::InferOutputShape(input, map);
        Tensor out = ShapeTransformOp::Run(input, map);

        std::cout << "Example 2: flatten-like\n";
        PrintShape(input.shape, "input.shape");
        PrintShape(outShape, "inferred out.shape");
        std::cout << "out.numel = " << out.numel() << " (data copied for demo)\n\n";
    }

    // ---------------- Example 3: Invalid map causing numel mismatch ----------------
    // (d0,d1)->(d0) would infer out shape [2], numel mismatch vs 6 => error
    {
        try {
            AffineMap bad;
            bad.numDims = 2;
            bad.results = { AffineExpr::Dim(0) };

            auto outShape = ShapeTransformOp::InferOutputShape(input, bad);
            (void)outShape;
        }
        catch (const std::exception& ex) {
            std::cout << "Example 3: expected failure: " << ex.what() << "\n";
        }
    }

    return 0;
}
