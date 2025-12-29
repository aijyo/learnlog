#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// --------------------------- ONNX Shape Op ---------------------------
//
// Returns a 1D int64 tensor that contains the shape of the input tensor.
//
// Given:
//   data shape = [d0, d1, ..., d{r-1}], rank = r
//   start (default 0), end (default r)
//
// Spec clipping rule (onnx Shape op):
// - If axis < 0, axis += rank
// - Then clamp to [0, rank]
// This is applied independently to start and end.
// Output:
//   out shape = [end - start]
//   out[i] = data.shape[start + i]
//
// If start > end after normalization, it's an error.
//
// Notes:
// - This op does NOT depend on data values, only on the metadata shape.
// - If rank is unknown, shape inference cannot proceed.
//
struct ShapeOp {
    // Normalize start/end per spec: negative indexing + clamp to [0, rank]
    static int64_t normalizeClampedPerSpec(int64_t axis, int64_t rank) {
        if (axis < 0) axis += rank;
        if (axis < 0) axis = 0;
        if (axis > rank) axis = rank;
        return axis;
    }

    // Verify start/end constraints when rank is known.
    static void verify(int64_t rank, int64_t start, const int64_t* endOpt) {
        int64_t s = normalizeClampedPerSpec(start, rank);
        int64_t e = endOpt ? normalizeClampedPerSpec(*endOpt, rank) : rank;
        if (s > e) {
            throw std::runtime_error("ShapeOp verify failed: start > end after normalization.");
        }
    }

    // Infer output tensor shape (1D) = [end-start].
    static std::vector<int64_t> inferOutputShape(int64_t rank, int64_t start, const int64_t* endOpt) {
        int64_t s = normalizeClampedPerSpec(start, rank);
        int64_t e = endOpt ? normalizeClampedPerSpec(*endOpt, rank) : rank;
        if (s > e) throw std::runtime_error("inferOutputShape: start > end.");
        return { e - s };
    }

    // Compute output values: int64 vector of selected dims.
    // dataShape must be known and rank = dataShape.size().
    static std::vector<int64_t> compute(const std::vector<int64_t>& dataShape,
        int64_t start,
        const int64_t* endOpt) {
        int64_t rank = static_cast<int64_t>(dataShape.size());
        int64_t s = normalizeClampedPerSpec(start, rank);
        int64_t e = endOpt ? normalizeClampedPerSpec(*endOpt, rank) : rank;
        if (s > e) throw std::runtime_error("ShapeOp compute: start > end.");

        std::vector<int64_t> out;
        out.reserve(static_cast<size_t>(e - s));
        for (int64_t i = s; i < e; ++i) {
            out.push_back(dataShape[static_cast<size_t>(i)]);
        }
        return out;
    }
};

// Pretty print helpers
static void PrintVec(const std::vector<int64_t>& v, const std::string& name) {
    std::cout << name << "=[";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i] << (i + 1 == v.size() ? "" : ", ");
    }
    std::cout << "]\n";
}

int main() {
    // Example input tensor shape: NCHW = [1, 3, 224, 224]
    std::vector<int64_t> dataShape = { 1, 3, 224, 224 };
    int64_t rank = static_cast<int64_t>(dataShape.size());

    // Case 1: Shape(data) => start=0, end=rank
    {
        int64_t start = 0;
        ShapeOp::verify(rank, start, nullptr);
        auto outShape = ShapeOp::inferOutputShape(rank, start, nullptr);
        auto outVal = ShapeOp::compute(dataShape, start, nullptr);

        PrintVec(outShape, "out_tensor_shape");
        PrintVec(outVal, "out_values");
    }

    // Case 2: Shape(data, start=1, end=3) => [C, H] = [3, 224]
    {
        int64_t start = 1;
        int64_t end = 3;
        ShapeOp::verify(rank, start, &end);
        auto outShape = ShapeOp::inferOutputShape(rank, start, &end);
        auto outVal = ShapeOp::compute(dataShape, start, &end);

        PrintVec(outShape, "out_tensor_shape(start=1,end=3)");
        PrintVec(outVal, "out_values(start=1,end=3)");
    }

    // Case 3: Out-of-range end => clipped to rank
    // end=999 is equivalent to end=rank per spec => returns full tail
    {
        int64_t start = 2;
        int64_t end = 999;
        ShapeOp::verify(rank, start, &end);
        auto outShape = ShapeOp::inferOutputShape(rank, start, &end);
        auto outVal = ShapeOp::compute(dataShape, start, &end);

        PrintVec(outShape, "out_tensor_shape(start=2,end=999)");
        PrintVec(outVal, "out_values(start=2,end=999)");
    }

    // Case 4: Negative start/end
    // start=-2 => rank-2 => 2, end=-1 => rank-1 => 3 => returns [H]=[224]
    {
        int64_t start = -2;
        int64_t end = -1;
        ShapeOp::verify(rank, start, &end);
        auto outShape = ShapeOp::inferOutputShape(rank, start, &end);
        auto outVal = ShapeOp::compute(dataShape, start, &end);

        PrintVec(outShape, "out_tensor_shape(start=-2,end=-1)");
        PrintVec(outVal, "out_values(start=-2,end=-1)");
    }

    // Case 5: start less than -rank => clipped to 0
    // start=-999 => clipped to 0; end=2 => returns [N,C] = [1,3]
    {
        int64_t start = -999;
        int64_t end = 2;
        ShapeOp::verify(rank, start, &end);
        auto outShape = ShapeOp::inferOutputShape(rank, start, &end);
        auto outVal = ShapeOp::compute(dataShape, start, &end);

        PrintVec(outShape, "out_tensor_shape(start=-999,end=2)");
        PrintVec(outVal, "out_values(start=-999,end=2)");
    }

    // Uncomment to see verify error: start > end after normalization
    // {
    //   int64_t start = 3;
    //   int64_t end = 1;
    //   ShapeOp::verify(rank, start, &end); // throws
    // }

    return 0;
}
