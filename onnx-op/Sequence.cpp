#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// --------------------------- Type System (Minimal) ---------------------------
//
// This file implements a minimal runtime + type refinement model for ONNX
// Sequence ops, inspired by onnx-mlir shape/type inference rules.
//
// Key ideas:
// - A Sequence holds a list of tensors, but also maintains a summarized
//   "element tensor type" (dtype + rank + shape with dynamic dims).
// - When inserting/constructing sequences, we merge tensor types using a
//   weakening rule similar to onnx-mlir's sequenceAddType().
// - Ops provided: SequenceEmpty, SequenceConstruct, SequenceInsert,
//   SequenceErase, SequenceAt, SequenceLength.
//
enum class DType {
    F32,
    I64
};

static const char* DTypeName(DType t) {
    switch (t) {
    case DType::F32: return "f32";
    case DType::I64: return "i64";
    default: return "unknown";
    }
}

struct TensorType {
    DType dtype = DType::F32;
    bool hasRank = false;              // false => unranked
    std::vector<int64_t> shape;        // valid only if hasRank==true
    // Use -1 to represent dynamic dimension (like ShapedType::kDynamic).

    static TensorType Unranked(DType dt) {
        TensorType t;
        t.dtype = dt;
        t.hasRank = false;
        return t;
    }

    static TensorType Ranked(DType dt, std::vector<int64_t> dims) {
        TensorType t;
        t.dtype = dt;
        t.hasRank = true;
        t.shape = std::move(dims);
        return t;
    }

    int64_t rank() const { return hasRank ? static_cast<int64_t>(shape.size()) : -1; }
};

static std::string TensorTypeToString(const TensorType& ty) {
    std::string s = "tensor<";
    if (!ty.hasRank) {
        s += "*x";
        s += DTypeName(ty.dtype);
        s += ">";
        return s;
    }
    s += "[";
    for (size_t i = 0; i < ty.shape.size(); ++i) {
        if (ty.shape[i] < 0) s += "?";
        else s += std::to_string(ty.shape[i]);
        if (i + 1 != ty.shape.size()) s += ",";
    }
    s += "]x";
    s += DTypeName(ty.dtype);
    s += ">";
    return s;
}

struct Tensor {
    TensorType type;
    std::vector<float> f32;     // demo storage for f32 only
    std::vector<int64_t> i64;   // demo storage for i64 only

    static int64_t Numel(const std::vector<int64_t>& shape) {
        if (shape.empty()) return 1;
        int64_t n = 1;
        for (int64_t d : shape) {
            if (d < 0) throw std::runtime_error("Numel: dynamic dim not allowed in runtime storage.");
            n *= d;
        }
        return n;
    }

    static Tensor MakeF32(std::vector<int64_t> shape, std::vector<float> data) {
        Tensor t;
        t.type = TensorType::Ranked(DType::F32, std::move(shape));
        if (static_cast<int64_t>(data.size()) != Numel(t.type.shape))
            throw std::runtime_error("MakeF32: data size mismatch.");
        t.f32 = std::move(data);
        return t;
    }

    static Tensor MakeI64Scalar(int64_t v) {
        Tensor t;
        t.type = TensorType::Ranked(DType::I64, {}); // rank-0 scalar
        t.i64 = { v };
        return t;
    }
};

struct SeqType {
    TensorType elementType;  // summarized element tensor type
    int64_t length;          // 0..N or -1 for dynamic length

    static SeqType Make(TensorType elem, int64_t len) {
        SeqType s;
        s.elementType = std::move(elem);
        s.length = len;
        return s;
    }
};

static std::string SeqTypeToString(const SeqType& st) {
    std::string s = "seq<";
    s += TensorTypeToString(st.elementType);
    s += ">, length=";
    if (st.length < 0) s += "?";
    else s += std::to_string(st.length);
    return s;
}

struct Sequence {
    SeqType type;
    std::vector<Tensor> elements; // runtime storage
};

// --------------------------- sequenceAddType ---------------------------
//
// Merge two tensor types into a weaker "summary" type, mirroring onnx-mlir:
//
// - DType must match.
// - If either is unranked => result becomes unranked of that dtype.
// - If ranks differ => result becomes unranked.
// - If ranks equal => each dim: keep if equal else make dynamic (-1).
//
static TensorType sequenceAddType(const TensorType& accumulated, const TensorType& additional) {
    if (accumulated.dtype != additional.dtype)
        throw std::runtime_error("sequenceAddType: element dtypes must match.");

    DType dt = accumulated.dtype;

    // Pick the weaker attr: known dim > unknown dim > unranked
    if (!accumulated.hasRank) return accumulated; // already weakest
    if (!additional.hasRank) return additional;   // additional is weaker => return it

    int64_t ra = accumulated.rank();
    int64_t rb = additional.rank();
    if (ra != rb) return TensorType::Unranked(dt);

    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(ra));
    for (int64_t i = 0; i < ra; ++i) {
        int64_t a = accumulated.shape[static_cast<size_t>(i)];
        int64_t b = additional.shape[static_cast<size_t>(i)];
        // If mismatch => dynamic
        out.push_back(a == b ? a : -1);
    }
    return TensorType::Ranked(dt, std::move(out));
}

// --------------------------- Sequence Ops ---------------------------

// SequenceEmpty(dtype=optional, default f32)
static Sequence SequenceEmpty(DType dtype = DType::F32) {
    Sequence s;
    // element type starts as unranked tensor of dtype (common in IR)
    s.type = SeqType::Make(TensorType::Unranked(dtype), /*len=*/0);
    s.elements.clear();
    return s;
}

// SequenceConstruct(tensors...) => seq length = n, element type = merged summary
static Sequence SequenceConstruct(const std::vector<Tensor>& inputs) {
    if (inputs.empty()) throw std::runtime_error("SequenceConstruct: requires at least one input.");

    TensorType elemTy = inputs[0].type;
    for (size_t i = 1; i < inputs.size(); ++i) {
        elemTy = sequenceAddType(elemTy, inputs[i].type);
    }

    Sequence s;
    s.type = SeqType::Make(elemTy, static_cast<int64_t>(inputs.size()));
    s.elements = inputs;
    return s;
}

// SequenceInsert(seq, tensor, position(optional))
// For simplicity, we support position in [0..len] and negative like python.
// If position omitted, insert at end.
static Sequence SequenceInsert(const Sequence& seq, const Tensor& tensor, int64_t* positionOpt = nullptr) {
    // Verify dtype compatibility with sequence element dtype.
    if (seq.type.elementType.dtype != tensor.type.dtype) {
        throw std::runtime_error("SequenceInsert: tensor dtype must match sequence element dtype.");
    }

    Sequence out = seq;

    int64_t len = static_cast<int64_t>(out.elements.size());
    int64_t pos = len; // default end
    if (positionOpt) {
        pos = *positionOpt;
        // Normalize negative
        if (pos < 0) pos += (len + 1);
        if (pos < 0 || pos > len) throw std::runtime_error("SequenceInsert: position out of range.");
    }

    out.elements.insert(out.elements.begin() + pos, tensor);

    // Infer/Refine output seq type
    if (seq.type.length == 0) {
        // When input seq is empty, inherit tensor type
        out.type = SeqType::Make(tensor.type, /*len=*/1);
    }
    else {
        int64_t newLen = (seq.type.length < 0) ? -1 : (seq.type.length + 1);
        TensorType merged = sequenceAddType(seq.type.elementType, tensor.type);
        out.type = SeqType::Make(merged, newLen);
    }

    return out;
}

// SequenceErase(seq, position(optional))
// If position omitted, erase last.
static Sequence SequenceErase(const Sequence& seq, int64_t* positionOpt = nullptr) {
    if (seq.elements.empty()) throw std::runtime_error("SequenceErase: cannot erase from empty sequence.");

    Sequence out = seq;
    int64_t len = static_cast<int64_t>(out.elements.size());
    int64_t pos = len - 1;
    if (positionOpt) {
        pos = *positionOpt;
        if (pos < 0) pos += len;
        if (pos < 0 || pos >= len) throw std::runtime_error("SequenceErase: position out of range.");
    }

    out.elements.erase(out.elements.begin() + pos);

    int64_t oldLenTy = seq.type.length;
    int64_t newLenTy = (oldLenTy < 0) ? -1 : (oldLenTy - 1);

    // Element type stays the same summary type (onnx-mlir does not refine it downward)
    out.type = SeqType::Make(seq.type.elementType, newLenTy);
    return out;
}

// SequenceAt(seq, index) => returns tensor
static Tensor SequenceAt(const Sequence& seq, int64_t index) {
    int64_t len = static_cast<int64_t>(seq.elements.size());
    if (len == 0) throw std::runtime_error("SequenceAt: empty sequence.");
    if (index < 0) index += len;
    if (index < 0 || index >= len) throw std::runtime_error("SequenceAt: index out of range.");

    Tensor t = seq.elements[static_cast<size_t>(index)];

    // Refine output type: if seq element type is ranked and output is unranked, refine.
    // In this runtime model, the stored tensor already has a concrete type.
    // We still show the conceptual refinement:
    if (seq.type.elementType.hasRank && !t.type.hasRank) {
        t.type = seq.type.elementType;
    }

    return t;
}

// SequenceLength(seq) => scalar i64 tensor
static Tensor SequenceLength(const Sequence& seq) {
    return Tensor::MakeI64Scalar(static_cast<int64_t>(seq.elements.size()));
}

// --------------------------- Demo Printing ---------------------------
static void PrintSequenceInfo(const Sequence& s, const std::string& name) {
    std::cout << name << ": " << SeqTypeToString(s.type)
        << ", runtime_len=" << s.elements.size() << "\n";
}

static void PrintTensorInfo(const Tensor& t, const std::string& name) {
    std::cout << name << ": " << TensorTypeToString(t.type) << "\n";
}

// --------------------------- Full Demo ---------------------------
int main() {
    // 1) SequenceEmpty (default f32)
    Sequence s0 = SequenceEmpty();
    PrintSequenceInfo(s0, "s0(empty)");

    // 2) Insert first tensor into empty seq => seq element type inherits tensor type
    Tensor a = Tensor::MakeF32({ 2, 3 }, { 0,1,2,3,4,5 });
    Sequence s1 = SequenceInsert(s0, a);
    PrintSequenceInfo(s1, "s1(insert a)");
    // element type becomes tensor<[2,3]xf32>, length=1

    // 3) Insert another tensor with different shape [2,5]
    Tensor b = Tensor::MakeF32({ 2, 5 }, { 10,11,12,13,14,15,16,17,18,19 });
    Sequence s2 = SequenceInsert(s1, b);
    PrintSequenceInfo(s2, "s2(insert b)");
    // element type merges to tensor<[2,?]xf32>, length=2

    // 4) SequenceAt
    Tensor t0 = SequenceAt(s2, 0);
    Tensor t1 = SequenceAt(s2, 1);
    PrintTensorInfo(t0, "SequenceAt(s2,0)");
    PrintTensorInfo(t1, "SequenceAt(s2,1)");

    // 5) SequenceLength
    Tensor len = SequenceLength(s2);
    PrintTensorInfo(len, "SequenceLength(s2)");
    std::cout << "SequenceLength(s2) value = " << len.i64[0] << "\n";

    // 6) SequenceErase (erase first element)
    int64_t pos0 = 0;
    Sequence s3 = SequenceErase(s2, &pos0);
    PrintSequenceInfo(s3, "s3(erase pos=0)");

    // 7) SequenceConstruct from list of tensors
    Sequence s4 = SequenceConstruct({ a, b });
    PrintSequenceInfo(s4, "s4(construct a,b)");

    return 0;
}
