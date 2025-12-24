#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ---------------------------- Tensor<T> ----------------------------

// English comments per your preference.

template <typename T>
struct Tensor {
    static_assert(std::is_arithmetic_v<T>, "Tensor<T> requires arithmetic T.");

    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<T> data;

    static std::vector<int64_t> StridesRowMajor(const std::vector<int64_t>& shape) {
        std::vector<int64_t> s(shape.size(), 1);
        for (int i = (int)shape.size() - 2; i >= 0; --i) {
            s[(size_t)i] = s[(size_t)i + 1] * shape[(size_t)i + 1];
        }
        return s;
    }

    static int64_t Numel(const std::vector<int64_t>& shape) {
        int64_t n = 1;
        for (auto d : shape) {
            if (d < 0) throw std::runtime_error("Negative dim.");
            n *= d;
        }
        return n;
    }

    Tensor() = default;

    Tensor(std::vector<int64_t> s, std::vector<T> v)
        : shape(std::move(s)), strides(StridesRowMajor(shape)), data(std::move(v)) {
        if ((int64_t)data.size() != Numel(shape)) throw std::runtime_error("Tensor size mismatch.");
    }

    T& at_flat(int64_t flat) { return data[(size_t)flat]; }
    const T& at_flat(int64_t flat) const { return data[(size_t)flat]; }

    int64_t offset(const std::vector<int64_t>& idx) const {
        if (idx.size() != shape.size()) throw std::runtime_error("Index rank mismatch.");
        int64_t off = 0;
        for (size_t i = 0; i < idx.size(); ++i) off += idx[i] * strides[i];
        return off;
    }
};

// ---------------------------- Einsum Parsing ----------------------------

struct EinsumSpec {
    std::vector<std::string> inSubs; // per input
    std::string outSub;             // output (may be empty if implicit)
    bool hasExplicitOut = false;
};

static std::string StripSpaces(const std::string& s) {
    std::string r;
    for (char c : s) if (!std::isspace((unsigned char)c)) r.push_back(c);
    return r;
}

static std::vector<std::string> Split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == delim) { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

static EinsumSpec ParseEquation(const std::string& equation, int numInputs) {
    std::string eq = StripSpaces(equation);

    size_t arrow = eq.find("->");
    std::string lhs = eq;
    std::string rhs;
    bool explicitOut = false;
    if (arrow != std::string::npos) {
        explicitOut = true;
        lhs = eq.substr(0, arrow);
        rhs = eq.substr(arrow + 2);
    }

    auto inSubs = Split(lhs, ',');
    if ((int)inSubs.size() != numInputs)
        throw std::runtime_error("Equation inputs count mismatch.");

    auto validateSub = [](const std::string& sub) {
        for (size_t i = 0; i < sub.size(); ++i) {
            char c = sub[i];
            if (c == '.') {
                if (i + 2 >= sub.size() || sub[i + 1] != '.' || sub[i + 2] != '.')
                    throw std::runtime_error("Invalid ellipsis.");
                i += 2;
                continue;
            }
            if (!std::isalpha((unsigned char)c))
                throw std::runtime_error("Invalid label char.");
        }
        };
    for (auto& s : inSubs) validateSub(s);
    if (explicitOut) validateSub(rhs);

    EinsumSpec spec;
    spec.inSubs = std::move(inSubs);
    spec.outSub = rhs;
    spec.hasExplicitOut = explicitOut;
    return spec;
}

// Token list: each token is "a".."Z" or "..." or synthetic "E0","E1",...
using Tokens = std::vector<std::string>;

static Tokens TokenizeSubscript(const std::string& sub) {
    Tokens t;
    for (size_t i = 0; i < sub.size(); ++i) {
        char c = sub[i];
        if (c == '.') { // ellipsis
            t.push_back("...");
            i += 2;
        }
        else {
            t.push_back(std::string(1, c));
        }
    }
    return t;
}

static bool IsEllipsisToken(const std::string& tok) { return tok == "..."; }

static int64_t CountNonEllipsisLabels(const Tokens& t) {
    int64_t n = 0;
    for (auto& s : t) if (!IsEllipsisToken(s)) n++;
    return n;
}

struct Expanded {
    std::vector<Tokens> inTok; // expanded inputs, no "..."
    Tokens outTok;             // expanded output, no "..."
    int64_t ellRank = 0;
};

static Expanded ExpandEllipsis(const EinsumSpec& spec,
    const std::vector<std::vector<int64_t>>& inShapes) {
    const int nIn = (int)spec.inSubs.size();
    std::vector<Tokens> rawIn(nIn);
    for (int i = 0; i < nIn; ++i) rawIn[i] = TokenizeSubscript(spec.inSubs[i]);

    int64_t globalEll = 0;
    for (int i = 0; i < nIn; ++i) {
        int64_t nonEll = CountNonEllipsisLabels(rawIn[i]);
        int64_t r = (int64_t)inShapes[i].size();
        int64_t e = r - nonEll;
        if (e < 0) throw std::runtime_error("Subscript has more labels than rank.");
        globalEll = std::max(globalEll, e);
    }

    Tokens ell;
    for (int64_t k = 0; k < globalEll; ++k) ell.push_back("E" + std::to_string(k));

    std::vector<Tokens> expIn(nIn);
    for (int i = 0; i < nIn; ++i) {
        Tokens out;
        int64_t nonEll = CountNonEllipsisLabels(rawIn[i]);
        int64_t r = (int64_t)inShapes[i].size();
        int64_t e = r - nonEll;

        Tokens ellForThis;
        if (e == globalEll) ellForThis = ell;
        else ellForThis.insert(ellForThis.end(), ell.begin() + (globalEll - e), ell.end());

        for (auto& tok : rawIn[i]) {
            if (IsEllipsisToken(tok)) out.insert(out.end(), ellForThis.begin(), ellForThis.end());
            else out.push_back(tok);
        }
        expIn[i] = std::move(out);
    }

    Tokens expOut;
    if (spec.hasExplicitOut) {
        Tokens rawOut = TokenizeSubscript(spec.outSub);
        for (auto& tok : rawOut) {
            if (IsEllipsisToken(tok)) expOut.insert(expOut.end(), ell.begin(), ell.end());
            else expOut.push_back(tok);
        }
    }
    else {
        // Implicit output:
        // 1) ellipsis E0.. in order
        // 2) labels that appear exactly once across all inputs, in order of first appearance
        std::unordered_map<std::string, int> count;
        std::vector<std::string> firstOrder;
        auto addLabel = [&](const std::string& l) {
            if (count.find(l) == count.end()) firstOrder.push_back(l);
            count[l]++;
            };
        for (auto& tks : expIn) for (auto& l : tks) addLabel(l);

        expOut.insert(expOut.end(), ell.begin(), ell.end());
        std::unordered_set<std::string> seen(expOut.begin(), expOut.end());
        for (auto& l : firstOrder) {
            if (count[l] == 1 && !seen.count(l)) {
                expOut.push_back(l);
                seen.insert(l);
            }
        }
    }

    return { std::move(expIn), std::move(expOut), globalEll };
}

// ---------------------------- Shape inference & verification ----------------------------

static int64_t BroadcastDim(int64_t a, int64_t b) {
    if (a == b) return a;
    if (a == 1) return b;
    if (b == 1) return a;
    throw std::runtime_error("Broadcast dimension mismatch.");
}

static std::vector<int64_t> InferOutputShape(const Expanded& ex,
    const std::vector<std::vector<int64_t>>& inShapes) {
    std::unordered_map<std::string, int64_t> labelDim;

    const int nIn = (int)ex.inTok.size();
    for (int i = 0; i < nIn; ++i) {
        const auto& toks = ex.inTok[i];
        const auto& shape = inShapes[i];
        if (toks.size() != shape.size()) throw std::runtime_error("Token-rank mismatch.");

        // Repeated labels inside one input => diagonal, dims must match exactly.
        std::unordered_map<std::string, int64_t> firstPos;
        for (size_t d = 0; d < toks.size(); ++d) {
            const std::string& lab = toks[d];
            int64_t dim = shape[d];

            auto itLocal = firstPos.find(lab);
            if (itLocal != firstPos.end()) {
                if (shape[(size_t)itLocal->second] != dim)
                    throw std::runtime_error("Repeated label requires equal dims (diagonal).");
            }
            else {
                firstPos[lab] = (int64_t)d;
            }

            auto it = labelDim.find(lab);
            if (it == labelDim.end()) labelDim[lab] = dim;
            else labelDim[lab] = BroadcastDim(it->second, dim);
        }
    }

    std::vector<int64_t> outShape;
    outShape.reserve(ex.outTok.size());
    for (auto& lab : ex.outTok) {
        auto it = labelDim.find(lab);
        if (it == labelDim.end()) throw std::runtime_error("Output label not found in inputs.");
        outShape.push_back(it->second);
    }
    return outShape;
}

// ---------------------------- Execution (templated) ----------------------------

template <typename T>
static Tensor<T> Einsum(const std::string& equation, const std::vector<Tensor<T>>& inputs) {
    if (inputs.empty()) throw std::runtime_error("No inputs.");
    const int nIn = (int)inputs.size();

    EinsumSpec spec = ParseEquation(equation, nIn);

    std::vector<std::vector<int64_t>> inShapes;
    inShapes.reserve(nIn);
    for (auto& t : inputs) inShapes.push_back(t.shape);

    Expanded ex = ExpandEllipsis(spec, inShapes);
    std::vector<int64_t> outShape = InferOutputShape(ex, inShapes);

    // Collect output labels set.
    std::unordered_set<std::string> outSet(ex.outTok.begin(), ex.outTok.end());

    // Collect all labels and counts.
    std::vector<std::string> allLabelsOrder;
    std::unordered_map<std::string, int> labelCount;
    for (auto& toks : ex.inTok) {
        for (auto& lab : toks) {
            if (labelCount.find(lab) == labelCount.end()) allLabelsOrder.push_back(lab);
            labelCount[lab]++;
        }
    }

    // Sum labels: in but not out.
    std::vector<std::string> sumLabels;
    for (auto& lab : allLabelsOrder) if (!outSet.count(lab)) sumLabels.push_back(lab);

    // label -> broadcasted dim
    std::unordered_map<std::string, int64_t> labelDim;
    for (int i = 0; i < nIn; ++i) {
        for (size_t d = 0; d < ex.inTok[i].size(); ++d) {
            const std::string& lab = ex.inTok[i][d];
            int64_t dim = inShapes[i][d];
            auto it = labelDim.find(lab);
            if (it == labelDim.end()) labelDim[lab] = dim;
            else labelDim[lab] = BroadcastDim(it->second, dim);
        }
    }

    // Iteration order: output labels, then sum labels.
    std::vector<std::string> iterLabels = ex.outTok;
    iterLabels.insert(iterLabels.end(), sumLabels.begin(), sumLabels.end());

    std::vector<int64_t> iterSizes;
    iterSizes.reserve(iterLabels.size());
    for (auto& lab : iterLabels) iterSizes.push_back(labelDim[lab]);

    // Precompute iter position.
    std::unordered_map<std::string, int64_t> iterPos;
    for (size_t i = 0; i < iterLabels.size(); ++i) iterPos[iterLabels[i]] = (int64_t)i;

    // Prepare output tensor.
    Tensor<T> out;
    out.shape = outShape;
    out.strides = Tensor<T>::StridesRowMajor(outShape);
    out.data.assign((size_t)Tensor<T>::Numel(outShape), T{ 0 });

    // For each input, tokens map dim->label.
    // Fast offset computation.
    auto inputOffsetFast = [&](int inIdx, const std::vector<int64_t>& labIndex) -> int64_t {
        const Tensor<T>& Tn = inputs[inIdx];
        const auto& toks = ex.inTok[inIdx];
        std::vector<int64_t> idx(Tn.shape.size(), 0);
        for (size_t d = 0; d < toks.size(); ++d) {
            const std::string& lab = toks[d];
            int64_t wanted = labIndex[(size_t)iterPos[lab]];
            idx[d] = (Tn.shape[d] == 1) ? 0 : wanted;
        }
        return Tn.offset(idx);
        };

    // Output index vector for writing.
    std::vector<int64_t> outIdx(ex.outTok.size(), 0);

    // Mixed radix counter over iter labels.
    const int64_t iterRank = (int64_t)iterLabels.size();
    std::vector<int64_t> counter((size_t)iterRank, 0);

    auto bump = [&]() -> bool {
        for (int64_t i = iterRank - 1; i >= 0; --i) {
            counter[(size_t)i]++;
            if (counter[(size_t)i] < iterSizes[(size_t)i]) return true;
            counter[(size_t)i] = 0;
        }
        return false;
        };

    // Accumulator type: for integer inputs, accumulate in double to reduce overflow risk.
    using AccT = std::conditional_t<std::is_floating_point_v<T>, double, long double>;

    while (true) {
        AccT prod = (AccT)1;
        for (int i = 0; i < nIn; ++i) {
            int64_t off = inputOffsetFast(i, counter);
            prod *= (AccT)inputs[i].at_flat(off);
        }

        for (size_t i = 0; i < ex.outTok.size(); ++i) {
            outIdx[i] = counter[(size_t)iterPos[ex.outTok[i]]];
        }
        int64_t outOff = out.offset(outIdx);

        // Accumulate.
        AccT cur = (AccT)out.at_flat(outOff);
        cur += prod;
        out.at_flat(outOff) = (T)cur;

        if (!bump()) break;
    }

    return out;
}

// ---------------------------- Demo ----------------------------

template <typename T>
static void PrintTensor(const Tensor<T>& t, const std::string& name, int maxN = 32) {
    std::cout << name << " shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        std::cout << t.shape[i] << (i + 1 < t.shape.size() ? "," : "");
    }
    std::cout << "] data=[";
    int n = (int)t.data.size();
    int k = std::min(n, maxN);
    for (int i = 0; i < k; ++i) {
        std::cout << (long double)t.data[i] << (i + 1 < k ? ", " : "");
    }
    if (n > k) std::cout << ", ...";
    std::cout << "]\n";
}

int main() {
    try {
        // Example 1: float matrix multiplication "ik,kj->ij"
        Tensor<float> A({ 2, 3 }, { 1,2,3, 4,5,6 });
        Tensor<float> B({ 3, 2 }, { 7,8, 9,10, 11,12 });
        auto C = Einsum<float>("ik,kj->ij", { A, B });
        PrintTensor(A, "A");
        PrintTensor(B, "B");
        PrintTensor(C, "C=A@B");

        // Example 2: int32 outer product "i,j->ij"
        Tensor<int32_t> v({ 3 }, { 1,2,3 });
        Tensor<int32_t> w({ 2 }, { 10,20 });
        auto O = Einsum<int32_t>("i,j->ij", { v, w });
        PrintTensor(O, "Outer(int32)");

        // Example 3: transpose "ij->ji"
        auto AT = Einsum<float>("ij->ji", { A });
        PrintTensor(AT, "Transpose(A)");

        // Example 4: trace "ii->" (scalar output is shape=[ ] => numel=1)
        Tensor<float> M({ 3,3 }, { 1,2,3, 4,5,6, 7,8,9 });
        auto tr = Einsum<float>("ii->", { M });
        PrintTensor(tr, "Trace(M)");

        std::cout << "Done.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
