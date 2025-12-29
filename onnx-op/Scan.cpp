#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ------------------------------ TinyTensor ------------------------------
//
// A minimal dense float tensor for demo/runtime execution.
// - Row-major contiguous storage.
// - Supports slicing the first dimension (axis=0) into a view-copy tensor.
// - Supports writing a step tensor into the big scan output at index t.
//
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> data;

  Tensor() = default;

  explicit Tensor(std::vector<int64_t> s)
      : shape(std::move(s)), data(static_cast<size_t>(numelFromShape(shape)), 0.0f) {}

  Tensor(std::vector<int64_t> s, std::vector<float> d)
      : shape(std::move(s)), data(std::move(d)) {
    if (static_cast<int64_t>(data.size()) != numel()) {
      throw std::runtime_error("Tensor data size mismatch with shape.");
    }
  }

  int64_t rank() const { return static_cast<int64_t>(shape.size()); }

  int64_t dim(int64_t i) const {
    if (i < 0 || i >= rank()) throw std::runtime_error("dim index out of range");
    return shape[static_cast<size_t>(i)];
  }

  int64_t numel() const { return numelFromShape(shape); }

  static int64_t numelFromShape(const std::vector<int64_t>& s) {
    if (s.empty()) return 1;
    int64_t n = 1;
    for (int64_t v : s) {
      if (v < 0) throw std::runtime_error("Negative dim is not supported in this demo runtime.");
      n *= v;
    }
    return n;
  }

  // Slice along axis=0: returns a new tensor of shape = shape[1:].
  // Copies data (simple and clear for demo).
  Tensor slice0(int64_t index) const {
    if (rank() < 1) throw std::runtime_error("slice0 requires rank >= 1");
    int64_t T = dim(0);
    if (index < 0 || index >= T) throw std::runtime_error("slice0 index out of range");

    std::vector<int64_t> outShape(shape.begin() + 1, shape.end());
    Tensor out(outShape);

    int64_t stepSize = out.numel();
    const float* src = data.data() + index * stepSize;
    std::copy(src, src + stepSize, out.data.begin());
    return out;
  }

  // Write `step` into this tensor at axis=0 index.
  // This tensor must have shape [T] + step.shape.
  void writeSlice0(int64_t index, const Tensor& step) {
    if (rank() < 1) throw std::runtime_error("writeSlice0 requires rank >= 1");
    if (index < 0 || index >= dim(0)) throw std::runtime_error("writeSlice0 index out of range");

    std::vector<int64_t> expect(shape.begin() + 1, shape.end());
    if (expect != step.shape) {
      throw std::runtime_error("writeSlice0 shape mismatch.");
    }

    int64_t stepSize = step.numel();
    float* dst = data.data() + index * stepSize;
    std::copy(step.data.begin(), step.data.end(), dst);
  }
};

// ------------------------------ ONNX Scan (Runtime) ------------------------------
//
// This is a runtime execution implementation for a common subset of ONNX Scan:
// - scan_input_axes not supported (assume 0)
// - scan_output_axes not supported (assume 0)
// - direction not supported (assume forward)
//
// Semantics:
// Inputs are: [v_initial..., scan_inputs...]
// Body signature (conceptually):
//   (v_t..., x_t...) -> (v_{t+1}..., y_t...)
// Outputs are:
//   [v_final..., scan_outputs...]
// Where scan_outputs are stacked along axis=0 for all time steps.
//
struct ScanRuntime {
  using BodyFn = std::function<
      std::pair<std::vector<Tensor>, std::vector<Tensor>>(
          const std::vector<Tensor>& /*states*/,
          const std::vector<Tensor>& /*scanStepInputs*/)>;

  struct Result {
    std::vector<Tensor> finalStates;
    std::vector<Tensor> scanOutputs;
  };

  static Result Run(
      const std::vector<Tensor>& initialStates,
      const std::vector<Tensor>& scanInputs,
      int64_t numScanInputs,
      const BodyFn& body) {
    // Basic validation
    if (numScanInputs <= 0) throw std::runtime_error("numScanInputs must be > 0");
    if (static_cast<int64_t>(scanInputs.size()) != numScanInputs)
      throw std::runtime_error("scanInputs.size() != numScanInputs");

    if (scanInputs.empty()) throw std::runtime_error("There must be 1 or more scan inputs.");

    // Determine sequence length from first scan input's axis=0
    if (scanInputs[0].rank() < 1) throw std::runtime_error("scan input must have rank >= 1");
    int64_t T = scanInputs[0].dim(0);

    // Ensure all scan inputs have same sequence length (axis=0)
    for (int64_t i = 0; i < numScanInputs; ++i) {
      if (scanInputs[static_cast<size_t>(i)].rank() < 1)
        throw std::runtime_error("scan input must have rank >= 1");
      if (scanInputs[static_cast<size_t>(i)].dim(0) != T)
        throw std::runtime_error("All scan inputs must have same dim(0) sequence length.");
    }

    // Initialize current states
    std::vector<Tensor> states = initialStates;

    // We don't know scan output shapes until we run the first step (like shape inference).
    bool allocated = false;
    std::vector<Tensor> scanOutputs; // final stacked outputs [T, ...]
    int64_t numScanOutputs = -1;

    // Run steps
    for (int64_t t = 0; t < T; ++t) {
      // Build step inputs: slice each scan input at time t => shape drops axis=0
      std::vector<Tensor> stepXs;
      stepXs.reserve(static_cast<size_t>(numScanInputs));
      for (int64_t i = 0; i < numScanInputs; ++i) {
        stepXs.push_back(scanInputs[static_cast<size_t>(i)].slice0(t));
      }

      // Call body: (states, stepXs) -> (newStates, stepYs)
      auto [newStates, stepYs] = body(states, stepXs);

      // Validate state count is stable
      if (t == 0) {
        if (newStates.size() != states.size()) {
          throw std::runtime_error("Body must return the same number of state variables.");
        }
      } else {
        if (newStates.size() != states.size()) {
          throw std::runtime_error("State variable count changed across steps.");
        }
      }

      // Allocate scan outputs on first step based on stepYs shapes
      if (!allocated) {
        numScanOutputs = static_cast<int64_t>(stepYs.size());
        scanOutputs.resize(static_cast<size_t>(numScanOutputs));

        for (int64_t k = 0; k < numScanOutputs; ++k) {
          std::vector<int64_t> outShape;
          outShape.reserve(static_cast<size_t>(stepYs[static_cast<size_t>(k)].rank() + 1));
          outShape.push_back(T); // leading sequence dimension
          outShape.insert(outShape.end(),
                          stepYs[static_cast<size_t>(k)].shape.begin(),
                          stepYs[static_cast<size_t>(k)].shape.end());
          scanOutputs[static_cast<size_t>(k)] = Tensor(outShape);
        }
        allocated = true;
      } else {
        // Check stepYs count stable
        if (static_cast<int64_t>(stepYs.size()) != numScanOutputs) {
          throw std::runtime_error("Scan output count changed across steps.");
        }
        // Check stepYs shape matches allocated output slices
        for (int64_t k = 0; k < numScanOutputs; ++k) {
          std::vector<int64_t> expect(scanOutputs[static_cast<size_t>(k)].shape.begin() + 1,
                                      scanOutputs[static_cast<size_t>(k)].shape.end());
          if (expect != stepYs[static_cast<size_t>(k)].shape) {
            throw std::runtime_error("Scan output step shape changed across steps.");
          }
        }
      }

      // Write stepYs into scanOutputs at index t
      for (int64_t k = 0; k < numScanOutputs; ++k) {
        scanOutputs[static_cast<size_t>(k)].writeSlice0(t, stepYs[static_cast<size_t>(k)]);
      }

      // Update states for next iteration
      states = std::move(newStates);
    }

    // Return final states and stacked scan outputs
    Result r;
    r.finalStates = std::move(states);
    r.scanOutputs = std::move(scanOutputs);
    return r;
  }
};

// ------------------------------ Demo: cumulative sum scan ------------------------------
//
// Example:
// - One state variable: running sum (scalar tensor shape [])
// - One scan input: x with shape [T]  (we represent it as [T] rank=1)
// - Body: sum_{t+1} = sum_t + x_t
//         y_t = sum_{t+1}
// Output:
// - final state: sum_T
// - scan output y: shape [T] (stacked), i.e., [T] + [] => [T]
//
static void PrintTensor1D(const Tensor& t, const std::string& name) {
  std::cout << name << " shape=[";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    std::cout << t.shape[i] << (i + 1 == t.shape.size() ? "" : ",");
  }
  std::cout << "] data=[";
  for (size_t i = 0; i < t.data.size(); ++i) {
    std::cout << t.data[i] << (i + 1 == t.data.size() ? "" : ", ");
  }
  std::cout << "]\n";
}

int main() {
  // Initial state: sum0 = 0 (scalar: shape [])
  Tensor sum0({}, {0.0f});

  // Scan input x: shape [T] = [5]
  Tensor x({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  // Body function
  ScanRuntime::BodyFn body = [](const std::vector<Tensor>& states,
                                const std::vector<Tensor>& stepXs)
      -> std::pair<std::vector<Tensor>, std::vector<Tensor>> {
    // states[0] is scalar sum_t (shape [])
    // stepXs[0] is scalar x_t (because we slice0 from [T] => shape [])
    float sum_t = states[0].data[0];
    float x_t = stepXs[0].data[0];

    float sum_next = sum_t + x_t;

    Tensor newSum({}, {sum_next});      // new state
    Tensor y({}, {sum_next});           // scan output for this step

    return {{newSum}, {y}};
  };

  auto res = ScanRuntime::Run(/*initialStates=*/{sum0},
                              /*scanInputs=*/{x},
                              /*numScanInputs=*/1,
                              /*body=*/body);

  // final state
  PrintTensor1D(res.finalStates[0], "final_sum");

  // scan output stacked: shape [T] because step y is scalar []
  PrintTensor1D(res.scanOutputs[0], "y");

  return 0;
}
