#pragma once
// Lightweight TextDetection wrapper.
// Provides path-based and cv::Mat-based APIs similar to simplified TextRecognition.
// This wrapper avoids BaseCVResult and does not require Abseil (uses absl_shim.h).

#include <string>
#include <utility>

#include <opencv2/core.hpp>

#include "utils/absl_shim.h"
#include "predictor.h"  // TextDetPredictor / TextDetPredictorParams / TextDetPredictorResult

#include "frame_def.h"
// Parameters for TextDetection wrapper.
// NOTE: Keep this in sync with TextDetPredictorParams fields you actually use.
struct TextDetectionParams {
    // Required: directory of det model (Paddle Inference model directory).
    std::string model_dir;

    // Optional runtime configuration.
    std::string device = "cpu";   // "cpu" or "gpu"
    std::string precision = "fp32";
    bool enable_mkldnn = true;
    int mkldnn_cache_capacity = 10;
    int cpu_threads = 8;

    // DetResizeForTest params.
    int limit_side_len = 960;
    std::string limit_type = "max"; // "max" or "min"
    int max_side_limit = 4000;

    // DBPostProcess params.
    float thresh = 0.3f;
    float box_thresh = 0.6f;
    float unclip_ratio = 1.5f;

    // Optional: override input shape (e.g., {3, 640, 640}).
    // Leave empty to use model default.
    std::vector<int> input_shape;
};

// TextDetection is a small wrapper around TextDetPredictor.
// It owns a predictor instance, builds it once, then offers single-image prediction.
class TextDetection {
public:
    explicit TextDetection(const TextDetectionParams& params);

    // (Re)build underlying predictor. Usually called in constructor.
    absl::Status Build();

    // Predict on a BGR image (cv::Mat).
    absl::StatusOr<TextDetPredictorResult> Predict(const cv::Mat& bgr);

    // Predict by reading image from disk (OpenCV imread).
    absl::StatusOr<TextDetPredictorResult> PredictPath(const std::string& image_path);

    // Predict from WoW captured BGRA frame (single sample).
    // If you don't need it, you can remove this API.
    absl::StatusOr<TextDetPredictorResult> PredictFrame(const BgraFrame& frame);
    // Access last build/predict error string if you prefer non-Status flow.
    const std::string& LastError() const { return last_error_; }

private:
    static TextDetPredictorParams ToPredictorParams(const TextDetectionParams& from);

private:
    TextDetectionParams params_;
    TextDetPredictor det_;
    bool built_ = false;
    std::string last_error_;
};
