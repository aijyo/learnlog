#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <opencv2/opencv.hpp>

#include "frame_def.h"
#include "utils/absl_shim.h"
#include "predictor.h"


// English:
// Params for text recognition wrapper (single image).
struct TextRecognitionParams {
    std::optional<std::string> model_name = std::nullopt;
    std::optional<std::string> model_dir = std::nullopt;
    std::optional<std::string> lang = std::nullopt;
    std::optional<std::string> ocr_version = std::nullopt;
    std::optional<std::string> vis_font_dir = std::nullopt;
    std::optional<std::string> device = std::nullopt;

    std::string precision = "fp32";
    bool enable_mkldnn = true;
    int mkldnn_cache_capacity = 10;
    int cpu_threads = 8;

    // English:
    // Rec model input shape override if needed, e.g. {3, 48, 320}
    std::optional<std::vector<int>> input_shape = std::nullopt;
};

class TextRecognition {
public:
    explicit TextRecognition(const TextRecognitionParams& params = TextRecognitionParams());

    // English:
    // Predict from a BGR image (single sample).
    absl::StatusOr<TextRecPredictorResult> Predict(const cv::Mat& bgr);

    // English:
    // Predict from an image file path (single sample).
    absl::StatusOr<TextRecPredictorResult> PredictPath(const std::string& image_path);

    // English:
    // Predict from WoW captured BGRA frame (single sample).
    // If you don't need it, you can remove this API.
    absl::StatusOr<TextRecPredictorResult> PredictFrame(const BgraFrame& frame);

    void CreateModel();
    absl::Status CheckParams();

    static TextRecPredictorParams ToTextRecognitionModelParams(const TextRecognitionParams& from);

private:
    TextRecognitionParams params_;
    std::unique_ptr<TextRecPredictor> rec_;
};
