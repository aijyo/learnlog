#pragma once

#include <opencv2/opencv.hpp>

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/absl_shim.h"

#include "CTCLabelDecode.h"
//#include "processors.h"
#include "utils/base_predictor.h"
#include "utils/utility.h"
#include "processors.h"

struct TextRecPredictorResult {
    std::string input_path = "";
    cv::Mat input_image;
    std::string rec_text = "";
    float rec_score = 0.0f;
    std::string vis_font = "";
};

struct TextRecPredictorParams {
    std::optional<std::string> model_name = std::nullopt;
    std::optional<std::string> model_dir = std::nullopt;
    std::optional<std::string> lang = "eng";
    std::optional<std::string> ocr_version = std::nullopt;
    std::optional<std::string> vis_font_dir = std::nullopt;
    std::optional<std::string> device = std::nullopt;

    std::string precision = "fp32";
    bool enable_mkldnn = true;
    int mkldnn_cache_capacity = 10;
    int cpu_threads = 8;

    // NOTE: Rec model often needs fixed input shape, e.g. {3, 48, 320}
    std::optional<std::vector<int>> input_shape = std::nullopt;
};

class TextRecPredictor : public BasePredictor {
public:
    explicit TextRecPredictor(const TextRecPredictorParams& params);

    // Build graph: preprocess ops + infer + postprocess ops
    absl::Status Build();

    // Predict from in-memory BGR image (single sample)
    absl::StatusOr<TextRecPredictorResult> Predict(const cv::Mat& bgr);

    // Predict from image path (single sample)
    absl::StatusOr<TextRecPredictorResult> PredictPath(const std::string& image_path);

    // Optional: keep your original model/lang validation logic
    absl::Status CheckRecModelParams();

private:
    bool initialized_ = false;
    std::unordered_map<std::string, std::unique_ptr<CTCLabelDecode>> post_op_;
    std::unique_ptr<PaddleInfer> infer_ptr_;
    TextRecPredictorParams params_;
};


// ============================
// Text Detection (DB) - simplified single-image API
// ============================

struct TextDetPredictorResult {
    std::string input_path = "";
    cv::Mat input_image;
    std::vector<std::vector<cv::Point2f>> dt_polys = {};
    std::vector<float> dt_scores = {};
};

struct TextDetPredictorParams {
    std::optional<std::string> model_name = std::nullopt;
    std::optional<std::string> model_dir = std::nullopt;
    std::optional<std::string> device = std::nullopt;

    std::string precision = "fp32";
    bool enable_mkldnn = true;
    int mkldnn_cache_capacity = 10;
    int cpu_threads = 8;

    // Optional overrides for det pre/post params.
    std::optional<int> limit_side_len = std::nullopt;
    std::optional<std::string> limit_type = std::nullopt;
    std::optional<int> max_side_limit = std::nullopt;

    std::optional<float> thresh = std::nullopt;
    std::optional<float> box_thresh = std::nullopt;
    std::optional<float> unclip_ratio = std::nullopt;

    // Some det exports provide a fixed input shape in yaml/json.
    std::optional<std::vector<int>> input_shape = std::nullopt;
};

class TextDetPredictor : public BasePredictor {
public:
    explicit TextDetPredictor(const TextDetPredictorParams& params);

    // Build graph: preprocess ops + infer + postprocess ops
    absl::Status Build();

    // Detect from in-memory BGR image (single sample)
    absl::StatusOr<TextDetPredictorResult> Predict(const cv::Mat& bgr);

    // Detect from image path (single sample)
    absl::StatusOr<TextDetPredictorResult> PredictPath(const std::string& image_path);

private:
    bool initialized_ = false;
    std::unordered_map<std::string, std::unique_ptr<DBPostProcess>> post_op_;
    std::unique_ptr<PaddleInfer> infer_ptr_;
    TextDetPredictorParams params_;
};
