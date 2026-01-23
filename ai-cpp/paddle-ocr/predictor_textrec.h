//#pragma once
//
//#include <opencv2/opencv.hpp>
//
//#include <optional>
//#include <string>
//#include <unordered_map>
//#include <utility>
//#include <vector>
//
//#include "utils/absl_shim.h"
//
//#include "CTCLabelDecode.h"
//#include "utils/base_predictor.h"
//#include "utils/utility.h"
//
//struct TextRecPredictorResult {
//    std::string input_path = "";
//    cv::Mat input_image;
//    std::string rec_text = "";
//    float rec_score = 0.0f;
//    std::string vis_font = "";
//};
//
//struct TextRecPredictorParams {
//    std::optional<std::string> model_name = std::nullopt;
//    std::optional<std::string> model_dir = std::nullopt;
//    std::optional<std::string> lang = std::nullopt;
//    std::optional<std::string> ocr_version = std::nullopt;
//    std::optional<std::string> vis_font_dir = std::nullopt;
//    std::optional<std::string> device = std::nullopt;
//
//    std::string precision = "fp32";
//    bool enable_mkldnn = true;
//    int mkldnn_cache_capacity = 10;
//    int cpu_threads = 8;
//
//    // NOTE: Rec model often needs fixed input shape, e.g. {3, 48, 320}
//    std::optional<std::vector<int>> input_shape = std::nullopt;
//};
//
//class TextRecPredictor : public BasePredictor {
//public:
//    explicit TextRecPredictor(const TextRecPredictorParams& params);
//
//    // Build graph: preprocess ops + infer + postprocess ops
//    absl::Status Build();
//
//    // Predict from in-memory BGR image (single sample)
//    absl::StatusOr<TextRecPredictorResult> Predict(const cv::Mat& bgr);
//
//    // Predict from image path (single sample)
//    absl::StatusOr<TextRecPredictorResult> PredictPath(const std::string& image_path);
//
//    // Optional: keep your original model/lang validation logic
//    absl::Status CheckRecModelParams();
//
//private:
//    std::unordered_map<std::string, std::unique_ptr<CTCLabelDecode>> post_op_;
//    std::unique_ptr<PaddleInfer> infer_ptr_;
//    TextRecPredictorParams params_;
//};
