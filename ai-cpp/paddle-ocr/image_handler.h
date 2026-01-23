#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "utils/absl_shim.h"
#include "frame_def.h"
// Forward declarations (use your real headers in cpp)
class TextDetection;   // your wrapper: det.Predict(Mat)/PredictPath(...)
class TextRecognition; // your wrapper: rec.Predict(Mat)/PredictPath(...)
class WowTextHandler; // your business processor (based on recognized text)

using TextHander = WowTextHandler;
// ImageHandler config for game screenshots.
struct ImageHandlerConfig {
    // If set, only process this ROI (in original screenshot coordinates).
    std::optional<cv::Rect> roi;

    // Enable/disable detection. If false, will directly recognize ROI/full image.
    bool enable_det = true;

    // Filters
    float det_score_thresh = 0.5f;
    float rec_score_thresh = 0.0f;
    int min_box_size = 8;

    // Crop/rectify
    bool rotate_crop = true;     // warpPerspective using quad
    int crop_padding = 2;        // pixels
    int max_crop_side = 640;     // cap crop size, 0 = no cap

    // Optional preprocessing for game UI crops
    bool to_gray = false;
    bool adaptive_threshold = false;

    // Sorting/merging
    bool sort_items = true;
    int sort_y_tol = 10;
    bool build_merged_text = true;
};

// ImageHandler encapsulates TextDetection + TextRecognition + TextHander.
class ImageHandler {
public:
    // You can pass your already-initialized det/rec/processor instances.
    // If you prefer ImageHandler to create them internally from params, I can adapt later.
    ImageHandler(std::shared_ptr<TextDetection> det,
        std::shared_ptr<TextRecognition> rec,
        std::shared_ptr<TextHander> processor,
        ImageHandlerConfig cfg = {});

    const ImageHandlerConfig& config() const { return cfg_; }
    void set_config(const ImageHandlerConfig& cfg) { cfg_ = cfg; }

    // Main APIs
    absl::StatusOr<OcrFrameResult> ProcessPath(const std::string& image_path);
    absl::StatusOr<OcrFrameResult> ProcessMat(const cv::Mat& bgr);

private:
    absl::StatusOr<OcrFrameResult> ProcessInternal(const cv::Mat& bgr_src);

    // Detection + recognition
    absl::StatusOr<std::vector<OcrInstance>> DetectAndRecognize(const cv::Mat& bgr_work,
        const cv::Rect& work_roi_in_src);

    absl::StatusOr<OcrInstance> RecognizeOne(const cv::Mat& bgr_work,
        const std::vector<cv::Point2f>& quad_work,
        float det_score,
        const cv::Rect& work_roi_in_src) const;

    absl::StatusOr<cv::Mat> CropQuad(const cv::Mat& bgr_work,
        const std::vector<cv::Point2f>& quad_work) const;

    // Processor step (TextRecProcessor logic)
    absl::Status ProcessByTextHander(OcrFrameResult& frame) const;

    // Helpers
    static cv::Rect ClampRect(const cv::Rect& r, const cv::Size& s);
    static cv::Rect QuadBoundingRect(const std::vector<cv::Point2f>& quad);
    static std::vector<cv::Point2f> OffsetQuad(const std::vector<cv::Point2f>& quad, int dx, int dy);
    static void SortInstances(std::vector<OcrInstance>& items, int y_tol);
    static std::string MergeTextTopDown(const std::vector<OcrInstance>& items, int y_tol);

private:
    std::shared_ptr<TextDetection> det_;
    std::shared_ptr<TextRecognition> rec_;
    std::shared_ptr<TextHander> processor_;
    ImageHandlerConfig cfg_;
};
