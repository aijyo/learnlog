#include "text_detection.h"

#include <opencv2/imgcodecs.hpp>

// -----------------------------
// Construction / Build
// -----------------------------

TextDetection::TextDetection(const TextDetectionParams& params)
    : params_(params), det_(ToPredictorParams(params)) {
    // Do not hard-exit here; let caller handle errors.
    auto st = Build();
    if (!st.ok()) {
        last_error_ = st.ToString();
    }
}

absl::Status TextDetection::Build() {
    last_error_.clear();

    if (params_.model_dir.empty()) {
        last_error_ = "TextDetection::Build: model_dir is empty";
        return absl::InvalidArgumentError(last_error_);
    }

    auto st = det_.Build();
    if (!st.ok()) {
        last_error_ = st.ToString();
        built_ = false;
        return st;
    }

    built_ = true;
    return absl::OkStatus();
}

// -----------------------------
// Predict
// -----------------------------

absl::StatusOr<TextDetPredictorResult> TextDetection::Predict(const cv::Mat& bgr) {
    last_error_.clear();

    if (!built_) {
        auto st = Build();
        if (!st.ok()) return st;
    }

    auto r = det_.Predict(bgr);
    if (!r.ok()) {
        last_error_ = r.status().ToString();
    }
    return r;
}

absl::StatusOr<TextDetPredictorResult> TextDetection::PredictPath(const std::string& image_path) {
    last_error_.clear();

    if (image_path.empty()) {
        last_error_ = "TextDetection::PredictPath: image_path is empty";
        return absl::InvalidArgumentError(last_error_);
    }

    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        last_error_ = absl::StrCat("TextDetection::PredictPath: failed to read image: ", image_path);
        return absl::NotFoundError(last_error_);
    }

    auto r = Predict(bgr);
    if (r.ok()) {
        // Preserve input path for convenience.
        r.value().input_path = image_path;
    }
    return r;
}

static cv::Mat MakeBgrFromBgraFrame(const BgraFrame& frame) {
    if (frame.data.empty() || frame.width <= 0 || frame.height <= 0) {
        return cv::Mat();
    }

    int stride = 0;
    if (frame.stride > 0) stride = frame.stride;
    else if (frame.stride > 0) stride = frame.stride;
    else stride = frame.width * 4;

    cv::Mat bgra(frame.height, frame.width, CV_8UC4, (void*)frame.data.data(), stride);
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    return bgr;
}

absl::StatusOr<TextDetPredictorResult> TextDetection::PredictFrame(const BgraFrame& frame) {
    last_error_.clear();

    if (!built_) {
        auto st = Build();
        if (!st.ok()) return st;
    }

    cv::Mat bgr = MakeBgrFromBgraFrame(frame);
    if (bgr.empty()) {
        return absl::InvalidArgumentError("TextRecognition::PredictFrame: invalid BGRA frame");
    }

    auto r = det_.Predict(bgr);
    if (!r.ok()) return r.status();

    // English:
    // Fill extra info if needed.
    auto out = r.value();
    out.input_path = "";  // frame has no path
    out.input_image = bgr;
    return out;
}
// -----------------------------
// Params conversion
// -----------------------------

TextDetPredictorParams TextDetection::ToPredictorParams(const TextDetectionParams& from) {
    TextDetPredictorParams to;

    // Required.
    to.model_dir = from.model_dir;

    // Runtime.
    to.device = from.device;
    to.precision = from.precision;
    to.enable_mkldnn = from.enable_mkldnn;
    to.mkldnn_cache_capacity = from.mkldnn_cache_capacity;
    to.cpu_threads = from.cpu_threads;

    // Resize.
    to.limit_side_len = from.limit_side_len;
    to.limit_type = from.limit_type;
    to.max_side_limit = from.max_side_limit;

    // Postprocess.
    to.thresh = from.thresh;
    to.box_thresh = from.box_thresh;
    to.unclip_ratio = from.unclip_ratio;

    // Optional shape override.
    if (!from.input_shape.empty()) {
        to.input_shape = from.input_shape;
    }

    // Keep batch_size = 1 in simplified wrapper.
    //to.batch_size = 1;

    return to;
}
