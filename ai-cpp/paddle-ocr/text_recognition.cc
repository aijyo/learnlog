#include "text_recognition.h"

#include <cstdarg>
#include <cstdio>

namespace {

    // English:
    // Minimal logger shim to remove ilogger.h dependency.
    inline void LogPrintf(const char* level, const char* fmt, ...) {
        std::fprintf(stderr, "[%s] ", (level ? level : "I"));
        va_list args;
        va_start(args, fmt);
        std::vfprintf(stderr, fmt, args);
        va_end(args);
        std::fprintf(stderr, "\n");
    }

}  // namespace

#ifndef INFO
#define INFO(...)  LogPrintf("I", __VA_ARGS__)
#endif
#ifndef INFOW
#define INFOW(...) LogPrintf("W", __VA_ARGS__)
#endif
#ifndef INFOE
#define INFOE(...) LogPrintf("E", __VA_ARGS__)
#endif

TextRecognition::TextRecognition(const TextRecognitionParams& params)
    : params_(params) {
    auto st = CheckParams();
    if (!st.ok()) {
        INFOE("Init TextRecognition failed: %s", st.ToString().c_str());
        std::exit(-1);
    }
    CreateModel();
}

absl::Status TextRecognition::CheckParams() {
    if (!params_.model_dir.has_value() || params_.model_dir->empty()) {
        return absl::NotFoundError("Require text recognition model_dir.");
    }
    return absl::OkStatus();
}

void TextRecognition::CreateModel() {
    if(!rec_)
        rec_.reset(new TextRecPredictor(ToTextRecognitionModelParams(params_)));
}

TextRecPredictorParams TextRecognition::ToTextRecognitionModelParams(
    const TextRecognitionParams& from) {
    TextRecPredictorParams to;

    // English:
    // Keep the same parameter mapping style as your original code.
    to.model_name = from.model_name;
    to.model_dir = from.model_dir;
    to.lang = from.lang;
    to.ocr_version = from.ocr_version;
    to.vis_font_dir = from.vis_font_dir;
    to.device = from.device;
    to.enable_mkldnn = from.enable_mkldnn;
    to.mkldnn_cache_capacity = from.mkldnn_cache_capacity;
    to.precision = from.precision;
    to.cpu_threads = from.cpu_threads;
    to.input_shape = from.input_shape;

    return to;
}

absl::StatusOr<TextRecPredictorResult> TextRecognition::Predict(const cv::Mat& bgr) {
    if (!rec_) {
        return absl::FailedPreconditionError("TextRecognition::Predict: rec_ is null");
    }
    return rec_->Predict(bgr);
}

absl::StatusOr<TextRecPredictorResult> TextRecognition::PredictPath(const std::string& image_path) {
    if (!rec_) {
        return absl::FailedPreconditionError("TextRecognition::PredictPath: rec_ is null");
    }
    return rec_->PredictPath(image_path);
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

absl::StatusOr<TextRecPredictorResult> TextRecognition::PredictFrame(const BgraFrame& frame) {
    if (!rec_) {
        return absl::FailedPreconditionError("TextRecognition::PredictFrame: rec_ is null");
    }

    cv::Mat bgr = MakeBgrFromBgraFrame(frame);
    if (bgr.empty()) {
        return absl::InvalidArgumentError("TextRecognition::PredictFrame: invalid BGRA frame");
    }

    auto r = rec_->Predict(bgr);
    if (!r.ok()) return r.status();

    // English:
    // Fill extra info if needed.
    auto out = r.value();
    out.input_path = "";  // frame has no path
    out.input_image = bgr;
    return out;
}
