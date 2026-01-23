//#include "./TextRecProcessor.h"
//
//#include <chrono>
//#include <cstdarg>
//#include <cstdio>
//
//namespace {
//
//    static int64_t NowMs()
//    {
//        using namespace std::chrono;
//        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
//    }
//
//    // English:
//    // Minimal logger shim to remove ilogger.h dependency.
//    inline void LogPrintf(const char* level, const char* fmt, ...)
//    {
//        std::fprintf(stderr, "[%s] ", (level ? level : "I"));
//        va_list args;
//        va_start(args, fmt);
//        std::vfprintf(stderr, fmt, args);
//        va_end(args);
//        std::fprintf(stderr, "\n");
//    }
//
//} // namespace
//
//#ifndef INFO
//#define INFO(...)  LogPrintf("I", __VA_ARGS__)
//#endif
//#ifndef INFOW
//#define INFOW(...) LogPrintf("W", __VA_ARGS__)
//#endif
//#ifndef INFOE
//#define INFOE(...) LogPrintf("E", __VA_ARGS__)
//#endif
//
//TextRecProcessor::TextRecProcessor(const Config& cfg)
//{
//    Init(cfg);
//}
//
//bool TextRecProcessor::Init(const Config& cfg)
//{
//    std::lock_guard<std::mutex> lk(mu_);
//    cfg_ = cfg;
//    last_error_.clear();
//    last_trigger_tick_ms_ = 0;
//
//    try {
//        // English:
//        // Create TextRecPredictor (Build is done inside constructor in our previous implementation).
//        rec_.reset(new TextRecPredictor(cfg_.rec));
//    }
//    catch (...) {
//        last_error_ = "TextRecPredictor init failed (exception)";
//        return false;
//    }
//
//    return true;
//}
//
//void TextRecProcessor::OnEvent(std::function<void(const Event&)> cb)
//{
//    std::lock_guard<std::mutex> lk(mu_);
//    cb_ = std::move(cb);
//}
//
//std::string TextRecProcessor::GetLastError() const
//{
//    std::lock_guard<std::mutex> lk(mu_);
//    return last_error_;
//}
//
//cv::Mat TextRecProcessor::MakeBgrMatFromBgraFrame(const BgraFrame& frame, bool force_bgra_to_bgr)
//{
//    // English:
//    // Assumptions about BgraFrame:
//    // - frame.data points to BGRA bytes
//    // - frame.width / frame.height are valid
//    // - frame.stride_bytes is the row bytes
//    //
//    // If your BgraFrame uses different field names, adjust here only.
//
//    if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0)
//        return cv::Mat();
//
//    // Try to guess stride field names.
//    // Common: stride or stride_bytes
//    int stride = 0;
//    if (frame.stride_bytes > 0) stride = frame.stride_bytes;
//    else if (frame.stride > 0) stride = frame.stride;
//    else stride = frame.width * 4;
//
//    cv::Mat bgra(frame.height, frame.width, CV_8UC4, (void*)frame.data, stride);
//
//    if (!force_bgra_to_bgr) {
//        // Keep 4 channels if user wants.
//        return bgra.clone();
//    }
//
//    cv::Mat bgr;
//    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
//    return bgr;
//}
//
//cv::Mat TextRecProcessor::ApplyBasicPreprocess(const cv::Mat& bgr, const Config& cfg)
//{
//    if (bgr.empty())
//        return bgr;
//
//    cv::Mat out = bgr;
//
//    if (cfg.enable_grayscale) {
//        cv::Mat gray;
//        if (out.channels() == 3) {
//            cv::cvtColor(out, gray, cv::COLOR_BGR2GRAY);
//        }
//        else if (out.channels() == 4) {
//            cv::cvtColor(out, gray, cv::COLOR_BGRA2GRAY);
//        }
//        else {
//            gray = out;
//        }
//        out = gray;
//    }
//
//    if (cfg.enable_binary_threshold) {
//        cv::Mat gray;
//        if (out.channels() == 1) {
//            gray = out;
//        }
//        else if (out.channels() == 3) {
//            cv::cvtColor(out, gray, cv::COLOR_BGR2GRAY);
//        }
//        else if (out.channels() == 4) {
//            cv::cvtColor(out, gray, cv::COLOR_BGRA2GRAY);
//        }
//        else {
//            gray = out;
//        }
//
//        cv::Mat bin;
//        cv::threshold(gray, bin, cfg.binary_threshold, 255, cv::THRESH_BINARY);
//        out = bin;
//    }
//
//    // English:
//    // TextRecPredictor::Predict expects a cv::Mat, typically BGR.
//    // If we produced grayscale/binary, convert back to BGR to keep behavior stable.
//    if (out.channels() == 1) {
//        cv::Mat back_bgr;
//        cv::cvtColor(out, back_bgr, cv::COLOR_GRAY2BGR);
//        out = back_bgr;
//    }
//
//    return out;
//}
//
//bool TextRecProcessor::OnFrame(const BgraFrame& frame)
//{
//    std::function<void(const Event&)> cb;
//    Config cfg_snapshot;
//
//    {
//        std::lock_guard<std::mutex> lk(mu_);
//        cb = cb_;
//        cfg_snapshot = cfg_;
//    }
//
//    const int64_t now = NowMs();
//
//    // English:
//    // Throttle: if too frequent, do nothing.
//    {
//        std::lock_guard<std::mutex> lk(mu_);
//        if (last_trigger_tick_ms_ != 0 &&
//            (now - last_trigger_tick_ms_) < cfg_snapshot.min_trigger_interval_ms) {
//            return false;
//        }
//    }
//
//    Event ev;
//    ev.frame_id = frame.frame_id;
//
//    if (!rec_) {
//        ev.rec_ok = false;
//        ev.rec_err = "rec predictor not initialized";
//        {
//            std::lock_guard<std::mutex> lk(mu_);
//            last_error_ = ev.rec_err;
//        }
//        if (cb) cb(ev);
//        return false;
//    }
//
//    // 1) BGRA -> BGR
//    cv::Mat bgr = MakeBgrMatFromBgraFrame(frame, cfg_snapshot.force_bgra_to_bgr);
//    if (bgr.empty()) {
//        ev.rec_ok = false;
//        ev.rec_err = "invalid frame (empty mat)";
//        {
//            std::lock_guard<std::mutex> lk(mu_);
//            last_error_ = ev.rec_err;
//        }
//        if (cb) cb(ev);
//        return false;
//    }
//
//    // 2) ROI crop (optional)
//    cv::Rect roi_rect(0, 0, bgr.cols, bgr.rows);
//    if (cfg_snapshot.roi.has_value()) {
//        roi_rect = cfg_snapshot.roi.value();
//        roi_rect &= cv::Rect(0, 0, bgr.cols, bgr.rows);
//        if (roi_rect.width <= 0 || roi_rect.height <= 0) {
//            ev.rec_ok = false;
//            ev.rec_err = "roi is out of range";
//            {
//                std::lock_guard<std::mutex> lk(mu_);
//                last_error_ = ev.rec_err;
//            }
//            if (cb) cb(ev);
//            return false;
//        }
//    }
//    ev.used_roi = roi_rect;
//
//    cv::Mat crop = bgr(roi_rect).clone();
//
//    // 3) Basic preprocess (optional, keep minimal)
//    crop = ApplyBasicPreprocess(crop, cfg_snapshot);
//
//    // 4) Predict
//    auto r = rec_->Predict(crop);
//    if (!r.ok()) {
//        ev.rec_ok = false;
//        ev.rec_err = r.status().ToString();
//
//        // English:
//        // Do NOT advance throttle timestamp on failure (avoid "failure lock").
//        {
//            std::lock_guard<std::mutex> lk(mu_);
//            last_error_ = ev.rec_err;
//        }
//
//        if (cb) cb(ev);
//        return false;
//    }
//
//    // Success: commit throttle timestamp now.
//    {
//        std::lock_guard<std::mutex> lk(mu_);
//        last_trigger_tick_ms_ = now;
//        last_error_.clear();
//    }
//
//    ev.rec_ok = true;
//    ev.rec_text = r.value().rec_text;
//    ev.rec_score = r.value().rec_score;
//
//    if (cb) cb(ev);
//    return true;
//}
