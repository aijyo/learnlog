//#pragma once
//#include <string>
//#include <vector>
//#include <functional>
//#include <mutex>
//#include <optional>
//
//#include <opencv2/opencv.hpp>
//
//#include "utils/absl_shim.h"
//
//
//// Reuse your WoW frame type definition.
//// NOTE: In your project, BgraFrame is defined in PaddleOcr.h or utils.h.
//// Keep it consistent with your existing code.
//#include "./PaddleOcr.h"          // for BgraFrame (and maybe other common structs)
//#include "./predictor.h"          // TextRecPredictor / TextRecPredictorParams / TextRecPredictorResult
//
//struct BgraFrame
//{
//    int width = 0;
//    int height = 0;
//    int stride = 0;                 // bytes per row
//    int64_t frame_id = 0;
//    std::vector<uint8_t> data;      // BGRA
//};
// 
//// A lightweight dispatcher for TextRecPredictor (single image -> text).
//class TextRecProcessor
//{
//public:
//    struct Config
//    {
//        // English:
//        // TextRecPredictor parameters (model dir, device, threads, etc.)
//        TextRecPredictorParams rec;
//
//        // English:
//        // Throttle invocation rate to reduce CPU usage.
//        int min_trigger_interval_ms = 200;
//
//        // English:
//        // Optional ROI in pixels (relative to the captured BGRA frame).
//        // If not set, we use the whole frame.
//        std::optional<cv::Rect> roi = std::nullopt;
//
//        // English:
//        // Basic pre-processing switches (keep minimal and stable).
//        bool enable_grayscale = false;
//        bool enable_binary_threshold = false;
//        int binary_threshold = 180; // 0..255
//
//        // English:
//        // If true, we will try to remove alpha channel and convert BGRA->BGR.
//        bool force_bgra_to_bgr = true;
//    };
//
//    struct Event
//    {
//        int64_t frame_id = 0;
//
//        bool rec_ok = false;
//        std::string rec_err;
//
//        std::string rec_text;
//        float rec_score = 0.0f;
//
//        // English:
//        // For debugging/visualization if needed.
//        cv::Rect used_roi{};
//    };
//
//public:
//    TextRecProcessor() = default;
//    explicit TextRecProcessor(const Config& cfg);
//
//    bool Init(const Config& cfg);
//
//    // English:
//    // Feed a BGRA frame into processor.
//    // Returns true if ran (not throttled) AND succeeded.
//    bool OnFrame(const BgraFrame& frame);
//
//    void OnEvent(std::function<void(const Event&)> cb);
//
//    std::string GetLastError() const;
//
//private:
//    static cv::Mat MakeBgrMatFromBgraFrame(const BgraFrame& frame, bool force_bgra_to_bgr);
//    static cv::Mat ApplyBasicPreprocess(const cv::Mat& bgr, const Config& cfg);
//
//private:
//    Config cfg_;
//    std::unique_ptr<TextRecPredictor> rec_;
//
//    std::function<void(const Event&)> cb_;
//    mutable std::mutex mu_;
//    std::string last_error_;
//    int64_t last_trigger_tick_ms_ = 0;
//};
