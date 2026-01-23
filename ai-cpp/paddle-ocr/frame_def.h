#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <opencv2/core.hpp>
// A simple BGRA frame container consistent with your existing pipeline.
struct BgraFrame
{
    int width = 0;
    int height = 0;
    int stride = 0;                 // bytes per row
    int64_t frame_id = 0;
    std::vector<uint8_t> data;      // BGRA
};


// A single OCR instance (one detected text region + recognized text).
struct OcrInstance {
    std::string text;                 // UTF-8
    float rec_score = 0.0f;           // recognizer confidence
    float det_score = 1.0f;           // detector confidence (if provided)
    std::vector<cv::Point2f> quad;      // 4 points in ORIGINAL screenshot coords
    cv::Rect bbox;                    // bounding rect in ORIGINAL screenshot coords
};

// OCR result for a screenshot (or ROI)
struct OcrFrameResult {
    cv::Size image_size;              // original screenshot size
    std::vector<OcrInstance> items;   // all recognized instances
    std::string merged_text;          // optional merged text (top-down)
};