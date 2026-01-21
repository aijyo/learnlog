#pragma once
#include <memory>
#include <string>
#include <vector>

#include "./utils/utils.h"

class PPOCR_API PaddleOcr
{
public:
    struct Line
    {
        std::string text;  // UTF-8 text
        int x1 = 0, y1 = 0, x2 = 0, y2 = 0; // bounding box
        float confidence = 0.0f; // 0..100
    };

public:
    PaddleOcr();
    ~PaddleOcr();

    PaddleOcr(const PaddleOcr&) = delete;
    PaddleOcr& operator=(const PaddleOcr&) = delete;

    // English:
    // Initialize PaddleOCR pipeline with explicit params (mirrors cli.cc -> GetPipelineMoudleParams()).
    // det_dir/rec_dir/cls_dir are model directories.
    // cls_dir can be empty if you don't use textline orientation classification.
    bool Init(const std::string& det_dir,
        const std::string& rec_dir,
        const std::string& cls_dir,
        const std::string& lang,
        const std::string& device = "cpu",
        const std::string& precision = "fp32",
        bool enable_mkldnn = false,
        int cpu_threads = 4);

    // English:
    // Optional tuning similar to cli flags
    void SetDetParams(int limit_side_len, const std::string& limit_type,
        float thresh, float box_thresh, float unclip_ratio);
    void SetRecParams(float score_thresh, int batch_size);

    // English:
    // Recognize text lines, sorted from top-to-bottom then left-to-right.
    bool RecognizeLinesTopDown(const BgraFrame& frame, std::vector<Line>& out_lines);

    // English:
    // Convenience: return concatenated lines (top-down), each line ends with '\n'.
    bool RecognizeTextTopDown(const BgraFrame& frame, std::string& out_text);

    const std::string& GetLastError() const { return last_error_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string last_error_;
};
