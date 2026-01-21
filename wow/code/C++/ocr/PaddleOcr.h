#pragma once
#include <string>
#include <vector>
#include <cstdint>

// English:
// A simple BGRA frame container consistent with your existing pipeline.
struct BgraFrame
{
    int width = 0;
    int height = 0;
    int stride = 0;                 // bytes per row
    int64_t frame_id = 0;
    std::vector<uint8_t> data;      // BGRA
};

// English:
// PaddleOCR result item (text + confidence + polygon/box).
struct OcrItem
{
    std::string text;
    float score = 0.0f;

    // English:
    // Polygon points in image coordinates: [x0,y0,x1,y1,x2,y2,x3,y3]
    std::vector<float> poly;
};

class PaddleOcr
{
public:
    struct Config
    {
        // English:
        // Paddle inference model dirs (exported inference model).
        // Each dir should contain model.pdmodel + model.pdiparams (or __model__ + __params__ for older).
        std::string det_model_dir;
        std::string rec_model_dir;
        std::string cls_model_dir;  // optional, can be empty
        std::string rec_dict_path;  // e.g. ppocr_keys_v1.txt

        // English:
        // Runtime settings
        bool use_gpu = false;
        int gpu_id = 0;
        int cpu_threads = 4;
        bool enable_mkldnn = false;

        // English:
        // OCR options
        bool use_angle_cls = false;
        int max_side_len = 960; // det resize constraint
        float det_db_thresh = 0.3f;
        float det_db_box_thresh = 0.6f;
        float det_db_unclip_ratio = 1.5f;

        // English:
        // For compatibility with your existing code
        int min_text_len = 1;
    };

    PaddleOcr() = default;
    explicit PaddleOcr(const Config& cfg);

    bool Init(const Config& cfg);
    bool IsInited() const { return inited_; }

    // English:
    // Run OCR on BGRA image, returns items in roughly top-down order.
    bool Run(const BgraFrame& frame, std::vector<OcrItem>& out_items, std::string& out_err);

    // English:
    // Convenience: returns plain text (top-down joined).
    bool RunText(const BgraFrame& frame, std::string& out_text_topdown, std::string& out_err);

private:
    Config cfg_;
    bool inited_ = false;

    // English:
    // Opaque pointers to predictors (det/cls/rec).
    // We keep them as void* to avoid leaking paddle headers in the public header.
    void* det_predictor_ = nullptr;
    void* rec_predictor_ = nullptr;
    void* cls_predictor_ = nullptr;

    // English:
    // Internal helpers
    bool CreatePredictors(std::string& err);
    void DestroyPredictors();
};
