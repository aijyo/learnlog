#pragma once
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <memory>

struct BgraFrame
{
    int width = 0;
    int height = 0;
    int stride = 0; // bytes per row
    std::vector<uint8_t> data;
    uint64_t frame_id = 0;
};

class SpellRecognizer
{
public:
    struct Result
    {
        int spell_id = -1;     // predicted class id (spellid index)
        int status = -1;       // e.g. 0=unavailable, 1=available (depends on your training)
        float spell_conf = 0;  // confidence/prob
        float status_conf = 0; // confidence/prob
        // Detected box in original image coordinates (skill box)
        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        uint64_t frame_id = 0;
    };

    struct Options
    {
        // YOLO
        int yolo_input_w = 320;
        int yolo_input_h = 320;
        float yolo_conf_thres = 0.25f;
        float yolo_iou_thres = 0.45f;

        // If YOLO is trained for a single class (skill box), keep as 1.
        // If multiple classes exist, set accordingly.
        int yolo_num_classes = 1;

        // MobileNet
        int icon_size = 64; // crop resize to 64x64
        // Normalization for MobileNet input:
        // If your training used ToTensor() only => [0,1], mean=0, std=1.
        // If used (x-0.5)/0.5 => mean=0.5, std=0.5.
        float mn_mean = 0.0f;
        float mn_std = 1.0f;

        // If your Mobilenet expects RGB order. Usually yes.
        bool mn_rgb = true;
    };

public:
    SpellRecognizer();
    ~SpellRecognizer();

    // model paths: yolo.onnx + mobilenet.onnx
    bool Init(const std::string& yolo_onnx_path,
        const std::string& mobilenet_onnx_path,
        const Options& opt = Options());

    std::optional<Result> Run(const BgraFrame& frame);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
