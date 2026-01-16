#include "spell_recognizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>

#include <onnxruntime_cxx_api.h>

namespace {

    // ---------------------------
    // Math / utility
    // ---------------------------

    static inline float Clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }

    struct BoxF
    {
        float x1, y1, x2, y2;
        float score;
        int cls;
    };

    static float IoU(const BoxF& a, const BoxF& b)
    {
        float xx1 = std::max(a.x1, b.x1);
        float yy1 = std::max(a.y1, b.y1);
        float xx2 = std::min(a.x2, b.x2);
        float yy2 = std::min(a.y2, b.y2);
        float w = std::max(0.0f, xx2 - xx1);
        float h = std::max(0.0f, yy2 - yy1);
        float inter = w * h;
        float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
        float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
        float uni = areaA + areaB - inter;
        return (uni <= 0.0f) ? 0.0f : (inter / uni);
    }

    static std::vector<BoxF> Nms(const std::vector<BoxF>& boxes, float iou_thres)
    {
        std::vector<BoxF> sorted = boxes;
        std::sort(sorted.begin(), sorted.end(), [](const BoxF& a, const BoxF& b) { return a.score > b.score; });

        std::vector<BoxF> keep;
        std::vector<char> removed(sorted.size(), 0);

        for (size_t i = 0; i < sorted.size(); ++i)
        {
            if (removed[i]) continue;
            keep.push_back(sorted[i]);
            for (size_t j = i + 1; j < sorted.size(); ++j)
            {
                if (removed[j]) continue;
                if (sorted[i].cls != sorted[j].cls) continue;
                if (IoU(sorted[i], sorted[j]) > iou_thres) removed[j] = 1;
            }
        }
        return keep;
    }

    // ---------------------------
    // Image helpers (no OpenCV)
    // ---------------------------

    static bool ValidateFrame(const BgraFrame& f)
    {
        if (f.width <= 0 || f.height <= 0) return false;
        if (f.stride < f.width * 4) return false;
        if ((int)f.data.size() < f.stride * f.height) return false;
        return true;
    }

    // Convert BGRA (frame) to RGB float [0..1], HWC
    static void BgraToRgbFloat01_HWC(const BgraFrame& f, std::vector<float>& out_rgb, int& out_w, int& out_h)
    {
        out_w = f.width;
        out_h = f.height;
        out_rgb.resize((size_t)out_w * out_h * 3);

        for (int y = 0; y < out_h; ++y)
        {
            const uint8_t* row = f.data.data() + (size_t)y * f.stride;
            for (int x = 0; x < out_w; ++x)
            {
                const uint8_t b = row[x * 4 + 0];
                const uint8_t g = row[x * 4 + 1];
                const uint8_t r = row[x * 4 + 2];
                // const uint8_t a = row[x * 4 + 3];

                const size_t idx = ((size_t)y * out_w + x) * 3;
                out_rgb[idx + 0] = r / 255.0f;
                out_rgb[idx + 1] = g / 255.0f;
                out_rgb[idx + 2] = b / 255.0f;
            }
        }
    }

    // Bilinear resize RGB float HWC
    static void ResizeRgbFloat_HWC_Bilinear(const float* src, int sw, int sh,
        float* dst, int dw, int dh)
    {
        const float scale_x = (dw > 1) ? (float)(sw - 1) / (float)(dw - 1) : 0.0f;
        const float scale_y = (dh > 1) ? (float)(sh - 1) / (float)(dh - 1) : 0.0f;

        for (int y = 0; y < dh; ++y)
        {
            float fy = y * scale_y;
            int y0 = (int)fy;
            int y1 = std::min(y0 + 1, sh - 1);
            float wy = fy - y0;

            for (int x = 0; x < dw; ++x)
            {
                float fx = x * scale_x;
                int x0 = (int)fx;
                int x1 = std::min(x0 + 1, sw - 1);
                float wx = fx - x0;

                const float* p00 = src + ((y0 * sw + x0) * 3);
                const float* p01 = src + ((y0 * sw + x1) * 3);
                const float* p10 = src + ((y1 * sw + x0) * 3);
                const float* p11 = src + ((y1 * sw + x1) * 3);

                float* pd = dst + ((y * dw + x) * 3);
                for (int c = 0; c < 3; ++c)
                {
                    float v0 = p00[c] * (1.0f - wx) + p01[c] * wx;
                    float v1 = p10[c] * (1.0f - wx) + p11[c] * wx;
                    pd[c] = v0 * (1.0f - wy) + v1 * wy;
                }
            }
        }
    }

    struct LetterboxInfo
    {
        float scale = 1.0f;
        int pad_x = 0;
        int pad_y = 0;
        int new_w = 0;
        int new_h = 0;
    };

    // Letterbox resize to (dw, dh), keep aspect ratio, pad with 0.
    // Input: RGB float HWC
    static void LetterboxRgbFloat_HWC(const float* src, int sw, int sh,
        float* dst, int dw, int dh,
        LetterboxInfo& info)
    {
        std::fill(dst, dst + (size_t)dw * dh * 3, 0.0f);

        float r = std::min((float)dw / (float)sw, (float)dh / (float)sh);
        int nw = (int)std::round(sw * r);
        int nh = (int)std::round(sh * r);
        int px = (dw - nw) / 2;
        int py = (dh - nh) / 2;

        info.scale = r;
        info.pad_x = px;
        info.pad_y = py;
        info.new_w = nw;
        info.new_h = nh;

        std::vector<float> resized((size_t)nw * nh * 3);
        ResizeRgbFloat_HWC_Bilinear(src, sw, sh, resized.data(), nw, nh);

        for (int y = 0; y < nh; ++y)
        {
            float* drow = dst + ((size_t)(y + py) * dw + px) * 3;
            const float* srow = resized.data() + (size_t)y * nw * 3;
            std::memcpy(drow, srow, (size_t)nw * 3 * sizeof(float));
        }
    }

    // Crop RGB float HWC from original (no padding), clamp to bounds
    static std::vector<float> CropRgbFloat_HWC(const float* src, int sw, int sh,
        int x1, int y1, int x2, int y2,
        int& cw, int& ch)
    {
        x1 = std::max(0, std::min(x1, sw - 1));
        y1 = std::max(0, std::min(y1, sh - 1));
        x2 = std::max(0, std::min(x2, sw));
        y2 = std::max(0, std::min(y2, sh));

        if (x2 <= x1) x2 = std::min(sw, x1 + 1);
        if (y2 <= y1) y2 = std::min(sh, y1 + 1);

        cw = x2 - x1;
        ch = y2 - y1;

        std::vector<float> out((size_t)cw * ch * 3);
        for (int y = 0; y < ch; ++y)
        {
            const float* srow = src + (((size_t)(y + y1) * sw + x1) * 3);
            float* drow = out.data() + (size_t)y * cw * 3;
            std::memcpy(drow, srow, (size_t)cw * 3 * sizeof(float));
        }
        return out;
    }

    // Convert RGB float HWC to CHW float (for ONNX input)
    // Optionally apply mean/std
    static void HwcToChw(const float* src_hwc, int w, int h,
        std::vector<float>& dst_chw,
        float mean, float stdv)
    {
        dst_chw.resize((size_t)3 * w * h);
        const float inv_std = (stdv == 0.0f) ? 1.0f : (1.0f / stdv);

        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                const size_t si = ((size_t)y * w + x) * 3;
                float r = src_hwc[si + 0];
                float g = src_hwc[si + 1];
                float b = src_hwc[si + 2];

                r = (r - mean) * inv_std;
                g = (g - mean) * inv_std;
                b = (b - mean) * inv_std;

                dst_chw[(size_t)0 * w * h + (size_t)y * w + x] = r;
                dst_chw[(size_t)1 * w * h + (size_t)y * w + x] = g;
                dst_chw[(size_t)2 * w * h + (size_t)y * w + x] = b;
            }
        }
    }

} // namespace

// ---------------------------
// SpellRecognizer::Impl
// ---------------------------

struct SpellRecognizer::Impl
{
    Options opt;

    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "SpellRecognizer" };
    Ort::SessionOptions sess_opt;

    std::unique_ptr<Ort::Session> yolo_sess;
    std::unique_ptr<Ort::Session> mn_sess;

    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> yolo_in_names_str;
    std::vector<std::string> yolo_out_names_str;
    std::vector<const char*> yolo_in_names;
    std::vector<const char*> yolo_out_names;

    std::vector<std::string> mn_in_names_str;
    std::vector<std::string> mn_out_names_str;
    std::vector<const char*> mn_in_names;
    std::vector<const char*> mn_out_names;

    bool InitSessions(const std::string& yolo_path, const std::string& mn_path)
    {
        sess_opt.SetIntraOpNumThreads(1);
        sess_opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        yolo_sess = std::make_unique<Ort::Session>(env, yolo_path.c_str(), sess_opt);
        mn_sess = std::make_unique<Ort::Session>(env, mn_path.c_str(), sess_opt);

        auto fill_names = [&](Ort::Session& s,
            std::vector<std::string>& in_str,
            std::vector<std::string>& out_str,
            std::vector<const char*>& in_c,
            std::vector<const char*>& out_c)
            {
                size_t nin = s.GetInputCount();
                size_t nout = s.GetOutputCount();
                in_str.clear(); out_str.clear();
                in_c.clear(); out_c.clear();
                in_str.reserve(nin); out_str.reserve(nout);
                in_c.reserve(nin); out_c.reserve(nout);

                for (size_t i = 0; i < nin; ++i)
                {
                    Ort::AllocatedStringPtr name = s.GetInputNameAllocated(i, allocator);
                    in_str.emplace_back(name.get());
                }
                for (size_t i = 0; i < nout; ++i)
                {
                    Ort::AllocatedStringPtr name = s.GetOutputNameAllocated(i, allocator);
                    out_str.emplace_back(name.get());
                }
                for (auto& ss : in_str) in_c.push_back(ss.c_str());
                for (auto& ss : out_str) out_c.push_back(ss.c_str());
            };

        fill_names(*yolo_sess, yolo_in_names_str, yolo_out_names_str, yolo_in_names, yolo_out_names);
        fill_names(*mn_sess, mn_in_names_str, mn_out_names_str, mn_in_names, mn_out_names);

        return true;
    }

    // ---------------------------
    // YOLO decode (supports common YOLOv8 ONNX shapes)
    // ---------------------------

    std::vector<BoxF> DecodeYolo(const float* out, const std::vector<int64_t>& shape,
        const LetterboxInfo& lb, int orig_w, int orig_h)
    {
        // Common shapes:
        // A) [1, num_preds, 5+nc]
        // B) [1, 5+nc, num_preds]
        // Some exports: [num_preds, 5+nc] or [1, num_preds, 4+nc] etc.
        // We'll assume obj+cls exist: 4 box + obj + nc.
        std::vector<BoxF> boxes;

        int64_t dim0 = (shape.size() > 0) ? shape[0] : 0;
        int64_t dim1 = (shape.size() > 1) ? shape[1] : 0;
        int64_t dim2 = (shape.size() > 2) ? shape[2] : 0;

        int nc = opt.yolo_num_classes;

        auto push_box = [&](float cx, float cy, float w, float h, float score, int cls)
            {
                // Undo letterbox:
                // x' = (x - pad) / scale
                float x1 = (cx - w * 0.5f);
                float y1 = (cy - h * 0.5f);
                float x2 = (cx + w * 0.5f);
                float y2 = (cy + h * 0.5f);

                x1 = (x1 - lb.pad_x) / lb.scale;
                y1 = (y1 - lb.pad_y) / lb.scale;
                x2 = (x2 - lb.pad_x) / lb.scale;
                y2 = (y2 - lb.pad_y) / lb.scale;

                x1 = std::max(0.0f, std::min(x1, (float)orig_w));
                y1 = std::max(0.0f, std::min(y1, (float)orig_h));
                x2 = std::max(0.0f, std::min(x2, (float)orig_w));
                y2 = std::max(0.0f, std::min(y2, (float)orig_h));

                BoxF b;
                b.x1 = x1; b.y1 = y1; b.x2 = x2; b.y2 = y2;
                b.score = score;
                b.cls = cls;
                boxes.push_back(b);
            };

        // Case A: [1, N, 5+nc]
        if (shape.size() == 3 && dim0 == 1 && dim2 >= (5 + nc))
        {
            int64_t N = dim1;
            int64_t C = dim2;
            for (int64_t i = 0; i < N; ++i)
            {
                const float* p = out + i * C;
                float cx = p[0];
                float cy = p[1];
                float ww = p[2];
                float hh = p[3];
                float obj = p[4];

                int best_cls = 0;
                float best_cls_score = (nc > 0) ? p[5] : 1.0f;
                for (int c = 1; c < nc; ++c)
                {
                    float cs = p[5 + c];
                    if (cs > best_cls_score) { best_cls_score = cs; best_cls = c; }
                }
                float score = obj * best_cls_score;
                if (score >= opt.yolo_conf_thres)
                    push_box(cx, cy, ww, hh, score, best_cls);
            }
        }
        // Case B: [1, 5+nc, N]
        else if (shape.size() == 3 && dim0 == 1 && dim1 >= (5 + nc))
        {
            int64_t C = dim1;
            int64_t N = dim2;

            const float* cx = out + 0 * N;
            const float* cy = out + 1 * N;
            const float* ww = out + 2 * N;
            const float* hh = out + 3 * N;
            const float* obj = out + 4 * N;
            const float* cls_base = out + 5 * N;

            for (int64_t i = 0; i < N; ++i)
            {
                int best_cls = 0;
                float best_cls_score = (nc > 0) ? cls_base[i] : 1.0f;
                for (int c = 1; c < nc; ++c)
                {
                    float cs = cls_base[(size_t)c * N + (size_t)i];
                    if (cs > best_cls_score) { best_cls_score = cs; best_cls = c; }
                }
                float score = obj[i] * best_cls_score;
                if (score >= opt.yolo_conf_thres)
                    push_box(cx[i], cy[i], ww[i], hh[i], score, best_cls);
            }
        }
        else
        {
            // Unknown shape -> return empty
        }

        return Nms(boxes, opt.yolo_iou_thres);
    }

    std::optional<BoxF> DetectSkillBox(const BgraFrame& frame,
        const std::vector<float>& rgb_hwc, int w, int h)
    {
        // Letterbox -> yolo input
        const int iw = opt.yolo_input_w;
        const int ih = opt.yolo_input_h;

        std::vector<float> yolo_hwc((size_t)iw * ih * 3);
        LetterboxInfo lb{};
        LetterboxRgbFloat_HWC(rgb_hwc.data(), w, h, yolo_hwc.data(), iw, ih, lb);

        // HWC -> CHW, no extra normalization (YOLO typically uses [0,1])
        std::vector<float> yolo_chw;
        HwcToChw(yolo_hwc.data(), iw, ih, yolo_chw, /*mean*/0.0f, /*std*/1.0f);

        // ONNX input tensor
        std::array<int64_t, 4> in_shape{ 1, 3, (int64_t)ih, (int64_t)iw };
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(mem, yolo_chw.data(), yolo_chw.size(),
            in_shape.data(), in_shape.size());

        auto outputs = yolo_sess->Run(Ort::RunOptions{ nullptr },
            yolo_in_names.data(), &input, 1,
            yolo_out_names.data(), yolo_out_names.size());

        if (outputs.empty()) return std::nullopt;

        // Use first output as detection head (common for YOLOv8 ONNX)
        Ort::Value& out0 = outputs[0];
        auto type_info = out0.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_shape = type_info.GetShape();
        const float* out_data = out0.GetTensorData<float>();

        std::vector<BoxF> boxes = DecodeYolo(out_data, out_shape, lb, w, h);
        if (boxes.empty()) return std::nullopt;

        // For skill box detection, usually take top-1
        return boxes[0];
    }

    // ---------------------------
    // MobileNet infer
    // ---------------------------

    std::optional<SpellRecognizer::Result> ClassifySpell(const BgraFrame& frame,
        const std::vector<float>& rgb_hwc, int w, int h,
        const BoxF& box)
    {
        // Crop box -> resize to 64x64 -> CHW -> normalize
        int x1 = (int)std::floor(box.x1);
        int y1 = (int)std::floor(box.y1);
        int x2 = (int)std::ceil(box.x2);
        int y2 = (int)std::ceil(box.y2);

        int cw = 0, ch = 0;
        std::vector<float> crop = CropRgbFloat_HWC(rgb_hwc.data(), w, h, x1, y1, x2, y2, cw, ch);

        const int S = opt.icon_size;
        std::vector<float> resized((size_t)S * S * 3);
        ResizeRgbFloat_HWC_Bilinear(crop.data(), cw, ch, resized.data(), S, S);

        // If your MobileNet expects BGR, swap here (default mn_rgb=true -> keep RGB)
        if (!opt.mn_rgb)
        {
            for (int i = 0; i < S * S; ++i)
                std::swap(resized[i * 3 + 0], resized[i * 3 + 2]);
        }

        std::vector<float> mn_chw;
        HwcToChw(resized.data(), S, S, mn_chw, opt.mn_mean, opt.mn_std);

        std::array<int64_t, 4> in_shape{ 1, 3, (int64_t)S, (int64_t)S };
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(mem, mn_chw.data(), mn_chw.size(),
            in_shape.data(), in_shape.size());

        auto outputs = mn_sess->Run(Ort::RunOptions{ nullptr },
            mn_in_names.data(), &input, 1,
            mn_out_names.data(), mn_out_names.size());
        if (outputs.empty()) return std::nullopt;

        // ---- Output handling (two common patterns) ----
        // Pattern 1: two outputs: [1, num_spells] and [1, num_status]
        // Pattern 2: one output containing both heads concatenated (custom)
        //
        // You MUST align this with your exported model.
        auto argmax = [](const float* p, int n, float& out_maxv) -> int
            {
                int idx = 0;
                float mv = p[0];
                for (int i = 1; i < n; ++i)
                {
                    if (p[i] > mv) { mv = p[i]; idx = i; }
                }
                out_maxv = mv;
                return idx;
            };

        SpellRecognizer::Result r;
        r.frame_id = frame.frame_id;
        r.x1 = x1; r.y1 = y1; r.x2 = x2; r.y2 = y2;

        if (outputs.size() >= 2)
        {
            // spell logits/prob
            {
                const float* p = outputs[0].GetTensorData<float>();
                auto s_info = outputs[0].GetTensorTypeAndShapeInfo();
                auto s_shape = s_info.GetShape();
                // expecting [1, K] or [K]
                int K = 0;
                if (s_shape.size() == 2) K = (int)s_shape[1];
                else if (s_shape.size() == 1) K = (int)s_shape[0];
                else return std::nullopt;

                float mv = 0.0f;
                int sid = argmax(p, K, mv);
                r.spell_id = sid;
                r.spell_conf = mv;
            }
            // status logits/prob (e.g., 2 classes)
            {
                const float* p = outputs[1].GetTensorData<float>();
                auto t_info = outputs[1].GetTensorTypeAndShapeInfo();
                auto t_shape = t_info.GetShape();
                int K = 0;
                if (t_shape.size() == 2) K = (int)t_shape[1];
                else if (t_shape.size() == 1) K = (int)t_shape[0];
                else return std::nullopt;

                float mv = 0.0f;
                int st = argmax(p, K, mv);
                r.status = st;
                r.status_conf = mv;
            }
        }
        else
        {
            // Single output fallback (custom layout)
            // TODO: If your model concatenates [spell_logits | status_logits],
            //       set split index below.
            const float* p = outputs[0].GetTensorData<float>();
            auto info = outputs[0].GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();

            int N = 0;
            if (shape.size() == 2) N = (int)shape[1];
            else if (shape.size() == 1) N = (int)shape[0];
            else return std::nullopt;

            // ---- IMPORTANT: you must set this correctly ----
            // Example: num_spells = 5000, status_classes=2 => N=5002, split=5000
            const int split_spell = N - 2; // assume last 2 are status
            if (split_spell <= 0) return std::nullopt;

            float mv_spell = 0.0f;
            int sid = argmax(p, split_spell, mv_spell);

            float mv_st = 0.0f;
            int st = argmax(p + split_spell, N - split_spell, mv_st);

            r.spell_id = sid;
            r.spell_conf = mv_spell;
            r.status = st;
            r.status_conf = mv_st;
        }

        return r;
    }
};

// ---------------------------
// SpellRecognizer public API
// ---------------------------

SpellRecognizer::SpellRecognizer() : impl_(std::make_unique<Impl>()) {}
SpellRecognizer::~SpellRecognizer() = default;

bool SpellRecognizer::Init(const std::string& yolo_onnx_path,
    const std::string& mobilenet_onnx_path,
    const Options& opt)
{
    impl_->opt = opt;
    try
    {
        return impl_->InitSessions(yolo_onnx_path, mobilenet_onnx_path);
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNXRuntime init failed: " << e.what() << std::endl;
        return false;
    }
}

std::optional<SpellRecognizer::Result> SpellRecognizer::Run(const BgraFrame& frame)
{
    if (!impl_ || !ValidateFrame(frame)) return std::nullopt;

    // Convert full frame BGRA -> RGB float [0,1] HWC
    std::vector<float> rgb_hwc;
    int w = 0, h = 0;
    BgraToRgbFloat01_HWC(frame, rgb_hwc, w, h);

    // 1) YOLO detect skill box
    std::optional<BoxF> box = impl_->DetectSkillBox(frame, rgb_hwc, w, h);
    if (!box.has_value()) return std::nullopt;

    // 2) MobileNet classify spellid + status
    return impl_->ClassifySpell(frame, rgb_hwc, w, h, box.value());
}
