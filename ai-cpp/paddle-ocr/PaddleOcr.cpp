#include "./PaddleOcr.h"

// Extra deps for PP-OCRv5 yaml/static_infer style
#include "utils/yaml_config.h"
#include "utils/utility.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <filesystem>

// Paddle Inference
#include "paddle_inference_api.h"

namespace fs = std::filesystem;

// ------------------------
// Internal implementation
// ------------------------
namespace {

    struct PaddleInferImpl {
        std::shared_ptr<paddle_infer::Predictor> det;
        std::shared_ptr<paddle_infer::Predictor> rec;
        std::shared_ptr<paddle_infer::Predictor> cls; // optional

        std::string det_in;
        std::string det_out;
        std::string rec_in;
        std::string rec_out;
        std::string cls_in;
        std::string cls_out;

        std::vector<std::string> dict; // ppocr_keys_v1
    };

    static bool EndsWith(const std::string& s, const std::string& suffix) {
        if (suffix.size() > s.size()) return false;
        return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
    }

    // English comment:
    // Trim leading and trailing ASCII whitespace.
    static std::string Trim(const std::string& s) {
        if (s.empty()) return s;
        size_t start = 0;
        size_t end = s.size();
        while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
        while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
        return s.substr(start, end - start);
    }

    // English comment:
    // Load non-empty lines from a UTF-8 text file. Lines starting with '#' are ignored.
    static std::vector<std::string> LoadDictTxt(const std::string& path) {
        std::ifstream ifs(path);
        std::vector<std::string> dict;
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::string s = Trim(line);
            if (s.empty() || s[0] == '#') continue;
            dict.push_back(s);
        }
        return dict;
    }

    // English comment:
    // Resolve recognition dictionary path from either:
    //  1) a dict txt path
    //  2) a yaml path (inference.yml/yaml) or a model directory containing yaml
    //  3) empty string -> use rec_model_dir to find yaml
    //
    // We follow static_infer's YamlConfig convention:
    //  - key: PostProcess.character_dict
    static bool ResolveRecDictPath(const std::string& dict_or_yaml,
        const std::string& rec_model_dir,
        std::string& out_dict_path,
        std::string& err) {
        err.clear();
        out_dict_path.clear();

        fs::path p(dict_or_yaml);
        bool is_empty = dict_or_yaml.empty();

        // If user directly passes a .txt file, use it.
        if (!is_empty && fs::exists(p) && fs::is_regular_file(p)) {
            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".txt") {
                out_dict_path = p.string();
                return true;
            }
        }

        // Determine the directory to load yaml from.
        fs::path yaml_dir;
        if (is_empty) {
            yaml_dir = fs::path(rec_model_dir);
        }
        else if (fs::exists(p) && fs::is_directory(p)) {
            yaml_dir = p;
        }
        else if (!dict_or_yaml.empty() && (EndsWith(dict_or_yaml, ".yml") || EndsWith(dict_or_yaml, ".yaml"))) {
            yaml_dir = p.parent_path();
        }
        else {
            // Maybe user passed a relative txt name; try rec_model_dir as base.
            fs::path candidate = fs::path(rec_model_dir) / p;
            if (fs::exists(candidate) && fs::is_regular_file(candidate)) {
                out_dict_path = candidate.string();
                return true;
            }
            err = "rec_dict_path is neither a dict txt nor a yaml file/dir: " + dict_or_yaml;
            return false;
        }

        if (yaml_dir.empty() || !fs::exists(yaml_dir) || !fs::is_directory(yaml_dir)) {
            err = "YAML directory does not exist: " + yaml_dir.string();
            return false;
        }

        // Load yaml config using provided implementation (static_infer).
        YamlConfig ycfg(yaml_dir.string());
        auto st = ycfg.GetString("PostProcess.character_dict", "");
        if (!st.ok()) {
            err = "Failed to read PostProcess.character_dict from yaml in dir: " + yaml_dir.string();
            return false;
        }
        std::string dict_rel = st.value();
        if (dict_rel.empty()) {
            err = "PostProcess.character_dict is empty in yaml under: " + yaml_dir.string();
            return false;
        }

        fs::path dict_path(dict_rel);
        if (dict_path.is_relative()) {
            dict_path = yaml_dir / dict_path;
        }
        if (!fs::exists(dict_path) || !fs::is_regular_file(dict_path)) {
            err = "Dict file does not exist: " + dict_path.string();
            return false;
        }

        out_dict_path = dict_path.string();
        return true;
    }

    // English:
    // Sort boxes in top-to-bottom then left-to-right order.
    static bool BoxLess(const OcrItem& a, const OcrItem& b) {
        if (a.poly.size() < 8 || b.poly.size() < 8)
            return a.poly.size() < b.poly.size();

        auto cy = [](const OcrItem& it) {
            return (it.poly[1] + it.poly[3] + it.poly[5] + it.poly[7]) * 0.25f;
            };
        auto cx = [](const OcrItem& it) {
            return (it.poly[0] + it.poly[2] + it.poly[4] + it.poly[6]) * 0.25f;
            };
        float ay = cy(a);
        float by = cy(b);
        if (std::fabs(ay - by) > 10.0f)
            return ay < by;
        return cx(a) < cx(b);
    }

    // English:
    // Greedy CTC decode. We assume:
    // - blank id = 0
    // - dict maps id-1 to character
    static std::string CtcGreedyDecode(const float* probs_or_logits, int T, int C,
        const std::vector<std::string>& dict,
        float* out_score) {
        const int blank_id = 0;
        int prev = -1;
        float score_sum = 0.0f;
        int score_cnt = 0;

        std::string text;
        for (int t = 0; t < T; ++t) {
            const float* p = probs_or_logits + t * C;
            int id = (int)(std::max_element(p, p + C) - p);
            float prob = p[id];
            if (id != blank_id && id != prev) {
                int dict_id = id - 1;
                if (dict_id >= 0 && dict_id < (int)dict.size())
                    text += dict[dict_id];
                score_sum += prob;
                score_cnt++;
            }
            prev = id;
        }
        if (out_score)
            *out_score = (score_cnt > 0) ? (score_sum / score_cnt) : 0.0f;
        return text;
    }

    // English:
    // Convert BGR float image (0..1) to NCHW tensor (RGB order).
    static void HwcToNchwRgb(const cv::Mat& img_f32_bgr, std::vector<float>& out_nchw) {
        const int H = img_f32_bgr.rows;
        const int W = img_f32_bgr.cols;
        out_nchw.assign((size_t)1 * 3 * H * W, 0.0f);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const cv::Vec3f v = img_f32_bgr.at<cv::Vec3f>(y, x);
                // BGR -> RGB
                out_nchw[(0 * 3 + 0) * H * W + y * W + x] = v[2];
                out_nchw[(0 * 3 + 1) * H * W + y * W + x] = v[1];
                out_nchw[(0 * 3 + 2) * H * W + y * W + x] = v[0];
            }
        }
    }

    // English:
    // A minimal contour->quad extraction for DB-style probability map.
    static std::vector<std::array<cv::Point2f, 4>> ExtractQuadsFromDbProb(
        const cv::Mat& prob_map,
        float bin_thresh,
        float box_thresh,
        float unclip_ratio)
    {
        std::vector<std::array<cv::Point2f, 4>> quads;
        if (prob_map.empty() || prob_map.type() != CV_32F)
            return quads;

        cv::Mat bin;
        cv::threshold(prob_map, bin, bin_thresh, 255, cv::THRESH_BINARY);
        bin.convertTo(bin, CV_8U);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        for (auto& c : contours) {
            if (c.size() < 3)
                continue;

            // Score by averaging prob inside contour's bounding rect.
            cv::Rect r = cv::boundingRect(c);
            r &= cv::Rect(0, 0, prob_map.cols, prob_map.rows);
            if (r.width <= 1 || r.height <= 1)
                continue;
            cv::Scalar mean_prob = cv::mean(prob_map(r));
            if ((float)mean_prob[0] < box_thresh)
                continue;

            cv::RotatedRect rr = cv::minAreaRect(c);
            cv::Point2f pts[4];
            rr.points(pts);

            // Unclip approximation: scale rect about center.
            cv::Point2f center = rr.center;
            for (int i = 0; i < 4; ++i) {
                pts[i] = center + (pts[i] - center) * unclip_ratio;
            }

            std::array<cv::Point2f, 4> quad;
            for (int i = 0; i < 4; ++i)
                quad[i] = pts[i];
            quads.push_back(quad);
        }
        return quads;
    }

    // English:
    // Order quad points consistently: [tl,tr,br,bl]
    static std::array<cv::Point2f, 4> OrderQuad(const std::array<cv::Point2f, 4>& q) {
        std::array<cv::Point2f, 4> out;
        std::vector<cv::Point2f> pts(q.begin(), q.end());
        auto sum = [](const cv::Point2f& p) { return p.x + p.y; };
        auto diff = [](const cv::Point2f& p) { return p.x - p.y; };
        out[0] = *std::min_element(pts.begin(), pts.end(), [&](auto& a, auto& b) { return sum(a) < sum(b); }); // tl
        out[2] = *std::max_element(pts.begin(), pts.end(), [&](auto& a, auto& b) { return sum(a) < sum(b); }); // br
        out[1] = *std::max_element(pts.begin(), pts.end(), [&](auto& a, auto& b) { return diff(a) < diff(b); }); // tr
        out[3] = *std::min_element(pts.begin(), pts.end(), [&](auto& a, auto& b) { return diff(a) < diff(b); }); // bl
        return out;
    }

    // English:
    // Locate Paddle inference model/params inside a model directory.
    static bool FindPaddleModelFiles(const std::string& model_dir, std::string& model_file, std::string& params_file) {
        model_file.clear();
        params_file.clear();

        fs::path dir(model_dir);
        if (!fs::exists(dir) || !fs::is_directory(dir))
            return false;

        // English comment:
        // Prefer PP-OCRv5 static_infer export: inference.json + inference.pdiparams.
        // Fallback to legacy exports: model.pdmodel + model.pdiparams, or __model__ + __params__.
        {
            auto status_or_paths = Utility::GetModelPaths(model_dir);
            if (status_or_paths.ok()) {
                const auto& mp = status_or_paths.value();
                auto it = mp.find("paddle");
                if (it != mp.end()) {
                    model_file = it->second.first;
                    params_file = it->second.second;
                    if (!model_file.empty() && !params_file.empty())
                        return true;
                }
            }
        }

        // Very old naming
        fs::path __model__ = dir / "__model__";
        fs::path __params__ = dir / "__params__";
        if (fs::exists(__model__) && fs::exists(__params__)) {
            model_file = __model__.string();
            params_file = __params__.string();
            return true;
        }

        return false;
    }

    static std::shared_ptr<paddle_infer::Predictor> CreateOnePredictor(
        const std::string& model_dir,
        bool use_gpu,
        int gpu_id,
        int cpu_threads,
        bool enable_mkldnn,
        std::string& err)
    {
        std::string model_file, params_file;
        if (!FindPaddleModelFiles(model_dir, model_file, params_file)) {
            err = "Cannot find Paddle model files in dir: " + model_dir;
            return nullptr;
        }

        paddle_infer::Config config;
        config.SetModel(model_file, params_file);

        // English:
        // Basic runtime configs.
        config.SwitchIrOptim(true);
        config.EnableMemoryOptim();
        config.DisableGlogInfo();

        if (use_gpu) {
            // English:
            // GPU memory pool in MB, adjust if needed.
            config.EnableUseGpu(512, gpu_id);
        }
        else {
            config.DisableGpu();
            config.SetCpuMathLibraryNumThreads(cpu_threads);
            if (enable_mkldnn) {
                config.EnableMKLDNN();
            }
        }

        try {
            return paddle_infer::CreatePredictor(config);
        }
        catch (const std::exception& e) {
            err = std::string("CreatePredictor failed: ") + e.what();
            return nullptr;
        }
    }

    // English:
    // A very small angle classifier wrapper:
    // - Resize to (cls_w, cls_h)
    // - Output assumed [1,2] or [1,2,1,1]
    // - label 1 => rotate 180
    static bool RunAngleClsIfEnabled(PaddleInferImpl* impl, const cv::Mat& crop_bgr, bool enabled, cv::Mat& out_bgr) {
        out_bgr = crop_bgr;
        if (!enabled || !impl || !impl->cls)
            return true;

        // Typical PaddleOCR cls input: [1,3,48,192]
        int cls_h = 48;
        int cls_w = 192;

        cv::Mat img;
        cv::resize(crop_bgr, img, cv::Size(cls_w, cls_h), 0, 0, cv::INTER_LINEAR);
        img.convertTo(img, CV_32FC3, 1.0 / 255.0);

        std::vector<float> nchw;
        HwcToNchwRgb(img, nchw);

        auto in = impl->cls->GetInputHandle(impl->cls_in);
        in->Reshape({ 1, 3, cls_h, cls_w });
        in->CopyFromCpu(nchw.data());

        impl->cls->Run();

        auto out = impl->cls->GetOutputHandle(impl->cls_out);
        std::vector<int> os = out->shape();
        int64_t n = 1;
        for (auto v : os) n *= v;

        std::vector<float> buf((size_t)n);
        out->CopyToCpu(buf.data());

        // Find argmax on last dim=2 (best-effort)
        int label = 0;
        if (n >= 2) {
            label = (buf[1] > buf[0]) ? 1 : 0;
        }

        if (label == 1) {
            cv::Mat rot;
            cv::rotate(crop_bgr, rot, cv::ROTATE_180);
            out_bgr = rot;
        }
        return true;
    }

} // namespace

// ------------------------
// PaddleOcr public methods
// ------------------------

PaddleOcr::PaddleOcr(const Config& cfg) {
    Init(cfg);
}

bool PaddleOcr::Init(const Config& cfg) {
    cfg_ = cfg;

    // English comment:
    // Try to override DET postprocess thresholds from det model yaml (static_infer style).
    // If yaml is missing or keys are absent, we keep user-provided defaults.
    if (!cfg_.det_model_dir.empty()) {
        try {
            YamlConfig det_cfg(cfg_.det_model_dir);
            auto t = det_cfg.GetFloat("PostProcess.thresh", cfg_.det_db_thresh);
            if (t.ok()) cfg_.det_db_thresh = t.value();
            auto bt = det_cfg.GetFloat("PostProcess.box_thresh", cfg_.det_db_box_thresh);
            if (bt.ok()) cfg_.det_db_box_thresh = bt.value();
            auto ur = det_cfg.GetFloat("PostProcess.unclip_ratio", cfg_.det_db_unclip_ratio);
            if (ur.ok()) cfg_.det_db_unclip_ratio = ur.value();
        }
        catch (...) {
            // Ignore yaml errors and continue with defaults.
        }
    }

    DestroyPredictors();
    std::string err;
    if (!CreatePredictors(err)) {
        inited_ = false;
        return false;
    }
    inited_ = true;
    return true;
}

bool PaddleOcr::CreatePredictors(std::string& err) {
    err.clear();

    auto impl = std::make_unique<PaddleInferImpl>();

    impl->det = CreateOnePredictor(cfg_.det_model_dir, cfg_.use_gpu, cfg_.gpu_id, cfg_.cpu_threads, cfg_.enable_mkldnn, err);
    if (!impl->det) return false;

    impl->rec = CreateOnePredictor(cfg_.rec_model_dir, cfg_.use_gpu, cfg_.gpu_id, cfg_.cpu_threads, cfg_.enable_mkldnn, err);
    if (!impl->rec) return false;

    if (cfg_.use_angle_cls && !cfg_.cls_model_dir.empty()) {
        std::string err2;
        impl->cls = CreateOnePredictor(cfg_.cls_model_dir, cfg_.use_gpu, cfg_.gpu_id, cfg_.cpu_threads, cfg_.enable_mkldnn, err2);
        if (!impl->cls) {
            // English:
            // If cls fails, we can fallback to no cls to keep OCR usable.
            impl->cls.reset();
        }
    }

    // English:
    // Use first input/output name to be robust across exported models.
    {
        auto in_names = impl->det->GetInputNames();
        auto out_names = impl->det->GetOutputNames();
        if (in_names.empty() || out_names.empty()) {
            err = "DET predictor has empty IO names.";
            return false;
        }
        impl->det_in = in_names[0];
        impl->det_out = out_names[0];
    }
    {
        auto in_names = impl->rec->GetInputNames();
        auto out_names = impl->rec->GetOutputNames();
        if (in_names.empty() || out_names.empty()) {
            err = "REC predictor has empty IO names.";
            return false;
        }
        impl->rec_in = in_names[0];
        impl->rec_out = out_names[0];
    }
    if (impl->cls) {
        auto in_names = impl->cls->GetInputNames();
        auto out_names = impl->cls->GetOutputNames();
        if (!in_names.empty() && !out_names.empty()) {
            impl->cls_in = in_names[0];
            impl->cls_out = out_names[0];
        }
        else {
            impl->cls.reset();
        }
    }

    {
        std::string dict_path, derr;
        if (!ResolveRecDictPath(cfg_.rec_dict_path, cfg_.rec_model_dir, dict_path, derr)) {
            err = derr;
            return false;
        }
        impl->dict = LoadDictTxt(dict_path);
    }
    if (impl->dict.empty()) {
        err = "Failed to load rec_dict_path or dict empty: " + cfg_.rec_dict_path;
        return false;
    }

    det_predictor_ = impl.release();
    rec_predictor_ = det_predictor_; // share the same impl
    cls_predictor_ = det_predictor_;
    return true;
}

void PaddleOcr::DestroyPredictors() {
    if (det_predictor_) {
        auto* impl = reinterpret_cast<PaddleInferImpl*>(det_predictor_);
        delete impl;
    }
    det_predictor_ = nullptr;
    rec_predictor_ = nullptr;
    cls_predictor_ = nullptr;
}

bool PaddleOcr::Run(const BgraFrame& frame, std::vector<OcrItem>& out_items, std::string& out_err) {
    out_items.clear();
    out_err.clear();
    if (!inited_ || !det_predictor_) {
        out_err = "PaddleOcr not initialized";
        return false;
    }
    if (frame.data.empty() || frame.width <= 0 || frame.height <= 0) {
        out_err = "Empty frame";
        return false;
    }

    auto* impl = reinterpret_cast<PaddleInferImpl*>(det_predictor_);

    cv::Mat bgra(frame.height, frame.width, CV_8UC4, (void*)frame.data.data(), frame.stride);
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

    // ---- det preprocess ----
    int max_side = std::max(bgr.cols, bgr.rows);
    float scale = 1.0f;
    if (max_side > cfg_.max_side_len) {
        scale = (float)cfg_.max_side_len / (float)max_side;
    }

    int det_w = (int)std::round(bgr.cols * scale);
    int det_h = (int)std::round(bgr.rows * scale);
    det_w = (det_w + 31) / 32 * 32;
    det_h = (det_h + 31) / 32 * 32;

    cv::Mat det_img;
    cv::resize(bgr, det_img, cv::Size(det_w, det_h), 0, 0, cv::INTER_LINEAR);
    det_img.convertTo(det_img, CV_32FC3, 1.0 / 255.0);

    std::vector<float> det_nchw;
    HwcToNchwRgb(det_img, det_nchw);

    // ---- det run ----
    try {
        auto det_in = impl->det->GetInputHandle(impl->det_in);
        det_in->Reshape({ 1, 3, det_h, det_w });
        det_in->CopyFromCpu(det_nchw.data());
        impl->det->Run();
    }
    catch (const std::exception& e) {
        out_err = std::string("DET Run failed: ") + e.what();
        return false;
    }

    // ---- det output ----
    cv::Mat prob_map;
    try {
        auto det_out = impl->det->GetOutputHandle(impl->det_out);
        std::vector<int> ds = det_out->shape();

        int out_h = 0, out_w = 0;
        if (ds.size() == 4) {          // [1,1,H,W]
            out_h = (int)ds[2];
            out_w = (int)ds[3];
        }
        else if (ds.size() == 3) {   // [1,H,W]
            out_h = (int)ds[1];
            out_w = (int)ds[2];
        }
        else {
            out_err = "Unexpected DET output shape";
            return false;
        }

        int64_t n = 1;
        for (auto v : ds) n *= v;
        std::vector<float> det_buf((size_t)n);
        det_out->CopyToCpu(det_buf.data());

        // English:
        // For [1,1,H,W], prob starts at det_buf[0].
        // For [1,H,W], prob starts at det_buf[0].
        prob_map = cv::Mat(out_h, out_w, CV_32F, det_buf.data()).clone();
    }
    catch (const std::exception& e) {
        out_err = std::string("DET output parse failed: ") + e.what();
        return false;
    }

    auto quads = ExtractQuadsFromDbProb(prob_map, cfg_.det_db_thresh, cfg_.det_db_box_thresh, cfg_.det_db_unclip_ratio);
    if (quads.empty()) {
        return true; // no text
    }

    // ---- rec per quad ----
    for (auto& q : quads) {
        auto oq = OrderQuad(q);

        // Scale back to original image coordinates
        std::array<cv::Point2f, 4> src;
        for (int i = 0; i < 4; ++i) {
            src[i].x = oq[i].x / scale;
            src[i].y = oq[i].y / scale;
            src[i].x = std::clamp(src[i].x, 0.0f, (float)(frame.width - 1));
            src[i].y = std::clamp(src[i].y, 0.0f, (float)(frame.height - 1));
        }

        float w = std::hypot(src[0].x - src[1].x, src[0].y - src[1].y);
        float h = std::hypot(src[0].x - src[3].x, src[0].y - src[3].y);
        if (w < 4.0f || h < 4.0f)
            continue;

        // Determine rec input size from predictor input shape (best-effort)
        int rec_H = 48;
        int rec_W_max = 320;
        try {
            auto rec_in = impl->rec->GetInputHandle(impl->rec_in);
            auto rs = rec_in->shape(); // might be empty before reshape in some versions
            if (rs.size() == 4) {
                if (rs[2] > 0) rec_H = (int)rs[2];
                if (rs[3] > 0) rec_W_max = (int)rs[3];
            }
        }
        catch (...) {
            // keep defaults
        }

        int crop_w = std::max(16, std::min((int)std::round(w * rec_H / h), rec_W_max));
        cv::Point2f dst[4] = { {0,0}, {(float)crop_w,0}, {(float)crop_w,(float)rec_H}, {0,(float)rec_H} };
        cv::Point2f src_pts[4] = { src[0], src[1], src[2], src[3] };
        cv::Mat M = cv::getPerspectiveTransform(src_pts, dst);

        cv::Mat crop;
        cv::warpPerspective(bgr, crop, M, cv::Size(crop_w, rec_H), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // Angle cls (optional)
        cv::Mat crop2 = crop;
        RunAngleClsIfEnabled(impl, crop, (cfg_.use_angle_cls && impl->cls), crop2);

        crop2.convertTo(crop2, CV_32FC3, 1.0 / 255.0);

        std::vector<float> rec_nchw;
        HwcToNchwRgb(crop2, rec_nchw);

        // ---- rec run ----
        try {
            auto rec_in = impl->rec->GetInputHandle(impl->rec_in);
            rec_in->Reshape({ 1, 3, rec_H, crop_w });
            rec_in->CopyFromCpu(rec_nchw.data());
            impl->rec->Run();
        }
        catch (...) {
            continue;
        }

        // ---- rec output ----
        std::string text;
        float score = 0.0f;
        try {
            auto rec_out = impl->rec->GetOutputHandle(impl->rec_out);
            auto rs = rec_out->shape();

            int T = 0;
            int C = 0;
            if (rs.size() == 3) {          // [1,T,C]
                T = (int)rs[1];
                C = (int)rs[2];
            }
            else if (rs.size() == 2) {   // [T,C]
                T = (int)rs[0];
                C = (int)rs[1];
            }
            else if (rs.size() == 4) {   // [1,T,1,C] or similar
                T = (int)rs[1];
                C = (int)rs[3];
            }
            else {
                continue;
            }

            int64_t n = 1;
            for (auto v : rs) n *= v;
            std::vector<float> rec_buf((size_t)n);
            rec_out->CopyToCpu(rec_buf.data());

            // English:
            // Flatten to [T,C] view.
            const float* p = rec_buf.data();
            // If shape is [1,T,C], p already starts at first element.
            // If shape is [1,T,1,C], still contiguous; treating as [T,C] works if the middle dim is 1.
            text = CtcGreedyDecode(p, T, C, impl->dict, &score);
        }
        catch (...) {
            continue;
        }

        if ((int)text.size() < cfg_.min_text_len)
            continue;

        OcrItem item;
        item.text = text;
        item.score = score;
        item.poly.resize(8);
        item.poly[0] = src[0].x; item.poly[1] = src[0].y;
        item.poly[2] = src[1].x; item.poly[3] = src[1].y;
        item.poly[4] = src[2].x; item.poly[5] = src[2].y;
        item.poly[6] = src[3].x; item.poly[7] = src[3].y;
        out_items.push_back(std::move(item));
    }

    std::sort(out_items.begin(), out_items.end(), BoxLess);
    return true;
}

bool PaddleOcr::RunText(const BgraFrame& frame, std::string& out_text_topdown, std::string& out_err) {
    out_text_topdown.clear();
    std::vector<OcrItem> items;
    if (!Run(frame, items, out_err))
        return false;

    std::string text;
    for (const auto& it : items) {
        if (!it.text.empty()) {
            if (!text.empty())
                text += "\n";
            text += it.text;
        }
    }
    out_text_topdown = text;
    return true;
}
