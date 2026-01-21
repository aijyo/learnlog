#include "paddle_ocr.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "src/api/pipelines/ocr.h"
#include "src/utils/args.h"

#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;


static std::string TickString() {
    // English comment: simple unique-ish suffix for temp filenames
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return std::to_string((long long)now);
}

static bool EnsureDir(const fs::path& p) {
    std::error_code ec;
    if (fs::exists(p, ec)) return true;
    return fs::create_directories(p, ec);
}

static bool ValidateFrame(const BgraFrame& f)
{
    if (f.width <= 0 || f.height <= 0 || f.stride <= 0) return false;
    return f.data.size() >= (size_t)f.stride * f.height;
}

static cv::Mat BgraToBgrMat(const BgraFrame& frame)
{
    // English:
    // Wrap BGRA buffer, convert to BGR, then clone to own memory.
    cv::Mat bgra(frame.height, frame.width, CV_8UC4,
        (void*)frame.data.data(), frame.stride);

    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    return bgr.clone();
}

static void SortTopDownLeftRight(std::vector<PaddleOcr::Line>& lines) {
    std::sort(lines.begin(), lines.end(), [](const auto& a, const auto& b) {
        if (a.y1 != b.y1) return a.y1 < b.y1;
        return a.x1 < b.x1;
        });
}

struct PaddleOcr::Impl {
    bool inited = false;
    PaddleOCRParams ocr_params;
    fs::path tmp_root;

    // English:
    // Parse SaveToJson output into Line list.
    bool ParseOcrJson(const fs::path& json_path,
        std::vector<PaddleOcr::Line>& out,
        std::string& err)
    {
        out.clear();

        std::ifstream ifs(json_path);
        if (!ifs.is_open()) {
            err = "Failed to open json: " + json_path.string();
            return false;
        }

        json j;
        try {
            ifs >> j;
        }
        catch (const std::exception& e) {
            err = std::string("JSON parse error: ") + e.what();
            return false;
        }

        // English:
        // Support both:
        // 1) array root
        // 2) { "results": [...] } or { "data": [...] }
        json arr;
        if (j.is_array()) {
            arr = j;
        }
        else if (j.is_object()) {
            if (j.contains("results") && j["results"].is_array())
                arr = j["results"];
            else if (j.contains("data") && j["data"].is_array())
                arr = j["data"];
            else {
                err = "Unsupported JSON schema (no results/data array)";
                return false;
            }
        }
        else {
            err = "Unsupported JSON root type";
            return false;
        }

        for (auto& item : arr) {
            if (!item.is_object()) continue;

            PaddleOcr::Line line;

            // text
            if (item.contains("text"))
                line.text = item["text"].get<std::string>();
            else if (item.contains("label"))
                line.text = item["label"].get<std::string>();
            else
                continue;

            // confidence
            double score = 0.0;
            if (item.contains("score")) score = item["score"].get<double>();
            else if (item.contains("confidence")) score = item["confidence"].get<double>();

            line.confidence = (score <= 1.0) ? float(score * 100.0) : float(score);

            // box / polygon / points
            const json* box = nullptr;
            if (item.contains("box")) box = &item["box"];
            else if (item.contains("points")) box = &item["points"];
            else if (item.contains("polygon")) box = &item["polygon"];

            if (box && box->is_array()) {
                int minx = INT32_MAX, miny = INT32_MAX;
                int maxx = INT32_MIN, maxy = INT32_MIN;

                for (auto& pt : *box) {
                    if (pt.is_array() && pt.size() >= 2) {
                        int x = pt[0].get<int>();
                        int y = pt[1].get<int>();
                        minx = std::min(minx, x);
                        miny = std::min(miny, y);
                        maxx = std::max(maxx, x);
                        maxy = std::max(maxy, y);
                    }
                }

                if (minx != INT32_MAX) {
                    line.x1 = minx; line.y1 = miny;
                    line.x2 = maxx; line.y2 = maxy;
                }
            }

            out.push_back(std::move(line));
        }

        // English:
        // Sort top-to-bottom, then left-to-right
        std::sort(out.begin(), out.end(), [](auto& a, auto& b) {
            if (a.y1 != b.y1) return a.y1 < b.y1;
            return a.x1 < b.x1;
            });

        return true;
    }
};

PaddleOcr::PaddleOcr() : impl_(new Impl()) {}
PaddleOcr::~PaddleOcr() = default;

bool PaddleOcr::Init(const std::string& det_dir,
    const std::string& rec_dir,
    const std::string& cls_dir,
    const std::string& lang,
    const std::string& device,
    const std::string& precision,
    bool enable_mkldnn,
    int cpu_threads)
{
    last_error_.clear();

    if (det_dir.empty() || rec_dir.empty()) {
        last_error_ = "det_dir or rec_dir is empty.";
        return false;
    }

    impl_->ocr_params = PaddleOCRParams{};
    impl_->ocr_params.text_detection_model_dir = det_dir;
    impl_->ocr_params.text_recognition_model_dir = rec_dir;

    // English comment: optional
    if (!cls_dir.empty()) {
        impl_->ocr_params.textline_orientation_model_dir = cls_dir;
        impl_->ocr_params.use_textline_orientation = true;
    }
    else {
        impl_->ocr_params.use_textline_orientation = false;
    }

    impl_->ocr_params.lang = lang;
    impl_->ocr_params.device = device;
    impl_->ocr_params.precision = precision;
    impl_->ocr_params.enable_mkldnn = enable_mkldnn;
    impl_->ocr_params.cpu_threads = cpu_threads;

    // English comment: create temp root folder
    impl_->tmp_root = fs::temp_directory_path() / "ppocr_class_tmp";
    if (!EnsureDir(impl_->tmp_root)) {
        last_error_ = "Failed to create temp folder: " + impl_->tmp_root.string();
        return false;
    }

    impl_->inited = true;
    return true;
}

void PaddleOcr::SetDetParams(int limit_side_len, const std::string& limit_type,
    float thresh, float box_thresh, float unclip_ratio)
{
    // English comment: mirror FLAGS_text_det_* in cli.cc
    impl_->ocr_params.text_det_limit_side_len = limit_side_len;
    impl_->ocr_params.text_det_limit_type = limit_type;
    impl_->ocr_params.text_det_thresh = thresh;
    impl_->ocr_params.text_det_box_thresh = box_thresh;
    impl_->ocr_params.text_det_unclip_ratio = unclip_ratio;
}

void PaddleOcr::SetRecParams(float score_thresh, int batch_size)
{
    // English comment: mirror FLAGS_text_rec_* in cli.cc
    impl_->ocr_params.text_rec_score_thresh = score_thresh;
    impl_->ocr_params.text_recognition_batch_size = batch_size;
}

bool PaddleOcr::RecognizeLinesTopDown(const BgraFrame& frame, std::vector<Line>& out_lines)
{
    last_error_.clear();
    out_lines.clear();

    if (!impl_->inited) {
        last_error_ = "PaddleOcr not initialized.";
        return false;
    }
    if (!ValidateFrame(frame)) {
        last_error_ = "Invalid BgraFrame.";
        return false;
    }

    // 1) frame -> temp png
    cv::Mat bgr = BgraToBgrMat(frame);
    if (bgr.empty()) {
        last_error_ = "Failed to convert BGRA to BGR.";
        return false;
    }

    const std::string tag = TickString();
    const fs::path img_path = impl_->tmp_root / ("in_" + tag + ".png");
    const fs::path out_dir = impl_->tmp_root / ("out_" + tag);

    if (!EnsureDir(out_dir)) {
        last_error_ = "Failed to create out dir: " + out_dir.string();
        return false;
    }

    if (!cv::imwrite(img_path.string(), bgr)) {
        last_error_ = "Failed to write temp image: " + img_path.string();
        return false;
    }

    // 2) run PaddleOCR pipeline (mirrors cli.cc: PaddleOCR(params).Predict(input_path))
    std::vector<std::unique_ptr<BaseCVResult>> outputs;
    try {
        outputs = PaddleOCR(impl_->ocr_params).Predict(img_path.string());
    }
    catch (const std::exception& e) {
        last_error_ = std::string("Predict exception: ") + e.what();
        return false;
    }
    catch (...) {
        last_error_ = "Predict exception: unknown.";
        return false;
    }

    // 3) SaveToJson, then parse json -> lines
    // English comment:
    // SaveToJson expects a save_path; cli uses FLAGS_save_path. Here we use out_dir.
    for (auto& r : outputs) {
        r->SaveToJson(out_dir.string());
    }

    // English comment:
    // We don't know exact json filename pattern; scan for the newest *.json under out_dir.
    fs::path json_file;
    bool found = false;

    for (auto& e : fs::directory_iterator(out_dir)) {
        if (!e.is_regular_file()) continue;
        if (e.path().extension() != ".json") continue;

        json_file = e.path();
        found = true;
        break;  // English: first json is enough
    }

    if (!found) {
        last_error_ = "No json output found in: " + out_dir.string();
        return false;
    }

    std::string err;
    if (!impl_->ParseOcrJson(json_file, out_lines, err)) {
        last_error_ = err;
        return false;
    }

    return true;
}

bool PaddleOcr::RecognizeTextTopDown(const BgraFrame& frame, std::string& out_text)
{
    out_text.clear();
    std::vector<Line> lines;
    if (!RecognizeLinesTopDown(frame, lines)) return false;

    std::ostringstream oss;
    for (const auto& l : lines) {
        oss << l.text << "\n";
    }
    out_text = oss.str();
    return true;
}
