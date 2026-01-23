#include "image_handler.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Include your real headers here
#include "text_detection.h"  // or "text_detection.h"
#include "text_recognition.h" // if you have, otherwise replace with your header
#include "text_handler.h"     // your processor

// -----------------------------
// Ctor
// -----------------------------

ImageHandler::ImageHandler(std::shared_ptr<TextDetection> det,
    std::shared_ptr<TextRecognition> rec,
    std::shared_ptr<TextHander> processor,
    ImageHandlerConfig cfg)
    : det_(std::move(det)),
    rec_(std::move(rec)),
    processor_(std::move(processor)),
    cfg_(std::move(cfg)) {}

// -----------------------------
// Public APIs
// -----------------------------

absl::StatusOr<OcrFrameResult> ImageHandler::ProcessPath(const std::string& image_path) {
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        return absl::InvalidArgumentError("ImageHandler::ProcessPath: failed to read image");
    }
    return ProcessMat(bgr);
}

absl::StatusOr<OcrFrameResult> ImageHandler::ProcessMat(const cv::Mat& bgr) {
    if (bgr.empty()) {
        return absl::InvalidArgumentError("ImageHandler::ProcessMat: input image is empty");
    }
    if (!rec_) {
        return absl::FailedPreconditionError("ImageHandler::ProcessMat: rec_ is null");
    }
    if (cfg_.enable_det && !det_) {
        return absl::FailedPreconditionError("ImageHandler::ProcessMat: enable_det=true but det_ is null");
    }
    if (!processor_) {
        return absl::FailedPreconditionError("ImageHandler::ProcessMat: processor_ is null");
    }
    return ProcessInternal(bgr);
}

// -----------------------------
// Pipeline
// -----------------------------

absl::StatusOr<OcrFrameResult> ImageHandler::ProcessInternal(const cv::Mat& bgr_src) {
    OcrFrameResult frame;
    frame.image_size = bgr_src.size();

    // Apply ROI if configured
    cv::Rect roi_in_src(0, 0, bgr_src.cols, bgr_src.rows);
    if (cfg_.roi.has_value()) {
        roi_in_src = ClampRect(cfg_.roi.value(), bgr_src.size());
        if (roi_in_src.width <= 0 || roi_in_src.height <= 0) {
            return absl::InvalidArgumentError("ImageHandler: ROI invalid after clamp");
        }
    }

    cv::Mat bgr_work = bgr_src(roi_in_src).clone();

    // If det disabled: recognize whole ROI as a single crop
    if (!cfg_.enable_det) {
        auto rec_r = rec_->Predict(bgr_work); // assumes your TextRecognition has Predict(Mat)
        if (!rec_r.ok()) return rec_r.status();

        //auto& rec_v = rec_r.value();
        OcrInstance one;
        one.text = rec_r->rec_text;        // adjust if your field name differs
        one.rec_score = rec_r->rec_score;  // adjust if your field name differs
        one.det_score = 1.0f;
        one.quad = {
          cv::Point(roi_in_src.x, roi_in_src.y),
          cv::Point(roi_in_src.x + roi_in_src.width - 1, roi_in_src.y),
          cv::Point(roi_in_src.x + roi_in_src.width - 1, roi_in_src.y + roi_in_src.height - 1),
          cv::Point(roi_in_src.x, roi_in_src.y + roi_in_src.height - 1),
        };
        one.bbox = roi_in_src;

        if (one.rec_score >= cfg_.rec_score_thresh) frame.items.push_back(std::move(one));
    }
    else {
        auto items = DetectAndRecognize(bgr_work, roi_in_src);
        if (!items.ok()) return items.status();
        frame.items = std::move(items.value());
    }

    if (cfg_.sort_items) {
        SortInstances(frame.items, cfg_.sort_y_tol);
    }
    if (cfg_.build_merged_text) {
        frame.merged_text = MergeTextTopDown(frame.items, cfg_.sort_y_tol);
    }

    // Run your business logic (TextHander) based on recognized text.
    // IMPORTANT: this step is where "根据文字内容，实现TextHander逻辑".
    auto st = ProcessByTextHander(frame);
    if (!st.ok()) return st;

    return frame;
}

// -----------------------------
// Detect + Recognize
// -----------------------------

absl::StatusOr<std::vector<OcrInstance>>
ImageHandler::DetectAndRecognize(const cv::Mat& bgr_work, const cv::Rect& work_roi_in_src) {
    auto det_r = det_->Predict(bgr_work); // assumes your TextDetection has Predict(Mat)
    if (!det_r.ok()) return det_r.status();

    const auto& polys = det_r->dt_polys;
    const auto& scores = det_r->dt_scores;

    std::vector<OcrInstance> out;
    out.reserve(polys.size());

    for (int i = 0; i < (int)polys.size(); ++i) {
        float det_score = (i < (int)scores.size()) ? scores[i] : 1.0f;
        if (det_score < cfg_.det_score_thresh) continue;

        const auto& quad_work = polys[i];
        if ((int)quad_work.size() < 4) continue;
        cv::Rect br = QuadBoundingRect(quad_work);
        if (br.width < cfg_.min_box_size || br.height < cfg_.min_box_size) continue;

        auto one = RecognizeOne(bgr_work, quad_work, det_score, work_roi_in_src);
        if (!one.ok()) continue;

        if (one->rec_score < cfg_.rec_score_thresh) continue;
        if (one->text.empty()) continue;

        out.push_back(std::move(one.value()));
    }

    return out;
}

absl::StatusOr<OcrInstance>
ImageHandler::RecognizeOne(const cv::Mat& bgr_work,
    const std::vector<cv::Point2f>& quad_work,
    float det_score,
    const cv::Rect& work_roi_in_src) const {
    auto crop_r = CropQuad(bgr_work, quad_work);
    if (!crop_r.ok()) return crop_r.status();

    cv::Mat crop = crop_r.value();

    // Optional preprocessing for UI text
    if (cfg_.to_gray) {
        cv::Mat gray;
        cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, crop, cv::COLOR_GRAY2BGR);
    }
    if (cfg_.adaptive_threshold) {
        cv::Mat gray, thr;
        cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
        cv::adaptiveThreshold(gray, thr, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 31, 5);
        cv::cvtColor(thr, crop, cv::COLOR_GRAY2BGR);
    }

    auto rec_r = rec_->Predict(crop); // assumes Predict(Mat) -> {text, score}
    if (!rec_r.ok()) return rec_r.status();

    // Convert quad from work coords to source coords
    std::vector<cv::Point2f> quad_src = OffsetQuad(quad_work, work_roi_in_src.x, work_roi_in_src.y);
    cv::Rect bbox_src = QuadBoundingRect(quad_src);

    OcrInstance inst;
    inst.text = rec_r->rec_text;       // adjust if needed
    inst.rec_score = rec_r->rec_score; // adjust if needed
    inst.det_score = det_score;
    inst.quad = std::move(quad_src);
    inst.bbox = bbox_src;
    return inst;
}

// -----------------------------
// Crop quad
// -----------------------------

absl::StatusOr<cv::Mat>
ImageHandler::CropQuad(const cv::Mat& bgr_work, const std::vector<cv::Point2f>& quad_work) const {
    if ((int)quad_work.size() < 4) {
        return absl::InvalidArgumentError("ImageHandler::CropQuad: quad size < 4");
    }

    // Pad bounding rect
    cv::Rect br = QuadBoundingRect(quad_work);
    br.x -= cfg_.crop_padding;
    br.y -= cfg_.crop_padding;
    br.width += cfg_.crop_padding * 2;
    br.height += cfg_.crop_padding * 2;
    br = ClampRect(br, bgr_work.size());
    if (br.width <= 0 || br.height <= 0) {
        return absl::InternalError("ImageHandler::CropQuad: invalid rect after clamp");
    }

    if (!cfg_.rotate_crop) {
        return bgr_work(br).clone();
    }

    // Warp perspective
    std::vector<cv::Point2f> src(4);
    for (int i = 0; i < 4; ++i) src[i] = cv::Point2f((float)quad_work[i].x, (float)quad_work[i].y);

    auto dist = [](const cv::Point2f& a, const cv::Point2f& b) {
        float dx = a.x - b.x, dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
        };

    float w1 = dist(src[0], src[1]);
    float w2 = dist(src[2], src[3]);
    float h1 = dist(src[0], src[3]);
    float h2 = dist(src[1], src[2]);

    int tw = std::max(1, (int)std::round(std::max(w1, w2)));
    int th = std::max(1, (int)std::round(std::max(h1, h2)));

    if (cfg_.max_crop_side > 0) {
        int mx = std::max(tw, th);
        if (mx > cfg_.max_crop_side) {
            float s = (float)cfg_.max_crop_side / (float)mx;
            tw = std::max(1, (int)std::round(tw * s));
            th = std::max(1, (int)std::round(th * s));
        }
    }

    std::vector<cv::Point2f> dst(4);
    dst[0] = cv::Point2f(0.f, 0.f);
    dst[1] = cv::Point2f((float)(tw - 1), 0.f);
    dst[2] = cv::Point2f((float)(tw - 1), (float)(th - 1));
    dst[3] = cv::Point2f(0.f, (float)(th - 1));

    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat warped;
    cv::warpPerspective(bgr_work, warped, M, cv::Size(tw, th), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return warped;
}

// -----------------------------
// TextHander step
// -----------------------------

absl::Status ImageHandler::ProcessByTextHander(OcrFrameResult& frame) const {
    // This is the only integration point you need to adapt to your TextHander API.
    // The intention:
    //  1) provide per-instance texts and positions
    //  2) allow processor to generate business outputs (e.g., castbar parsing, target name, etc.)
    //
    // Typical options:
    //  - processor_->Process(frame.items, &frame)  (in-place enrich)
    //  - processor_->Handle(frame.merged_text)     (text-only)
    //
    // TODO: Adjust function name and parameters to match your TextHander.
    //
    // Example (text-only):
    //   return processor_->Process(frame.merged_text);
    //
    // Example (structured):
    //   return processor_->Process(frame.items, &frame.merged_text);

    // ---- DEFAULT SAFE FALLBACK ----
    // If your processor currently doesn't expose a suitable method yet,
    // return OK to keep pipeline working.
    //
    // Replace this line with your real call.
    //(void)frame;
    processor_->ProcessByTextHander(frame);
    return absl::OkStatus();
}

// -----------------------------
// Helpers
// -----------------------------

cv::Rect ImageHandler::ClampRect(const cv::Rect& r, const cv::Size& s) {
    int x = std::max(0, r.x);
    int y = std::max(0, r.y);
    int w = std::min(r.width, s.width - x);
    int h = std::min(r.height, s.height - y);
    if (w < 0) w = 0;
    if (h < 0) h = 0;
    return cv::Rect(x, y, w, h);
}

cv::Rect ImageHandler::QuadBoundingRect(const std::vector<cv::Point2f>& quad) {
    if (quad.empty()) return cv::Rect();
    float minx = quad[0].x, maxx = quad[0].x;
    float miny = quad[0].y, maxy = quad[0].y;
    for (const auto& p : quad) {
        minx = std::min(minx, p.x);
        maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y);
        maxy = std::max(maxy, p.y);
    }
    return cv::Rect(minx, miny, std::max<float>(0, maxx - minx + 1), std::max<float>(0, maxy - miny + 1));
}

std::vector<cv::Point2f> ImageHandler::OffsetQuad(const std::vector<cv::Point2f>& quad, int dx, int dy) {
    std::vector<cv::Point2f> out = quad;
    for (auto& p : out) {
        p.x += dx;
        p.y += dy;
    }
    return out;
}

void ImageHandler::SortInstances(std::vector<OcrInstance>& items, int y_tol) {
    std::sort(items.begin(), items.end(), [y_tol](const OcrInstance& a, const OcrInstance& b) {
        int ay = a.bbox.y;
        int by = b.bbox.y;
        if (std::abs(ay - by) <= y_tol) return a.bbox.x < b.bbox.x;
        return ay < by;
        });
}

std::string ImageHandler::MergeTextTopDown(const std::vector<OcrInstance>& items, int y_tol) {
    // Merge items into multiple lines by y clustering.
    // This is helpful for feeding text-only processors.
    std::string out;
    int last_y = std::numeric_limits<int>::min();
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& it = items[i];
        if (it.text.empty()) continue;

        if (!out.empty()) {
            if (std::abs(it.bbox.y - last_y) > y_tol) out.push_back('\n');
            else out.push_back(' ');
        }
        out += it.text;
        last_y = it.bbox.y;
    }
    return out;
}
