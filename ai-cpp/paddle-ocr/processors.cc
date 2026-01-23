#include "processors.h"

#include <stdexcept>

#include "utils/utility.h"

DetResizeForTest::DetResizeForTest(const DetResizeForTestParam& params) {
    if (params.input_shape.has_value()) {
        input_shape_ = params.input_shape.value();
        resize_type_ = 3;
    }
    else if (params.image_shape.has_value()) {
        image_shape_ = params.image_shape.value();
        resize_type_ = 1;
        if (params.keep_ratio.has_value()) {
            keep_ratio_ = params.keep_ratio.value();
        }
    }
    else if (params.limit_side_len.has_value()) {
        limit_side_len_ = params.limit_side_len.value();
        limit_type_ = params.limit_type.value_or("min");
    }
    else if (params.resize_long.has_value()) {
        resize_type_ = 2;
        resize_long_ = params.resize_long.value_or(960);
    }
    else {
        limit_side_len_ = 736;
        limit_type_ = "min";
    }
    if (params.max_side_limit.has_value()) {
        max_side_limit_ = params.max_side_limit.value();
    }
}

absl::StatusOr<std::vector<cv::Mat>>
DetResizeForTest::Apply(std::vector<cv::Mat>& input,
    const void* param_ptr) const {
    if (input.empty()) {
        return absl::InvalidArgumentError("Input image vector is empty.");
    }
    std::vector<cv::Mat> results;
    if (param_ptr != nullptr) {
        const DetResizeForTestParam* param =
            static_cast<const DetResizeForTestParam*>(param_ptr);
        for (const auto& img : input) {
            auto res = Resize(
                img,
                param->limit_side_len.has_value() ? param->limit_side_len.value()
                : limit_side_len_,
                param->limit_type.has_value() ? param->limit_type.value()
                : limit_type_,
                param->max_side_limit.has_value() ? param->max_side_limit.value()
                : max_side_limit_);
            if (!res.ok())
                return res.status();
            results.push_back(res.value());
        }
    }
    else {
        for (const auto& img : input) {
            auto res = Resize(img, limit_side_len_, limit_type_, max_side_limit_);
            if (!res.ok())
                return res.status();
            results.push_back(res.value());
        }
    }
    return results;
}

absl::StatusOr<cv::Mat> DetResizeForTest::Resize(const cv::Mat& img,
    int limit_side_len,
    const std::string& limit_type,
    int max_side_limit) const {
    int src_h = img.rows;
    int src_w = img.cols;
    if (src_h + src_w < 64) {
        cv::Mat padded = ImagePadding(img);
        src_h = padded.rows;
        src_w = padded.cols;
        return Resize(padded, limit_side_len, limit_type, max_side_limit);
    }

    switch (resize_type_) {
    case 0:
        return ResizeImageType0(img, limit_side_len, limit_type, max_side_limit);
    case 1:
        return ResizeImageType1(img);
    case 2:
        return ResizeImageType2(img);
    case 3:
        return ResizeImageType3(img);
    default:
        return absl::InvalidArgumentError("Unknown resize_type: " +
            std::to_string(resize_type_));
    }
}

cv::Mat DetResizeForTest::ImagePadding(const cv::Mat& img, int value) const {
    int h = img.rows, w = img.cols, c = img.channels();
    int pad_h = std::max(32, h);
    int pad_w = std::max(32, w);
    cv::Mat im_pad = cv::Mat::zeros(pad_h, pad_w, img.type());
    im_pad.setTo(cv::Scalar::all(value));
    img.copyTo(im_pad(cv::Rect(0, 0, w, h)));
    return im_pad;
}

absl::StatusOr<cv::Mat>
DetResizeForTest::ResizeImageType0(const cv::Mat& img, int limit_side_len,
    const std::string& limit_type,
    int max_side_limit) const {
    int h = img.rows, w = img.cols;
    float ratio = 1.f;
    if (limit_type == "max") {
        if (std::max(h, w) > limit_side_len)
            ratio = float(limit_side_len) / std::max(h, w);
    }
    else if (limit_type == "min") {
        if (std::min(h, w) < limit_side_len)
            ratio = float(limit_side_len) / std::min(h, w);
    }
    else if (limit_type == "resize_long") {
        ratio = float(limit_side_len) / std::max(h, w);
    }
    else {
        return absl::InvalidArgumentError("Not supported limit_type: " +
            limit_type);
    }
    int resize_h = int(h * ratio);
    int resize_w = int(w * ratio);

    if (std::max(resize_h, resize_w) > max_side_limit) {
        ratio = float(max_side_limit) / std::max(resize_h, resize_w);
        resize_h = int(resize_h * ratio);
        resize_w = int(resize_w * ratio);
    }
    resize_h = std::max(int(std::round(resize_h / 32.0) * 32), 32);
    resize_w = std::max(int(std::round(resize_w / 32.0) * 32), 32);

    if (resize_h == h && resize_w == w)
        return img;
    if (resize_h <= 0 || resize_w <= 0)
        return absl::InvalidArgumentError("resize_w/h <= 0");
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h));
    return resized;
}

absl::StatusOr<cv::Mat>
DetResizeForTest::ResizeImageType1(const cv::Mat& img) const {
    int resize_h = image_shape_[0];
    int resize_w = image_shape_[1];
    int ori_h = img.rows, ori_w = img.cols;
    if (keep_ratio_) {
        resize_w = int(ori_w * (float(resize_h) / ori_h));
        int N = int(std::ceil(resize_w / 32.0));
        resize_w = N * 32;
    }
    if (resize_h == ori_h && resize_w == ori_w)
        return img;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h));
    return resized;
}

absl::StatusOr<cv::Mat>
DetResizeForTest::ResizeImageType2(const cv::Mat& img) const {
    int h = img.rows, w = img.cols;
    int resize_h = h, resize_w = w;
    float ratio;
    if (resize_h > resize_w)
        ratio = float(resize_long_) / resize_h;
    else
        ratio = float(resize_long_) / resize_w;

    resize_h = int(resize_h * ratio);
    resize_w = int(resize_w * ratio);

    int max_stride = 128;
    resize_h = ((resize_h + max_stride - 1) / max_stride) * max_stride;
    resize_w = ((resize_w + max_stride - 1) / max_stride) * max_stride;

    if (resize_h == h && resize_w == w)
        return img;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h));
    return resized;
}

absl::StatusOr<cv::Mat>
DetResizeForTest::ResizeImageType3(const cv::Mat& img) const {
    if (input_shape_.size() != INPUTSHAPE)
        return absl::InvalidArgumentError("input_shape not set for type " +
            std::to_string(INPUTSHAPE));
    int resize_h = input_shape_[1];
    int resize_w = input_shape_[2];
    int ori_h = img.rows, ori_w = img.cols;
    if (resize_h == ori_h && resize_w == ori_w)
        return img;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resize_w, resize_h));
    return resized;
}


#include <cmath>
#include <algorithm>

// -----------------------------
// DBPostProcess (simplified implementation)
// -----------------------------

DBPostProcess::DBPostProcess(const DBPostProcessParams& params)
    : thresh_(params.thresh.value_or(0.3f)),
    box_thresh_(params.box_thresh.value_or(0.7f)),
    unclip_ratio_(params.unclip_ratio.value_or(2.0f)),
    max_candidates_(params.max_candidates),
    min_size_(params.min_size),
    use_dilation_(params.use_dilation),
    score_mode_(params.score_mode),
    box_type_(params.box_type) {
    // NOTE: Lite version only supports quad boxes.
    if (score_mode_ != "slow" && score_mode_ != "fast") score_mode_ = "fast";
    if (box_type_ != "quad") box_type_ = "quad";
}

absl::StatusOr<std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::operator()(const cv::Mat& preds,
    const std::vector<int>& img_shapes,
    std::optional<float> thresh,
    std::optional<float> box_thresh,
    std::optional<float> unclip_ratio) const {
    if (img_shapes.size() < 2) {
        return absl::InvalidArgumentError("DBPostProcess: img_shapes must be [src_h, src_w]");
    }
    return Process(preds, img_shapes,
        thresh.value_or(thresh_),
        box_thresh.value_or(box_thresh_),
        unclip_ratio.value_or(unclip_ratio_));
}

absl::StatusOr<std::vector<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>>
DBPostProcess::Apply(const cv::Mat& preds, const std::vector<int>& img_shapes,
    std::optional<float> thresh,
    std::optional<float> box_thresh,
    std::optional<float> unclip_ratio) const {
    std::vector<
        std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
        db_result = {};

    auto preds_batch = Utility::SplitBatch(preds);

    if (!preds_batch.ok()) {
        return preds_batch.status();
    }
    for (const auto& pred : preds_batch.value()) {
        auto result = Process(pred, img_shapes, thresh.value_or(thresh_),
            box_thresh.value_or(box_thresh_),
            unclip_ratio.value_or(unclip_ratio_));

        if (!result.ok()) {
            return result.status();
        }
        db_result.push_back(result.value());
    }

    return db_result;
}

absl::StatusOr<std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::Process(const cv::Mat& pred, const std::vector<int>& img_shape,
    float thresh, float box_thresh, float unclip_ratio) const {
    if (pred.empty()) {
        return absl::InvalidArgumentError("DBPostProcess::Process: pred is empty");
    }
    if (img_shape.size() < 2) {
        return absl::InvalidArgumentError("DBPostProcess::Process: img_shape must be [src_h, src_w]");
    }

    // Accept pred in shape (H,W) or (1,1,H,W)/(1,H,W). We reshape to 2D.
    cv::Mat pred_single = pred;
    if (pred_single.dims > 2) {
        const int H = pred_single.size[pred_single.dims - 2];
        const int W = pred_single.size[pred_single.dims - 1];
        pred_single = pred_single.reshape(1, H);
        if (pred_single.cols != W) {
            // reshape(1, H) should yield HxW; if not, fallback to clone then reshape by vector
            pred_single = pred.clone().reshape(1, H);
        }
    }
    if (pred_single.type() != CV_32FC1 && pred_single.type() != CV_32F) {
        return absl::InvalidArgumentError("DBPostProcess::Process: pred must be CV_32FC1");
    }

    cv::Mat segmentation = pred_single > thresh;
    cv::Mat mask;
    if (use_dilation_) {
        cv::Mat kernel = (cv::Mat_<uchar>(2, 2) << 1, 1, 1, 1);
        cv::dilate(segmentation, mask, kernel);
    }
    else {
        mask = segmentation;
    }

    const int src_h = img_shape[0];
    const int src_w = img_shape[1];

    // Lite: quad only.
    return BoxesFromBitmap(pred_single, mask, src_w, src_h, box_thresh, unclip_ratio);
}

absl::StatusOr<std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::BoxesFromBitmap(const cv::Mat& pred, const cv::Mat& bitmap,
    int dest_width, int dest_height,
    float box_thresh, float unclip_ratio) const {
    std::vector<std::vector<cv::Point2f>> boxes;
    std::vector<float> scores;

    const float width_scale = static_cast<float>(dest_width) / bitmap.cols;
    const float height_scale = static_cast<float>(dest_height) / bitmap.rows;

    cv::Mat bitmap_uint8;
    bitmap.convertTo(bitmap_uint8, CV_8UC1, 255.0);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap_uint8, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    int num_contours = std::min(static_cast<int>(contours.size()), max_candidates_);

    for (int i = 0; i < num_contours; ++i) {
        const auto& contour = contours[i];
        if ((int)contour.size() < 3) continue;

        cv::RotatedRect rect = cv::minAreaRect(contour);
        float short_side = std::min(rect.size.width, rect.size.height);
        if (short_side < (float)min_size_) continue;

        cv::Point2f pts[4];
        rect.points(pts);
        std::vector<cv::Point2f> quad(4);
        for (int k = 0; k < 4; ++k) quad[k] = pts[k];
        quad = OrderQuad(quad);

        std::vector<cv::Point> quad_i(4);
        for (int k = 0; k < 4; ++k) {
            quad_i[k] = cv::Point((int)std::round(quad[k].x), (int)std::round(quad[k].y));
        }

        float score = 0.0f;
        if (score_mode_ == "slow") score = BoxScoreSlow(pred, quad_i);
        else score = BoxScoreFast(pred, quad_i);

        if (score < box_thresh) continue;

        // Unclip (lite approximation)
        std::vector<cv::Point2f> quad_unclip = UnclipLite(quad, unclip_ratio);
        quad_unclip = OrderQuad(quad_unclip);

        // Map to original image coords
        for (auto& p : quad_unclip) {
            p.x *= width_scale;
            p.y *= height_scale;
        }
        ClipQuad(quad_unclip, dest_width, dest_height);

        boxes.push_back(std::move(quad_unclip));
        scores.push_back(score);
    }

    return std::make_pair(std::move(boxes), std::move(scores));
}

float DBPostProcess::BoxScoreFast(const cv::Mat& pred, const std::vector<cv::Point>& box) const {
    cv::Rect r = cv::boundingRect(box) & cv::Rect(0, 0, pred.cols, pred.rows);
    if (r.width <= 0 || r.height <= 0) return 0.0f;
    return (float)cv::mean(pred(r))[0];
}

float DBPostProcess::BoxScoreSlow(const cv::Mat& pred, const std::vector<cv::Point>& box) const {
    cv::Rect r = cv::boundingRect(box) & cv::Rect(0, 0, pred.cols, pred.rows);
    if (r.width <= 0 || r.height <= 0) return 0.0f;

    std::vector<std::vector<cv::Point>> polys(1);
    polys[0].reserve(box.size());
    for (auto p : box) polys[0].push_back(cv::Point(p.x - r.x, p.y - r.y));

    cv::Mat mask(r.height, r.width, CV_8UC1, cv::Scalar(0));
    cv::fillPoly(mask, polys, cv::Scalar(255));
    return (float)cv::mean(pred(r), mask)[0];
}

static inline float L2(const cv::Point2f& a, const cv::Point2f& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<cv::Point2f>
DBPostProcess::UnclipLite(const std::vector<cv::Point2f>& poly, float unclip_ratio) const {
    // Lite approximation of polygon offset:
    // distance = area * ratio / perimeter
    // scale points away from centroid by factor ~= 1 + distance / avg_radius
    if (poly.size() < 3) return poly;

    double area2 = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        const auto& p0 = poly[i];
        const auto& p1 = poly[(i + 1) % poly.size()];
        area2 += (double)p0.x * (double)p1.y - (double)p1.x * (double)p0.y;
    }
    const double area = std::fabs(area2) * 0.5;

    double peri = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        peri += (double)L2(poly[i], poly[(i + 1) % poly.size()]);
    }
    if (peri <= 1e-6) return poly;

    const double distance = area * (double)unclip_ratio / peri;

    cv::Point2f c(0.f, 0.f);
    for (auto& p : poly) { c.x += p.x; c.y += p.y; }
    c.x /= (float)poly.size();
    c.y /= (float)poly.size();

    double ravg = 0.0;
    for (auto& p : poly) ravg += (double)L2(p, c);
    ravg /= (double)poly.size();
    if (ravg <= 1e-6) return poly;

    const double scale = 1.0 + distance / ravg;

    std::vector<cv::Point2f> out = poly;
    for (auto& p : out) {
        p.x = c.x + (float)((p.x - c.x) * scale);
        p.y = c.y + (float)((p.y - c.y) * scale);
    }
    return out;
}

std::vector<cv::Point2f>
DBPostProcess::OrderQuad(const std::vector<cv::Point2f>& pts) {
    // Order: top-left, top-right, bottom-right, bottom-left.
    std::vector<cv::Point2f> p = pts;
    if (p.size() != 4) return p;

    std::sort(p.begin(), p.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
        });

    cv::Point2f tl = p[0], tr = p[1], bl = p[2], br = p[3];
    if (tl.x > tr.x) std::swap(tl, tr);
    if (bl.x > br.x) std::swap(bl, br);
    return { tl, tr, br, bl };
}

void DBPostProcess::ClipQuad(std::vector<cv::Point2f>& quad, int w, int h) {
    for (auto& p : quad) {
        p.x = std::max(0.f, std::min(p.x, (float)(w - 1)));
        p.y = std::max(0.f, std::min(p.y, (float)(h - 1)));
    }
}
