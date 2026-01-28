#include "char_segmenter.h"
#include <opencv2/imgproc.hpp>

static inline int ClampInt(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static cv::Vec3i HexToBgr_(const std::string& hex) {
    // Equivalent to Python hex_to_bgr()
    std::string s = hex;
    if (!s.empty() && s[0] == '#') {
        s.erase(s.begin());
    }
    if (s.size() != 6) {
        throw std::runtime_error("hex color must be like #RRGGBB");
    }

    int r = std::stoi(s.substr(0, 2), nullptr, 16);
    int g = std::stoi(s.substr(2, 2), nullptr, 16);
    int b = std::stoi(s.substr(4, 2), nullptr, 16);

    // Return BGR, int16-compatible
    return cv::Vec3i(b, g, r);
}

CharSegmenter::CharSegmenter(CharSegmenterConfig cfg) : cfg_(cfg) {}

cv::Mat CharSegmenter::BinarizeWhiteBg_(const cv::Mat& bgr) const {
    CV_Assert(!bgr.empty());
    CV_Assert(bgr.type() == CV_8UC3);

    const int h = bgr.rows;
    const int w = bgr.cols;

    // Parse background color: equivalent to hex_to_bgr()
    const cv::Vec3i bg = HexToBgr_(cfg_.bgHex);

    cv::Mat fg01(h, w, CV_8U, cv::Scalar(0));

    for (int y = 0; y < h; ++y) {
        const cv::Vec3b* src = bgr.ptr<cv::Vec3b>(y);
        uint8_t* dst = fg01.ptr<uint8_t>(y);

        for (int x = 0; x < w; ++x) {
            // int16 arithmetic (same as numpy astype(int16))
            const int db = std::abs(static_cast<int>(src[x][0]) - bg[0]);
            const int dg = std::abs(static_cast<int>(src[x][1]) - bg[1]);
            const int dr = std::abs(static_cast<int>(src[x][2]) - bg[2]);

            // L_infty distance (max channel diff)
            const int dist = std::max(db, std::max(dg, dr));

            // Foreground if dist >= threshold
            dst[x] = (dist >= cfg_.bgDistThr) ? 1 : 0;
        }
    }

    return fg01;  // 0/1 mask, identical semantics to Python
}

std::vector<cv::Range> CharSegmenter::FindLineRunsByProjection_(const cv::Mat& fg01) const {
    const int h = fg01.rows;
    const int w = fg01.cols;

    // Horizontal projection: sum of fg pixels per row
    std::vector<int> rowSum(h, 0);
    for (int y = 0; y < h; ++y) {
        rowSum[y] = cv::countNonZero(fg01.row(y));
    }

    const int activeThr = std::max(2, static_cast<int>(0.002 * w));

    std::vector<cv::Range> runs;
    int y = 0;
    while (y < h) {
        if (rowSum[y] < activeThr) {
            ++y;
            continue;
        }
        int y0 = y;
        while (y < h && rowSum[y] >= activeThr) ++y;
        int y1 = y - 1;

        if ((y1 - y0 + 1) >= cfg_.minLineRunH) {
            y0 = ClampInt(y0 - cfg_.padLineY, 0, h - 1);
            y1 = ClampInt(y1 + cfg_.padLineY, 0, h - 1);
            runs.emplace_back(y0, y1 + 1); // cv::Range is [start, end)
        }
    }
    return runs;
}

std::vector<cv::Range> CharSegmenter::SplitTokensByBlankColumns_(const cv::Mat& lineFg01) const {
    const int h = lineFg01.rows;
    const int w = lineFg01.cols;

    // Vertical projection: sum of fg pixels per column
    std::vector<int> colSum(w, 0);
    for (int x = 0; x < w; ++x) {
        colSum[x] = cv::countNonZero(lineFg01.col(x));
    }

    auto isBlank = [&](int x) -> bool {
        return colSum[x] <= cfg_.blankColThr;
        };

    // Find blank runs (separators)
    std::vector<cv::Range> blankRuns;
    int x = 0;
    while (x < w) {
        if (!isBlank(x)) {
            ++x;
            continue;
        }
        int x0 = x;
        while (x < w && isBlank(x)) ++x;
        int x1 = x - 1;
        if ((x1 - x0 + 1) >= cfg_.minBlankRun) {
            blankRuns.emplace_back(x0, x1 + 1);
        }
    }

    // Tokens are between blank runs
    std::vector<cv::Range> tokens;
    int prevEnd = 0;
    if (!blankRuns.empty()) {
        prevEnd = 0;
        for (const auto& br : blankRuns) {
            int t0 = prevEnd;
            int t1 = br.start - 1;
            if (t1 >= t0 && (t1 - t0 + 1) >= cfg_.minTokenW) {
                tokens.emplace_back(t0, t1 + 1);
            }
            prevEnd = br.end;
        }
        if (prevEnd <= w - 1) {
            int t0 = prevEnd;
            int t1 = w - 1;
            if (t1 >= t0 && (t1 - t0 + 1) >= cfg_.minTokenW) {
                tokens.emplace_back(t0, t1 + 1);
            }
        }
    }
    else {
        // No blanks => one token
        tokens.emplace_back(0, w);
    }

    // Drop pure-blank tokens (should be rare)
    std::vector<cv::Range> out;
    for (const auto& t : tokens) {
        cv::Mat roi = lineFg01.colRange(t);
        if (cv::countNonZero(roi) == 0) continue;
        out.push_back(t);
    }
    return out;
}

cv::Mat CharSegmenter::ResizeKeepLineHeightGray_(const cv::Mat& cropGrayBg0) const {
    // English: Resize to a fixed square size while preserving the original line height.
    // This matches the Python logic: normalize_to_square() BEFORE binarization.
    const int outH = cfg_.outSize;
    const int outW = cfg_.outSize;
    const int pad = cfg_.outPad;

    const int h = std::max(1, cropGrayBg0.rows);
    const int w = std::max(1, cropGrayBg0.cols);

    const int ah = std::max(1, outH - 2 * pad);
    // English: Scale based on height; keep line height consistent.
    const double scale = static_cast<double>(ah) / static_cast<double>(h);
    int nh = std::max(1, static_cast<int>(std::round(h * scale)));
    int nw = std::max(1, static_cast<int>(std::round(w * scale)));

    // English: Use INTER_LINEAR in grayscale stage; this helps expand antialias gradients
    // for more stable Otsu thresholding.
    cv::Mat resized;
    cv::resize(cropGrayBg0, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(outH, outW, CV_8U, cv::Scalar(0));

    // English: Center-crop if the resized patch exceeds output bounds.
    if (nw > outW) {
        int sx = (nw - outW) / 2;
        resized = resized.colRange(sx, sx + outW).clone();
        nw = resized.cols;
        nh = resized.rows;
    }
    if (nh > outH) {
        int sy = (nh - outH) / 2;
        resized = resized.rowRange(sy, sy + outH).clone();
        nw = resized.cols;
        nh = resized.rows;
    }

    const int y0 = (outH - nh) / 2;
    const int x0 = (outW - nw) / 2;
    resized.copyTo(out(cv::Rect(x0, y0, nw, nh)));
    return out;
}

cv::Mat CharSegmenter::BinarizeOtsuAndClean_(const cv::Mat& gray) const {
    CV_Assert(gray.type() == CV_8U);

    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // English: If foreground becomes dominant (e.g., inverted), flip it.
    const int fg = cv::countNonZero(bin);
    if (fg > (bin.rows * bin.cols) / 2) {
        cv::bitwise_not(bin, bin);
    }

    // English: Remove tiny bridges/noise that often cause 0->8 or '.'->'_' mistakes.
    const cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, k);

    return bin;
}

cv::Mat CharSegmenter::NormalizeResizeThenBinarize_(const cv::Mat& cropGrayBg0, const cv::Mat& /*cropFg01*/) const {
    // English: Background pixels must already be set to 0 in cropGrayBg0.
    // Resize FIRST (grayscale), then Otsu binarize, then morphology clean.
    cv::Mat grayNorm = ResizeKeepLineHeightGray_(cropGrayBg0);
    return BinarizeOtsuAndClean_(grayNorm);
}

std::vector<CharCrop> CharSegmenter::Segment(const cv::Mat& bgr, cv::Mat* debugOverlayBgr) const {
    CV_Assert(!bgr.empty());
    CV_Assert(bgr.type() == CV_8UC3);

    const int H = bgr.rows;
    const int W = bgr.cols;

    cv::Mat fg01 = BinarizeWhiteBg_(bgr);

    // Debug overlay
    cv::Mat overlay;
    if (debugOverlayBgr) overlay = bgr.clone();

    // Split lines by projection
    std::vector<cv::Range> lineRuns = FindLineRunsByProjection_(fg01);

    std::vector<CharCrop> out;
    out.reserve(512);

    for (int li = 0; li < static_cast<int>(lineRuns.size()); ++li) {
        const auto& yr = lineRuns[li];
        int y0 = yr.start;
        int y1 = yr.end - 1;

        // Expand one background row above/below for the entire line (uniform height)
        int yy0 = ClampInt(y0 - cfg_.keepBgRows, 0, H - 1);
        int yy1 = ClampInt(y1 + cfg_.keepBgRows, 0, H - 1);

        if (debugOverlayBgr) {
            cv::rectangle(overlay, cv::Rect(0, yy0, W, yy1 - yy0 + 1), cv::Scalar(0, 255, 0), 1);
        }

        cv::Mat lineFg01 = fg01.rowRange(yy0, yy1 + 1);

        // Split tokens by blank columns (spaces)
        std::vector<cv::Range> tokens = SplitTokensByBlankColumns_(lineFg01);

        for (const auto& tr : tokens) {
            int x0 = tr.start;
            int x1 = tr.end - 1;

            // Expand one bg column at left/right (bffb behavior)
            int xx0 = ClampInt(x0 - cfg_.keepBgCols, 0, W - 1);
            int xx1 = ClampInt(x1 + cfg_.keepBgCols, 0, W - 1);

            cv::Rect bbox(xx0, yy0, xx1 - xx0 + 1, yy1 - yy0 + 1);

            // English: Build grayscale crop from original BGR, then set background pixels to 0.
            // This keeps antialias information for better Otsu thresholding.
            cv::Mat tokenBgr = bgr(bbox);
            cv::Mat tokenGray;
            cv::cvtColor(tokenBgr, tokenGray, cv::COLOR_BGR2GRAY);

            cv::Mat tokenFg01 = fg01(bbox); // 0/1
            // Background mask: tokenFg01 == 0
            cv::Mat bgMask;
            cv::compare(tokenFg01, 0, bgMask, cv::CmpTypes::CMP_EQ);
            tokenGray.setTo(0, bgMask);

            // English: Normalize with "resize then binarize" to match Python.
            cv::Mat norm = NormalizeResizeThenBinarize_(tokenGray, tokenFg01);

            CharCrop cc;
            cc.bbox = bbox;
            cc.lineIndex = li;
            cc.norm = norm;

            out.push_back(std::move(cc));

            if (debugOverlayBgr) {
                cv::rectangle(overlay, bbox, cv::Scalar(0, 0, 255), 1);
            }
        }
    }

    if (debugOverlayBgr) *debugOverlayBgr = overlay;
    return out;
}
