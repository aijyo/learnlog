#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct CharCrop {
    // Bounding box in original image coordinates
    cv::Rect bbox;

    // Line index
    int lineIndex = -1;

    // Normalized grayscale binary-like image (0 or 255), fixed size
    cv::Mat norm;
};

struct CharSegmenterConfig {
    // Binarize threshold for white background
    // gray < thr => foreground
    //int binarizeThr = 200;
    std::string bgHex = "#404040";  // Background color in hex, e.g. "#404040"
    int bgDistThr = 25;             // Color distance threshold

    // Line splitting params
    int minLineRunH = 6;
    int padLineY = 1;

    // Column splitting params (space gap)
    int blankColThr = 0;       // col_sum <= blankColThr considered blank
    int minBlankRun = 2;       // consecutive blank columns to split
    int minTokenW = 1;

    // Keep background around the token/line
    int keepBgRows = 1;        // keep N background rows above/below each line
    int keepBgCols = 1;        // keep N background cols left/right of each token

    // Normalize output
    int outSize = 32;          // output square size
    int outPad = 2;            // inner padding when scaling
};

class CharSegmenter {
public:
    explicit CharSegmenter(CharSegmenterConfig cfg = {});

    // Segment characters from an input BGR image
    std::vector<CharCrop> Segment(const cv::Mat& bgr, cv::Mat* debugOverlayBgr = nullptr) const;

private:
    cv::Mat BinarizeWhiteBg_(const cv::Mat& bgr) const;

    std::vector<cv::Range> FindLineRunsByProjection_(const cv::Mat& fg01) const;

    std::vector<cv::Range> SplitTokensByBlankColumns_(const cv::Mat& lineFg01) const;

    // Normalization aligned with the updated Python script:
    // - Segment by background color distance (fg01)
    // - For each token, build a grayscale crop and set background pixels to 0
    // - Resize to outSize FIRST (grayscale), then binarize (Otsu), then clean with morphology
    cv::Mat NormalizeResizeThenBinarize_(const cv::Mat& cropGrayBg0, const cv::Mat& cropFg01) const;

    cv::Mat ResizeKeepLineHeightGray_(const cv::Mat& cropGrayBg0) const;

    cv::Mat BinarizeOtsuAndClean_(const cv::Mat& gray) const;

private:
    CharSegmenterConfig cfg_;
};
