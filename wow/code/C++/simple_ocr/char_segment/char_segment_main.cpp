#include "char_segmenter.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string Pad3(int v) {
    // English: Zero-pad to 3 digits for stable file ordering.
    std::ostringstream oss;
    if (v < 0) v = 0;
    if (v < 10) oss << "00" << v;
    else if (v < 100) oss << "0" << v;
    else oss << v;
    return oss.str();
}

static std::string StemName(const std::string& path) {
    // English: Return filename without extension.
    std::filesystem::path p(path);
    return p.stem().string();
}

static bool EnsureDir(const std::filesystem::path& dir) {
    // English: Create directory if missing.
    std::error_code ec;
    if (std::filesystem::exists(dir, ec)) return true;
    return std::filesystem::create_directories(dir, ec);
}

int main(int argc, char** argv) {

    std::string inputPath = R"(D:\3rd\char_segment.png)";
    std::string outDirStr = R"(D:\3rd\out_chars)";
    if (argc >= 2) {
        //std::cout
        //    << "Usage:\n"
        //    << "  char_seg <input_image> [output_dir]\n\n"
        //    << "Example:\n"
        //    << "  char_seg D:/tmp/test.png D:/tmp/out\n";
        //return 1;

        inputPath = argv[1];
        outDirStr = (argc >= 3) ? argv[2] : "./out_chars";
    }

    const std::filesystem::path outDir(outDirStr);
    if (!EnsureDir(outDir)) {
        std::cerr << "[ERR] Failed to create output dir: " << outDir.string() << "\n";
        return 2;
    }

    cv::Mat bgr = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        std::cerr << "[ERR] Failed to read image: " << inputPath << "\n";
        return 3;
    }

    // English: Configure segmenter if you want to override defaults.
    CharSegmenterConfig cfg;
     cfg.bgHex = "#313131";
    // cfg.bgDistThr = 25;
     cfg.outSize = 32;

    CharSegmenter seg(cfg);

    cv::Mat overlay;
    std::vector<CharCrop> chars = seg.Segment(bgr, &overlay);

    const std::string base = StemName(inputPath);

    // English: Save debug overlay.
    if (!overlay.empty()) {
        const auto overlayPath = (outDir / (base + "_debug_overlay.png")).string();
        cv::imwrite(overlayPath, overlay);
    }

    // English: Save each segmented char as a standalone binary PNG (0/255).
    // Naming: <base>_L<line>_<idx>.png
    int saved = 0;
    int idxInLine = 0;
    int lastLine = -1;

    for (int i = 0; i < (int)chars.size(); ++i) {
        const auto& c = chars[i];

        if (c.lineIndex != lastLine) {
            lastLine = c.lineIndex;
            idxInLine = 0;
        }

        cv::Mat out = c.norm;
        if (out.empty()) continue;

        // English: Make sure it is CV_8U and 0/255.
        if (out.type() != CV_8U) {
            out.convertTo(out, CV_8U);
        }
        // English: If it is 0/1, scale up.
        double mn = 0.0, mx = 0.0;
        cv::minMaxLoc(out, &mn, &mx);
        if (mx <= 1.0) {
            out = out * 255;
        }

        std::string name =
            base + "_L" + Pad3(c.lineIndex) + "_" + Pad3(idxInLine) + ".png";

        const auto outPath = (outDir / name).string();
        if (cv::imwrite(outPath, out)) {
            ++saved;
        }
        ++idxInLine;
    }

    std::cout << "[OK] input: " << inputPath << "\n"
              << "[OK] outDir: " << outDir.string() << "\n"
              << "[OK] chars:  " << chars.size() << "\n"
              << "[OK] saved:  " << saved << "\n";

    return 0;
}
