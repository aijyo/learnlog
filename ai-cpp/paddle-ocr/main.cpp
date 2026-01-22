#include <windows.h>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "./PaddleOcr.h"

// English:
// Convert cv::Mat (BGR/BGRA/GRAY) to BgraFrame.
static bool MatToBgraFrame(const cv::Mat& src, BgraFrame& out) {
    if (src.empty())
        return false;

    cv::Mat bgra;
    if (src.type() == CV_8UC4) {
        bgra = src;
    } else if (src.type() == CV_8UC3) {
        cv::cvtColor(src, bgra, cv::COLOR_BGR2BGRA);
    } else if (src.type() == CV_8UC1) {
        cv::cvtColor(src, bgra, cv::COLOR_GRAY2BGRA);
    } else {
        return false;
    }

    out.width = bgra.cols;
    out.height = bgra.rows;
    out.stride = (int)bgra.step;
    out.data.assign(bgra.data, bgra.data + (size_t)out.stride * (size_t)out.height);
    return true;
}

int main(int argc, char** argv) {
    //SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    //AddDllDirectory(L"D:/code/mycode/c++/ocr/paddle-onnx/build/Debug"); // current directory (your exe folder)
    // Default ONNX + dict location you provided
    const std::string base_dir = R"(D:\code\mycode\c++\ocr\paddle-onnx)";

    std::string det_dir = base_dir + "/PP-OCRv5_server_det";
    std::string rec_dir = base_dir + "/PP-OCRv5_server_rec";
    std::string dict_txt = base_dir + "/PP-OCRv5_server_rec/inference.yml";
    std::string img_path = "./demo.png"; // exe directory

    // Optional overrides: <det.onnx> <rec.onnx> <dict.txt> <image>
    if (argc >= 2) det_dir = argv[1];
    if (argc >= 3) rec_dir = argv[2];
    if (argc >= 4) dict_txt = argv[3];
    if (argc >= 5) img_path = argv[4];

    std::cout << "[INFO] det  : " << det_dir << "\n";
    std::cout << "[INFO] rec  : " << rec_dir << "\n";
    std::cout << "[INFO] dict : " << dict_txt << "\n";
    std::cout << "[INFO] image: " << img_path << "\n";

    PaddleOcr::Config cfg;
    cfg.det_model_dir = det_dir;
    cfg.rec_model_dir = rec_dir;
    cfg.rec_dict_path = dict_txt;
    cfg.cpu_threads = 4;
    cfg.max_side_len = 960;
    cfg.det_db_thresh = 0.3f;
    cfg.det_db_box_thresh = 0.6f;
    cfg.det_db_unclip_ratio = 1.5f;

    PaddleOcr ocr;
    if (!ocr.Init(cfg)) {
        std::cerr << "[ERROR] OCR Init failed\n";
        return 1;
    }

    cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to read image: " << img_path << "\n";
        return 2;
    }

    BgraFrame frame;
    frame.frame_id = 1;
    if (!MatToBgraFrame(img, frame)) {
        std::cerr << "[ERROR] Unsupported image format\n";
        return 3;
    }

    std::vector<OcrItem> items;
    std::string err;
    if (!ocr.Run(frame, items, err)) {
        std::cerr << "[ERROR] OCR Run failed: " << err << "\n";
        return 4;
    }

    std::cout << "OCR items: " << items.size() << "\n";
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& it = items[i];
        std::cout << "[" << i << "] score=" << it.score << " text=\"" << it.text << "\"";
        if (it.poly.size() >= 8) {
            std::cout << " poly=(";
            for (int k = 0; k < 8; ++k) {
                std::cout << it.poly[k] << (k == 7 ? "" : ",");
            }
            std::cout << ")";
        }
        std::cout << "\n";
    }

    std::string text_topdown;
    if (ocr.RunText(frame, text_topdown, err)) {
        std::cout << "\n--- Text (top-down) ---\n" << text_topdown << "\n";
    }

    return 0;
}
