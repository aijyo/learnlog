//#include <windows.h>
//#include <iostream>
//#include <string>
//#include <vector>
//
//#include <opencv2/opencv.hpp>
//
////#include "./PaddleOcr.h"
////#include "TextRecProcessor.h"
//#include "image_handler.h"
//
//// English:
//// Convert cv::Mat (BGR/BGRA/GRAY) to BgraFrame.
//static bool MatToBgraFrame(const cv::Mat& src, BgraFrame& out) {
//    if (src.empty())
//        return false;
//
//    cv::Mat bgra;
//    if (src.type() == CV_8UC4) {
//        bgra = src;
//    } else if (src.type() == CV_8UC3) {
//        cv::cvtColor(src, bgra, cv::COLOR_BGR2BGRA);
//    } else if (src.type() == CV_8UC1) {
//        cv::cvtColor(src, bgra, cv::COLOR_GRAY2BGRA);
//    } else {
//        return false;
//    }
//
//    out.width = bgra.cols;
//    out.height = bgra.rows;
//    out.stride = (int)bgra.step;
//    out.data.assign(bgra.data, bgra.data + (size_t)out.stride * (size_t)out.height);
//    return true;
//}
//
//int main(int argc, char** argv) {
//    //SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
//    //AddDllDirectory(L"D:/code/mycode/c++/ocr/paddle-onnx/build/Debug"); // current directory (your exe folder)
//    // Default ONNX + dict location you provided
//    const std::string base_dir = R"(D:\code\wow\paddle-ocr)";
//
//    std::string det_dir = base_dir + "/PP-OCRv5_server_det_infer";
//    std::string rec_dir = base_dir + "/PP-OCRv5_server_rec_infer";
//    std::string dict_txt = base_dir + "/PP-OCRv5_server_rec_infer/inference.yml";
//    std::string img_path = R"(D:\code\wow\paddle-ocr\ocr_demo.png)"; // exe directory
//
//    // Optional overrides: <det.onnx> <rec.onnx> <dict.txt> <image>
//    if (argc >= 2) det_dir = argv[1];
//    if (argc >= 3) rec_dir = argv[2];
//    if (argc >= 4) dict_txt = argv[3];
//    if (argc >= 5) img_path = argv[4];
//
//    std::cout << "[INFO] det  : " << det_dir << "\n";
//    std::cout << "[INFO] rec  : " << rec_dir << "\n";
//    std::cout << "[INFO] dict : " << dict_txt << "\n";
//    std::cout << "[INFO] image: " << img_path << "\n";
//
//    TextRecProcessor::Config cfg;
//    cfg.det.model_dir = det_dir;
//    cfg.rec_model_dir = rec_dir;
//    cfg.rec_dict_path = dict_txt;
//    cfg.cpu_threads = 4;
//    cfg.max_side_len = 960;
//    cfg.det_db_thresh = 0.3f;
//    cfg.det_db_box_thresh = 0.6f;
//    cfg.det_db_unclip_ratio = 1.5f;
//
//    PaddleOcr ocr;
//    if (!ocr.Init(cfg)) {
//        std::cerr << "[ERROR] OCR Init failed\n";
//        return 1;
//    }
//
//    cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
//    if (img.empty()) {
//        std::cerr << "[ERROR] Failed to read image: " << img_path << "\n";
//        return 2;
//    }
//
//    BgraFrame frame;
//    frame.frame_id = 1;
//    if (!MatToBgraFrame(img, frame)) {
//        std::cerr << "[ERROR] Unsupported image format\n";
//        return 3;
//    }
//
//    std::vector<OcrItem> items;
//    std::string err;
//    if (!ocr.Run(frame, items, err)) {
//        std::cerr << "[ERROR] OCR Run failed: " << err << "\n";
//        return 4;
//    }
//
//    std::cout << "OCR items: " << items.size() << "\n";
//    for (size_t i = 0; i < items.size(); ++i) {
//        const auto& it = items[i];
//        std::cout << "[" << i << "] score=" << it.score << " text=\"" << it.text << "\"";
//        if (it.poly.size() >= 8) {
//            std::cout << " poly=(";
//            for (int k = 0; k < 8; ++k) {
//                std::cout << it.poly[k] << (k == 7 ? "" : ",");
//            }
//            std::cout << ")";
//        }
//        std::cout << "\n";
//    }
//
//    std::string text_topdown;
//    if (ocr.RunText(frame, text_topdown, err)) {
//        std::cout << "\n--- Text (top-down) ---\n" << text_topdown << "\n";
//    }
//
//    return 0;
//}
#include <iostream>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

#include "utils/absl_shim.h"
#include "image_handler.h"
#include "text_detection.h"
#include "text_recognition.h"
#include "text_handler.h"

static void PrintOcrFrame(const OcrFrameResult& frame) {
    std::cout << "OCR items: " << frame.items.size() << "\n";
    for (size_t i = 0; i < frame.items.size(); ++i) {
        const auto& it = frame.items[i];
        std::cout << "[" << i << "] rec_score=" << it.rec_score
            << " det_score=" << it.det_score
            << " text=\"" << it.text << "\"";

        if (it.quad.size() >= 4) {
            std::cout << " quad=(";
            for (int k = 0; k < 4; ++k) {
                std::cout << it.quad[k].x << "," << it.quad[k].y;
                if (k != 3) std::cout << " ";
            }
            std::cout << ")";
        }

        std::cout << " bbox=(" << it.bbox.x << "," << it.bbox.y
            << "," << it.bbox.width << "," << it.bbox.height << ")";

        std::cout << "\n";
    }

    if (!frame.merged_text.empty()) {
        std::cout << "\n--- Text (top-down) ---\n" << frame.merged_text << "\n";
    }
}

int main(int argc, char** argv) {
    //SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    //AddDllDirectory(L"D:/code/mycode/c++/ocr/paddle-onnx/build/Debug");

    const std::string base_dir = R"(D:\code\wow\paddle-ocr)";

    std::string det_dir = base_dir + "/PP-OCRv5_server_det_infer";
    std::string rec_dir = base_dir + "/PP-OCRv5_server_rec_infer";
    //std::string dict_yml = base_dir + "/PP-OCRv5_server_rec_infer/inference.yml";
    std::string img_path = R"(D:\code\wow\paddle-ocr\ocr_demo.png)";

    // Optional overrides: <det_dir> <rec_dir> <dict_yml> <image>
    if (argc >= 2) det_dir = argv[1];
    if (argc >= 3) rec_dir = argv[2];
    //if (argc >= 4) dict_yml = argv[3];
    if (argc >= 4) img_path = argv[3];

    std::cout << "[INFO] det  : " << det_dir << "\n";
    std::cout << "[INFO] rec  : " << rec_dir << "\n";
    //std::cout << "[INFO] dict : " << dict_yml << "\n";
    std::cout << "[INFO] image: " << img_path << "\n";

    // -----------------------------
    // 1) Build TextDetection
    // -----------------------------
    TextDetectionParams det_params;
    det_params.model_dir = det_dir;

    det_params.device = "cpu";
    det_params.precision = "fp32";
    det_params.enable_mkldnn = true;
    det_params.mkldnn_cache_capacity = 10;
    det_params.cpu_threads = 4;

    // Align with your old cfg:
    det_params.limit_side_len = 960;
    det_params.thresh = 0.3f;
    det_params.box_thresh = 0.6f;
    det_params.unclip_ratio = 1.5f;

    auto det = std::make_shared<TextDetection>(det_params);
    {
        auto st = det->Build();
        if (!st.ok()) {
            std::cerr << "[ERROR] TextDetection Build failed: " << st.message() << "\n";
            return 1;
        }
    }

    // -----------------------------
    // 2) Build TextRecognition
    // -----------------------------
    TextRecognitionParams rec_params;
    // 关键：TextRecognitionParams 里没有 dict_path 字段，
    // 一般是从 model_dir 里的 inference.yml 或模型配置里读取。
    // 你这里传 rec_dir 即可（目录中包含 inference.yml）。
    rec_params.model_dir = rec_dir;

    // 如果你的实现支持把 yml 当作 model_dir，也可以切到：
    // rec_params.model_dir = dict_yml;

    rec_params.device = "cpu";
    rec_params.precision = "fp32";
    rec_params.enable_mkldnn = true;
    rec_params.mkldnn_cache_capacity = 10;
    rec_params.cpu_threads = 4;

    auto rec = std::make_shared<TextRecognition>(rec_params);
    {
        auto st = rec->CheckParams();
        if (!st.ok()) {
            std::cerr << "[ERROR] TextRecognition CheckParams failed: " << st.message() << "\n";
            return 1;
        }
        rec->CreateModel();
    }

    // -----------------------------
    // 3) Build WowTextHandler (business logic)
    // -----------------------------
    WowTextHandler::Config text_cfg;
    text_cfg.min_rec_score = 0.0f;
    text_cfg.max_ui_lines = 5;

    auto text_handler = std::make_shared<WowTextHandler>(text_cfg);

    // Optional: register callback to receive parsed UI
    text_handler->SetCallback([](const WowTextHandler::ParsedUi& ui) {
        // You can hook your game logic here.
        // Keep it minimal for now:
        std::cout << "\n[ParsedUi]\n";
        std::cout << "l1=" << ui.l1 << "\n";
        std::cout << "l2=" << ui.l2 << "\n";
        std::cout << "l3=" << ui.l3 << "\n";
        std::cout << "l4=" << ui.l4 << "\n";
        std::cout << "l5=" << ui.l5 << "\n";
        std::cout << "slot=" << ui.action_slot << " spell_id=" << ui.spell_id
            << " spell_name=" << ui.spell_name << "\n";
        });

    // -----------------------------
    // 4) Create ImageHandler + run
    // -----------------------------
    ImageHandlerConfig ih_cfg;
    ih_cfg.enable_det = true;
    ih_cfg.det_score_thresh = 0.3f;   // 参考你原 det_db_thresh
    ih_cfg.rec_score_thresh = 0.0f;
    ih_cfg.sort_items = true;
    ih_cfg.sort_y_tol = 10;
    ih_cfg.build_merged_text = true;

    // 如果你只想跑 WoW 插件的 UI 区域，建议设置 ROI：
    // ih_cfg.roi = cv::Rect(x, y, w, h);

    ImageHandler handler(det, rec, text_handler, ih_cfg);

    auto r = handler.ProcessPath(img_path);
    if (!r.ok()) {
        std::cerr << "[ERROR] ImageHandler ProcessPath failed: " << r.status().message() << "\n";
        return 2;
    }

    PrintOcrFrame(r.value());
    return 0;
}
