// main_capture_ocr.cpp
// Capture ROI frames via WGC and run PaddleOCR on each frame.
// Comments are in English as requested.

#define NOMINMAX
#include <windows.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// Your ROI capturer
#include "win_capture.h"
#include "image_pipeline.h"
#include "text_handler.h"

#include "src/api/models/doc_img_orientation_classification.h"
#include "src/api/models/text_detection.h"
#include "src/api/models/text_image_unwarping.h"
#include "src/api/models/text_recognition.h"
#include "src/api/models/textline_orientation_classification.h"
#include "src/api/pipelines/doc_preprocessor.h"
#include "src/api/pipelines/ocr.h"
#include "src/utils/args.h"
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>


static const std::unordered_set<std::string> SUPPORT_MODE_PIPELINE = {
    "ocr",
    "doc_preprocessor",
};

static const std::unordered_set<std::string> SUPPORT_MODE_MODEL = {
    "text_image_unwarping", "doc_img_orientation_classification",
    "textline_orientation_classification", "text_detection",
    "text_recognition" };

// -------------------------------
// Helpers: monitor + ROI selection
// -------------------------------
static BOOL CALLBACK EnumMonitorsProc_(HMONITOR hMon, HDC, LPRECT, LPARAM user)
{
    auto* out = reinterpret_cast<std::vector<HMONITOR>*>(user);
    out->push_back(hMon);
    return TRUE;
}

static HMONITOR PickPrimaryMonitor_()
{
    // If you want more control, replace this with your own monitor picker.
    POINT pt{ 0, 0 };
    return MonitorFromPoint(pt, MONITOR_DEFAULTTOPRIMARY);
}

static RECT MakeRoi_(int x, int y, int w, int h)
{
    RECT r{};
    r.left = x;
    r.top = y;
    r.right = x + w;
    r.bottom = y + h;
    return r;
}

// -------------------------------
// Helpers: BgraFrame -> cv::Mat
// -------------------------------
static cv::Mat BgraFrameToMat_(const BgraFrame& f)
{
    // Create a CV_8UC4 Mat that shares memory with frame.data.
    // Note: This Mat is only valid as long as 'f.data' stays alive.
    return cv::Mat(f.height, f.width, CV_8UC4, (void*)f.data.data(), (size_t)f.stride);
}

// -------------------------------
// ADAPTER: PaddleOCR invocation
// -------------------------------
// You MUST adapt this part to match your actual ocr.h/ocr.cc API.
//
// Goal: provide a single function `RunOcrAndPrint_(ocr, mat_bgr)`.
//
// Common patterns you might have:
// 1) ocr.Ocr(cv::Mat bgr) -> vector<Line> or OCRResult
// 2) ocr.Run(cv::Mat bgr, ...) -> ...
// 3) ocr.Detect/Recognize pipeline, etc.
#include "src/pipelines/ocr/result.h"
static std::vector<std::string> recognition_image_(ImagePipeline& ocr, const cv::Mat& bgr)
{
    std::vector<std::string> results;
    auto inputs = std::vector{ bgr };
    auto rt_info = ocr.Predict(inputs);
    if (!rt_info.empty())
    {
        auto& item = rt_info.front();
        auto* ocr = dynamic_cast<OCRResult*>(item.get());

        auto&& texts = ocr->Texts();
        results = texts;
        printf("==============begin==============\n");
        for (auto index = 0; index < texts.size(); ++index)
        {
            printf("image content line[%d]: %s\n\n\n\n", index, texts[index].c_str());
        }

        printf("==============end==============\n");

        //auto content = ocr->Str();
        //printf("image content: %s\n\n\n\n", rt_info.front()->Str().c_str());
    }
    return results;
}

// -----------------------------
// Utility: get HMONITOR by point / rect
// -----------------------------
static HMONITOR GetMonitorFromPoint(POINT pt)
{
    return MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST);
}

static HMONITOR GetMonitorFromRectCenter(const RECT& rcVirtual)
{
    POINT c{};
    c.x = (rcVirtual.left + rcVirtual.right) / 2;
    c.y = (rcVirtual.top + rcVirtual.bottom) / 2;
    return GetMonitorFromPoint(c);
}

static bool GetMonitorRect(HMONITOR mon, RECT& outRcMonitorVirtual)
{
    MONITORINFO mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(mon, &mi)) return false;
    outRcMonitorVirtual = mi.rcMonitor; // virtual screen coordinates
    return true;
}

static inline void NormalizeRect(RECT& r)
{
    if (r.left > r.right) std::swap(r.left, r.right);
    if (r.top > r.bottom) std::swap(r.top, r.bottom);
}

// Convert virtual-screen ROI to monitor-local ROI, clipping to that monitor.
// Returns false if ROI doesn't intersect with the monitor.
static bool VirtualToMonitorLocalRect(HMONITOR mon, const RECT& rcVirtual, RECT& outLocal)
{
    MONITORINFO mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(mon, &mi))
        return false;

    RECT v = rcVirtual;
    NormalizeRect(v);

    RECT clipped{};
    if (!IntersectRect(&clipped, &v, &mi.rcMonitor))
        return false;

    outLocal.left = clipped.left - mi.rcMonitor.left;
    outLocal.top = clipped.top - mi.rcMonitor.top;
    outLocal.right = clipped.right - mi.rcMonitor.left;
    outLocal.bottom = clipped.bottom - mi.rcMonitor.top;

    NormalizeRect(outLocal);
    return (outLocal.right > outLocal.left) && (outLocal.bottom > outLocal.top);
}

// --------- global stop flag ---------
static std::atomic<bool> g_stop{ false };

static BOOL WINAPI ConsoleCtrlHandler_(DWORD ctrlType) {
    // English comment:
    // Handle Ctrl+C / console close to stop threads gracefully.
    switch (ctrlType) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        g_stop.store(true);
        return TRUE;
    default:
        return FALSE;
    }
}

int main(int argc, char** argv)
{
    // English comment:
    // Install console handler to stop on Ctrl+C.
    ::SetConsoleCtrlHandler(ConsoleCtrlHandler_, TRUE);

    // --------- Config (adjust to your project) ----------
    const std::string base_dir = R"(D:\code\gitcode\PaddleOCR\deploy\cpp_infer\wow)";
    std::string det_dir = base_dir + R"(\PP-OCRv5_mobile_det_infer)";
    std::string rec_dir = base_dir + R"(\PP-OCRv5_mobile_rec_infer)";
    std::string cls_dir = "";   // optional
    std::string dict_path = ""; // optional
    int cpu_threads = 4;
    bool use_gpu = false;
    int gpu_id = 0;

    // 1) Select region in virtual-screen coordinates
    RECT rcVirtual{};
    {
        ScreenRegionSelector selector(720, 640);
        if (!selector.SelectRegionVirtual(rcVirtual))
        {
            MessageBoxW(nullptr, L"Selection canceled or invalid.", L"Info", MB_OK);
            return 0;
        }
    }

    // 2) Pick monitor by selection center
    HMONITOR mon = GetMonitorFromRectCenter(rcVirtual);
    if (!mon)
    {
        MessageBoxW(nullptr, L"Failed to get monitor.", L"Error", MB_OK | MB_ICONERROR);
        return 0;
    }

    // 3) Convert virtual ROI -> monitor local ROI
    RECT roiLocal{};
    if (!VirtualToMonitorLocalRect(mon, rcVirtual, roiLocal))
    {
        MessageBoxW(nullptr, L"Failed to get roi local rect.", L"Error", MB_OK | MB_ICONERROR);
        return 0;
    }

    // --------- Init OCR pipeline ----------
    OCRPipelineParams config;
    {
        config.text_detection_model_dir = det_dir;
        config.text_detection_model_name = "PP-OCRv5_mobile_det";

        config.text_recognition_model_dir = rec_dir;
        config.text_recognition_model_name = "PP-OCRv5_mobile_rec";

        config.use_doc_orientation_classify = false;
        config.use_doc_unwarping = false;
        config.use_textline_orientation = false;

        config.device = "cpu";
        config.cpu_threads = 4;
        config.thread_num = 1;

        config.precision = "fp32";
        config.enable_mkldnn = true;
        config.mkldnn_cache_capacity = 10;

        config.lang = "eng";
        config.paddlex_config = R"(D:\code\gitcode\PaddleOCR\deploy\cpp_infer\wow\ocr.yaml)";
    }

    ImagePipeline image_pipeline(config);

    // --------- Init Capture ----------
    WgcRoiCapturer cap;
    if (!cap.StartMonitorRoi(mon, roiLocal, 20))
    {
        std::cerr << "Failed to start WGC ROI capture.\n";
        return 1;
    }

    std::cout << "Capture started. Press Ctrl+C to exit.\n";

    // --------- Main thread: TextHandler ----------
    // English comment:
    // TextHandler::Start() blocks with a message loop. Run it on main thread.
    TextHandler::Options th_opt;
    th_opt.com_port = "COM3";
    th_opt.baud = 9600;
    th_opt.addr = 0x00;
    th_opt.wait_ack = true;
    th_opt.debug = false;
    th_opt.trigger_vk = 'X';          // default trigger key = '1'
    // th_opt.target_exe = L"Wow.exe"; // optional
    th_opt.user_config = R"(D:\code\gitcode\PaddleOCR\deploy\cpp_infer\wow\user_keybinds.json)";

    TextHandler handler(th_opt);
    handler.SetMappedKey("1");        // default mapped key; can be changed at runtime

    // --------- OCR thread ----------
    auto ocr_callback = [&]()
        {
            // English comment:
            // Pull latest frame and OCR in a dedicated worker thread.
            uint64_t last_frame_id = 0;

            while (!g_stop.load())
            {
                BgraFrame f;
                if (!cap.TryGetLatest(f))
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                if (f.frame_id == last_frame_id)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                last_frame_id = f.frame_id;

                // Convert BGRA -> BGR
                cv::Mat bgra = BgraFrameToMat_(f);
                cv::Mat bgr;
                cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

                try
                {
                    auto&& texts = recognition_image_(image_pipeline, bgr);
                    handler.set_texts(std::move(texts));
                }
                catch (const std::exception& e)
                {
                    std::cerr << "OCR exception: " << e.what() << "\n";
                }

                // Throttle
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }

            // English comment:
            // Thread exits when g_stop is set.

        };
    std::thread ocr_thread(std::move(ocr_callback));

    // English comment:
    // If you want OCR rt_info to change mapping dynamically:
    // you can call handler.SetMappedKey(...) from OCR thread,
    // but prefer to use a thread-safe queue or atomic string swap.

    // Start() blocks until it quits (or you call handler.Stop()).
    // We rely on Ctrl+C to set g_stop and then we ask handler.Stop().
    std::thread stop_watcher([&]() {
        // English comment:
        // Monitor stop flag and stop the TextHandler message loop.
        while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        handler.Stop();
        });

    handler.Start(); // blocking

    // --------- Shutdown ----------
    g_stop.store(true);

    if (stop_watcher.joinable())
        stop_watcher.join();

    // Stop capture if your capturer exposes Stop()
    // cap.Stop();

    if (ocr_thread.joinable())
        ocr_thread.join();

    std::cout << "Exited.\n";
    return 0;
}

//
//// -------------------------------
//// MAIN
//// -------------------------------
//int main(int argc, char** argv)
//{
//    // --------- Config (adjust to your project) ----------
//    // If cli.cc used flags, you can read args here similarly.
//    // Example defaults:
//    const std::string base_dir = R"(D:\code\gitcode\PaddleOCR\deploy\cpp_infer\wow)";
//    std::string det_dir = base_dir + R"(\PP-OCRv5_mobile_det_infer)";
//    std::string rec_dir = base_dir + R"(\PP-OCRv5_mobile_rec_infer)";
//    std::string cls_dir = ""; // optional
//    std::string dict_path = ""; // optional (if your wrapper still uses dict)
//    int cpu_threads = 4;
//    bool use_gpu = false;
//    int gpu_id = 0;
//
//    // ROI: monitor-local coordinates
//    //SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
//    // 1) Select region in virtual-screen coordinates
//    RECT rcVirtual{};
//    {
//        ScreenRegionSelector selector(720, 640);
//        //ScreenRegionSelector selector(38, 38);
//        if (!selector.SelectRegionVirtual(rcVirtual))
//        {
//            MessageBoxW(nullptr, L"Selection canceled or invalid.", L"Info", MB_OK);
//            return 0;
//        }
//    }
//
//    // 2) Pick monitor by selection center
//    HMONITOR mon = GetMonitorFromRectCenter(rcVirtual);
//    if (!mon)
//    {
//        MessageBoxW(nullptr, L"Failed to get monitor.", L"Error", MB_OK | MB_ICONERROR);
//        return 0;
//    }
//
//    // 3) Convert virtual ROI -> monitor local ROI
//    RECT roiLocal{};
//    auto bSuc = VirtualToMonitorLocalRect(mon, rcVirtual, roiLocal);
//    if (!bSuc)
//    {
//        MessageBoxW(nullptr, L"Failed to get roi local rect.", L"Error", MB_OK | MB_ICONERROR);
//        return 0;
//    }
//
//
//    // Allow quick override from args:
//    // argv[1]=roi_x argv[2]=roi_y argv[3]=roi_w argv[4]=roi_h
//    //if (argc >= 5)
//    //{
//    //    int x = std::stoi(argv[1]);
//    //    int y = std::stoi(argv[2]);
//    //    int w = std::stoi(argv[3]);
//    //    int h = std::stoi(argv[4]);
//    //    roi = MakeRoi_(x, y, w, h);
//    //}
//
//    // --------- Init OCR ----------
//    // You MUST adapt this construction/init to match your ocr.h implementation.
//    //PaddleOCR ocr;
//    //{
//    //    // Example init API (CHANGE to match your wrapper):
//    //    PaddleOCRParams cfg;
//    //    cfg.det_model_dir = det_dir;
//    //    cfg.rec_model_dir = rec_dir;
//    //    cfg.cls_model_dir = cls_dir;
//    //    cfg.dict_path = dict_path;
//    //    cfg.use_gpu = use_gpu;
//    //    cfg.gpu_id = gpu_id;
//    //    cfg.cpu_threads = cpu_threads;
//
//    //    if (!ocr.Init(cfg))  // <-- CHANGE THIS to match your ocr.h
//    //    {
//    //        std::cerr << "OCR init failed.\n";
//    //        return 1;
//    //    }
//    //}
//    OCRPipelineParams config;
//    {
//        // ---------- Detection ----------
//        config.text_detection_model_dir = det_dir;
//         config.text_detection_model_name = "PP-OCRv5_mobile_det";
//
//        // ---------- Recognition ----------
//        config.text_recognition_model_dir = rec_dir;
//         config.text_recognition_model_name = "PP-OCRv5_mobile_rec";
//
//        // ---------- Disable document-level preprocess ----------
//        config.use_doc_orientation_classify = false;
//        config.use_doc_unwarping = false;
//
//        // ---------- Disable textline orientation ----------
//        config.use_textline_orientation = false;
//
//        // ---------- Runtime / device ----------
//        config.device = "cpu";
//        config.cpu_threads = 8;
//        config.thread_num = 1;
//
//        // ---------- Precision / MKLDNN ----------
//        config.precision = "fp32";       // 默认 fp32
//        config.enable_mkldnn = true;     // 默认 true
//        config.mkldnn_cache_capacity = 10;
//
//        // ---------- Language / OCR version ----------
//        config.lang = "eng";
//        // config.lang = "ch";
//        // config.ocr_version = "PP-OCRv5";
//        config.paddlex_config = R"(D:\code\gitcode\PaddleOCR\deploy\cpp_infer\wow\ocr.yaml)";
//    }
//
//    ImagePipeline image_pipeline(config);
//
//    // --------- Init Capture ----------
//    //HMONITOR mon = PickPrimaryMonitor_();
//    //if (!mon)
//    //{
//    //    std::cerr << "Failed to pick monitor.\n";
//    //    return 1;
//    //}
//
//    WgcRoiCapturer cap;
//    if (!cap.StartMonitorRoi(mon, roiLocal, 20))
//    {
//        std::cerr << "Failed to start WGC ROI capture.\n";
//        return 1;
//    }
//
//    std::cout << "Capture started. Press Ctrl+C to exit.\n";
//
//    // --------- Loop: pull latest frame and OCR ----------
//    uint64_t last_frame_id = 0;
//    while (true)
//    {
//        BgraFrame f;
//        if (!cap.TryGetLatest(f))
//        {
//            std::this_thread::sleep_for(std::chrono::milliseconds(10));
//            continue;
//        }
//
//        if (f.frame_id == last_frame_id)
//        {
//            std::this_thread::sleep_for(std::chrono::milliseconds(10));
//            continue;
//        }
//        last_frame_id = f.frame_id;
//
//        // Convert BGRA -> BGR
//        cv::Mat bgra = BgraFrameToMat_(f);
//        cv::Mat bgr;
//        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
//
//        // Run OCR and print
//        try
//        {
//            recognition_image_(image_pipeline, bgr);
//        }
//        catch (const std::exception& e)
//        {
//            std::cerr << "OCR exception: " << e.what() << "\n";
//        }
//
//        // Throttle a bit to avoid spamming OCR every frame
//        std::this_thread::sleep_for(std::chrono::milliseconds(200));
//    }
//
//    // cap.Stop(); // unreachable in this loop
//    // return 0;
//}
