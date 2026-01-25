//// win_main.cpp (embedded controller)
//// Integrated version: exposes Start(json_config) / Stop(reason) for ui_win_main.cpp.
//// No standalone main() by default.
//// Comments are in English as requested.
//
//#define NOMINMAX
//#include <windows.h>
//#include <atomic>
//#include <chrono>
//#include <iostream>
//#include <memory>
//#include <mutex>
//#include <string>
//#include <thread>
//#include <unordered_set>
//#include <vector>
//
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//
//#include "third_party/nlohmann/json.hpp"
//using nlohmann::json;
//
//// Your ROI capturer + pipeline + handler
//#include "win_capture.h"
//#include "image_pipeline.h"
//#include "text_handler.h"
//
//// Pipelines / models
//#include "src/api/models/doc_img_orientation_classification.h"
//#include "src/api/models/text_detection.h"
//#include "src/api/models/text_image_unwarping.h"
//#include "src/api/models/text_recognition.h"
//#include "src/api/models/textline_orientation_classification.h"
//#include "src/api/pipelines/doc_preprocessor.h"
//#include "src/api/pipelines/ocr.h"
//#include "src/utils/args.h"
//
//#include "src/pipelines/ocr/result.h"
//
//
//static bool ReadTextFileUtf8(const std::wstring& path, std::string& out) {
//    out.clear();
//    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
//    if (h == INVALID_HANDLE_VALUE) return false;
//
//    LARGE_INTEGER sz{};
//    if (!GetFileSizeEx(h, &sz) || sz.QuadPart <= 0 || sz.QuadPart > (10LL * 1024 * 1024)) {
//        CloseHandle(h);
//        return false;
//    }
//
//    out.resize((size_t)sz.QuadPart);
//    DWORD read = 0;
//    BOOL ok = ReadFile(h, out.data(), (DWORD)out.size(), &read, nullptr);
//    CloseHandle(h);
//    return ok && read == out.size();
//}
//
//// -------------------------------
//// Helpers: monitor + rect conversion
//// -------------------------------
//static RECT MakeVirtualRectFromXYWH_(int x, int y, int w, int h)
//{
//    RECT r{};
//    r.left = x;
//    r.top = y;
//    r.right = x + w;
//    r.bottom = y + h;
//    return r;
//}
//
//static inline void NormalizeRect_(RECT& r)
//{
//    if (r.left > r.right) std::swap(r.left, r.right);
//    if (r.top > r.bottom) std::swap(r.top, r.bottom);
//}
//
//static HMONITOR GetMonitorFromPoint_(POINT pt)
//{
//    return MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST);
//}
//
//static HMONITOR GetMonitorFromRectCenter_(const RECT& rcVirtual)
//{
//    POINT c{};
//    c.x = (rcVirtual.left + rcVirtual.right) / 2;
//    c.y = (rcVirtual.top + rcVirtual.bottom) / 2;
//    return GetMonitorFromPoint_(c);
//}
//
//// Convert virtual-screen ROI to monitor-local ROI, clipping to that monitor.
//// Returns false if ROI doesn't intersect with the monitor.
//static bool VirtualToMonitorLocalRect_(HMONITOR mon, const RECT& rcVirtual, RECT& outLocal)
//{
//    MONITORINFO mi{};
//    mi.cbSize = sizeof(mi);
//    if (!GetMonitorInfoW(mon, &mi))
//        return false;
//
//    RECT v = rcVirtual;
//    NormalizeRect_(v);
//
//    RECT clipped{};
//    if (!IntersectRect(&clipped, &v, &mi.rcMonitor))
//        return false;
//
//    outLocal.left = clipped.left - mi.rcMonitor.left;
//    outLocal.top = clipped.top - mi.rcMonitor.top;
//    outLocal.right = clipped.right - mi.rcMonitor.left;
//    outLocal.bottom = clipped.bottom - mi.rcMonitor.top;
//
//    NormalizeRect_(outLocal);
//    return (outLocal.right > outLocal.left) && (outLocal.bottom > outLocal.top);
//}
//
//// -------------------------------
//// Helpers: BgraFrame -> cv::Mat
//// -------------------------------
//static cv::Mat BgraFrameToMat_(const BgraFrame& f)
//{
//    // English comment:
//    // Create a CV_8UC4 Mat that shares memory with frame.data.
//    return cv::Mat(f.height, f.width, CV_8UC4, (void*)f.data.data(), (size_t)f.stride);
//}
//
//// -------------------------------
//// Adapter: OCR invocation
//// -------------------------------
//static std::vector<std::string> recognition_image_(ImagePipeline& pipeline, const cv::Mat& bgr)
//{
//    std::vector<std::string> results;
//    auto inputs = std::vector{ bgr };
//    auto rt_info = pipeline.Predict(inputs);
//    if (!rt_info.empty())
//    {
//        auto& item = rt_info.front();
//        auto* ocr = dynamic_cast<OCRResult*>(item.get());
//        if (ocr) results = ocr->Texts();
//    }
//    return results;
//}
//
//// -------------------------------
//// Public API: Start/Stop/IsRunning
//// -------------------------------
//namespace wowapp {
//
//    struct RuntimeConfig
//    {
//        // OCR
//        std::string det_dir;
//        std::string rec_dir;
//        std::string paddlex_config;
//        std::string det_model_name = "PP-OCRv5_mobile_det";
//        std::string rec_model_name = "en_PP-OCRv4_mobile_rec";
//        std::string device = "cpu";
//        int cpu_threads = 1;
//        int thread_num = 1;
//        std::string precision = "fp32";
//        bool enable_mkldnn = false;
//        int mkldnn_cache_capacity = 10;
//        std::string lang = "eng";
//
//        // Serial / hotkeys
//        std::string com_name = "COM3";
//        int baud = 9600;
//        int addr = 0;
//        bool wait_ack = true;
//        bool debug = false;
//        int trigger_vk = (int)'X';
//        int switch_vk = (int)VK_F12;
//        bool auto_type = true;
//
//        // Keybind map
//        std::string user_keybinds_path;
//
//        // Capture ROI (virtual screen coords)
//        RECT rcVirtual{};
//    };
//
//    struct State
//    {
//        std::mutex mu;
//
//        std::atomic<bool> running{ false };
//        std::atomic<bool> stop_flag{ false };
//        std::atomic<int> stop_reason{ 0 };
//
//        std::thread main_thread;
//
//        // Owned by main_thread
//        std::unique_ptr<TextHandler> handler;
//        std::unique_ptr<WgcRoiCapturer> cap;
//        std::unique_ptr<ImagePipeline> pipeline;
//        std::thread ocr_thread;
//    };
//
//    static State g_state;
//
//    // English comment:
//    // Parse JSON string (UTF-8) into RuntimeConfig.
//    // Expected structure is compatible with ui_win_main's config.json schema.
//    static bool ParseConfig_(const std::string& json_utf8, RuntimeConfig& out, std::string& err)
//    {
//        json cfg;
//        try {
//            cfg = json::parse(json_utf8);
//        }
//        catch (...) {
//            err = "invalid json";
//            return false;
//        }
//
//        // Defaults are kept if fields are missing.
//        try {
//            if (cfg.contains("ocr")) {
//                auto& o = cfg["ocr"];
//                if (o.contains("det_dir")) out.det_dir = o["det_dir"].get<std::string>();
//                if (o.contains("rec_dir")) out.rec_dir = o["rec_dir"].get<std::string>();
//                if (o.contains("paddlex_config")) out.paddlex_config = o["paddlex_config"].get<std::string>();
//                if (o.contains("text_detection_model_name")) out.det_model_name = o["text_detection_model_name"].get<std::string>();
//                if (o.contains("text_recognition_model_name")) out.rec_model_name = o["text_recognition_model_name"].get<std::string>();
//                if (o.contains("device")) out.device = o["device"].get<std::string>();
//                if (o.contains("cpu_threads")) out.cpu_threads = o["cpu_threads"].get<int>();
//                if (o.contains("thread_num")) out.thread_num = o["thread_num"].get<int>();
//                if (o.contains("precision")) out.precision = o["precision"].get<std::string>();
//                if (o.contains("enable_mkldnn")) out.enable_mkldnn = o["enable_mkldnn"].get<bool>();
//                if (o.contains("mkldnn_cache_capacity")) out.mkldnn_cache_capacity = o["mkldnn_cache_capacity"].get<int>();
//                if (o.contains("lang")) out.lang = o["lang"].get<std::string>();
//            }
//
//            if (cfg.contains("serial")) {
//                auto& s = cfg["serial"];
//                if (s.contains("comName")) out.com_name = s["comName"].get<std::string>();
//                if (s.contains("baud")) out.baud = s["baud"].get<int>();
//                if (s.contains("addr")) out.addr = s.value("addr", 0);
//                if (s.contains("wait_ack")) out.wait_ack = s.value("wait_ack", true);
//                if (s.contains("debug")) out.debug = s.value("debug", false);
//            }
//
//            if (cfg.contains("hotkeys")) {
//                auto& h = cfg["hotkeys"];
//                if (h.contains("trigger_vk")) out.trigger_vk = h["trigger_vk"].get<int>();
//                if (h.contains("switch_vk")) out.switch_vk = h["switch_vk"].get<int>();
//                if (h.contains("auto_type")) out.auto_type = h["auto_type"].get<bool>();
//            }
//
//            if (cfg.contains("user_keybinds_path")) {
//                out.user_keybinds_path = cfg["user_keybinds_path"].get<std::string>();
//            }
//
//            if (!cfg.contains("capture")) {
//                err = "missing capture";
//                return false;
//            }
//            auto& c = cfg["capture"];
//            int x = c.value("x", 0);
//            int y = c.value("y", 0);
//            int w = c.value("w", 0);
//            int h = c.value("h", 0);
//            bool use_saved = c.value("use_saved_region", false);
//            if (!use_saved || w <= 0 || h <= 0) {
//                err = "invalid capture region";
//                return false;
//            }
//            out.rcVirtual = MakeVirtualRectFromXYWH_(x, y, w, h);
//        }
//        catch (...) {
//            err = "bad schema";
//            return false;
//        }
//
//        // Required fields: det/rec/paddlex_config and mapping path (depending on your pipeline)
//        if (out.det_dir.empty() || out.rec_dir.empty() || out.paddlex_config.empty()) {
//            err = "missing ocr paths";
//            return false;
//        }
//        if (out.user_keybinds_path.empty()) {
//            err = "missing user_keybinds_path";
//            return false;
//        }
//        return true;
//    }
//
//    static void ShutdownOwnedObjects_(State& st)
//    {
//        // English comment:
//        // Ensure threads are stopped and objects released.
//        if (st.handler) {
//            st.handler->Stop();
//        }
//        if (st.ocr_thread.joinable()) {
//            st.ocr_thread.join();
//        }
//        if (st.cap) {
//            // If your capturer has Stop(), call it here.
//            // st.cap->Stop();
//        }
//        st.pipeline.reset();
//        st.cap.reset();
//        st.handler.reset();
//    }
//
//    static void MainThread_(RuntimeConfig cfg)
//    {
//        State& st = g_state;
//
//        // 1) Monitor & ROI convert
//        HMONITOR mon = GetMonitorFromRectCenter_(cfg.rcVirtual);
//        if (!mon) {
//            st.running.store(false);
//            return;
//        }
//
//        RECT roiLocal{};
//        if (!VirtualToMonitorLocalRect_(mon, cfg.rcVirtual, roiLocal)) {
//            st.running.store(false);
//            return;
//        }
//
//        // 2) Init OCR pipeline
//        OCRPipelineParams p;
//        p.text_detection_model_dir = cfg.det_dir;
//        p.text_detection_model_name = cfg.det_model_name;
//        p.text_recognition_model_dir = cfg.rec_dir;
//        p.text_recognition_model_name = cfg.rec_model_name;
//
//        p.use_doc_orientation_classify = false;
//        p.use_doc_unwarping = false;
//        p.use_textline_orientation = false;
//
//        p.device = cfg.device;
//        p.cpu_threads = cfg.cpu_threads;
//        p.thread_num = cfg.thread_num;
//
//        p.precision = cfg.precision;
//        p.enable_mkldnn = cfg.enable_mkldnn;
//        p.mkldnn_cache_capacity = cfg.mkldnn_cache_capacity;
//
//        p.lang = cfg.lang;
//        p.paddlex_config = cfg.paddlex_config;
//
//        st.pipeline = std::make_unique<ImagePipeline>(p);
//
//        // 3) Init capture
//        st.cap = std::make_unique<WgcRoiCapturer>();
//        if (!st.cap->StartMonitorRoi(mon, roiLocal, 20)) {
//            ShutdownOwnedObjects_(st);
//            st.running.store(false);
//            return;
//        }
//
//        // 4) Init TextHandler
//        TextHandler::Options th_opt;
//        th_opt.com_port = cfg.com_name;
//        th_opt.baud = cfg.baud;
//        th_opt.addr = (uint8_t)cfg.addr;
//        th_opt.wait_ack = cfg.wait_ack;
//        th_opt.debug = cfg.debug;
//        th_opt.trigger_vk = (UINT)cfg.trigger_vk;
//        th_opt.switch_vk = (UINT)cfg.switch_vk;
//        th_opt.auto_type = cfg.auto_type;
//        th_opt.user_config = cfg.user_keybinds_path;
//
//        st.handler = std::make_unique<TextHandler>(th_opt);
//        st.handler->SetMappedKey("1");
//
//        // 5) OCR thread
//        st.ocr_thread = std::thread([&st]() {
//            uint64_t last_frame_id = 0;
//
//            while (!st.stop_flag.load()) {
//                if (!st.cap || !st.pipeline || !st.handler) {
//                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
//                    continue;
//                }
//
//                BgraFrame f;
//                if (!st.cap->TryGetLatest(f)) {
//                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
//                    continue;
//                }
//                if (f.frame_id == last_frame_id) {
//                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
//                    continue;
//                }
//                last_frame_id = f.frame_id;
//
//                cv::Mat bgra = BgraFrameToMat_(f);
//                cv::Mat bgr;
//                cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
//
//                try {
//                    auto texts = recognition_image_(*st.pipeline, bgr);
//                    st.handler->set_texts(std::move(texts));
//                }
//                catch (...) {
//                    // swallow
//                }
//
//                std::this_thread::sleep_for(std::chrono::milliseconds(200));
//            }
//            });
//
//        // 6) Run TextHandler loop (blocking)
//        // English comment:
//        // Stop() will call handler->Stop(), which should exit Start() loop.
//        st.handler->Start();
//
//        // 7) Shutdown
//        st.stop_flag.store(true);
//        ShutdownOwnedObjects_(st);
//        st.running.store(false);
//    }
//
//    bool Start(const std::string& json_config_utf8)
//    {
//        std::lock_guard<std::mutex> lk(g_state.mu);
//        if (g_state.running.load()) {
//            return false;
//        }
//
//        RuntimeConfig cfg;
//        std::string err;
//        if (!ParseConfig_(json_config_utf8, cfg, err)) {
//            std::cerr << "Start() config error: " << err << "\n";
//            return false;
//        }
//
//        g_state.stop_flag.store(false);
//        g_state.stop_reason.store(0);
//        g_state.running.store(true);
//
//        g_state.main_thread = std::thread([cfg]() mutable {
//            MainThread_(cfg);
//            });
//
//        return true;
//    }
//
//    void Stop(int reason)
//    {
//        std::unique_lock<std::mutex> lk(g_state.mu);
//        if (!g_state.running.load()) return;
//
//        g_state.stop_reason.store(reason);
//        g_state.stop_flag.store(true);
//
//        if (g_state.handler) {
//            g_state.handler->Stop();
//        }
//
//        lk.unlock();
//
//        if (g_state.main_thread.joinable()) {
//            g_state.main_thread.join();
//        }
//
//        // Ensure final state
//        std::lock_guard<std::mutex> lk2(g_state.mu);
//        g_state.running.store(false);
//    }
//
//    bool IsRunning()
//    {
//        return g_state.running.load();
//    }
//
//} // namespace wowapp
//
//// Optional: standalone main for testing
//// #define WOWAPP_STANDALONE to enable.
////
//// When enabled, it expects a config.json path as argv[1] and reads it to start.
////
//// #ifdef WOWAPP_STANDALONE
//// int main(int argc, char** argv) { ... }
//// #endif
