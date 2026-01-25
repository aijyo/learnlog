#pragma once
#define NOMINMAX
#include <windows.h>
#include <string>
#include <atomic>
#include <cstdint>
#include <vector>
#include <mutex>

#include "./text_analyze.h"
#include "./user_config.h"
#include "./common_def.h"
class KeyboardControl;

class TextHandler {
public:
    struct Options {
        std::string com_port = "COM3";
        int baud = 9600;
        uint8_t addr = 0x00;
        bool wait_ack = true;
        bool debug = false;
        utils::AutoMode auto_type = utils::AutoMode::kAssistant;
        float auto_time = 0.6; //600ms;
        // auto break
        bool auto_break = false;
        float break_time = 0.7; //  break when remain < 400ms, <0 mean break when < (gcd)
        std::string break_shortcut = "F5";

        // Trigger key: default '1'
        // Use VK_1 / '1' (0x31) for main keyboard number row.
        int trigger_vk = '1';
        bool ignore_trigger_case = false;
        int switch_vk = VK_F12;
        bool ignore_switch_case = false;

        // Optional: only active when foreground exe == target_exe (empty means any)
        std::wstring target_exe;
        std::string user_config = R"(D:\code\gitcode\PaddleOCR\deploy\cpp_infer\wow\user_keybinds.json)";
    };

public:
    explicit TextHandler(const Options& opt);
    ~TextHandler();

    TextHandler(const TextHandler&) = delete;
    TextHandler& operator=(const TextHandler&) = delete;

    // Set mapped key spec, e.g. "1", "CTRL+a", "ALT+tab"
    void SetMappedKey(const std::string& key_spec);

    void set_texts(std::vector<std::string>&& texts);
    // Start/Stop the hook + serial
    bool Start();
    void Stop();

    bool IsRunning() const { return running_.load(); }

private:
    bool OpenSerial_();
    void CloseSerial_();

    bool IsForegroundTarget_() const;

    // Hook management
    bool InstallHook_();
    void UninstallHook_();

    // Message loop (blocking)
    void RunMessageLoop_();

    // Hook callback
    static LRESULT CALLBACK LowLevelKeyboardProc_(int nCode, WPARAM wParam, LPARAM lParam);

    // Instance handler
    LRESULT OnKeyboardEvent_(int nCode, WPARAM wParam, LPARAM lParam);

    // Analyze OCR texts -> shortcut key spec (e.g. "CTRL+a", "ALT+tab").
    // Return true auto run
    bool AnalyzeShortcutFromTexts_(const std::vector<std::string>& texts);

    // Normalize and validate shortcut key spec.
    // Return empty if invalid / unsupported.
    std::string NormalizeShortcutSpec_(std::string s);

    void OnTexts_();
private:
    Options opt_;

    //bool hooking_ = false;
    std::atomic<bool> running_{ false };

    // Track whether we have pressed mapped key due to trigger key,
    // so we can release correctly.
    std::atomic<bool> mapped_down_from_trigger_{ false };

    void* hook_ = nullptr;          // HHOOK
    KeyboardControl* kb_ = nullptr; // owned in cpp

    // singleton instance for hook callback
    static TextHandler* s_instance_;

    UserConfig user_config_;
    TextAnalyze text_analyze_;
    // Thread-safe state shared between OCR thread and hook thread.
    std::mutex state_mu_;
    std::vector<std::string> latest_texts_;
    std::string new_shortcut_ = "1"; // keep your default mapping

    //spell info
    float gcd_ = .0f;
    float scd_ = .0f;
    uint64_t spellid_ = 0;
    //std::string  ctrl_type_;

    //
    // The thread id that owns RunMessageLoop_(). We post custom messages to it.
    DWORD msg_thread_id_ = 0;

    // English comment:
    // Custom message used to notify the message loop that OCR texts were updated.
    static constexpr UINT kMsgTextsUpdated_ = WM_APP + 0x4A31;
};
