#pragma once
#include <string>
#include <vector>
#include <functional>
#include <mutex>

#include "./PaddleOcr.h"

// English:
// Parsed spell/target info structures are kept compatible with your existing UI logic.
struct ParsedSpellState
{
    bool is_casting = false;
    bool is_on_cooldown = false;
    bool is_ready = false;
    bool is_silenced = false;

    float cooldown_remaining_sec = 0.0f;
    float cast_remaining_sec = 0.0f;
    float silenced_remaining_sec = 0.0f;
};

struct ParsedTargetState
{
    bool has_target = false;
    bool target_is_casting = false;

    std::string target_name;
    std::string target_cast_spell;
    float target_cast_remaining_sec = 0.0f;
};

// English:
// New dispatcher name required by user: PaddleOcrDispathcer (typo kept intentionally).
class PaddleOcrDispathcer
{
public:
    struct Config
    {
        // English:
        // PaddleOCR config
        PaddleOcr::Config ocr;

        // English:
        // Throttle callback/ocr invocation rate to avoid high CPU usage.
        int min_trigger_interval_ms = 200;

        // English:
        // If true, ParseFromText() will run after OCR and set parse fields.
        bool enable_parse = true;

        // English:
        // Join lines by '\n' when composing topdown text.
        bool join_lines_with_newline = true;
    };

    struct Event
    {
        int64_t frame_id = 0;

        // English:
        // OCR results
        bool ocr_ok = false;
        std::string ocr_err;

        std::vector<OcrItem> ocr_items;       // includes boxes + score + text
        std::string ocr_text_topdown;         // joined by order (top-down best-effort)

        // English:
        // Parse results (optional)
        bool parse_ok = false;
        std::string parse_reason;

        ParsedSpellState spell;
        ParsedTargetState target;
    };

public:
    PaddleOcrDispathcer() = default;
    explicit PaddleOcrDispathcer(const Config& cfg);

    bool Init(const Config& cfg);

    // English:
    // Feed a BGRA frame into dispatcher.
    // Returns true if OCR ran (not throttled) AND OCR succeeded.
    bool OnFrame(const BgraFrame& frame);

    void OnEvent(std::function<void(const Event&)> cb);

    std::string GetLastError() const;

private:
    // English:
    // Build top-down text from OCR items.
    static std::string BuildTopDownText(const std::vector<OcrItem>& items, bool join_with_newline);

    bool ParseFromText(const std::string& text, Event& ev);

private:
    Config cfg_;
    PaddleOcr ocr_;

    std::function<void(const Event&)> cb_;
    mutable std::mutex mu_;

    std::string last_error_;

    int64_t last_trigger_tick_ms_ = 0;
};

// English:
// Backward compatible alias: your old name still works.
using PaddleOcrImageDispatcher = PaddleOcrDispathcer;
