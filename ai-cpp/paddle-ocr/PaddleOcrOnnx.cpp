//#include "./PaddleOcrOnnx.h"
//
//#include <chrono>
//#include <regex>
//#include <algorithm>
//
//namespace {
//
//    static int64_t NowMs()
//    {
//        using namespace std::chrono;
//        return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
//    }
//
//    // English:
//    // Best-effort extraction helpers (you can harden these with your UI text patterns).
//    static bool ExtractFirstFloat(const std::string& s, float& out)
//    {
//        static const std::regex re(R"(([0-9]+(?:\.[0-9]+)?))");
//        std::smatch m;
//        if (!std::regex_search(s, m, re))
//            return false;
//        out = std::stof(m[1].str());
//        return true;
//    }
//
//} // namespace
//
//PaddleOcrDispathcer::PaddleOcrDispathcer(const Config& cfg)
//{
//    Init(cfg);
//}
//
//bool PaddleOcrDispathcer::Init(const Config& cfg)
//{
//    std::lock_guard<std::mutex> lk(mu_);
//    cfg_ = cfg;
//    last_error_.clear();
//    last_trigger_tick_ms_ = 0;
//
//    if (!ocr_.Init(cfg_.ocr)) {
//        last_error_ = "PaddleOcr init failed";
//        return false;
//    }
//    return true;
//}
//
//void PaddleOcrDispathcer::OnEvent(std::function<void(const Event&)> cb)
//{
//    std::lock_guard<std::mutex> lk(mu_);
//    cb_ = std::move(cb);
//}
//
//std::string PaddleOcrDispathcer::GetLastError() const
//{
//    std::lock_guard<std::mutex> lk(mu_);
//    return last_error_;
//}
//
//std::string PaddleOcrDispathcer::BuildTopDownText(const std::vector<OcrItem>& items, bool join_with_newline)
//{
//    // English:
//    // PaddleOcr::Run() already sorts items roughly top-down in our implementation,
//    // but we don't rely on it too much. We simply join in current order.
//    std::string out;
//    for (const auto& it : items) {
//        if (it.text.empty())
//            continue;
//        if (!out.empty())
//            out += (join_with_newline ? "\n" : " ");
//        out += it.text;
//    }
//    return out;
//}
//
//bool PaddleOcrDispathcer::OnFrame(const BgraFrame& frame)
//{
//    std::function<void(const Event&)> cb;
//    Config cfg_snapshot;
//
//    {
//        std::lock_guard<std::mutex> lk(mu_);
//        cb = cb_;
//        cfg_snapshot = cfg_;
//    }
//
//    const int64_t now = NowMs();
//
//    // English:
//    // Throttle: If too frequent, do nothing and return false.
//    {
//        std::lock_guard<std::mutex> lk(mu_);
//        if (last_trigger_tick_ms_ != 0 &&
//            (now - last_trigger_tick_ms_) < cfg_snapshot.min_trigger_interval_ms) {
//            return false;
//        }
//    }
//
//    Event ev;
//    ev.frame_id = frame.frame_id;
//
//    // English:
//    // Prefer Run() to get items/boxes; then build topdown text.
//    std::string err;
//    std::vector<OcrItem> items;
//    const bool ok = ocr_.Run(frame, items, err);
//    if (!ok) {
//        ev.ocr_ok = false;
//        ev.ocr_err = err;
//        ev.parse_ok = false;
//        ev.parse_reason = "ocr-failed";
//
//        // English:
//        // Do NOT advance throttle timestamp on failure (avoid "failure lock").
//        {
//            std::lock_guard<std::mutex> lk(mu_);
//            last_error_ = err;
//        }
//
//        if (cb)
//            cb(ev);
//        return false;
//    }
//
//    // English:
//    // Success: commit throttle timestamp now.
//    {
//        std::lock_guard<std::mutex> lk(mu_);
//        last_trigger_tick_ms_ = now;
//        last_error_.clear();
//    }
//
//    ev.ocr_ok = true;
//    ev.ocr_items = std::move(items);
//    ev.ocr_text_topdown = BuildTopDownText(ev.ocr_items, cfg_snapshot.join_lines_with_newline);
//
//    if (cfg_snapshot.enable_parse) {
//        ev.parse_ok = ParseFromText(ev.ocr_text_topdown, ev);
//        if (!ev.parse_ok && ev.parse_reason.empty())
//            ev.parse_reason = "no-match";
//    }
//    else {
//        ev.parse_ok = false;
//        ev.parse_reason = "parse-disabled";
//    }
//
//    if (cb)
//        cb(ev);
//
//    return true;
//}
//
//bool PaddleOcrDispathcer::ParseFromText(const std::string& text, Event& ev)
//{
//    // English:
//    // This is intentionally best-effort. Replace these heuristics with your exact UI strings.
//    // We only set flags if we see keywords.
//
//    const std::string& t = text;
//    bool any = false;
//
//    // Spell states
//    if (t.find("施法") != std::string::npos || t.find("Casting") != std::string::npos) {
//        ev.spell.is_casting = true;
//        any = true;
//    }
//    if (t.find("冷却") != std::string::npos ||
//        t.find("CD") != std::string::npos ||
//        t.find("Cooldown") != std::string::npos) {
//        ev.spell.is_on_cooldown = true;
//        any = true;
//
//        float v = 0.0f;
//        if (ExtractFirstFloat(t, v))
//            ev.spell.cooldown_remaining_sec = v;
//    }
//    if (t.find("就绪") != std::string::npos || t.find("Ready") != std::string::npos) {
//        ev.spell.is_ready = true;
//        any = true;
//    }
//    if (t.find("沉默") != std::string::npos || t.find("Silence") != std::string::npos) {
//        ev.spell.is_silenced = true;
//        any = true;
//
//        float v = 0.0f;
//        if (ExtractFirstFloat(t, v))
//            ev.spell.silenced_remaining_sec = v;
//    }
//
//    // Target states
//    if (t.find("目标") != std::string::npos || t.find("Target") != std::string::npos) {
//        ev.target.has_target = true;
//        any = true;
//    }
//    if (t.find("目标施法") != std::string::npos || t.find("Target Casting") != std::string::npos) {
//        ev.target.target_is_casting = true;
//        any = true;
//    }
//
//    ev.parse_reason = any ? "ok" : "no-signal";
//    return any;
//}
