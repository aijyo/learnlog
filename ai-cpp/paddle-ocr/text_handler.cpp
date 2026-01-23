#include "text_handler.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>

void WowTextHandler::SetCallback(Callback cb) {
    std::lock_guard<std::mutex> lk(mu_);
    cb_ = std::move(cb);
}

absl::Status WowTextHandler::ProcessByTextHander(OcrFrameResult& frame) const {
    ParsedUi parsed;

    // 1) Reconstruct lines from OCR tokens (top-down).
    auto lines = ReconstructLinesTopDown(frame.items, cfg_);

    // Fill raw l1~l5 (missing lines allowed).
    if (lines.size() > 0) parsed.l1 = lines[0];
    if (lines.size() > 1) parsed.l2 = lines[1];
    if (lines.size() > 2) parsed.l3 = lines[2];
    if (lines.size() > 3) parsed.l4 = lines[3];
    if (lines.size() > 4) parsed.l5 = lines[4];

    // 2) Also fill merged_text for debug/logging.
    {
        std::ostringstream oss;
        for (size_t i = 0; i < lines.size(); ++i) {
            oss << lines[i];
            if (i + 1 < lines.size()) oss << "\n";
        }
        frame.merged_text = oss.str();
    }

    // 3) Parse each line (best-effort).
    // English:
    // We do NOT fail hard if some line cannot be parsed,
    // because OCR can be noisy; we only fail if the result is totally unusable.
    bool ok1 = ParseLine1(parsed.l1, parsed);
    bool ok2 = ParseLine2(parsed.l2, parsed);
    bool ok3 = ParseLine3(parsed.l3, parsed);
    bool ok4 = ParseLine4Target(parsed.l4, parsed);
    bool ok5 = ParseLine5Silence(parsed.l5, parsed);

    // Minimal sanity: slot & spell are most useful for your automation.
    if (!ok1) {
        return absl::InvalidArgumentError("UI.l1 parse failed (slot/type/spell not found). merged_text=\n" + frame.merged_text);
    }

    // 4) Callback to your game logic.
    Callback cb;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cb = cb_;
    }
    if (cb) cb(parsed);

    // Best-effort parsing should still return OK if l1 is ok.
    (void)ok2; (void)ok3; (void)ok4; (void)ok5;
    return absl::OkStatus();
}

// ---------------------------
// Line reconstruction
// ---------------------------
static inline float CenterY(const cv::Rect& r) {
    return r.y + r.height * 0.5f;
}
static inline float CenterX(const cv::Rect& r) {
    return r.x + r.width * 0.5f;
}

std::vector<std::string> WowTextHandler::ReconstructLinesTopDown(
    const std::vector<OcrInstance>& items,
    const Config& cfg)
{
    struct Token {
        std::string text;
        cv::Rect bbox;
        float score = 0.0f;
    };

    std::vector<Token> tokens;
    tokens.reserve(items.size());
    for (const auto& it : items) {
        if (cfg.min_rec_score > 0.0f && it.rec_score < cfg.min_rec_score) continue;
        if (it.text.empty()) continue;
        Token t;
        t.text = it.text;
        t.bbox = it.bbox;
        t.score = it.rec_score;
        tokens.push_back(std::move(t));
    }

    if (tokens.empty()) return {};

    // Sort by y then x.
    std::sort(tokens.begin(), tokens.end(), [](const Token& a, const Token& b) {
        if (a.bbox.y != b.bbox.y) return a.bbox.y < b.bbox.y;
        return a.bbox.x < b.bbox.x;
        });

    // Estimate typical line height.
    std::vector<int> heights;
    heights.reserve(tokens.size());
    for (auto& t : tokens) heights.push_back(std::max(1, t.bbox.height));
    std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
    float median_h = static_cast<float>(heights[heights.size() / 2]);
    float y_tol = std::max(2.0f, median_h * cfg.line_merge_y_tolerance_mul);

    // Cluster into lines by centerY proximity.
    struct Line {
        float cy = 0.0f;
        std::vector<Token> parts;
    };

    std::vector<Line> lines;
    for (const auto& tk : tokens) {
        float cy = CenterY(tk.bbox);

        int best_i = -1;
        float best_d = 1e9f;
        for (int i = 0; i < (int)lines.size(); ++i) {
            float d = std::abs(lines[i].cy - cy);
            if (d < best_d) { best_d = d; best_i = i; }
        }

        if (best_i >= 0 && best_d <= y_tol) {
            // Merge into existing line.
            auto& ln = lines[best_i];
            ln.parts.push_back(tk);
            // Update cy (running average).
            ln.cy = (ln.cy * (ln.parts.size() - 1) + cy) / (float)ln.parts.size();
        }
        else {
            // New line.
            Line ln;
            ln.cy = cy;
            ln.parts.push_back(tk);
            lines.push_back(std::move(ln));
        }
    }

    // Sort lines by cy (top -> down).
    std::sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) {
        return a.cy < b.cy;
        });

    // Convert each line's tokens into a string (left -> right).
    std::vector<std::string> out;
    out.reserve(lines.size());

    for (auto& ln : lines) {
        std::sort(ln.parts.begin(), ln.parts.end(), [](const Token& a, const Token& b) {
            if (a.bbox.x != b.bbox.x) return a.bbox.x < b.bbox.x;
            return a.bbox.width < b.bbox.width;
            });

        std::ostringstream oss;
        for (size_t i = 0; i < ln.parts.size(); ++i) {
            std::string s = ln.parts[i].text;
            if (s.empty()) continue;
            if (i > 0) oss << " ";
            oss << s;
        }

        auto line = NormalizeLine(oss.str());
        if (!line.empty()) out.push_back(line);

        if ((int)out.size() >= cfg.max_ui_lines) break; // UI only needs top 5 lines
    }

    return out;
}

std::string WowTextHandler::Trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

std::string WowTextHandler::NormalizeLine(const std::string& s) {
    // English:
    // OCR may produce inconsistent spaces. We:
    // 1) collapse multiple spaces
    // 2) remove spaces around some punctuations (= : / ( ) %)
    std::string t = Trim(s);
    if (t.empty()) return t;

    // Collapse spaces.
    std::string collapsed;
    collapsed.reserve(t.size());
    bool last_space = false;
    for (char c : t) {
        if (std::isspace((unsigned char)c)) {
            if (!last_space) collapsed.push_back(' ');
            last_space = true;
        }
        else {
            collapsed.push_back(c);
            last_space = false;
        }
    }

    // Remove spaces around punctuation tokens commonly used by lua formatting.
    auto strip_around = [&](char p) {
        std::string r;
        r.reserve(collapsed.size());
        for (size_t i = 0; i < collapsed.size(); ++i) {
            char c = collapsed[i];
            if (c == p) {
                // remove trailing space in r
                while (!r.empty() && r.back() == ' ') r.pop_back();
                r.push_back(p);
                // skip following spaces
                size_t j = i + 1;
                while (j < collapsed.size() && collapsed[j] == ' ') ++j;
                i = j - 1;
            }
            else {
                r.push_back(c);
            }
        }
        collapsed = std::move(r);
        };

    strip_around('=');
    strip_around(':');
    strip_around('/');
    strip_around('(');
    strip_around(')');
    strip_around('%');

    return Trim(collapsed);
}

// ---------------------------
// Parsing helpers
// ---------------------------
bool WowTextHandler::ParseYN(const std::string& v, bool& out) {
    if (v == "Y" || v == "y" || v == "1") { out = true; return true; }
    if (v == "N" || v == "n" || v == "0") { out = false; return true; }
    return false;
}

float WowTextHandler::ParseTimeSeconds(const std::string& token) {
    // token example: "1.2s" or "0.5m" or "0"
    std::string t = token;
    if (t.empty()) return 0.0f;
    if (t == "0") return 0.0f;

    float mul = 1.0f;
    if (!t.empty() && (t.back() == 's' || t.back() == 'S')) {
        mul = 1.0f;
        t.pop_back();
    }
    else if (!t.empty() && (t.back() == 'm' || t.back() == 'M')) {
        mul = 60.0f;
        t.pop_back();
    }

    try {
        return std::stof(t) * mul;
    }
    catch (...) {
        return 0.0f;
    }
}

WowTextHandler::ParsedCooldown WowTextHandler::ParseCooldownToken(const std::string& token) {
    ParsedCooldown cd;
    if (token.empty() || token == "0") {
        cd.on_cd = false;
        return cd;
    }

    // "1.2s/30.0s"
    auto pos = token.find('/');
    if (pos == std::string::npos) {
        // tolerate: "1.2s"
        cd.remain_sec = ParseTimeSeconds(token);
        cd.duration_sec = 0.0f;
        cd.on_cd = (cd.remain_sec > 0.0f);
        return cd;
    }

    auto a = token.substr(0, pos);
    auto b = token.substr(pos + 1);
    cd.remain_sec = ParseTimeSeconds(a);
    cd.duration_sec = ParseTimeSeconds(b);
    cd.on_cd = (cd.remain_sec > 0.0f && cd.duration_sec > 0.0f);
    return cd;
}

WowTextHandler::ParsedCharges WowTextHandler::ParseChargesToken(const std::string& token) {
    ParsedCharges chg;
    if (token.empty() || token == "-") return chg;

    // Example: "1/2(3.4s)" or "1/2"
    // cur/max
    auto slash = token.find('/');
    if (slash == std::string::npos) return chg;

    std::string cur_s = token.substr(0, slash);

    std::string rest = token.substr(slash + 1);
    std::string max_s = rest;

    // optional "(time)"
    auto lp = rest.find('(');
    if (lp != std::string::npos) {
        max_s = rest.substr(0, lp);
        auto rp = rest.find(')', lp + 1);
        if (rp != std::string::npos) {
            auto time_s = rest.substr(lp + 1, rp - (lp + 1));
            chg.recover_remain_sec = ParseTimeSeconds(time_s);
        }
    }

    try {
        chg.cur = std::stoi(cur_s);
        chg.max = std::stoi(max_s);
        chg.valid = true;
    }
    catch (...) {
        return ParsedCharges{};
    }

    return chg;
}

bool WowTextHandler::ParseLine1(const std::string& l1, ParsedUi& out) {
    // lua:
    // "slot=%d  type=%s(%s)  spell=%s  %s"
    // After NormalizeLine:
    // "slot=68 type=spell(123) spell=456 Fireball"
    std::string s = NormalizeLine(l1);
    if (s.empty()) return false;

    // Use regex for robustness.
    // slot=(\d+)\s+type=([^\s(]+)\(([^)]+)\)\s+spell=([^\s]+)\s+(.*)
    static const std::regex re(R"(slot=(\d+)\s+type=([^\s(]+)\(([^)]+)\)\s+spell=([^\s]+)\s*(.*))",
        std::regex::icase);
    std::smatch m;
    if (!std::regex_match(s, m, re)) return false;

    out.action_slot = std::stoi(m[1].str());
    out.action_type = m[2].str();
    out.action_id = m[3].str();

    auto spell_s = m[4].str();
    if (spell_s == "-" || spell_s == "0") out.spell_id = 0;
    else {
        try { out.spell_id = std::stoi(spell_s); }
        catch (...) { out.spell_id = 0; }
    }

    out.spell_name = Trim(m[5].str());
    return true;
}

bool WowTextHandler::ParseLine2(const std::string& l2, ParsedUi& out) {
    // lua:
    // "CD=%s  CHG=%s  usable=%s mana?=%s range=%s cur=%s atk=%s rep=%s"
    std::string s = NormalizeLine(l2);
    if (s.empty()) return false;

    // Example normalized:
    // "CD=1.2s/30.0s CHG=1/2(3.4s) usable=Y mana?=N range=in cur=N atk=N rep=N"
    static const std::regex re(
        R"(CD=([^\s]+)\s+CHG=([^\s]+)\s+usable=([YN])\s+mana\?=([YN])\s+range=([^\s]+)\s+cur=([YN])\s+atk=([YN])\s+rep=([YN]))",
        std::regex::icase
    );

    std::smatch m;
    if (!std::regex_match(s, m, re)) return false;

    out.cd = ParseCooldownToken(m[1].str());
    out.charges = ParseChargesToken(m[2].str());

    ParseYN(m[3].str(), out.usable);
    ParseYN(m[4].str(), out.not_enough);
    out.range = m[5].str();
    ParseYN(m[6].str(), out.is_current);
    ParseYN(m[7].str(), out.is_attack);
    ParseYN(m[8].str(), out.is_auto_repeat);

    return true;
}

bool WowTextHandler::ParseLine3(const std::string& l3, ParsedUi& out) {
    // lua:
    // "GCD=%s  playerCast=%s  combat=%s mount=%s dead=%s"
    // playerCast contains spaces, so parse by anchors.
    std::string s = NormalizeLine(l3);
    if (s.empty()) return false;

    // Find anchors.
    auto pos_combat = s.find(" combat=");
    if (pos_combat == std::string::npos) return false;

    std::string left = s.substr(0, pos_combat);
    std::string right = s.substr(pos_combat + 1); // skip leading space

    // left: "GCD=... playerCast=..."
    // right: "combat=Y mount=N dead=N"
    auto pos_pc = left.find(" playerCast=");
    if (pos_pc == std::string::npos) return false;

    std::string gcd_part = left.substr(0, pos_pc);
    std::string cast_part = left.substr(pos_pc + 1); // "playerCast=..."

    // gcd
    auto pos_eq = gcd_part.find("GCD=");
    if (pos_eq == std::string::npos) return false;
    out.gcd = ParseCooldownToken(gcd_part.substr(pos_eq + 4));

    // playerCast
    // cast_part example:
    // "playerCast=NONE"
    // "playerCast=CAST:xxx(123):1.2s THIS"
    {
        auto pos = cast_part.find("playerCast=");
        std::string v = (pos == std::string::npos) ? "" : cast_part.substr(pos + 11);
        v = Trim(v);

        if (v == "NONE" || v.empty()) {
            out.player_cast.active = false;
            out.player_cast.mode = "NONE";
        }
        else {
            out.player_cast.active = true;

            // Split last token "THIS"/"OTHER" if exists.
            std::string tail;
            auto sp = v.rfind(' ');
            if (sp != std::string::npos) {
                tail = v.substr(sp + 1);
                if (tail == "THIS" || tail == "OTHER") {
                    out.player_cast.match_this_spell = (tail == "THIS");
                    v = v.substr(0, sp);
                }
            }

            // v: "CAST:Name(123):1.2s" or "CHAN:Name(123):0.8s"
            // mode before first ':'
            auto c1 = v.find(':');
            if (c1 != std::string::npos) {
                out.player_cast.mode = v.substr(0, c1);
                std::string rest = v.substr(c1 + 1);

                // name(spellid)
                auto lp = rest.rfind('(');
                auto rp = rest.rfind(')');
                if (lp != std::string::npos && rp != std::string::npos && rp > lp) {
                    out.player_cast.name = rest.substr(0, lp);
                    try { out.player_cast.spell_id = std::stoi(rest.substr(lp + 1, rp - lp - 1)); }
                    catch (...) { out.player_cast.spell_id = 0; }
                }

                // optional ":time"
                auto c2 = rest.rfind(':');
                if (c2 != std::string::npos) {
                    std::string time_s = rest.substr(c2 + 1);
                    out.player_cast.remain_sec = ParseTimeSeconds(time_s);
                }
            }
        }
    }

    // right: combat/mount/dead
    // "combat=Y mount=N dead=N"
    {
        static const std::regex re(R"(combat=([YN])\s+mount=([YN])\s+dead=([YN]))", std::regex::icase);
        std::smatch m;
        if (std::regex_search(right, m, re)) {
            ParseYN(m[1].str(), out.in_combat);
            ParseYN(m[2].str(), out.is_mounted);
            ParseYN(m[3].str(), out.is_dead);
        }
    }

    return true;
}

bool WowTextHandler::ParseLine4Target(const std::string& l4, ParsedUi& out) {
    // lua:
    // no target -> "target: -"
    // else:
    // "target:NAME  HP=88%  d=1  atk=Y enemy=Y elite  aura=1/2  cast=CAST:xxx(123):1.2s"
    std::string s = NormalizeLine(l4);
    if (s.empty()) return false;

    if (s == "target:-" || s == "target: -" || s.find("target:-") != std::string::npos) {
        out.target.exists = false;
        return true;
    }

    if (s.find("target:") != 0) return false;
    out.target.exists = true;

    // name until " HP="
    auto pos_hp = s.find(" HP=");
    if (pos_hp == std::string::npos) return false;

    out.target.name = s.substr(strlen("target:"), pos_hp - strlen("target:"));
    out.target.name = Trim(out.target.name);

    // HP
    {
        static const std::regex re_hp(R"(HP=(\d+)%))", std::regex::icase);
        std::smatch m;
        if (std::regex_search(s, m, re_hp)) {
            out.target.hp_pct = std::stoi(m[1].str());
        }
    }

    // dist tier token is right after "HP=xx%" (lua prints distTier as returned string, e.g. "d=1" or "-")
    // We just capture the next token after "HP=..%"
    {
        static const std::regex re_dist(R"(HP=\d+%\s+([^\s]+))", std::regex::icase);
        std::smatch m;
        if (std::regex_search(s, m, re_dist)) {
            out.target.dist_tier = m[1].str();
        }
    }

    // atk/enemy
    {
        static const std::regex re_flags(R"(atk=([YN])\s+enemy=([YN]))", std::regex::icase);
        std::smatch m;
        if (std::regex_search(s, m, re_flags)) {
            ParseYN(m[1].str(), out.target.can_attack);
            ParseYN(m[2].str(), out.target.is_enemy);
        }
    }

    // classification: token after "enemy=Y/N"
    {
        static const std::regex re_class(R"(enemy=[YN]\s+([^\s]+))", std::regex::icase);
        std::smatch m;
        if (std::regex_search(s, m, re_class)) {
            out.target.classification = m[1].str();
        }
    }

    // aura
    {
        static const std::regex re_aura(R"(aura=(\d+)\/(\d+))", std::regex::icase);
        std::smatch m;
        if (std::regex_search(s, m, re_aura)) {
            out.target.buffs = std::stoi(m[1].str());
            out.target.debuffs = std::stoi(m[2].str());
        }
    }

    // cast=...
    {
        auto pos = s.find(" cast=");
        if (pos != std::string::npos) {
            std::string cast_s = s.substr(pos + 1); // "cast=..."
            // cast=NONE or cast=CAST:xxx(123):1.2s
            if (cast_s.find("cast=NONE") != std::string::npos) {
                out.target.cast.active = false;
                out.target.cast.mode = "NONE";
            }
            else {
                // Extract after "cast="
                auto pe = cast_s.find("cast=");
                std::string v = cast_s.substr(pe + 5);
                out.target.cast.active = true;

                // v: "CAST:Name(123):1.2s"
                auto c1 = v.find(':');
                if (c1 != std::string::npos) {
                    out.target.cast.mode = v.substr(0, c1);
                    std::string rest = v.substr(c1 + 1);

                    auto lp = rest.rfind('(');
                    auto rp = rest.rfind(')');
                    if (lp != std::string::npos && rp != std::string::npos && rp > lp) {
                        out.target.cast.name = rest.substr(0, lp);
                        try { out.target.cast.spell_id = std::stoi(rest.substr(lp + 1, rp - lp - 1)); }
                        catch (...) { out.target.cast.spell_id = 0; }
                    }

                    auto c2 = rest.rfind(':');
                    if (c2 != std::string::npos) {
                        out.target.cast.remain_sec = ParseTimeSeconds(rest.substr(c2 + 1));
                    }
                }
            }
        }
    }

    return true;
}

bool WowTextHandler::ParseLine5Silence(const std::string& l5, ParsedUi& out) {
    // lua:
    // "silence=0"
    // or "silence=1.2s Kick"
    std::string s = NormalizeLine(l5);
    if (s.empty()) return false;

    if (s == "silence=0") {
        out.silence.active = false;
        out.silence.remain_sec = 0.0f;
        out.silence.name.clear();
        return true;
    }

    if (s.find("silence=") != 0) return false;

    // split: "silence=TIME NAME..."
    auto sp = s.find(' ');
    std::string left = (sp == std::string::npos) ? s : s.substr(0, sp);
    std::string right = (sp == std::string::npos) ? "" : Trim(s.substr(sp + 1));

    auto eq = left.find('=');
    if (eq == std::string::npos) return false;

    std::string time_s = left.substr(eq + 1);
    out.silence.active = true;
    out.silence.remain_sec = ParseTimeSeconds(time_s);
    out.silence.name = right;
    return true;
}
