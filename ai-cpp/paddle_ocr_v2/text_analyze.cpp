//#include "text_analyze.h"
//
//#include <algorithm>
//#include <cctype>
//#include <string>
//#include <utility>
//#include <vector>
//
//namespace {
//
//    // ------------------------------
//    // String helpers
//    // ------------------------------
//
//    // Trim leading/trailing whitespace.
//    std::string Trim(std::string s) {
//        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
//        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
//        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
//        return s;
//    }
//
//    // Return true if a character can be part of a key.
//    // We accept uppercase letters/digits/underscore, and allow '0' due to OCR confusion.
//    bool IsKeyChar(unsigned char c) {
//        if (c >= 'A' && c <= 'Z') return true;
//        if (c >= '0' && c <= '9') return true;
//        if (c == '_') return true;
//        return false;
//    }
//
//    // Detect lines like "SPELL=" or "NAME=" with empty value.
//    // We use it to merge with the next line.
//    bool EndsWithKeyEquals(const std::string& line) {
//        std::string t = Trim(line);
//        if (t.size() < 2) return false;
//        if (t.back() != '=') return false;
//
//        // Verify the key part (before '=') looks like a key.
//        const size_t eq = t.rfind('=');
//        if (eq == std::string::npos || eq == 0) return false;
//
//        // Key should be mostly key chars.
//        for (size_t i = 0; i < eq; ++i) {
//            unsigned char c = static_cast<unsigned char>(t[i]);
//            // Allow OCR '0' in key and normal uppercase/digit/underscore.
//            if (!IsKeyChar(c)) return false;
//        }
//        return true;
//    }
//
//    // Next line is likely a standalone value if it does not contain '=' and is not too long.
//    bool LooksLikeStandaloneValue(const std::string& line) {
//        std::string t = Trim(line);
//        if (t.empty()) return false;
//        if (t.find('=') != std::string::npos) return false;
//        if (t.size() > 96) return false;
//        return true;
//    }
//
//    // Normalize OCR-confusable keys: SL0T -> SLOT, etc.
//    std::string NormalizeKey(std::string key) {
//        // Replace '0' with 'O' inside key, because OCR often mixes them.
//        for (char& c : key) {
//            if (c == '0') c = 'O';
//        }
//
//        // Common expected keys in your overlay. Add more if you see OCR variants.
//        if (key == "SLOT" || key == "SLOT") return "SLOT";
//        if (key == "SL0T" || key == "SLOT") return "SLOT"; // extra safety
//        if (key == "SLO T") return "SLOT";                  // rare spacing glitch
//
//        // Keep other keys unchanged.
//        return key;
//    }
//
//    // Fix O/o -> 0 for numeric-like values.
//    void FixNumericO0(std::string& v) {
//        for (char& c : v) {
//            if (c == 'O' || c == 'o') c = '0';
//        }
//    }
//
//    // Normalize value based on key type.
//    std::string NormalizeValueForKey(const std::string& key, std::string value) {
//        value = Trim(std::move(value));
//        if (value.empty()) return value;
//
//        // Keys that are typically numeric/float-ish in your output.
//        // This helps cases like "SCD=O.0O".
//        if (key == "SLOT" || key == "SPELL" || key == "REMAIN" || key == "SCD" ||
//            key == "PHP" || key == "BUF" || key == "DEBUF") {
//            FixNumericO0(value);
//        }
//        return value;
//    }
//
//    // Extract key-value pairs from a line without relying on whitespace.
//    // It finds all occurrences of "...KEY=VALUE..." where KEY is [A-Z0-9_]+ (allowing OCR '0'),
//    // then slices VALUE until the start of the next KEY=.
//    std::vector<std::pair<std::string, std::string>> ExtractKVs(const std::string& line) {
//        struct Hit {
//            size_t key_begin;
//            size_t eq_pos;
//            std::string key;
//        };
//
//        std::vector<Hit> hits;
//        hits.reserve(8);
//
//        // Find '=' positions then backtrack to get key.
//        for (size_t eq = 0; eq < line.size(); ++eq) {
//            if (line[eq] != '=') continue;
//
//            // Backtrack to find key end at eq-1.
//            if (eq == 0) continue;
//            size_t end = eq; // key ends at eq-1
//            size_t beg = eq;
//            while (beg > 0) {
//                unsigned char c = static_cast<unsigned char>(line[beg - 1]);
//                if (!IsKeyChar(c)) break;
//                --beg;
//            }
//
//            // Key length must be >= 2 to reduce false positives.
//            if (end <= beg || (end - beg) < 2) continue;
//
//            std::string raw_key = line.substr(beg, end - beg);
//
//            // Heuristic: prefer keys that look uppercase-ish (OCR may output digits too).
//            // We don't strictly enforce uppercase because OCR may insert '0'.
//            hits.push_back({ beg, eq, std::move(raw_key) });
//        }
//
//        if (hits.empty()) return {};
//
//        // Sort by eq_pos to compute value slices.
//        std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
//            return a.eq_pos < b.eq_pos;
//            });
//
//        std::vector<std::pair<std::string, std::string>> out;
//        out.reserve(hits.size());
//
//        for (size_t i = 0; i < hits.size(); ++i) {
//            const Hit& h = hits[i];
//
//            size_t value_begin = h.eq_pos + 1;
//            size_t value_end = (i + 1 < hits.size()) ? hits[i + 1].key_begin : line.size();
//
//            if (value_begin >= line.size()) continue;
//            if (value_begin > value_end) continue;
//
//            std::string key = NormalizeKey(h.key);
//            std::string val = line.substr(value_begin, value_end - value_begin);
//            val = NormalizeValueForKey(key, std::move(val));
//
//            if (!key.empty() && !val.empty()) {
//                out.emplace_back(std::move(key), std::move(val));
//            }
//        }
//
//        return out;
//    }
//
//} // namespace
//
//// ------------------------------
//// TextAnalyze implementation
//// ------------------------------
//
//TextAnalyze::TextAnalyze() {
//}
//
//// Parse a single (already repaired) line into kv_.
//// This function supports multiple KEY=VALUE pairs in one line.
//void TextAnalyze::parse_line(const std::string& line) {
//    const auto kvs = ExtractKVs(line);
//    for (const auto& kv : kvs) {
//        kv_[kv.first] = kv.second;
//    }
//}
//
//void TextAnalyze::set_texts(const std::vector<std::string>& lines)
//{
//    kv_.clear();
//
//    // 1) Clean: trim & drop empty lines
//    std::vector<std::string> cleaned;
//    cleaned.reserve(lines.size());
//    for (auto s : lines) {
//        s = Trim(std::move(s));
//        if (!s.empty()) cleaned.push_back(std::move(s));
//    }
//
//    // 2) Repair OCR line-splitting:
//    //    Merge "KEY=" with the next standalone value line (no '=').
//    std::vector<std::string> repaired;
//    repaired.reserve(cleaned.size());
//
//    for (size_t i = 0; i < cleaned.size(); ++i) {
//        std::string cur = cleaned[i];
//
//        if (EndsWithKeyEquals(cur) && (i + 1) < cleaned.size() && LooksLikeStandaloneValue(cleaned[i + 1])) {
//            // Merge without adding space to preserve compact format.
//            // If you prefer, you can add a single space: cur += " " + cleaned[i + 1];
//            cur += cleaned[i + 1];
//            ++i; // consume next line
//        }
//
//        repaired.push_back(std::move(cur));
//    }
//
//    // 3) Parse repaired lines
//    for (const auto& line : repaired) {
//        parse_line(line);
//    }
//
//    return;
//}
//
//std::string TextAnalyze::get_key(const std::string& key) const {
//    // Keep interface unchanged: caller can query original key names.
//    // We normalize common key OCR issues here too (optional but helpful).
//    std::string nk = NormalizeKey(key);
//
//    auto it = kv_.find(nk);
//    if (it == kv_.end()) return {};
//    return it->second;
//}
//
//bool TextAnalyze::has_key(const std::string& key) const {
//    std::string nk = NormalizeKey(key);
//    return kv_.find(nk) != kv_.end();
//}
//
//const std::unordered_map<std::string, std::string>& TextAnalyze::data() const {
//    return kv_;
//}
// TextAnalyze.cpp
#include "text_analyze.h"

#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace {

    // Fix common OCR confusions and filter chars.
    // Keep digits, '.', '=', and optionally '-' for negative numbers.
    static std::string NormalizeOcrText(const std::string& in, bool keep_minus = false) {
        std::string out;
        out.reserve(in.size());

        for (char ch : in) {
            unsigned char uch = static_cast<unsigned char>(ch);
            char c = static_cast<char>(uch);

            // Normalize common OCR mistakes
            switch (c) {
            case 'o': case 'O':
                c = '0'; break;
            case 'l': case 'I': case '|':
                c = '1'; break;
            case 'S':
                c = '5'; break;
            case 'B':
                c = '8'; break;
            default:
                break;
            }

            // Filter allowed characters
            if ((c >= '0' && c <= '9') || c == '.' || c == '=' || (keep_minus && c == '-')) {
                out.push_back(c);
            }
        }
        return out;
    }

    // Split by '=' and drop empty tokens.
    static std::vector<std::string> SplitNonEmpty(const std::string& s, char delim) {
        std::vector<std::string> parts;
        std::string token;
        std::stringstream ss(s);

        while (std::getline(ss, token, delim)) {
            if (!token.empty()) parts.push_back(token);
        }
        return parts;
    }

    // Remove leading/trailing dots (OCR sometimes produces ".00" or "0.00.")
    static void TrimDots(std::string& s) {
        while (!s.empty() && s.front() == '.') s.erase(s.begin());
        while (!s.empty() && s.back() == '.') s.pop_back();
    }

    // Validate a numeric token: digits with optional single dot and optional leading '-'.
    // This is stricter than std::stod, and avoids weird tokens.
    static bool IsValidNumberToken(const std::string& s, bool allow_minus = false) {
        if (s.empty()) return false;
        size_t i = 0;
        if (allow_minus && s[0] == '-') {
            if (s.size() == 1) return false;
            i = 1;
        }

        bool seen_digit = false;
        bool seen_dot = false;
        for (; i < s.size(); ++i) {
            char c = s[i];
            if (c >= '0' && c <= '9') {
                seen_digit = true;
                continue;
            }
            if (c == '.') {
                if (seen_dot) return false;
                seen_dot = true;
                continue;
            }
            return false;
        }
        return seen_digit;
    }

    // Build value as "b=c" from tokens[1], tokens[2]
    static std::string JoinTwoWithEq(const std::string& b, const std::string& c) {
        return b + "=" + c;
    }

} // namespace

TextAnalyze::TextAnalyze() = default;

void TextAnalyze::set_texts(const std::vector<std::string>& lines) {
    kv_.clear();
    kv_.reserve(lines.size());
    for (const auto& line : lines) {
        parse_line(line);
    }
}

std::string TextAnalyze::get_key(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) return std::string();
    return it->second;
}
//
//int TextAnalyze::target() const
//{
//    return target_;
//}
//
//std::string TextAnalyze::target_str(int target/* = -1*/) const
//{
//    if(target < 0) target = target_;
//
//    switch (target)
//    {
//    case 0:
//        return "NO_TARGET";
//    case 1:
//        return "FRIENDLY";
//    case 2:
//        return "HOSTILE_NPC";
//    case 3:
//        return "HOSTILE_PLAYER";
//    case 4:
//        return "NEUTRAL";
//    default:
//        return "UNKNOWN";
//    }
//}

bool TextAnalyze::has_key(const std::string& key) const {
    return kv_.find(key) != kv_.end();
}

const std::unordered_map<std::string, std::string>& TextAnalyze::data() const {
    return kv_;
}

void TextAnalyze::parse_line(const std::string& line) {
    // Keep minus disabled by default. Enable if you expect negative numbers.
    const bool keep_minus = false;

    // 1) Normalize and filter noise
    std::string norm = NormalizeOcrText(line, keep_minus);
    if (norm.empty()) return;

    // 2) Split by '='
    auto parts = SplitNonEmpty(norm, '=');
    if (parts.size() < 3) {
        printf("parse line faild: %s\n", norm.c_str());
        return;
    }

    // If OCR inserted extra '=' somehow, only take the first 3 numeric-like tokens.
    // Example: "185358==0.00=0.00" -> parts = ["185358","0.00","0.00"]
    // Example: "185358=0.00=0.00=..." -> we take first 3.
    std::string a = parts[0];
    std::string b = parts[1];
    std::string c = parts[2];
    ////0	没有目标
    ////    1	友方目标（Friend）
    ////    2	敌对 NPC
    ////    3	敌对 玩家
    ////    4	中立 / 不可攻击 / 其它
    //std::string d = parts[3];
    // target casting remain(for break)
    std::string e;
    if (parts.size() > 3)
    {
        e = parts[3];
    }

    TrimDots(a);
    TrimDots(b);
    TrimDots(c);
    //TrimDots(d);
    TrimDots(e);

    // 3) Validate tokens
    if (!IsValidNumberToken(a, keep_minus)) return;
    if (!IsValidNumberToken(b, keep_minus)) return;
    if (!IsValidNumberToken(c, keep_minus)) return;
    //if (!IsValidNumberToken(d, keep_minus)) return;
    if (!e.empty() && !IsValidNumberToken(e, keep_minus)) return;

    // 4) Store: key = a, value = "b=c"
    // This fits your existing get_key/has_key API.
    //kv_[a] = JoinTwoWithEq(b, c);

    kv_["spell"] = a;
    kv_["gcd"] = b;
    kv_["scd"] = c;
    //kv_["target"] = d;
    kv_["tremain"] = e;

    //target_ = d.empty() ? 0 : d[0] - '0';

}
