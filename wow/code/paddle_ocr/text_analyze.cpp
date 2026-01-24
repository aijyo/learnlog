#include "text_analyze.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace {

    // ------------------------------
    // String helpers
    // ------------------------------

    // Trim leading/trailing whitespace.
    std::string Trim(std::string s) {
        auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        return s;
    }

    // Return true if a character can be part of a key.
    // We accept uppercase letters/digits/underscore, and allow '0' due to OCR confusion.
    bool IsKeyChar(unsigned char c) {
        if (c >= 'A' && c <= 'Z') return true;
        if (c >= '0' && c <= '9') return true;
        if (c == '_') return true;
        return false;
    }

    // Detect lines like "SPELL=" or "NAME=" with empty value.
    // We use it to merge with the next line.
    bool EndsWithKeyEquals(const std::string& line) {
        std::string t = Trim(line);
        if (t.size() < 2) return false;
        if (t.back() != '=') return false;

        // Verify the key part (before '=') looks like a key.
        const size_t eq = t.rfind('=');
        if (eq == std::string::npos || eq == 0) return false;

        // Key should be mostly key chars.
        for (size_t i = 0; i < eq; ++i) {
            unsigned char c = static_cast<unsigned char>(t[i]);
            // Allow OCR '0' in key and normal uppercase/digit/underscore.
            if (!IsKeyChar(c)) return false;
        }
        return true;
    }

    // Next line is likely a standalone value if it does not contain '=' and is not too long.
    bool LooksLikeStandaloneValue(const std::string& line) {
        std::string t = Trim(line);
        if (t.empty()) return false;
        if (t.find('=') != std::string::npos) return false;
        if (t.size() > 96) return false;
        return true;
    }

    // Normalize OCR-confusable keys: SL0T -> SLOT, etc.
    std::string NormalizeKey(std::string key) {
        // Replace '0' with 'O' inside key, because OCR often mixes them.
        for (char& c : key) {
            if (c == '0') c = 'O';
        }

        // Common expected keys in your overlay. Add more if you see OCR variants.
        if (key == "SLOT" || key == "SLOT") return "SLOT";
        if (key == "SL0T" || key == "SLOT") return "SLOT"; // extra safety
        if (key == "SLO T") return "SLOT";                  // rare spacing glitch

        // Keep other keys unchanged.
        return key;
    }

    // Fix O/o -> 0 for numeric-like values.
    void FixNumericO0(std::string& v) {
        for (char& c : v) {
            if (c == 'O' || c == 'o') c = '0';
        }
    }

    // Normalize value based on key type.
    std::string NormalizeValueForKey(const std::string& key, std::string value) {
        value = Trim(std::move(value));
        if (value.empty()) return value;

        // Keys that are typically numeric/float-ish in your output.
        // This helps cases like "SCD=O.0O".
        if (key == "SLOT" || key == "SPELL" || key == "REMAIN" || key == "SCD" ||
            key == "PHP" || key == "BUF" || key == "DEBUF") {
            FixNumericO0(value);
        }
        return value;
    }

    // Extract key-value pairs from a line without relying on whitespace.
    // It finds all occurrences of "...KEY=VALUE..." where KEY is [A-Z0-9_]+ (allowing OCR '0'),
    // then slices VALUE until the start of the next KEY=.
    std::vector<std::pair<std::string, std::string>> ExtractKVs(const std::string& line) {
        struct Hit {
            size_t key_begin;
            size_t eq_pos;
            std::string key;
        };

        std::vector<Hit> hits;
        hits.reserve(8);

        // Find '=' positions then backtrack to get key.
        for (size_t eq = 0; eq < line.size(); ++eq) {
            if (line[eq] != '=') continue;

            // Backtrack to find key end at eq-1.
            if (eq == 0) continue;
            size_t end = eq; // key ends at eq-1
            size_t beg = eq;
            while (beg > 0) {
                unsigned char c = static_cast<unsigned char>(line[beg - 1]);
                if (!IsKeyChar(c)) break;
                --beg;
            }

            // Key length must be >= 2 to reduce false positives.
            if (end <= beg || (end - beg) < 2) continue;

            std::string raw_key = line.substr(beg, end - beg);

            // Heuristic: prefer keys that look uppercase-ish (OCR may output digits too).
            // We don't strictly enforce uppercase because OCR may insert '0'.
            hits.push_back({ beg, eq, std::move(raw_key) });
        }

        if (hits.empty()) return {};

        // Sort by eq_pos to compute value slices.
        std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
            return a.eq_pos < b.eq_pos;
            });

        std::vector<std::pair<std::string, std::string>> out;
        out.reserve(hits.size());

        for (size_t i = 0; i < hits.size(); ++i) {
            const Hit& h = hits[i];

            size_t value_begin = h.eq_pos + 1;
            size_t value_end = (i + 1 < hits.size()) ? hits[i + 1].key_begin : line.size();

            if (value_begin >= line.size()) continue;
            if (value_begin > value_end) continue;

            std::string key = NormalizeKey(h.key);
            std::string val = line.substr(value_begin, value_end - value_begin);
            val = NormalizeValueForKey(key, std::move(val));

            if (!key.empty() && !val.empty()) {
                out.emplace_back(std::move(key), std::move(val));
            }
        }

        return out;
    }

} // namespace

// ------------------------------
// TextAnalyze implementation
// ------------------------------

TextAnalyze::TextAnalyze() {
}

// Parse a single (already repaired) line into kv_.
// This function supports multiple KEY=VALUE pairs in one line.
void TextAnalyze::parse_line(const std::string& line) {
    const auto kvs = ExtractKVs(line);
    for (const auto& kv : kvs) {
        kv_[kv.first] = kv.second;
    }
}

void TextAnalyze::set_texts(const std::vector<std::string>& lines)
{
    kv_.clear();

    // 1) Clean: trim & drop empty lines
    std::vector<std::string> cleaned;
    cleaned.reserve(lines.size());
    for (auto s : lines) {
        s = Trim(std::move(s));
        if (!s.empty()) cleaned.push_back(std::move(s));
    }

    // 2) Repair OCR line-splitting:
    //    Merge "KEY=" with the next standalone value line (no '=').
    std::vector<std::string> repaired;
    repaired.reserve(cleaned.size());

    for (size_t i = 0; i < cleaned.size(); ++i) {
        std::string cur = cleaned[i];

        if (EndsWithKeyEquals(cur) && (i + 1) < cleaned.size() && LooksLikeStandaloneValue(cleaned[i + 1])) {
            // Merge without adding space to preserve compact format.
            // If you prefer, you can add a single space: cur += " " + cleaned[i + 1];
            cur += cleaned[i + 1];
            ++i; // consume next line
        }

        repaired.push_back(std::move(cur));
    }

    // 3) Parse repaired lines
    for (const auto& line : repaired) {
        parse_line(line);
    }

    return;
}

std::string TextAnalyze::get_key(const std::string& key) const {
    // Keep interface unchanged: caller can query original key names.
    // We normalize common key OCR issues here too (optional but helpful).
    std::string nk = NormalizeKey(key);

    auto it = kv_.find(nk);
    if (it == kv_.end()) return {};
    return it->second;
}

bool TextAnalyze::has_key(const std::string& key) const {
    std::string nk = NormalizeKey(key);
    return kv_.find(nk) != kv_.end();
}

const std::unordered_map<std::string, std::string>& TextAnalyze::data() const {
    return kv_;
}
