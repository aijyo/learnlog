#include "text_analyze.h"

#include <sstream>

// Parse all lines on construction
TextAnalyze::TextAnalyze() {
}

// Split a line into space-separated tokens and parse key=value
void TextAnalyze::parse_line(const std::string& line) {
    std::istringstream iss(line);
    std::string token;

    while (iss >> token) {
        auto pos = token.find('=');
        if (pos == std::string::npos) {
            continue;
        }

        std::string key = token.substr(0, pos);
        std::string value = token.substr(pos + 1);

        kv_[key] = value;
    }
}

void TextAnalyze::set_texts(const std::vector<std::string>& lines)
{
    for (const auto& line : lines) {
        parse_line(line);
    }
    return ;
}

// Get value by key
std::string TextAnalyze::get_key(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) {
        return {};
    }
    return it->second;
}

// Check if key exists
bool TextAnalyze::has_key(const std::string& key) const {
    return kv_.find(key) != kv_.end();
}

// Return raw data map
const std::unordered_map<std::string, std::string>&
TextAnalyze::data() const {
    return kv_;
}
