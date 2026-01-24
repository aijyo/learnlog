#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class TextAnalyze {
public:
    // Construct from multiple lines of text
    TextAnalyze();

    void set_texts(const std::vector<std::string>& lines);
    // Get value by key, return empty string if not found
    std::string get_key(const std::string& key) const;

    // Check whether a key exists
    bool has_key(const std::string& key) const;

    // Debug helper: dump all key-value pairs
    const std::unordered_map<std::string, std::string>& data() const;

private:
    void parse_line(const std::string& line);

private:
    std::unordered_map<std::string, std::string> kv_;
};
