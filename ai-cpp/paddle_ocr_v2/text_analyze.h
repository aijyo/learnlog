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

    ////| 数字 | 含义             |
    ////| ---- - | -------------- |
    ////| **0 * *| 没有目标           |
    ////| **1 * *| 友方目标（Friend）   |
    ////| **2 * *| 敌对 NPC         |
    ////| **3 * *| 敌对 玩家          |
    ////| **4 * *| 中立 / 不可攻击 / 其它 |
    //int target() const;
    //std::string target_str(int target = -1) const;
    // Check whether a key exists
    bool has_key(const std::string& key) const;

    // Debug helper: dump all key-value pairs
    const std::unordered_map<std::string, std::string>& data() const;

private:
    void parse_line(const std::string& line);

private:
    //int target_ = 0;
    std::unordered_map<std::string, std::string> kv_;
};
