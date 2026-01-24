#include "user_config.h"
#include <fstream>
#include <sstream>
// ====================== Implementation ======================

inline bool UserConfig::ParseSpellId_(const std::string& s, uint64_t* out) {
    // Parse spellid key string -> uint32
    // Accept only base-10 digits.
    if (!out || s.empty()) return false;
    uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + static_cast<uint64_t>(c - '0');
        if (v > 0xFFFFFFFFull) return false;
    }
    *out = static_cast<uint64_t>(v);
    return true;
}

inline bool UserConfig::ParseEntryObject_(const nlohmann::json& obj, Entry* out, std::string* err) {
    // Parse one entry object: {"button":"...","key":"...","name":"...","slotId":68}
    if (!out) return false;
    if (!obj.is_object()) {
        if (err) *err = "Entry is not an object.";
        return false;
    }

    Entry e;

    // button (optional)
    if (obj.contains("button")) {
        if (!obj["button"].is_string()) {
            if (err) *err = "\"button\" is not a string.";
            return false;
        }
        e.button = obj["button"].get<std::string>();
    }

    // key (required for your use-case)
    if (obj.contains("key")) {
        if (!obj["key"].is_string()) {
            if (err) *err = "\"key\" is not a string.";
            return false;
        }
        e.key = obj["key"].get<std::string>();
    }
    else {
        // Allow missing key but keep empty
        e.key.clear();
    }

    // name (optional)
    if (obj.contains("name")) {
        if (!obj["name"].is_string()) {
            if (err) *err = "\"name\" is not a string.";
            return false;
        }
        e.name = obj["name"].get<std::string>();
    }

    // slotId (optional)
    if (obj.contains("slotId")) {
        if (!obj["slotId"].is_number_integer()) {
            if (err) *err = "\"slotId\" is not an integer.";
            return false;
        }
        e.slotId = obj["slotId"].get<int>();
    }

    *out = std::move(e);
    return true;
}


bool UserConfig::set_path(const std::string& path, std::string* err_msg) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        if (err_msg) {
            *err_msg = "Failed to open config file: " + path;
        }
        return false;
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();

    if (!ifs.good() && !ifs.eof()) {
        if (err_msg) {
            *err_msg = "Failed while reading config file: " + path;
        }
        return false;
    }

    const std::string content = oss.str();
    if (content.empty()) {
        if (err_msg) {
            *err_msg = "Config file is empty: " + path;
        }
        return false;
    }

    // Reuse JSON-string based config loader
    return set_config(content, err_msg);
}

inline bool UserConfig::set_config(const std::string& json_str, std::string* err_msg) {
    // Reset old config
    map_.clear();

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(json_str);
    }
    catch (const std::exception& e) {
        if (err_msg) *err_msg = std::string("JSON parse failed: ") + e.what();
        return false;
    }

    if (!root.is_object()) {
        if (err_msg) *err_msg = "Root is not an object.";
        return false;
    }

    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string spellid_str = it.key();
        uint64_t spellid = 0;
        if (!ParseSpellId_(spellid_str, &spellid)) {
            // Skip invalid spellid key
            continue;
        }

        const nlohmann::json& val = it.value();
        std::vector<Entry> entries;

        if (val.is_object()) {
            Entry e;
            std::string perr;
            if (!ParseEntryObject_(val, &e, &perr)) {
                if (err_msg) *err_msg = "Entry parse failed for spellid=" + spellid_str + ": " + perr;
                return false;
            }
            entries.push_back(std::move(e));
        }
        else if (val.is_array()) {
            for (const auto& item : val) {
                Entry e;
                std::string perr;
                if (!ParseEntryObject_(item, &e, &perr)) {
                    if (err_msg) *err_msg = "Entry parse failed for spellid=" + spellid_str + ": " + perr;
                    return false;
                }
                entries.push_back(std::move(e));
            }
        }
        else {
            // Ignore unsupported type
            continue;
        }

        // Only store if we have at least one entry
        if (!entries.empty()) {
            map_[spellid] = std::move(entries);
        }
    }

    return true;
}

inline std::vector<std::string> UserConfig::GetKeys(uint64_t spellid) const {
    std::vector<std::string> out;
    auto it = map_.find(spellid);
    if (it == map_.end()) return out;

    out.reserve(it->second.size());
    for (const auto& e : it->second) {
        out.push_back(e.key);
    }
    return out;
}

std::string UserConfig::GetKeyBySpellId(uint64_t spellid) const {
    auto it = map_.find(spellid);
    if (it == map_.end()) {
        return {};
    }

    const auto& entries = it->second;
    if (entries.empty()) {
        return {};
    }

    // Return "key" field of the first json item
    return entries.front().key;
}

inline std::string UserConfig::GetPrimaryKey(uint64_t spellid) const {
    auto it = map_.find(spellid);
    if (it == map_.end() || it->second.empty()) return {};
    return it->second.front().key;
}

inline const std::vector<UserConfig::Entry>& UserConfig::GetEntries(uint64_t spellid) const {
    auto it = map_.find(spellid);
    if (it == map_.end()) return empty_;
    return it->second;
}

inline bool UserConfig::HasSpell(uint64_t spellid) const {
    return map_.find(spellid) != map_.end();
}
