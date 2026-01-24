#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

#include "third_party/nlohmann/json.hpp"

class UserConfig {
public:
    struct Entry {
        std::string button;
        std::string key;
        std::string name;
        int slotId = -1;
    };

public:
    UserConfig() = default;

    // Parse and set configuration from a JSON string produced by Lua.
    // Returns true on success; false on parse/format error.
    bool set_config(const std::string& json_str, std::string* err_msg = nullptr);

    // Load config from a file path
    bool set_path(const std::string& path, std::string* err_msg = nullptr);
    // Return all keybind strings for a spellid (e.g., {"2","D"}).
    // Empty if not found.
    std::vector<std::string> GetKeys(uint64_t spellid) const;

    // Get key string from json item by spellid
    std::string GetKeyBySpellId(uint64_t spellid) const;

    // Return the first keybind string for a spellid, empty if not found.
    std::string GetPrimaryKey(uint64_t spellid) const;

    // Return all parsed entries for a spellid, empty if not found.
    const std::vector<Entry>& GetEntries(uint64_t spellid) const;

    // Check whether spellid exists.
    bool HasSpell(uint64_t spellid) const;

private:
    static bool ParseSpellId_(const std::string& s, uint64_t* out);
    static bool ParseEntryObject_(const nlohmann::json& obj, Entry* out, std::string* err);

private:
    std::unordered_map<uint64_t, std::vector<Entry>> map_;
    std::vector<Entry> empty_;
};

