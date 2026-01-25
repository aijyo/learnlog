#pragma once
#include <cstdint>
#include <string>

namespace utils
{
    enum class AutoMode :int
    {
        kDefault = 1,
        kAssistant = 1,
        kAutoSpell = 2,
        kHalfAuto = 4
    };

    AutoMode next_mode(AutoMode current);
    bool is_equal(AutoMode l, AutoMode r);
    std::string to_string(AutoMode mode);
    std::wstring to_wstring(AutoMode mode);
    AutoMode index_mode(int index);
    int mode_index(AutoMode mode);
}