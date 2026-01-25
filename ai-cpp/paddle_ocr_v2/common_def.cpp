#include "./common_def.h"


namespace utils
{

    AutoMode next_mode(AutoMode current)
    {
        AutoMode result = AutoMode::kDefault;
        switch (current)
        {
        case utils::AutoMode::kAssistant:
            result = AutoMode::kAutoSpell;
            break;
        case utils::AutoMode::kAutoSpell:
            result = AutoMode::kHalfAuto;
            break;
        case utils::AutoMode::kHalfAuto:
            result = AutoMode::kHalfAuto;
            break;
        default:
            break;
        }

        return result;
    }


    bool is_equal(AutoMode l, AutoMode r)
    {
        return ((int)l & (int)r) == 0;
    }


    std::string to_string(AutoMode mode)
    {
        std::string result;
        switch (mode)
        {
        case utils::AutoMode::kAssistant:
            result = "辅助";
            break;
        case utils::AutoMode::kAutoSpell:
            result = "自动";
            break;
        case utils::AutoMode::kHalfAuto:
            result = "半自动";
            break;
        default:
            break;
        }
        return result;
    }
    std::wstring to_wstring(AutoMode mode)
    {
        std::wstring result;
        switch (mode)
        {
        case utils::AutoMode::kAssistant:
            result = L"辅助";
            break;
        case utils::AutoMode::kAutoSpell:
            result = L"自动";
            break;
        case utils::AutoMode::kHalfAuto:
            result = L"半自动";
            break;
        default:
            break;
        }
        return result;
    }

    AutoMode index_mode(int index)
    {
        AutoMode result = AutoMode::kDefault;
        switch (index)
        {
        case 0:
            result = AutoMode::kAssistant;
            break;
        case 1:
            result = AutoMode::kAutoSpell;
            break;
        case 2:
            result = AutoMode::kHalfAuto;
            break;
        default:
            break;
        }
        return result;
    }


    int mode_index(AutoMode mode)
    {
        int index = 0;
        switch (mode)
        {
        case utils::AutoMode::kAssistant:
            index = 0;
            break;
        case utils::AutoMode::kAutoSpell:
            index = 1;
            break;
        case utils::AutoMode::kHalfAuto:
            index = 2;
            break;
        default:
            break;
        }

        return index;
    }
}