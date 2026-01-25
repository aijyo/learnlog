// main.cpp
// Build: /std:c++17, link SetupAPI.lib
// Notes: Code comments are in English as requested.

#define UNICODE
#define _UNICODE

#include <windows.h>
#include <commctrl.h>
#include <setupapi.h>
#include <devguid.h>
#include <initguid.h>
#include <string>
#include <vector>
#include <sstream>
#include <optional>
#include <algorithm>
#include <commdlg.h>
#include "third_party/nlohmann/json.hpp"
#include "ui_select_region.h"
#include "win_command.h"
#include "./common_def.h"
using nlohmann::json;

// Forward declarations for embedded runtime (implemented in win_main.cpp)
namespace wowapp {
    bool Start(const std::string& json_config_utf8);
    void Stop(int reason);
    bool IsRunning();
}


#pragma comment(lib, "Comctl32.lib")
#pragma comment(lib, "SetupAPI.lib")
#pragma comment(lib, "Comdlg32.lib")


// -----------------------------
// Small helpers
// -----------------------------
static std::wstring GetExeDir() {
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring p = path;
    size_t pos = p.find_last_of(L"\\/");
    if (pos == std::wstring::npos) return L".";
    return p.substr(0, pos);
}

static std::wstring JoinPath(const std::wstring& a, const std::wstring& b) {
    if (a.empty()) return b;
    if (a.back() == L'\\' || a.back() == L'/') return a + b;
    return a + L"\\" + b;
}

static bool ReadTextFileUtf8(const std::wstring& path, std::string& out) {
    out.clear();
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(h, &sz) || sz.QuadPart <= 0 || sz.QuadPart > (10LL * 1024 * 1024)) {
        CloseHandle(h);
        return false;
    }

    out.resize((size_t)sz.QuadPart);
    DWORD read = 0;
    BOOL ok = ReadFile(h, out.data(), (DWORD)out.size(), &read, nullptr);
    CloseHandle(h);
    return ok && read == out.size();
}

static std::wstring GetDirNameW(const std::wstring& path) {
    // English comment:
    // Return directory part of a path. If no separator found, return ".".
    size_t pos = path.find_last_of(L"\\/");
    if (pos == std::wstring::npos) return L".";
    return path.substr(0, pos);
}

static std::wstring GetDefaultJsonPath() {
    return JoinPath(GetExeDir(), L"default.json");
}

static bool OpenJsonFileDialog(HWND owner, const std::wstring& defaultPath, std::wstring& outPath) {
    wchar_t fileBuf[MAX_PATH]{};
    // Pre-fill with default path if it exists; otherwise empty.
    wcsncpy_s(fileBuf, defaultPath.c_str(), _TRUNCATE);

    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFile = fileBuf;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = L"JSON Files (*.json)\0*.json\0All Files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_EXPLORER;

    if (GetOpenFileNameW(&ofn)) {
        outPath = fileBuf;
        return true;
    }
    return false;
}


static std::wstring ToW(const std::string& s) {
    if (s.empty()) return L"";
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), len);
    return w;
}

static std::string ToU8(const std::wstring& w) {
    if (w.empty()) return "";
    int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), len, nullptr, nullptr);
    return s;
}

static bool ReadAllTextFileW(const std::wstring& path, std::wstring& outText) {
    outText.clear();
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(h, &sz) || sz.QuadPart <= 0 || sz.QuadPart > (1024LL * 1024LL * 10LL)) { // 10MB limit
        CloseHandle(h);
        return false;
    }

    std::vector<char> buf((size_t)sz.QuadPart);
    DWORD read = 0;
    if (!ReadFile(h, buf.data(), (DWORD)buf.size(), &read, nullptr) || read != buf.size()) {
        CloseHandle(h);
        return false;
    }
    CloseHandle(h);

    // Assume UTF-8 w/o BOM for simplicity
    std::string s(buf.begin(), buf.end());
    outText = ToW(s);
    return true;
}

static std::wstring GetWindowTextWString(HWND hEdit) {
    int len = GetWindowTextLengthW(hEdit);
    std::wstring w(len, L'\0');
    GetWindowTextW(hEdit, w.data(), len + 1);
    return w;
}

static void SetEditText(HWND hEdit, const std::wstring& w) {
    SetWindowTextW(hEdit, w.c_str());
}

static bool FileExists(const std::wstring& path) {
    DWORD attr = GetFileAttributesW(path.c_str());
    return (attr != INVALID_FILE_ATTRIBUTES) && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

static void AppendLog(HWND hLogEdit, const std::wstring& line) {
    // Append text to a multiline edit and scroll to bottom.
    int len = GetWindowTextLengthW(hLogEdit);
    SendMessageW(hLogEdit, EM_SETSEL, (WPARAM)len, (LPARAM)len);

    std::wstring msg = line;
    if (!msg.empty() && msg.back() != L'\n') msg += L"\r\n";
    SendMessageW(hLogEdit, EM_REPLACESEL, FALSE, (LPARAM)msg.c_str());
    SendMessageW(hLogEdit, EM_SCROLLCARET, 0, 0);
}


static LRESULT CALLBACK EditCtrlASubclassProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam,
    UINT_PTR uIdSubclass, DWORD_PTR dwRefData) {
    // English comment:
    // Add Ctrl+A "Select All" behavior for edit controls.
    if (uMsg == WM_KEYDOWN) {
        if ((wParam == 'A' || wParam == 'a') && (GetKeyState(VK_CONTROL) & 0x8000)) {
            SendMessageW(hWnd, EM_SETSEL, 0, -1);
            return 0;
        }
    }
    if (uMsg == WM_NCDESTROY) {
        RemoveWindowSubclass(hWnd, EditCtrlASubclassProc, uIdSubclass);
    }
    return DefSubclassProc(hWnd, uMsg, wParam, lParam);
}

static std::wstring FormatHotkeyDisplay(UINT vk, bool caseInsensitive) {
    // Digits
    if (vk >= '0' && vk <= '9') {
        return std::wstring(1, (wchar_t)vk);
    }
    // Letters
    if (vk >= 'A' && vk <= 'Z') {
        wchar_t up = (wchar_t)vk;
        wchar_t low = (wchar_t)(vk - 'A' + 'a');
        if (caseInsensitive) {
            std::wstring s;
            s.push_back(up);
            s.push_back(L',');
            s.push_back(low);
            return s; // "X,x"
        }
        return std::wstring(1, up); // "X"
    }
    // Function keys
    if (vk >= VK_F1 && vk <= VK_F12) {
        int n = (int)vk - (int)VK_F1 + 1;
        wchar_t buf[16]{};
        _snwprintf_s(buf, _TRUNCATE, L"F%d", n);
        return buf;
    }
    // Common keys (optional)
    if (vk == VK_SPACE) return L"Space";
    if (vk == VK_RETURN) return L"Enter";
    if (vk == VK_TAB) return L"Tab";
    if (vk == VK_ESCAPE) return L"Esc";

    return L"(Unknown)";
}

// -----------------------------
// COM port enumeration via SetupAPI
// Returns friendly name + COMx extracted.
// -----------------------------
struct ComPortItem {
    std::wstring friendly;
    std::wstring comName; // e.g., "COM5"
};

static std::optional<std::wstring> ExtractComNameFromFriendly(const std::wstring& friendly) {
    // Typical: "USB-SERIAL CH340 (COM5)"
    auto pos = friendly.rfind(L"(COM");
    if (pos == std::wstring::npos) return std::nullopt;
    auto end = friendly.find(L")", pos);
    if (end == std::wstring::npos) return std::nullopt;
    std::wstring inside = friendly.substr(pos + 1, end - (pos + 1)); // "COM5"
    if (inside.size() >= 3 && inside.rfind(L"COM", 0) == 0) return inside;
    return std::nullopt;
}

static std::vector<ComPortItem> EnumerateComPorts() {
    std::vector<ComPortItem> out;

    HDEVINFO devInfo = SetupDiGetClassDevsW(&GUID_DEVCLASS_PORTS, nullptr, nullptr, DIGCF_PRESENT);
    if (devInfo == INVALID_HANDLE_VALUE) return out;

    SP_DEVINFO_DATA devData{};
    devData.cbSize = sizeof(devData);

    for (DWORD i = 0; SetupDiEnumDeviceInfo(devInfo, i, &devData); ++i) {
        WCHAR buf[512] = { 0 };
        DWORD regType = 0;
        DWORD size = 0;

        if (SetupDiGetDeviceRegistryPropertyW(
            devInfo, &devData, SPDRP_FRIENDLYNAME, &regType,
            (PBYTE)buf, sizeof(buf), &size)) {

            std::wstring friendly = buf;
            auto com = ExtractComNameFromFriendly(friendly);
            if (com.has_value()) {
                out.push_back({ friendly, com.value() });
            }
        }
    }

    SetupDiDestroyDeviceInfoList(devInfo);

    // Sort by COM number if possible.
    auto comNum = [](const std::wstring& com) -> int {
        // "COM12" -> 12
        if (com.size() <= 3) return 0;
        return _wtoi(com.c_str() + 3);
        };
    std::sort(out.begin(), out.end(), [&](const ComPortItem& a, const ComPortItem& b) {
        return comNum(a.comName) < comNum(b.comName);
        });
    return out;
}

static std::string WToU8(const std::wstring& w) { return ToU8(w); }
static std::wstring U8ToW(const std::string& s) { return ToW(s); }
struct AppConfig {
    std::wstring configJsonPath;   // base config json file path

    // Mapping
    bool mappingUseFile = true;
    std::wstring mappingFilePath;  // when mappingUseFile==true
    std::wstring mappingJsonText;  // when mappingUseFile==false

    // Serial
    std::wstring comFriendly;      // optional store friendly name
    std::wstring comName;          // "COM5"
    int baud = 115200;

    // Hotkeys
    int oneKeyVk = 0;              // VK code
    bool ignore_trigger_case = false;

    int toggleKeyVk = 0;
    bool ignore_switch_case = false;

    // Mode display
    //std::wstring modeType = L"auto spell";
    utils::AutoMode modeType = utils::AutoMode::kAssistant;

    // Capture region
    int x = 0, y = 0, w = 0, h = 0;
};

static void to_json(json& j, const AppConfig& c) {
    j = json{
        {"configJsonPath", WToU8(c.configJsonPath)},
        {"mapping", {
            {"useFile", c.mappingUseFile},
            {"filePath", WToU8(c.mappingFilePath)},
            {"jsonText", WToU8(c.mappingJsonText)}
        }},
        {"serial", {
            {"comFriendly", WToU8(c.comFriendly)},
            {"comName", WToU8(c.comName)},
            {"baud", c.baud}
        }},
        {"hotkeys", {
            {"oneKeyVk", c.oneKeyVk},
            {"ignore_trigger_case", c.ignore_trigger_case},
            {"toggleKeyVk", c.toggleKeyVk},
            {"ignore_switch_case", c.ignore_switch_case}
        }},
        {"ui", {
            {"modeType",(int)c.modeType}
        }},
        {"capture", {
            {"x", c.x}, {"y", c.y}, {"w", c.w}, {"h", c.h}
        }}
    };
}

static void from_json(const json& j, AppConfig& c) {
    auto getStrW = [&](const json& obj, const char* key, const std::wstring& def = L"") -> std::wstring {
        if (!obj.contains(key) || !obj[key].is_string()) return def;
        return U8ToW(obj[key].get<std::string>());
        };

    c.configJsonPath = getStrW(j, "configJsonPath", c.configJsonPath);

    if (j.contains("mapping") && j["mapping"].is_object()) {
        auto& m = j["mapping"];
        c.mappingUseFile = m.value("useFile", c.mappingUseFile);
        c.mappingFilePath = getStrW(m, "filePath", c.mappingFilePath);
        c.mappingJsonText = getStrW(m, "jsonText", c.mappingJsonText);
    }

    if (j.contains("serial") && j["serial"].is_object()) {
        auto& s = j["serial"];
        c.comFriendly = getStrW(s, "comFriendly", c.comFriendly);
        c.comName = getStrW(s, "comName", c.comName);
        c.baud = s.value("baud", c.baud);
    }

    if (j.contains("hotkeys") && j["hotkeys"].is_object()) {
        auto& h = j["hotkeys"];
        c.oneKeyVk = h.value("oneKeyVk", c.oneKeyVk);
        c.ignore_trigger_case = h.value("ignore_trigger_case", c.ignore_trigger_case);
        c.toggleKeyVk = h.value("toggleKeyVk", c.toggleKeyVk);
        c.ignore_switch_case = h.value("ignore_switch_case", c.ignore_switch_case);
    }

    if (j.contains("ui") && j["ui"].is_object()) {
        auto& u = j["ui"];
        //c.modeType = getStrW(u, "modeType", c.modeType);
        c.modeType =(utils::AutoMode) u.value("modeType", (int)c.modeType);
    }

    if (j.contains("capture") && j["capture"].is_object()) {
        auto& cap = j["capture"];
        c.x = cap.value("x", c.x);
        c.y = cap.value("y", c.y);
        c.w = cap.value("w", c.w);
        c.h = cap.value("h", c.h);
    }
}

static bool WriteTextFileUtf8(const std::wstring& path, const std::string& text) {
    HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD written = 0;
    BOOL ok = WriteFile(h, text.data(), (DWORD)text.size(), &written, nullptr);
    CloseHandle(h);
    return ok && written == text.size();
}


static json DefaultWinMainConfigJson_() {
    // English comment:
    // Provide default config values compatible with wow_app (embedded runtime).
    // Default OCR model paths are relative to exe_dir\model:
    //   model\ocr.yaml
    //   model\PP-OCRv5_mobile_det_infer
    //   model\en_PP-OCRv4_mobile_rec_infer
    json j;

    const std::wstring exeDir = GetExeDir();
    const std::wstring modelDir = JoinPath(exeDir, L"model");
    const std::wstring detDir = JoinPath(modelDir, L"PP-OCRv5_mobile_det_infer");
    const std::wstring recDir = JoinPath(modelDir, L"en_PP-OCRv4_mobile_rec_infer");
    const std::wstring yamlPath = JoinPath(modelDir, L"ocr.yaml");

    j["ocr"] = {
        {"det_dir",  WToU8(detDir)},
        {"rec_dir",  WToU8(recDir)},
        {"text_detection_model_name", "PP-OCRv5_mobile_det"},
        {"text_recognition_model_name", "en_PP-OCRv4_mobile_rec"},
        {"paddlex_config", WToU8(yamlPath)},
        {"device", "cpu"},
        {"cpu_threads", 1},
        {"thread_num", 1},
        {"precision", "fp32"},
        {"enable_mkldnn", false},
        {"mkldnn_cache_capacity", 10},
        {"lang", "eng"}
    };
    j["serial"] = {
        {"comName", "COM3"},
        {"baud", 9600},
        {"addr", 0},
        {"wait_ack", true},
        {"debug", false}
    };
    j["hotkeys"] = {
        {"trigger_vk", (int)'1'},
        {"ignore_trigger_case", false},
        {"switch_vk", (int)VK_F12},
        {"ignore_switch_case", false},
        {"auto_mode", 1},
        {"delay_time", 300},
        {"break_time", 700},
        {"auto_time", 700}
    };
    j["capture"] = {
        {"x", 0}, {"y", 0}, {"w", 0}, {"h", 0},
        {"use_saved_region", false}
    };

    // Default: put user_keybinds.json next to config.json in exe dir.
    j["user_keybinds_path"] = WToU8(JoinPath(exeDir, L"user_keybinds.json"));
    return j;
}

static bool LoadJsonUtf8_(const std::wstring& path, json& out) {
    std::string text;
    if (!ReadTextFileUtf8(path, text)) return false;
    try {
        out = json::parse(text);
        return true;
    }
    catch (...) {
        return false;
    }
}

static bool SaveJsonUtf8_(const std::wstring& path, const json& j) {
    return WriteTextFileUtf8(path, j.dump(2));
}

// -----------------------------
// Hotkey picker dialog (modal)
// Captures a single key: 0-9, A-Z, F1-F12.
// -----------------------------
struct HotkeyResult {
    bool ok = false;
    UINT vk = 0;
    bool caseInsensitive = false; // only meaningful for A-Z
};

static bool IsAllowedVk(UINT vk) {
    if (vk >= '0' && vk <= '9') return true;
    if (vk >= 'A' && vk <= 'Z') return true;
    if (vk >= 'a' && vk <= 'z') return true;
    if (vk >= VK_F1 && vk <= VK_F12) return true;
    return false;
}

static std::wstring VkToLabel(UINT vk) {
    if (vk >= '0' && vk <= '9') return std::wstring(1, (wchar_t)vk);
    if (vk >= 'A' && vk <= 'Z') return std::wstring(1, (wchar_t)vk);
    if (vk >= 'a' && vk <= 'z') return std::wstring(1, (wchar_t)vk);
    if (vk >= VK_F1 && vk <= VK_F12) {
        int n = (int)vk - (int)VK_F1 + 1;
        std::wstringstream ss;
        ss << L"F" << n;
        return ss.str();
    }
    return L"(Unknown)";
}
// Hotkey picker dialog (modal) - EDIT input mode
// Only the first valid key is accepted; subsequent inputs are ignored.
class HotkeyPickerDialog {
public:
    HotkeyResult ShowModal(HWND parent, const std::wstring& title) {
        result_ = {};
        title_ = title;
        done_ = false;

        RegisterClassOnce_();

        HWND dlg = CreateWindowExW(
            WS_EX_DLGMODALFRAME,
            kClassName_,
            title_.c_str(),
            WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
            CW_USEDEFAULT, CW_USEDEFAULT, 520, 230,
            parent, nullptr, GetModuleHandleW(nullptr), this);

        ShowWindow(dlg, SW_SHOW);
        UpdateWindow(dlg);

        // Disable parent while modal
        EnableWindow(parent, FALSE);

        MSG msg;
        while (!done_ && GetMessageW(&msg, nullptr, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        EnableWindow(parent, TRUE);
        SetActiveWindow(parent);

        return result_;
    }

private:
    struct KeyItem {
        std::wstring label;
        UINT vk;
        bool isLetter; // Whether case-insensitive option applies
    };

    static constexpr const wchar_t* kClassName_ = L"HotkeyPickerDialogComboClass";

    static void RegisterClassOnce_() {
        static bool registered = false;
        if (registered) return;

        WNDCLASSW wc{};
        wc.lpfnWndProc = WndProcThunk_;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = kClassName_;
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        RegisterClassW(&wc);

        registered = true;
    }

    static LRESULT CALLBACK WndProcThunk_(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        HotkeyPickerDialog* self = nullptr;
        if (msg == WM_NCCREATE) {
            auto cs = (CREATESTRUCTW*)lParam;
            self = (HotkeyPickerDialog*)cs->lpCreateParams;
            SetWindowLongPtrW(hWnd, GWLP_USERDATA, (LONG_PTR)self);
        }
        else {
            self = (HotkeyPickerDialog*)GetWindowLongPtrW(hWnd, GWLP_USERDATA);
        }
        if (!self) return DefWindowProcW(hWnd, msg, wParam, lParam);
        return self->WndProc_(hWnd, msg, wParam, lParam);
    }

    void BuildKeyList_() {
        items_.clear();
        items_.reserve(100);

        // Digits 0-9
        for (wchar_t c = L'0'; c <= L'9'; ++c) {
            items_.push_back({ std::wstring(1, c), (UINT)c, false });
        }

        // Letters A-Z
        for (wchar_t c = L'A'; c <= L'Z'; ++c) {
            items_.push_back({ std::wstring(1, c), (UINT)c, true });
        }

        // Function keys F1-F12
        for (int i = 1; i <= 12; ++i) {
            std::wstringstream ss;
            ss << L"F" << i;
            items_.push_back({ ss.str(), (UINT)(VK_F1 + (i - 1)), false });
        }

        // Common keys
        items_.push_back({ L"Space", VK_SPACE, false });
        items_.push_back({ L"Enter", VK_RETURN, false });
        items_.push_back({ L"Tab", VK_TAB, false });
        items_.push_back({ L"Esc", VK_ESCAPE, false });
        items_.push_back({ L"Backspace", VK_BACK, false });

        items_.push_back({ L"Insert", VK_INSERT, false });
        items_.push_back({ L"Delete", VK_DELETE, false });
        items_.push_back({ L"Home", VK_HOME, false });
        items_.push_back({ L"End", VK_END, false });
        items_.push_back({ L"PageUp", VK_PRIOR, false });
        items_.push_back({ L"PageDown", VK_NEXT, false });

        items_.push_back({ L"Up", VK_UP, false });
        items_.push_back({ L"Down", VK_DOWN, false });
        items_.push_back({ L"Left", VK_LEFT, false });
        items_.push_back({ L"Right", VK_RIGHT, false });

        // If you also want to allow pure modifier keys, uncomment:
        // items_.push_back({ L"Shift", VK_SHIFT, false });
        // items_.push_back({ L"Ctrl", VK_CONTROL, false });
        // items_.push_back({ L"Alt", VK_MENU, false });
        // items_.push_back({ L"CapsLock", VK_CAPITAL, false });
    }

    void PopulateCombo_() {
        SendMessageW(combo_, CB_RESETCONTENT, 0, 0);

        for (size_t i = 0; i < items_.size(); ++i) {
            int idx = (int)SendMessageW(combo_, CB_ADDSTRING, 0, (LPARAM)items_[i].label.c_str());
            // Store item index in combo item data
            SendMessageW(combo_, CB_SETITEMDATA, idx, (LPARAM)i);
        }

        // Default select the first item
        SendMessageW(combo_, CB_SETCURSEL, 0, 0);
        UpdateCaseInsensitiveEnabled_();
    }

    void UpdateCaseInsensitiveEnabled_() {
        int cur = (int)SendMessageW(combo_, CB_GETCURSEL, 0, 0);
        if (cur < 0) {
            EnableWindow(checkCase_, FALSE);
            return;
        }
        size_t itemIndex = (size_t)SendMessageW(combo_, CB_GETITEMDATA, cur, 0);
        if (itemIndex >= items_.size()) {
            EnableWindow(checkCase_, FALSE);
            return;
        }

        bool enable = items_[itemIndex].isLetter;
        EnableWindow(checkCase_, enable ? TRUE : FALSE);

        if (!enable) {
            // Force unchecked if not a letter
            SendMessageW(checkCase_, BM_SETCHECK, BST_UNCHECKED, 0);
        }
    }

    void OnOK_() {
        int cur = (int)SendMessageW(combo_, CB_GETCURSEL, 0, 0);
        if (cur < 0) {
            MessageBoxW(hwnd_, L"请选择一个按键。", L"Hotkey", MB_OK | MB_ICONWARNING);
            return;
        }

        size_t itemIndex = (size_t)SendMessageW(combo_, CB_GETITEMDATA, cur, 0);
        if (itemIndex >= items_.size()) {
            MessageBoxW(hwnd_, L"选择无效，请重新选择。", L"Hotkey", MB_OK | MB_ICONWARNING);
            return;
        }

        const auto& it = items_[itemIndex];
        bool caseInsensitive = false;
        if (it.isLetter) {
            caseInsensitive = (SendMessageW(checkCase_, BM_GETCHECK, 0, 0) == BST_CHECKED);
        }

        result_.ok = true;
        result_.vk = it.vk;
        result_.caseInsensitive = caseInsensitive;

        DestroyWindow(hwnd_);
    }

    void OnCancel_() {
        result_ = {}; // ok=false
        DestroyWindow(hwnd_);
    }

    LRESULT WndProc_(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        switch (msg) {
        case WM_CREATE: {
            hwnd_ = hWnd;

            BuildKeyList_();

            CreateWindowW(L"STATIC",
                L"选择一个单独按键（常用键已预置）：",
                WS_CHILD | WS_VISIBLE,
                16, 16, 460, 20,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            combo_ = CreateWindowW(L"COMBOBOX",
                L"",
                WS_CHILD | WS_VISIBLE | WS_BORDER | CBS_DROPDOWNLIST,
                16, 44, 300, 400,
                hWnd, (HMENU)101, GetModuleHandleW(nullptr), nullptr);

            checkCase_ = CreateWindowW(L"BUTTON",
                L"忽视大小写（仅对 A-Z 生效）",
                WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                16, 84, 260, 24,
                hWnd, (HMENU)102, GetModuleHandleW(nullptr), nullptr);
            SendMessageW(checkCase_, BM_SETCHECK, BST_UNCHECKED, 0);

            btnOK_ = CreateWindowW(L"BUTTON",
                L"确定",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                310, 150, 90, 30,
                hWnd, (HMENU)1, GetModuleHandleW(nullptr), nullptr);

            btnCancel_ = CreateWindowW(L"BUTTON",
                L"取消",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                410, 150, 90, 30,
                hWnd, (HMENU)2, GetModuleHandleW(nullptr), nullptr);

            PopulateCombo_();
            SetFocus(combo_);
            return 0;
        }
        case WM_COMMAND: {
            const int id = LOWORD(wParam);
            const int code = HIWORD(wParam);

            if (id == 101 && code == CBN_SELCHANGE) {
                UpdateCaseInsensitiveEnabled_();
                return 0;
            }

            if (id == 1) { // OK
                OnOK_();
                return 0;
            }
            if (id == 2) { // Cancel
                OnCancel_();
                return 0;
            }
            break;
        }
        case WM_CLOSE:
            OnCancel_();
            return 0;

        case WM_DESTROY:
            // Do NOT call PostQuitMessage here.
            // Just mark modal loop done.
            done_ = true;
            hwnd_ = nullptr;
            return 0;
        }
        return DefWindowProcW(hWnd, msg, wParam, lParam);
    }

private:
    HotkeyResult result_{};
    std::wstring title_;

    bool done_ = false;

    HWND hwnd_ = nullptr;
    HWND combo_ = nullptr;
    HWND checkCase_ = nullptr;
    HWND btnOK_ = nullptr;
    HWND btnCancel_ = nullptr;

    std::vector<KeyItem> items_;
};

// -----------------------------
// ImageView: custom control to display BgraFrame
// -----------------------------
class ImageView {
public:
    bool Create(HWND parent, int x, int y, int w, int h, int id) {
        WNDCLASSW wc{};
        wc.lpfnWndProc = WndProcThunk_;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"ImageViewControlClass";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
        RegisterClassW(&wc);

        hwnd_ = CreateWindowW(
            wc.lpszClassName,
            L"",
            WS_CHILD | WS_VISIBLE | WS_BORDER,
            x, y, w, h,
            parent,
            (HMENU)(INT_PTR)id,
            wc.hInstance,
            this);

        return hwnd_ != nullptr;
    }

    HWND hwnd() const { return hwnd_; }

    void SetFrame(const BgraFrame& f) {
        frame_ = f;
        InvalidateRect(hwnd_, nullptr, TRUE);
    }

private:
    static LRESULT CALLBACK WndProcThunk_(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        ImageView* self = nullptr;
        if (msg == WM_NCCREATE) {
            auto cs = (CREATESTRUCTW*)lParam;
            self = (ImageView*)cs->lpCreateParams;
            SetWindowLongPtrW(hWnd, GWLP_USERDATA, (LONG_PTR)self);
        }
        else {
            self = (ImageView*)GetWindowLongPtrW(hWnd, GWLP_USERDATA);
        }
        if (!self) return DefWindowProcW(hWnd, msg, wParam, lParam);
        return self->WndProc_(hWnd, msg, wParam, lParam);
    }

    LRESULT WndProc_(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        switch (msg) {
        case WM_PAINT: {
            PAINTSTRUCT ps{};
            HDC hdc = BeginPaint(hWnd, &ps);

            RECT rc{};
            GetClientRect(hWnd, &rc);

            int viewW = rc.right - rc.left;
            int viewH = rc.bottom - rc.top;

            // Fill background
            FillRect(hdc, &rc, (HBRUSH)(COLOR_BTNFACE + 1));

            if (frame_.width > 0 && frame_.height > 0 && !frame_.data.empty()) {
                int imgW = frame_.width;
                int imgH = frame_.height;

                // -------- scale calculation (never upscale) --------
                double scale = 1.0;
                if (imgW > viewW || imgH > viewH) {
                    double sx = (double)viewW / imgW;
                    double sy = (double)viewH / imgH;
                    scale = (sx < sy) ? sx : sy;
                }

                int drawW = (int)(imgW * scale);
                int drawH = (int)(imgH * scale);

                int drawX = (viewW - drawW) / 2;
                int drawY = (viewH - drawH) / 2;

                BITMAPINFO bmi{};
                bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                bmi.bmiHeader.biWidth = imgW;
                bmi.bmiHeader.biHeight = -imgH; // top-down
                bmi.bmiHeader.biPlanes = 1;
                bmi.bmiHeader.biBitCount = 32;
                bmi.bmiHeader.biCompression = BI_RGB;

                // Use StretchDIBits only when scaling down
                SetStretchBltMode(hdc, HALFTONE);
                SetBrushOrgEx(hdc, 0, 0, nullptr);

                StretchDIBits(
                    hdc,
                    drawX, drawY, drawW, drawH,
                    0, 0, imgW, imgH,
                    frame_.data.data(),
                    &bmi,
                    DIB_RGB_COLORS,
                    SRCCOPY
                );
            }
            else {
                DrawTextW(hdc, L"(No frame)", -1, &rc,
                    DT_CENTER | DT_VCENTER | DT_SINGLELINE);
            }

            EndPaint(hWnd, &ps);
            return 0;
        }

        }
        return DefWindowProcW(hWnd, msg, wParam, lParam);
    }

private:
    HWND hwnd_ = nullptr;
    BgraFrame frame_{};
};

// -----------------------------
// Config window: base config json path
// -----------------------------
class ConfigWindow {
public:
    bool Create(HWND parent) {
        parent_ = parent;

        WNDCLASSW wc{};
        wc.lpfnWndProc = WndProcThunk_;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"ConfigWindowClass";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        RegisterClassW(&wc);

        hwnd_ = CreateWindowExW(
            WS_EX_TOOLWINDOW,
            wc.lpszClassName,
            L"Config",
            WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
            CW_USEDEFAULT, CW_USEDEFAULT, 520, 160,
            parent, nullptr, wc.hInstance, this);

        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);
        return hwnd_ != nullptr;
    }

    HWND hwnd() const { return hwnd_; }
    std::wstring base_config_json_path() const { return baseConfigPath_; }

private:
    static LRESULT CALLBACK WndProcThunk_(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        ConfigWindow* self = nullptr;
        if (msg == WM_NCCREATE) {
            auto cs = (CREATESTRUCTW*)lParam;
            self = (ConfigWindow*)cs->lpCreateParams;
            SetWindowLongPtrW(hWnd, GWLP_USERDATA, (LONG_PTR)self);
        }
        else {
            self = (ConfigWindow*)GetWindowLongPtrW(hWnd, GWLP_USERDATA);
        }
        if (!self) return DefWindowProcW(hWnd, msg, wParam, lParam);
        return self->WndProc_(hWnd, msg, wParam, lParam);
    }

    LRESULT WndProc_(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        switch (msg) {
        case WM_CREATE: {
            CreateWindowW(L"STATIC", L"Base config JSON path:",
                WS_CHILD | WS_VISIBLE,
                16, 20, 160, 20,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            editBase_ = CreateWindowW(L"EDIT", L"",
                WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                180, 18, 300, 24,
                hWnd, (HMENU)1001, GetModuleHandleW(nullptr), nullptr);

            btnSave_ = CreateWindowW(L"BUTTON", L"Save",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                390, 60, 90, 28,
                hWnd, (HMENU)1002, GetModuleHandleW(nullptr), nullptr);

            if (!baseConfigPath_.empty()) SetEditText(editBase_, baseConfigPath_);
            return 0;
        }
        case WM_COMMAND: {
            if (LOWORD(wParam) == 1002) {
                baseConfigPath_ = GetWindowTextWString(editBase_);
                MessageBoxW(hWnd, L"Saved.", L"Config", MB_OK | MB_ICONINFORMATION);
                return 0;
            }
            break;
        }
        case WM_CLOSE:
            if (wowapp::IsRunning()) {
                wowapp::Stop(0);
            }
            DestroyWindow(hWnd);
            return 0;
        case WM_DESTROY:
            hwnd_ = nullptr;
            return 0;
        }
        return DefWindowProcW(hWnd, msg, wParam, lParam);
    }

private:
    HWND parent_ = nullptr;
    HWND hwnd_ = nullptr;

    HWND editBase_ = nullptr;
    HWND btnSave_ = nullptr;

    std::wstring baseConfigPath_;
};

// -----------------------------
// Main window
// -----------------------------
class MainWindow {
public:
    bool Create() {
        INITCOMMONCONTROLSEX icc{ sizeof(icc), ICC_STANDARD_CLASSES | ICC_WIN95_CLASSES };
        InitCommonControlsEx(&icc);

        WNDCLASSW wc{};
        wc.lpfnWndProc = WndProcThunk_;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"MainWindowClass";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        RegisterClassW(&wc);

        hwnd_ = CreateWindowW(
            wc.lpszClassName,
            L"WoW Helper - Control Panel",
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT, 1100, 720,
            nullptr, nullptr, wc.hInstance, this);

        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);
        return hwnd_ != nullptr;
    }

    HWND hwnd() const { return hwnd_; }

private:
    enum : int {
        ID_BTN_OPEN_CONFIG = 2001,
        ID_EDIT_MAPPING_PATH = 2002,
        ID_BTN_APPLY = 2003,
        ID_BTN_START = 2101,
        ID_BTN_STOP = 2102,
        ID_LOG = 2004,

        ID_COMBO_COM = 2005,
        ID_EDIT_BAUD = 2006,

        ID_BTN_PICK_ONEKEY = 2007,
        ID_CHECK_CASE_INSENSITIVE = 2008,

        ID_BTN_PICK_TOGGLEKEY = 2009,
        ID_STATIC_MODE = 2010,

        ID_STATIC_REGION = 2011,
        ID_BTN_TEST_CAPTURE = 2015,
        ID_BTN_SELECT_REGION = 2012,

        ID_IMAGEVIEW = 2013,
        ID_BTN_REFRESH_COM = 2014,

        ID_MAPPING_JSON_FROM_FILE = 3001,
        ID_PAST_JSON_MAPPING = 3002,
        ID_SELECT_JSON_MAPPING = 3003,
    };

    static LRESULT CALLBACK WndProcThunk_(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        MainWindow* self = nullptr;
        if (msg == WM_NCCREATE) {
            auto cs = (CREATESTRUCTW*)lParam;
            self = (MainWindow*)cs->lpCreateParams;
            SetWindowLongPtrW(hWnd, GWLP_USERDATA, (LONG_PTR)self);
        }
        else {
            self = (MainWindow*)GetWindowLongPtrW(hWnd, GWLP_USERDATA);
        }
        if (!self) return DefWindowProcW(hWnd, msg, wParam, lParam);
        return self->WndProc_(hWnd, msg, wParam, lParam);
    }
    void Layout_() {
        // Fixed layout with clearer grouping.
        const int margin = 16;
        const int gapY = 10;
        const int gapX = 10;

        RECT rc{};
        GetClientRect(hwnd_, &rc);

        int winW = rc.right - rc.left;
        int winH = rc.bottom - rc.top;

        int x = margin;
        int y = margin;

        // ---------- Row 1: Config + Apply ----------
        // Config button on the left
        MoveWindow(btnPickConfig_, x, y, 140, 28, TRUE);

        // Buttons aligned right: Stop | Start | Apply
        int btnW = 100;
        int btnH = 28;
        int rightX = winW - margin;

        // Apply (right-most)
        MoveWindow(btnApply_, rightX - btnW, y, btnW, btnH, TRUE);
        rightX -= (btnW + gapX);

        // Start
        MoveWindow(btnStart_, rightX - btnW, y, btnW, btnH, TRUE);
        rightX -= (btnW + gapX);

        // Stop
        MoveWindow(btnStop_, rightX - btnW, y, btnW, btnH, TRUE);

        y += 28 + gapY;

        // ---------- Group: Key Mapping (radio + file + paste) ----------
        // Row: radio buttons
        MoveWindow(radioMapFile_, x, y, 110, 22, TRUE);
        MoveWindow(radioMapPaste_, x + 120, y, 130, 22, TRUE);

        y += 22 + gapY;

        // Row: file picker + path (take full width, leaving some right padding)
        int pickW = 120;
        int pathW = (winW - margin * 2) - pickW - gapX;
        if (pathW < 200) pathW = 200;

        MoveWindow(btnPickMappingFile_, x, y, pickW, 26, TRUE);
        MoveWindow(editMappingFilePath_, x + pickW + gapX, y, pathW, 26, TRUE);

        y += 26 + gapY;

        // Paste box: multi-line, width matches mapping row
        // Height can be adaptive; keep reasonable default
        int pasteH = 110;
        MoveWindow(editMappingJsonPaste_, x, y, (winW - margin * 2), pasteH, TRUE);

        y += pasteH + (gapY + 6);

        // ============================================================
        // NOTE:
        // Remove old mapping controls to avoid duplicate UI:
        // stMappingPath_, editMappingPath_ should not be used anymore.
        // If you still have them, do NOT MoveWindow them here.
        // ============================================================

        // ---------- Group: COM + Baud (one row) ----------
        // Layout: "COM:" [combo] [refresh]   "Baud:" [edit]
        int comLabelW = 50;
        int comComboW = 320;
        int refreshW = 90;
        int baudLabelW = 55;
        int baudEditW = 120;

        MoveWindow(stCom_, x, y + 4, comLabelW, 20, TRUE);
        MoveWindow(comboCom_, x + comLabelW, y, comComboW, 200, TRUE);
        MoveWindow(btnRefreshCom_, x + comLabelW + comComboW + gapX, y, refreshW, 26, TRUE);

        int baudX = x + comLabelW + comComboW + gapX + refreshW + 2 * gapX;
        MoveWindow(stBaud_, baudX, y + 4, baudLabelW, 20, TRUE);
        MoveWindow(editBaud_, baudX + baudLabelW, y, baudEditW, 26, TRUE);

        y += 26 + gapY;

        // ---------- Group: OneKey + Case-insensitive ----------
        // "OneKey:" [Select] [value] [checkbox]
        int oneLabelW = 180;
        int selW = 120;
        int valW = 80;
        int chkW = 180;

        MoveWindow(stOneKey_, x, y + 4, oneLabelW, 20, TRUE);
        MoveWindow(btnPickOneKey_, x + oneLabelW, y, selW, 26, TRUE);
        MoveWindow(stOneKeyValue_, x + oneLabelW + selW + gapX, y + 4, valW, 20, TRUE);

        y += 26 + gapY;

        // ---------- Group: Toggle + Mode ----------
        // "Toggle:" [Select] [value]   "Mode:" [text]
        int toggleLabelW = 180;
        int modeLabelW = 55;
        int modeW = 220;

        MoveWindow(stToggleKey_, x, y + 4, toggleLabelW, 20, TRUE);
        MoveWindow(btnPickToggleKey_, x + toggleLabelW, y, selW, 26, TRUE);
        MoveWindow(stToggleValue_, x + toggleLabelW + selW + gapX, y + 4, valW, 20, TRUE);

        int modeX = x + toggleLabelW + selW + gapX + valW + 3 * gapX;
        MoveWindow(stModeLabel_, modeX, y + 4, modeLabelW, 20, TRUE);
        MoveWindow(stMode_, x + 610, y - 2, 200, 200, TRUE);

        y += 26 + gapY;


        // ---------- Group: Region + Select ----------
        // "Capture region:" [text] [Test capture] [Select region]
        int regionLabelW = 110;
        int regionTestBtnW = 120;
        int regionSelectBtnW = 140;
        int regionTextW = (winW - margin * 2) - regionLabelW - gapX - regionTestBtnW - gapX - regionSelectBtnW - gapX;
        if (regionTextW < 200) regionTextW = 200;

        MoveWindow(stRegionLabel_, x, y + 4, regionLabelW, 20, TRUE);
        MoveWindow(stRegion_, x + regionLabelW + gapX, y + 4, regionTextW, 20, TRUE);

        int btnX = x + regionLabelW + gapX + regionTextW + gapX;
        MoveWindow(btnTestCapture_, btnX, y, regionTestBtnW, 26, TRUE);
        MoveWindow(btnSelectRegion_, btnX + regionTestBtnW + gapX, y, regionSelectBtnW, 26, TRUE);

        y += 26 + gapY;

        // ---------- Bottom: Image + Log split ----------
        int bottomTop = y + 6;
        int bottomH = winH - bottomTop - margin;
        if (bottomH < 200) bottomH = 200;

        int splitGap = 16;
        int leftW = 520;
        if (winW < 1100) {
            // adapt to smaller window: make image smaller, keep log visible
            leftW = max(360, (winW - margin * 2 - splitGap) / 2);
        }
        int rightW = (winW - margin * 2) - leftW - splitGap;
        if (rightW < 260) rightW = 260;

        int imageX = x;
        int imageY = bottomTop;
        int imageW = leftW;
        int imageH = bottomH;

        MoveWindow(imageView_.hwnd(), imageX, imageY, imageW, imageH, TRUE);

        int logX = x + leftW + splitGap;
        int logY = bottomTop;
        int logW = rightW;
        int logH = bottomH;

        MoveWindow(stLogLabel_, logX, logY - 22, 120, 20, TRUE);
        MoveWindow(editLog_, logX, logY, logW, logH, TRUE);
    }

    void RefreshComList_() {
        SendMessageW(comboCom_, CB_RESETCONTENT, 0, 0);
        comItems_ = EnumerateComPorts();
        for (size_t i = 0; i < comItems_.size(); ++i) {
            SendMessageW(comboCom_, CB_ADDSTRING, 0, (LPARAM)comItems_[i].friendly.c_str());
        }
        if (!comItems_.empty()) {
            SendMessageW(comboCom_, CB_SETCURSEL, 0, 0);
        }
        AppendLog(editLog_, L"[COM] Refreshed COM ports.");
    }

    bool ValidateAndApply_() {
        // 1) Config.json path: if missing, create a default one.
        if (baseConfigPath_.empty()) {
            baseConfigPath_ = JoinPath(GetExeDir(), L"config.json");
        }
        if (!FileExists(baseConfigPath_)) {
            json init = DefaultWinMainConfigJson_();
            if (!SaveJsonUtf8_(baseConfigPath_, init)) {
                MessageBoxW(hwnd_, L"Failed to create default config.json.", L"Apply", MB_OK | MB_ICONERROR);
                return false;
            }
            AppendLog(editLog_, L"[CFG] Created default config: " + baseConfigPath_);
        }

        // 2) Mapping JSON: file or pasted
        std::wstring mappingJson;
        if (mappingUseFile_) {
            std::wstring mappingPath = GetWindowTextWString(editMappingFilePath_);
            if (mappingPath.empty() || !FileExists(mappingPath)) {
                MessageBoxW(hwnd_, L"Key mapping JSON file path is empty or not found.", L"Apply", MB_OK | MB_ICONERROR);
                return false;
            }
            if (!ReadAllTextFileW(mappingPath, mappingJson)) {
                MessageBoxW(hwnd_, L"Failed to read mapping JSON file.", L"Apply", MB_OK | MB_ICONERROR);
                return false;
            }
        }
        else {
            mappingJson = GetWindowTextWString(editMappingJsonPaste_);
            // trim simple
            auto isSpace = [](wchar_t c) { return c == L' ' || c == L'\r' || c == L'\n' || c == L'\t'; };
            while (!mappingJson.empty() && isSpace(mappingJson.front())) mappingJson.erase(mappingJson.begin());
            while (!mappingJson.empty() && isSpace(mappingJson.back())) mappingJson.pop_back();

            if (mappingJson.empty()) {
                MessageBoxW(hwnd_, L"Please paste mapping JSON content.", L"Apply", MB_OK | MB_ICONERROR);
                return false;
            }
        }

        std::wstring baudStr = GetWindowTextWString(editBaud_);
        int baud = _wtoi(baudStr.c_str());
        if (baud <= 0) {
            MessageBoxW(hwnd_, L"Invalid baud rate.", L"Apply", MB_OK | MB_ICONERROR);
            return false;
        }

        int comSel = (int)SendMessageW(comboCom_, CB_GETCURSEL, 0, 0);
        if (comSel < 0 || comSel >= (int)comItems_.size()) {
            MessageBoxW(hwnd_, L"Please select a COM port.", L"Apply", MB_OK | MB_ICONERROR);
            return false;
        }

        if (!oneKeyVk_.has_value()) {
            MessageBoxW(hwnd_, L"Please select a 'one-key output' hotkey.", L"Apply", MB_OK | MB_ICONERROR);
            return false;
        }

        if (!toggleKeyVk_.has_value()) {
            MessageBoxW(hwnd_, L"Please select a 'toggle function' hotkey.", L"Apply", MB_OK | MB_ICONERROR);
            return false;
        }

        // Here you would:
        // 1) Load base config json path from ConfigWindow (if any)
        // 2) Load mapping json
        // 3) Open serial port using selected COM and baud
        // 4) Apply your runtime settings
        //
        // This demo only logs them.

        std::wstringstream ss;
        std::wstring use_file = mappingUseFile_ ? L"[APPLY] Mapping source: file" : L"[APPLY] Mapping source: pasted text";
        ss << L"[APPLY] mapping=" << use_file
            << L", com=" << comItems_[comSel].comName
            << L", baud=" << baud
            << L", oneKey=" << VkToLabel(oneKeyVk_.value())
            << L", toggleKey=" << VkToLabel(toggleKeyVk_.value())
            << L", ignore_trigger_case=" << (ignore_trigger_case_ ? L"true" : L"false")
            << L", ignore_switch_case=" << (ignore_switch_case_ ? L"true" : L"false");
        AppendLog(editLog_, ss.str());

        // Update current mode text.
        // Default is "auto spell" as you requested.
        int sel = (int)SendMessageW(stMode_, CB_GETCURSEL, 0, 0);
        //std::wstring mode = L"自动";
        //if (sel >= 0) {
        //    wchar_t buf[64]{};
        //    SendMessageW(stMode_, CB_GETLBTEXT, sel, (LPARAM)buf);
        //    mode = buf;
        //}
        utils::AutoMode mode_type = utils::index_mode(sel);

        AppendLog(editLog_, L"[MODE] Current mode: " + utils::to_wstring(mode_type));

        // 5) Persist runtime config for win_main.cpp
        // English comment:
        // - Write mapping JSON to user_keybinds.json (same folder as config.json)
        // - Merge/update config.json with serial/hotkeys/capture/user_keybinds_path
        {
            std::wstring cfgPath = baseConfigPath_.empty() ? JoinPath(GetExeDir(), L"config.json") : baseConfigPath_;
            std::wstring cfgDir = GetDirNameW(cfgPath);
            std::wstring userKeybindsPath = JoinPath(cfgDir, L"user_keybinds.json");

            // Write mapping json text (UTF-8)
            if (!WriteTextFileUtf8(userKeybindsPath, ToU8(mappingJson))) {
                AppendLog(editLog_, L"[CFG] Failed to write user_keybinds.json: " + userKeybindsPath);
                MessageBoxW(hwnd_, L"Failed to write user_keybinds.json.", L"Apply", MB_OK | MB_ICONERROR);
                return false;
            }

            json cfg;
            if (!LoadJsonUtf8_(cfgPath, cfg)) {
                cfg = DefaultWinMainConfigJson_();
            }

            // Serial
            cfg["serial"]["comName"] = WToU8(comItems_[comSel].comName);
            cfg["serial"]["baud"] = baud;

            // Hotkeys: map UI oneKey/toggleKey into win_main trigger/switch.
            cfg["hotkeys"]["trigger_vk"] = (int)oneKeyVk_.value();
            cfg["hotkeys"]["ignore_trigger_case"] = ignore_trigger_case_;
            cfg["hotkeys"]["switch_vk"] = (int)toggleKeyVk_.value();
            cfg["hotkeys"]["ignore_switch_case"] = ignore_switch_case_;
            cfg["hotkeys"]["auto_mode"] = mode_type;
            cfg["hotkeys"]["delay_time"] = 300;
            cfg["hotkeys"]["break_time"] = 700;
            cfg["hotkeys"]["auto_time"] = 700;

            // Capture: store the selected virtual-screen region
            cfg["capture"]["x"] = region_.left;
            cfg["capture"]["y"] = region_.top;
            cfg["capture"]["w"] = (region_.right - region_.left);
            cfg["capture"]["h"] = (region_.bottom - region_.top);
            cfg["capture"]["use_saved_region"] = true;

            // Mapping path consumed by TextHandler in win_main
            cfg["user_keybinds_path"] = WToU8(userKeybindsPath);

            if (!SaveJsonUtf8_(cfgPath, cfg)) {
                AppendLog(editLog_, L"[CFG] Failed to write config.json: " + cfgPath);
                MessageBoxW(hwnd_, L"Failed to write config.json.", L"Apply", MB_OK | MB_ICONERROR);
                return false;
            }

            AppendLog(editLog_, L"[CFG] Updated: " + cfgPath);
            AppendLog(editLog_, L"[CFG] Updated: " + userKeybindsPath);
        }

        MessageBoxW(hwnd_, L"Apply succeeded.", L"Apply", MB_OK | MB_ICONINFORMATION);
        return true;
    }


    void UpdateRunUiState_() {
        bool running = wowapp::IsRunning();

        // Buttons
        EnableWindow(btnStart_, running ? FALSE : TRUE);
        EnableWindow(btnStop_, running ? TRUE : FALSE);
        EnableWindow(btnApply_, running ? FALSE : TRUE);

        // Disable config edits while running to avoid inconsistent state
        EnableWindow(btnPickConfig_, running ? FALSE : TRUE);
        EnableWindow(radioMapFile_, running ? FALSE : TRUE);
        EnableWindow(radioMapPaste_, running ? FALSE : TRUE);
        EnableWindow(btnPickMappingFile_, (running || !mappingUseFile_) ? FALSE : TRUE);
        EnableWindow(editMappingFilePath_, (running || !mappingUseFile_) ? FALSE : TRUE);
        EnableWindow(editMappingJsonPaste_, (running || mappingUseFile_) ? FALSE : TRUE);

        EnableWindow(comboCom_, running ? FALSE : TRUE);
        EnableWindow(btnRefreshCom_, running ? FALSE : TRUE);
        EnableWindow(editBaud_, running ? FALSE : TRUE);

        EnableWindow(btnPickOneKey_, running ? FALSE : TRUE);
        EnableWindow(btnPickToggleKey_, running ? FALSE : TRUE);

        EnableWindow(stMode_, running ? FALSE : TRUE);
        EnableWindow(btnTestCapture_, running ? FALSE : TRUE);
        EnableWindow(btnSelectRegion_, running ? FALSE : TRUE);
    }

    void UpdateRegionLabel_() {
        std::wstringstream ss;
        ss << L"x=" << region_.left
            << L", y=" << region_.top
            << L", w=" << (region_.right - region_.left)
            << L", h=" << (region_.bottom - region_.top);
        SetWindowTextW(stRegion_, ss.str().c_str());
    }

    void DemoSetFrame_() {
        // Demo: create a checker pattern BGRA frame.
        BgraFrame f;
        f.width = 320;
        f.height = 240;
        f.stride = f.width * 4;
        f.data.resize((size_t)f.stride * f.height);
        f.frame_id = 1;

        for (int y = 0; y < f.height; ++y) {
            for (int x = 0; x < f.width; ++x) {
                bool on = ((x / 16) % 2) ^ ((y / 16) % 2);
                uint8_t v = on ? 220 : 60;
                size_t idx = (size_t)y * f.stride + (size_t)x * 4;
                // BGRA
                f.data[idx + 0] = v;
                f.data[idx + 1] = v;
                f.data[idx + 2] = v;
                f.data[idx + 3] = 255;
            }
        }
        imageView_.SetFrame(f);
        AppendLog(editLog_, L"[IMG] Demo frame updated.");
    }

    AppConfig CollectConfigFromUI_() {
        AppConfig c;

        // Config path
        c.configJsonPath = baseConfigPath_;

        // Mapping mode
        c.mappingUseFile = mappingUseFile_;
        c.mappingFilePath = GetWindowTextWString(editMappingFilePath_);
        c.mappingJsonText = GetWindowTextWString(editMappingJsonPaste_);

        // Serial
        c.baud = _wtoi(GetWindowTextWString(editBaud_).c_str());
        if (c.baud <= 0) c.baud = 115200;

        int sel = (int)SendMessageW(comboCom_, CB_GETCURSEL, 0, 0);
        if (sel >= 0 && sel < (int)comItems_.size()) {
            c.comFriendly = comItems_[sel].friendly;
            c.comName = comItems_[sel].comName;
        }

        // Hotkeys
        c.oneKeyVk = oneKeyVk_.has_value() ? (int)oneKeyVk_.value() : 0;
        c.ignore_trigger_case = ignore_trigger_case_;

        c.toggleKeyVk = toggleKeyVk_.has_value() ? (int)toggleKeyVk_.value() : 0;
        c.ignore_switch_case = ignore_switch_case_;

        // Mode from combobox
        int selMode = (int)SendMessageW(stMode_, CB_GETCURSEL, 0, 0);
        c.modeType = utils::index_mode(selMode);

        // Region
        c.x = region_.left;
        c.y = region_.top;
        c.w = region_.right - region_.left;
        c.h = region_.bottom - region_.top;

        return c;
    }

    void ApplyConfigToUI_(const AppConfig& c) {
        // Config
        if (!c.configJsonPath.empty())
            baseConfigPath_ = c.configJsonPath;

        // Mapping mode UI
        mappingUseFile_ = c.mappingUseFile;
        SendMessageW(radioMapFile_, BM_SETCHECK, mappingUseFile_ ? BST_CHECKED : BST_UNCHECKED, 0);
        SendMessageW(radioMapPaste_, BM_SETCHECK, !mappingUseFile_ ? BST_CHECKED : BST_UNCHECKED, 0);

        SetWindowTextW(editMappingFilePath_, c.mappingFilePath.c_str());
        SetWindowTextW(editMappingJsonPaste_, c.mappingJsonText.c_str());

        EnableWindow(btnPickMappingFile_, mappingUseFile_ ? TRUE : FALSE);
        EnableWindow(editMappingFilePath_, mappingUseFile_ ? TRUE : FALSE);
        EnableWindow(editMappingJsonPaste_, mappingUseFile_ ? FALSE : TRUE);

        // Serial
        {
            wchar_t buf[32]{};
            _snwprintf_s(buf, _TRUNCATE, L"%d", c.baud > 0 ? c.baud : 115200);
            SetWindowTextW(editBaud_, buf);
        }

        // Refresh COM list first, then select if possible
        RefreshComList_();
        if (!c.comName.empty()) {
            for (int i = 0; i < (int)comItems_.size(); ++i) {
                if (comItems_[i].comName == c.comName) {
                    SendMessageW(comboCom_, CB_SETCURSEL, i, 0);
                    break;
                }
            }
        }

        // Hotkeys
        if (c.oneKeyVk != 0) {
            oneKeyVk_ = (UINT)c.oneKeyVk;
            ignore_trigger_case_ = c.ignore_trigger_case;
            SetWindowTextW(stOneKeyValue_, FormatHotkeyDisplay((UINT)c.oneKeyVk, ignore_trigger_case_).c_str());
        }
        else {
            SetWindowTextW(stOneKeyValue_, L"(none)");
        }

        if (c.toggleKeyVk != 0) {
            toggleKeyVk_ = (UINT)c.toggleKeyVk;
            ignore_switch_case_ = c.ignore_switch_case;
            SetWindowTextW(stToggleValue_, FormatHotkeyDisplay((UINT)c.toggleKeyVk, ignore_switch_case_).c_str());
        }
        else {
            SetWindowTextW(stToggleValue_, L"(none)");
        }

        // Mode
        //auto SelectModeByText = [&](const std::wstring& text) {
        //    int cnt = (int)SendMessageW(stMode_, CB_GETCOUNT, 0, 0);
        //    for (int i = 0; i < cnt; ++i) {
        //        wchar_t buf[64]{};
        //        SendMessageW(stMode_, CB_GETLBTEXT, i, (LPARAM)buf);
        //        if (text == buf) {
        //            SendMessageW(stMode_, CB_SETCURSEL, i, 0);
        //            return;
        //        }
        //    }
        //    // fallback
        //    SendMessageW(stMode_, CB_SETCURSEL, 0, 0);
        //    };
        //SelectModeByText(c.modeType.empty() ? L"自动" : c.modeType);
        auto mode_index = utils::mode_index(c.modeType);
        SendMessageW(stMode_, CB_SETCURSEL, mode_index, 0);

        // Region
        region_.left = c.x;
        region_.top = c.y;
        region_.right = c.x + c.w;
        region_.bottom = c.y + c.h;
        UpdateRegionLabel_();
    }

    bool SaveDefaultConfig_() {
        AppConfig c = CollectConfigFromUI_();
        json j = c;

        std::wstring path = GetDefaultJsonPath();
        std::string text = j.dump(2);

        if (!WriteTextFileUtf8(path, text)) {
            AppendLog(editLog_, L"[CFG] Failed to write default.json");
            return false;
        }
        AppendLog(editLog_, L"[CFG] Saved: " + path);
        return true;
    }

    bool LoadDefaultConfig_() {
        std::wstring path = GetDefaultJsonPath();
        std::string text;
        if (!ReadTextFileUtf8(path, text)) {
            AppendLog(editLog_, L"[CFG] default.json not found (skip).");
            return false;
        }

        try {
            json j = json::parse(text);
            AppConfig c = j.get<AppConfig>();
            ApplyConfigToUI_(c);
            AppendLog(editLog_, L"[CFG] Loaded: " + path);
            return true;
        }
        catch (...) {
            AppendLog(editLog_, L"[CFG] Failed to parse default.json");
            return false;
        }
    }

    LRESULT WndProc_(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        switch (msg) {
        case WM_CREATE: {
            // Controls creation
            btnPickConfig_ = CreateWindowW(L"BUTTON", L"Config.json...",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_BTN_OPEN_CONFIG, GetModuleHandleW(nullptr), nullptr);

            // Default config path = exe_dir\config.json
            baseConfigPath_ = JoinPath(GetExeDir(), L"config.json");

            //stMappingPath_ = CreateWindowW(L"STATIC", L"Key mapping JSON file path:",
            //    WS_CHILD | WS_VISIBLE,
            //    0, 0, 0, 0,
            //    hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);
            CreateWindowW(L"STATIC", L"Key Mapping JSON:",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0, hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            radioMapFile_ = CreateWindowW(L"BUTTON", L"From file",
                WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON,
                0, 0, 0, 0, hWnd, (HMENU)ID_MAPPING_JSON_FROM_FILE, GetModuleHandleW(nullptr), nullptr);

            radioMapPaste_ = CreateWindowW(L"BUTTON", L"Paste JSON",
                WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON,
                0, 0, 0, 0, hWnd, (HMENU)ID_PAST_JSON_MAPPING, GetModuleHandleW(nullptr), nullptr);

            // default = file mode
            SendMessageW(radioMapFile_, BM_SETCHECK, BST_CHECKED, 0);
            mappingUseFile_ = true;

            btnPickMappingFile_ = CreateWindowW(L"BUTTON", L"Select file...",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0, hWnd, (HMENU)ID_SELECT_JSON_MAPPING, GetModuleHandleW(nullptr), nullptr);

            editMappingFilePath_ = CreateWindowW(L"EDIT", L"",
                WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                0, 0, 0, 0, hWnd, (HMENU)3004, GetModuleHandleW(nullptr), nullptr);

            editMappingJsonPaste_ = CreateWindowW(L"EDIT", L"",
                WS_CHILD | WS_VISIBLE | WS_BORDER | ES_MULTILINE | ES_AUTOVSCROLL | WS_VSCROLL,
                0, 0, 0, 0, hWnd, (HMENU)3005, GetModuleHandleW(nullptr), nullptr);


            // English comment:
            // Enable Ctrl+A select-all for mapping edit boxes.
            SetWindowSubclass(editMappingFilePath_, EditCtrlASubclassProc, 1, 0);
            SetWindowSubclass(editMappingJsonPaste_, EditCtrlASubclassProc, 1, 0);

            // Start with paste editor disabled (file mode)
            EnableWindow(editMappingJsonPaste_, FALSE);

            editMappingPath_ = CreateWindowW(L"EDIT", L"",
                WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_EDIT_MAPPING_PATH, GetModuleHandleW(nullptr), nullptr);

            btnApply_ = CreateWindowW(L"BUTTON", L"Apply",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_BTN_APPLY, GetModuleHandleW(nullptr), nullptr);

            btnStart_ = CreateWindowW(L"BUTTON", L"Start",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_BTN_START, GetModuleHandleW(nullptr), nullptr);

            btnStop_ = CreateWindowW(L"BUTTON", L"Stop",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_BTN_STOP, GetModuleHandleW(nullptr), nullptr);

            stCom_ = CreateWindowW(L"STATIC", L"COM:",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            comboCom_ = CreateWindowW(L"COMBOBOX", L"",
                WS_CHILD | WS_VISIBLE | WS_BORDER | CBS_DROPDOWNLIST,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_COMBO_COM, GetModuleHandleW(nullptr), nullptr);

            btnRefreshCom_ = CreateWindowW(L"BUTTON", L"Refresh",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_BTN_REFRESH_COM, GetModuleHandleW(nullptr), nullptr);

            stBaud_ = CreateWindowW(L"STATIC", L"Baud:",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            editBaud_ = CreateWindowW(L"EDIT", L"9600",
                WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_EDIT_BAUD, GetModuleHandleW(nullptr), nullptr);

            stOneKey_ = CreateWindowW(L"STATIC", L"One-key output hotkey:",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            btnPickOneKey_ = CreateWindowW(L"BUTTON", L"SELECT",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_BTN_PICK_ONEKEY, GetModuleHandleW(nullptr), nullptr);

            stOneKeyValue_ = CreateWindowW(L"STATIC", L"(none)",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            stToggleKey_ = CreateWindowW(L"STATIC", L"Toggle function hotkey:",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            btnPickToggleKey_ = CreateWindowW(L"BUTTON", L"SELECT",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_BTN_PICK_TOGGLEKEY, GetModuleHandleW(nullptr), nullptr);

            stToggleValue_ = CreateWindowW(L"STATIC", L"(none)",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            stModeLabel_ = CreateWindowW(L"STATIC", L"Mode:",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            stMode_ = CreateWindowW(L"COMBOBOX", L"",
                WS_CHILD | WS_VISIBLE | WS_BORDER | CBS_DROPDOWNLIST,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_STATIC_MODE, GetModuleHandleW(nullptr), nullptr);

            // Fill options
            for (int index = 0; index < 3; index++)
            {
                SendMessageW(stMode_, CB_ADDSTRING, 0, (LPARAM)utils::to_wstring(utils::index_mode(index)).c_str());
            }

            // Default = 自动 (index 0)
            SendMessageW(stMode_, CB_SETCURSEL, 0, 0);

            stRegionLabel_ = CreateWindowW(L"STATIC", L"Capture region:",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            stRegion_ = CreateWindowW(L"STATIC", L"x=0,y=0,w=0,h=0",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_STATIC_REGION, GetModuleHandleW(nullptr), nullptr);

            btnTestCapture_ = CreateWindowW(L"BUTTON", L"Test capture",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_BTN_TEST_CAPTURE, GetModuleHandleW(nullptr), nullptr);

            btnSelectRegion_ = CreateWindowW(L"BUTTON", L"Select region",
                WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_BTN_SELECT_REGION, GetModuleHandleW(nullptr), nullptr);

            stLogLabel_ = CreateWindowW(L"STATIC", L"Log:",
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0,
                hWnd, nullptr, GetModuleHandleW(nullptr), nullptr);

            editLog_ = CreateWindowW(L"EDIT", L"",
                WS_CHILD | WS_VISIBLE | WS_BORDER |
                ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY | WS_VSCROLL,
                0, 0, 0, 0,
                hWnd, (HMENU)ID_LOG, GetModuleHandleW(nullptr), nullptr);

            imageView_.Create(hWnd, 0, 0, 0, 0, ID_IMAGEVIEW);

            // Initial state
            region_ = { 0, 0, 0, 0 };
            UpdateRegionLabel_();
            RefreshComList_();
            LoadDefaultConfig_();
            DemoSetFrame_();
            AppendLog(editLog_, L"[INIT] UI created.");
            return 0;
        }
        case WM_SIZE: {
            Layout_();
            UpdateRunUiState_();
            return 0;
        }
        case WM_COMMAND: {
            switch (LOWORD(wParam)) {
            case ID_BTN_REFRESH_COM: {
                RefreshComList_();
                return 0;
            }
            case ID_BTN_PICK_ONEKEY: {
                HotkeyPickerDialog dlg;
                auto r = dlg.ShowModal(hwnd_, L"选择一键输出按键");
                if (r.ok) {
                    oneKeyVk_ = r.vk;
                    ignore_trigger_case_ = r.caseInsensitive;

                    SetWindowTextW(stOneKeyValue_, FormatHotkeyDisplay(r.vk, r.caseInsensitive).c_str());
                    AppendLog(editLog_, L"[KEY] One-key hotkey selected.");
                }
                return 0;
            }

            case ID_BTN_PICK_TOGGLEKEY: {
                HotkeyPickerDialog dlg;
                auto r = dlg.ShowModal(hwnd_, L"选择切换功能按键");
                if (r.ok) {
                    toggleKeyVk_ = r.vk;
                    ignore_switch_case_ = r.caseInsensitive;

                    SetWindowTextW(stToggleValue_, FormatHotkeyDisplay(r.vk, r.caseInsensitive).c_str());
                    AppendLog(editLog_, L"[KEY] Toggle hotkey selected.");
                }
                return 0;
            }


            case ID_BTN_TEST_CAPTURE: {
                // English comment:
                // Capture using current region_ and show in ImageView. If region is invalid, do nothing.
                int w = (region_.right - region_.left);
                int h = (region_.bottom - region_.top);
                if (w <= 0 || h <= 0) {
                    AppendLog(editLog_, L"[CAP] Invalid region, skip test capture.");
                    return 0;
                }

                BgraFrame frame;
                if (!ScreenRegionSelector::CaptureRegionToBgraFrame(region_, frame)) {
                    AppendLog(editLog_, L"[CAP] Test capture failed.");
                    MessageBoxW(hwnd_, L"截图失败（Capture failed）", L"Test capture", MB_OK | MB_ICONERROR);
                    return 0;
                }

                frame.frame_id = 0;
                imageView_.SetFrame(frame);
                AppendLog(editLog_, L"[IMG] Test capture updated.");
                return 0;
            }

            case ID_BTN_SELECT_REGION: {
                ScreenRegionSelector selector;
                RECT rc{};
                if (!selector.SelectRegionVirtual(rc)) {
                    AppendLog(editLog_, L"[CAP] Region selection canceled.");
                    return 0;
                }

                // Save region
                region_ = rc;
                UpdateRegionLabel_();
                AppendLog(editLog_, L"[CAP] Region selected.");

                // Capture screenshot immediately and show in ImageView
                BgraFrame frame;
                if (!ScreenRegionSelector::CaptureRegionToBgraFrame(region_, frame)) {
                    AppendLog(editLog_, L"[CAP] Capture failed.");
                    MessageBoxW(hwnd_, L"截图失败（Capture failed）", L"Select region", MB_OK | MB_ICONERROR);
                    return 0;
                }

                frame.frame_id = 0;
                imageView_.SetFrame(frame);
                AppendLog(editLog_, L"[IMG] Region screenshot updated.");
                return 0;
            }

            case ID_BTN_APPLY: {
                if (wowapp::IsRunning()) {
                    MessageBoxW(hwnd_, L"正在运行中，请先 Stop 再 Apply。", L"Apply", MB_OK | MB_ICONINFORMATION);
                    return 0;
                }
                // 1) Validate (keep your existing validation, but update for new mapping modes)
                if (!ValidateAndApply_()) {
                    MessageBoxW(hwnd_, L"Apply failed.", L"Apply", MB_OK | MB_ICONERROR);
                    return 0;
                }

                // 2) Save default.json
                if (!SaveDefaultConfig_()) {
                    MessageBoxW(hwnd_, L"Apply succeeded, but failed to save default.json.", L"Apply", MB_OK | MB_ICONWARNING);
                    return 0;
                }

                MessageBoxW(hwnd_, L"Apply succeeded and saved default.json.", L"Apply", MB_OK | MB_ICONINFORMATION);
                return 0;
            }


            case ID_BTN_START: {
                if (wowapp::IsRunning()) {
                    UpdateRunUiState_();
                    return 0;
                }

                // Ensure config is valid and persisted
                if (!ValidateAndApply_()) {
                    MessageBoxW(hwnd_, L"Start failed (Apply failed).", L"Start", MB_OK | MB_ICONERROR);
                    UpdateRunUiState_();
                    return 0;
                }

                // Load current config.json and start embedded runtime
                std::wstring cfgPath = baseConfigPath_.empty() ? JoinPath(GetExeDir(), L"config.json") : baseConfigPath_;
                json cfg;
                if (!LoadJsonUtf8_(cfgPath, cfg)) {
                    MessageBoxW(hwnd_, L"Failed to load config.json for Start().", L"Start", MB_OK | MB_ICONERROR);
                    UpdateRunUiState_();
                    return 0;
                }

                std::string cfgStr = cfg.dump();
                if (!wowapp::Start(cfgStr)) {
                    MessageBoxW(hwnd_, L"Start() returned false. Please check your paths and region.", L"Start", MB_OK | MB_ICONERROR);
                    UpdateRunUiState_();
                    return 0;
                }

                AppendLog(editLog_, L"[RUN] Started.");
                UpdateRunUiState_();
                return 0;
            }

            case ID_BTN_STOP: {
                if (!wowapp::IsRunning()) {
                    UpdateRunUiState_();
                    return 0;
                }
                wowapp::Stop(1);
                AppendLog(editLog_, L"[RUN] Stopped.");
                UpdateRunUiState_();
                return 0;
            }


            case ID_BTN_OPEN_CONFIG: {
                std::wstring def = baseConfigPath_;
                std::wstring sel;
                if (OpenJsonFileDialog(hwnd_, def, sel)) {
                    baseConfigPath_ = sel;
                    AppendLog(editLog_, L"[CFG] Config selected: " + baseConfigPath_);
                    MessageBoxW(hwnd_, baseConfigPath_.c_str(), L"Config.json selected", MB_OK | MB_ICONINFORMATION);
                }
                return 0;
            }
            case ID_STATIC_MODE: {
                if (HIWORD(wParam) == CBN_SELCHANGE) {
                    int sel = (int)SendMessageW(stMode_, CB_GETCURSEL, 0, 0);
                    wchar_t buf[64]{};
                    SendMessageW(stMode_, CB_GETLBTEXT, sel, (LPARAM)buf);
                    AppendLog(editLog_, std::wstring(L"[MODE] Changed to: ") + buf);
                }
                return 0;
            }

            case ID_MAPPING_JSON_FROM_FILE: { // From file
                if (HIWORD(wParam) == BN_CLICKED) {
                    mappingUseFile_ = true;
                    EnableWindow(btnPickMappingFile_, TRUE);
                    EnableWindow(editMappingFilePath_, TRUE);
                    EnableWindow(editMappingJsonPaste_, FALSE);
                    AppendLog(editLog_, L"[MAP] Mode=From file");
                }
                return 0;
            }

            case ID_PAST_JSON_MAPPING: { // Paste JSON
                if (HIWORD(wParam) == BN_CLICKED) {
                    mappingUseFile_ = false;
                    EnableWindow(btnPickMappingFile_, FALSE);
                    EnableWindow(editMappingFilePath_, FALSE);
                    EnableWindow(editMappingJsonPaste_, TRUE);
                    AppendLog(editLog_, L"[MAP] Mode=Paste JSON");
                }
                return 0;
            }

            case ID_SELECT_JSON_MAPPING: { // Select mapping file...
                std::wstring cur = GetWindowTextWString(editMappingFilePath_);
                std::wstring def = cur.empty() ? JoinPath(GetExeDir(), L"key_mapping.json") : cur;
                std::wstring sel;
                if (OpenJsonFileDialog(hwnd_, def, sel)) {
                    SetWindowTextW(editMappingFilePath_, sel.c_str());
                    AppendLog(editLog_, L"[MAP] File selected: " + sel);
                }
                return 0;
            }

            default:
                break;
            }
            break;
        }
        case WM_CLOSE:
            DestroyWindow(hWnd);
            return 0;
        case WM_DESTROY:
            if (wowapp::IsRunning()) {
                wowapp::Stop(0);
            }
            PostQuitMessage(0);
            return 0;
        }
        return DefWindowProcW(hWnd, msg, wParam, lParam);
    }

private:
    HWND hwnd_ = nullptr;

    // Config window
    std::unique_ptr<ConfigWindow> configWindow_;

    // Controls
    HWND btnOpenConfig_ = nullptr;
    std::wstring baseConfigPath_;     // Config.json selected path
    std::wstring mappingJsonText_;    // pasted json (optional cache)
    bool mappingUseFile_ = true;      // true=file mode, false=paste mode

    HWND btnPickConfig_ = nullptr;    // reuse your btnOpenConfig_ or rename
    HWND radioMapFile_ = nullptr;
    HWND radioMapPaste_ = nullptr;
    HWND btnPickMappingFile_ = nullptr;
    HWND editMappingFilePath_ = nullptr; // single-line path
    HWND editMappingJsonPaste_ = nullptr; // multi-line pasted json

    HWND stMappingPath_ = nullptr;
    HWND editMappingPath_ = nullptr;
    HWND btnApply_ = nullptr;
    HWND btnStart_ = nullptr;
    HWND btnStop_ = nullptr;

    HWND stCom_ = nullptr;
    HWND comboCom_ = nullptr;
    HWND btnRefreshCom_ = nullptr;

    HWND stBaud_ = nullptr;
    HWND editBaud_ = nullptr;

    HWND stOneKey_ = nullptr;
    HWND btnPickOneKey_ = nullptr;
    HWND stOneKeyValue_ = nullptr;
    bool ignore_trigger_case_ = false;
    bool ignore_switch_case_ = false;

    HWND stToggleKey_ = nullptr;
    HWND btnPickToggleKey_ = nullptr;
    HWND stToggleValue_ = nullptr;

    HWND stModeLabel_ = nullptr;
    HWND stMode_ = nullptr;

    HWND stRegionLabel_ = nullptr;
    HWND stRegion_ = nullptr;
    HWND btnTestCapture_ = nullptr;
    HWND btnSelectRegion_ = nullptr;
    HWND stLogLabel_ = nullptr;
    HWND editLog_ = nullptr;

    ImageView imageView_;

    // Data/state
    std::vector<ComPortItem> comItems_;
    std::optional<UINT> oneKeyVk_;
    std::optional<UINT> toggleKeyVk_;
    RECT region_{};
};

// -----------------------------

#include "ui_main_app.h"

int RunUiApp() {
    MainWindow win;
    if (!win.Create()) return 0;

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return 0;
}
