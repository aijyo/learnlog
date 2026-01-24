#include "keyboard_control.h"

#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

static std::string ToLower_(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static std::vector<std::string> SplitPlus_(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == '+') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        }
        else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    // trim spaces
    for (auto& t : out) {
        size_t a = t.find_first_not_of(" \t\r\n");
        size_t b = t.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) t.clear();
        else t = t.substr(a, b - a + 1);
    }
    out.erase(std::remove_if(out.begin(), out.end(), [](const std::string& x) { return x.empty(); }), out.end());
    return out;
}

KeyboardControl::KeyboardControl() = default;

KeyboardControl::~KeyboardControl() {
    Close();
}

bool KeyboardControl::Open(const std::string& port, int baud) {
    Close();

    std::string p = port;
    // Win32 needs "\\\\.\\COM10" style for COM>=10
    if (p.rfind("\\\\.\\", 0) != 0) {
        std::string up = p;
        std::transform(up.begin(), up.end(), up.begin(), ::toupper);
        if (up.rfind("COM", 0) == 0) {
            // Always use \\.\ form for safety
            p = "\\\\.\\" + p;
        }
    }

    HANDLE h = ::CreateFileA(
        p.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (h == INVALID_HANDLE_VALUE) {
        hComm_ = nullptr;
        return false;
    }
    hComm_ = (void*)h;

    // Reasonable buffers
    ::SetupComm(h, 4096, 4096);
    ::PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR | PURGE_RXABORT | PURGE_TXABORT);

    if (!ConfigurePort_(baud)) {
        Close();
        return false;
    }
    return true;
}

void KeyboardControl::Close() {
    if (hComm_) {
        ::CloseHandle((HANDLE)hComm_);
        hComm_ = nullptr;
    }
}

bool KeyboardControl::IsOpen() const {
    return hComm_ != nullptr;
}

bool KeyboardControl::ConfigurePort_(int baud) {
    if (!hComm_) return false;
    HANDLE h = (HANDLE)hComm_;

    DCB dcb{};
    dcb.DCBlength = sizeof(dcb);
    if (!::GetCommState(h, &dcb)) return false;

    dcb.BaudRate = baud;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;

    dcb.fBinary = TRUE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;
    dcb.fOutxCtsFlow = FALSE;
    dcb.fOutxDsrFlow = FALSE;
    dcb.fInX = FALSE;
    dcb.fOutX = FALSE;

    if (!::SetCommState(h, &dcb)) return false;

    COMMTIMEOUTS to{};
    // Reason:
    // We implement our own ReadExact timeout by setting non-infinite timeouts.
    to.ReadIntervalTimeout = 10;
    to.ReadTotalTimeoutMultiplier = 0;
    to.ReadTotalTimeoutConstant = 10;
    to.WriteTotalTimeoutMultiplier = 0;
    to.WriteTotalTimeoutConstant = 50;
    if (!::SetCommTimeouts(h, &to)) return false;

    return true;
}

bool KeyboardControl::WriteAll_(const uint8_t* data, size_t size) {
    if (!hComm_) return false;
    HANDLE h = (HANDLE)hComm_;

    size_t sent = 0;
    while (sent < size) {
        DWORD w = 0;
        if (!::WriteFile(h, data + sent, (DWORD)(size - sent), &w, nullptr)) return false;
        if (w == 0) return false;
        sent += (size_t)w;
    }
    return true;
}

bool KeyboardControl::ReadExact_(uint8_t* out, size_t n, uint32_t timeout_ms) {
    if (!hComm_) return false;
    HANDLE h = (HANDLE)hComm_;

    auto t0 = std::chrono::steady_clock::now();
    size_t got = 0;
    while (got < n) {
        DWORD r = 0;
        if (!::ReadFile(h, out + got, (DWORD)(n - got), &r, nullptr)) return false;
        if (r > 0) {
            got += (size_t)r;
            continue;
        }
        // no data, check deadline
        auto t1 = std::chrono::steady_clock::now();
        auto ms = (uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (ms >= timeout_ms) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

uint8_t KeyboardControl::ChecksumSum8_(const std::vector<uint8_t>& bytes) {
    // SUM = low-8-bit of (HEAD+ADDR+CMD+LEN+DATA)【filecite】
    uint32_t s = 0;
    for (uint8_t b : bytes) s += b;
    return (uint8_t)(s & 0xFF);
}

std::vector<uint8_t> KeyboardControl::BuildFrame_(uint8_t addr, uint8_t cmd, const std::vector<uint8_t>& payload) {
    // Frame: 57 AB addr cmd len data sum【filecite】
    if (payload.size() > 64) {
        throw std::runtime_error("CH9329 payload length must be <= 64");
    }
    std::vector<uint8_t> head = { kHead0, kHead1, addr, cmd, (uint8_t)payload.size() };
    std::vector<uint8_t> body = head;
    body.insert(body.end(), payload.begin(), payload.end());
    uint8_t sum = ChecksumSum8_(body);
    body.push_back(sum);
    return body;
}

bool KeyboardControl::ReadOneFrame_(FrameResp& out) {
    // Find header 0x57 0xAB【filecite】
    out = FrameResp{};
    uint8_t b = 0;
    uint8_t prev = 0;
    bool have_prev = false;

    auto start = std::chrono::steady_clock::now();
    auto deadline = start + std::chrono::milliseconds(timeout_ms_);

    std::vector<uint8_t> raw;

    while (std::chrono::steady_clock::now() < deadline) {
        DWORD r = 0;
        if (!::ReadFile((HANDLE)hComm_, &b, 1, &r, nullptr)) return false;
        if (r == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        raw.push_back(b);
        if (have_prev && prev == kHead0 && b == kHead1) break;
        prev = b;
        have_prev = true;
    }

    if (raw.size() < 2 || raw[raw.size() - 2] != kHead0 || raw.back() != kHead1) {
        return false;
    }

    uint8_t rest3[3]{};
    if (!ReadExact_(rest3, 3, timeout_ms_)) return false;
    raw.insert(raw.end(), rest3, rest3 + 3);

    uint8_t addr = rest3[0];
    uint8_t cmd = rest3[1];
    uint8_t len = rest3[2];

    std::vector<uint8_t> data(len);
    if (len > 0) {
        if (!ReadExact_(data.data(), len, timeout_ms_)) return false;
        raw.insert(raw.end(), data.begin(), data.end());
    }

    uint8_t sum = 0;
    if (!ReadExact_(&sum, 1, timeout_ms_)) return false;
    raw.push_back(sum);

    uint8_t expect = ChecksumSum8_(std::vector<uint8_t>(raw.begin(), raw.end() - 1));
    if (expect != sum) {
        out.addr = addr;
        out.cmd = -1; // checksum mismatch mark【filecite】
        out.len = len;
        out.data = std::move(data);
        out.raw = std::move(raw);
        return true;
    }

    out.addr = addr;
    out.cmd = (int)cmd;
    out.len = len;
    out.data = std::move(data);
    out.raw = std::move(raw);
    return true;
}

bool KeyboardControl::ParseKeySpec_(const std::string& spec, uint8_t& out_modifier, uint8_t& out_keycode) {
    // Parse "a", "enter", "SHIFT+1", "CTRL+c" ...【filecite】
    auto parts = SplitPlus_(spec);
    if (parts.empty()) return false;

    std::string base = ToLower_(parts.back());
    parts.pop_back();

    uint8_t mod = 0;
    for (auto& m0 : parts) {
        std::string m = ToLower_(m0);
        if (m == "ctrl" || m == "lctrl") mod |= MOD_LCTRL;
        else if (m == "shift" || m == "lshift") mod |= MOD_LSHIFT;
        else if (m == "alt" || m == "lalt") mod |= MOD_LALT;
        else if (m == "win" || m == "gui" || m == "lgui") mod |= MOD_LGUI;
        else if (m == "rctrl") mod |= MOD_RCTRL;
        else if (m == "rshift") mod |= MOD_RSHIFT;
        else if (m == "ralt") mod |= MOD_RALT;
        else if (m == "rgui" || m == "rwin") mod |= MOD_RGUI;
        else return false;
    }

    const auto& mp = HidKeyMap();
    auto it = mp.find(base);
    if (it == mp.end()) return false;

    out_modifier = mod;
    out_keycode = it->second;
    return true;
}

const std::unordered_map<std::string, uint8_t>& KeyboardControl::HidKeyMap() {
    // Subset based on your Python HID_KEYCODES table【filecite】
    static const std::unordered_map<std::string, uint8_t> mp = [] {
        std::unordered_map<std::string, uint8_t> m;
        // a-z
        for (int i = 0; i < 26; ++i) {
            std::string k(1, char('a' + i));
            m[k] = (uint8_t)(0x04 + i);
        }
        // digits
        m["1"] = 0x1E; m["2"] = 0x1F; m["3"] = 0x20; m["4"] = 0x21; m["5"] = 0x22;
        m["6"] = 0x23; m["7"] = 0x24; m["8"] = 0x25; m["9"] = 0x26; m["0"] = 0x27;

        // common keys
        m["enter"] = 0x28;
        m["esc"] = 0x29;
        m["backspace"] = 0x2A;
        m["tab"] = 0x2B;
        m["space"] = 0x2C;

        m["-"] = 0x2D;
        m["="] = 0x2E;
        m["["] = 0x2F;
        m["]"] = 0x30;
        m["\\"] = 0x31;
        m[";"] = 0x33;
        m["'"] = 0x34;
        m["`"] = 0x35;
        m[","] = 0x36;
        m["."] = 0x37;
        m["/"] = 0x38;

        // Useful extras (not in your Python list, but commonly needed)
        m["capslock"] = 0x39;
        m["f1"] = 0x3A; m["f2"] = 0x3B; m["f3"] = 0x3C; m["f4"] = 0x3D;
        m["f5"] = 0x3E; m["f6"] = 0x3F; m["f7"] = 0x40; m["f8"] = 0x41;
        m["f9"] = 0x42; m["f10"] = 0x43; m["f11"] = 0x44; m["f12"] = 0x45;
        m["left"] = 0x50; m["right"] = 0x4F; m["up"] = 0x52; m["down"] = 0x51;
        m["delete"] = 0x4C;
        return m;
        }();
    return mp;
}

bool KeyboardControl::SendCmd(uint8_t cmd, const std::vector<uint8_t>& payload,
    std::optional<uint8_t>* out_status) {
    if (!IsOpen()) return false;

    std::vector<uint8_t> frame;
    try {
        frame = BuildFrame_(addr_, cmd, payload);
    }
    catch (...) {
        return false;
    }

    if (debug_) {
        std::ostringstream oss;
        oss << "TX:";
        for (auto b : frame) oss << " " << std::hex << std::uppercase << (int)b;
        std::cout << oss.str() << std::dec << "\n";
    }

    if (!WriteAll_(frame.data(), frame.size())) return false;

    if (!wait_ack_ || addr_ == 0xFF) {
        if (out_status) *out_status = std::nullopt;
        return true;
    }

    FrameResp resp;
    if (!ReadOneFrame_(resp)) return false;

    if (debug_) {
        std::ostringstream oss;
        oss << "RX:";
        for (auto b : resp.raw) oss << " " << std::hex << std::uppercase << (int)b;
        std::cout << oss.str() << std::dec << "\n";
    }

    // checksum mismatch marker (cmd=-1)【filecite】
    if (resp.cmd == -1) {
        if (out_status) *out_status = 0xE4; // DEF_CMD_ERR_SUM in your Python table
        return false;
    }

    // Normal response: cmd|0x80 ; Error response: cmd|0xC0【filecite】
    const int ok_cmd = (int)(cmd | 0x80);
    const int err_cmd = (int)(cmd | 0xC0);
    if (resp.cmd != ok_cmd && resp.cmd != err_cmd) {
        return false;
    }

    if (resp.data.empty()) {
        return false;
    }

    uint8_t status = resp.data[0];
    if (out_status) *out_status = status;
    return status == 0x00;
}

bool KeyboardControl::SendKeyboardReport(const uint8_t report[8]) {
    std::vector<uint8_t> payload(report, report + 8);
    std::optional<uint8_t> st;
    return SendCmd(kCmdSendKeyboardReport, payload, &st);
}

static bool HasKey_(const uint8_t rep[8], uint8_t kc) {
    for (int i = 2; i < 8; ++i) {
        if (rep[i] == kc) return true;
    }
    return false;
}

static void AddKey_(uint8_t rep[8], uint8_t kc) {
    if (kc == 0) return;
    if (HasKey_(rep, kc)) return;
    for (int i = 2; i < 8; ++i) {
        if (rep[i] == 0) {
            rep[i] = kc;
            return;
        }
    }
    // English comment:
    // If more than 6 keys are pressed simultaneously, extra keys are ignored.
}

static void RemoveKey_(uint8_t rep[8], uint8_t kc) {
    if (kc == 0) return;
    for (int i = 2; i < 8; ++i) {
        if (rep[i] == kc) {
            rep[i] = 0;
        }
    }
}

//bool KeyboardControl::KeyDown(const std::string& keySpec) {
//    uint8_t mod = 0, kc = 0;
//    if (!ParseKeySpec_(keySpec, mod, kc)) return false;
//
//    // 8-byte report: [modifier][0][k1][k2][k3][k4][k5][k6]【filecite】
//    uint8_t rep[8] = { mod, 0x00, kc, 0, 0, 0, 0, 0 };
//    return SendKeyboardReport(rep);
//}

bool KeyboardControl::KeyDown(const std::string& keySpec) {
    uint8_t mod = 0, kc = 0;
    if (!ParseKeySpec_(keySpec, mod, kc)) return false;

    // English comment:
    // Update stateful report:
    // 1) OR-in modifier bits
    // 2) Add keycode into the 6-key array
    current_report_[0] |= mod;
    AddKey_(current_report_, kc);

    // Ensure reserved byte stays 0
    current_report_[1] = 0x00;

    return SendKeyboardReport(current_report_);
}
//bool KeyboardControl::KeyUp(const std::string& keySpec) {
//    uint8_t rep[8] = { 0,0,0,0,0,0,0,0 };
//    return SendKeyboardReport(rep);
//}
bool KeyboardControl::KeyUp(const std::string& keySpec) {
    // English comment:
    // Empty => release all keys and modifiers.
    if (keySpec.empty()) {
        std::memset(current_report_, 0, sizeof(current_report_));
        return SendKeyboardReport(current_report_);
    }

    uint8_t mod = 0, kc = 0;
    if (!ParseKeySpec_(keySpec, mod, kc)) return false;

    // English comment:
    // Release only specified modifiers and/or keycode from current state.
    if (mod != 0) {
        current_report_[0] &= (uint8_t)(~mod);
    }
    RemoveKey_(current_report_, kc);

    // Ensure reserved byte stays 0
    current_report_[1] = 0x00;

    return SendKeyboardReport(current_report_);
}

//bool KeyboardControl::TapKey(const std::string& keySpec, int hold_ms, int gap_ms) {
//    if (!KeyDown(keySpec)) return false;
//    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, hold_ms)));
//    if (!KeyUp()) return false;
//    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, gap_ms)));
//    return true;
//}
bool KeyboardControl::TapKey(const std::string& keySpec, int hold_ms, int gap_ms) {
    if (!KeyDown(keySpec)) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, hold_ms)));
    if (!KeyUp(keySpec)) return false; 
    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, gap_ms)));
    return true;
}
// ---------------- Mouse ----------------

static int ClampI8_(int v) {
    return std::max(-127, std::min(127, v));
}

bool KeyboardControl::SendMouseReport_(uint8_t buttons, int dx, int dy, int wheel) {
    // Note:
    // Common CH9329 mouse payload is 4 bytes: [buttons][x][y][wheel]
    // If your board differs, adjust here.
    dx = ClampI8_(dx);
    dy = ClampI8_(dy);
    wheel = ClampI8_(wheel);

    std::vector<uint8_t> payload;
    payload.push_back(buttons);
    payload.push_back((uint8_t)(int8_t)dx);
    payload.push_back((uint8_t)(int8_t)dy);
    payload.push_back((uint8_t)(int8_t)wheel);

    std::optional<uint8_t> st;
    return SendCmd(kCmdSendMouseReport, payload, &st);
}

bool KeyboardControl::MouseMove(int dx, int dy) {
    return SendMouseReport_(0, dx, dy, 0);
}

bool KeyboardControl::MouseWheel(int wheel) {
    return SendMouseReport_(0, 0, 0, wheel);
}

bool KeyboardControl::MouseButtonDown(uint8_t buttons_mask) {
    return SendMouseReport_(buttons_mask, 0, 0, 0);
}

bool KeyboardControl::MouseButtonUp() {
    return SendMouseReport_(0, 0, 0, 0);
}

bool KeyboardControl::MouseClick(uint8_t buttons_mask, int hold_ms, int gap_ms) {
    if (!MouseButtonDown(buttons_mask)) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, hold_ms)));
    if (!MouseButtonUp()) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(std::max(0, gap_ms)));
    return true;
}
