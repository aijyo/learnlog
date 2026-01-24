#include "text_handler.h"
#include "./keyboard_control/keyboard_control.h"

#include <cstdio>
#include <memory>
#include <algorithm>
#include <sstream>
#include <charconv>
#include <string>

TextHandler* TextHandler::s_instance_ = nullptr;

TextHandler::TextHandler(const Options& opt)
    : opt_(opt) {
}

TextHandler::~TextHandler() {
    Stop();
}


// --- helpers ---
static std::string ToLowerCopy_(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static std::string TrimCopy_(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static void ReplaceAllInPlace_(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

void TextHandler::set_texts(std::vector<std::string>&& texts) {
    // English comment:
    // OCR thread calls here. We store latest texts and update mapped shortcut atomically under mutex.
    std::string new_spec = AnalyzeShortcutFromTexts_(texts);

    {
        std::lock_guard<std::mutex> lk(state_mu_);
        latest_texts_ = std::move(texts);

        // English comment:
        // If we found a valid shortcut, update mapping.
        // If not found, keep previous mapping (do not override).
        if (!new_spec.empty()) {
            mapped_key_spec_ = std::move(new_spec);
        }
    }
}

static bool split(const std::string& s,
    std::string& out1,
    std::string& out2,
    char split_ch) {
    auto pos = s.find(split_ch);
    if (pos == std::string::npos) {
        return false;
    }

    out1 = s.substr(0, pos);
    out2 = s.substr(pos + 1);
    return true;
}

static bool string_to_u64(const std::string& s, uint64_t& out) {
    auto res = std::from_chars(s.data(), s.data() + s.size(), out, 10);
    return res.ec == std::errc() && res.ptr == s.data() + s.size();
}

std::string TextHandler::AnalyzeShortcutFromTexts_(const std::vector<std::string>& texts) {
    // English comment:
    // Try to find a shortcut-like pattern from OCR outputs.
    // We prefer the first valid shortcut found.
    text_analyze_.set_texts(texts);
    auto strSpellid = text_analyze_.get_key("SPELL");
    uint64_t spellid = 0;
    string_to_u64(strSpellid, spellid);

    auto gcdRemain = text_analyze_.get_key("REMAIN");
    auto scd = text_analyze_.get_key("SCD");
    std::string cd;
    std::string total_cd;
    if (!scd.empty())
    {
        split(scd, cd, total_cd, '/');
    }
    auto shortcut = user_config_.GetKeyBySpellId(spellid);

    // Nothing found
    return shortcut;
}

std::string TextHandler::NormalizeShortcutSpec_(std::string s) {
    // English comment:
    // Normalize shortcut string to the keySpec format that KeyboardControl expects:
    // e.g. "CTRL+A" -> "CTRL+a", "alt+TAB" -> "ALT+tab", "Win+R" -> "WIN+r".
    s = TrimCopy_(s);
    if (s.empty()) return "";

    // remove extra spaces around '+'
    ReplaceAllInPlace_(s, " +", "+");
    ReplaceAllInPlace_(s, "+ ", "+");
    ReplaceAllInPlace_(s, " ", ""); // after cleanup, remove remaining spaces

    // unify delimiter
    ReplaceAllInPlace_(s, "-", "+"); // tolerate "ctrl-a" -> "ctrl+a" (optional)

    // lower then rebuild with canonical upper mods
    std::string lower = ToLowerCopy_(s);

    // Quick allowlist for single keys
    // English comment:
    // For single key like "1" / "f1" / "tab", keep as-is (normalized).
    auto has_plus = (lower.find('+') != std::string::npos);
    if (!has_plus) {
        // canonical some names
        if (lower == "escape") lower = "esc";
        return lower;
    }

    // Split by '+'
    std::vector<std::string> parts;
    {
        std::stringstream ss(lower);
        std::string tok;
        while (std::getline(ss, tok, '+')) {
            tok = TrimCopy_(tok);
            if (!tok.empty()) parts.push_back(tok);
        }
    }
    if (parts.size() < 2) return "";

    // last token is the base key
    std::string key = parts.back();
    parts.pop_back();

    // normalize modifiers
    // English comment:
    // Keep only known modifiers, ignore unknown tokens.
    std::vector<std::string> mods_out;
    mods_out.reserve(parts.size());
    for (auto& m : parts) {
        if (m == "control") m = "ctrl";
        if (m == "command") m = "gui";
        if (m == "windows") m = "win";
        if (m == "option") m = "alt";

        if (m == "ctrl" || m == "shift" || m == "alt" || m == "win" || m == "gui" ||
            m == "lctrl" || m == "lshift" || m == "lalt" || m == "lgui" ||
            m == "rctrl" || m == "rshift" || m == "ralt" || m == "rgui") {
            mods_out.push_back(m);
        }
        else {
            // unknown token -> reject
            return "";
        }
    }

    // canonicalize key aliases
    if (key == "escape") key = "esc";
    if (key == "return") key = "enter";
    if (key == "del") key = "delete";

    // rebuild with uppercase modifiers (KeyboardControl parser is case-insensitive,
    // but we output in a consistent style)
    std::string out;
    for (size_t i = 0; i < mods_out.size(); ++i) {
        std::string mm = mods_out[i];
        // English comment:
        // Canonical mod token to upper for readability.
        std::transform(mm.begin(), mm.end(), mm.begin(), [](unsigned char c) { return (char)std::toupper(c); });
        if (!out.empty()) out += "+";
        out += mm;
    }

    // key keep lower (like "a", "tab", "f1", "1")
    if (!out.empty()) out += "+";
    out += key;

    return out;
}

void TextHandler::SetMappedKey(const std::string& key_spec) {
    // English comment:
    // Update the key mapping. This controls what we send over serial when trigger key is pressed.
    mapped_key_spec_ = key_spec.empty() ? "1" : key_spec;
}

bool TextHandler::OpenSerial_() {
    // English comment:
    // Create and open KeyboardControl.
    if (kb_) return true;

    auto kb = std::make_unique<KeyboardControl>();
    kb->SetAddr(opt_.addr);
    kb->SetWaitAck(opt_.wait_ack);
    kb->SetDebug(opt_.debug);
    kb->SetTimeoutMs(500);

    if (!kb->Open(opt_.com_port, opt_.baud)) {
        std::printf("[TextHandler] Open serial failed: %s baud=%d\n",
            opt_.com_port.c_str(), opt_.baud);
        return false;
    }

    kb_ = kb.release();
    std::printf("[TextHandler] Serial OK: %s baud=%d addr=0x%02X wait_ack=%d\n",
        opt_.com_port.c_str(), opt_.baud, opt_.addr, opt_.wait_ack ? 1 : 0);
    return true;
}

void TextHandler::CloseSerial_() {
    if (!kb_) return;
    kb_->Close();
    delete kb_;
    kb_ = nullptr;
}

bool TextHandler::IsForegroundTarget_() const {
    // English comment:
    // Same idea as redirect_input.cpp: if target exe is empty, accept any foreground process.
    if (opt_.target_exe.empty())
        return true;

    HWND fg = ::GetForegroundWindow();
    if (!fg)
        return false;

    DWORD pid = 0;
    ::GetWindowThreadProcessId(fg, &pid);
    if (!pid)
        return false;

    HANDLE h = ::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h)
        return false;

    wchar_t path[MAX_PATH] = { 0 };
    DWORD sz = MAX_PATH;
    bool ok = false;

    if (::QueryFullProcessImageNameW(h, 0, path, &sz)) {
        const wchar_t* filename = wcsrchr(path, L'\\');
        filename = filename ? filename + 1 : path;
        ok = (_wcsicmp(filename, opt_.target_exe.c_str()) == 0);
    }

    ::CloseHandle(h);
    return ok;
}

bool TextHandler::InstallHook_() {
    // English comment:
    // Install WH_KEYBOARD_LL hook; needs message loop to keep working.
    if (hook_) return true;

    s_instance_ = this;
    HHOOK h = ::SetWindowsHookExW(WH_KEYBOARD_LL, LowLevelKeyboardProc_,
        ::GetModuleHandleW(nullptr), 0);
    if (!h) {
        std::printf("[TextHandler] SetWindowsHookEx failed: %lu\n", ::GetLastError());
        s_instance_ = nullptr;
        return false;
    }

    hook_ = (void*)h;
    return true;
}

void TextHandler::UninstallHook_() {
    if (!hook_) return;
    ::UnhookWindowsHookEx((HHOOK)hook_);
    hook_ = nullptr;
    s_instance_ = nullptr;
}

bool TextHandler::Start() {
    if (running_.load())
        return true;

    if (!OpenSerial_())
        return false;

    if (!InstallHook_()) {
        CloseSerial_();
        return false;
    }

    user_config_.set_path(opt_.user_config);
    running_.store(true);

    std::printf("[TextHandler] Running.\n");
    std::printf("[TextHandler] Trigger VK=0x%02X, mapped='%s', targetExe='%ls'\n",
        opt_.trigger_vk, mapped_key_spec_.c_str(),
        opt_.target_exe.empty() ? L"(any)" : opt_.target_exe.c_str());
    std::printf("[TextHandler] Press Ctrl+C to exit (or call Stop()).\n");

    // English comment:
    // Message loop blocks current thread. If you need async, run Start() in a dedicated thread.
    RunMessageLoop_();
    return true;
}

void TextHandler::Stop() {
    if (!running_.load())
        return;

    running_.store(false);

    // English comment:
    // Release any pressed mapped key.
    if (kb_) {
        kb_->KeyUp();
    }
    mapped_down_from_trigger_.store(false);

    UninstallHook_();
    CloseSerial_();

    // English comment:
    // Quit message loop if running on current thread.
    ::PostQuitMessage(0);

    std::printf("[TextHandler] Stopped.\n");
}

void TextHandler::RunMessageLoop_() {
    MSG msg{};
    while (running_.load()) {
        int ret = GetMessageW(&msg, nullptr, 0, 0);
        if (ret > 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
            continue;
        }
        if (ret == 0) {
            // WM_QUIT received. This ends the thread's message loop.
            break;
        }
        // ret == -1 -> error
        break;
    }
}

LRESULT CALLBACK TextHandler::LowLevelKeyboardProc_(int nCode, WPARAM wParam, LPARAM lParam) {
    if (!s_instance_) {
        return ::CallNextHookEx(nullptr, nCode, wParam, lParam);
    }
    return s_instance_->OnKeyboardEvent_(nCode, wParam, lParam);
}

LRESULT TextHandler::OnKeyboardEvent_(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode != HC_ACTION)
        return ::CallNextHookEx((HHOOK)hook_, nCode, wParam, lParam);

    const KBDLLHOOKSTRUCT* k = reinterpret_cast<KBDLLHOOKSTRUCT*>(lParam);

    // English comment:
    // Filter injected events to avoid recursion (same as redirect_input.cpp).
    if (k->flags & LLKHF_INJECTED)
        return ::CallNextHookEx((HHOOK)hook_, nCode, wParam, lParam);

    // Optional: only active for target exe.
    if (!IsForegroundTarget_())
        return ::CallNextHookEx((HHOOK)hook_, nCode, wParam, lParam);

    const DWORD vk = k->vkCode;
    const bool isDown = (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN);
    const bool isUp = (wParam == WM_KEYUP || wParam == WM_SYSKEYUP);

    if ((int)vk == opt_.trigger_vk) {
        // English comment:
        // Remap trigger key to mapped key over serial, and swallow trigger key events.
        if (!kb_) {
            // No serial -> don't swallow to avoid breaking user input unexpectedly.
            return ::CallNextHookEx((HHOOK)hook_, nCode, wParam, lParam);
        }

        std::string mapped;
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            mapped = mapped_key_spec_;
        }
        if (isDown) {
            // English comment:
            // Forward every trigger keydown (including auto-repeat) as mapped keydown.
            if (!mapped_down_from_trigger_.load()) {
                mapped_down_from_trigger_.store(true);
            }
            kb_->KeyDown(mapped);
        }
        else if (isUp) {
            if (mapped_down_from_trigger_.load()) {
                mapped_down_from_trigger_.store(false);
                kb_->KeyUp(mapped); // release mapped shortcut
            }
        }

        // Swallow original trigger key events.
        return 1;
    }

    return ::CallNextHookEx((HHOOK)hook_, nCode, wParam, lParam);
}
