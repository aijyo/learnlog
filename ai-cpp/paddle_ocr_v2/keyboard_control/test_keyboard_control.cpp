#include "test_keyboard_control.h"
#include "keyboard_control.h"

#include <windows.h>
#include <cstdio>
#include <memory>

KeyboardControlTester::KeyboardControlTester(const Options& opt)
    : opt_(opt) {
}

bool KeyboardControlTester::Open_() {
    // English comment:
    // Create and open the KeyboardControl instance.
    if (kb_) return true;

    auto kb = std::make_unique<KeyboardControl>();
    kb->SetAddr(opt_.addr);
    kb->SetWaitAck(opt_.wait_ack);
    kb->SetDebug(opt_.debug);
    kb->SetTimeoutMs(500);

    if (!kb->Open(opt_.com_port, opt_.baud)) {
        std::printf("[TEST] Open failed: %s baud=%d\n", opt_.com_port.c_str(), opt_.baud);
        return false;
    }

    kb_ = kb.release();
    std::printf("[TEST] Open OK: %s baud=%d addr=0x%02X wait_ack=%d\n",
        opt_.com_port.c_str(), opt_.baud, opt_.addr, opt_.wait_ack ? 1 : 0);
    return true;
}

void KeyboardControlTester::Close_() {
    if (!kb_) return;
    kb_->Close();
    delete kb_;
    kb_ = nullptr;
}

bool KeyboardControlTester::Tap(const std::string& key_spec, int hold_ms, int gap_ms) {
    // English comment:
    // Match python tap_key(): key down -> sleep(hold_ms) -> key up -> sleep(gap_ms)
    if (!Open_()) return false;

    const int h = (hold_ms >= 0) ? hold_ms : opt_.hold_ms;
    const int g = (gap_ms >= 0) ? gap_ms : opt_.gap_ms;

    if (!kb_->KeyDown(key_spec)) {
        std::printf("[TEST] KeyDown failed: %s\n", key_spec.c_str());
        return false;
    }

    ::Sleep((DWORD)h);

    if (!kb_->KeyUp()) {
        std::printf("[TEST] KeyUp failed: %s\n", key_spec.c_str());
        return false;
    }

    ::Sleep((DWORD)g);
    return true;
}

bool KeyboardControlTester::SendRawReport(uint8_t modifier, uint8_t keycode, int hold_ms, int gap_ms) {
    // English comment:
    // Send raw 8-byte keyboard report, identical to python build_keyboard_report(mod, kc).
    if (!Open_()) return false;

    const int h = (hold_ms >= 0) ? hold_ms : opt_.hold_ms;
    const int g = (gap_ms >= 0) ? gap_ms : opt_.gap_ms;

    uint8_t down[8] = { modifier, 0x00, keycode, 0, 0, 0, 0, 0 };
    if (!kb_->SendKeyboardReport(down)) {
        std::printf("[TEST] SendKeyboardReport DOWN failed: mod=0x%02X kc=0x%02X\n", modifier, keycode);
        return false;
    }

    ::Sleep((DWORD)h);

    uint8_t up[8] = { 0,0,0,0,0,0,0,0 };
    if (!kb_->SendKeyboardReport(up)) {
        std::printf("[TEST] SendKeyboardReport UP failed\n");
        return false;
    }

    ::Sleep((DWORD)g);
    return true;
}

bool KeyboardControlTester::TestCtrlA() {
    // English comment:
    // Equivalent to python: tap_key("CTRL+a")
    std::printf("[TEST] CTRL+A\n");
    return Tap("CTRL+a");
}

bool KeyboardControlTester::TestCtrlC() {
    std::printf("[TEST] CTRL+C\n");
    return Tap("CTRL+c");
}

bool KeyboardControlTester::TestAltTab() {
    std::printf("[TEST] ALT+TAB\n");
    return Tap("ALT+tab", /*hold_ms=*/80, /*gap_ms=*/80);
}

bool KeyboardControlTester::RunBasicSuite() {
    // English comment:
    // A minimal suite similar to what you do in Python quick testing.
    if (!Open_()) return false;

    bool ok = true;

    ok = ok && Tap("a");
    ok = ok && Tap("ENTER");
    ok = ok && TestCtrlA();
    ok = ok && TestCtrlC();
    ok = ok && TestAltTab();

    // English comment:
    // Demonstrate raw report version (CTRL+A):
    // modifier = MOD_LCTRL, keycode = HID('a')=0x04.
    ok = ok && SendRawReport(KeyboardControl::MOD_LCTRL, 0x04);

    Close_();
    std::printf("[TEST] Suite result: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}
