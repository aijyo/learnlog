#pragma once
#include <string>
#include <vector>
#include <cstdint>

class KeyboardControl;

class KeyboardControlTester {
public:
    struct Options {
        std::string com_port = "COM3";
        int baud = 9600;
        uint8_t addr = 0x00;
        bool wait_ack = true;
        bool debug = false;
        int hold_ms = 30;
        int gap_ms = 30;
    };

public:
    explicit KeyboardControlTester(const Options& opt);

    // Run a small suite similar to your Python CLI demo usage.
    bool RunBasicSuite();

    // Focused tests
    bool TestCtrlA();
    bool TestCtrlC();
    bool TestAltTab();

    // Generic test: match python tap_key(key_spec, hold_ms, gap_ms).
    bool Tap(const std::string& key_spec, int hold_ms = -1, int gap_ms = -1);

    // Advanced: raw 8-byte keyboard report (same shape as python build_keyboard_report).
    bool SendRawReport(uint8_t modifier, uint8_t keycode, int hold_ms = -1, int gap_ms = -1);

private:
    bool Open_();
    void Close_();

private:
    Options opt_;
    KeyboardControl* kb_ = nullptr; // Owned in cpp
};
