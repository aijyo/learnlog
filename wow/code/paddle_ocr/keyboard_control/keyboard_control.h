#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

class KeyboardControl {
public:
    // CH9329 protocol constants (from your Python reference)
    static constexpr uint8_t kHead0 = 0x57;
    static constexpr uint8_t kHead1 = 0xAB;

    // Commands (keyboard is confirmed in your Python file)
    static constexpr uint8_t kCmdGetInfo = 0x01;
    static constexpr uint8_t kCmdSendKeyboardReport = 0x02; // 8-byte keyboard report【filecite】
    // Mouse command below is a common CH9329 convention; verify with your board doc if needed.
    static constexpr uint8_t kCmdSendMouseReport = 0x03;

    // HID modifiers (from your Python reference)
    enum Mod : uint8_t {
        MOD_LCTRL = 1 << 0,
        MOD_LSHIFT = 1 << 1,
        MOD_LALT = 1 << 2,
        MOD_LGUI = 1 << 3,
        MOD_RCTRL = 1 << 4,
        MOD_RSHIFT = 1 << 5,
        MOD_RALT = 1 << 6,
        MOD_RGUI = 1 << 7
    };

    // Mouse buttons (HID convention)
    enum MouseBtn : uint8_t {
        MOUSE_LEFT = 1 << 0,
        MOUSE_RIGHT = 1 << 1,
        MOUSE_MIDDLE = 1 << 2,
        MOUSE_BACK = 1 << 3,
        MOUSE_FORWARD = 1 << 4,
    };

public:
    KeyboardControl();
    ~KeyboardControl();

    KeyboardControl(const KeyboardControl&) = delete;
    KeyboardControl& operator=(const KeyboardControl&) = delete;

    // port examples: "COM6" or "\\\\.\\COM12"
    bool Open(const std::string& port, int baud = 9600);
    void Close();
    bool IsOpen() const;

    // Protocol options
    void SetAddr(uint8_t addr) { addr_ = addr; }
    void SetWaitAck(bool wait_ack) { wait_ack_ = wait_ack; }
    void SetDebug(bool debug) { debug_ = debug; }
    void SetTimeoutMs(uint32_t ms) { timeout_ms_ = ms; }

    // High-level keyboard operations
    bool TapKey(const std::string& keySpec, int hold_ms = 30, int gap_ms = 30);
    bool KeyDown(const std::string& keySpec);
    bool KeyUp(const std::string& keySpec= ""); // release all if keySpec empty

    // Low-level: send a keyboard report directly
    // report[0]=modifier, report[1]=0, report[2..7]=up to 6 keycodes
    bool SendKeyboardReport(const uint8_t report[8]);

    // Mouse operations (verify cmd/payload with your board if needed)
    // Relative move: dx,dy in [-127..127], wheel in [-127..127]
    bool MouseMove(int dx, int dy);
    bool MouseWheel(int wheel);
    bool MouseButtonDown(uint8_t buttons_mask);
    bool MouseButtonUp(); // release all buttons
    bool MouseClick(uint8_t buttons_mask, int hold_ms = 20, int gap_ms = 20);

    // Advanced: send raw cmd/payload
    bool SendCmd(uint8_t cmd, const std::vector<uint8_t>& payload,
        std::optional<uint8_t>* out_status = nullptr);

public:
    // Key map helper (subset; extend as you like)
    static const std::unordered_map<std::string, uint8_t>& HidKeyMap();

private:
    struct FrameResp {
        uint8_t addr = 0;
        int cmd = 0; // cmd=-1 means checksum mismatch
        uint8_t len = 0;
        std::vector<uint8_t> data;
        std::vector<uint8_t> raw;
    };

private:
    // Serial I/O (Win32)
    bool ConfigurePort_(int baud);
    bool WriteAll_(const uint8_t* data, size_t size);
    bool ReadExact_(uint8_t* out, size_t n, uint32_t timeout_ms);

    // Protocol utilities
    static uint8_t ChecksumSum8_(const std::vector<uint8_t>& bytes);
    static std::vector<uint8_t> BuildFrame_(uint8_t addr, uint8_t cmd, const std::vector<uint8_t>& payload);
    bool ReadOneFrame_(FrameResp& out);

    // Key parsing
    bool ParseKeySpec_(const std::string& spec, uint8_t& out_modifier, uint8_t& out_keycode);

    // Mouse report
    bool SendMouseReport_(uint8_t buttons, int dx, int dy, int wheel);

private:
    void* hComm_ = nullptr; // HANDLE
    uint8_t addr_ = 0x00;
    bool wait_ack_ = true;
    bool debug_ = false;
    uint32_t timeout_ms_ = 500;
    uint8_t current_report_[8] = { 0 };
};
