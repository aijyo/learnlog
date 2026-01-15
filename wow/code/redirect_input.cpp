// key_remap.cpp
// Build: cl /EHsc /W4 key_remap.cpp user32.lib
#include <windows.h>
#include <cstdio>
#include <atomic>

static HHOOK g_hook = nullptr;

// Optional: only remap when foreground process == target exe name (set empty to disable)
static const wchar_t* g_targetExe = L""; // e.g. L"A.exe"

// Track whether we have "pressed A" due to X, so we can release correctly.
static std::atomic<bool> g_aDownFromX{ false };

static bool IsForegroundTarget()
{
    if (g_targetExe[0] == L'\0')
        return true;

    HWND fg = GetForegroundWindow();
    if (!fg)
        return false;

    DWORD pid = 0;
    GetWindowThreadProcessId(fg, &pid);
    if (!pid)
        return false;

    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h)
        return false;

    wchar_t path[MAX_PATH] = { 0 };
    DWORD sz = MAX_PATH;
    bool ok = false;

    // Query full path of foreground process
    if (QueryFullProcessImageNameW(h, 0, path, &sz))
    {
        // Extract filename
        const wchar_t* filename = wcsrchr(path, L'\\');
        filename = filename ? filename + 1 : path;

        ok = (_wcsicmp(filename, g_targetExe) == 0);
    }

    CloseHandle(h);
    return ok;
}

static void SendKey(WORD vk, bool down)
{
    INPUT in = {};
    in.type = INPUT_KEYBOARD;
    in.ki.wVk = vk;
    in.ki.wScan = 0;
    in.ki.time = 0;
    in.ki.dwExtraInfo = 0;
    in.ki.dwFlags = down ? 0 : KEYEVENTF_KEYUP;
    SendInput(1, &in, sizeof(INPUT));
}

static LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    if (nCode != HC_ACTION)
        return CallNextHookEx(g_hook, nCode, wParam, lParam);

    const KBDLLHOOKSTRUCT* k = reinterpret_cast<KBDLLHOOKSTRUCT*>(lParam);

    // Filter out injected events to avoid recursion
    if (k->flags & LLKHF_INJECTED)
        return CallNextHookEx(g_hook, nCode, wParam, lParam);

    // Optional: only work when target app is foreground
    if (!IsForegroundTarget())
        return CallNextHookEx(g_hook, nCode, wParam, lParam);

    // We remap 'X' -> 'A' (virtual-key codes)
    // VK_X = 0x58, VK_A = 0x41
    const DWORD vk = k->vkCode;

    const bool isKeyDown = (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN);
    const bool isKeyUp = (wParam == WM_KEYUP || wParam == WM_SYSKEYUP);

    if (vk == 'X' || vk =='x')
    {
        if (isKeyDown)
        {
            // First time we see X down, mark state "A is logically held"
            if (!g_aDownFromX.load())
                g_aDownFromX.store(true);

            // IMPORTANT: forward every X keydown (including auto-repeat) as A keydown
            SendKey('A', true);
        }
        else if (isKeyUp)
        {
            if (g_aDownFromX.load())
            {
                g_aDownFromX.store(false);
                // Release A once when X is released
                SendKey('A', false);
            }
        }

        // Swallow original X events so target apps never see X
        return 1;
    }

    return CallNextHookEx(g_hook, nCode, wParam, lParam);
}

int wmain()
{
    wprintf(L"[key_remap] Running. Remap X -> A. TargetExe='%s'\n", g_targetExe[0] ? g_targetExe : L"(any)");
    wprintf(L"Press Ctrl+C to exit.\n");

    g_hook = SetWindowsHookExW(WH_KEYBOARD_LL, LowLevelKeyboardProc, GetModuleHandleW(nullptr), 0);
    if (!g_hook)
    {
        wprintf(L"SetWindowsHookEx failed: %lu\n", GetLastError());
        return 1;
    }

    // Message loop required for hook to work reliably
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    UnhookWindowsHookEx(g_hook);
    return 0;
}
