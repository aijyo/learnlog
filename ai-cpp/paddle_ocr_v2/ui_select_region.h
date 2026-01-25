#pragma once
#define NOMINMAX
#include <windows.h>
#include "./utils/utils.h" // BgraFrame

class ScreenRegionSelector
{
public:
    // maxW/maxH: 0 means unlimited.
    explicit ScreenRegionSelector(int maxW = 0, int maxH = 0)
        : maxW_(maxW), maxH_(maxH)
    {
        if (maxW_ < 0) maxW_ = 0;
        if (maxH_ < 0) maxH_ = 0;
    }

    bool SelectRegionVirtual(RECT& outVirtualRect)
    {
        outVirtualRect = { 0, 0, 0, 0 };

        HINSTANCE hInst = GetModuleHandleW(nullptr);
        RegisterClassOnce(hInst);

        vx_ = GetSystemMetrics(SM_XVIRTUALSCREEN);
        vy_ = GetSystemMetrics(SM_YVIRTUALSCREEN);
        vw_ = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        vh_ = GetSystemMetrics(SM_CYVIRTUALSCREEN);

        canceled_ = false;
        done_ = false;
        dragging_ = false;
        has_selection_ = false;

        anchor_screen_ = { 0, 0 };
        cur_screen_ = { 0, 0 };
        sel_virtual_ = { 0, 0, 0, 0 };

        hwnd_ = CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_LAYERED,
            kClassName,
            L"RegionSelectorOverlay",
            WS_POPUP,
            vx_, vy_, vw_, vh_,
            nullptr, nullptr, hInst, this);

        if (!hwnd_)
            return false;

        SetLayeredWindowAttributes(hwnd_, 0, (BYTE)80, LWA_ALPHA);

        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);
        SetCursor(LoadCursor(nullptr, IDC_CROSS));

        SetCapture(hwnd_);
        MSG msg{};
        while (!done_) {
            // GetMessage will block, so wake it by posting WM_NULL when done_ set from other paths if needed.
            BOOL ret = GetMessageW(&msg, nullptr, 0, 0);
            if (ret <= 0) break; // ret==0: WM_QUIT, ret==-1: error

            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        if (hwnd_)
        {
            ReleaseCapture();
            DestroyWindow(hwnd_);
            hwnd_ = nullptr;
        }

        if (canceled_)
            return false;

        RECT v = sel_virtual_;
        NormalizeRect(v);
        if ((v.right - v.left) <= 0 || (v.bottom - v.top) <= 0)
            return false;

        outVirtualRect = v;
        return true;
    }

private:
    static inline const wchar_t* kClassName = L"RegionSelectorOverlayClass_V3";

    HWND hwnd_ = nullptr;

    int vx_ = 0, vy_ = 0, vw_ = 0, vh_ = 0;

    int maxW_ = 0;
    int maxH_ = 0;

    bool canceled_ = false;
    bool done_ = false;
    bool dragging_ = false;

    // Whether we currently have a valid selection (user released left button at least once)
    bool has_selection_ = false;

    POINT anchor_screen_{};
    POINT cur_screen_{};
    RECT sel_virtual_{};

    static void NormalizeRect(RECT& r)
    {
        if (r.left > r.right) std::swap(r.left, r.right);
        if (r.top > r.bottom) std::swap(r.top, r.bottom);
    }

    static int ClampInt(int v, int lo, int hi)
    {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    POINT ClampCursorToMaxBox(const POINT& anchor, const POINT& cursor) const
    {
        POINT out = cursor;

        if (maxW_ > 0)
        {
            if (cursor.x >= anchor.x)
                out.x = ClampInt(cursor.x, anchor.x, anchor.x + maxW_);
            else
                out.x = ClampInt(cursor.x, anchor.x - maxW_, anchor.x);
        }

        if (maxH_ > 0)
        {
            if (cursor.y >= anchor.y)
                out.y = ClampInt(cursor.y, anchor.y, anchor.y + maxH_);
            else
                out.y = ClampInt(cursor.y, anchor.y - maxH_, anchor.y);
        }

        return out;
    }

    POINT ScreenToOverlayClient(const POINT& p_screen) const
    {
        POINT p = p_screen;
        p.x -= vx_;
        p.y -= vy_;
        return p;
    }

    static void RegisterClassOnce(HINSTANCE hInst)
    {
        static bool registered = false;
        if (registered) return;

        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.hInstance = hInst;
        wc.lpszClassName = kClassName;
        wc.lpfnWndProc = &ScreenRegionSelector::WndProcThunk;
        wc.hCursor = LoadCursor(nullptr, IDC_CROSS);
        wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
        wc.style = CS_HREDRAW | CS_VREDRAW;

        RegisterClassExW(&wc);
        registered = true;
    }

    static LRESULT CALLBACK WndProcThunk(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {
        ScreenRegionSelector* self = nullptr;
        if (msg == WM_NCCREATE)
        {
            auto cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
            self = reinterpret_cast<ScreenRegionSelector*>(cs->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)self);
            self->hwnd_ = hwnd;
        }
        else
        {
            self = reinterpret_cast<ScreenRegionSelector*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        }

        if (!self)
            return DefWindowProcW(hwnd, msg, wParam, lParam);

        return self->WndProc(hwnd, msg, wParam, lParam);
    }

    void ClearSelection()
    {
        // Clear current selection and wait for a new left-button down.
        has_selection_ = false;
        dragging_ = false;
        sel_virtual_ = { 0, 0, 0, 0 };
        InvalidateRect(hwnd_, nullptr, FALSE);
    }

    void UpdateSelectionFromCursor()
    {
        POINT raw{};
        GetCursorPos(&raw);

        cur_screen_ = ClampCursorToMaxBox(anchor_screen_, raw);

        sel_virtual_.left = anchor_screen_.x;
        sel_virtual_.top = anchor_screen_.y;
        sel_virtual_.right = cur_screen_.x;
        sel_virtual_.bottom = cur_screen_.y;
    }

    bool HasValidSelection() const
    {
        RECT v = sel_virtual_;
        NormalizeRect(v);
        return (v.right > v.left) && (v.bottom > v.top);
    }

    LRESULT WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {
        switch (msg)
        {
        case WM_SETCURSOR:
            SetCursor(LoadCursor(nullptr, IDC_CROSS));
            return TRUE;

        case WM_KEYDOWN:
        {
            if (wParam == VK_ESCAPE)
            {
                canceled_ = true;
                done_ = true;
                //PostQuitMessage(0);
                ReleaseCapture();                 // optional: ensure capture released early
                DestroyWindow(hwnd_);             // close overlay window
                hwnd_ = nullptr;
                return 0;
            }
            if (wParam == VK_RETURN)
            {
                // Confirm only when we have a valid selection.
                if (has_selection_ && HasValidSelection())
                {
                    done_ = true;
                    ReleaseCapture();                 // optional: ensure capture released early
                    DestroyWindow(hwnd_);             // close overlay window
                    hwnd_ = nullptr;
                    return 0;
                }
                return 0;
            }
            return 0;
        }

        case WM_RBUTTONDOWN:
        {
            // Right-click means "not ok, reselect" (do NOT exit).
            ClearSelection();
            return 0;
        }

        case WM_LBUTTONDOWN:
        {
            dragging_ = true;

            GetCursorPos(&anchor_screen_);
            cur_screen_ = anchor_screen_;

            sel_virtual_ = { anchor_screen_.x, anchor_screen_.y, cur_screen_.x, cur_screen_.y };

            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }

        case WM_MOUSEMOVE:
        {
            if (dragging_ && (wParam & MK_LBUTTON))
            {
                UpdateSelectionFromCursor();
                InvalidateRect(hwnd, nullptr, FALSE);
            }
            return 0;
        }

        case WM_LBUTTONUP:
        {
            if (dragging_)
            {
                dragging_ = false;
                UpdateSelectionFromCursor();

                // Mark we have a selection, but do NOT finish.
                has_selection_ = HasValidSelection();

                InvalidateRect(hwnd, nullptr, FALSE);
            }
            return 0;
        }

        case WM_PAINT:
        {
            PAINTSTRUCT ps{};
            HDC hdc = BeginPaint(hwnd, &ps);

            RECT client{};
            GetClientRect(hwnd, &client);
            FillRect(hdc, &client, (HBRUSH)GetStockObject(BLACK_BRUSH));

            RECT v = sel_virtual_;
            NormalizeRect(v);

            if ((v.right > v.left) && (v.bottom > v.top))
            {
                POINT tl = ScreenToOverlayClient(POINT{ v.left, v.top });
                POINT br = ScreenToOverlayClient(POINT{ v.right, v.bottom });

                RECT r_client{ tl.x, tl.y, br.x, br.y };
                NormalizeRect(r_client);

                HPEN pen = CreatePen(PS_SOLID, 2, RGB(255, 0, 0));
                HGDIOBJ oldPen = SelectObject(hdc, pen);
                HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(HOLLOW_BRUSH));

                Rectangle(hdc, r_client.left, r_client.top, r_client.right, r_client.bottom);

                SelectObject(hdc, oldBrush);
                SelectObject(hdc, oldPen);
                DeleteObject(pen);
            }

            EndPaint(hwnd, &ps);
            return 0;
        }
        }

        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
public:
    // Capture a screen region (virtual screen coordinates) into BgraFrame (BGRA 32bpp).
    static bool CaptureRegionToBgraFrame(const RECT& vr, BgraFrame& out) {
        out = {};

        RECT r = vr;
        if (r.right <= r.left || r.bottom <= r.top) return false;

        const int w = r.right - r.left;
        const int h = r.bottom - r.top;

        HDC hdcScreen = GetDC(nullptr);
        if (!hdcScreen) return false;

        HDC hdcMem = CreateCompatibleDC(hdcScreen);
        if (!hdcMem) {
            ReleaseDC(nullptr, hdcScreen);
            return false;
        }

        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = w;
        bmi.bmiHeader.biHeight = -h;              // top-down
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;            // BGRA
        bmi.bmiHeader.biCompression = BI_RGB;

        void* dibBits = nullptr;
        HBITMAP hbm = CreateDIBSection(hdcScreen, &bmi, DIB_RGB_COLORS, &dibBits, nullptr, 0);
        if (!hbm || !dibBits) {
            if (hbm) DeleteObject(hbm);
            DeleteDC(hdcMem);
            ReleaseDC(nullptr, hdcScreen);
            return false;
        }

        HGDIOBJ oldObj = SelectObject(hdcMem, hbm);

        // Copy from screen into DIB
        // Note: coordinates are in virtual screen space; BitBlt on screen DC accepts those.
        BOOL ok = BitBlt(hdcMem, 0, 0, w, h, hdcScreen, r.left, r.top, SRCCOPY);
        if (!ok) {
            SelectObject(hdcMem, oldObj);
            DeleteObject(hbm);
            DeleteDC(hdcMem);
            ReleaseDC(nullptr, hdcScreen);
            return false;
        }

        // Copy pixels out
        out.width = w;
        out.height = h;
        out.stride = w * 4;
        out.data.resize((size_t)out.stride * (size_t)h);
        memcpy(out.data.data(), dibBits, out.data.size());

        // Cleanup
        SelectObject(hdcMem, oldObj);
        DeleteObject(hbm);
        DeleteDC(hdcMem);
        ReleaseDC(nullptr, hdcScreen);

        return true;
    }
};
