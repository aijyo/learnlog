//// ScreenSnip.cpp
//// A minimal Win32 screen region selector (red rectangle) + capture bitmap.
//// Comments are in English as requested.
//
//#define NOMINMAX
//#include <windows.h>
//#include <windowsx.h>
//#include <cstdint>
//#include <string>
//
//#include <vector>
//#include <shlwapi.h>
//
//// Optional: WIC PNG saving
//#include <wincodec.h>
//#pragma comment(lib, "windowscodecs.lib")
//
//#pragma comment(lib, "shlwapi.lib")
//
//static std::wstring GetExeDir()
//{
//    wchar_t buf[MAX_PATH]{};
//    GetModuleFileNameW(nullptr, buf, MAX_PATH);
//    PathRemoveFileSpecW(buf);
//    return buf;
//}
//
//static std::wstring GetTempDir()
//{
//    wchar_t buf[MAX_PATH]{};
//    GetTempPathW(MAX_PATH, buf);
//    return buf;
//}
//
//static bool EnsureDirEndsWithSlash(std::wstring& s)
//{
//    if (!s.empty() && s.back() != L'\\' && s.back() != L'/') s.push_back(L'\\');
//    return true;
//}
//
//// Save PNG to exe dir first; if fails, save to temp dir.
//static std::wstring PickWritablePngPath()
//{
//    std::wstring dir = GetExeDir();
//    EnsureDirEndsWithSlash(dir);
//    std::wstring p1 = dir + L"snip.png";
//
//    // Quick write test: try create file.
//    HANDLE h = CreateFileW(p1.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
//        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
//    if (h != INVALID_HANDLE_VALUE)
//    {
//        CloseHandle(h);
//        DeleteFileW(p1.c_str());
//        return p1;
//    }
//
//    std::wstring tmp = GetTempDir();
//    EnsureDirEndsWithSlash(tmp);
//    return tmp + L"snip.png";
//}
//
//static void ShowLastErrorBox(const wchar_t* title, const wchar_t* action)
//{
//    DWORD e = GetLastError();
//    wchar_t* msg = nullptr;
//    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
//        FORMAT_MESSAGE_IGNORE_INSERTS,
//        nullptr, e, 0, (LPWSTR)&msg, 0, nullptr);
//
//    std::wstring text = std::wstring(action) + L"\n\nGetLastError=" + std::to_wstring(e) +
//        L"\n" + (msg ? msg : L"");
//    MessageBoxW(nullptr, text.c_str(), title, MB_ICONERROR);
//
//    if (msg) LocalFree(msg);
//}
//class ScreenSnip
//{
//public:
//    // Let user drag to select a screen region. Returns true if a non-empty region was selected.
//    bool SelectRegion(RECT& outRect)
//    {
//        outRect = { 0,0,0,0 };
//
//        EnsureDpiAwareness_();
//
//        HINSTANCE hInst = GetModuleHandleW(nullptr);
//        RegisterWindowClassOnce_(hInst);
//
//        // Create a topmost fullscreen overlay window.
//        const int vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
//        const int vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
//        const int vw = GetSystemMetrics(SM_CXVIRTUALSCREEN);
//        const int vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);
//
//        selecting_ = true;
//        done_ = false;
//        canceled_ = false;
//        hasAnchor_ = false;
//        anchor_ = { 0,0 };
//        current_ = { 0,0 };
//        selected_ = { 0,0,0,0 };
//
//        hwnd_ = CreateWindowExW(
//            WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_LAYERED,
//            kClassName_,
//            L"ScreenSnipOverlay",
//            WS_POPUP,
//            vx, vy, vw, vh,
//            nullptr, nullptr, hInst, this);
//
//        if (!hwnd_) return false;
//
//        // Semi-transparent dark overlay.
//        SetLayeredWindowAttributes(hwnd_, 0, (BYTE)80, LWA_ALPHA);
//
//        ShowWindow(hwnd_, SW_SHOW);
//        UpdateWindow(hwnd_);
//
//        // Capture mouse so we keep receiving events.
//        SetCapture(hwnd_);
//
//        // Run a local message loop until selection completes or canceled.
//        MSG msg;
//        while (!done_ && GetMessageW(&msg, nullptr, 0, 0))
//        {
//            TranslateMessage(&msg);
//            DispatchMessageW(&msg);
//        }
//
//        if (hwnd_)
//        {
//            ReleaseCapture();
//            DestroyWindow(hwnd_);
//            hwnd_ = nullptr;
//        }
//
//        if (canceled_) return false;
//
//        NormalizeRect_(selected_);
//        if ((selected_.right - selected_.left) <= 0 || (selected_.bottom - selected_.top) <= 0)
//            return false;
//
//        outRect = selected_;
//        return true;
//    }
//
//    // Capture a region from the screen into an HBITMAP. Caller owns the returned bitmap (DeleteObject).
//    static HBITMAP CaptureRegionBitmap(const RECT& rc)
//    {
//        RECT r = rc;
//        NormalizeRect_(r);
//
//        const int w = r.right - r.left;
//        const int h = r.bottom - r.top;
//        if (w <= 0 || h <= 0) return nullptr;
//
//        HDC hdcScreen = GetDC(nullptr);
//        if (!hdcScreen) return nullptr;
//
//        HDC hdcMem = CreateCompatibleDC(hdcScreen);
//        if (!hdcMem)
//        {
//            ReleaseDC(nullptr, hdcScreen);
//            return nullptr;
//        }
//
//        HBITMAP hbmp = CreateCompatibleBitmap(hdcScreen, w, h);
//        if (!hbmp)
//        {
//            DeleteDC(hdcMem);
//            ReleaseDC(nullptr, hdcScreen);
//            return nullptr;
//        }
//
//        HGDIOBJ old = SelectObject(hdcMem, hbmp);
//
//        // Copy pixels from screen.
//        BitBlt(hdcMem, 0, 0, w, h, hdcScreen, r.left, r.top, SRCCOPY | CAPTUREBLT);
//
//        SelectObject(hdcMem, old);
//        DeleteDC(hdcMem);
//        ReleaseDC(nullptr, hdcScreen);
//
//        return hbmp;
//    }
//
//    // Optional: Save HBITMAP to PNG using WIC. Returns true if saved.
//    static bool SaveHBitmapToPng(const std::wstring& path, HBITMAP hbmp)
//    {
//        if (!hbmp) return false;
//
//        IWICImagingFactory* factory = nullptr;
//        IWICBitmapEncoder* encoder = nullptr;
//        IWICBitmapFrameEncode* frame = nullptr;
//        IWICStream* stream = nullptr;
//        IWICBitmap* wicBitmap = nullptr;
//        UINT width = 0, height = 0;
//
//        HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
//        const bool didInit = SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE;
//
//        hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
//                              IID_PPV_ARGS(&factory));
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = factory->CreateStream(&stream);
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE);
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder);
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = encoder->Initialize(stream, WICBitmapEncoderNoCache);
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = encoder->CreateNewFrame(&frame, nullptr);
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = frame->Initialize(nullptr);
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = factory->CreateBitmapFromHBITMAP(hbmp, nullptr, WICBitmapUseAlpha, &wicBitmap);
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = wicBitmap->GetSize(&width, &height);
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = frame->SetSize(width, height);
//        if (FAILED(hr)) goto Cleanup;
//
//        // Force a common pixel format
//        WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
//        hr = frame->SetPixelFormat(&fmt);
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = frame->WriteSource(wicBitmap, nullptr);
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = frame->Commit();
//        if (FAILED(hr)) goto Cleanup;
//
//        hr = encoder->Commit();
//        if (FAILED(hr)) goto Cleanup;
//
//    Cleanup:
//        if (frame) frame->Release();
//        if (encoder) encoder->Release();
//        if (stream) stream->Release();
//        if (wicBitmap) wicBitmap->Release();
//        if (factory) factory->Release();
//
//        // Only call CoUninitialize if we initialized in this function.
//        if (didInit && hr != RPC_E_CHANGED_MODE) CoUninitialize();
//
//        return SUCCEEDED(hr);
//    }
//
//private:
//    static inline const wchar_t* kClassName_ = L"ScreenSnipOverlayClass";
//
//    HWND hwnd_ = nullptr;
//    bool selecting_ = false;
//    bool done_ = false;
//    bool canceled_ = false;
//    bool hasAnchor_ = false;
//    POINT anchor_{};
//    POINT current_{};
//    RECT selected_{};
//
//    static void NormalizeRect_(RECT& r)
//    {
//        if (r.left > r.right) std::swap(r.left, r.right);
//        if (r.top > r.bottom) std::swap(r.top, r.bottom);
//    }
//
//    void EnsureDpiAwareness_()
//    {
//        // Best-effort: enable per-monitor DPI awareness on newer Windows.
//        // This avoids coordinate mismatch on high-DPI.
//        HMODULE user32 = GetModuleHandleW(L"user32.dll");
//        if (!user32) return;
//
//        using SetDpiAwarenessContextFn = DPI_AWARENESS_CONTEXT(WINAPI*)(DPI_AWARENESS_CONTEXT);
//        auto fn = (SetDpiAwarenessContextFn)GetProcAddress(user32, "SetProcessDpiAwarenessContext");
//        if (fn)
//        {
//            fn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
//        }
//        else
//        {
//            // Fallback for older Windows.
//            SetProcessDPIAware();
//        }
//    }
//
//    void RegisterWindowClassOnce_(HINSTANCE hInst)
//    {
//        static bool registered = false;
//        if (registered) return;
//
//        WNDCLASSEXW wc{};
//        wc.cbSize = sizeof(wc);
//        wc.hInstance = hInst;
//        wc.lpszClassName = kClassName_;
//        wc.lpfnWndProc = &ScreenSnip::WndProcThunk_;
//        wc.hCursor = LoadCursorW(nullptr, IDC_CROSS);
//        wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
//        wc.style = CS_HREDRAW | CS_VREDRAW;
//        RegisterClassExW(&wc);
//
//        registered = true;
//    }
//
//    static LRESULT CALLBACK WndProcThunk_(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
//    {
//        ScreenSnip* self = nullptr;
//
//        if (msg == WM_NCCREATE)
//        {
//            auto cs = (CREATESTRUCTW*)lParam;
//            self = (ScreenSnip*)cs->lpCreateParams;
//            SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)self);
//            self->hwnd_ = hwnd;
//        }
//        else
//        {
//            self = (ScreenSnip*)GetWindowLongPtrW(hwnd, GWLP_USERDATA);
//        }
//
//        if (!self) return DefWindowProcW(hwnd, msg, wParam, lParam);
//        return self->WndProc_(hwnd, msg, wParam, lParam);
//    }
//
//    LRESULT WndProc_(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
//    {
//        switch (msg)
//        {
//        case WM_KEYDOWN:
//            if (wParam == VK_ESCAPE)
//            {
//                canceled_ = true;
//                done_ = true;
//                PostQuitMessage(0);
//            }
//            return 0;
//
//        case WM_LBUTTONDOWN:
//        {
//            hasAnchor_ = true;
//            anchor_.x = GET_X_LPARAM(lParam);
//            anchor_.y = GET_Y_LPARAM(lParam);
//            current_ = anchor_;
//
//            selected_ = { anchor_.x, anchor_.y, current_.x, current_.y };
//            InvalidateRect(hwnd, nullptr, FALSE);
//            return 0;
//        }
//        case WM_MOUSEMOVE:
//        {
//            if (hasAnchor_ && (wParam & MK_LBUTTON))
//            {
//                current_.x = GET_X_LPARAM(lParam);
//                current_.y = GET_Y_LPARAM(lParam);
//                selected_ = { anchor_.x, anchor_.y, current_.x, current_.y };
//                InvalidateRect(hwnd, nullptr, FALSE);
//            }
//            return 0;
//        }
//        case WM_LBUTTONUP:
//        {
//            if (hasAnchor_)
//            {
//                current_.x = GET_X_LPARAM(lParam);
//                current_.y = GET_Y_LPARAM(lParam);
//                selected_ = { anchor_.x, anchor_.y, current_.x, current_.y };
//
//                done_ = true;
//                PostQuitMessage(0);
//            }
//            return 0;
//        }
//        case WM_RBUTTONDOWN:
//            // Right click cancels
//            canceled_ = true;
//            done_ = true;
//            PostQuitMessage(0);
//            return 0;
//
//        case WM_PAINT:
//        {
//            PAINTSTRUCT ps{};
//            HDC hdc = BeginPaint(hwnd, &ps);
//
//            // Paint a dark background (window is already layered for alpha).
//            RECT client{};
//            GetClientRect(hwnd, &client);
//            FillRect(hdc, &client, (HBRUSH)GetStockObject(BLACK_BRUSH));
//
//            // Draw the selection rectangle in red.
//            if (hasAnchor_)
//            {
//                RECT r = selected_;
//                NormalizeRect_(r);
//
//                HPEN pen = CreatePen(PS_SOLID, 2, RGB(255, 0, 0));
//                HGDIOBJ oldPen = SelectObject(hdc, pen);
//                HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(HOLLOW_BRUSH));
//
//                Rectangle(hdc, r.left, r.top, r.right, r.bottom);
//
//                SelectObject(hdc, oldBrush);
//                SelectObject(hdc, oldPen);
//                DeleteObject(pen);
//            }
//
//            EndPaint(hwnd, &ps);
//            return 0;
//        }
//
//        case WM_DESTROY:
//            return 0;
//        }
//
//        return DefWindowProcW(hwnd, msg, wParam, lParam);
//    }
//};
//
//// ---------------------- Example usage ----------------------
//// Build as a Windows subsystem app, or change to console as you like.
////int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int)
//int main()
//{
//    ScreenSnip snip;
//
//    RECT rc{};
//    if (!snip.SelectRegion(rc))
//        return 0;
//
//    HBITMAP hbmp = ScreenSnip::CaptureRegionBitmap(rc);
//    if (!hbmp)
//        return 0;
//
//    // Example: save to PNG (optional)
//    //// Make sure the folder exists.
//    //ScreenSnip::SaveHBitmapToPng(L".\\snip.png", hbmp);
//    std::wstring outPath = PickWritablePngPath();
//    if (!ScreenSnip::SaveHBitmapToPng(outPath, hbmp))
//    {
//        ShowLastErrorBox(L"Save PNG failed", L"Can't open file for write or encode PNG.");
//    }
//    else
//    {
//        MessageBoxW(nullptr, (L"Saved: " + outPath).c_str(), L"OK", MB_OK);
//    }
//
//    DeleteObject(hbmp);
//    return 0;
//}
