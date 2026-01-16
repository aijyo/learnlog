// wgc_roi_capture.cpp
// Minimal WGC ROI capture + region selection overlay (red rectangle).
// Comments are in English as requested.

#define NOMINMAX
#include <windows.h>
#include <windowsx.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <shlwapi.h>
#include <wrl/client.h>

#include <cmath>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")
#pragma comment(lib, "Shlwapi.lib")

#include <unknwn.h>
#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <Windows.Graphics.DirectX.Direct3D11.interop.h>


using Microsoft::WRL::ComPtr;

#include "utils.h"
#include "bgra_consumer_thread.h"
#include "bgra_png_saver.h"
// -----------------------------
// IGraphicsCaptureItemInterop
// -----------------------------
struct __declspec(uuid("3628e81b-3cac-4c60-b7f4-23ce0e0c3356"))
    IGraphicsCaptureItemInterop : IUnknown
{
    virtual HRESULT __stdcall CreateForWindow(HWND window, REFIID riid, void** result) = 0;
    virtual HRESULT __stdcall CreateForMonitor(HMONITOR monitor, REFIID riid, void** result) = 0;
};

// -----------------------------
// Utility: DPI awareness
// -----------------------------
static void EnablePerMonitorDpiAwareness()
{
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    if (!user32) return;

    using Fn = DPI_AWARENESS_CONTEXT(WINAPI*)(DPI_AWARENESS_CONTEXT);
    auto fn = (Fn)GetProcAddress(user32, "SetProcessDpiAwarenessContext");
    if (fn)
    {
        fn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
    else
    {
        SetProcessDPIAware();
    }
}

// -----------------------------
// Utility: get HMONITOR by point / rect
// -----------------------------
static HMONITOR GetMonitorFromPoint(POINT pt)
{
    return MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST);
}

static HMONITOR GetMonitorFromRectCenter(const RECT& rcVirtual)
{
    POINT c{};
    c.x = (rcVirtual.left + rcVirtual.right) / 2;
    c.y = (rcVirtual.top + rcVirtual.bottom) / 2;
    return GetMonitorFromPoint(c);
}

static bool GetMonitorRect(HMONITOR mon, RECT& outRcMonitorVirtual)
{
    MONITORINFO mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(mon, &mi)) return false;
    outRcMonitorVirtual = mi.rcMonitor; // virtual screen coordinates
    return true;
}

static inline void NormalizeRect(RECT& r)
{
    if (r.left > r.right) std::swap(r.left, r.right);
    if (r.top > r.bottom) std::swap(r.top, r.bottom);
}

// Convert virtual-screen ROI to monitor-local ROI, clipping to that monitor.
// Returns false if ROI doesn't intersect with the monitor.
static bool VirtualToMonitorLocalRect(HMONITOR mon, const RECT& rcVirtual, RECT& outLocal)
{
    MONITORINFO mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(mon, &mi))
        return false;

    RECT v = rcVirtual;
    NormalizeRect(v);

    RECT clipped{};
    if (!IntersectRect(&clipped, &v, &mi.rcMonitor))
        return false;

    outLocal.left = clipped.left - mi.rcMonitor.left;
    outLocal.top = clipped.top - mi.rcMonitor.top;
    outLocal.right = clipped.right - mi.rcMonitor.left;
    outLocal.bottom = clipped.bottom - mi.rcMonitor.top;

    NormalizeRect(outLocal);
    return (outLocal.right > outLocal.left) && (outLocal.bottom > outLocal.top);
}

// Clamp local ROI into [0..W/H]
static void ClampRoi(RECT& r, int w, int h)
{
    r.left = std::max<LONG>(0, std::min<LONG>(r.left, w));
    r.right = std::max<LONG>(0, std::min<LONG>(r.right, w));
    r.top = std::max<LONG>(0, std::min<LONG>(r.top, h));
    r.bottom = std::max<LONG>(0, std::min<LONG>(r.bottom, h));
    NormalizeRect(r);
}

// -----------------------------
// ScreenRegionSelector: overlay window with red rectangle
// Returns RECT in virtual screen coordinates.
// Adds: max selection size + right-click reset
// -----------------------------
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
        SetCursor(LoadCursorW(nullptr, IDC_CROSS));

        SetCapture(hwnd_);

        MSG msg;
        while (!done_ && GetMessageW(&msg, nullptr, 0, 0))
        {
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
        wc.hCursor = LoadCursorW(nullptr, IDC_CROSS);
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
            SetCursor(LoadCursorW(nullptr, IDC_CROSS));
            return TRUE;

        case WM_KEYDOWN:
        {
            if (wParam == VK_ESCAPE)
            {
                canceled_ = true;
                done_ = true;
                PostQuitMessage(0);
                return 0;
            }
            if (wParam == VK_RETURN)
            {
                // Confirm only when we have a valid selection.
                if (has_selection_ && HasValidSelection())
                {
                    done_ = true;
                    PostQuitMessage(0);
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
};

// -----------------------------
// WGC helpers: create IDirect3DDevice from ID3D11Device
// -----------------------------
static winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice
CreateDirect3DDeviceFromD3D11(ID3D11Device* d3dDevice)
{
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
    winrt::check_hresult(d3dDevice->QueryInterface(IID_PPV_ARGS(dxgiDevice.GetAddressOf())));

    winrt::com_ptr<IInspectable> inspectable;
    winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.Get(), inspectable.put()));

    return inspectable.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
}

static Microsoft::WRL::ComPtr<ID3D11Texture2D>
GetTextureFromSurface(winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface const& surface)
{
    Microsoft::WRL::ComPtr<ID3D11Texture2D> tex;

    // Get raw IUnknown from WinRT object
    IUnknown* unk = winrt::get_unknown(surface);

    // QI to IDirect3DDxgiInterfaceAccess (defined in windows.graphics.directx.direct3d11.interop.h)
    Microsoft::WRL::ComPtr<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess> access;
    winrt::check_hresult(unk->QueryInterface(IID_PPV_ARGS(access.GetAddressOf())));

    // Finally get the DX interface (ID3D11Texture2D)
    winrt::check_hresult(access->GetInterface(__uuidof(ID3D11Texture2D), (void**)tex.GetAddressOf()));
    return tex;
}

// -----------------------------
// Create GraphicsCaptureItem for monitor
// -----------------------------
static winrt::Windows::Graphics::Capture::GraphicsCaptureItem
CreateItemForMonitor(HMONITOR mon)
{
    // Must have an apartment initialized for WinRT.
    // Using MTA is typical for capture worker logic.
    winrt::init_apartment(winrt::apartment_type::multi_threaded);

    auto factory = winrt::get_activation_factory<
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem,
        IGraphicsCaptureItemInterop>();

    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
    winrt::check_hresult(factory->CreateForMonitor(mon, winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(), winrt::put_abi(item)));
    return item;
}

// -----------------------------
// WgcRoiCapturer: capture a monitor, output CPU BGRA ROI frames
// -----------------------------
class WgcRoiCapturer
{
public:
    WgcRoiCapturer(BgraFrameConsumerThread::FrameCallback cb = nullptr)
        : consumer_(cb)
    {
    }

    bool StartMonitorRoi(HMONITOR mon, RECT roiMonitorLocal, int fps)
    {
        Stop();

        if (!mon) return false;
        targetFps_ = (fps <= 0) ? 60 : fps;

        // Create capture item
        item_ = CreateItemForMonitor(mon);
        if (!item_) return false;

        // Create D3D11 device
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
        D3D_FEATURE_LEVEL fl{};
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            flags, levels, _countof(levels),
            D3D11_SDK_VERSION,
            d3dDevice_.GetAddressOf(),
            &fl,
            d3dCtx_.GetAddressOf());
        if (FAILED(hr)) return false;

        d3dWinrtDevice_ = CreateDirect3DDeviceFromD3D11(d3dDevice_.Get());

        // Set ROI (monitor-local coords)
        NormalizeRect(roiMonitorLocal);
        roi_ = roiMonitorLocal;
        roiDirty_ = true;

        // Create frame pool + session
        auto size = item_.Size(); // size of the capture item in pixels
        itemW_ = size.Width;
        itemH_ = size.Height;

        ClampRoi(roi_, itemW_, itemH_);

        // Must be MTA for free-threaded capture in console apps.
        winrt::init_apartment(winrt::apartment_type::multi_threaded);// Create a free-threaded frame pool so FrameArrived doesn't require a UI dispatcher.
        framePool_ = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
            d3dWinrtDevice_,                        // IDirect3DDevice
            winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,                              // frame count
            size               // capture item size
        );


        //framePool_ = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::Create(
        //    d3dWinrtDevice_,
        //    winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
        //    2,
        //    size);

        session_ = framePool_.CreateCaptureSession(item_);
        // Cursor capture optional:
        // session_.IsCursorCaptureEnabled(true);

        frameArrivedRevoker_ = framePool_.FrameArrived(
            winrt::auto_revoke,
            [this](auto const& pool, auto const&)
            {
                OnFrameArrived(pool);
            });

        session_.StartCapture();
        running_.store(true);

        consumer_.Start();
        return true;
    }

    void Stop()
    {
        consumer_.Stop();
        running_.store(false);

        if (frameArrivedRevoker_) frameArrivedRevoker_.revoke();
        if (session_) session_.Close();
        if (framePool_) framePool_.Close();

        session_ = nullptr;
        framePool_ = nullptr;
        item_ = nullptr;
        d3dWinrtDevice_ = nullptr;

        roiGpuTex_.Reset();
        roiStageTex_.Reset();
        d3dCtx_.Reset();
        d3dDevice_.Reset();

        {
            std::lock_guard<std::mutex> lk(mtx_);
            latest_ = {};
            frameId_ = 0;
        }
    }

    // Get the latest frame (BGRA). Copies out (thread-safe).
    bool TryGetLatest(BgraFrame& out)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (latest_.data.empty()) return false;
        out = latest_;
        return true;
    }

private:
    void OnFrameArrived(winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& pool)
    {
        if (!running_.load()) return;

        auto frame = pool.TryGetNextFrame();
        if (!frame) return;

        auto srcTex = GetTextureFromSurface(frame.Surface());
        if (!srcTex) return;

        D3D11_TEXTURE2D_DESC srcDesc{};
        srcTex->GetDesc(&srcDesc);

        RECT roiLocal{};
        {
            std::lock_guard<std::mutex> lk(mtxRoi_);
            roiLocal = roi_;
        }
        ClampRoi(roiLocal, (int)srcDesc.Width, (int)srcDesc.Height);

        const int roiW = roiLocal.right - roiLocal.left;
        const int roiH = roiLocal.bottom - roiLocal.top;
        if (roiW <= 0 || roiH <= 0) return;

        if (NeedRecreateRoiTextures_(roiW, roiH, srcDesc.Format))
        {
            CreateRoiTextures_(roiW, roiH, srcDesc.Format);
        }

        // GPU copy ROI
        D3D11_BOX box{};
        box.left = (UINT)roiLocal.left;
        box.top = (UINT)roiLocal.top;
        box.front = 0;
        box.right = (UINT)roiLocal.right;
        box.bottom = (UINT)roiLocal.bottom;
        box.back = 1;

        d3dCtx_->CopySubresourceRegion(roiGpuTex_.Get(), 0, 0, 0, 0, srcTex.Get(), 0, &box);

        // GPU -> staging
        d3dCtx_->CopyResource(roiStageTex_.Get(), roiGpuTex_.Get());

        // Map to CPU
        D3D11_MAPPED_SUBRESOURCE mapped{};
        HRESULT hr = d3dCtx_->Map(roiStageTex_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) return;

        //BgraFrame out;
        //out.width = roiW;
        //out.height = roiH;
        //out.stride = (int)mapped.RowPitch;
        //out.data.resize((size_t)mapped.RowPitch * roiH);
        //memcpy(out.data.data(), mapped.pData, out.data.size());
        //out.frame_id = ++frameId_;
        BgraFrame out;
        out.width = roiW;
        out.height = roiH;

        // Make output tightly packed: stride = width * 4
        out.stride = roiW * 4;
        out.data.resize((size_t)out.stride * roiH);

        const uint8_t* src = reinterpret_cast<const uint8_t*>(mapped.pData);
        uint8_t* dst = out.data.data();

        for (int y = 0; y < roiH; ++y)
        {
            // Copy only valid pixels, drop RowPitch padding
            memcpy(dst + (size_t)y * out.stride,
                src + (size_t)y * mapped.RowPitch,
                (size_t)out.stride);
        }

        out.frame_id = ++frameId_;

        d3dCtx_->Unmap(roiStageTex_.Get(), 0);

        consumer_.Submit(std::move(out));
    }

    bool NeedRecreateRoiTextures_(int roiW, int roiH, DXGI_FORMAT fmt)
    {
        if (!roiGpuTex_ || !roiStageTex_) return true;
        if (roiW != roiW_ || roiH != roiH_) return true;
        if (fmt != roiFmt_) return true;
        return false;
    }

    void CreateRoiTextures_(int roiW, int roiH, DXGI_FORMAT fmt)
    {
        roiGpuTex_.Reset();
        roiStageTex_.Reset();

        roiW_ = roiW;
        roiH_ = roiH;
        roiFmt_ = fmt;

        D3D11_TEXTURE2D_DESC td{};
        td.Width = (UINT)roiW;
        td.Height = (UINT)roiH;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = fmt;              // typically B8G8R8A8_UNORM
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = 0;
        td.CPUAccessFlags = 0;

        winrt::check_hresult(d3dDevice_->CreateTexture2D(&td, nullptr, roiGpuTex_.GetAddressOf()));

        td.Usage = D3D11_USAGE_STAGING;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

        winrt::check_hresult(d3dDevice_->CreateTexture2D(&td, nullptr, roiStageTex_.GetAddressOf()));
    }

private:
    std::atomic<bool> running_{ false };
    int targetFps_ = 60;

    // Capture item size
    int itemW_ = 0;
    int itemH_ = 0;

    // D3D
    ComPtr<ID3D11Device> d3dDevice_;
    ComPtr<ID3D11DeviceContext> d3dCtx_;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice d3dWinrtDevice_{ nullptr };

    // WGC
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item_{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool framePool_{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession session_{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::FrameArrived_revoker frameArrivedRevoker_{};

    // ROI (monitor-local)
    std::mutex mtxRoi_;
    RECT roi_{ 0,0,0,0 };
    bool roiDirty_ = false;

    // ROI textures
    int roiW_ = 0;
    int roiH_ = 0;
    DXGI_FORMAT roiFmt_ = DXGI_FORMAT_UNKNOWN;
    ComPtr<ID3D11Texture2D> roiGpuTex_;
    ComPtr<ID3D11Texture2D> roiStageTex_;

    // Latest frame
    std::mutex mtx_;
    BgraFrame latest_{};
    uint64_t frameId_ = 0;
    BgraFrameConsumerThread consumer_;
};

#include <string>
#include <sstream>
#include <iostream>
static BOOL CALLBACK EnumMonProc(HMONITOR hMon, HDC, LPRECT, LPARAM)
{
    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(hMon, &mi))
        return TRUE;

    auto& r = mi.rcMonitor;

    std::wstringstream ss;
    ss << L"[MON] " << mi.szDevice
        << L" rcMonitor=(" << r.left << L"," << r.top << L")-(" << r.right << L"," << r.bottom << L")"
        << L" size=" << (r.right - r.left) << L"x" << (r.bottom - r.top)
        << L" primary=" << ((mi.dwFlags & MONITORINFOF_PRIMARY) ? L"YES" : L"NO")
        << L"\n";

    std::wcout << ss.str();
    return TRUE;
}

static void DumpAllMonitors()
{
    // Enumerate all monitors and print virtual-screen rectangles
    EnumDisplayMonitors(nullptr, nullptr, EnumMonProc, 0);
}
// -----------------------------
// Demo main: select region -> choose monitor -> start capture -> pull frames at fps
// -----------------------------
//int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int)
int main()
{
    //DumpAllMonitors(); 
    //SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    // 1) Select region in virtual-screen coordinates
    ScreenRegionSelector selector(64,64);
    RECT rcVirtual{};
    if (!selector.SelectRegionVirtual(rcVirtual))
    {
        MessageBoxW(nullptr, L"Selection canceled or invalid.", L"Info", MB_OK);
        return 0;
    }

    // 2) Pick monitor by selection center
    HMONITOR mon = GetMonitorFromRectCenter(rcVirtual);
    if (!mon)
    {
        MessageBoxW(nullptr, L"Failed to get monitor.", L"Error", MB_OK | MB_ICONERROR);
        return 0;
    }

    // 3) Convert virtual ROI -> monitor local ROI
    RECT roiLocal{};
    auto bSuc = VirtualToMonitorLocalRect(mon, rcVirtual, roiLocal);

    // 4) Start WGC capture at target FPS
    const int fps = 60; // Change to 30/60 as needed
    BgraFramePngSaver saver;

    WgcRoiCapturer cap(
        [&](BgraFrame&& f)
        {
            // f.data contains BGRA bytes (stride * height)
            // Do your processing here (e.g., icon recognition)
            // Keep it fast; if heavy, offload again or use a bounded queue design.
            printf("Consume frame_id=%llu width=%d height=%d stride=%d\n", f.frame_id, f.width,f.height, f.stride);
            //saver.Save(f);
        });
    if (!cap.StartMonitorRoi(mon, roiLocal, fps))
    {
        MessageBoxW(nullptr, L"Failed to start WGC capture.", L"Error", MB_OK | MB_ICONERROR);
        return 0;
    }

    // 5) Pull latest frames in a loop (low latency: always take newest)
    //    Press ESC in console-less app: you can close window or just run for N seconds here.
    const int secondsToRun = 5;
    const int totalTicks = secondsToRun * fps;

    auto tickDur = std::chrono::microseconds((int)(1000000.0 / fps));

    for (int i = 0; i < totalTicks; ++i)
    {
        auto t0 = std::chrono::steady_clock::now();

        //BgraFrame fr;
        //if (cap.TryGetLatest(fr))
        //{
        //    // Print minimal info (debug output)
        //    wchar_t buf[256];
        //    swprintf_s(buf, L"Frame %llu  %dx%d  stride=%d\n",
        //        (unsigned long long)fr.frame_id, fr.width, fr.height, fr.stride);
        //    OutputDebugStringW(buf);

        //    // fr.data is BGRA bytes. You can feed to OpenCV / model here.
        //}

        // Throttle loop to desired fps
        auto t1 = std::chrono::steady_clock::now();
        auto used = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        if (used < tickDur)
        {
            std::this_thread::sleep_for(tickDur - used);
        }
    }

    cap.Stop();
    MessageBoxW(nullptr, L"Capture finished (see Output window).", L"Done", MB_OK);
    return 0;
}
