// win_capture.cpp
// WGC monitor ROI capturer -> CPU BGRA frames.
// Comments are in English as requested.

#define NOMINMAX
#include "win_capture.h"

#include <unknwn.h>

// IMPORTANT: do NOT manually define IGraphicsCaptureItemInterop.
// Use the official interop headers instead.
#include <Windows.Graphics.Capture.Interop.h>
#include <Windows.Graphics.DirectX.Direct3D11.interop.h> // IDirect3DDxgiInterfaceAccess + CreateDirect3D11DeviceFromDXGIDevice

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

using Microsoft::WRL::ComPtr;

static inline void NormalizeRect_(RECT& r)
{
    if (r.left > r.right) std::swap(r.left, r.right);
    if (r.top > r.bottom) std::swap(r.top, r.bottom);
}

static inline void ClampRect_(RECT& r, int w, int h)
{
    r.left = std::max<LONG>(0, std::min<LONG>(r.left, w));
    r.right = std::max<LONG>(0, std::min<LONG>(r.right, w));
    r.top = std::max<LONG>(0, std::min<LONG>(r.top, h));
    r.bottom = std::max<LONG>(0, std::min<LONG>(r.bottom, h));
    NormalizeRect_(r);
}

static winrt::Windows::Graphics::Capture::GraphicsCaptureItem CreateItemForMonitor_(HMONITOR mon)
{
    auto factory = winrt::get_activation_factory<
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem,
        IGraphicsCaptureItemInterop>();

    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
    winrt::check_hresult(factory->CreateForMonitor(
        mon,
        winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(),
        winrt::put_abi(item)));

    return item;
}

static winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice CreateWinrtD3DDevice_(ID3D11Device* d3dDevice)
{
    ComPtr<IDXGIDevice> dxgiDevice;
    winrt::check_hresult(d3dDevice->QueryInterface(IID_PPV_ARGS(dxgiDevice.GetAddressOf())));

    winrt::com_ptr<IInspectable> inspectable;
    winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(
        dxgiDevice.Get(),
        inspectable.put()));

    return inspectable.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
}

static ComPtr<ID3D11Texture2D> GetTextureFromSurface_(winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface const& surface)
{
    // IDirect3DDxgiInterfaceAccess is declared in Windows.Graphics.DirectX.Direct3D11.interop.h
    winrt::com_ptr<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess> access = surface.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();

    ComPtr<ID3D11Texture2D> tex;
    winrt::check_hresult(access->GetInterface(__uuidof(ID3D11Texture2D),
        reinterpret_cast<void**>(tex.GetAddressOf())));
    return tex;
}

WgcRoiCapturer::WgcRoiCapturer(FrameCallback cb)
    : cb_(std::move(cb))
{
    // Prefer MTA for capture in worker threads/console apps.
    try { winrt::init_apartment(winrt::apartment_type::multi_threaded); }
    catch (...) {}
}

WgcRoiCapturer::~WgcRoiCapturer()
{
    Stop();
}

bool WgcRoiCapturer::StartMonitorRoi(HMONITOR mon, RECT roiMonitorLocal, int fps)
{
    Stop();
    if (!mon) return false;

    targetFps_ = (fps <= 0) ? 60 : fps;

    item_ = CreateItemForMonitor_(mon);
    if (!item_) return false;

    // Create D3D11 device
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL fl{};

    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        levels,
        _countof(levels),
        D3D11_SDK_VERSION,
        d3dDevice_.GetAddressOf(),
        &fl,
        d3dCtx_.GetAddressOf());

    if (FAILED(hr)) return false;

    d3dWinrtDevice_ = CreateWinrtD3DDevice_(d3dDevice_.Get());

    NormalizeRect_(roiMonitorLocal);
    {
        std::lock_guard<std::mutex> lk(mtxRoi_);
        roi_ = roiMonitorLocal;
    }

    // Create frame pool + session
    auto size = item_.Size();

    framePool_ = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
        d3dWinrtDevice_,
        winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
        2,
        size);

    session_ = framePool_.CreateCaptureSession(item_);

    frameArrivedRevoker_ = framePool_.FrameArrived(
        winrt::auto_revoke,
        [this](auto const& pool, auto const&)
        {
            this->OnFrameArrived(pool);
        });

    running_.store(true);
    session_.StartCapture();
    return true;
}

void WgcRoiCapturer::Stop()
{
    if (!running_.exchange(false))
        return;

    try
    {
        frameArrivedRevoker_.revoke();
        if (session_) session_.Close();
        if (framePool_) framePool_.Close();
    }
    catch (...) {}

    session_ = nullptr;
    framePool_ = nullptr;
    item_ = nullptr;

    ReleaseRoiTextures_();

    d3dCtx_.Reset();
    d3dDevice_.Reset();
    d3dWinrtDevice_ = nullptr;

    {
        std::lock_guard<std::mutex> lk(mtxLatest_);
        latest_ = {};
        frameId_ = 0;
    }
}

void WgcRoiCapturer::SetRoi(RECT roiMonitorLocal)
{
    NormalizeRect_(roiMonitorLocal);
    std::lock_guard<std::mutex> lk(mtxRoi_);
    roi_ = roiMonitorLocal;
}

bool WgcRoiCapturer::TryGetLatest(BgraFrame& out)
{
    std::lock_guard<std::mutex> lk(mtxLatest_);
    if (latest_.data.empty()) return false;
    out = latest_;
    return true;
}

void WgcRoiCapturer::ReleaseRoiTextures_()
{
    roiGpuTex_.Reset();
    roiStageTex_.Reset();
    roiW_ = 0;
    roiH_ = 0;
    roiFmt_ = DXGI_FORMAT_UNKNOWN;
}

void WgcRoiCapturer::EnsureRoiTextures_(int roiW, int roiH, DXGI_FORMAT fmt)
{
    if (roiGpuTex_ && roiStageTex_ && roiW_ == roiW && roiH_ == roiH && roiFmt_ == fmt)
        return;

    ReleaseRoiTextures_();

    roiW_ = roiW;
    roiH_ = roiH;
    roiFmt_ = fmt;

    D3D11_TEXTURE2D_DESC td{};
    td.Width = (UINT)roiW;
    td.Height = (UINT)roiH;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = fmt;
    td.SampleDesc.Count = 1;

    // GPU texture (copy target)
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = 0;
    td.CPUAccessFlags = 0;
    winrt::check_hresult(d3dDevice_->CreateTexture2D(&td, nullptr, roiGpuTex_.GetAddressOf()));

    // Staging texture (CPU readback)
    td.Usage = D3D11_USAGE_STAGING;
    td.BindFlags = 0;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    winrt::check_hresult(d3dDevice_->CreateTexture2D(&td, nullptr, roiStageTex_.GetAddressOf()));
}

void WgcRoiCapturer::OnFrameArrived(winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& pool)
{
    if (!running_.load()) return;

    auto frame = pool.TryGetNextFrame();
    if (!frame) return;

    auto srcTex = GetTextureFromSurface_(frame.Surface());
    if (!srcTex) return;

    D3D11_TEXTURE2D_DESC srcDesc{};
    srcTex->GetDesc(&srcDesc);

    RECT roiLocal{};
    {
        std::lock_guard<std::mutex> lk(mtxRoi_);
        roiLocal = roi_;
    }

    ClampRect_(roiLocal, (int)srcDesc.Width, (int)srcDesc.Height);

    const int roiW = roiLocal.right - roiLocal.left;
    const int roiH = roiLocal.bottom - roiLocal.top;
    if (roiW <= 0 || roiH <= 0) return;

    EnsureRoiTextures_(roiW, roiH, srcDesc.Format);

    // Copy ROI region from src -> roiGpuTex_
    D3D11_BOX box{};
    box.left = (UINT)roiLocal.left;
    box.top = (UINT)roiLocal.top;
    box.front = 0;
    box.right = (UINT)roiLocal.right;
    box.bottom = (UINT)roiLocal.bottom;
    box.back = 1;

    d3dCtx_->CopySubresourceRegion(roiGpuTex_.Get(), 0, 0, 0, 0, srcTex.Get(), 0, &box);
    d3dCtx_->CopyResource(roiStageTex_.Get(), roiGpuTex_.Get());

    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = d3dCtx_->Map(roiStageTex_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) return;

    BgraFrame out;
    out.width = roiW;
    out.height = roiH;
    out.stride = roiW * 4; // tight packing
    out.data.resize((size_t)out.stride * roiH);

    const uint8_t* src = (const uint8_t*)mapped.pData;
    uint8_t* dst = out.data.data();

    for (int y = 0; y < roiH; ++y)
    {
        memcpy(dst + (size_t)y * out.stride,
            src + (size_t)y * mapped.RowPitch,
            (size_t)out.stride);
    }

    d3dCtx_->Unmap(roiStageTex_.Get(), 0);

    out.frame_id = ++frameId_;

    {
        std::lock_guard<std::mutex> lk(mtxLatest_);
        latest_ = out;
    }

    if (cb_) cb_(out);
}
