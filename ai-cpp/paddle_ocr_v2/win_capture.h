#pragma once
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>
#include <cstdint>
#include <functional>
#include <mutex>
#include <atomic>
#include <vector>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

#include "./utils/utils.h" // BgraFrame

// -----------------------------
// ScreenRegionSelector: overlay window with red rectangle
// Returns RECT in virtual screen coordinates.
// Adds: max selection size + right-click reset
// -----------------------------

class WgcRoiCapturer
{
public:
    using FrameCallback = std::function<void(const BgraFrame&)>;

    explicit WgcRoiCapturer(FrameCallback cb = nullptr);
    ~WgcRoiCapturer();

    WgcRoiCapturer(const WgcRoiCapturer&) = delete;
    WgcRoiCapturer& operator=(const WgcRoiCapturer&) = delete;

    bool StartMonitorRoi(HMONITOR mon, RECT roiMonitorLocal, int fps = 60);
    void Stop();

    void SetRoi(RECT roiMonitorLocal);
    bool TryGetLatest(BgraFrame& out);

    bool IsRunning() const { return running_.load(); }

private:
    void OnFrameArrived(winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& pool);

    void EnsureRoiTextures_(int roiW, int roiH, DXGI_FORMAT fmt);
    void ReleaseRoiTextures_();

private:
    std::atomic<bool> running_{ false };
    int targetFps_ = 60;

    // D3D11
    Microsoft::WRL::ComPtr<ID3D11Device> d3dDevice_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> d3dCtx_;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice d3dWinrtDevice_{ nullptr };

    // WGC
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item_{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool framePool_{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession session_{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::FrameArrived_revoker frameArrivedRevoker_{};

    // ROI (monitor-local coords)
    std::mutex mtxRoi_;
    RECT roi_{ 0,0,0,0 };

    // ROI textures
    int roiW_ = 0;
    int roiH_ = 0;
    DXGI_FORMAT roiFmt_ = DXGI_FORMAT_UNKNOWN;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> roiGpuTex_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> roiStageTex_;

    // Latest
    std::mutex mtxLatest_;
    BgraFrame latest_{};
    uint64_t frameId_ = 0;

    FrameCallback cb_;
};
