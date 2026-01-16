//// WgcCapture.h/.cpp merged for brevity.
//// Comments are in English.
//
//#define NOMINMAX
//#include <windows.h>
//#include <d3d11.h>
//#include <dxgi1_2.h>
//#include <wrl/client.h>
//#include <vector>
//#include <atomic>
//#include <mutex>
//
//#include <winrt/base.h>
//#include <winrt/Windows.Foundation.h>
//#include <winrt/Windows.Graphics.Capture.h>
//#include <winrt/Windows.Graphics.DirectX.h>
//#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
//
//using Microsoft::WRL::ComPtr;
//
//static winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice
//CreateDirect3DDeviceFromD3D11(ID3D11Device* d3dDevice)
//{
//    ComPtr<IDXGIDevice> dxgiDevice;
//    d3dDevice->QueryInterface(IID_PPV_ARGS(&dxgiDevice));
//
//    winrt::com_ptr<::IInspectable> inspectable;
//    // IDirect3DDevice interop
//    auto hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.Get(), inspectable.put_void());
//    winrt::check_hresult(hr);
//
//    return inspectable.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
//}
//
//// Helper to get ID3D11Texture2D from a WinRT surface
//static ComPtr<ID3D11Texture2D>
//GetTextureFromSurface(winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface const& surface)
//{
//    ComPtr<ID3D11Texture2D> tex;
//
//    auto access = surface.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
//    winrt::check_hresult(access->GetInterface(IID_PPV_ARGS(&tex)));
//    return tex;
//}
//
//struct BgraFrame
//{
//    int width = 0;
//    int height = 0;
//    int stride = 0;
//    std::vector<uint8_t> data;
//    uint64_t frame_id = 0;
//};
//
//class WgcRoiCapturer
//{
//public:
//    // item: capture target (monitor or window)
//    bool Start(winrt::Windows::Graphics::Capture::GraphicsCaptureItem const& item, int fps)
//    {
//        Stop();
//
//        if (!item) return false;
//
//        winrt::init_apartment(winrt::apartment_type::multi_threaded);
//
//        // Create D3D11 device
//        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
//        D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
//
//        D3D_FEATURE_LEVEL fl{};
//        HRESULT hr = D3D11CreateDevice(
//            nullptr,
//            D3D_DRIVER_TYPE_HARDWARE,
//            nullptr,
//            flags,
//            levels,
//            _countof(levels),
//            D3D11_SDK_VERSION,
//            d3dDevice_.GetAddressOf(),
//            &fl,
//            d3dCtx_.GetAddressOf());
//        if (FAILED(hr)) return false;
//
//        d3dWinrtDevice_ = CreateDirect3DDeviceFromD3D11(d3dDevice_.Get());
//
//        // Create frame pool + session
//        // Use B8G8R8A8 for easy BGRA path
//        auto size = item.Size();
//        framePool_ = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::Create(
//            d3dWinrtDevice_,
//            winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
//            2, // frame pool size
//            size);
//
//        session_ = framePool_.CreateCaptureSession(item);
//        // For lower latency, keep this true if you want borderless capture; optional:
//        // session_.IsCursorCaptureEnabled(true);
//
//        // Frame arrived handler
//        frameArrivedRevoker_ = framePool_.FrameArrived(
//            winrt::auto_revoke,
//            [this](auto const& pool, auto const&)
//            {
//                OnFrameArrived(pool);
//            });
//
//        session_.StartCapture();
//        running_.store(true);
//
//        // You can throttle consumer side based on fps if needed.
//        targetFps_ = fps;
//        return true;
//    }
//
//    void Stop()
//    {
//        running_.store(false);
//
//        if (frameArrivedRevoker_) frameArrivedRevoker_.revoke();
//        if (session_) session_.Close();
//        if (framePool_) framePool_.Close();
//
//        session_ = nullptr;
//        framePool_ = nullptr;
//        d3dWinrtDevice_ = nullptr;
//
//        d3dCtx_.Reset();
//        d3dDevice_.Reset();
//
//        {
//            std::lock_guard<std::mutex> lk(mtx_);
//            latest_ = {};
//            frameId_ = 0;
//        }
//    }
//
//    // Set ROI in *capture texture coordinates* (screen/window pixel space)
//    void SetRoi(RECT roi)
//    {
//        std::lock_guard<std::mutex> lk(mtx_);
//        roi_ = roi;
//        roiDirty_ = true;
//    }
//
//    // Non-blocking fetch of the latest frame (copy out)
//    bool TryGetLatest(BgraFrame& out)
//    {
//        std::lock_guard<std::mutex> lk(mtx_);
//        if (latest_.data.empty()) return false;
//        out = latest_;
//        return true;
//    }
//
//private:
//    void OnFrameArrived(winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& pool)
//    {
//        if (!running_.load()) return;
//
//        auto frame = pool.TryGetNextFrame();
//        if (!frame) return;
//
//        auto surface = frame.Surface();
//        auto srcTex = GetTextureFromSurface(surface);
//        if (!srcTex) return;
//
//        D3D11_TEXTURE2D_DESC srcDesc{};
//        srcTex->GetDesc(&srcDesc);
//
//        RECT roiLocal{};
//        {
//            std::lock_guard<std::mutex> lk(mtx_);
//            roiLocal = roi_;
//            if (!HasValidRoi_(roiLocal))
//            {
//                // Default to full frame if ROI not set
//                roiLocal.left = 0;
//                roiLocal.top = 0;
//                roiLocal.right = (LONG)srcDesc.Width;
//                roiLocal.bottom = (LONG)srcDesc.Height;
//            }
//            ClampRoi_(roiLocal, (int)srcDesc.Width, (int)srcDesc.Height);
//
//            // Recreate textures if ROI size changed
//            if (roiDirty_ || RoiSizeChanged_(roiLocal))
//            {
//                CreateRoiTextures_(
//                    roiLocal.right - roiLocal.left,
//                    roiLocal.bottom - roiLocal.top,
//                    srcDesc.Format);
//                lastRoi_ = roiLocal;
//                roiDirty_ = false;
//            }
//        }
//
//        const int roiW = roiLocal.right - roiLocal.left;
//        const int roiH = roiLocal.bottom - roiLocal.top;
//        if (roiW <= 0 || roiH <= 0) return;
//
//        // 1) GPU copy ROI region into a small default-usage texture
//        D3D11_BOX box{};
//        box.left = (UINT)roiLocal.left;
//        box.top = (UINT)roiLocal.top;
//        box.front = 0;
//        box.right = (UINT)roiLocal.right;
//        box.bottom = (UINT)roiLocal.bottom;
//        box.back = 1;
//
//        d3dCtx_->CopySubresourceRegion(roiGpuTex_.Get(), 0, 0, 0, 0, srcTex.Get(), 0, &box);
//
//        // 2) Copy to staging for CPU read
//        d3dCtx_->CopyResource(roiStageTex_.Get(), roiGpuTex_.Get());
//
//        // 3) Map to CPU
//        D3D11_MAPPED_SUBRESOURCE mapped{};
//        HRESULT hr = d3dCtx_->Map(roiStageTex_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
//        if (FAILED(hr)) return;
//
//        BgraFrame frameOut;
//        frameOut.width = roiW;
//        frameOut.height = roiH;
//        frameOut.stride = (int)mapped.RowPitch;
//        frameOut.data.resize((size_t)mapped.RowPitch * roiH);
//        memcpy(frameOut.data.data(), mapped.pData, frameOut.data.size());
//        frameOut.frame_id = ++frameId_;
//
//        d3dCtx_->Unmap(roiStageTex_.Get(), 0);
//
//        // Publish latest
//        {
//            std::lock_guard<std::mutex> lk(mtx_);
//            latest_ = std::move(frameOut);
//        }
//    }
//
//    bool HasValidRoi_(RECT r)
//    {
//        return (r.right > r.left) && (r.bottom > r.top);
//    }
//
//    void ClampRoi_(RECT& r, int w, int h)
//    {
//        r.left = max(0, min(r.left, w));
//        r.right = max(0, min(r.right, w));
//        r.top = max(0, min(r.top, h));
//        r.bottom = max(0, min(r.bottom, h));
//        if (r.right < r.left) std::swap(r.right, r.left);
//        if (r.bottom < r.top) std::swap(r.bottom, r.top);
//    }
//
//    bool RoiSizeChanged_(RECT r)
//    {
//        return (r.right - r.left) != (lastRoi_.right - lastRoi_.left) ||
//            (r.bottom - r.top) != (lastRoi_.bottom - lastRoi_.top) ||
//            !roiGpuTex_ || !roiStageTex_;
//    }
//
//    void CreateRoiTextures_(int roiW, int roiH, DXGI_FORMAT fmt)
//    {
//        roiGpuTex_.Reset();
//        roiStageTex_.Reset();
//
//        // Default GPU texture (fast CopySubresourceRegion target)
//        D3D11_TEXTURE2D_DESC td{};
//        td.Width = roiW;
//        td.Height = roiH;
//        td.MipLevels = 1;
//        td.ArraySize = 1;
//        td.Format = fmt; // typically DXGI_FORMAT_B8G8R8A8_UNORM
//        td.SampleDesc.Count = 1;
//        td.Usage = D3D11_USAGE_DEFAULT;
//        td.BindFlags = 0;
//        td.CPUAccessFlags = 0;
//
//        winrt::check_hresult(d3dDevice_->CreateTexture2D(&td, nullptr, roiGpuTex_.GetAddressOf()));
//
//        // Staging texture for CPU read
//        td.Usage = D3D11_USAGE_STAGING;
//        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
//
//        winrt::check_hresult(d3dDevice_->CreateTexture2D(&td, nullptr, roiStageTex_.GetAddressOf()));
//    }
//
//private:
//    std::atomic<bool> running_{ false };
//    int targetFps_ = 60;
//
//    ComPtr<ID3D11Device> d3dDevice_;
//    ComPtr<ID3D11DeviceContext> d3dCtx_;
//    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice d3dWinrtDevice_{ nullptr };
//
//    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool framePool_{ nullptr };
//    winrt::Windows::Graphics::Capture::GraphicsCaptureSession session_{ nullptr };
//    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::FrameArrived_revoker frameArrivedRevoker_{};
//
//    std::mutex mtx_;
//    RECT roi_{ 0,0,0,0 };
//    RECT lastRoi_{ 0,0,0,0 };
//    bool roiDirty_ = false;
//
//    ComPtr<ID3D11Texture2D> roiGpuTex_;
//    ComPtr<ID3D11Texture2D> roiStageTex_;
//
//    BgraFrame latest_{};
//    uint64_t frameId_ = 0;
//};
