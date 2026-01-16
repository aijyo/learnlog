#pragma once
// BgraFramePngSaver.h/.cpp (single file example)
// Build: MSVC, /std:c++17

#include <windows.h>
#include <wincodec.h>   // WIC
#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <iomanip>

// Link with: Windowscodecs.lib (usually auto by SDK; if needed add in project settings)
#pragma comment(lib, "windowscodecs.lib")

#include "utils.h"

class BgraFramePngSaver
{
public:
    // If outputDir is empty, default to "<exe_dir>\\data"
    explicit BgraFramePngSaver(std::wstring outputDir = L"")
    {
        EnsureComInitialized_();
        output_dir_ = outputDir.empty() ? DefaultDataDir_() : std::filesystem::path(outputDir);
        std::error_code ec;
        std::filesystem::create_directories(output_dir_, ec); // best-effort
    }

    // Save frame to PNG. If fileName is empty -> "frame_<frame_id>.png"
    // Returns true on success. Optionally returns full path in outPath.
    bool Save(const BgraFrame& frame, const std::wstring& fileName = L"", std::wstring* outPath = nullptr)
    {
        if (!IsFrameValid_(frame))
            return false;

        const std::wstring finalName = fileName.empty() ? MakeDefaultName_(frame.frame_id) : fileName;
        const std::filesystem::path fullPath = output_dir_ / finalName;

        if (outPath) *outPath = fullPath.wstring();

        return SavePngWic_(frame, fullPath.wstring());
    }

    const std::filesystem::path& OutputDir() const { return output_dir_; }

private:
    // --- COM init RAII (thread-affine) ---
    void EnsureComInitialized_()
    {
        // It's OK if COM is already initialized on this thread.
        // We keep a flag only for potential future extension; we don't CoUninitialize here
        // because caller might manage COM lifetime. CoUninitialize mismatches can cause issues.
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE)
        {
            // If RPC_E_CHANGED_MODE: COM already initialized as STA; still OK to use WIC.
        }
    }

    static bool IsFrameValid_(const BgraFrame& f)
    {
        if (f.width <= 0 || f.height <= 0) return false;
        if (f.stride < f.width * 4) return false;
        const size_t need = static_cast<size_t>(f.stride) * static_cast<size_t>(f.height);
        if (f.data.size() < need) return false;
        return true;
    }

    static std::filesystem::path DefaultDataDir_()
    {
        wchar_t exePath[MAX_PATH] = {};
        DWORD n = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        std::filesystem::path p = (n > 0) ? std::filesystem::path(exePath) : std::filesystem::current_path();
        std::filesystem::path dir = p.has_filename() ? p.parent_path() : p;
        return dir / L"data";
    }

    static std::wstring MakeDefaultName_(uint64_t frameId)
    {
        std::wostringstream oss;
        oss << L"frame_" << frameId << L".png";
        return oss.str();
    }

    static bool SavePngWic_(const BgraFrame& frame, const std::wstring& path)
    {
        // Create WIC factory
        IWICImagingFactory* factory = nullptr;
        HRESULT hr = CoCreateInstance(
            CLSID_WICImagingFactory2, nullptr, CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&factory));
        if (FAILED(hr))
        {
            // Fallback for older systems
            hr = CoCreateInstance(
                CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                IID_PPV_ARGS(&factory));
        }
        if (FAILED(hr) || !factory) return false;

        IWICStream* stream = nullptr;
        IWICBitmapEncoder* encoder = nullptr;
        IWICBitmapFrameEncode* frameEncode = nullptr;
        IPropertyBag2* props = nullptr;

        bool ok = false;

        do
        {
            hr = factory->CreateStream(&stream);
            if (FAILED(hr) || !stream) break;

            hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE);
            if (FAILED(hr)) break;

            hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder);
            if (FAILED(hr) || !encoder) break;

            hr = encoder->Initialize(stream, WICBitmapEncoderNoCache);
            if (FAILED(hr)) break;

            hr = encoder->CreateNewFrame(&frameEncode, &props);
            if (FAILED(hr) || !frameEncode) break;

            hr = frameEncode->Initialize(props);
            if (FAILED(hr)) break;

            hr = frameEncode->SetSize(static_cast<UINT>(frame.width), static_cast<UINT>(frame.height));
            if (FAILED(hr)) break;

            // We write as 32bpp BGRA directly
            WICPixelFormatGUID pixelFormat = GUID_WICPixelFormat32bppBGRA;
            hr = frameEncode->SetPixelFormat(&pixelFormat);
            if (FAILED(hr)) break;

            // Some encoders may change pixelFormat; we accept BGRA only here
            if (pixelFormat != GUID_WICPixelFormat32bppBGRA)
                break;

            hr = frameEncode->WritePixels(
                static_cast<UINT>(frame.height),
                static_cast<UINT>(frame.stride),
                static_cast<UINT>(frame.stride * frame.height),
                const_cast<BYTE*>(reinterpret_cast<const BYTE*>(frame.data.data())));
            if (FAILED(hr)) break;

            hr = frameEncode->Commit();
            if (FAILED(hr)) break;

            hr = encoder->Commit();
            if (FAILED(hr)) break;

            ok = true;
        } while (false);

        if (props) props->Release();
        if (frameEncode) frameEncode->Release();
        if (encoder) encoder->Release();
        if (stream) stream->Release();
        factory->Release();

        return ok;
    }

private:
    std::filesystem::path output_dir_;
};
