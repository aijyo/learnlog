#pragma once
#include <d3d11.h>
#include <dxgi1_2.h>        // IDXGIFactory2, IDXGISwapChain1
#include <d3dcompiler.h>    // D3DCompile
#include <wrl.h>            // Microsoft::WRL::ComPtr
#include <DirectXMath.h>
#include <memory>
#include <iostream>
//#include "simple-image-render.h"

//using I420Frame = lib_d3d::I420Frame;
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

enum class ImageFormat {
	kUnkown = 0,
	kRGBA, // RGBA format
	kBGRA, // BGRA format
	kI420, // I420 format
	kNV12, // NV12 format
};

using Microsoft::WRL::ComPtr;
using uid_t = std::uint64_t;
class I420Frame {
public:
	I420Frame(int width, int height);
	I420Frame(int width, int height, const uint8_t* src_y, int src_stride_y,
		const uint8_t* src_u, int src_stride_u, const uint8_t* src_v, int src_stride_v);
	~I420Frame();

	I420Frame(const I420Frame&) = delete;
	I420Frame& operator=(const I420Frame&) = delete;

	I420Frame(I420Frame&& other) noexcept;
	I420Frame& operator=(I420Frame&& other) noexcept;

	uint8_t* data_y() const { return data_y_; }
	uint8_t* data_u() const { return data_u_; }
	uint8_t* data_v() const { return data_v_; }

	int width() const { return width_; }
	int height() const { return height_; }
	ImageFormat format() const { return ImageFormat::kI420; }

	int stride_y() const { return width_; }
	int stride_u() const { return width_ / 2; }
	int stride_v() const { return width_ / 2; }

	void clear();

private:
	void allocate();

	int width_ = 0;
	int height_ = 0;
	uint8_t* data_y_ = nullptr;
	uint8_t* data_u_ = nullptr;
	uint8_t* data_v_ = nullptr;
	std::unique_ptr<uint8_t[]> buffer_;
};

//using Microsoft::WRL::ComPtr;

struct VSInput {
	DirectX::XMFLOAT3 pos;   // POSITION
	DirectX::XMFLOAT2 tex;   // TEXCOORD
};

class D3D11VideoRenderer {
public:
	D3D11VideoRenderer();
	~D3D11VideoRenderer();

	void set_window(HWND hwnd);
	void SetFrame(std::shared_ptr<I420Frame> frame);
	void Render();

private:
	HWND                           hwnd_ = nullptr;
	int                            width_ = 0;
	int                            height_ = 0;

	// D3D11 核心接口
	Microsoft::WRL::ComPtr<ID3D11Device>           device_;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext>    context_;
	Microsoft::WRL::ComPtr<IDXGISwapChain1>        swapChain_;           // 正确的 DXGI 交换链
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> renderTargetView_;

	// YUV 纹理 & 资源视图
	Microsoft::WRL::ComPtr<ID3D11Texture2D>        texY_, texU_, texV_;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srvY_, srvU_, srvV_;

	// 着色器与输入布局
	Microsoft::WRL::ComPtr<ID3D11VertexShader>     vs_;
	Microsoft::WRL::ComPtr<ID3D11PixelShader>      ps_;
	Microsoft::WRL::ComPtr<ID3D11InputLayout>      inputLayout_;
	Microsoft::WRL::ComPtr<ID3D11Buffer>           vertexBuffer_;

	std::shared_ptr<I420Frame>     currentFrame_;

	void InitializeD3D();
	void Cleanup();
	void GetWindowDimensions();
	void CreateRenderTarget();
	void CreateShadersAndLayout();
	void CreateQuad();
	void CreateTextures();
	void UpdateTextures();
};