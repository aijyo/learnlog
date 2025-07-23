// D3D11VideoRenderer.cpp
#include "d3d11-i420-render.h"
#include <d3dcompiler.h>
#include <stdexcept>
#include <cstring>
#include <DirectXColors.h>

// HLSL：简单 VS + YUV→RGB PS
static const char* kShaderSrc = R"(
struct VSInput { float3 pos : POSITION; float2 tex : TEXCOORD; };
struct PSInput { float4 pos : SV_POSITION; float2 tex : TEXCOORD; };

PSInput VSMain(VSInput vin) {
    PSInput o; o.pos = float4(vin.pos, 1); o.tex = vin.tex; return o;
}

Texture2D yTex : register(t0);
Texture2D uTex : register(t1);
Texture2D vTex : register(t2);
SamplerState smp  : register(s0);

float4 PSMain(PSInput i) : SV_Target {
    float y = yTex.Sample(smp, i.tex).r;
    float u = uTex.Sample(smp, i.tex).r - 0.5;
    float v = vTex.Sample(smp, i.tex).r - 0.5;
    float r = y + 1.402 * v;
    float g = y - 0.344136 * u - 0.714136 * v;
    float b = y + 1.772 * u;
    return float4(r, g, b, 1);
}
)";

D3D11VideoRenderer::D3D11VideoRenderer() {}
D3D11VideoRenderer::~D3D11VideoRenderer() { Cleanup(); }

void D3D11VideoRenderer::set_window(HWND hwnd) {
	hwnd_ = hwnd;
	GetWindowDimensions();
	InitializeD3D();
}

void D3D11VideoRenderer::GetWindowDimensions() {
	RECT rc; GetClientRect(hwnd_, &rc);
	width_ = rc.right - rc.left;
	height_ = rc.bottom - rc.top;
}

void D3D11VideoRenderer::InitializeD3D() {
	// 1. 创建设备和上下文（先用临时 swapChain 描述，实际后面替换）
	ComPtr<IDXGIFactory2> factory2;
	D3D_FEATURE_LEVEL fl;
	UINT flags = 0;
#ifdef _DEBUG
	flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
	HRESULT hr = D3D11CreateDevice(
		nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
		nullptr, 0, D3D11_SDK_VERSION,
		&device_, &fl, &context_);
	if (FAILED(hr)) { std::cerr << "CreateDevice 失败\n"; return; }

	// 2. 拿到 IDXGIFactory2
	ComPtr<IDXGIDevice> dxgiDev;
	device_->QueryInterface(IID_PPV_ARGS(&dxgiDev));
	ComPtr<IDXGIAdapter> adapter;
	dxgiDev->GetAdapter(&adapter);
	adapter->GetParent(IID_PPV_ARGS(&factory2));

	// 3. Flip-model 交换链描述
	DXGI_SWAP_CHAIN_DESC1 scd = {};
	scd.Width = width_;
	scd.Height = height_;
	scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
	scd.BufferCount = 2;
	scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	scd.SampleDesc.Count = 1;

	// 4. 创建交换链
	factory2->CreateSwapChainForHwnd(
		device_.Get(), hwnd_, &scd,
		nullptr, nullptr, &swapChain_);

	// 5. 创建 RTV & 着色器 & 四边形
	CreateRenderTarget();
	CreateShadersAndLayout();
	CreateQuad();
}

void D3D11VideoRenderer::CreateRenderTarget() {
	ComPtr<ID3D11Texture2D> bb;
	swapChain_->GetBuffer(0, IID_PPV_ARGS(&bb));
	device_->CreateRenderTargetView(bb.Get(), nullptr, &renderTargetView_);
}

void D3D11VideoRenderer::CreateShadersAndLayout() {
	ComPtr<ID3DBlob> vsBlob, psBlob, err;
	D3DCompile(kShaderSrc, strlen(kShaderSrc), nullptr, nullptr, nullptr,
		"VSMain", "vs_5_0", 0, 0, &vsBlob, &err);
	device_->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
		nullptr, &vs_);

	D3DCompile(kShaderSrc, strlen(kShaderSrc), nullptr, nullptr, nullptr,
		"PSMain", "ps_5_0", 0, 0, &psBlob, &err);
	device_->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(),
		nullptr, &ps_);

	// 输入布局
	D3D11_INPUT_ELEMENT_DESC ie[] = {
		{"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(VSInput,pos),D3D11_INPUT_PER_VERTEX_DATA,0},
		{"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,   0,offsetof(VSInput,tex),D3D11_INPUT_PER_VERTEX_DATA,0},
	};
	device_->CreateInputLayout(
		ie, _countof(ie),
		vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
		&inputLayout_);
}

void D3D11VideoRenderer::CreateQuad() {
	struct V { DirectX::XMFLOAT3 p; DirectX::XMFLOAT2 t; };
	V v[4] = {
		{{-1,-1,0},{0,1}}, {{1,-1,0},{1,1}},
		{{-1, 1,0},{0,0}}, {{1, 1,0},{1,0}},
	};
	D3D11_BUFFER_DESC bd = {};
	bd.Usage = D3D11_USAGE_DEFAULT;
	bd.ByteWidth = sizeof(v);
	bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	D3D11_SUBRESOURCE_DATA sd = { v };
	device_->CreateBuffer(&bd, &sd, &vertexBuffer_);
}

void D3D11VideoRenderer::SetFrame(std::shared_ptr<I420Frame> frame) {
	currentFrame_ = std::move(frame);
	CreateTextures();
	UpdateTextures();
}

void D3D11VideoRenderer::CreateTextures() {
	D3D11_TEXTURE2D_DESC td = {};
	td.MipLevels = 1; td.ArraySize = 1; td.Format = DXGI_FORMAT_R8_UNORM;
	td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	td.SampleDesc.Count = 1;

	td.Width = currentFrame_->width();
	td.Height = currentFrame_->height();
	device_->CreateTexture2D(&td, nullptr, &texY_);

	td.Width /= 2; td.Height /= 2;
	device_->CreateTexture2D(&td, nullptr, &texU_);
	device_->CreateTexture2D(&td, nullptr, &texV_);

	D3D11_SHADER_RESOURCE_VIEW_DESC srd = {};
	srd.Format = DXGI_FORMAT_R8_UNORM;
	srd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srd.Texture2D.MipLevels = 1;

	device_->CreateShaderResourceView(texY_.Get(), &srd, &srvY_);
	device_->CreateShaderResourceView(texU_.Get(), &srd, &srvU_);
	device_->CreateShaderResourceView(texV_.Get(), &srd, &srvV_);
}

void D3D11VideoRenderer::UpdateTextures() {
	context_->UpdateSubresource(texY_.Get(), 0, nullptr,
		currentFrame_->data_y(), currentFrame_->stride_y(), 0);
	context_->UpdateSubresource(texU_.Get(), 0, nullptr,
		currentFrame_->data_u(), currentFrame_->stride_u(), 0);
	context_->UpdateSubresource(texV_.Get(), 0, nullptr,
		currentFrame_->data_v(), currentFrame_->stride_v(), 0);
}

void D3D11VideoRenderer::Render() {
	// 1. RTV + 清屏
	context_->OMSetRenderTargets(1, renderTargetView_.GetAddressOf(), nullptr);
	FLOAT clear[4] = { 0,0,0,1 };
	context_->ClearRenderTargetView(renderTargetView_.Get(), clear);

	// 2. 设置视口
	D3D11_VIEWPORT vp = { 0,0,
		(FLOAT)width_,(FLOAT)height_,0.0f,1.0f };
	context_->RSSetViewports(1, &vp);

	// 3. 绑定管线
	context_->IASetInputLayout(inputLayout_.Get());
	context_->VSSetShader(vs_.Get(), nullptr, 0);
	context_->PSSetShader(ps_.Get(), nullptr, 0);
	// 创建两个数组用于存储 stride 和 offset
	UINT stride[] = { sizeof(VSInput) };
	UINT offset[] = { 0 };

	// 调用 IASetVertexBuffers，传入这些数组
	context_->IASetVertexBuffers(0, 1, vertexBuffer_.GetAddressOf(), stride, offset);
	context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

	// 4. 绑定 YUV 纹理+采样器
	context_->PSSetShaderResources(0, 1, srvY_.GetAddressOf());
	context_->PSSetShaderResources(1, 1, srvU_.GetAddressOf());
	context_->PSSetShaderResources(2, 1, srvV_.GetAddressOf());
	static ComPtr<ID3D11SamplerState> smp;
	if (!smp) {
		D3D11_SAMPLER_DESC sd = {};
		sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
		device_->CreateSamplerState(&sd, &smp);
	}

	//D3D11_RASTERIZER_DESC rd = {};
	////rd.FillMode = D3D11_FILL_SOLID;
	////rd.CullMode = D3D11_CULL_NONE;       // 关闭背面剔除
	//rd.CullMode = D3D11_CULL_BACK;
	//rd.FrontCounterClockwise = TRUE;   // CCW 算前面，CW 算背面
	//rd.FrontCounterClockwise = FALSE;
	D3D11_RASTERIZER_DESC rd = {};
	rd.FillMode = D3D11_FILL_SOLID;      // 一定要设
	rd.CullMode = D3D11_CULL_BACK;       // 或者 CULL_NONE
	rd.FrontCounterClockwise = TRUE;                 // FALSE＝顺时针为“正面”
	rd.DepthClipEnable = TRUE;                  // 推荐打开深度裁剪
	static  ComPtr<ID3D11RasterizerState> rs;
	if (!rs) {
		device_->CreateRasterizerState(&rd, &rs);
	}
	context_->RSSetState(rs.Get());

	context_->PSSetSamplers(0, 1, smp.GetAddressOf());

	context_->OMSetDepthStencilState(nullptr, 0);  // 禁用深度测试

	// 5. Draw & Present
	context_->Draw(4, 0);
	swapChain_->Present(1, 0);
}

void D3D11VideoRenderer::Cleanup() {
	// ComPtr 会自动释放，无需手动 Release()
}