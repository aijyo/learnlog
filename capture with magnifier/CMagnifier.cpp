#include "CMagnifier.h"

#include <dwmapi.h>



//HWND m_hMagnifierWnd;
//RECT m_rcPos;
//HBITMAP m_hMemBitmap;
//int m_nWidth;
//int m_nHeight;
//bool m_bFilter;
//std::vector<HWND> m_vcFilterWnd;
//
////m11 m12
////m21 m22
////dx dy
//float m_transforMatrix[6];
CMagnifier::CMagnifier()
	: m_hMagnifierWnd(NULL)
	, m_hMemBitmap(NULL)
	, m_nWidth(0)
	, m_nHeight(0)
	, m_bFilterExclude(false)
{
	memset(&m_rcCapture, 0, sizeof(RECT));
	memset(m_transforMatrix, 0, sizeof(m_transforMatrix));
	m_transforMatrix[0] = 1.0;
	m_transforMatrix[3] = 1.0;
}

CMagnifier::~CMagnifier()
{
	if (m_hMemBitmap)
	{
		DeleteObject(m_hMemBitmap);
		m_hMemBitmap = NULL;
	}
}

void CMagnifier::setFilterList(const std::vector<HWND>& vcFilter, bool bExclude /* = true */)
{
	m_vcFilterWnd = vcFilter;
	m_bFilterExclude = bExclude;
}
void CMagnifier::setCaptureRect(const RECT& pos)
{
	m_rcCapture = pos;
	m_nWidth = pos.right - pos.left;
	m_nHeight = pos.bottom - pos.top;
}

void CMagnifier::setPaintWindow(HWND hwnd)
{
	m_hMagnifierWnd = hwnd;
}

HBITMAP CMagnifier::RemakeOsBitmaps(HDC hScreenDC)
{
	int nRight; // edi
	int nBottom; // ebx
	HBITMAP hMemBitmap; // ecx
	HBITMAP result; // eax
	int nWidth; // edi
	int nHeight; // ebx
	HDC hScreenDC_1; // [esp+Ch] [ebp-4h]

	hScreenDC_1 = hScreenDC;
	nRight = m_rcCapture.right;
	nBottom = m_rcCapture.bottom;
	hMemBitmap = m_hMemBitmap;
	result = hMemBitmap;
	nWidth = nRight -m_rcCapture.left;
	nHeight = nBottom - m_rcCapture.right;
	if (hMemBitmap && (m_nWidth != nWidth || m_nHeight != nHeight))
	{
		DeleteObject(hMemBitmap);
		hScreenDC = hScreenDC_1;
		result = 0;
		m_hMemBitmap = NULL;
	}
	if (!result)
	{
		result = CreateCompatibleBitmap(hScreenDC, nWidth, nHeight);
		m_hMemBitmap = result;
		m_nWidth = nWidth;
		m_nHeight = nHeight;
	}
	return result;
}

unsigned int CMagnifier::RunFilterList(HDC hMemDC, int width, int height, int left, int top)
{
	HDC hMemDC_2; // esi RunFilterList((DWORD *)(pThis + 56), hMemDC, width, height, *(DWORD *)(pThis + 4), *(DWORD *)(pThis + 8));
	unsigned int v8; // edi
	HRGN rgn_1; // eax
	HRGN rgn_2; // ebx
	signed int v11; // eax
	bool v12; // sf
	//DWORD* v13; // eax
	//unsigned int v14; // ecx
	HDC v15; // esi
	int v16; // eax
	HBRUSH v17; // esi
	signed int v18; // eax
	unsigned int index; // [esp+1Ch] [ebp-1Ch]
	RECT rc; // [esp+24h] [ebp-14h] BYREF

	hMemDC_2 = hMemDC;

	//if (m_bFilterExclude)
	//	return 0;
	if (m_vcFilterWnd.empty())
		return 0;
	v8 = 0;
	SetLastError(0);
	rgn_1 = CreateRectRgn(0, 0, width, height);
	rgn_2 = rgn_1;
	if (rgn_1)
	{
		SelectClipRgn(hMemDC_2, rgn_1);
	}
	else
	{
		v11 = GetLastError();
		v8 = v11;
		v12 = v11 < 0;
		if (v11 > 0)
		{
			v8 = (unsigned __int16)v11 | 0x80070000;
			v12 = 1;
		}
		if (!v12)
			v8 = (unsigned int)-2147467259;
	}
	//v13 = &m_vcFilterWnd[0];
	for (index = 0; index < m_vcFilterWnd.size(); ++index)
	{
		if (GetWindowRect(m_vcFilterWnd[index], &rc))
		{
			rc.left -= left;
			v15 = hMemDC;
			rc.right -= left;
			rc.top -= top;
			rc.bottom -= top;
			v16 = ExcludeClipRect(hMemDC, rc.left, rc.top, rc.right, rc.bottom);
			if (!v16)
			{
				v8 = (unsigned int)-2147467259;
				goto LABEL_19;
			}
			if (v16 == 1)
				goto LABEL_19;
		}
	}
	v17 = CreateSolidBrush(0x808080u);
	if (v17)
	{
		rc.left = 0;
		rc.top = 0;
		rc.right = width;
		rc.bottom = height;
		FillRect(hMemDC, &rc, v17);
		DeleteObject(v17);
		v15 = hMemDC;
	}
	else
	{
		v18 = GetLastError();
		v15 = hMemDC;
		v8 = v18;
		if (v18 > 0)
			v8 = (unsigned __int16)v18 | 0x80070000;
	}
LABEL_19:
	if (rgn_2)
	{
		SelectClipRgn(v15, rgn_2);
		DeleteObject(rgn_2);
	}
	return v8;
}

BOOL CMagnifier::DrawCursor(HDC hDC, int left, int top)
{
	BOOL result; // eax
	CURSORINFO pci; // [esp+8h] [ebp-30h] BYREF
	ICONINFO piconinfo; // [esp+1Ch] [ebp-1Ch] BYREF
	POINT curPoint; // [esp+30h] [ebp-8h] BYREF

	curPoint.x = 0;
	curPoint.y = 0;
	pci.cbSize = 20;
	result = GetCursorInfo(&pci);
	if (result)
	{
		if (pci.hCursor)
		{
			if ((pci.flags & 1) != 0)
			{
				result = GetPhysicalCursorPos(&curPoint);
				if (result)
				{
					result = GetIconInfo(pci.hCursor, &piconinfo);
					if (result)
					{
						if (piconinfo.hbmColor)
							DeleteObject(piconinfo.hbmColor);
						if (piconinfo.hbmMask)
							DeleteObject(piconinfo.hbmMask);
						result = DrawIcon(hDC, curPoint.x - piconinfo.xHotspot - left, curPoint.y - piconinfo.yHotspot - top, pci.hCursor);
					}
				}
			}
		}
	}
	return result;
}

//m11 m12
//m21 m22
//dx dy
int TransformPoint(POINT& point, float matrix[])
{
	int result = 0;
	int x = point.x;
	int y = point.y;

	//x' = m11*x + m21*y + dx
	//y' = m22*y + m12*x + dy
	point.x = (int)(matrix[0] * x + matrix[2] * y + matrix[4]);
	point.y = (int)(matrix[3] * y + matrix[1] * x + matrix[5]);
	return result;
}

BOOL CMagnifier::CutOut(HWND hMagWnd, HDC hMagDC, HDC hdcSrc)
{
	POINT cursorPoints; // [esp+8h] [ebp-10h] BYREF
	POINT cursorPoint; // [esp+10h] [ebp-8h] BYREF

	GetPhysicalCursorPos(&cursorPoint);
	cursorPoints = cursorPoint;
	MapWindowPoints(0, hMagWnd, &cursorPoints, 1u);
	return BitBlt(hMagDC, cursorPoints.x - 22, cursorPoints.y - 22, 44, 44, hdcSrc, cursorPoint.x - 22, cursorPoint.y - 22, 0xCC0020u);
}

HDC CMagnifier::MagnifierPaint(HDC magnifierWndDC)
{
	HDC hScreenDC; // eax
	HDC hScreenDC_1; // esi
	HDC hMemDC; // ebx
	HGDIOBJ hOldBitmap; // [esp+10h] [ebp-40h]
	HDC hScreenDC_2; // [esp+14h] [ebp-3Ch]
	int width; // [esp+1Ch] [ebp-34h]
	int height; // [esp+20h] [ebp-30h]
	RECT rcMagnifer; // [esp+24h] [ebp-2Ch] BYREF
	POINT points[3];
	POINT& ptLeftTop = points[0]; // [esp+34h] [ebp-1Ch] BYREF
	POINT& ptRightTop = points[1]; // [esp+3Ch] [ebp-14h] BYREF
	POINT& ptLeftBottom = points[2]; // [esp+44h] [ebp-Ch] BYREF
	RECT& rcCapture = m_rcCapture;
	//RECT rcPos = { 0 };
	//GetWindowRect(m_hMagnifierWnd, &rcPos);

	hScreenDC = GetDC(NULL);
	hScreenDC_1 = hScreenDC;
	hScreenDC_2 = hScreenDC;
	if (m_hMagnifierWnd && hScreenDC)
	{
		hMemDC = CreateCompatibleDC(hScreenDC);
		if (hMemDC)
		{
			width = rcCapture.right - rcCapture.left;//  v15 = pThis[3] - pThis[1];
			height = rcCapture.bottom - rcCapture.top;//  v16 = pThis[4] - pThis[2];
			//RemakeOsBitmaps(hScreenDC_1);
			{
				HBITMAP hMemBitmap; // ecx
				hMemBitmap = m_hMemBitmap;
				if (hMemBitmap && (m_nWidth != width || m_nHeight != height))
				{
					DeleteObject(hMemBitmap);
					hScreenDC = hScreenDC_1;
					m_hMemBitmap = NULL;
				}
				if (!m_hMemBitmap)
				{
					m_hMemBitmap = CreateCompatibleBitmap(hScreenDC, width, height);
					m_nWidth = width;
					m_nHeight = height;
				}
			}
			hOldBitmap = SelectObject(hMemDC, m_hMemBitmap);
			BitBlt(hMemDC, 0, 0, width, height, 0, 0, 0, 0x42u);
			BitBlt(hMemDC, 0, 0, width, height, hScreenDC_1, rcCapture.left, rcCapture.top, 0xC0CC0020);
			RunFilterList(hMemDC, width, height, rcCapture.left, rcCapture.top);
			if ((GetWindowLongW(m_hMagnifierWnd, -16) & 1) != 0)
				DrawCursor(hMemDC, rcCapture.left, rcCapture.top);
			GetClientRect(m_hMagnifierWnd, &rcMagnifer);
			IntersectClipRect(magnifierWndDC, rcMagnifer.left, rcMagnifer.top, rcMagnifer.right, rcMagnifer.bottom);
			ptRightTop.x = width;
			ptLeftTop.x = 0;
			ptLeftTop.y = 0;
			ptRightTop.y = 0;
			ptLeftBottom.x = 0;
			ptLeftBottom.y = height;
			// point matrix
			TransformPoint(ptLeftTop, m_transforMatrix);
			TransformPoint(ptRightTop, m_transforMatrix);
			TransformPoint(ptLeftBottom, m_transforMatrix);

			if ((GetWindowLongW(m_hMagnifierWnd, -16) & 4) != 0)
				BitBlt(hMemDC, 0, 0, width, height, 0, 0, 0, 0x550009u);
			PlgBlt(magnifierWndDC, points, hMemDC, 0, 0, width, height, 0, 0, 0);
			//BitBlt(hMemDC, 0, 0, width, height, hMemDC, 0, 0, /*0x550009u*/SRCCOPY);
			hScreenDC_1 = hScreenDC_2;
			if ((GetWindowLongW(m_hMagnifierWnd, -16) & 2) != 0)
				CutOut(m_hMagnifierWnd, magnifierWndDC, hScreenDC_2);
			SelectObject(hMemDC, hOldBitmap);
			DeleteDC(hMemDC);
		}
		hScreenDC = (HDC)ReleaseDC(0, hScreenDC_1);
	}
	return hScreenDC;
}

#include <Windef.h>
#include <d3d9.h>
#pragma comment(lib, "d3d9.lib")  // located in DirectX SDK

static const int kBytesPerPixel = 4;
int CMagnifier::DoCapture(HWND hMagnifierWindow)
{
	char* lpMem = (char*)GetWindowLongW(hMagnifierWindow, 0);
	int pThis = *((DWORD*)lpMem + 46);
	//DWORD dwContinue = 1;
	do 
	{
		if (!pThis) break;
		int nCount = *(DWORD*)(pThis + 7152);

		if(nCount < 1) break;

		for (int index = 0; index < nCount; index++)
		{
			//xFilterTextureD3D9(pThis, index, dwContinue);

			IDirect3DDevice9* d3d_device_ = nullptr;
			IDirect3DTexture9* texture_ = nullptr;
			//IDirect3DVertexBuffer9* vertex_buffer_ = nullptr;
			IDirect3DSurface9* pSurface = nullptr;
			IDirect3DSurface9* pPlainSurface = nullptr;
			D3DSURFACE_DESC pDesc;
			int pThis_7128; // ecx
			int pThis_7604; // edi
			int pThis_7128_2; // [esp+98h] [ebp-60h]


			D3DLOCKED_RECT rcLock = { 0 };

			RECT rcCapture = m_rcCapture;
			RECT rcClipped = { 0 };
			int nWidth = 0;
			int nHeight = 0;

			pThis_7128 = *(DWORD*)(pThis + 4 * index + 7128);
			pThis_7604 = *(DWORD*)(pThis + 4 * index + 7604);
			pThis_7128_2 = pThis_7128;

			d3d_device_ = (IDirect3DDevice9*)(*(DWORD*)(*(DWORD*)pThis_7128_2));
			texture_ = (IDirect3DTexture9*)(*(DWORD*)(*(DWORD*)pThis_7604));


			int v7 = 52 * *(DWORD*)(pThis + 4 * index + 7576);;
			RECT rcScreen = *(const RECT*)(pThis + v7 + 7268);
			BOOL bSuc = IntersectRect(&rcClipped, &rcScreen, &rcCapture);
			do
			{
				if (!bSuc) break;

				OffsetRect(&rcClipped, -rcCapture.left, -rcCapture.top);

				nWidth = rcClipped.right - rcClipped.left;
				nHeight = rcClipped.bottom - rcClipped.top;

				HRESULT result = (*(int(__thiscall**)(DWORD, int, DWORD, int*))(*(DWORD*)pThis_7604 + 68))(
					*(DWORD*)(*(DWORD*)pThis_7604 + 68),
					pThis_7604,
					0,
					(int*)&pDesc);                          // public: virtual long __stdcall CMipMap::GetLevelDesc(unsigned int, struct _D3DSURFACE_DESC *)

				//HRESULT result = texture_->GetLevelDesc(0, &pDesc);


				pSurface = nullptr;
				result = (*(int(__thiscall**)(DWORD, int, DWORD, int*))(*(DWORD*)pThis_7604 + 72))(
					*(DWORD*)(*(DWORD*)pThis_7604 + 72),
					pThis_7604,
					0,
					(int*)&pSurface);                    // public: virtual long __stdcall CMipMap::GetSurfaceLevel(unsigned int, struct IDirect3DSurface9 * *)
				//result = texture_->GetSurfaceLevel(0, &pSurface);
				int flag = 0;
				if (result < 0) break;

				pPlainSurface = nullptr;
				result = (*(int(__thiscall**)(DWORD, int, int, int, int, int, int*, DWORD))(*(DWORD*)pThis_7128_2 + 144))(// STDMETHOD(CreateOffscreenPlainSurface)(THIS_ UINT Width,UINT Height,D3DFORMAT Format,D3DPOOL Pool,IDirect3DSurface9** ppSurface,HANDLE* pSharedHandle) PURE;
					*(DWORD*)(*(DWORD*)pThis_7128_2 + 144),
					pThis_7128_2,
					nWidth,
					nHeight,
					pDesc.Format,
					D3DPOOL_SYSTEMMEM,
					(int*)&pPlainSurface,
					0);
				//result = d3d_device_->CreateOffscreenPlainSurface(nWidth, nHeight, pDesc.Format, D3DPOOL_SYSTEMMEM, &pPlainSurface, 0);

				if (result < 0)
				{
					flag = 19;
					break;
				}

				RECT rcSource = { 0, 0, nWidth, nHeight };
				bSuc = IntersectRect(&rcClipped, &rcSource, &rcClipped);
				if (!bSuc)
				{
					break;
				}
				result = (*(int(__thiscall**)(DWORD, int, int, int))(*(DWORD*)pThis_7128_2 + 128))(// STDMETHOD(GetRenderTargetData)(THIS_ IDirect3DSurface9* pRenderTarget,IDirect3DSurface9* pDestSurface) PURE;
					*(DWORD*)(*(DWORD*)pThis_7128_2 + 128),
					pThis_7128_2,
					(int)pSurface,
					(int)pPlainSurface);                  // public: virtual long __stdcall CBaseDevice::GetRenderTargetData(struct IDirect3DSurface9 *, struct IDirect3DSurface9 *)
				//result = d3d_device_->GetRenderTargetData(pSurface, pPlainSurface);

				if (result < 0)
				{
					break;
				}

				result = pPlainSurface->LockRect(&rcLock, &rcClipped, D3DLOCK_NOSYSLOCK);

				if (result < 0)
					break;

				int dataLen = kBytesPerPixel * nWidth * nHeight;
				//if(bResize)
				m_vcImage.resize(dataLen);
				char* pBuf = (char*)&m_vcImage[0];
				int stride = nWidth * kBytesPerPixel;

				const char* rpixels = (const char*)rcLock.pBits;
				int rpitch = rcLock.Pitch;

				if (stride == rpitch)
				{
					memcpy(pBuf, rpixels, dataLen);
				}
				else
				{
					for (int i = 0; i < nHeight; ++i)
					{
						memcpy(pBuf, rpixels, stride);
						pBuf += stride;
						rpixels += rpitch;
					}
				}
				pPlainSurface->UnlockRect();


				//D3DFMT_R8G8B8 = 20,
				//D3DFMT_A8R8G8B8 = 21,

			} while (false);

			if (pSurface)
			{
				pSurface->Release();
				pSurface = nullptr;
			}

			if (pPlainSurface)
			{
				pPlainSurface->Release();
				pPlainSurface = nullptr;
			}
		}

	} while (false);
	return 0;
}

int CMagnifier::DoCaptureWin11(HWND hMagnifierWindow)
{
	char* lpMem = (char*)GetWindowLongW(hMagnifierWindow, 0);
	int pThis = *((DWORD*)lpMem + 46);
	//DWORD dwContinue = 1;
	do
	{
		if (!pThis) break;
		int nCount = *(DWORD*)(pThis + 7152);

		if (nCount < 1) break;

		for (int index = 0; index < nCount; index++)
		{
			//xFilterTextureD3D9(pThis, index, dwContinue);

			IDirect3DDevice9* d3d_device_ = nullptr;
			IDirect3DTexture9* texture_ = nullptr;
			//IDirect3DVertexBuffer9* vertex_buffer_ = nullptr;
			IDirect3DSurface9* pSurface = nullptr;
			IDirect3DSurface9* pPlainSurface = nullptr;
			D3DSURFACE_DESC pDesc;
			int pThis_7128; // ecx
			int pThis_7604; // edi
			int pThis_7128_2; // [esp+98h] [ebp-60h]


			D3DLOCKED_RECT rcLock = { 0 };

			RECT rcCapture = m_rcCapture;
			RECT rcClipped = { 0 };
			int nWidth = 0;
			int nHeight = 0;

			pThis_7128 = *(DWORD*)(pThis + 4 * index + 7128);
			pThis_7604 = *(DWORD*)(pThis + 4 * index + 7604);
			pThis_7128_2 = pThis_7128;

			d3d_device_ = (IDirect3DDevice9*)(*(DWORD*)(*(DWORD*)pThis_7128_2));
			texture_ = (IDirect3DTexture9*)(*(DWORD*)(*(DWORD*)pThis_7604));


			int v7 = 52 * *(DWORD*)(pThis + 4 * index + 7576);;
			RECT rcScreen = *(const RECT*)(pThis + v7 + 7268);
			BOOL bSuc = IntersectRect(&rcClipped, &rcScreen, &rcCapture);
			do
			{
				if (!bSuc) break;

				OffsetRect(&rcClipped, -rcCapture.left, -rcCapture.top);

				nWidth = rcClipped.right - rcClipped.left;
				nHeight = rcClipped.bottom - rcClipped.top;

				HRESULT result = (*(int(__thiscall**)(DWORD, int, DWORD, int*))(*(DWORD*)pThis_7604 + 68))(
					*(DWORD*)(*(DWORD*)pThis_7604 + 68),
					pThis_7604,
					0,
					(int*)&pDesc);                          // public: virtual long __stdcall CMipMap::GetLevelDesc(unsigned int, struct _D3DSURFACE_DESC *)

				//result = texture_->GetLevelDesc(0, &pDesc);


				pSurface = nullptr;
				result = (*(int(__thiscall**)(DWORD, int, DWORD, int*))(*(DWORD*)pThis_7604 + 72))(
					*(DWORD*)(*(DWORD*)pThis_7604 + 72),
					pThis_7604,
					0,
					(int*)&pSurface);                    // public: virtual long __stdcall CMipMap::GetSurfaceLevel(unsigned int, struct IDirect3DSurface9 * *)
				//result = texture_->GetSurfaceLevel(0, &pSurface);
				int flag = 0;
				if (result < 0) break;

				pPlainSurface = nullptr;
				result = (*(int(__thiscall**)(DWORD, int, int, int, int, int, int*, DWORD))(*(DWORD*)pThis_7128_2 + 144))(// STDMETHOD(CreateOffscreenPlainSurface)(THIS_ UINT Width,UINT Height,D3DFORMAT Format,D3DPOOL Pool,IDirect3DSurface9** ppSurface,HANDLE* pSharedHandle) PURE;
					*(DWORD*)(*(DWORD*)pThis_7128_2 + 144),
					pThis_7128_2,
					nWidth,
					nHeight,
					pDesc.Format,
					D3DPOOL_SYSTEMMEM,
					(int*)&pPlainSurface,
					0);
				//result = d3d_device_->CreateOffscreenPlainSurface(nWidth, nHeight, pDesc.Format, D3DPOOL_SYSTEMMEM, &pPlainSurface, 0);

				if (result < 0)
				{
					flag = 19;
					break;
				}

				RECT rcSource = { 0, 0, nWidth, nHeight };
				bSuc = IntersectRect(&rcClipped, &rcSource, &rcClipped);
				if (!bSuc)
				{
					break;
				}
				result = (*(int(__thiscall**)(DWORD, int, int, int))(*(DWORD*)pThis_7128_2 + 128))(// STDMETHOD(GetRenderTargetData)(THIS_ IDirect3DSurface9* pRenderTarget,IDirect3DSurface9* pDestSurface) PURE;
					*(DWORD*)(*(DWORD*)pThis_7128_2 + 128),
					pThis_7128_2,
					(int)pSurface,
					(int)pPlainSurface);                  // public: virtual long __stdcall CBaseDevice::GetRenderTargetData(struct IDirect3DSurface9 *, struct IDirect3DSurface9 *)
				//result = d3d_device_->GetRenderTargetData(pSurface, pPlainSurface);

				if (result < 0)
				{
					break;
				}

				result = pPlainSurface->LockRect(&rcLock, &rcClipped, D3DLOCK_NOSYSLOCK);

				if (result < 0)
					break;

				int dataLen = kBytesPerPixel * nWidth * nHeight;
				//if(bResize)
				m_vcImage.resize(dataLen);
				char* pBuf = (char*)&m_vcImage[0];
				int stride = nWidth * kBytesPerPixel;

				const char* rpixels = (const char*)rcLock.pBits;
				int rpitch = rcLock.Pitch;

				if (stride == rpitch)
				{
					memcpy(pBuf, rpixels, dataLen);
				}
				else
				{
					for (int i = 0; i < nHeight; ++i)
					{
						memcpy(pBuf, rpixels, stride);
						pBuf += stride;
						rpixels += rpitch;
					}
				}
				pPlainSurface->UnlockRect();


				//D3DFMT_R8G8B8 = 20,
				//D3DFMT_A8R8G8B8 = 21,

			} while (false);

			if (pSurface)
			{
				pSurface->Release();
				pSurface = nullptr;
			}

			if (pPlainSurface)
			{
				pPlainSurface->Release();
				pPlainSurface = nullptr;
			}
		}

	} while (false);
	return 0;
}

int CMagnifier::xFilterTextureD3D9(int pThis, unsigned int index, DWORD& bContinue)
{
	//assert(*(HWND*)pThis == hMagWindow);
	bContinue = 12;

	IDirect3DDevice9* d3d_device_ = nullptr;
	IDirect3DTexture9* texture_ = nullptr;
	//IDirect3DVertexBuffer9* vertex_buffer_ = nullptr;
	IDirect3DSurface9* pSurface = nullptr;
	IDirect3DSurface9* pPlainSurface = nullptr;
	D3DSURFACE_DESC pDesc;
	int pThis_7128; // ecx
	int pThis_7604; // edi
	int pThis_7128_2; // [esp+98h] [ebp-60h]


	D3DLOCKED_RECT rcLock = { 0 };

	RECT rcCapture = m_rcCapture;
	RECT rcClipped = { 0 };
	int nWidth = 0;
	int nHeight = 0;

	pThis_7128 = *(DWORD*)(pThis + 4 * index + 7128);
	pThis_7604 = *(DWORD*)(pThis + 4 * index + 7604);
	pThis_7128_2 = pThis_7128;

	d3d_device_ = (IDirect3DDevice9*)(*(DWORD*)(*(DWORD*)pThis_7128_2));
	texture_ = (IDirect3DTexture9*)(*(DWORD*)(*(DWORD*)pThis_7604));


	int v7 = 52 * *(DWORD*)(pThis + 4 * index + 7576);;
	RECT rcScreen = *(const RECT*)(pThis + v7 + 7268);
	HRESULT result = IntersectRect(&rcClipped, &rcScreen, &rcCapture);
	do 
	{
		if (!result) break;

		OffsetRect(&rcClipped, rcCapture.left, rcCapture.top);

		nWidth = rcClipped.right - rcClipped.left;
		nHeight = rcClipped.bottom - rcClipped.top;

		result = (*(int(__thiscall**)(DWORD, int, DWORD, int*))(*(DWORD*)pThis_7604 + 68))(
			*(DWORD*)(*(DWORD*)pThis_7604 + 68),
			pThis_7604,
			0,
			(int*)&pDesc);                          // public: virtual long __stdcall CMipMap::GetLevelDesc(unsigned int, struct _D3DSURFACE_DESC *)

		//result = texture_->GetLevelDesc(0, &pDesc);


		pSurface = nullptr;
		result = (*(int(__thiscall**)(DWORD, int, DWORD, int*))(*(DWORD*)pThis_7604 + 72))(
			*(DWORD*)(*(DWORD*)pThis_7604 + 72),
			pThis_7604,
			0,
			(int*)&pSurface);                    // public: virtual long __stdcall CMipMap::GetSurfaceLevel(unsigned int, struct IDirect3DSurface9 * *)
		//result = texture_->GetSurfaceLevel(0, &pSurface);
		int flag = 0;
		if (result < 0) break;

		pPlainSurface = nullptr;
		result = (*(int(__thiscall**)(DWORD, int, int, int, int, int, int*, DWORD))(*(DWORD*)pThis_7128_2 + 144))(// STDMETHOD(CreateOffscreenPlainSurface)(THIS_ UINT Width,UINT Height,D3DFORMAT Format,D3DPOOL Pool,IDirect3DSurface9** ppSurface,HANDLE* pSharedHandle) PURE;
			*(DWORD*)(*(DWORD*)pThis_7128_2 + 144),
			pThis_7128_2,
			nWidth,
			nHeight,
			pDesc.Format,
			D3DPOOL_SYSTEMMEM,
			(int*)&pPlainSurface,
			0);
		//result = d3d_device_->CreateOffscreenPlainSurface(nWidth, nHeight, pDesc.Format, D3DPOOL_SYSTEMMEM, &pPlainSurface, 0);

		if (result < 0)
		{
			flag = 19;
			break;
		}

		RECT rcSource = { 0, 0, nWidth, nHeight };
		if (!IntersectRect(&rcClipped, &rcSource, &rcClipped))
		{
			break;
		}
		result = (*(int(__thiscall**)(DWORD, int, int, int))(*(DWORD*)pThis_7128_2 + 128))(// STDMETHOD(GetRenderTargetData)(THIS_ IDirect3DSurface9* pRenderTarget,IDirect3DSurface9* pDestSurface) PURE;
			*(DWORD*)(*(DWORD*)pThis_7128_2 + 128),
			pThis_7128_2,
			(int)pSurface,
			(int)pPlainSurface);                  // public: virtual long __stdcall CBaseDevice::GetRenderTargetData(struct IDirect3DSurface9 *, struct IDirect3DSurface9 *)
		//result = d3d_device_->GetRenderTargetData(pSurface, pPlainSurface);

		if (result < 0)
		{
			break;
		}

		result = pPlainSurface->LockRect(&rcLock, &rcClipped, D3DLOCK_NOSYSLOCK);

		if (result < 0)
			break;

		int dataLen = kBytesPerPixel * nWidth * nHeight;
		//if(bResize)
		m_vcImage.reserve(dataLen);
		char* pBuf = (char*)&m_vcImage[0];
		int stride = nWidth * kBytesPerPixel;

		const char* rpixels = (const char*)rcLock.pBits;
		int rpitch = rcLock.Pitch;

		if (stride == rpitch)
		{
			memcpy(pBuf, rpixels, dataLen);
		}
		else
		{
			for (int i = 0; i < nHeight; ++i)
			{
				memcpy(pBuf, rpixels, stride);
				pBuf += stride;
				rpixels += rpitch;
			}
		}
		pPlainSurface->UnlockRect();


		//D3DFMT_R8G8B8 = 20,
		//D3DFMT_A8R8G8B8 = 21,

	} while (false);

	if (pSurface)
	{
		pSurface->Release();
		pSurface = nullptr;
	}

	if (pPlainSurface)
	{
		pPlainSurface->Release();
		pPlainSurface = nullptr;
	}
	return 0;
}
