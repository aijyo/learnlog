#pragma once

#include <windows.h>
#include <wincodec.h>
#include <magnification.h>

#include <vector>

class CMagnifier
{
public:
	CMagnifier();
	~CMagnifier();

	void setFilterList(const std::vector<HWND>& vcFilter, bool bExclude = true);
	void setCaptureRect(const RECT& pos);
	void setPaintWindow(HWND hwnd);


	int DoCapture(HWND hMagnifierWindow);
	int DoCaptureWin11(HWND hMagnifierWindow);
	HDC MagnifierPaint(HDC magnifierWndDC);
	const std::vector<std::uint8_t>& GetImage() { return m_vcImage; }
protected:
	HBITMAP RemakeOsBitmaps(HDC hScreenDC);
	unsigned int RunFilterList(HDC hMemDC, int width, int height, int left, int top);
	BOOL DrawCursor(HDC hDC, int a2, int a3);
	BOOL CutOut(HWND hwnd, HDC a2, HDC hdcSrc);

	int xFilterTextureD3D9(int pThis, unsigned int index, DWORD& bContinue);
private:
	HWND m_hMagnifierWnd;
	RECT m_rcCapture;
	HBITMAP m_hMemBitmap;
	int m_nWidth;
	int m_nHeight;
//#define MW_FILTERMODE_EXCLUDE   0
//#define MW_FILTERMODE_INCLUDE   1
	bool m_bFilterExclude;					//
	std::vector<HWND> m_vcFilterWnd;

	//m11 m12
	//m21 m22
	//dx dy
	float m_transforMatrix[6];

	// data
	std::vector<std::uint8_t> m_vcImage;
	 
};

