/*************************************************************************************************
*
* File: MagnifierSample.cpp
*
* Description: Implements a simple control that magnifies the screen, using the 
* Magnification API.
*
* The magnification window is quarter-screen by default but can be resized.
* To make it full-screen, use the Maximize button or double-click the caption
* bar. To return to partial-screen mode, click on the application icon in the 
* taskbar and press ESC. 
*
* In full-screen mode, all keystrokes and mouse clicks are passed through to the
* underlying focused application. In partial-screen mode, the window can receive the 
* focus. 
*
* Multiple monitors are not supported.
*
* 
* Requirements: To compile, link to Magnification.lib. The sample must be run with 
* elevated privileges.
*
* The sample is not designed for multimonitor setups.
* 
*  This file is part of the Microsoft WinfFX SDK Code Samples.
* 
*  Copyright (C) Microsoft Corporation.  All rights reserved.
* 
* This source code is intended only as a supplement to Microsoft
* Development Tools and/or on-line documentation.  See these other
* materials for detailed information regarding Microsoft code samples.
* 
* THIS CODE AND INFORMATION ARE PROVIDED AS IS WITHOUT WARRANTY OF ANY
* KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
* PARTICULAR PURPOSE.
* 
*************************************************************************************************/

// Ensure that the following definition is in effect before winuser.h is included.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0603    
#endif

#include <windows.h>
#include <wincodec.h>
#include <magnification.h>
#include <fstream>

#include "CMagnifier.h"
// For simplicity, the sample uses a constant magnification factor.
#define MAGFACTOR  1.0f
#define RESTOREDWINDOWSTYLES WS_SIZEBOX | WS_SYSMENU | WS_CLIPCHILDREN | WS_CAPTION | WS_MAXIMIZEBOX

// Global variables and strings.
HINSTANCE           hInst;
const TCHAR         WindowClassName[] = TEXT("MagnifierWindow");
const TCHAR         PaintWindowClassName[] = TEXT("PaintMagnifierWindow");
const TCHAR         WindowTitle[]= TEXT("Screen Magnifier Sample");
const UINT          timerInterval = 16; // close to the refresh rate @60hz
HWND                hwndMag;
HWND                hwndHost;
RECT                magWindowRect;
RECT                hostWindowRect;

// Forward declarations.
ATOM                RegisterHostWindowClass(HINSTANCE hInstance);
BOOL                SetupMagnifier(HINSTANCE hinst);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
void CALLBACK       UpdateMagWindow(HWND hwnd, UINT uMsg, UINT_PTR idEvent, DWORD dwTime);
void                GoFullScreen();
void                GoPartialScreen();
//int                 CaptureFrameFromMagnifier();
BOOL                isFullScreen = FALSE;

CMagnifier          gMagnifier;
//HWND                gPaintWnd = NULL;
//
// FUNCTION: WinMain()
//
// PURPOSE: Entry point for the application.
//
int APIENTRY WinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE /*hPrevInstance*/,
                     _In_ LPSTR     /*lpCmdLine*/,
                     _In_ int       nCmdShow)
{
    if (FALSE == MagInitialize())
    {
        return 0;
    }
    if (FALSE == SetupMagnifier(hInstance))
    {
        return 0;
    }

	int i = nCmdShow;
	i = 0;
    ShowWindow(hwndHost, nCmdShow);

	SetWindowPos(hwndMag, NULL,
		magWindowRect.left, magWindowRect.top, magWindowRect.right, magWindowRect.bottom, 0);
    UpdateWindow(hwndHost);

    // Create a timer to update the control.
    UINT_PTR timerId = SetTimer(hwndHost, 0, timerInterval, UpdateMagWindow);

    // Main message loop.
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Shut down.
    KillTimer(NULL, timerId);
    MagUninitialize();
    return (int) msg.wParam;
}

//
// FUNCTION: HostWndProc()
//
// PURPOSE: Window procedure for the window that hosts the magnifier control.
//
LRESULT CALLBACK HostWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message) 
    {
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE)
        {
            if (isFullScreen) 
            {
                GoPartialScreen();
            }
        }
        break;

    case WM_SYSCOMMAND:
        if (GET_SC_WPARAM(wParam) == SC_MAXIMIZE)
        {
            GoFullScreen();
            //ShowWindow(hWnd, SW_HIDE);
        }
        else
        {
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    case WM_SIZE:
        if ( hwndMag != NULL )
        {
            GetClientRect(hWnd, &magWindowRect);
            // Resize the control to fill the window.
			SetWindowPos(hwndMag, NULL,
				magWindowRect.left, magWindowRect.top, magWindowRect.right, magWindowRect.bottom, 0);
			//SetWindowPos(hwndMag, NULL,
			//	0, 0, magWindowRect.right - magWindowRect.left, magWindowRect.bottom - magWindowRect.top, 0);
        }
        break;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;  
}

//
// FUNCTION: HostWndProc()
//
// PURPOSE: Window procedure for the window that hosts the magnifier control.
//
LRESULT CALLBACK PaintWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    //HDC hMemDC;
    //PAINTSTRUCT paint;
	switch (message)
	{
	case WM_PAINT:
    {
        //hMemDC = BeginPaint(hWnd, &paint);
        //if (hMemDC)
        //{
        //    //gMagnifier.MagnifierPaint(hMemDC);
        //}
        //EndPaint(hWnd, &paint);
        break;
    }
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}
//
//  FUNCTION: RegisterHostWindowClass()
//
//  PURPOSE: Registers the window class for the window that contains the magnification control.
//
ATOM RegisterHostWindowClass(HINSTANCE hInstance)
{
    WNDCLASSEX wcex = {};

    wcex.cbSize = sizeof(WNDCLASSEX); 
    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = HostWndProc;
    wcex.hInstance      = hInstance;
    wcex.hCursor        = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(1 + COLOR_BTNFACE);
    wcex.lpszClassName  = WindowClassName;

    return RegisterClassEx(&wcex);
}


ATOM RegisterPaintWindowClass(HINSTANCE hInstance)
{
	WNDCLASSEX wcex = {};

	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = PaintWndProc;
	wcex.hInstance = hInstance;
	wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(1 + COLOR_BTNFACE);
	wcex.lpszClassName = PaintWindowClassName;

	return RegisterClassEx(&wcex);
}

#define UNUSED(a) (void*)(a)

BOOL CALLBACK OnMagImageScalingCallback(
	HWND hwnd,
	void* srcdata,
	MAGIMAGEHEADER srcheader,
	void* destdata,
	MAGIMAGEHEADER destheader,
	RECT unclipped,
	RECT clipped,
	HRGN dirty) {

    UNUSED(hwnd);
    UNUSED(srcdata);
    UNUSED(&srcheader);
    UNUSED(destdata);
    UNUSED(&destheader);
    UNUSED(&unclipped);
    UNUSED(&clipped);
    UNUSED(dirty);
    int i = 0;
    i = 1;
	return TRUE;
}

//
// FUNCTION: SetupMagnifier
//
// PURPOSE: Creates the windows and initializes magnification.
//
#include <iostream>
BOOL SetupMagnifier(HINSTANCE hinst)
{
    // Set bounds of host window according to screen size.
    //hostWindowRect.top = 0;
    //hostWindowRect.bottom = GetSystemMetrics(SM_CYSCREEN) / 4;  // top quarter of screen
    //hostWindowRect.left = GetSystemMetrics(SM_CXSCREEN);
    //hostWindowRect.right = hostWindowRect.left + GetSystemMetrics(SM_CXSCREEN);

	// 获取桌面x坐标，可以为负值
	int xScreen = ::GetSystemMetrics(SM_XVIRTUALSCREEN);
	std::cout << "x坐标：" << xScreen << std::endl;
	// 获取桌面y坐标，可以为负值
	int yScreen = ::GetSystemMetrics(SM_YVIRTUALSCREEN);
	std::cout << "y坐标：" << yScreen << std::endl;
	// 获取桌面总宽度
	int cxScreen = ::GetSystemMetrics(SM_CXVIRTUALSCREEN);
	std::cout << "总宽度：" << cxScreen << std::endl;
	// 获取桌面总高度
	int cyScreen = ::GetSystemMetrics(SM_CYVIRTUALSCREEN);
	std::cout << "高度：" << cyScreen << std::endl;
	//获取屏幕数量
	int nScreenCount = ::GetSystemMetrics(SM_CMONITORS);
	std::cout << "屏幕数量：" << nScreenCount << std::endl;

	hostWindowRect.top = yScreen;
	hostWindowRect.bottom = cyScreen/2 ;  // top quarter of screen
	hostWindowRect.left = xScreen;
	hostWindowRect.right = cxScreen /2;

    // Create the host window.
    RegisterHostWindowClass(hinst);
    hwndHost = CreateWindowEx(WS_EX_TOPMOST | WS_EX_LAYERED, 
        WindowClassName, WindowTitle, 
        RESTOREDWINDOWSTYLES,
        hostWindowRect.left, hostWindowRect.top, hostWindowRect.right - hostWindowRect.left, hostWindowRect.bottom - hostWindowRect.top, NULL, NULL, hInst, NULL);
    if (!hwndHost)
    {
        return FALSE;
    }

    SetWindowPos(hwndHost, NULL, hostWindowRect.left, hostWindowRect.top, hostWindowRect.right - hostWindowRect.left, hostWindowRect.bottom - hostWindowRect.top, FALSE);
    // Make the window opaque.
    SetLayeredWindowAttributes(hwndHost, 0, 255, LWA_ALPHA);

    // Create a magnifier control that fills the client area.
    GetClientRect(hwndHost, &magWindowRect);
    hwndMag = CreateWindow(WC_MAGNIFIER, TEXT("MagnifierWindow"), 
        WS_CHILD | MS_SHOWMAGNIFIEDCURSOR | WS_VISIBLE,
        magWindowRect.left, magWindowRect.top, magWindowRect.right, magWindowRect.bottom, hwndHost, NULL, hInst, NULL );
    if (!hwndMag)
    {
        return FALSE;
    }

 //   RegisterPaintWindowClass(hinst);
	//gPaintWnd = CreateWindow(WC_MAGNIFIER, PaintWindowClassName, WS_VISIBLE,
	//	magWindowRect.left, magWindowRect.top, magWindowRect.right, magWindowRect.bottom, hwndHost, NULL, hInst, NULL);
 //   ShowWindow(gPaintWnd, SW_NORMAL);

    std::vector<HWND> vcFilter = { (HWND)0x006F1064 };
    //gMagnifier.setFilterList(vcFilter);
    //gMagnifier.setPaintWindow(gPaintWnd);

    // Set the magnification factor.
    MAGTRANSFORM matrix;
    memset(&matrix, 0, sizeof(matrix));
    matrix.v[0][0] = MAGFACTOR;
    matrix.v[1][1] = MAGFACTOR;
    matrix.v[2][2] = 1.0f;

    BOOL ret = MagSetWindowTransform(hwndMag, &matrix);

    if (ret)
    {
        MAGCOLOREFFECT magEffectInvert = 
        {{ // MagEffectInvert
            { -1.0f,  0.0f,  0.0f,  0.0f,  0.0f },
            {  0.0f, -1.0f,  0.0f,  0.0f,  0.0f },
            {  0.0f,  0.0f, -1.0f,  0.0f,  0.0f },
            {  0.0f,  0.0f,  0.0f,  1.0f,  0.0f },
            {  1.0f,  1.0f,  1.0f,  0.0f,  1.0f } 
        }};

        //ret = MagSetColorEffect(hwndMag,&magEffectInvert);
    }
	//HWND fWnd = (HWND)0x000911D0;
	ret = MagSetWindowFilterList(hwndMag, MW_FILTERMODE_EXCLUDE, 1, (HWND*)(&vcFilter[0]));

    //ret = MagSetImageScalingCallback(hwndMag, &OnMagImageScalingCallback);
    return ret;  
    //return TRUE;
}


//
// FUNCTION: GoFullScreen()
//
// PURPOSE: Makes the host window full-screen by placing non-client elements outside the display.
//
void GoFullScreen()
{
    isFullScreen = TRUE;
    // The window must be styled as layered for proper rendering. 
    // It is styled as transparent so that it does not capture mouse clicks.
    SetWindowLong(hwndHost, GWL_EXSTYLE, WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT);
    // Give the window a system menu so it can be closed on the taskbar.
    SetWindowLong(hwndHost, GWL_STYLE,  WS_CAPTION | WS_SYSMENU);

    // Calculate the span of the display area.
    HDC hDC = GetDC(NULL);
    int xSpan = GetSystemMetrics(SM_CXSCREEN);
    int ySpan = GetSystemMetrics(SM_CYSCREEN);
    ReleaseDC(NULL, hDC);

	// Calculate the size of system elements.
	//int xBorder = GetSystemMetrics(SM_CXFRAME);
	//int yCaption = GetSystemMetrics(SM_CYCAPTION);
	//int yBorder = GetSystemMetrics(SM_CYFRAME);

	// Calculate the window origin and span for full-screen mode.
	//int xOrigin = -xBorder;
	//int yOrigin = -yBorder - yCaption;
	int xOrigin = 0;
	int yOrigin = 0;
	//xSpan += 2 * xBorder;
	//ySpan += 2 * yBorder + yCaption;

	SetWindowPos(hwndHost, HWND_TOPMOST, xOrigin, yOrigin, xSpan, ySpan,
		SWP_SHOWWINDOW | SWP_NOZORDER | SWP_NOACTIVATE);
}

//
// FUNCTION: GoPartialScreen()
//
// PURPOSE: Makes the host window resizable and focusable.
//
void GoPartialScreen()
{
    isFullScreen = FALSE;

    SetWindowLong(hwndHost, GWL_EXSTYLE, WS_EX_TOPMOST | WS_EX_LAYERED);
    SetWindowLong(hwndHost, GWL_STYLE, RESTOREDWINDOWSTYLES);
    SetWindowPos(hwndHost, HWND_TOPMOST, 
        hostWindowRect.left, hostWindowRect.top, hostWindowRect.right, hostWindowRect.bottom, 
        SWP_SHOWWINDOW | SWP_NOZORDER | SWP_NOACTIVATE);
}



int CaptureFrameFromMagnifier()
{
    int result = 0;
    HDC pDC;// 源DC
    HWND hwndTarget = (HWND)0x00030E88;

    static bool bTest = false;
    if (bTest)
    {
        static HWND hwnd = hwndTarget;
        hwndTarget = hwnd;
    }
	pDC = ::GetDC(hwndTarget);//获取屏幕DC(0为全屏，句柄则为窗口)

	RECT rc = { 0 };
	GetClientRect(hwndTarget, &rc);

	int kBytesPerPixel = 4;

    //int nTop = rc.top;
    //int nLeft = rc.left;
	int nWidth = rc.right - rc.left;
	int nHeight = rc.bottom - rc.top;
	int bytes_per_row = nWidth * kBytesPerPixel;
	int buffer_size = bytes_per_row * nHeight; /*bmp.bmWidthBytes* bmp.bmHeight*/

    //int BitPerPixel = ::GetDeviceCaps(pDC, BITSPIXEL);//获得颜色模式
    //if (width == 0 && height == 0)//默认宽度和高度为全屏
    //{
    //    width = ::GetDeviceCaps(pDC, HORZRES); //设置图像宽度全屏
    //    height = ::GetDeviceCaps(pDC, VERTRES); //设置图像高度全屏
    //}
    HDC memDC;//内存DC
    memDC = ::CreateCompatibleDC(pDC);
    HBITMAP memBitmap, oldmemBitmap;//建立和屏幕兼容的bitmap
    memBitmap = ::CreateCompatibleBitmap(pDC, nWidth, nHeight);
	oldmemBitmap = (HBITMAP)::SelectObject(memDC, memBitmap);//将memBitmap选入内存DC
	//BitBlt(memDC, 0, 0, nWidth, nHeight, pDC, nLeft, nTop, SRCCOPY);//图像宽度高度和截取位置
    
	const auto ret = ::PrintWindow(hwndTarget, memDC, PW_CLIENTONLY | PW_RENDERFULLCONTENT);

    if (ret)
    {
        int i = 0;
        i = 1;
    }
    //if (hwndTarget == ::GetDesktopWindow())
    //{
    //    BitBlt(memDC, 0, 0, nWidth, nHeight, pDC, nLeft, nTop, SRCCOPY);//图像宽度高度和截取位置
    //}
    //else
    //{
    //    bool bret = ::PrintWindow(hwndTarget, memDC, PW_CLIENTONLY);
    //    if (!bret)
    //    {
    //        BitBlt(memDC, 0, 0, nWidth, nHeight, pDC, nLeft, nTop, SRCCOPY);//图像宽度高度和截取位置
    //    }
    //}
    //以下代码保存memDC中的位图到文件
    BITMAP bmp;
    ::GetObject(memBitmap, sizeof(BITMAP), &bmp);;//获得位图信息

	std::ofstream file("d:\\xxxxtest.bmp");//打开文件用于写,若文件不存在就创建它
	if (!file)
        return -1; //打开文件失败则结束运行


    BITMAPINFOHEADER bih = { 0 };//位图信息头
    bih.biBitCount = bmp.bmBitsPixel;//每个像素字节大小
    bih.biCompression = BI_RGB;
    bih.biHeight = bmp.bmHeight;//高度
    bih.biPlanes = 1;
    bih.biSize = sizeof(BITMAPINFOHEADER);
    bih.biSizeImage = bmp.bmWidthBytes * bmp.bmHeight;//图像数据大小
    bih.biWidth = bmp.bmWidth;//宽度

    BITMAPFILEHEADER bfh = { 0 };//位图文件头
    bfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);//到位图数据的偏移量
    bfh.bfSize = bfh.bfOffBits + bmp.bmWidthBytes * bmp.bmHeight;//文件总的大小
    bfh.bfType = (WORD)0x4d42;

	file.write((const char*)&bfh, sizeof(BITMAPFILEHEADER));
	file.write((const char*)&bih, sizeof(BITMAPINFOHEADER));
    //fwrite(&bfh, 1, sizeof(BITMAPFILEHEADER), fp);//写入位图文件头
    //fwrite(&bih, 1, sizeof(BITMAPINFOHEADER), fp);//写入位图信息头
    byte* p = new byte[buffer_size];//申请内存保存位图数据
    GetDIBits(memDC, (HBITMAP)memBitmap, 0, nHeight, p,
		(LPBITMAPINFO)&bih, DIB_RGB_COLORS);//获取位图数据
	file.write((const char*)p, bmp.bmWidthBytes * bmp.bmHeight);
	//fwrite(p, 1, bmp.bmWidthBytes * bmp.bmHeight, fp);//写入位图数据
    delete[] p;
    file.close();
    //fclose(fp);
    //HWND sBitHwnd = GetDlgItem(g_Hwnd, IDC_STATIC_IMG);
    /*返回内存中的位图句柄 还原原来的内存DC位图句柄 不能直接用 memBitmap我测试好像是不行不知道为什么*/
    HBITMAP oleImage = (HBITMAP)::SelectObject(memDC, oldmemBitmap);
    //oleImage = (HBITMAP)SendMessage(sBitHwnd, STM_SETIMAGE, IMAGE_BITMAP, (LPARAM)oleImage);
//#if 0
//    /*这种方法也能把位图显示到picture 控件上*/
//    HDC bitDc = NULL;
//    bitDc = ::GetDC(sBitHwnd);
//    BitBlt(bitDc, 0, 0, bmp.bmWidth, bmp.bmHeight, memDC, 0, 0, SRCCOPY); //内存DC映射到屏幕DC
//    ReleaseDC(sBitHwnd, bitDc);
//    /*如果需要把位图转换*/
//    /*
//    CImage image;
//    image.Create(nWidth, nHeight, nBitPerPixel);
//    BitBlt(image.GetDC(), 0, 0, nWidth, nHeight, hdcSrc, 0, 0, SRCCOPY);
//    ::ReleaseDC(NULL, hdcSrc);
//    image.ReleaseDC();
//    image.Save(path, Gdiplus::ImageFormatPNG);//ImageFormatJPEG
//    */
//#endif
    DeleteObject(memBitmap);
    DeleteObject(oleImage);
    DeleteDC(memDC);
    ReleaseDC(hwndTarget, pDC);


	return result;
}


//
// FUNCTION: UpdateMagWindow()
//
// PURPOSE: Sets the source rectangle and updates the window. Called by a timer.
//
void CALLBACK UpdateMagWindow(HWND /*hwnd*/, UINT /*uMsg*/, UINT_PTR /*idEvent*/, DWORD /*dwTime*/)
{
	POINT mousePoint;
	GetCursorPos(&mousePoint);

	int width = (int)((magWindowRect.right - magWindowRect.left) / MAGFACTOR);
	//int height = (int)((magWindowRect.bottom - magWindowRect.top) / MAGFACTOR);
	RECT sourceRect;
	//sourceRect.left = mousePoint.x - width / 2;
	//sourceRect.top = mousePoint.y - height / 2;

	//// Don't scroll outside desktop area.
	//if (sourceRect.left < 0)
	//{
	//	sourceRect.left = 0;
	//}
	//if (sourceRect.left > GetSystemMetrics(SM_CXSCREEN) - width)
	//{
	//	sourceRect.left = GetSystemMetrics(SM_CXSCREEN) - width;
	//}
	//sourceRect.right = sourceRect.left + width;

	//if (sourceRect.top < 0)
	//{
	//	sourceRect.top = 0;
	//}
	//if (sourceRect.top > GetSystemMetrics(SM_CYSCREEN) - height)
	//{
	//	sourceRect.top = GetSystemMetrics(SM_CYSCREEN) - height;
	//}
	//sourceRect.bottom = sourceRect.top + height;

	//sourceRect.left += GetSystemMetrics(SM_CXSCREEN);
	//sourceRect.right = sourceRect.left + width;

	// Set the source rectangle for the magnifier control.

	RECT rc = { 0 };
    //GetWindowRect(hwndHost, &rc);
    //GetWindowRect(hwndHost, &sourceRect);
	GetWindowRect(hwndMag, &sourceRect);
	sourceRect.left += GetSystemMetrics(SM_CXSCREEN);
	sourceRect.right = sourceRect.left + width;
    //POINT point = { 0 };
    //ClientToScreen(hwndMag, &point);
    //OffsetRect(&sourceRect, -point.x, -point.y);

	//__try {
		MagSetWindowSource(hwndMag, sourceRect);


        //UpdateWindow(gPaintWnd);
        // 
		//HDC hMemDC = GetDC(gPaintWnd);
		//gMagnifier.MagnifierPaint(hMemDC);
		//ReleaseDC(NULL, hMemDC);

	//}
	//__except (EXCEPTION_EXECUTE_HANDLER)
	//{
	//	int i = 0;
	//	i = 1;
	//}


	gMagnifier.setCaptureRect(sourceRect);
	gMagnifier.DoCapture(hwndMag);
	static int count = 0;

	auto data = gMagnifier.GetImage();
	if (count % 100 == 0 && data.size() > 0)
	{
		std::ofstream file("d:\\xxxxtest.bmp");//打开文件用于写,若文件不存在就创建它
		if (!file)
			return; //打开文件失败则结束运行

		int nWidth = sourceRect.right - sourceRect.left;
		int nHeight = sourceRect.bottom - sourceRect.top;

		BITMAPINFOHEADER bih = { 0 };//位图信息头
		bih.biBitCount = 4;//每个像素字节大小
		bih.biCompression = BI_RGB;
		bih.biHeight = nHeight;//高度
		bih.biPlanes = 1;
		bih.biSize = sizeof(BITMAPINFOHEADER);
		bih.biSizeImage = 4 * nWidth * nHeight;//图像数据大小
		bih.biWidth = nWidth;//宽度

		BITMAPFILEHEADER bfh = { 0 };//位图文件头
		bfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);//到位图数据的偏移量
		bfh.bfSize = bfh.bfOffBits + 4 * nWidth * nHeight;//文件总的大小
		bfh.bfType = (WORD)0x4d42;

		file.write((const char*)&bfh, sizeof(BITMAPFILEHEADER));
		file.write((const char*)&bih, sizeof(BITMAPINFOHEADER));
		//fwrite(&bfh, 1, sizeof(BITMAPFILEHEADER), fp);//写入位图文件头
		//fwrite(&bih, 1, sizeof(BITMAPINFOHEADER), fp);//写入位图信息头
		file.write((const char*)&data[0], 4 * nWidth * nHeight);
		//fwrite(p, 1, bmp.bmWidthBytes * bmp.bmHeight, fp);//写入位图数据

		file.close();
	}
	else
	{
		int i = 0;
		i = count;
	}
	//// Reclaim topmost status, to prevent unmagnified menus from remaining in view. 
	//SetWindowPos(hwndHost, HWND_TOPMOST, 0, 0, 0, 0,
	//	SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE);

	// Force redraw.
	InvalidateRect(hwndMag, NULL, TRUE);
	//test
	//CaptureFrameFromMagnifier();
	{

    }
}
//
//#include <iostream>
//#include <Windows.h>
//#include <GdiPlus.h> // 保存图片用到了GDI+
//#include <windowsx.h>
//#include <atlbase.h> // 字符串转换用到
//#include <string>
//#include <functional>
//#include <vector>
//#include <memory>
//#include <dwmapi.h>
//
//#pragma comment(lib, "gdiplus.lib") // 保存图片需要
//#pragma comment(lib, "Dwmapi.lib")  // 判断是否是隐形窗口以及获取窗口大小会用到
//
//// 为了将屏幕和窗口进行统一,因此使用了结构体
//struct WindowInfo
//{
//	HWND hwnd; /* 为空表示屏幕截图 */
//	std::string desc; // 窗口标题
//	RECT rect{ 0,0,0,0 }; /* hwnd不为空时,此参数无效 */
//	void* tempPointer = nullptr;
//};
//
//namespace GdiplusUtil
//{
//	static int GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
//	{
//		UINT  num = 0;          // number of image encoders
//		UINT  size = 0;         // size of the image encoder array in bytes
//
//		Gdiplus::ImageCodecInfo* pImageCodecInfo = NULL;
//
//		Gdiplus::GetImageEncodersSize(&num, &size);
//		if (size == 0)
//			return -1;  // Failure
//
//		pImageCodecInfo = (Gdiplus::ImageCodecInfo*)(malloc(size));
//		if (pImageCodecInfo == NULL)
//			return -1;  // Failure
//
//		Gdiplus::GetImageEncoders(num, size, pImageCodecInfo);
//
//		for (UINT j = 0; j < num; ++j) {
//			if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0) {
//				*pClsid = pImageCodecInfo[j].Clsid;
//				free(pImageCodecInfo);
//				return j;  // Success
//			}
//		}
//
//		free(pImageCodecInfo);
//		return -1;  // Failure
//	}
//
//	// 将bitmap对象保存为png图片
//	static bool SaveBitmapAsPng(const std::shared_ptr<Gdiplus::Bitmap>& bitmap, const std::string& filename)
//	{
//		if (bitmap == nullptr) return false;
//		CLSID png_clsid;
//		WCHAR type[] = L"image/png";
//		GetEncoderClsid(type, &png_clsid);
//		Gdiplus::Status ok = bitmap->Save(CA2W(filename.c_str(), CP_ACP), &png_clsid, nullptr);
//		return ok == Gdiplus::Status::Ok;
//	}
//}
//
//class WindowCapture
//{
//public:
//	using BitmapPtr = std::shared_ptr<Gdiplus::Bitmap>;
//
//	static BitmapPtr Capture(const WindowInfo& wnd_info)
//	{
//		HDC hWndDC = GetWindowDC(wnd_info.hwnd);
//		RECT capture_rect{ 0,0,0,0 }; // 最终要截取的区域
//		RECT wnd_rect; // 窗口区域
//		RECT real_rect; // 真实的窗口区域,实际上也不是百分百准确
//
//		if (wnd_info.hwnd) {
//			::GetWindowRect(wnd_info.hwnd, &wnd_rect);
//			DwmGetWindowAttribute(wnd_info.hwnd, DWMWINDOWATTRIBUTE::DWMWA_EXTENDED_FRAME_BOUNDS, &real_rect, sizeof(RECT));
//			int offset_left = real_rect.left - wnd_rect.left;
//			int offset_top = real_rect.top - wnd_rect.top;
//			capture_rect = RECT{ offset_left,offset_top,real_rect.right - real_rect.left + offset_left,real_rect.bottom - real_rect.top + offset_top };
//		}
//		else {
//			capture_rect = wnd_info.rect;
//		}
//
//		int width = capture_rect.right - capture_rect.left;
//		int height = capture_rect.bottom - capture_rect.top;
//
//		HDC hMemDC = CreateCompatibleDC(hWndDC);
//		HBITMAP hBitmap = CreateCompatibleBitmap(hWndDC, width, height);
//		SelectObject(hMemDC, hBitmap);
//
//		BitmapPtr bitmap;
//		// 获取指定区域的rgb数据
//		bool ok = BitBlt(hMemDC, 0, 0, width, height, hWndDC, capture_rect.left, capture_rect.top, SRCCOPY);
//		// hBitmap就是得到的图片对象,转GDI的Bitmap进行保存
//		if (ok) bitmap = std::make_shared<Gdiplus::Bitmap>(hBitmap, nullptr);
//
//		DeleteDC(hWndDC);
//		DeleteDC(hMemDC);
//		DeleteObject(hBitmap);
//
//		return bitmap;
//	}
//};
//
//class Enumerator
//{
//public:
//	using EnumCallback = std::function<void(const WindowInfo&)>;
//
//	static bool EnumMonitor(EnumCallback callback)
//	{
//		// 调用Win32Api进行显示器遍历
//		return ::EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, (LPARAM)&callback);
//	}
//
//	static bool EnumWindow(EnumCallback callback)
//	{
//		// 调用Win32Api进行窗口遍历
//		return ::EnumWindows(EnumWindowsProc, (LPARAM)&callback);
//	}
//
//private:
//	static BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam)
//	{
//		//::GetParent获取的有可能是所有者窗口,因此使用GetAncestor获取父窗口句柄
//		HWND parent = ::GetAncestor(hwnd, GA_PARENT);
//		HWND desktop = ::GetDesktopWindow(); // 获取桌面的句柄
//		TCHAR szTitle[MAX_PATH] = { 0 };
//		::GetWindowText(hwnd, szTitle, MAX_PATH); // 获取标题
//
//		// 排除父窗口不是桌面的
//		if (parent != nullptr && parent != desktop) return TRUE;
//
//		// 排除标题为空的
//		if (strcmp(szTitle, "") == 0) return TRUE;
//
//		// 排除最小化窗口(因为获取最小化窗口的区域数据是不对的,因此也没办法进行截图等操作)
//		if (::IsIconic(hwnd)) return TRUE;
//
//		// 排除不可见窗口,被其他窗口遮挡的情况是可见的
//		if (!::IsWindowVisible(hwnd)) return TRUE;
//
//		// 排除对用户隐形的窗口,参考[https://docs.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute]
//		DWORD flag = 0;
//		DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &flag, sizeof(flag));
//		if (flag) return TRUE;
//
//		if (lParam) {
//			WindowInfo wnd_info{ hwnd,(LPCSTR)CT2A(szTitle, CP_ACP) };
//			EnumCallback* callback_ptr = reinterpret_cast<EnumCallback*>(lParam);
//			callback_ptr->operator()(wnd_info);
//		}
//		return TRUE;
//	}
//
//	static BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData)
//	{
//		MONITORINFOEX mi;
//		lprcMonitor = nullptr;
//		hdcMonitor = nullptr;
//		mi.cbSize = sizeof(MONITORINFOEX);
//		GetMonitorInfo(hMonitor, &mi);
//		if (dwData) {
//			std::string device_name = (LPCSTR)CT2A(mi.szDevice, CP_ACP);
//			if (mi.dwFlags == MONITORINFOF_PRIMARY) device_name += "(Primary)"; // 主显示器,可根据需要进行操作
//			WindowInfo wnd_info{ nullptr, device_name, mi.rcMonitor };
//
//			EnumCallback* callback = reinterpret_cast<EnumCallback*>(dwData);
//			(*callback)(wnd_info);
//		}
//		return TRUE;
//	}
//};
//
//namespace TestCase
//{
//	void Run()
//	{
//		std::vector<WindowInfo> window_vec; // 用来保存窗口信息
//		// 枚举显示器
//		Enumerator::EnumMonitor([&window_vec](const WindowInfo& wnd_info)
//			{
//				window_vec.push_back(wnd_info);
//			});
//		// 计算生成所有屏幕加在一起的区域大小
//		if (window_vec.size() > 0) { // 也可大于1,这样只有一个显示器时不会显示全屏选项
//			int width = 0, height = 0;
//			for (const auto& wnd_info : window_vec) {
//				width += wnd_info.rect.right - wnd_info.rect.left;
//				int h = wnd_info.rect.bottom - wnd_info.rect.top;
//				if (h > height) height = h; // 高度可能不一样,需要以最高的为准
//			}
//			WindowInfo wnd_info{ nullptr, "FullScreen", { 0, 0, width, height} };
//			window_vec.push_back(wnd_info);
//		}
//		// 枚举窗口
//		Enumerator::EnumWindow([&window_vec](const WindowInfo& wnd_info)
//			{
//				window_vec.push_back(wnd_info);
//			});
//		// 示例: 遍历找到的所有窗口,将每一个都截图到指定路径,文件夹需存在,程序不会自己创建文件夹
//		int cnt = 1;
//
//		for (const auto& window : window_vec) {
//			printf("%2d. %s\n", cnt, window.desc.c_str());
//
//
//
//
//			auto bitmap = WindowCapture::Capture(window);
//			if (bitmap) GdiplusUtil::SaveBitmapAsPng(bitmap, std::to_string(cnt) + ".png");
//			++cnt;
//		}
//	}
//}
//
//int main()
//{
//	/************************ GDI+ 初始化 ***************************/
//	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
//	ULONG_PTR token;
//	Gdiplus::GdiplusStartup(&token, &gdiplusStartupInput, NULL);
//	/***********************************************************/
//
//	TestCase::Run();
//
//	Gdiplus::GdiplusShutdown(token); // 关闭GDI
//	return 0;
//}
