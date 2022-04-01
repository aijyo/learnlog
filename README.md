# learnlog
get window covered region:

#include <windows.h>
#include <dwmapi.h>

#include <QPainter>

using wnd_t = HWND;

bool IsWindowCovered(wnd_t hwnd, QRect rect)
{
	if (!hwnd) {
		return false;
	}
	bool bIsCovered = false;
	RECT hwndRect = { 0,0,0,0 };
	if (rect.isEmpty()) {
		::GetWindowRect(hwnd, &hwndRect);
	}
	else {
		hwndRect.left = rect.left();
		hwndRect.right = rect.right();
		hwndRect.top = rect.top();
		hwndRect.bottom = rect.bottom();
	}

	HRGN rgn = ::CreateRectRgn(hwndRect.left,
		hwndRect.top,
		hwndRect.right,
		hwndRect.bottom);


	HWND desktopWin = GetDesktopWindow();
	HWND hParentWnd = ::GetAncestor(hwnd, GA_PARENT);
	HWND hChildWnd = hwnd;
	while (hChildWnd != nullptr) {
		HWND topWnd = ::GetTopWindow(hParentWnd);
		do {
			if (topWnd == hChildWnd) { //���hChildWnd�Ѿ��ڵ�ǰz�򶥲�����Ͳ����ٱ�����
				break;
			}
			RECT topWndRect = { 0,0,0,0 };
			::GetWindowRect(topWnd, &topWndRect);
			RECT tempRect = { 0,0,0,0 };
			//�ɼ���������С�����������洰�ڣ���Ҫ�ж��Ƿ��ڵ��Ĵ������ཻ���� ���п�����סĿ�괰��
			if (::IsWindowVisible(topWnd)
				&& topWnd != desktopWin
				&& !::IsIconic(topWnd)
				&& IntersectRect(&tempRect, &topWndRect, &hwndRect) != 0)
			{
				BYTE alpha;
				BOOL success = GetLayeredWindowAttributes(topWnd, nullptr, &alpha, nullptr);
				if (success && alpha == 0) {
					topWnd = GetNextWindow(topWnd, GW_HWNDNEXT);
					continue;
				}
				HWND ownedWin = GetWindow(topWnd, GW_OWNER);

				if ((ownedWin != nullptr) || (ownedWin == nullptr && hParentWnd == desktopWin)) {
					char className[1024] = { 0 };
					char winTitle[1024] = { 0 };
					GetClassNameA(topWnd, className, 1024);
					GetWindowTextA(topWnd, winTitle, 1024);

					// work with desktop window
					if (strcmp(className, "WorkerW")
						&& strcmp(className, "Progman")) {
						if (strcmp(className, "Windows.UI.Core.CoreWindow")
							|| strcmp(className, "ApplicationFrameWindow")) {
							int pvAttr = 0;
							DwmGetWindowAttribute(topWnd, DWMWA_CLOAKED, &pvAttr, 4);
							// equals 0 means the window is visible
							if (pvAttr != 0) {
								topWnd = GetNextWindow(topWnd, GW_HWNDNEXT);
								continue;
							}
						}
						if (!QString("2333").isEmpty()) 
						{
							//LOG(INFO) << " window title: " << winTitle
							//	<< " window class: " << className
							//	<< " HWND: " << topWnd
							//	<< "rect: (" << topWndRect.left
							//	<< "," << topWndRect.top
							//	<< "," << topWndRect.right
							//	<< "," << topWndRect.bottom
							//	<< ")";
						}
						HRGN topWndRgn = ::CreateRectRgn(topWndRect.left, topWndRect.top, topWndRect.right, topWndRect.bottom);
						int cbRet = ::CombineRgn(rgn, rgn, topWndRgn, RGN_DIFF);
						DeleteObject(topWndRgn);
					}
				}
			}
			topWnd = GetNextWindow(topWnd, GW_HWNDNEXT);
		} while (topWnd != nullptr);

		hChildWnd = hParentWnd;
		hParentWnd = ::GetAncestor(hParentWnd, GA_PARENT);
		if (hChildWnd == GetDesktopWindow()) {
			break;
		}
	}
	DWORD uRegionSize = GetRegionData(rgn, sizeof(RGNDATA), NULL);  // Send NULL request to get the storage size
	char* pRawRgnData = new char[uRegionSize];
	RGNDATA* pRgnData = (RGNDATA*)pRawRgnData;   // Allocate space for the region data
	DWORD uSizeCheck = GetRegionData(rgn, uRegionSize, pRgnData);

	QPainter painter;
	if (uSizeCheck == uRegionSize) {
		DWORD nCnt = pRgnData->rdh.nCount;
		if (nCnt == 0) {
			bIsCovered = true;
		}
		else {

			HDC hdc = GetWindowDC(desktopWin);  // ��ȡһ���ɹ���ͼ��DC���������ֱ������������
			HPEN hpen1 = CreatePen(PS_SOLID, 1, RGB(255, 0, 0)); // ������ɫ1���ؿ��ȵ�ʵ�߻���
			HBRUSH hbrush1 = CreateSolidBrush(RGB(0, 0, 255));     // ����һ��ʵ����ɫ��ˢ
			// ��hpen1��hbrush1ѡ��HDC��������HDCԭ���Ļ��ʺͻ�ˢ
			HPEN hpen_old = (HPEN)SelectObject(hdc, hpen1);
			HBRUSH hbrush_old = (HBRUSH)SelectObject(hdc, hbrush1);
			//PaintRgn(hdc, rgn);
			FrameRgn(hdc, rgn, hbrush1, 2, 2);
			// �ָ�ԭ���Ļ��ʺͻ�ˢ
			SelectObject(hdc, hpen_old);
			SelectObject(hdc, hbrush_old);
		}
	}
	DeleteObject(rgn);
	return bIsCovered;
}

int desktopDraw()
{

	HDC hdc = GetWindowDC(GetDesktopWindow());  // ��ȡһ���ɹ���ͼ��DC���������ֱ������������

	HPEN hpen1 = CreatePen(PS_SOLID, 1, RGB(255, 0, 0)); // ������ɫ1���ؿ��ȵ�ʵ�߻���
	//������ɫ5���ؿ��ȵ����ۻ��ʣ�������봴����������Ļ��������MSDN
	HPEN hpen2 = CreatePen(PS_DASH, 5, RGB(0, 255, 0));
	HBRUSH hbrush1 = CreateSolidBrush(RGB(0, 0, 255));     // ����һ��ʵ����ɫ��ˢ

	HBRUSH hbrush2 = (HBRUSH)GetStockObject(NULL_BRUSH);// ����һ��͸���Ļ�ˢ




	// ��hpen1��hbrush1ѡ��HDC��������HDCԭ���Ļ��ʺͻ�ˢ
	HPEN hpen_old = (HPEN)SelectObject(hdc, hpen1);
	HBRUSH hbrush_old = (HBRUSH)SelectObject(hdc, hbrush1);



	Rectangle(hdc, 40, 30, 40 + 200, 30 + 50);// ��(40,30)����һ����200���أ���50���صľ���
	SelectObject(hdc, hpen2);  // ��hpen1��hbrush1��Ȼ����(40,100)��Ҳ��һ�����Σ������кβ��
	SelectObject(hdc, hbrush2);
	Rectangle(hdc, 40, 100, 40 + 200, 100 + 50);


	Ellipse(hdc, 40, 200, 40 + 200, 200 + 50);// ������Բ����
	MoveToEx(hdc, 0, 600, NULL);// ����(0,600)��(800,0)��ֱ�߿���
	LineTo(hdc, 800, 0);
	SetPixel(hdc, 700, 500, RGB(255, 255, 0));// ��(700,500)�������Ƶ㣬���������ֻ��һ���ش�С����ϸϸ�Ŀ������ҵ�

   // �ָ�ԭ���Ļ��ʺͻ�ˢ
	SelectObject(hdc, hpen_old);
	SelectObject(hdc, hbrush_old);



	return 0;
}
