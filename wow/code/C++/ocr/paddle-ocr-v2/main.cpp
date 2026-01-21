//#include "ppocr_dll.h"
//#include <iostream>
//#include <vector>
//
//int main() {
//    void* h = ppocr_create();
//    char err[2048] = {};
//
//    int rc = ppocr_run(
//        h,
//        "ocr",
//        "D:/test.jpg",
//        "D:/out",
//        "--device=cpu --precision=fp32 --enable_mkldnn=false",
//        err, (int)sizeof(err));
//
//    if (rc != 0) {
//        std::cout << "ppocr_run failed rc=" << rc << " err=" << err << "\n";
//    }
//
//    ppocr_destroy(h);
//    return 0;
//}
