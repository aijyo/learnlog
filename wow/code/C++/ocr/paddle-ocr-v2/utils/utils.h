#pragma once
#include <cstdint>
#include <vector>
#if defined(_WIN32)
#if defined(PPOCR_DLL_EXPORTS)
#define PPOCR_API __declspec(dllexport)
#else
#define PPOCR_API __declspec(dllimport)
#endif
#else
#define PPOCR_API
#endif


struct BgraFrame
{
    int width = 0;
    int height = 0;
    int stride = 0; // bytes per row
    std::vector<uint8_t> data;
    uint64_t frame_id = 0;
};

//
//#include <string>
//
//#ifdef _WIN32
//#include <windows.h>
//#else
//#include <sys/stat.h>
//#include <unistd.h>
//#endif
//
//static bool FileExists(const std::string& path) {
//#ifdef _WIN32
//    DWORD attrs = GetFileAttributesA(path.c_str());
//    // English comment:
//    // INVALID_FILE_ATTRIBUTES means path does not exist.
//    // Also ensure it is not a directory.
//    return (attrs != INVALID_FILE_ATTRIBUTES) && ((attrs & FILE_ATTRIBUTE_DIRECTORY) == 0);
//#else
//    return access(path.c_str(), F_OK) == 0;
//#endif
//}
//
//static bool DirExists(const std::string& path) {
//#ifdef _WIN32
//    DWORD attrs = GetFileAttributesA(path.c_str());
//    return (attrs != INVALID_FILE_ATTRIBUTES) && ((attrs & FILE_ATTRIBUTE_DIRECTORY) != 0);
//#else
//    struct stat st;
//    return (stat(path.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
//#endif
//}
//
//static std::string JoinPath(const std::string& a, const std::string& b) {
//    if (a.empty()) return b;
//    char sep =
//#ifdef _WIN32
//        '\\';
//#else
//        '/';
//#endif
//    if (a.back() == '/' || a.back() == '\\') return a + b;
//    return a + sep + b;
//}
