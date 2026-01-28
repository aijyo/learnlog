cmake_minimum_required(VERSION 3.20)
project(char_seg_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Option A (recommended):
# Pass OpenCV_DIR from command line:
#   -DOpenCV_DIR="D:/opencv/opencv/sources/build/install/x64/vc17/lib"
#
# ---- Option B (fallback guesses):
if(NOT OpenCV_DIR)
  # Try common OpenCV install layouts
  set(_OPENCV_ROOT "D:/opencv/opencv/sources/build/install")
  if(EXISTS "${_OPENCV_ROOT}/x64/vc17/lib/OpenCVConfig.cmake")
    set(OpenCV_DIR "${_OPENCV_ROOT}/x64/vc17/lib")
  elseif(EXISTS "${_OPENCV_ROOT}/lib/cmake/opencv4/OpenCVConfig.cmake")
    set(OpenCV_DIR "${_OPENCV_ROOT}/lib/cmake/opencv4")
  endif()
endif()

find_package(OpenCV REQUIRED)

add_executable(char_seg
  char_segment_main.cpp
  char_segmenter.cpp
  char_segmenter.h
)

target_include_directories(char_seg PRIVATE
  "${OpenCV_INCLUDE_DIRS}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(char_seg PRIVATE
  ${OpenCV_LIBS}
)

# Nice-to-have for MSVC
if(MSVC)
  target_compile_definitions(char_seg PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_options(char_seg PRIVATE /W4 /permissive-)
endif()
