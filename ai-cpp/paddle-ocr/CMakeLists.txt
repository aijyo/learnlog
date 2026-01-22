cmake_minimum_required(VERSION 3.20)

project(ocr_paddle_demo LANGUAGES CXX)

# =========================
# Language / global settings
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

# English comment:
# Avoid Windows headers defining min/max macros.
add_definitions(-DNOMINMAX)

# =========================
# MSVC: static CRT (/MT, /MTd)
# =========================
if (MSVC)
    # English comment:
    # Force static runtime to match vcpkg x64-windows-static and produce a self-contained exe (CRT-wise).
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# =========================
# OpenCV from vcpkg (static triplet recommended: x64-windows-static)
# =========================
# English comment:
# With vcpkg toolchain, OpenCV exports config package targets like opencv_core, opencv_imgproc...
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs highgui)


find_package(yaml-cpp CONFIG REQUIRED)
# =========================
# Paddle Inference (local prebuilt)
# Expected layout (as your screenshot):
#   3rd/paddle_inference/paddle/include/...
#   3rd/paddle_inference/paddle/lib/...
#   3rd/paddle_inference/third_party/...
# =========================
set(PADDLE_ROOT   "${CMAKE_CURRENT_SOURCE_DIR}/3rd/paddle_inference")
set(PADDLE_DIR    "${PADDLE_ROOT}/paddle")
set(PADDLE_INC    "${PADDLE_DIR}/include")
set(PADDLE_LIBDIR "${PADDLE_DIR}/lib")

# English comment:
# Validate header/lib existence early for better error messages.
if (NOT EXISTS "${PADDLE_INC}/paddle_inference_api.h")
    message(FATAL_ERROR "Cannot find paddle_inference_api.h at: ${PADDLE_INC}")
endif()

set(PADDLE_INFER_LIB "${PADDLE_LIBDIR}/paddle_inference.lib")
if (NOT EXISTS "${PADDLE_INFER_LIB}")
    message(FATAL_ERROR "Cannot find paddle_inference.lib at: ${PADDLE_LIBDIR}")
endif()

# =========================
# Target
# =========================
add_executable(ocr_paddle_demo
    main.cpp
    PaddleOcr.cpp
    PaddleOcr.h
	PaddleOcrOnnx.cpp
	PaddleOcrOnnx.h
	utils/utility.h
	utils/utility.cc
	utils/yaml_config.h
	utils/yaml_config.cc
	utils/dirent.h
	utils/absl_shim.h
)

# =========================
# Include dirs
# =========================
target_include_directories(ocr_paddle_demo PRIVATE
    "${PADDLE_INC}"
)

# =========================
# Link libraries
# =========================
# OpenCV (static from vcpkg x64-windows-static)
target_link_libraries(ocr_paddle_demo PRIVATE
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
	yaml-cpp::yaml-cpp
)

# Paddle Inference import lib
target_link_libraries(ocr_paddle_demo PRIVATE
    "${PADDLE_INFER_LIB}"
)

# English comment:
# Paddle inference on Windows often needs these system libs.
if (WIN32)
    target_link_libraries(ocr_paddle_demo PRIVATE
        shlwapi
        psapi
        ws2_32
        iphlpapi
        advapi32
        ole32
        shell32
        userenv
        bcrypt
    )
endif()

# =========================
# Compile options
# =========================
if (MSVC)
    target_compile_options(ocr_paddle_demo PRIVATE /W3)
endif()

# =========================
# Runtime: copy Paddle DLLs to output folder
# NOTE: Most Windows Paddle inference packages are "import lib + dll".
# This step ensures the exe can run without manual copying.
# =========================
if (WIN32)
    # English comment:
    # Main DLLs next to paddle_inference.lib
    file(GLOB _PADDLE_DLLS
        "${PADDLE_LIBDIR}/*.dll"
    )

    # English comment:
    # Many runtime deps are shipped in third_party.
    file(GLOB_RECURSE _PADDLE_3P_DLLS
        "${PADDLE_ROOT}/third_party/**/*.dll"
    )

    set(_ALL_PADDLE_DLLS ${_PADDLE_DLLS} ${_PADDLE_3P_DLLS})

    if (_ALL_PADDLE_DLLS)
        add_custom_command(TARGET ocr_paddle_demo POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_ALL_PADDLE_DLLS}
                $<TARGET_FILE_DIR:ocr_paddle_demo>
        )
        message(STATUS "Paddle DLLs will be copied to output directory.")
    else()
        message(STATUS "No Paddle DLLs found to copy. (Maybe fully-static build or DLLs in different path.)")
    endif()
endif()

# # =========================
# # Visual Studio folder grouping (optional)
# # =========================
# source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES
    # main.cpp
    # PaddleOcr.cpp
    # PaddleOcr.h
    # PaddleOcrOnnx.cpp
    # PaddleOcrOnnx.h
# )
