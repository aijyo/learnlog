# Windows 上基于 MLIR 的 onnx-mlir 构建指南（含补丁与原因）

> 目的：整理在 Windows（clang-cl/MSVC）环境下成功编译 `onnx-mlir.exe` 的关键步骤与**源码/CMake 修复**，方便后续可重复构建与排错。  
> 说明：本文代码片段中的**注释均为英文**，正文说明为中文。

---

## 环境与前置

- 工具链：VS 2022（含 `clang-cl`）、CMake、Ninja（或 VS 生成器）。
- 依赖源码：`llvm-project`（已编好 MLIR）、`onnx-mlir`。
- 其他依赖：`protobuf_install`、`abs_install`（abseil）。
- 可选：Python 3.12 venv（若要编 docs/demo 或测试时需要 `onnx` 包）。

> PowerShell 变量写法应使用 `$root_dir\...`（**不要**用 cmd 的 `%root_dir%`）。

---

## 一键构建脚本（保存为 `build-onnx-mlir.ps1`）

> 脚本用途：自动打补丁（下方列出所有修改）、可选创建 3.12 venv、配置 CMake 并仅构建核心目标。  
> 你也可以按“关键改动”小节手工修改源码/CMake，再按“构建命令”自行编译。

```powershell
<# 
  build-onnx-mlir.ps1
  -----------------------------------------------------------------------------
  Reproducible build script for onnx-mlir on Windows (clang-cl + Ninja or VS).
  - Applies cross-platform OpenMP ExternalProject fix (no 'sh -c', Windows DLL)
  - Fixes MSVC/clang-cl flags in Python/Compiler CMakeLists (no -frtti/-fexceptions)
  - Makes Window/Elementwise M_PI and OMSort C-runtime warnings clean under /WX
  - Optional: create a Python 3.12 venv with onnx==1.17.0 for docs/tests
  - Configures and builds core targets (defaults to onnx-mlir CLI tool)
#>

[CmdletBinding()]
param(
  [string]$RootDir         = "D:\code\gitcode",
  [string]$LlvmProjDir     = "$RootDir\llvm-project",
  [string]$OnnxMlirDir     = "$RootDir\onnx-mlir",
  [string]$BuildDir        = "$RootDir\onnx-mlir\build",
  [string]$ProtobufInstall = "$RootDir\protobuf_install",
  [string]$AbslInstall     = "$RootDir\abs_install",
  [string]$Generator       = "Ninja",   # Or: "Visual Studio 17 2022"
  [string]$ClangCL         = "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin\clang-cl.exe",
  [switch]$CreatePy312Venv,
  [string]$PyVenvDir       = "D:\venvs\py312",
  [switch]$DisableTests,   # -DBUILD_TESTING=OFF
  [switch]$SkipPatches,
  [switch]$Reconfigure,
  [switch]$BuildAll,       # build ALL, else only onnx-mlir
  [switch]$AlsoBuildPythonMods
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Section([string]$msg) { Write-Host "==== $msg ====" -ForegroundColor Cyan }
function Save-Backup([string]$path) { if (Test-Path $path) { $bak="$path.bak"; if (-not (Test-Path $bak)) { Copy-Item $path $bak } } }
function Ensure-Dir([string]$d) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null } }

# Optional: create Python 3.12 venv with onnx
$PythonExe = $null
if ($CreatePy312Venv) {
  Write-Section "Create Python 3.12 venv and install onnx==1.17.0"
  Ensure-Dir $PyVenvDir
  & py -3.12 -m venv $PyVenvDir
  & "$PyVenvDir\Scripts\python.exe" -m pip install -U pip
  & "$PyVenvDir\Scripts\python.exe" -m pip install "onnx==1.17.0" "protobuf>=4,<6" numpy
  $PythonExe = "$PyVenvDir\Scripts\python.exe"
}

# Patches (idempotent)
if (-not $SkipPatches) {
  Write-Section "Apply patches"

  # 1) Replace src/Runtime/omp/CMakeLists.txt
  $OmpCmake = Join-Path $OnnxMlirDir "src\Runtime\omp\CMakeLists.txt"
  if (Test-Path $OmpCmake) {
    $content = Get-Content $OmpCmake -Raw
    if ($content -match 'CONFIGURE_COMMAND\s+sh\s+-c') {
      Save-Backup $OmpCmake
@'
# SPDX-License-Identifier: Apache-2.0

if(OMP_SUPPORTED)
  set(OMP_TOPDIR ${CMAKE_CURRENT_BINARY_DIR})
  set_directory_properties(PROPERTIES EP_BASE ${OMP_TOPDIR})

  if(NOT DEFINED OPENMP_SOURCE_DIR)
    set(OPENMP_SOURCE_DIR ${LLVM_BUILD_MAIN_SRC_DIR}/../openmp)
  endif()

  include(ExternalProject)

  set(_EP_GENERATOR ${CMAKE_GENERATOR})
  if(NOT _EP_GENERATOR)
    set(_EP_GENERATOR "Ninja")
  endif()

  if(WIN32)
    # Windows cannot build static libomp; enforce shared.
    set(_LIBOMP_ENABLE_SHARED ON)
    set(_OMP_LIB_NAME libomp.lib)   # Import/implib
    set(_OMP_DLL_NAME libomp.dll)   # Runtime DLL
    set(_OMPRUNTIME_OUT ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.lib)
  else()
    # Non-Windows: keep static to embed into model.so
    set(_LIBOMP_ENABLE_SHARED OFF)
    set(_OMP_LIB_NAME libomp.a)
    set(_OMPRUNTIME_OUT ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.a)
  endif()

  ExternalProject_Add(OMomp
    SOURCE_DIR    ${OPENMP_SOURCE_DIR}
    BINARY_DIR    ${OMP_TOPDIR}/openmp-build
    INSTALL_DIR   ${OMP_TOPDIR}/install
    STAMP_DIR     ${OMP_TOPDIR}/stamp
    TMP_DIR       ${OMP_TOPDIR}/tmp
    DOWNLOAD_COMMAND ""                    # Use local source tree
    CMAKE_GENERATOR ${_EP_GENERATOR}
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DLIBOMP_ENABLE_SHARED=${_LIBOMP_ENABLE_SHARED}
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${OMP_TOPDIR}/openmp-build --target omp
    INSTALL_COMMAND ""
  )

  add_custom_target(ompruntime
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OMP_TOPDIR}/openmp-build/runtime/src/${_OMP_LIB_NAME} ${_OMPRUNTIME_OUT}
    DEPENDS OMomp
    BYPRODUCTS ${OMP_TOPDIR}/openmp-build/runtime/src/${_OMP_LIB_NAME}
  )

  if(WIN32)
    add_custom_command(TARGET ompruntime POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${OMP_TOPDIR}/openmp-build/runtime/src/${_OMP_DLL_NAME}
              ${ONNX_MLIR_LIBRARY_PATH}/${_OMP_DLL_NAME})
  endif()

  install(FILES ${_OMPRUNTIME_OUT} DESTINATION lib)
  if(WIN32)
    install(FILES ${ONNX_MLIR_LIBRARY_PATH}/${_OMP_DLL_NAME} DESTINATION bin)
  endif()

  message(STATUS "OpenMP support           : ON")
else()
  message(STATUS "OpenMP support           : OFF")
endif()
'@ | Set-Content -Encoding UTF8 $OmpCmake
    }
  }

  # 2) Guard _USE_MATH_DEFINES in Elementwise.cpp
  $Elem = Join-Path $OnnxMlirDir "src\Conversion\ONNXToKrnl\Math\Elementwise.cpp"
  if (Test-Path $Elem) {
    $txt = Get-Content $Elem -Raw
    if ($txt -match '^\s*#define\s+_USE_MATH_DEFINES\s*$' -and $txt -notmatch '#ifndef\s+_USE_MATH_DEFINES') {
      Save-Backup $Elem
      $txt = $txt -replace '^\s*#define\s+_USE_MATH_DEFINES\s*$', "#ifndef _USE_MATH_DEFINES`r`n#define _USE_MATH_DEFINES`r`n#endif`r`n#include <cmath>"
      Set-Content -Encoding UTF8 $Elem $txt
    }
  }

  # 3) OMSort.inc: const-correct casts + OM_NULL + init compFunc + default fallback
  $OMSort = Join-Path $OnnxMlirDir "src\Runtime\OMSort.inc"
  if (Test-Path $OMSort) {
    $s = Get-Content $OMSort -Raw
    $changed = $false
    if ($s -notmatch 'define\s+OM_NULL') {
      $prefix = @'
/* Cross-language null pointer literal: C++ uses nullptr, C uses NULL. */
#ifndef OM_NULL
#  ifdef __cplusplus
#    define OM_NULL nullptr
#  else
#    define OM_NULL NULL
#  endif
#endif
'@
      Save-Backup $OMSort
      $s = $prefix + "`r`n" + $s
      $changed = $true
    }
    if ($s -match '\(uint64_t \*\)\(idx1p\)') { $s = $s -replace '\(uint64_t \*\)\(idx1p\)', '(const uint64_t *)(idx1p)'; $changed = $true }
    if ($s -match '\(uint64_t \*\)\(idx2p\)') { $s = $s -replace '\(uint64_t \*\)\(idx2p\)', '(const uint64_t *)(idx2p)'; $changed = $true }
    if ($s -match 'compareFunctionType\s*\*\s*compFunc\s*;' ) { $s = $s -replace 'compareFunctionType\s*\*\s*compFunc\s*;', 'compareFunctionType *compFunc = OM_NULL;'; $changed = $true }
    if ($s -match 'default:\s*\R(?!\s*compFunc\s*=\s*OM_NULL;)') { $s = $s -replace 'default:\s*', "default:`r`n  compFunc = OM_NULL;`r`n"; $changed = $true }
    if ($changed) { Set-Content -Encoding UTF8 $OMSort $s }
  }

  # 4) src/runtime/python CMakeLists: MSVC-safe flags
  $PyRtCmk = Join-Path $OnnxMlirDir "src\runtime\python\CMakeLists.txt"
  if (Test-Path $PyRtCmk) {
    $t = Get-Content $PyRtCmk -Raw
    if ($t -notmatch 'CMAKE_CXX_COMPILER_FRONTEND_VARIANT') {
      Save-Backup $PyRtCmk
@'
# >>> Added: compiler-appropriate flags for pybind modules on MSVC/clang-cl
if (TARGET PyRuntimeC)
  if (MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(PyRuntimeC PRIVATE /EHsc /GR /bigobj)
  else()
    target_compile_options(PyRuntimeC PRIVATE -frtti -fexceptions)
  endif()
endif()

if (TARGET PyCompileAndRuntimeC)
  if (MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(PyCompileAndRuntimeC PRIVATE /EHsc /GR /bigobj)
  else()
    target_compile_options(PyCompileAndRuntimeC PRIVATE -frtti -fexceptions)
  endif()
endif()
# <<< Added
'@ | Add-Content -Encoding UTF8 $PyRtCmk
    }
  }

  # 5) src/Compiler CMakeLists: MSVC-safe flags for PyCompile
  $CompCmk = Join-Path $OnnxMlirDir "src\Compiler\CMakeLists.txt"
  if (Test-Path $CompCmk) {
    $c = Get-Content $CompCmk -Raw
    if ($c -match 'pybind11_add_module\s*\(\s*PyCompile' -and $c -notmatch 'CMAKE_CXX_COMPILER_FRONTEND_VARIANT') {
      Save-Backup $CompCmk
      $c = $c -replace '(pybind11_add_module\s*\(\s*PyCompile[^\)]*\)\s*[\r\n]+add_dependencies\(\s*PyCompile[^\)]*\))', '$1' + @'

# >>> Added: compiler-appropriate flags for PyCompile
if (MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  target_compile_options(PyCompile PRIVATE /EHsc /GR /bigobj)
else()
  target_compile_options(PyCompile PRIVATE -frtti -fexceptions)
endif()
# <<< Added
'@
      Set-Content -Encoding UTF8 $CompCmk $c
    }
  }
}

# Configure
Write-Section "Configure CMake"
Ensure-Dir $BuildDir
Set-Location $BuildDir
if ($Reconfigure) {
  Remove-Item -Recurse -Force .\CMakeCache.txt -ErrorAction Ignore
  Remove-Item -Recurse -Force .\src\Runtime\omp\OMomp-prefix -ErrorAction Ignore
  Remove-Item -Recurse -Force .\src\Runtime\omp\openmp-build -ErrorAction Ignore
}
$commonArgs = @(
  $OnnxMlirDir,
  "-G", $Generator,
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_PREFIX_PATH=$ProtobufInstall;$AbslInstall",
  "-Dabsl_DIR=$($AbslInstall)\lib\cmake\absl",
  "-DLLVM_LIT_ARGS=-v",
  "-DMLIR_DIR=$($LlvmProjDir)\build\lib\cmake\mlir",
  "-DONNX_MLIR_ENABLE_STABLEHLO=OFF",
  "-DONNX_MLIR_ENABLE_WERROR=ON",
  "-DCMAKE_C_COMPILER=$ClangCL",
  "-DCMAKE_CXX_COMPILER=$ClangCL",
  "-DCMAKE_CXX_FLAGS=/D_USE_MATH_DEFINES"
)
if ($DisableTests) { $commonArgs += "-DBUILD_TESTING=OFF" }
if ($PythonExe)    { $commonArgs += "-DPython3_EXECUTABLE=$PythonExe" }
& cmake @commonArgs

# Build
Write-Section "Build"
if ($BuildAll) { & cmake --build . --config Release }
else { & cmake --build . --config Release --target onnx-mlir }
if ($AlsoBuildPythonMods) {
  & cmake --build . --config Release --target PyRuntimeC
  & cmake --build . --config Release --target PyCompile
}
Write-Section "Done"
```

---

## 关键改动与原因（含代码）

### 1. `src/Runtime/omp/CMakeLists.txt`（**替换**）

**原因：**  
原文件使用 `sh -c` 且强制 `LIBOMP_ENABLE_SHARED=OFF` 生成 `libomp.a`，这在 Windows 下不可行（无 `sh`，且 Windows 不提供静态 `libomp.a`）。

**修复点：**  
- 移除 `sh -c`，改为标准 `ExternalProject_Add` + `CMAKE_ARGS`；  
- Windows 下强制 `LIBOMP_ENABLE_SHARED=ON`，复制 `libomp.lib` 和 `libomp.dll`；  
- 非 Windows 维持静态 `.a`，对齐原意。

> 见上面脚本内的同名多行字符串（完整替换内容）。

---

### 2. `src/Conversion/ONNXToKrnl/Math/Elementwise.cpp`（**微调**）

**原因：**  
在全局通过 `/D_USE_MATH_DEFINES` 时，文件内再次 `#define _USE_MATH_DEFINES` 会触发 **macro redefined**，在 `/WX` 下变错误。

**修复点：** 用 `#ifndef` 宏卫士并包含 `<cmath>`。

```cpp
// Avoid redefining macro when it is provided via compiler flags (/D_USE_MATH_DEFINES).
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
```

---

### 3. `src/Runtime/OMSort.inc`（**多处 C 兼容与告警清理**）

**原因：**  
- 将 `const void*` 强转为 `uint64_t*` 触发 `-Wcast-qual`；  
- 在 C 源编译单元中使用 `nullptr` 不可用；  
- `compFunc` 可能未初始化（`-Wsometimes-uninitialized`）。

**修复点：**  
- 对 `idx1p/idx2p` 的强转加 `const`；  
- 引入跨 C/C++ 的空指针宏 `OM_NULL`；  
- `compFunc` 初始化为 `OM_NULL`，并在 `default:` 兜底。

```c
/* Cross-language null pointer literal: C++ uses nullptr, C uses NULL. */
#ifndef OM_NULL
#  ifdef __cplusplus
#    define OM_NULL nullptr
#  else
#    define OM_NULL NULL
#  endif
#endif

/* Make casts const-correct for qsort-style comparators. */
const uint64_t idx1 = *((const uint64_t *)(idx1p));
const uint64_t idx2 = *((const uint64_t *)(idx2p));

/* Ensure pointer is initialized to avoid -Wsometimes-uninitialized under /WX. */
compareFunctionType *compFunc = OM_NULL;

/* Fallback in default case to keep a single return path. */
default:
  compFunc = OM_NULL;
  break;
```

---

### 4. `src/runtime/python/CMakeLists.txt`（**MSVC/clang-cl 参数兼容**）

**原因：**  
`clang-cl` 不接受 `-frtti / -fexceptions / -flto`（GNU 风格），应使用 `/GR /EHsc`（LTO 则是 `/GL`/`/LTCG`）。

**修复点：** 按前端区分编译开关，避免“未知参数变错误”。

```cmake
# Use compiler-appropriate flags: GNU/Clang(GNU) vs MSVC/clang-cl.
if (MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  # MSVC or clang-cl: /GR and /EHsc are already correct.
  target_compile_options(PyRuntimeC PRIVATE /EHsc /GR /bigobj)
else()
  # GCC/Clang (GNU frontend)
  target_compile_options(PyRuntimeC PRIVATE -frtti -fexceptions)
endif()
```

> 如有 `PyCompileAndRuntimeC` 目标，同理添加一段。

*（可选改进：包装/复制产物用 `cmake -E copy_if_different` 与 `$<TARGET_FILE:...>`，跨平台复制 `.pyd/.so`。）*

---

### 5. `src/Compiler/CMakeLists.txt`（**`PyCompile` 的编译开关**）

**原因：**  
同上，`pybind11_add_module(PyCompile ...)` 目标在 clang-cl 下不要用 GNU 开关。

**修复点：**

```cmake
# After: pybind11_add_module(PyCompile PyOMCompileSession.cpp)
if (MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  target_compile_options(PyCompile PRIVATE /EHsc /GR /bigobj)
else()
  target_compile_options(PyCompile PRIVATE -frtti -fexceptions)
  # target_compile_options(PyCompile PRIVATE -flto)  # optional on Unix
endif()
```

---

## 构建配置与命令

### 配置（示例，Ninja + clang-cl）

```powershell
cmake $root_dir\onnx-mlir -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="$root_dir\protobuf_install;$root_dir\abs_install" `
  -Dabsl_DIR="$root_dir\abs_install\lib\cmake\absl" `
  -DLLVM_LIT_ARGS=-v `
  -DMLIR_DIR="$root_dir\llvm-project\build\lib\cmake\mlir" `
  -DONNX_MLIR_ENABLE_STABLEHLO=OFF `
  -DONNX_MLIR_ENABLE_WERROR=ON `
  -DCMAKE_C_COMPILER="D:\...\clang-cl.exe" `
  -DCMAKE_CXX_COMPILER="D:\...\clang-cl.exe" `
  -DCMAKE_CXX_FLAGS="/D_USE_MATH_DEFINES" `
  -DBUILD_TESTING=OFF `
  -DPython3_EXECUTABLE="D:/venvs/py312/Scripts/python.exe"   # 若需 docs/test
```

### 构建

```powershell
# 只编核心 CLI（推荐）
cmake --build . --config Release --target onnx-mlir

# 也可：
cmake --build . --config Release --target onnx-mlir-opt
cmake --build . --config Release --target PyRuntimeC
cmake --build . --config Release --target PyCompile
```

---

## 常见问题与排查

- **OpenMP 源与构建失败**：原脚本 `sh -c`、Windows 静态 `libomp.a` 均不可用。按上文完整替换 CMakeLists。
- **`M_PI` 未定义**：确保 `/D_USE_MATH_DEFINES` 或在需要处 `<cmath>` + 宏卫士。
- **`-frtti/-fexceptions` 报未知参数**：在 MSVC/clang-cl 下使用 `/GR /EHsc`。
- **`-Wcast-qual`、`-Wsometimes-uninitialized`**：按上文修复 `OMSort.inc`。
- **文档示例/测试跑 Python 报 `ModuleNotFoundError: onnx`**：  
  1) 走 3.12 venv 并 `pip install onnx==1.17.0`；或  
  2) 关闭 `-DBUILD_TESTING=OFF` 并只构建核心目标。

---

如需将此文档进一步拆分为“最小更改版”（仅列必要补丁的 diff）、或适配 VS 生成器（`-G "Visual Studio 17 2022"`）的专用配置样例，请根据你的仓库风格微调即可。
