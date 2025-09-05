方案 A（最稳）：安装官方 LLVM（带 CMake 包）

安装 Windows 官方预编译 LLVM（llvm.org 的 .exe 安装包）。
默认会装到：

C:\Program Files\LLVM\


其中必有：

C:\Program Files\LLVM\lib\cmake\llvm\LLVMConfig.cmake


用 VS x64 Native Tools 打开命令行，全新配置 TVM：

cd D:\code\gitcode\tvm
rmdir /S /Q build
mkdir build & cd build

cmake -G "Visual Studio 17 2022" -A x64 -T v143 ^
  -DUSE_LLVM=ON ^
  -DLLVM_DIR="C:\Program Files\LLVM\lib\cmake\llvm" ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
  ..

cmake --build . --config Release --parallel


这样做“编译器/运行库/STL/LLVM”整体与 VS 工具链对齐，避免你之前的 __std_* 链接符号问题。

方案 B：自己用 VS 编译一套 LLVM（自然包含 CMake 包）

如果你想完全与本机 toolset 匹配，自己编一次 LLVM：

:: 在 x64 Native Tools 中
cmake -S llvm -B D:\llvm-build -G "Visual Studio 17 2022" -A x64 -T v143 ^
  -DLLVM_ENABLE_PROJECTS="clang;lld" ^
  -DLLVM_USE_CRT_RELEASE=MD ^
  -DCMAKE_BUILD_TYPE=Release

cmake --build D:\llvm-build --config Release --target INSTALL --parallel


然后指向：

-DLLVM_DIR=D:\llvm-build\install\lib\cmake\llvm

方案 C：继续用 Conda 的 LLVM（有风险）

Conda 的 LLVM 通常带 LLVMConfig.cmake，路径一般是：

%CONDA_PREFIX%\Library\lib\cmake\llvm


配置命令例子：

conda activate yourenv

cd D:\code\gitcode\tvm
rmdir /S /Q build
mkdir build & cd build

cmake -G "Visual Studio 17 2022" -A x64 -T v143 ^
  -DUSE_LLVM=ON ^
  -DLLVM_DIR="%CONDA_PREFIX%\Library\lib\cmake\llvm" ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
  ..
cmake --build . --config Release --parallel


⚠️ 但这条路容易再次命中你之前的 __std_search_1 / __std_remove_8 等符号不匹配（因为 Conda 的 LLVM 可能用不同版本的 MSVC STL 编的）。一旦再出现同类报错，就改用方案 A 或 B。

常见坑与排查

大小写：CMake 找的是 LLVMConfig.cmake（Windows 不区分大小写，但文件名要对）。

确认路径：LLVM_DIR 必须是包含 LLVMConfig.cmake 的目录，而不是 bin 或 lib 本身。

统一运行库：保持 /MD（上面已用 -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL）。

不要混用 VS 自带 LLVM（缺 cmake 包）和 Conda/官方 LLVM（有 cmake 包）的头文件/库。

清理缓存：改 LLVM_DIR 后，务必删除 build 目录重新 CMake，避免缓存误导。
