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


是的，用 Conda 环境时，优先用「Ninja + clang-cl」构建最不容易串工具链；但是否能完全避免 __std_* 链接报错，关键仍在于：用“同一套 MSVC 运行库/SDK”。所以要做到两点：
1）在 VS 的 x64 Native Tools 命令行里构建（继承 MSVC/Windows SDK 环境）；2）LLVM_DIR 指向 Conda 的 LLVM（带 LLVMConfig.cmake），同时编译器选 clang-cl（Conda 提供）。

下面给你一份「稳妥可复现」的清单（按步骤照做）。

推荐做法（Ninja + clang-cl + Conda LLVM）
0) 打开正确的终端

打开 x64 Native Tools Command Prompt for VS 2022（很重要！这样 link.exe、MSVC STL、Windows SDK 都来自同一套 VS）。

然后在这个终端里 conda activate yourenv（把 Conda 的 clang/LLVM 放进 PATH）。

1) 快速自检（确认你用的是对的工具）
:: MSVC 编译器版本（必须是 19.3x 或更新）
cl

:: 链接器（link.exe 来自 VS）
where link

:: Conda 的 clang-cl / lld-link（可选）
where clang-cl
where lld-link

:: Conda 的 LLVM CMake 包（确保存在 LLVMConfig.cmake）
dir %CONDA_PREFIX%\Library\lib\cmake\llvm\LLVMConfig.cmake

2) 重新配置 + 构建（Ninja）
:: Clean build folder
cd D:\code\gitcode\tvm
rmdir /S /Q build
mkdir build & cd build

:: Configure with Ninja + clang-cl, LLVM from Conda, /MD runtime
cmake -G "Ninja" ^
  -DCMAKE_C_COMPILER=clang-cl ^
  -DCMAKE_CXX_COMPILER=clang-cl ^
  -DCMAKE_LINKER=link ^
  -DUSE_LLVM=ON ^
  -DLLVM_DIR="%CONDA_PREFIX%\Library\lib\cmake\llvm" ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
  -DCMAKE_BUILD_TYPE=Release ^
  ..

:: Build
cmake --build . --parallel


要点说明（英文注释便于你留存脚本）：

-G "Ninja"：避免 VS 生成器掺入多余 toolset 逻辑，构建更“纯粹”。

-DCMAKE_CXX_COMPILER=clang-cl：使用 Conda 的 clang-cl 前端；

-DCMAKE_LINKER=link：仍用 VS 的 link.exe，这样 STL/运行库/SDK 与 VS 完全一致。

若你想用 Conda 的 lld-link，把这行改成 -DCMAKE_LINKER=lld-link；但通常 link.exe 更稳（与 MSVC 配套）。

-DLLVM_DIR：指向 Conda 的 CMake 包（必须包含 LLVMConfig.cmake）。

-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL：统一 /MD 运行库，避免混用 /MT。

构建一定要在 x64 Native Tools 里进行，这样 INCLUDE/LIB/WindowsSdkDir 等环境变量对齐。

常见问题与变体
Q1：我还是遇到 __std_search_1 / __std_remove_8？

这表示 Conda 的 LLVM 是用更新 MSVC STL 编出来的，但 你当前 VS 的 STL 不够新（即使你在 x64 Native Tools 里）。
修复：

更新 VS2022 工具集到最新（Installer 里更新 v143 + 最新 Win10/11 SDK），然后重新打开 x64 Native Tools 终端再试；

或者不用 Conda 的 LLVM：安装 官方 LLVM（自带 LLVMConfig.cmake），把 LLVM_DIR 指到 C:\Program Files\LLVM\lib\cmake\llvm；

或者自己用当前 VS 工具链编一套 LLVM（保证完全匹配），再把 LLVM_DIR 指向你自己安装的 LLVM。

Q2：能不能“全套用 Conda 的工具链”？

理论可行，但在 Windows 上仍要依赖 VS 的 headers/SDK/运行库，完全脱离 VS 很难。最稳是上面推荐的“clang-cl（Conda）+ link.exe（VS）+ VS STL/SDK”组合。

Q3：VS 生成器（-G "Visual Studio 17 2022"）可以吗？

可以，但更容易“混入”VS 的 toolset 选择逻辑，Ninja + 显式指定编译器更干净。如果坚持 VS 生成器，建议加 -T v143，并确保 LLVM_DIR 与运行库一致。

Q4：我想强行规避 __std_* helper？

只有在重编 LLVM 本体时加宏 _DISABLE_VECTORIZED_ALGORITHMS=1 才有效，对现成二进制包无效。不建议优先这么做。

最后给你一版“故障排查清单”

在 x64 Native Tools for VS 2022 里构建（不是普通 CMD/PowerShell）。

cl 版本 >= 19.3x；link.exe 来自 VS。

LLVM_DIR 指向 包含 LLVMConfig.cmake 的目录（Conda/官方/自编均可），且只用这一套 LLVM 的头/库。

统一 /MD：-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL。

不要手动加 /NODEFAULTLIB:msvcprt 之类的屏蔽默认库选项。

变更工具链后，删 build 目录重新 CMake（避免缓存污染）。

总结：
在 Conda 环境下，Ninja + clang-cl确实是最佳实践，但并不是“银弹”。真正关键是让链接所用的 MSVC STL/运行库与 LLVM 的编译版本不落差。按上面的组合（x64 Native Tools + Conda LLVM_DIR + Ninja + clang-cl + /MD），一般就能稳定通过；若仍见 __std_*，就更新 VS 或改用官方/自编 LLVM 来彻底对齐。
