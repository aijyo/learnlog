
编译 LLVM
cd D:\code\gitcode
mkdir build-llvm
mkdir install-llvm

运行 CMake 配置
cmake -S D:\code\gitcode\llvm-project\llvm -B D:\code\gitcode\build-llvm ^
  -G "Visual Studio 17 2022" -A x64 -T v143 ^
  -DLLVM_ENABLE_PROJECTS="clang;lld" ^
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" ^
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF ^
  -DLLVM_OPTIMIZED_TABLEGEN=ON ^
  -DLLVM_USE_CRT_RELEASE=MD ^
  -DCMAKE_INSTALL_PREFIX="D:\code\gitcode\install-llvm" ^
  -DCMAKE_C_FLAGS="/utf-8" -DCMAKE_CXX_FLAGS="/utf-8"


cmake -S D:\code\gitcode\llvm-project\llvm -B D:\code\gitcode\build-llvm ^
  -G "Ninja" ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_LINKER=link ^
  -DLLVM_ENABLE_PROJECTS="clang;lld" ^
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" ^
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF ^
  -DLLVM_OPTIMIZED_TABLEGEN=ON ^
  -DLLVM_USE_CRT_RELEASE=MD ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX="D:\code\gitcode\install-llvm"

4. 编译与安装

如果是 VS 生成器：

cmake --build D:\code\gitcode\build-llvm --config Release --parallel


如果是 Ninja：

cmake --build D:\code\gitcode\build-llvm --target INSTALL --parallel


5. 执行安装命令
Visual Studio 生成器（多配置）
cmake --install D:\code\gitcode\build-llvm --config Release
Ninja 生成器（单配置）
cmake --install D:\code\gitcode\build-llvm --config Release


注意：目标名大小写不敏感，INSTALL 和 install 都行。
