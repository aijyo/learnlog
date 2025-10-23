# 设置编译工具变量：可以选择 "Visual Studio" 或 "Ninja"
$BUILD_TOOL = "Visual Studio"  # 将此值修改为 "Visual Studio" 或 "Ninja" 来切换构建工具

# 设置 protobuf 和 MLIR 相关路径
$root_dir = Get-Location
$protobuf_version = "21.12"
$llvm_commit_hash = "fc44a4fcd3c54be927c15ddd9211aca1501633e7"
$lit_path = "$root_dir\llvm-project\build\bin\lit.exe"  # 根据安装位置调整

# 确定使用的 CMake 生成器
if ($BUILD_TOOL -eq "Visual Studio") {
    $cmake_generator = "Visual Studio 17 2022"  # 使用 Visual Studio 2022
} elseif ($BUILD_TOOL -eq "Ninja") {
    $cmake_generator = "Ninja"  # 使用 Ninja
} else {
    Write-Host "Unknown BUILD_TOOL: $BUILD_TOOL. Using Ninja by default."
    $cmake_generator = "Ninja"
}

# 创建并构建 protobuf
Write-Host "Cloning protobuf repository..."
git clone -b v$protobuf_version --recursive https://github.com/protocolbuffers/protobuf.git

Write-Host "Building protobuf..."
$protobuf_build_dir = "$root_dir\protobuf_build"
New-Item -ItemType Directory -Path $protobuf_build_dir -Force
cd $protobuf_build_dir
cmake $root_dir\protobuf\cmake -G $cmake_generator `
   -DCMAKE_INSTALL_PREFIX="$root_dir\protobuf_install" `
   -DCMAKE_BUILD_TYPE=Release `
   -Dprotobuf_BUILD_EXAMPLES=OFF `
   -Dprotobuf_BUILD_SHARED_LIBS=OFF `
   -Dprotobuf_BUILD_TESTS=OFF `
   -Dprotobuf_MSVC_STATIC_RUNTIME=OFF `
   -Dprotobuf_WITH_ZLIB=OFF

# add_definitions(/bigobj) 需要将protobuf cmakelists.txt中的这行注释掉
cmake --build . --config Release
cmake --build . --config Release --target install

# 更新 PATH 环境变量
$env:PATH = "$root_dir\protobuf_install\bin;$env:PATH"

# 安装 Python protobuf
Write-Host "Installing protobuf via pip..."
python3 -m pip install protobuf==4.21.12

# 克隆并构建 MLIR
Write-Host "Cloning llvm-project repository..."
git clone -n https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout $llvm_commit_hash
cd ..

Write-Host "Building MLIR..."
$llvm_build_dir = "$root_dir\llvm-project\build"
New-Item -ItemType Directory -Path $llvm_build_dir -Force
cd $llvm_build_dir

cmake $root_dir\llvm-project\llvm -G $cmake_generator `
   -DCMAKE_INSTALL_PREFIX="$root_dir\llvm-project\build\install" `
   -DLLVM_ENABLE_PROJECTS="mlir;clang" `
   -DLLVM_ENABLE_RUNTIMES="openmp" `
   -DLLVM_TARGETS_TO_BUILD="host" `
   -DCMAKE_BUILD_TYPE=Release `
   -DLLVM_ENABLE_ASSERTIONS=ON `
   -DLLVM_ENABLE_RTTI=ON `
   -DLLVM_ENABLE_ZLIB=OFF `
   -DLLVM_INSTALL_UTILS=ON `
   -DENABLE_LIBOMPTARGET=OFF `
   -DLLVM_ENABLE_LIBEDIT=OFF

cmake --build . --config Release
cmake --build . --config Release --target install
cmake --build . --config Release --target check-mlir

# 克隆并构建 onnx-mlir
Write-Host "Cloning onnx-mlir repository..."
git clone --recursive https://github.com/onnx/onnx-mlir.git

Write-Host "Building onnx-mlir..."
$onnx_mlir_build_dir = "$root_dir\onnx-mlir\build"
New-Item -ItemType Directory -Path $onnx_mlir_build_dir -Force
cd $onnx_mlir_build_dir

cmake $root_dir\onnx-mlir -G $cmake_generator `
   -DCMAKE_BUILD_TYPE=Release `
   -DCMAKE_PREFIX_PATH="%root_dir%\protobuf_install;%root_dir%\abs_install" `
   -Dabsl_DIR="$root_dir\abs_install\lib\cmake\absl" `
   -DLLVM_LIT_ARGS=-v `
   -DMLIR_DIR="$root_dir\llvm-project\build\lib\cmake\mlir" `
   -DONNX_MLIR_ENABLE_STABLEHLO=OFF `
   -DONNX_MLIR_ENABLE_WERROR=ON

cmake --build . --config Release

Write-Host "Build complete! The onnx-mlir executable should appear in the Debug/bin or Release/bin directory."

# 打开 Visual Studio 2022 解决方案
Write-Host "Opening Visual Studio solution..."
Start-Process "$onnx_mlir_build_dir\onnx-mlir.sln"



cmake $root_dir\onnx-mlir -G "Visual Studio 17 2022" `
   -A x64 `
   -DCMAKE_BUILD_TYPE=Release `
   -DCMAKE_PREFIX_PATH="$root_dir\protobuf_install" `
   -DLLVM_LIT_ARGS=-v `
   -DMLIR_DIR="$root_dir\llvm-project\build\lib\cmake\mlir" `
   -DONNX_MLIR_ENABLE_STABLEHLO=OFF `
   -DONNX_MLIR_ENABLE_WERROR=ON `
   -DCMAKE_CXX_COMPILER="clang-cl.exe" `
   -DCMAKE_C_COMPILER="clang-cl.exe"
   
   
     
abs 问题：
git clone https://github.com/abseil/abseil-cpp.git

cd abseil-cpp
mkdir build
cd build

cmake .. -G "Ninja" -DCMAKE_INSTALL_PREFIX="$root_dir\abs_install" -DCMAKE_CXX_STANDARD=17

cmake .. -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX="$root_dir\abs_install" -DCMAKE_CXX_STANDARD=17
cmake --build . --config Release
cmake --install . --config Release


