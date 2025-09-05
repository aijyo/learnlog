# Windows 上编译安装 TVM 与 LLVM 的完整流程

本文记录了在 Windows 环境下成功编译 LLVM + TVM 的完整经验步骤。  

环境要求：  
- Windows 10/11  
- Visual Studio 2022 (v143 工具链)  
- Conda Python 环境  

---

## 一、编译与安装 LLVM

### 1. Clone 源码并切换分支
```bash
git clone https://github.com/llvm/llvm-project.git
cd D:\code\gitcode\llvm-project
git checkout release/XX   # 切换到最新 release 分支
```

### 2. 创建目录
```bash
cd D:\code\gitcode\llvm-project
mkdir build-llvm
mkdir install-llvm
```

### 3. CMake 配置
```bash
cmake -S D:\code\gitcode\llvm-project\llvm -B D:\code\gitcode\llvm-project\build-llvm ^
  -G "Visual Studio 17 2022" -A x64 -T v143 ^
  -DLLVM_ENABLE_PROJECTS="clang;lld" ^
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" ^
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF ^
  -DLLVM_OPTIMIZED_TABLEGEN=ON ^
  -DLLVM_USE_CRT_RELEASE=MD ^
  -DCMAKE_INSTALL_PREFIX="D:\code\gitcode\llvm-project\install-llvm" ^
  -DCMAKE_C_FLAGS="/utf-8" -DCMAKE_CXX_FLAGS="/utf-8"
```

### 4. 编译与安装
```bash
cmake --build D:\code\gitcode\llvm-project\build-llvm --config Release 
cmake --install D:\code\gitcode\llvm-project\build-llvm --config Release
```

---

## 二、编译与安装 TVM

### 1. 配置 Conda 环境
```bash
conda init
conda activate tvm-build-venv
```

### 2. Clone TVM 源码
```bash
git clone https://github.com/apache/tvm.git
cd tvm
rm -rf build && mkdir build && cd build
```

### 3. 配置 `config.cmake`
复制并修改配置：
```bash
cp ../cmake/config.cmake .
```

编辑 `config.cmake`，添加以下内容：
```cmake
set(CMAKE_BUILD_TYPE Release)

set(USE_LLVM ON)
set(LLVM_DIR "D:/code/gitcode/llvm-project/install-llvm/lib/cmake/llvm")

set(HIDE_PRIVATE_SYMBOLS ON)

set(USE_CUDA OFF)
set(USE_METAL OFF)
set(USE_VULKAN OFF)
set(USE_OPENCL OFF)
set(USE_CUBLAS OFF)
set(USE_CUDNN OFF)
set(USE_CUTLASS OFF)
```

### 4. CMake 配置与编译
```bash
###set PATH=D:\code\gitcode\llvm-project\install-llvm\bin;%PATH%
$env:PATH = "D:\code\gitcode\llvm-project\install-llvm\bin;" + $env:PATH
where llvm-config
where llc

cd D:/code/gitcode/tvm/build
cmake ..
cmake --build . --config Release --parallel
```

### 5. Python 绑定与安装

#### (1) 安装 FFI
```bash
cd D:/code/gitcode/tvm/ffi
pip install .
cd ..
```

#### (2) 设置环境变量
```bash
set TVM_LIBRARY_PATH="D:/code/gitcode/tvm/build"
```

#### (3) 安装 TVM Python 包
```bash
cd D:/code/gitcode/tvm/build
pip install -e D:/code/gitcode/tvm/python
```

---

## 三、验证安装

进入 Python 交互环境：
```python
import tvm
print(tvm.__version__)
```

如果提示缺少依赖库（如 `psutil`），执行：
```bash
python -m pip install psutil
```

---

✅ 至此，TVM + LLVM 已经在 Windows 上编译完成，可以正常使用 `import tvm`。  
