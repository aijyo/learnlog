Windows 上编译安装 TVM 指南
1. 设置 PowerShell 脚本执行策略
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force


允许执行本地和受信任的远程 PowerShell 脚本，避免执行报错。

2. 创建和管理 Conda 环境

删除已有环境（可选）：

conda env remove -n tvm-build-venv


创建新的 Conda 环境并安装依赖：

conda create -n tvm-build-venv -c conda-forge llvmdev>=15 cmake>=3.24 git python=3.11


初始化 Conda shell 集成：

conda init 
conda activate tvm-build-venv

3. 安装 Vcpkg（可选，用于 C++ 依赖管理）
git clone https://github.com/microsoft/vcpkg.git D:\vcpkg


Vcpkg 可用于安装 C++ 库和管理依赖，如 Vulkan、OpenCL 等。

4. 安装 Zlib
conda install -c conda-forge zlib


TVM 编译需要 Zlib，确保安装并记录路径。

5. 编译 TVM

配置 CMake：

cd <tvm_root>/build
cmake .. -DCMAKE_BUILD_TYPE=Release `
  -DZLIB_INCLUDE_DIR="D:/ProgramData/anaconda3/envs/tvm-build-venv/Library/include" `
  -DZLIB_LIBRARY="D:/ProgramData/anaconda3/envs/tvm-build-venv/Library/lib/zlib.lib"


编译 TVM：

cmake --build . --config Release --parallel $(nproc)


注意：Windows 上 Release 构建生成的 Python 扩展为 .dll 文件，需要后续处理。

6. 安装 Python 包
6.1 安装 FFI 模块
cd <tvm_root>/build/ffi
pip install .
cd ../..


注意：FFI 模块在 build/ffi 目录下安装，而不是 TVM 根目录。

6.2 安装 TVM Python 包

激活目标 Python 环境：

conda activate <your-own-env>
conda install python  # 确保 Python 已安装


设置 TVM 构建目录路径（PYTHONPATH 或环境变量）：

$env:TVM_LIBRARY_PATH = "<tvm_root>/build"


安装 TVM：

pip install -e <tvm_root>/python


安装完成后，你可以在 Python 中执行 import tvm 并加载 tvm_ffi。

7. 常见注意事项

路径问题

Windows 上编译生成的 tvm_ffi.dll 需重命名为 .pyd 并放到 Python 包目录，或将 build 输出路径加入 PYTHONPATH。

Conda 环境

安装和运行 TVM 的 Python 必须在同一个 Conda 环境中，避免依赖不一致。

Release vs Debug

Python 扩展和构建类型必须一致（Release 或 Debug），否则可能出现 tvm_ffi 导入失败或 ABI 错误。

子模块

TVM 依赖 DLPack 等 Git 子模块，确保 clone 时使用 --recursive 或初始化子模块：

git submodule update --init --recursive
