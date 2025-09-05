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
cd tvm
rm -rf build && mkdir build && cd build
# Specify the build configuration via CMake options
cp ../cmake/config.cmake .

然后再 config.cmake文件中添加：
set(CMAKE_BUILD_TYPE Release)

set(USE_LLVM ON)

set(HIDE_PRIVATE_SYMBOLS ON)

set(USE_CUDA   OFF)

set(USE_METAL  OFF)

set(USE_VULKAN OFF)

set(USE_OPENCL OFF)

set(USE_CUBLAS OFF)

set(USE_CUDNN  OFF)

set(USE_CUTLASS OFF)

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

更新 LLVM 的推荐操作步骤（适用于 Windows + MSVC 2022）
1. 设置 Conda 优先使用 conda-forge 通道

为了从 conda-forge 获取最新、与 VS2022 最新编译器兼容的 LLVM，你可以将 conda-forge 放到优先通道（确保在 .condarc 中优先）：

conda config --add channels conda-forge
conda config --set channel_priority strict


这样在执行 conda install 时，会优先从 conda-forge 获取包版本。

Stack Overflow

2. 安装最新可用的 llvmdev

在设置好频道优先级之后，执行：

conda install llvmdev


这将安装来自 conda-forge 的最新构建版本（目前为 v21.1.0）
Anaconda
。

此版本默认已使用 VS2022 编译器 构建，与 MSVC 14.3x 兼容，可避免符号错误等问题 
conda-forge.org
+1
。

6>  正在创建库 D:/code/gitcode/tvm/build/Release/tvm_allvisible.lib 和对象 D:/code/gitcode/tvm/build/Release/tvm_allvisible.exp
7>LLVMBitReader.lib(BitcodeReader.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
7>LLVMCore.lib(AutoUpgrade.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
7>LLVMWindowsDriver.lib(MSVCPaths.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
7>LLVMVectorize.lib(SeedCollection.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
7>LLVMInstrumentation.lib(DataFlowSanitizer.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
7>LLVMAsmPrinter.lib(CodeViewDebug.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
7>LLVMX86CodeGen.lib(X86Subtarget.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
6>LLVMBitReader.lib(BitcodeReader.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
6>LLVMCore.lib(AutoUpgrade.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
6>LLVMWindowsDriver.lib(MSVCPaths.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
6>LLVMVectorize.lib(SeedCollection.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
6>LLVMInstrumentation.lib(DataFlowSanitizer.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
6>LLVMAsmPrinter.lib(CodeViewDebug.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
6>LLVMX86CodeGen.lib(X86Subtarget.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
7>LLVMSPIRVCodeGen.lib(SPIRVEmitIntrinsics.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
7>LLVMSPIRVCodeGen.lib(SPIRVBuiltins.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
7>LLVMAArch64CodeGen.lib(AArch64Arm64ECCallLowering.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
6>LLVMSPIRVCodeGen.lib(SPIRVEmitIntrinsics.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
6>LLVMSPIRVCodeGen.lib(SPIRVBuiltins.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
6>LLVMAArch64CodeGen.lib(AArch64Arm64ECCallLowering.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_search_1
7>LLVMX86Desc.lib(X86InstComments.cpp.obj) : error LNK2019: 无法解析的外部符号 __std_find_first_of_trivial_pos_1，函数 "unsigned __int64 __cdecl std::_Find_first_of_pos_vectorized<char,char>(char const * const,unsigned __int64,char const * const,unsigned __int64)" (??$_Find_first_of_pos_vectorized@DD@std@@YA_KQEBD_K01@Z) 中引用了该符号
7>LLVMProfileData.lib(InstrProf.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_find_first_of_trivial_pos_1
7>LLVMSPIRVCodeGen.lib(SPIRVBuiltins.cpp.obj) : error LNK2019: 无法解析的外部符号 __std_find_end_1，函数 "char const * __cdecl std::_Find_end_vectorized<char const ,char const >(char const * const,char const * const,char const * const,unsigned __int64)" (??$_Find_end_vectorized@$$CBD$$CBD@std@@YAPEBDQEBD00_K@Z) 中引用了该符号
7>LLVMSupport.lib(StringRef.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_find_end_1
7>LLVMCore.lib(Dominators.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMTransformUtils.lib(CodeLayout.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMX86Desc.lib(X86InstComments.cpp.obj) : error LNK2019: 无法解析的外部符号 __std_find_first_of_trivial_pos_1，函数 "unsigned __int64 __cdecl std::_Find_first_of_pos_vectorized<char,char>(char const * const,unsigned __int64,char const * const,unsigned __int64)" (??$_Find_first_of_pos_vectorized@DD@std@@YA_KQEBD_K01@Z) 中引用了该符号
6>LLVMProfileData.lib(InstrProf.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_find_first_of_trivial_pos_1
6>LLVMSPIRVCodeGen.lib(SPIRVBuiltins.cpp.obj) : error LNK2019: 无法解析的外部符号 __std_find_end_1，函数 "char const * __cdecl std::_Find_end_vectorized<char const ,char const >(char const * const,char const * const,char const * const,unsigned __int64)" (??$_Find_end_vectorized@$$CBD$$CBD@std@@YAPEBDQEBD00_K@Z) 中引用了该符号
6>LLVMSupport.lib(StringRef.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_find_end_1
6>LLVMCore.lib(Dominators.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMTransformUtils.lib(CodeLayout.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMTransformUtils.lib(LoopConstrainer.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMTransformUtils.lib(LoopConstrainer.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMAnalysis.lib(MemorySSA.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMAnalysis.lib(MemorySSAUpdater.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMScalarOpts.lib(GVNHoist.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMTransformUtils.lib(SSAUpdaterBulk.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMTransformUtils.lib(LoopUnroll.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMTransformUtils.lib(PromoteMemoryToRegister.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMCodeGen.lib(WinEHPrepare.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMCodeGen.lib(MachineDominators.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMCodeGen.lib(MachinePostDominators.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMScalarOpts.lib(ADCE.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMSelectionDAG.lib(SelectionDAGISel.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMCodeGen.lib(MachineFunction.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMCodeGen.lib(MachineBlockPlacement.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMCodeGen.lib(RegisterCoalescer.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMVectorize.lib(VPlanConstruction.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMVectorize.lib(VPlanTransforms.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMVectorize.lib(DependencyGraph.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMInstrumentation.lib(DataFlowSanitizer.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMVectorize.lib(SLPVectorizer.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMVectorize.lib(VPlanAnalysis.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMVectorize.lib(VPlan.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMVectorize.lib(VPlanVerifier.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
7>LLVMDemangle.lib(RustDemangle.cpp.obj) : error LNK2019: 无法解析的外部符号 __std_remove_1，函数 "char * __cdecl std::_Remove_vectorized<char,char>(char * const,char * const,char)" (??$_Remove_vectorized@DD@std@@YAPEADQEAD0D@Z) 中引用了该符号
7>D:\code\gitcode\tvm\build\Release\tvm.dll : fatal error LNK1120: 5 个无法解析的外部命令
6>LLVMAnalysis.lib(MemorySSA.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMAnalysis.lib(MemorySSAUpdater.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMScalarOpts.lib(GVNHoist.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMTransformUtils.lib(SSAUpdaterBulk.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMTransformUtils.lib(LoopUnroll.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMTransformUtils.lib(PromoteMemoryToRegister.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMCodeGen.lib(WinEHPrepare.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMCodeGen.lib(MachineDominators.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMCodeGen.lib(MachinePostDominators.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMScalarOpts.lib(ADCE.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMSelectionDAG.lib(SelectionDAGISel.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMCodeGen.lib(MachineFunction.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMCodeGen.lib(MachineBlockPlacement.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMCodeGen.lib(RegisterCoalescer.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMVectorize.lib(VPlanConstruction.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMVectorize.lib(VPlanTransforms.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMVectorize.lib(DependencyGraph.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMInstrumentation.lib(DataFlowSanitizer.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMVectorize.lib(SLPVectorizer.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMVectorize.lib(VPlanAnalysis.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMVectorize.lib(VPlan.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMVectorize.lib(VPlanVerifier.cpp.obj) : error LNK2001: 无法解析的外部符号 __std_remove_8
6>LLVMDemangle.lib(RustDemangle.cpp.obj) : error LNK2019: 无法解析的外部符号 __std_remove_1，函数 "char * __cdecl std::_Remove_vectorized<char,char>(char * const,char * const,char)" (??$_Remove_vectorized@DD@std@@YAPEADQEAD0D@Z) 中引用了该符号
6>D:\code\gitcode\tvm\build\Release\tvm_allvisible.dll : fatal error LNK1120: 5 个无法解析的外部命令


但核心思路是：把“编译器/标准库(STL) + 运行库 + LLVM 二进制”的来源统一。你现在是“VS 的 MSVC 工具链 + Conda 的 LLVM”，两边的 MSVC STL 内部符号版本对不上，才会出现 __std_search_1 / __std_remove_8 之类的未解析。

给你两条可行路径，任选其一（不要混搭）：

路线 1：统一到 VS 工具链（最省心）

把 TVM 改为链接 VS 安装的 LLVM（而不是 Conda 的 LLVM）。这样“编译器/标准库/LLVM”全部来自 VS 工具链，版本天然对齐。

在 VS Installer 里更新到最新 VS2022 + v143 工具集 + Win10/11 SDK。

找到 VS 自带的 LLVM 的 CMake 包路径（常见位置其一）：

C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\lib\cmake\llvm


用 “x64 Native Tools Command Prompt for VS 2022” 全新配置/编译 TVM：

cd D:\code\gitcode\tvm
rmdir /S /Q build
mkdir build & cd build

cmake -G "Visual Studio 17 2022" -A x64 -T v143 ^
  -DUSE_LLVM=ON ^
  -DLLVM_DIR="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\lib\cmake\llvm" ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
  ..

cmake --build . --config Release --parallel


要点：

强制 -T v143；

用 VS 的 LLVM_DIR；

统一运行库为 /MD（MultiThreadedDLL），避免和外部库冲突。

路线 2：统一到 Conda（进阶）

让编译 TVM 的编译器/链接器/标准库与 LLVM 都来自 Conda。难点是 clang-cl 仍需调用本机 VS 的头文件/库，一旦版本差异仍会踩坑；因此这条路通常需要你用本机 VS 工具链重编 LLVM（或确保 Conda 的 LLVM 和你本机的 MSVC STL 完全匹配）。如果你坚持 Conda：

A. 先尽量“同一个工具链打包”：

激活你的 Conda 环境；

使用 Ninja + clang-cl 配置（尽量减少 VS 生成器的干预）：

conda activate yourenv

cd D:\code\gitcode\tvm
rmdir /S /Q build
mkdir build & cd build

cmake -G "Ninja" ^
  -DCMAKE_C_COMPILER=clang-cl ^
  -DCMAKE_CXX_COMPILER=clang-cl ^
  -DUSE_LLVM=ON ^
  -DLLVM_DIR="%CONDA_PREFIX%\Library\lib\cmake\llvm" ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ^
  ..

cmake --build . --parallel


B. 如果还是报 __std_*，说明 Conda 打的 LLVM 用到更新或不同的 MSVC STL 内部符号。此时有两种补救：

用你机器的 VS 工具链把 LLVM 重编一遍（保证与本机 MSVC 完全一致，且 /MD），然后用这个 LLVM_DIR 去编 TVM；

或者放弃 Conda 的 LLVM，改走路线 1（最稳）。

注：给 LLVM 加 _DISABLE_VECTORIZED_ALGORITHMS=1 能避免这些 __std_* 辅助符号，但这必须重编 LLVM 本体才生效，对现有 Conda 二进制无效。

常见坑位自检

混用了 /MT 和 /MD：统一成 /MD（上面用 -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL）。

链接命令里意外有 /NODEFAULTLIB:msvcprt 等屏蔽默认库。

构建不是在 x64 Native Tools 里进行，导致 toolset 漂移。

LLVM_DIR 没指向你真正想用的那套 LLVM（VS 的或 Conda 的）。

最简结论

最简单可靠的对齐法：别用 Conda 的 LLVM 做链接；改用 VS 自带/官方安装包的 LLVM（把 LLVM_DIR 指到 VS 的 cmake 目录），然后在 VS 的 x64 Native Tools 里全新配置编译。这样编译器、STL、运行库、LLVM 全部一致，问题就消失了。
