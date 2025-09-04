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

然后再 config.cmake文件中添加：set(USE_LLVM ON)

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
