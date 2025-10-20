clone代码之后 
LLVM 编译： https://llvm.org/docs/CMake.html

MLIR 编译(包括LLVM):https://mlir.llvm.org/getting_started/

cmake ..\llvm  -G "Ninja"  -DLLVM_ENABLE_PROJECTS=mlir  -DLLVM_BUILD_EXAMPLES=ON  -DLLVM_TARGETS_TO_BUILD="Native"  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON

cmake ..\llvm  -G "Visual Studio 17 2022"  -DLLVM_ENABLE_PROJECTS=mlir  -DLLVM_BUILD_EXAMPLES=ON  -DLLVM_TARGETS_TO_BUILD="Native"  -DCMAKE_BUILD_TYPE=Release -Thost=x64 -DLLVM_ENABLE_ASSERTIONS=ON
