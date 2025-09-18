exe中链接  obj文件
方案 A：用 CMake 直接把 .obj 当源文件加进目标（推荐）

runner 目标链接你现成的 D:\code\llvm\build\aa.obj：

# CMakeLists.txt

set(EXT_OBJ "D:/code/llvm/build/aa.obj")

add_executable(runner src/runner.cpp "${EXT_OBJ}")

# Tell CMake this is already-compiled object, not C/C++ source
set_source_files_properties("${EXT_OBJ}" PROPERTIES
  EXTERNAL_OBJECT TRUE
  GENERATED TRUE)

# 如果你还需要保证先生成 aa.obj，可以：
# add_custom_target(gen_aa DEPENDS "${EXT_OBJ}")
# add_dependencies(runner gen_aa)


runner.cpp 里直接声明并调用就行：

// All comments in English as requested.

#include <cstdio>

// The object must export 'foo' with C name (no C++ mangling)
extern "C" double foo(double, double);

int main() {
  std::printf("%f\n", foo(40, 2));
  return 0;
}


注意

确保 aa.obj 和 runner.exe 同为 x64（/MACHINE:X64），否则会链接失败。

你的 IR 里函数要用 ExternalLinkage 且命名 foo，生成 .obj 时等价于 extern "C" double foo(double,double)。

如果名字不确定，用 dumpbin /symbols D:\code\llvm\build\aa.obj | findstr foo 看实际符号名。

方案 B：在 VS 项目属性里加 .obj（图形界面）

VS → 项目属性 → Linker → Input → Additional Dependencies → 直接填入：

D:\code\llvm\build\aa.obj


保存后重链即可。

方案 C：命令行直接把 .obj 交给链接器

如果你自己用 cl/link：

cl /c /nologo /EHsc /MD /I<includes> src\runner.cpp /Fo:build\runner.obj
link /nologo build\runner.obj D:\code\llvm\build\aa.obj /OUT:build\runner.exe

方案 D（“写在 cpp 里”的折中）：把 .obj 打包成 .lib，然后 #pragma comment(lib, ...)

MSVC 的 #pragma comment(lib, "...") 只能指定库（.lib），不能直接塞 .obj。
所以先把你的 aa.obj 打成一个静态库：

lib /nologo /OUT:D:\code\llvm\build\aa.lib D:\code\llvm\build\aa.obj


然后在 runner.cpp 顶部写：

// All comments in English as requested.

#pragma comment(lib, "D:\\code\\llvm\\build\\aa.lib")

#include <cstdio>
extern "C" double foo(double, double);

int main() {
  std::printf("%f\n", foo(40, 2));
  return 0;
}


这样无需改 CMake也能把库带进来（MSVC 专属）。同样要注意位数一致、符号名 unmangled。
