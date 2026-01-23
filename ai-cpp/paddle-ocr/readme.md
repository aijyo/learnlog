vcpkg install yaml-cpp:x64-windows-static
# 如果你也想用 vcpkg 的 OpenCV 静态：
vcpkg install opencv:x64-windows-static

camke :

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows-static
  
  
cmake --build build --config Debug
