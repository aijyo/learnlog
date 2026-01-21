参考文档安装依赖：https://www.paddleocr.ai/latest/version3.x/deployment/cpp/OCR_windows.html#21-demo
OPENCV_DIR OpenCV_DIR 是两个不同的值。 默认编译的都是release，因为是静态库，所以都编release，编 debug浪费时间
cmake -S . -B build `
  -DOPENCV_DIR=D:/3rd/opencv/sources/build/install `
  -DOpenCV_DIR=D:/3rd/opencv/sources/build/install/lib/cmake/opencv4 `
  -DPADDLE_LIB=D:/3rd/paddle_inference
