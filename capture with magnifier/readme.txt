放大镜的接口使用可以参考：\Windows-classic-samples-master\Samples\Magnification\cpp\Fullscreen的 FullscreenMagnifierSample

zoom 的解决方案看上去是：通过hook CBaseDevice::StretchRect 完成的
