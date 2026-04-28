// colornet.cpp — compiled with -frtti (global default).
// Does NOT subclass ncnn::Layer. Calls colorization_impl() via extern "C"
// which lives in colornet_impl.cpp (compiled with -fno-rtti).
// This split eliminates the "typeinfo for ncnn::Layer" linker error entirely.
#include <include/colornet.h>
#include <opencv2/opencv.hpp>

// Implemented in colornet_impl.cpp (compiled with -fno-rtti)
extern "C" int colorization_impl(const unsigned char* bgr_data, int w, int h,
                                  unsigned char* out_data,       int out_w, int out_h,
                                  const char* model_path);

int colorization(const cv::Mat& bgr, cv::Mat& out_image, const std::string& model_path) {
    if (bgr.empty()) return -1;
    // Allocate output buffer same size as input
    out_image.create(bgr.rows, bgr.cols, CV_8UC3);
    int ret = colorization_impl(
        bgr.data,       bgr.cols,       bgr.rows,
        out_image.data, out_image.cols, out_image.rows,
        model_path.c_str()
    );
    return ret;
}
