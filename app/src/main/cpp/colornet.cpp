// colornet.cpp — compiled with -frtti (global default).
// Handles all OpenCV operations. Passes raw float buffers to colornet_impl.cpp
// which contains Sig17Slice (ncnn::Layer subclass) compiled with -fno-rtti.
#include <include/colornet.h>
#include <opencv2/opencv.hpp>
#include <vector>

// Implemented in colornet_impl.cpp (compiled with -fno-rtti, no OpenCV includes)
extern "C" int colorization_ncnn(
        const float* L_data,
        float* ab_data,
        int w_out, int h_out,
        const char* model_path);

int colorization(const cv::Mat& bgr, cv::Mat& out_image, const std::string& model_path) {
    if (bgr.empty()) return -1;

    // 1. BGR -> LAB, extract L channel, resize to 256x256
    cv::Mat base;
    bgr.convertTo(base, CV_32F, 1.0 / 255.0);
    cv::Mat lab;
    cvtColor(base, lab, cv::COLOR_BGR2Lab);
    cv::Mat L;
    cv::extractChannel(lab, L, 0);
    cv::Mat L_256;
    cv::resize(L, L_256, cv::Size(256, 256));
    // L_256 is continuous float — safe to pass as raw pointer

    // 2. Allocate ab buffer: ncnn output is ~128x128 (Sig17Slice halves),
    //    plus 2 extra floats at the end to receive actual ncnn output dimensions
    const int MAX_NCNN_OUT = 256 * 256; // safe upper bound
    std::vector<float> ab_buf(MAX_NCNN_OUT * 2 + 2, 0.f);

    int ret = colorization_ncnn(
        (const float*)L_256.data,
        ab_buf.data(),
        base.cols, base.rows,
        model_path.c_str()
    );
    if (ret < 0) return ret;

    // 3. Read actual ncnn output dimensions from end of buffer
    int ncnn_w = (int)ab_buf[MAX_NCNN_OUT * 2];
    int ncnn_h = (int)ab_buf[MAX_NCNN_OUT * 2 + 1];
    if (ncnn_w <= 0 || ncnn_h <= 0) return -2;

    int ncnn_pixels = ncnn_w * ncnn_h;

    // 4. Wrap a and b channels, resize to original image size
    cv::Mat a(ncnn_h, ncnn_w, CV_32F, ab_buf.data());
    cv::Mat b(ncnn_h, ncnn_w, CV_32F, ab_buf.data() + ncnn_pixels);
    cv::resize(a, a, base.size());
    cv::resize(b, b, base.size());

    // 5. Merge L + a + b back to BGR
    cv::Mat chn[] = {L, a, b};
    cv::Mat lab_out;
    cv::merge(chn, 3, lab_out);
    cv::Mat color;
    cvtColor(lab_out, color, cv::COLOR_Lab2BGR);
    color.convertTo(out_image, CV_8UC3, 255.0);
    return 0;
}
