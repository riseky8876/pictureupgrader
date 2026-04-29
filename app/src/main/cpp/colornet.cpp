#include <include/colornet.h>
#include <opencv2/opencv.hpp>
#include <vector>

extern "C" int colorization_ncnn(
        const float* L_data,
        float* ab_data,
        int w_out, int h_out,
        const char* model_path,
        const char* step_log_path);

// step_log_path set by native-lib.cpp before calling colorization()
static char g_step_log_path[512] = {0};
void colorization_set_step_log(const char* p) {
    strncpy(g_step_log_path, p, sizeof(g_step_log_path)-1);
}

int colorization(const cv::Mat& bgr, cv::Mat& out_image, const std::string& model_path) {
    if (bgr.empty()) return -1;

    cv::Mat base;
    bgr.convertTo(base, CV_32F, 1.0/255.0);
    cv::Mat lab;
    cvtColor(base, lab, cv::COLOR_BGR2Lab);
    cv::Mat L;
    cv::extractChannel(lab, L, 0);
    base.release();

    cv::Mat L_256;
    cv::resize(L, L_256, cv::Size(256,256));

    const int MAX_NCNN_OUT = 256*256;
    std::vector<float> ab_buf(MAX_NCNN_OUT*2 + 2, 0.f);

    int ret = colorization_ncnn(
        (const float*)L_256.data,
        ab_buf.data(),
        bgr.cols, bgr.rows,
        model_path.c_str(),
        g_step_log_path);
    if (ret < 0) return ret;

    int ncnn_w = (int)ab_buf[MAX_NCNN_OUT*2];
    int ncnn_h = (int)ab_buf[MAX_NCNN_OUT*2+1];
    if (ncnn_w <= 0 || ncnn_h <= 0) return -2;

    int ncnn_pixels = ncnn_w * ncnn_h;
    cv::Mat a(ncnn_h, ncnn_w, CV_32F, ab_buf.data());
    cv::Mat b(ncnn_h, ncnn_w, CV_32F, ab_buf.data() + ncnn_pixels);
    cv::resize(a, a, bgr.size());
    cv::resize(b, b, bgr.size());

    cv::Mat chn[] = {L, a, b};
    cv::Mat lab_out;
    cv::merge(chn, 3, lab_out);
    cv::Mat color;
    cvtColor(lab_out, color, cv::COLOR_Lab2BGR);
    color.convertTo(out_image, CV_8UC3, 255.0);
    return 0;
}
