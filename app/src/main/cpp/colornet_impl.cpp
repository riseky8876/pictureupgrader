// colornet_impl.cpp — compiled with -fno-rtti via CMakeLists set_source_files_properties
// Isolated from colornet.cpp so the extern "C" colorization() in colornet.cpp
// can be compiled normally with -frtti (matching native-lib.cpp).
// Sig17Slice lives here ONLY — no typeinfo referenced from -frtti translation units.
#include <cstdio>
#include <net.h>
#include <layer.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

// Forward declaration of the C linkage function we implement below
extern "C" int colorization_impl(const unsigned char* bgr_data, int w, int h,
                                  unsigned char* out_data,       int out_w, int out_h,
                                  const char* model_path);

class Sig17Slice : public ncnn::Layer {
public:
    Sig17Slice() { one_blob_only = true; }

    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                const ncnn::Option& opt) const override {
        int bw = bottom_blob.w, bh = bottom_blob.h, bc = bottom_blob.c;
        int outw = bw / 2, outh = bh / 2;
        top_blob.create(outw, outh, bc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty()) return -100;
#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < bc; p++) {
            const float* ptr = bottom_blob.channel(p % bc).row((p / bc) % 2) + ((p / bc) / 2);
            float* outptr = top_blob.channel(p);
            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) { *outptr++ = *ptr; ptr += 2; }
                ptr += bw;
            }
        }
        return 0;
    }
};
DEFINE_LAYER_CREATOR(Sig17Slice)

extern "C" int colorization_impl(const unsigned char* bgr_data, int w, int h,
                                   unsigned char* out_data,       int out_w, int out_h,
                                   const char* model_path) {
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.register_custom_layer("Sig17Slice", Sig17Slice_layer_creator);
    std::string mp(model_path);
    if (net.load_param((mp + "/siggraph17_color_sim.param").c_str())) return -1;
    if (net.load_model((mp + "/siggraph17_color_sim.bin").c_str()))   return -1;

    // wrap raw data into cv::Mat without copy
    cv::Mat bgr(h, w, CV_8UC3, (void*)bgr_data);
    cv::Mat base; bgr.convertTo(base, CV_32F, 1.0/255);

    cv::Mat lab, L, input_img;
    cvtColor(base, lab, cv::COLOR_BGR2Lab);
    cv::extractChannel(lab, L, 0);
    resize(L, input_img, cv::Size(256, 256));

    ncnn::Mat in_L(256, 256, 1, (void*)input_img.data);
    in_L = in_L.clone();

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in_L);
    ncnn::Mat out;
    ex.extract("out_ab", out);

    cv::Mat a(out.h, out.w, CV_32F, (float*)out.data);
    cv::Mat b(out.h, out.w, CV_32F, (float*)out.data + out.w * out.h);
    cv::resize(a, a, base.size());
    cv::resize(b, b, base.size());

    cv::Mat chn[] = {L, a, b};
    cv::merge(chn, 3, lab);
    cv::Mat color;
    cvtColor(lab, color, cv::COLOR_Lab2BGR);
    color.convertTo(color, CV_8UC3, 255);

    // write result to caller's buffer
    cv::Mat out_mat(out_h, out_w, CV_8UC3, out_data);
    cv::resize(color, out_mat, cv::Size(out_w, out_h));
    return 0;
}
