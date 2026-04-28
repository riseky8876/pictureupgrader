// colornet_impl.cpp — compiled with -fno-rtti (see CMakeLists.txt)
// Must NOT include any OpenCV header that uses typeid (e.g. flann/any.h).
// Only includes ncnn headers (built with -fno-rtti, safe) and basic C headers.
// cv::Mat pixel operations are done via raw pointers passed from colornet.cpp.
#include <cstdio>
#include <cstring>
#include <string>
#include <net.h>
#include <layer.h>
#include <omp.h>

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

// Called from colornet.cpp (compiled with -frtti + OpenCV).
// Receives pre-processed L channel as float buffer (256x256),
// returns ab channels as float buffer (h*w*2 floats, output size matches input L).
extern "C" int colorization_ncnn(
        const float* L_data,    // input: L channel 256x256 float
        float* ab_data,         // output: ab channels h_out*w_out*2 floats
        int w_out, int h_out,   // size of ab output (original image size)
        const char* model_path)
{
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.register_custom_layer("Sig17Slice", Sig17Slice_layer_creator);

    std::string mp(model_path);
    if (net.load_param((mp + "/siggraph17_color_sim.param").c_str())) return -1;
    if (net.load_model((mp + "/siggraph17_color_sim.bin").c_str()))   return -1;

    // wrap L buffer — ncnn Mat(w, h, channels, data)
    ncnn::Mat in_L(256, 256, 1, (void*)L_data);
    in_L = in_L.clone(); // ensure ncnn owns the data

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in_L);
    ncnn::Mat out;
    ex.extract("out_ab", out); // shape: [2, out.h, out.w]

    // out contains a and b channels contiguously
    // copy to caller's buffer (caller will resize to original image size)
    int ncnn_pixels = out.w * out.h;
    memcpy(ab_data,                out.data,                          ncnn_pixels * sizeof(float)); // a
    memcpy(ab_data + ncnn_pixels, (float*)out.data + ncnn_pixels,    ncnn_pixels * sizeof(float)); // b
    // pass ncnn output size back via first two floats is not needed —
    // caller knows ncnn output is 256/2=128 wide (Sig17Slice halves it),
    // but we store the actual out dimensions in ab_data[-2],[-1] is fragile.
    // Instead store w/h as the last 2 floats after ab data.
    ab_data[ncnn_pixels * 2]     = (float)out.w;
    ab_data[ncnn_pixels * 2 + 1] = (float)out.h;

    return 0;
}
