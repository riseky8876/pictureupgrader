// compiled with -fno-rtti — no OpenCV includes
#include <cstdio>
#include <cstring>
#include <string>
#include <net.h>
#include <layer.h>

class Sig17Slice : public ncnn::Layer {
public:
    Sig17Slice() { one_blob_only = true; }
    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                const ncnn::Option& opt) const override {
        int bw = bottom_blob.w, bh = bottom_blob.h, bc = bottom_blob.c;
        int outw = bw/2, outh = bh/2;
        top_blob.create(outw, outh, bc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty()) return -100;
        for (int p = 0; p < bc; p++) {
            const float* ptr = bottom_blob.channel(p%bc).row((p/bc)%2) + ((p/bc)/2);
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

static FILE* g_step_f = nullptr;
static void impl_step(const char* s) {
    if (g_step_f) { rewind(g_step_f); fprintf(g_step_f, "COL-impl: %s\n", s); fflush(g_step_f); }
}

extern "C" int colorization_ncnn(
        const float* L_data,
        float* ab_data,
        int w_out, int h_out,
        const char* model_path,
        const char* step_log_path)  // NEW: pass step log path
{
    // open step log for granular tracking
    if (step_log_path && step_log_path[0])
        g_step_f = fopen(step_log_path, "w");

    impl_step("init net");
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 2;  // limit threads to reduce peak memory

    impl_step("register layer");
    net.register_custom_layer("Sig17Slice", Sig17Slice_layer_creator);

    impl_step("load param");
    std::string mp(model_path);
    if (net.load_param((mp + "/siggraph17_color_sim.param").c_str())) {
        impl_step("load param FAILED");
        if (g_step_f) fclose(g_step_f);
        return -1;
    }

    impl_step("load bin");
    if (net.load_model((mp + "/siggraph17_color_sim.bin").c_str())) {
        impl_step("load bin FAILED");
        if (g_step_f) fclose(g_step_f);
        return -1;
    }

    impl_step("create extractor");
    ncnn::Extractor ex = net.create_extractor();

    impl_step("wrap input");
    ncnn::Mat in_L(256, 256, 1, (void*)L_data);
    in_L = in_L.clone();

    impl_step("set input");
    ex.input("input", in_L);

    impl_step("extract output");
    ncnn::Mat out;
    ex.extract("out_ab", out);

    impl_step("copy output");
    if (out.empty()) {
        impl_step("out empty!");
        if (g_step_f) fclose(g_step_f);
        return -2;
    }

    int ncnn_pixels = out.w * out.h;
    const int MAX_NCNN_OUT = 256*256;
    if (ncnn_pixels > MAX_NCNN_OUT) {
        impl_step("out too large!");
        if (g_step_f) fclose(g_step_f);
        return -3;
    }
    memcpy(ab_data,               out.data,                       ncnn_pixels * sizeof(float));
    memcpy(ab_data + ncnn_pixels, (float*)out.data + ncnn_pixels, ncnn_pixels * sizeof(float));
    ab_data[MAX_NCNN_OUT*2]     = (float)out.w;
    ab_data[MAX_NCNN_OUT*2 + 1] = (float)out.h;

    impl_step("done");
    if (g_step_f) fclose(g_step_f);
    return 0;
}
