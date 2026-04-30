#include <include/realesrgan.h>
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "PU", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PU", __VA_ARGS__)

namespace wsdsb {

RealESRGAN::RealESRGAN() : scale(2), tile_size(400), tile_pad(10) {}
RealESRGAN::~RealESRGAN() { net_.clear(); }

int RealESRGAN::Load(const std::string& model_path) {
    net_.opt.use_vulkan_compute = false;
    net_.opt.num_threads = 2;
    std::string param = model_path + "/real_esrgan.param";
    std::string bin   = model_path + "/real_esrgan.bin";
    if (net_.load_param(param.c_str()) < 0) { LOGE("load param failed: %s", param.c_str()); return -1; }
    if (net_.load_model(bin.c_str())   < 0) { LOGE("load bin failed: %s",   bin.c_str());   return -1; }
    for (auto i : net_.input_indexes())  input_indexes_.push_back(i);
    for (auto i : net_.output_indexes()) output_indexes_.push_back(i);
    LOGI("RealESRGAN loaded OK");
    return 0;
}

void RealESRGAN::PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor) {
    const cv::Mat* mat = (const cv::Mat*)input_data;
    ncnn::Mat in = ncnn::Mat::from_pixels(mat->data, ncnn::Mat::PIXEL_BGR2RGB,
                                          mat->cols, mat->rows);
    in.substract_mean_normalize(0, norm_vals_);
    input_tensor.push_back(Tensor_t(in));
}

void RealESRGAN::Run(const std::vector<Tensor_t>& input_tensor,
                     std::vector<Tensor_t>& output_tensor) {
    ncnn::Extractor ex = net_.create_extractor();
    for (int i = 0; i < (int)input_indexes_.size(); i++)
        ex.input(input_indexes_[i], input_tensor[i].data);
    for (int i = 0; i < (int)output_indexes_.size(); i++) {
        ncnn::Mat out;
        ex.extract(output_indexes_[i], out);
        output_tensor.push_back(Tensor_t(out));
    }
}

void RealESRGAN::PostProcess(const std::vector<Tensor_t>&,
                              std::vector<Tensor_t>& output_tensor, void* result) {
    if (output_tensor.empty() || output_tensor[0].data.empty()) return;
    ncnn::Mat& out = output_tensor[0].data;
    std::vector<cv::Mat> channels;
    for (int i = 0; i < 3; i++) {
        cv::Mat c(out.h, out.w, CV_32FC1, (float*)out.data + i * out.w * out.h);
        channels.push_back(c.clone());
    }
    cv::Mat f, u8;
    cv::merge(channels, f);
    f.convertTo(u8, CV_8UC3, 255.0);
    cv::cvtColor(u8, *(cv::Mat*)result, cv::COLOR_RGB2BGR);
}

int RealESRGAN::Process(const cv::Mat& input_img, void* result) {
    if (input_img.empty()) return -1;

    // Cap input to 400x400 max so entire image fits in one ncnn pass.
    // No tiling = no seam artifacts. Output will be 2x this size,
    // then we resize back to 2x the ORIGINAL dimensions.
    const int MAX_SIDE = 400;
    int orig_w = input_img.cols;
    int orig_h = input_img.rows;

    cv::Mat small;
    if (orig_w > MAX_SIDE || orig_h > MAX_SIDE) {
        float s = (orig_w > orig_h)
                  ? (float)MAX_SIDE / orig_w
                  : (float)MAX_SIDE / orig_h;
        cv::resize(input_img, small,
                   cv::Size((int)(orig_w * s), (int)(orig_h * s)),
                   0, 0, cv::INTER_AREA);
    } else {
        small = input_img.clone();
    }
    LOGI("ESR: input %dx%d → small %dx%d", orig_w, orig_h, small.cols, small.rows);

    std::vector<Tensor_t> in_t, out_t;
    PreProcess((void*)&small, in_t);
    Run(in_t, out_t);

    cv::Mat sr_small;
    PostProcess(in_t, out_t, (void*)&sr_small);
    if (sr_small.empty()) { LOGE("ESR: sr_small empty"); return -1; }
    LOGI("ESR: sr_small %dx%d", sr_small.cols, sr_small.rows);

    // Resize SR output to 2x original size
    cv::Mat* out = (cv::Mat*)result;
    cv::resize(sr_small, *out,
               cv::Size(orig_w * scale, orig_h * scale),
               0, 0, cv::INTER_LANCZOS4);
    LOGI("ESR: final output %dx%d", out->cols, out->rows);
    return 0;
}

int RealESRGAN::Padding(const cv::Mat&, cv::Mat&, int&, int&) { return 0; }

void RealESRGAN::Tensor2Image(const ncnn::Mat&, int, int, cv::Mat&) {}

} // namespace wsdsb
