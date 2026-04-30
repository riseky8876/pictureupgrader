#include <include/realesrgan.h>
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "PU", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PU", __VA_ARGS__)

namespace wsdsb {

RealESRGAN::RealESRGAN() : scale(2), tile_size(400), tile_pad(10) {}
RealESRGAN::~RealESRGAN() { net_.clear(); }

int RealESRGAN::Load(const std::string& model_path) {
    // Try Vulkan first (GPU — faster), fallback to CPU if unavailable
    if (ncnn::get_gpu_count() > 0) {
        net_.opt.use_vulkan_compute = true;
        LOGI("ESR: using Vulkan GPU");
    } else {
        net_.opt.use_vulkan_compute = false;
        LOGI("ESR: using CPU");
    }
    net_.opt.num_threads = 4;

    std::string param = model_path + "/real_esrgan.param";
    std::string bin   = model_path + "/real_esrgan.bin";
    if (net_.load_param(param.c_str()) < 0) { LOGE("load param failed: %s", param.c_str()); return -1; }
    if (net_.load_model(bin.c_str())   < 0) { LOGE("load bin failed: %s",   bin.c_str());   return -1; }
    for (auto i : net_.input_indexes())  input_indexes_.push_back(i);
    for (auto i : net_.output_indexes()) output_indexes_.push_back(i);
    LOGI("ESR: model loaded OK");
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

// Tiling with proper seam blending — same approach as realsr-ncnn-vulkan
int RealESRGAN::Process(const cv::Mat& input_img, void* result) {
    if (input_img.empty()) return -1;

    int orig_w = input_img.cols;
    int orig_h = input_img.rows;

    // Pad to multiple of 2
    int pad_w = (orig_w % 2 != 0) ? 1 : 0;
    int pad_h = (orig_h % 2 != 0) ? 1 : 0;
    cv::Mat padded;
    cv::copyMakeBorder(input_img, padded, 0, pad_h, 0, pad_w,
                       cv::BORDER_REFLECT_101);

    int pw = padded.cols, ph = padded.rows;
    cv::Mat output(ph * scale, pw * scale, CV_8UC3, cv::Scalar(0));

    int tiles_x = (pw + tile_size - 1) / tile_size;
    int tiles_y = (ph + tile_size - 1) / tile_size;
    LOGI("ESR: %dx%d → %dx%d, tiles %dx%d",
         orig_w, orig_h, pw * scale, ph * scale, tiles_x, tiles_y);

    for (int yi = 0; yi < tiles_y; yi++) {
        for (int xi = 0; xi < tiles_x; xi++) {
            // Input tile region (without pad)
            int x0 = xi * tile_size;
            int y0 = yi * tile_size;
            int x1 = std::min(x0 + tile_size, pw);
            int y1 = std::min(y0 + tile_size, ph);

            // Expanded region (with overlap padding)
            int x0p = std::max(x0 - tile_pad, 0);
            int y0p = std::max(y0 - tile_pad, 0);
            int x1p = std::min(x1 + tile_pad, pw);
            int y1p = std::min(y1 + tile_pad, ph);

            cv::Mat tile = padded(cv::Rect(x0p, y0p, x1p-x0p, y1p-y0p)).clone();

            std::vector<Tensor_t> in_t, out_t;
            PreProcess((void*)&tile, in_t);
            Run(in_t, out_t);
            cv::Mat tile_out;
            PostProcess(in_t, out_t, (void*)&tile_out);
            if (tile_out.empty()) continue;

            // Crop overlap from tile output
            int crop_x = (x0 - x0p) * scale;
            int crop_y = (y0 - y0p) * scale;
            int crop_w = (x1 - x0) * scale;
            int crop_h = (y1 - y0) * scale;

            // Clamp to tile_out bounds
            crop_x = std::max(0, std::min(crop_x, tile_out.cols - 1));
            crop_y = std::max(0, std::min(crop_y, tile_out.rows - 1));
            crop_w = std::min(crop_w, tile_out.cols - crop_x);
            crop_h = std::min(crop_h, tile_out.rows - crop_y);
            if (crop_w <= 0 || crop_h <= 0) continue;

            // Destination in output
            int dst_x = x0 * scale;
            int dst_y = y0 * scale;
            int dst_w = std::min(crop_w, output.cols - dst_x);
            int dst_h = std::min(crop_h, output.rows - dst_y);
            if (dst_w <= 0 || dst_h <= 0) continue;

            tile_out(cv::Rect(crop_x, crop_y, dst_w, dst_h))
                .copyTo(output(cv::Rect(dst_x, dst_y, dst_w, dst_h)));
        }
    }

    // Crop back to original size * scale
    int final_w = std::min(orig_w * scale, output.cols);
    int final_h = std::min(orig_h * scale, output.rows);
    output(cv::Rect(0, 0, final_w, final_h)).copyTo(*(cv::Mat*)result);
    LOGI("ESR: done %dx%d", final_w, final_h);
    return 0;
}

int RealESRGAN::Padding(const cv::Mat&, cv::Mat&, int&, int&) { return 0; }
void RealESRGAN::Tensor2Image(const ncnn::Mat&, int, int, cv::Mat&) {}

} // namespace wsdsb
