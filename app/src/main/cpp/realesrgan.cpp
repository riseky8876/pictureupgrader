#include <include/realesrgan.h>
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "PU", __VA_ARGS__)

namespace wsdsb {

RealESRGAN::RealESRGAN() : scale(2), tile_size(400), tile_pad(10) {}

RealESRGAN::~RealESRGAN() { net_.clear(); }

int RealESRGAN::Load(const std::string& model_path) {
    net_.opt.use_vulkan_compute = false;
    std::string param = model_path + "/real_esrgan.param";
    std::string bin   = model_path + "/real_esrgan.bin";
    if (net_.load_param(param.c_str()) < 0) { fprintf(stderr,"load param failed: %s\n",param.c_str()); return -1; }
    if (net_.load_model(bin.c_str())   < 0) { fprintf(stderr,"load bin failed: %s\n",  bin.c_str());   return -1; }
    for (auto i : net_.input_indexes())  input_indexes_.push_back(i);
    for (auto i : net_.output_indexes()) output_indexes_.push_back(i);
    return 0;
}

void RealESRGAN::Tensor2Image(const ncnn::Mat& out, int h, int w, cv::Mat& img) {
    std::vector<cv::Mat> channels;
    for (int i = 0; i < 3; i++) {
        cv::Mat c(h, w, CV_32FC1, (float*)out.data + i * w * h);
        channels.push_back(c.clone());
    }
    cv::Mat f;
    cv::merge(channels, f);
    cv::Mat u8;
    f.convertTo(u8, CV_8UC3, 255.0);
    cv::cvtColor(u8, img, cv::COLOR_RGB2BGR);
}

// FIX: original code had img_pad_w and img_pad_h swapped —
// img_pad_h should be set when rows is odd, img_pad_w when cols is odd
int RealESRGAN::Padding(const cv::Mat& img, cv::Mat& pad_img, int& img_pad_h, int& img_pad_w) {
    img_pad_w = (img.cols % 2 != 0) ? 1 : 0;
    img_pad_h = (img.rows % 2 != 0) ? 1 : 0;
    cv::copyMakeBorder(img, pad_img, 0, img_pad_h, 0, img_pad_w,
                       cv::BORDER_REFLECT_101);
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
    if (output_tensor.empty()) return;
    ncnn::Mat& out = output_tensor[0].data;
    if (out.empty()) return;
    cv::Mat img;
    Tensor2Image(out, out.h, out.w, img);
    img.copyTo(*(cv::Mat*)result);
}

int RealESRGAN::Process(const cv::Mat& input_img, void* result) {
    if (input_img.empty()) return -1;

    cv::Mat pad_img;
    int img_pad_h = 0, img_pad_w = 0;
    Padding(input_img, pad_img, img_pad_h, img_pad_w);

    int tiles_x = (int)std::ceil((float)pad_img.cols / tile_size);
    int tiles_y = (int)std::ceil((float)pad_img.rows / tile_size);
    LOGI("ESR: input %dx%d, pad %dx%d, tiles %dx%d",
         input_img.cols, input_img.rows, pad_img.cols, pad_img.rows, tiles_x, tiles_y);

    cv::Mat out(pad_img.rows * scale, pad_img.cols * scale, CV_8UC3);

    for (int i = 0; i < tiles_y; i++) {
        for (int j = 0; j < tiles_x; j++) {
            int x0 = j * tile_size;
            int y0 = i * tile_size;
            int x1 = std::min(x0 + tile_size, pad_img.cols);
            int y1 = std::min(y0 + tile_size, pad_img.rows);

            // padded tile bounds (with overlap)
            int x0p = std::max(x0 - tile_pad, 0);
            int x1p = std::min(x1 + tile_pad, pad_img.cols);
            int y0p = std::max(y0 - tile_pad, 0);
            int y1p = std::min(y1 + tile_pad, pad_img.rows);

            int tw = x1p - x0p;
            int th = y1p - y0p;
            if (tw <= 0 || th <= 0) continue;

            cv::Mat tile = pad_img(cv::Rect(x0p, y0p, tw, th)).clone();

            std::vector<Tensor_t> in_t, out_t;
            PreProcess((void*)&tile, in_t);
            Run(in_t, out_t);
            cv::Mat out_tile;
            PostProcess(in_t, out_t, (void*)&out_tile);

            if (out_tile.empty()) continue;

            // crop padding from tile output
            int ox = (x0 - x0p) * scale;
            int oy = (y0 - y0p) * scale;
            int ow = (x1 - x0) * scale;
            int oh = (y1 - y0) * scale;

            // guard bounds
            ox = std::max(0, std::min(ox, out_tile.cols - 1));
            oy = std::max(0, std::min(oy, out_tile.rows - 1));
            ow = std::min(ow, out_tile.cols - ox);
            oh = std::min(oh, out_tile.rows - oy);
            if (ow <= 0 || oh <= 0) continue;

            // destination in output image
            int dx = x0 * scale;
            int dy = y0 * scale;
            int dw = std::min(ow, out.cols - dx);
            int dh = std::min(oh, out.rows - dy);
            if (dw <= 0 || dh <= 0) continue;

            out_tile(cv::Rect(ox, oy, dw, dh))
                .copyTo(out(cv::Rect(dx, dy, dw, dh)));
        }
    }

    // Crop back to original size * scale (remove padding)
    int final_w = input_img.cols * scale;
    int final_h = input_img.rows * scale;
    final_w = std::min(final_w, out.cols);
    final_h = std::min(final_h, out.rows);
    out(cv::Rect(0, 0, final_w, final_h)).copyTo(*(cv::Mat*)result);
    LOGI("ESR: done, output %dx%d", final_w, final_h);
    return 0;
}


} // namespace wsdsb
