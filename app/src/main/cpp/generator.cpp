#include <include/generator.h>

namespace wsdsb {

Generator::~Generator() {
    generator_net_.clear();
}

int Generator::Load(const std::string &model_path) {
    std::string generator_param_path = model_path + "/generator.param";
    std::string generator_model_path = model_path + "/generator.bin";
    generator_net_.opt.use_vulkan_compute = false;
    generator_net_.opt.use_fp16_packed = false;
    generator_net_.opt.use_fp16_storage = false;
    generator_net_.opt.use_fp16_arithmetic = false;
    generator_net_.opt.use_bf16_storage = true;

    if (generator_net_.load_param(generator_param_path.c_str()) < 0) {
        fprintf(stderr, "open param file %s failed\n", generator_param_path.c_str());
        return -1;
    }
    if (generator_net_.load_model(generator_model_path.c_str()) < 0) {
        fprintf(stderr, "open bin file %s failed\n", generator_model_path.c_str());
        return -1;
    }

    // FIX: initialise all 6 slots to -1 (invalid) before lookup.
    // Original code used resize(6) which leaves values as garbage/0,
    // causing generator_ex.input(garbage_index) = segfault when a blob
    // name is not found in the model.
    input_indexes_.assign(6, -1);

    const auto &blobs = generator_net_.blobs();
    for (int i = 0; i < (int)blobs.size(); ++i) {
        const auto &b = blobs[i];
        if      (b.name == "enc_feat_32")  input_indexes_[0] = i;
        else if (b.name == "enc_feat_64")  input_indexes_[1] = i;
        else if (b.name == "enc_feat_128") input_indexes_[2] = i;
        else if (b.name == "enc_feat_256") input_indexes_[3] = i;
        else if (b.name == "style_feat")   input_indexes_[4] = i;
        else if (b.name == "input")        input_indexes_[5] = i;
    }

    // Validate all blobs were found
    for (int i = 0; i < 6; i++) {
        if (input_indexes_[i] < 0) {
            fprintf(stderr, "Generator: required input blob [%d] not found in model\n", i);
            return -1;
        }
    }

    for (const auto &output : generator_net_.output_indexes())
        output_indexes_.push_back(output);

    return 0;
}

void Generator::PreProcess(const void*, std::vector<Tensor_t>&) {}

void Generator::Normlize(const ncnn::Mat &output, std::vector<float> &output_norm) {
    int size = output.c * output.h * output.w;
    output_norm.resize(size);
    std::copy((float*)output.data, (float*)output.data + size, output_norm.begin());
    for (int i = 0; i < size; ++i) {
        float val = std::max(-1.f, std::min(1.f, output_norm[i]));
        output_norm[i] = (val + 1.f) / 2.f;
    }
}

void Generator::Tensor2Image(std::vector<float> &output_tensor, int img_h, int img_w, cv::Mat &output_img) {
    std::vector<cv::Mat> mat_list;
    for (int i = 0; i < 3; ++i) {
        cv::Mat mat(img_h, img_w, CV_32FC1, (void*)(output_tensor.data() + i * img_w * img_h));
        mat_list.push_back(mat.clone()); // clone so mat_list owns data
    }
    cv::Mat result_img_f;
    cv::merge(mat_list, result_img_f);
    cv::Mat result_img;
    result_img_f.convertTo(result_img, CV_8UC3, 255.0, 0);
    cv::cvtColor(result_img, output_img, cv::COLOR_RGB2BGR);
}

void Generator::Run(const std::vector<Tensor_t> &input_tensor, std::vector<Tensor_t> &output_tensor) {
    // FIX: guard against mismatched input tensor count
    if ((int)input_tensor.size() < (int)input_indexes_.size()) {
        fprintf(stderr, "Generator::Run: input_tensor size %d < expected %d\n",
                (int)input_tensor.size(), (int)input_indexes_.size());
        return;
    }
    ncnn::Extractor generator_ex = generator_net_.create_extractor();
    for (int i = 0; i < (int)input_indexes_.size(); ++i) {
        if (input_indexes_[i] < 0) continue; // skip unresolved blobs
        generator_ex.input(input_indexes_[i], input_tensor[i].data);
    }
    for (int output_index : output_indexes_) {
        ncnn::Mat out;
        generator_ex.extract(output_index, out);
        output_tensor.push_back(Tensor_t(out));
    }
}

void Generator::PostProcess(const std::vector<Tensor_t>&, std::vector<Tensor_t> &output_tensor, void *result) {
    if (output_tensor.empty()) {
        fprintf(stderr, "Generator::PostProcess: output_tensor empty\n");
        return;
    }
    ncnn::Mat &out = output_tensor[0].data;
    if (out.empty()) {
        fprintf(stderr, "Generator::PostProcess: out Mat empty\n");
        return;
    }
    std::vector<float> out_norm;
    Normlize(out, out_norm);
    cv::Mat out_img;
    Tensor2Image(out_norm, out.h, out.w, out_img);
    out_img.copyTo(((CodeFormerResult_t*)result)->restored_face);
}

int Generator::Process(const cv::Mat&, void *result) {
    if (!result) return -1;
    auto* cf = (CodeFormerResult_t*)result;
    if (cf->output_tensors.empty()) {
        fprintf(stderr, "Generator::Process: output_tensors from encoder is empty\n");
        return -1;
    }
    std::vector<Tensor_t> output_tensor;
    Run(cf->output_tensors, output_tensor);
    PostProcess(cf->output_tensors, output_tensor, result);
    return 0;
}

} // namespace wsdsb
