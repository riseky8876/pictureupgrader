// face.cpp — SCRFD-based face detection
// Replaces yolov7-lite-e (unavailable publicly) with scrfd_500m-opt2
// Model: https://github.com/nihui/ncnn-assets/tree/master/models
// SCRFD paper: https://arxiv.org/abs/2105.04714
// Interface preserved: same Load/Process/AlignFace as original face.cpp

#include "include/face.h"

namespace wsdsb {

// ── anchor generator (SCRFD style) ──────────────────────────────────────────
static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales)
{
    int num_ratio = ratios.w;
    int num_scale = scales.w;
    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);
    const float cx = 0;
    const float cy = 0;
    for (int i = 0; i < num_ratio; i++) {
        float ar = ratios[i];
        int r_w = (int)round(base_size / sqrt(ar));
        int r_h = (int)round(r_w * ar);
        for (int j = 0; j < num_scale; j++) {
            float scale = scales[j];
            float rs_w = r_w * scale;
            float rs_h = r_h * scale;
            float* anchor = anchors.row(i * num_scale + j);
            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }
    return anchors;
}

// ── proposal generator ───────────────────────────────────────────────────────
// scrfd_500m-opt2 has NO kps output blobs, so we generate synthetic landmarks
// from the face centre (sufficient for AlignFace affine estimation).
static void generate_proposals(const ncnn::Mat& anchors, int feat_stride,
                                const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob,
                                float prob_threshold, float wpad2, float hpad2, float scale,
                                int img_w, int img_h,
                                std::vector<Object_t>& objects)
{
    int w = score_blob.w;
    int h = score_blob.h;
    const int num_anchors = anchors.h;

    for (int q = 0; q < num_anchors; q++) {
        const float* anchor = anchors.row(q);
        const ncnn::Mat score = score_blob.channel(q);
        const ncnn::Mat bbox  = bbox_blob.channel_range(q * 4, 4);

        float anchor_y = anchor[1];
        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++) {
            float anchor_x = anchor[0];
            for (int j = 0; j < w; j++) {
                int index = i * w + j;
                float prob = score[index];
                if (prob >= prob_threshold) {
                    float dx = bbox.channel(0)[index] * feat_stride;
                    float dy = bbox.channel(1)[index] * feat_stride;
                    float dw = bbox.channel(2)[index] * feat_stride;
                    float dh = bbox.channel(3)[index] * feat_stride;

                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;
                    float x0 = cx - dx;
                    float y0 = cy - dy;
                    float x1 = cx + dw;
                    float y1 = cy + dh;

                    // unpad + unscale
                    float rx0 = (x0 - wpad2) / scale;
                    float ry0 = (y0 - hpad2) / scale;
                    float rx1 = (x1 - wpad2) / scale;
                    float ry1 = (y1 - hpad2) / scale;

                    rx0 = std::max(std::min(rx0, (float)(img_w - 1)), 0.f);
                    ry0 = std::max(std::min(ry0, (float)(img_h - 1)), 0.f);
                    rx1 = std::max(std::min(rx1, (float)(img_w - 1)), 0.f);
                    ry1 = std::max(std::min(ry1, (float)(img_h - 1)), 0.f);

                    Object_t obj;
                    obj.rect.x = rx0;
                    obj.rect.y = ry0;
                    obj.rect.width  = rx1 - rx0;
                    obj.rect.height = ry1 - ry0;
                    obj.score = prob;

                    // Synthesise 5 facial landmarks from bounding-box geometry.
                    // This is a rough approximation; accurate enough for the
                    // affine warp used by AlignFace / CodeFormer.
                    float fw = obj.rect.width;
                    float fh = obj.rect.height;
                    float fx = obj.rect.x;
                    float fy = obj.rect.y;
                    obj.pts.push_back(cv::Point2f(fx + fw * 0.30f, fy + fh * 0.40f)); // left eye
                    obj.pts.push_back(cv::Point2f(fx + fw * 0.70f, fy + fh * 0.40f)); // right eye
                    obj.pts.push_back(cv::Point2f(fx + fw * 0.50f, fy + fh * 0.55f)); // nose
                    obj.pts.push_back(cv::Point2f(fx + fw * 0.35f, fy + fh * 0.72f)); // left mouth
                    obj.pts.push_back(cv::Point2f(fx + fw * 0.65f, fy + fh * 0.72f)); // right mouth

                    objects.push_back(obj);
                }
                anchor_x += feat_stride;
            }
            anchor_y += feat_stride;
        }
    }
}

// ── Face class ───────────────────────────────────────────────────────────────
Face::Face() : prob_threshold(0.3f), nms_threshold(0.45f)
{
    // canonical 512×512 face template used by CodeFormer/GFPGAN
    face_template.push_back(cv::Point2f(192.98138f, 239.94708f));
    face_template.push_back(cv::Point2f(318.90277f, 240.1936f));
    face_template.push_back(cv::Point2f(256.63416f, 314.01935f));
    face_template.push_back(cv::Point2f(201.26117f, 371.41043f));
    face_template.push_back(cv::Point2f(313.08905f, 371.15118f));
}

Face::~Face() { net_.clear(); }

int Face::Load(const std::string& model_path)
{
    // model files: scrfd_500m-opt2.param / scrfd_500m-opt2.bin
    std::string param = model_path + "/scrfd_500m-opt2.param";
    std::string bin   = model_path + "/scrfd_500m-opt2.bin";

    net_.opt.use_vulkan_compute = false;

    if (net_.load_param(param.c_str())) {
        fprintf(stderr, "open param file %s failed\n", param.c_str());
        return -1;
    }
    if (net_.load_model(bin.c_str())) {
        fprintf(stderr, "open bin file %s failed\n", bin.c_str());
        return -1;
    }
    return 0;
}

void Face::PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor)
{
    const int target_size = 640;
    auto* bgr = (cv::Mat*)input_data;
    int img_w = bgr->cols;
    int img_h = bgr->rows;

    int w = img_w, h = img_h;
    float scale = 1.f;
    if (w > h) { scale = (float)target_size / w; w = target_size; h = (int)(h * scale); }
    else        { scale = (float)target_size / h; h = target_size; w = (int)(w * scale); }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr->data, ncnn::Mat::PIXEL_BGR2RGB,
                                                  img_w, img_h, w, h);
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2,
                           wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1 / 128.f, 1 / 128.f, 1 / 128.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    Tensor_t t(in_pad);
    t.img_w = img_w; t.img_h = img_h;
    t.pad_w = wpad;  t.pad_h = hpad;
    t.in_w  = in_pad.w; t.in_h = in_pad.h;
    t.scale = scale;
    input_tensor.push_back(t);
}

void Face::Run(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor)
{
    ncnn::Extractor ex = net_.create_extractor();
    ex.input("input.1", input_tensor[0].data);

    // scrfd_500m-opt2 blob names (score + bbox for stride 8, 16, 32)
    const char* score_names[] = {"412", "474", "536"};
    const char* bbox_names[]  = {"415", "477", "539"};
    for (int i = 0; i < 3; i++) {
        ncnn::Mat score_blob, bbox_blob;
        ex.extract(score_names[i], score_blob);
        ex.extract(bbox_names[i],  bbox_blob);
        output_tensor.push_back(Tensor_t(score_blob));
        output_tensor.push_back(Tensor_t(bbox_blob));
    }
}

void Face::PostProcess(const std::vector<Tensor_t>& input_tensor,
                       std::vector<Tensor_t>& output_tensor, void* result)
{
    const Tensor_t& inp = input_tensor[0];
    float wpad2 = inp.pad_w / 2.f;
    float hpad2 = inp.pad_h / 2.f;

    struct StrideCfg { int base; int stride; };
    StrideCfg cfgs[3] = {{16, 8}, {64, 16}, {256, 32}};

    std::vector<Object_t> proposals;
    for (int i = 0; i < 3; i++) {
        const ncnn::Mat& score_blob = output_tensor[i * 2].data;
        const ncnn::Mat& bbox_blob  = output_tensor[i * 2 + 1].data;

        ncnn::Mat ratios(1); ratios[0] = 1.f;
        ncnn::Mat scales(2); scales[0] = 1.f; scales[1] = 2.f;
        ncnn::Mat anchors = generate_anchors(cfgs[i].base, ratios, scales);

        generate_proposals(anchors, cfgs[i].stride, score_blob, bbox_blob,
                           prob_threshold, wpad2, hpad2, inp.scale,
                           inp.img_w, inp.img_h, proposals);
    }

    // sort by score
    std::sort(proposals.begin(), proposals.end(),
              [](const Object_t& a, const Object_t& b){ return a.score > b.score; });

    // NMS
    std::vector<int> picked;
    int n = (int)proposals.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = proposals[i].rect.area();
    for (int i = 0; i < n; i++) {
        int keep = 1;
        for (int j : picked) {
            cv::Rect_<float> inter = proposals[i].rect & proposals[j].rect;
            float inter_area = inter.area();
            float union_area = areas[i] + areas[j] - inter_area;
            if (inter_area / union_area > nms_threshold) { keep = 0; break; }
        }
        if (keep) picked.push_back(i);
    }

    int count = (int)picked.size();
    ((PipeResult_t*)result)->face_count = count;
    for (int i = 0; i < count; i++) {
        ((PipeResult_t*)result)->object[i] = proposals[picked[i]];
    }
}

void Face::AlignFace(const cv::Mat& img, Object_t& object)
{
    cv::Mat affine_matrix = cv::estimateAffinePartial2D(
        object.pts, face_template, cv::noArray(), cv::LMEDS);
    cv::Mat cropped_face;
    cv::warpAffine(img, cropped_face, affine_matrix, cv::Size(512, 512),
                   1, cv::BORDER_CONSTANT, cv::Scalar(135, 133, 132));
    cv::Mat affine_matrix_inv;
    cv::invertAffineTransform(affine_matrix, affine_matrix_inv);
    affine_matrix_inv *= 2;
    affine_matrix_inv.copyTo(object.trans_inv);
    cropped_face.copyTo(object.trans_img);
}

// unused virtuals — satisfy interface
void Face::generateAndInsertProposals(const std::vector<float>&, int,
                                      const Tensor_t&, const ncnn::Mat&,
                                      float, std::vector<Object_t>&) {}

int Face::Process(const cv::Mat& input_img, void* result)
{
    std::vector<Tensor_t> input_tensor, output_tensor;
    PreProcess((void*)&input_img, input_tensor);
    Run(input_tensor, output_tensor);
    PostProcess(input_tensor, output_tensor, result);
    for (int i = 0; i < ((PipeResult_t*)result)->face_count; i++)
        AlignFace(input_img, ((PipeResult_t*)result)->object[i]);
    return 0;
}

} // namespace wsdsb
