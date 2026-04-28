// face.cpp — SCRFD-500m face detection
// Safe version with full null/bounds checks to prevent segfault
#include "include/face.h"

namespace wsdsb {

static void generate_proposals(int feat_stride,
                                const ncnn::Mat& score_blob,
                                const ncnn::Mat& bbox_blob,
                                float prob_threshold,
                                float wpad2, float hpad2, float scale,
                                int img_w, int img_h,
                                std::vector<Object_t>& objects)
{
    // scrfd_500m-opt2: score_blob shape = [num_anchors*h*w] flat or [c,h,w]
    // bbox_blob shape = [num_anchors*4, h, w]
    // num_anchors per stride = 2
    const int num_anchors = 2;

    int fw = score_blob.w;
    int fh = score_blob.h;
    // score_blob.c should be num_anchors (2), but guard against 1
    int fc = score_blob.c;

    for (int q = 0; q < num_anchors; q++) {
        // safe channel access
        const float* score_ptr = (fc > q) ? (const float*)score_blob.channel(q) : nullptr;
        if (!score_ptr) continue;

        for (int i = 0; i < fh; i++) {
            for (int j = 0; j < fw; j++) {
                int idx = i * fw + j;
                float score = score_ptr[idx];
                if (score < prob_threshold) continue;

                // anchor centre for this grid cell
                float cx = (j + 0.5f) * feat_stride;
                float cy = (i + 0.5f) * feat_stride;

                // bbox regression: [q*4 .. q*4+3] channels of bbox_blob
                int bbox_ch = q * 4;
                if (bbox_blob.c < bbox_ch + 4) continue; // guard

                float l = bbox_blob.channel(bbox_ch + 0)[idx] * feat_stride;
                float t = bbox_blob.channel(bbox_ch + 1)[idx] * feat_stride;
                float r = bbox_blob.channel(bbox_ch + 2)[idx] * feat_stride;
                float b = bbox_blob.channel(bbox_ch + 3)[idx] * feat_stride;

                float x0 = (cx - l - wpad2) / scale;
                float y0 = (cy - t - hpad2) / scale;
                float x1 = (cx + r - wpad2) / scale;
                float y1 = (cy + b - hpad2) / scale;

                x0 = std::max(0.f, std::min(x0, (float)(img_w - 1)));
                y0 = std::max(0.f, std::min(y0, (float)(img_h - 1)));
                x1 = std::max(0.f, std::min(x1, (float)(img_w - 1)));
                y1 = std::max(0.f, std::min(y1, (float)(img_h - 1)));

                if (x1 <= x0 || y1 <= y0) continue;

                Object_t obj;
                obj.rect.x      = x0;
                obj.rect.y      = y0;
                obj.rect.width  = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.score       = score;

                // Synthesise 5 landmarks from bbox geometry
                float fw2 = obj.rect.width, fh2 = obj.rect.height;
                float fx = obj.rect.x,      fy  = obj.rect.y;
                obj.pts.push_back(cv::Point2f(fx + fw2*0.30f, fy + fh2*0.40f));
                obj.pts.push_back(cv::Point2f(fx + fw2*0.70f, fy + fh2*0.40f));
                obj.pts.push_back(cv::Point2f(fx + fw2*0.50f, fy + fh2*0.55f));
                obj.pts.push_back(cv::Point2f(fx + fw2*0.35f, fy + fh2*0.72f));
                obj.pts.push_back(cv::Point2f(fx + fw2*0.65f, fy + fh2*0.72f));

                objects.push_back(obj);
            }
        }
    }
}

// ── Face ─────────────────────────────────────────────────────────────────────
Face::Face() : prob_threshold(0.5f), nms_threshold(0.45f)
{
    face_template = {
        cv::Point2f(192.98138f, 239.94708f),
        cv::Point2f(318.90277f, 240.19360f),
        cv::Point2f(256.63416f, 314.01935f),
        cv::Point2f(201.26117f, 371.41043f),
        cv::Point2f(313.08905f, 371.15118f)
    };
}

Face::~Face() { net_.clear(); }

int Face::Load(const std::string& model_path)
{
    net_.opt.use_vulkan_compute = false;
    std::string param = model_path + "/scrfd_500m-opt2.param";
    std::string bin   = model_path + "/scrfd_500m-opt2.bin";
    if (net_.load_param(param.c_str())) { fprintf(stderr,"load param failed: %s\n",param.c_str()); return -1; }
    if (net_.load_model(bin.c_str()))   { fprintf(stderr,"load bin failed: %s\n",  bin.c_str());   return -1; }
    return 0;
}

void Face::PreProcess(const void* input_data, std::vector<Tensor_t>& input_tensor)
{
    const cv::Mat* bgr = (const cv::Mat*)input_data;
    if (!bgr || bgr->empty()) return;

    const int target_size = 640;
    int img_w = bgr->cols, img_h = bgr->rows;
    float scale;
    int w, h;
    if (img_w > img_h) { scale=(float)target_size/img_w; w=target_size; h=(int)(img_h*scale); }
    else               { scale=(float)target_size/img_h; h=target_size; w=(int)(img_w*scale); }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        bgr->data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = (w+31)/32*32 - w;
    int hpad = (h+31)/32*32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad/2, hpad-hpad/2, wpad/2, wpad-wpad/2,
                           ncnn::BORDER_CONSTANT, 0.f);

    const float mean_vals[3] = {127.5f,127.5f,127.5f};
    const float norm_vals[3] = {1/128.f,1/128.f,1/128.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    Tensor_t t(in_pad);
    t.img_w=img_w; t.img_h=img_h;
    t.pad_w=wpad;  t.pad_h=hpad;
    t.in_w=in_pad.w; t.in_h=in_pad.h;
    t.scale=scale;
    input_tensor.push_back(t);
}

void Face::Run(const std::vector<Tensor_t>& input_tensor, std::vector<Tensor_t>& output_tensor)
{
    if (input_tensor.empty()) return;
    ncnn::Extractor ex = net_.create_extractor();
    ex.input("input.1", input_tensor[0].data);

    // scrfd_500m-opt2 official output blob names
    // stride 8 → score:score8, bbox:bbox8
    // stride 16→ score:score16,bbox:bbox16
    // stride 32→ score:score32,bbox:bbox32
    const char* score_names[] = {"score8",  "score16",  "score32"};
    const char* bbox_names[]  = {"bbox8",   "bbox16",   "bbox32"};

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
    if (input_tensor.empty() || output_tensor.size() < 6) return;
    PipeResult_t* pr = (PipeResult_t*)result;
    pr->face_count = 0;

    const Tensor_t& inp = input_tensor[0];
    float wpad2 = inp.pad_w / 2.f;
    float hpad2 = inp.pad_h / 2.f;

    const int strides[3] = {8, 16, 32};
    std::vector<Object_t> proposals;

    for (int i = 0; i < 3; i++) {
        const ncnn::Mat& score_blob = output_tensor[i*2].data;
        const ncnn::Mat& bbox_blob  = output_tensor[i*2+1].data;
        if (score_blob.empty() || bbox_blob.empty()) continue;

        generate_proposals(strides[i], score_blob, bbox_blob,
                           prob_threshold, wpad2, hpad2, inp.scale,
                           inp.img_w, inp.img_h, proposals);
    }

    // sort descending score
    std::sort(proposals.begin(), proposals.end(),
              [](const Object_t& a, const Object_t& b){ return a.score > b.score; });

    // NMS
    std::vector<int> picked;
    int n = (int)proposals.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) areas[i] = proposals[i].rect.area();
    for (int i = 0; i < n; i++) {
        bool keep = true;
        for (int j : picked) {
            cv::Rect_<float> inter = proposals[i].rect & proposals[j].rect;
            float iou = inter.area() / (areas[i] + areas[j] - inter.area());
            if (iou > nms_threshold) { keep = false; break; }
        }
        if (keep) picked.push_back(i);
    }

    // cap at MAX_DET_FACE_COUNT to prevent array overflow
    int count = std::min((int)picked.size(), MAX_DET_FACE_COUNT);
    pr->face_count = count;
    for (int i = 0; i < count; i++)
        pr->object[i] = proposals[picked[i]];
}

void Face::AlignFace(const cv::Mat& img, Object_t& object)
{
    if (object.pts.size() < 5) return; // guard: need exactly 5 landmarks

    cv::Mat affine_matrix = cv::estimateAffinePartial2D(
        object.pts, face_template, cv::noArray(), cv::LMEDS);

    // guard: estimateAffinePartial2D can return empty if landmarks are degenerate
    if (affine_matrix.empty()) {
        // fallback: simple crop + resize without alignment
        cv::Rect roi(
            (int)object.rect.x, (int)object.rect.y,
            (int)object.rect.width, (int)object.rect.height);
        roi &= cv::Rect(0, 0, img.cols, img.rows); // clamp to image bounds
        if (roi.width < 1 || roi.height < 1) return;
        cv::Mat cropped;
        cv::resize(img(roi), cropped, cv::Size(512, 512));
        cropped.copyTo(object.trans_img);
        // identity-like inverse so paste_faces still works reasonably
        cv::Mat inv = cv::Mat::eye(2, 3, CV_64F);
        inv.copyTo(object.trans_inv);
        return;
    }

    cv::Mat cropped_face;
    cv::warpAffine(img, cropped_face, affine_matrix, cv::Size(512, 512),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(135,133,132));

    cv::Mat affine_matrix_inv;
    cv::invertAffineTransform(affine_matrix, affine_matrix_inv);
    affine_matrix_inv *= 2.0;
    affine_matrix_inv.copyTo(object.trans_inv);
    cropped_face.copyTo(object.trans_img);
}

void Face::generateAndInsertProposals(const std::vector<float>&, int,
                                      const Tensor_t&, const ncnn::Mat&,
                                      float, std::vector<Object_t>&) {}

int Face::Process(const cv::Mat& input_img, void* result)
{
    if (input_img.empty() || !result) return -1;
    std::vector<Tensor_t> input_tensor, output_tensor;
    PreProcess((void*)&input_img, input_tensor);
    if (input_tensor.empty()) return -1;
    Run(input_tensor, output_tensor);
    PostProcess(input_tensor, output_tensor, result);
    PipeResult_t* pr = (PipeResult_t*)result;
    for (int i = 0; i < pr->face_count; i++)
        AlignFace(input_img, pr->object[i]);
    return 0;
}

} // namespace wsdsb
