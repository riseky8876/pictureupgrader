#include <include/pipeline.h>
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "PU", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PU", __VA_ARGS__)

namespace wsdsb {

PipeLine::PipeLine() {}
PipeLine::~PipeLine() {}

static void paste_faces(const cv::Mat& restored_face,
                         cv::Mat& trans_inv, cv::Mat& bg) {
    if (restored_face.empty() || trans_inv.empty() || bg.empty()) return;

    cv::Mat inv = trans_inv.clone();
    inv.at<double>(0,2) += 1.0;
    inv.at<double>(1,2) += 1.0;

    cv::Mat inv_restored;
    cv::warpAffine(restored_face, inv_restored, inv, bg.size(), 1, 0);

    cv::Mat mask = cv::Mat::ones(512, 512, CV_8UC1) * 255;
    cv::Mat inv_mask;
    cv::warpAffine(mask, inv_mask, inv, bg.size(), 1, 0);

    cv::Mat inv_mask_erosion;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4,4));
    cv::erode(inv_mask, inv_mask_erosion, kernel);

    int total_face_area = cv::countNonZero(inv_mask_erosion);
    int w_edge = std::max(1, (int)(std::sqrt((float)total_face_area) / 20));
    int erosion_radius = std::max(2, w_edge * 2);
    int blur_size = std::max(1, w_edge * 2);
    if (blur_size % 2 == 0) blur_size++;

    cv::Mat inv_mask_center;
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                       cv::Size(erosion_radius, erosion_radius));
    cv::erode(inv_mask_erosion, inv_mask_center, kernel);

    cv::Mat inv_soft_mask;
    cv::GaussianBlur(inv_mask_center, inv_soft_mask, cv::Size(blur_size, blur_size), 0, 0, 4);

    cv::Mat inv_soft_mask_f;
    inv_soft_mask.convertTo(inv_soft_mask_f, CV_32F, 1/255.f);

    cv::Mat pasted;
    cv::bitwise_and(inv_restored, inv_restored, pasted, inv_mask_erosion);

    for (int h = 0; h < bg.rows; h++) {
        cv::Vec3b* bg_ptr   = bg.ptr<cv::Vec3b>(h);
        cv::Vec3b* face_ptr = pasted.ptr<cv::Vec3b>(h);
        float*     mask_ptr = inv_soft_mask_f.ptr<float>(h);
        for (int w2 = 0; w2 < bg.cols; w2++) {
            float m = mask_ptr[w2];
            bg_ptr[w2][0] = (uchar)(bg_ptr[w2][0]*(1-m) + face_ptr[w2][0]*m);
            bg_ptr[w2][1] = (uchar)(bg_ptr[w2][1]*(1-m) + face_ptr[w2][1]*m);
            bg_ptr[w2][2] = (uchar)(bg_ptr[w2][2]*(1-m) + face_ptr[w2][2]*m);
        }
    }
}

int PipeLine::CreatePipeLine(PipelineConfig_t& cfg) {
    pipeline_config_ = cfg;
    return 0; // defer model loading to Apply() to avoid holding all models in RAM
}

int PipeLine::Apply(const cv::Mat& input_img, cv::Mat& output_img) {
    const std::string& mp = pipeline_config_.model_path;
    LOGI("Apply: start, image %dx%d", input_img.cols, input_img.rows);

    // ── Step 1: Background upsampling (load → run → free) ────────────────
    cv::Mat bg_upsample;
    if (pipeline_config_.bg_upsample) {
        LOGI("Apply: loading RealESRGAN");
        auto esrgan = std::make_unique<RealESRGAN>();
        if (esrgan->Load(mp) < 0) {
            LOGE("Apply: RealESRGAN load failed, using resize fallback");
            cv::resize(input_img, bg_upsample,
                       cv::Size(input_img.cols*2, input_img.rows*2), 0, 0, cv::INTER_LINEAR);
        } else {
            LOGI("Apply: running RealESRGAN");
            esrgan->Process(input_img, (void*)&bg_upsample);
            LOGI("Apply: RealESRGAN done");
        }
        esrgan.reset(); // free model memory immediately
    } else {
        cv::resize(input_img, bg_upsample,
                   cv::Size(input_img.cols*2, input_img.rows*2), 0, 0, cv::INTER_LINEAR);
    }
    LOGI("Apply: bg_upsample %dx%d", bg_upsample.cols, bg_upsample.rows);

    // ── Step 2: Face detection (load → run → free) ────────────────────────
    LOGI("Apply: loading Face detector");
    auto face_det = std::make_unique<Face>();
    if (face_det->Load(mp) < 0) {
        LOGE("Apply: Face load failed, skipping face enhancement");
        bg_upsample.copyTo(output_img);
        return 0;
    }
    auto pipe_result = std::make_unique<PipeResult_t>();
    pipe_result->face_count = 0;
    face_det->Process(input_img, pipe_result.get());
    face_det.reset(); // free face model
    LOGI("Apply: detected %d faces", pipe_result->face_count);

    // ── Step 3: CodeFormer per face (load once, run N faces, free) ────────
    if (pipe_result->face_count > 0) {
        LOGI("Apply: loading CodeFormer");
        auto codeformer = std::make_unique<CodeFormer>();
        if (codeformer->Load(mp) < 0) {
            LOGE("Apply: CodeFormer load failed, skipping face enhancement");
        } else {
            for (int i = 0; i < pipe_result->face_count; i++) {
                LOGI("Apply: processing face %d/%d", i+1, pipe_result->face_count);
                codeformer->Process(pipe_result->object[i].trans_img,
                                    pipe_result->codeformer_result[i]);
                paste_faces(pipe_result->codeformer_result[i].restored_face,
                            pipe_result->object[i].trans_inv,
                            bg_upsample);
                // free restored face immediately to save RAM
                pipe_result->codeformer_result[i].restored_face.release();
            }
        }
        codeformer.reset(); // free CodeFormer
    }

    bg_upsample.copyTo(output_img);
    LOGI("Apply: done, output %dx%d", output_img.cols, output_img.rows);
    return 0;
}

} // namespace wsdsb
