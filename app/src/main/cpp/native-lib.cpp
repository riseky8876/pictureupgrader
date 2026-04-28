#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <include/pipeline.h>
#include <include/colornet.h>

#define LIMIT_PIX 1500

class JniString {
public:
    JniString(JNIEnv *env, jstring str) : env_(env), str_(str), chars_(env->GetStringUTFChars(str, JNI_FALSE)) {}
    ~JniString() { env_->ReleaseStringUTFChars(str_, chars_); }
    const char* c_str() const { return chars_; }
private:
    JNIEnv *env_;
    jstring str_;
    const char* chars_;
};

extern "C" {

jboolean processImage(JNIEnv *env, jstring in_path, jstring out_path, jstring model_dir,
                      std::function<jboolean(const cv::Mat&, cv::Mat&, const char*)> processFunc) {
    jboolean result = JNI_FALSE;
    JniString imagepath(env, in_path);
    JniString outpath(env, out_path);
    JniString mdir(env, model_dir);

    try {
        cv::Mat img = cv::imread(imagepath.c_str(), 1);
        cv::Mat out_image;
        if (img.empty()) {
            fprintf(stderr, "cv::imread %s failed\n", imagepath.c_str());
            throw std::runtime_error("Failed to read image");
        }
        result = processFunc(img, out_image, mdir.c_str());
        if (result == JNI_TRUE)
            cv::imwrite(outpath.c_str(), out_image);
    } catch (const std::exception& e) {
        fprintf(stderr, "Caught exception: %s\n", e.what());
        result = JNI_FALSE;
    }
    return result;
}

JNIEXPORT jboolean JNICALL
Java_com_ernesto_pictureupgrader_MainActivity_imgSupResolution(JNIEnv *env, jobject,
                                                               jstring in_path, jstring out_path,
                                                               jstring model_dir) {
    return processImage(env, in_path, out_path, model_dir,
        [](const cv::Mat& img, cv::Mat& out_image, const char* mdir) {
            wsdsb::PipelineConfig_t cfg;
            wsdsb::PipeLine pipe;
            cfg.model_path = std::string(mdir);
            cfg.bg_upsample = true;
            if (pipe.CreatePipeLine(cfg) < 0) return JNI_FALSE;

            // FIX: use separate in/out variables — pipe.Apply(x, x) is undefined behavior
            int w = img.cols, h = img.rows;
            cv::Mat resized;
            if (w >= h && w > LIMIT_PIX) {
                int nw = LIMIT_PIX, nh = (int)((double)LIMIT_PIX / w * h);
                cv::resize(img, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
            } else if (h > w && h > LIMIT_PIX) {
                int nh = LIMIT_PIX, nw = (int)((double)LIMIT_PIX / h * w);
                cv::resize(img, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
            } else {
                resized = img.clone();
            }
            pipe.Apply(resized, out_image);  // separate src/dst
            return JNI_TRUE;
        });
}

JNIEXPORT jboolean JNICALL
Java_com_ernesto_pictureupgrader_MainActivity_imgColouration(JNIEnv *env, jobject,
                                                             jstring in_path, jstring out_path,
                                                             jstring model_dir) {
    return processImage(env, in_path, out_path, model_dir,
        [](const cv::Mat& img, cv::Mat& out_image, const char* mdir) {
            if (colorization(img, out_image, mdir) < 0) return JNI_FALSE;
            return JNI_TRUE;
        });
}

} // extern "C"
