#include <jni.h>
#include <string>
#include <signal.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <include/pipeline.h>
#include <include/colornet.h>

#define TAG "PU"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static char g_crash_log_path[512] = {0};
static char g_step_log_path[512]  = {0};

static void write_step(const char* step) {
    LOGI("STEP: %s", step);
    if (g_step_log_path[0]) {
        FILE* f = fopen(g_step_log_path, "w");
        if (f) { fprintf(f, "%s\n", step); fclose(f); }
    }
}

static void crash_handler(int sig) {
    char last_step[256] = "unknown";
    if (g_step_log_path[0]) {
        FILE* f = fopen(g_step_log_path, "r");
        if (f) { fgets(last_step, sizeof(last_step), f); fclose(f); }
    }
    if (g_crash_log_path[0]) {
        FILE* f = fopen(g_crash_log_path, "w");
        if (f) {
            fprintf(f, "CRASH: signal %d (%s)\nLast step: %s\n", sig, strsignal(sig), last_step);
            fclose(f);
        }
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

extern "C" {

JNIEXPORT void JNICALL
Java_com_ernesto_pictureupgrader_MainActivity_initCrashHandler(
        JNIEnv* env, jobject, jstring crash_log_path) {
    const char* p = env->GetStringUTFChars(crash_log_path, nullptr);
    strncpy(g_crash_log_path, p, sizeof(g_crash_log_path)-1);
    snprintf(g_step_log_path, sizeof(g_step_log_path), "%s.step", p);
    env->ReleaseStringUTFChars(crash_log_path, p);
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    signal(SIGBUS,  crash_handler);
    signal(SIGILL,  crash_handler);
}

// ── Super Resolution ──────────────────────────────────────────────────────────
JNIEXPORT jboolean JNICALL
Java_com_ernesto_pictureupgrader_MainActivity_imgSupResolution(
        JNIEnv* env, jobject,
        jstring in_path, jstring out_path, jstring model_dir) {

    const char* imagepath = env->GetStringUTFChars(in_path,   nullptr);
    const char* outpath   = env->GetStringUTFChars(out_path,  nullptr);
    const char* mdir      = env->GetStringUTFChars(model_dir, nullptr);
    jboolean result = JNI_FALSE;

    write_step("SR: imread");
    cv::Mat img = cv::imread(imagepath, cv::IMREAD_COLOR);
    if (img.empty()) { LOGE("imread failed"); goto done_sr; }

    {
        // Aggressively limit input size to avoid OOM
        // Each pixel = 3 bytes, ncnn needs ~10x for intermediate buffers
        // 600x600 * 3 * 10 = ~10MB — safe for most devices
        write_step("SR: resize");
        cv::Mat resized;
        const int LIMIT = 600;
        int w = img.cols, h = img.rows;
        if (w >= h && w > LIMIT) {
            cv::resize(img, resized, cv::Size(LIMIT, (int)((double)LIMIT/w*h)));
        } else if (h > w && h > LIMIT) {
            cv::resize(img, resized, cv::Size((int)((double)LIMIT/h*w), LIMIT));
        } else {
            resized = img.clone();
        }
        img.release(); // free original immediately
        LOGI("SR: resized to %dx%d", resized.cols, resized.rows);

        write_step("SR: CreatePipeLine");
        wsdsb::PipelineConfig_t cfg;
        cfg.model_path  = std::string(mdir);
        cfg.bg_upsample = true;
        wsdsb::PipeLine pipe;
        if (pipe.CreatePipeLine(cfg) < 0) { LOGE("CreatePipeLine failed"); goto done_sr; }

        write_step("SR: Apply");
        cv::Mat out_image;
        pipe.Apply(resized, out_image);
        resized.release();

        write_step("SR: imwrite");
        if (out_image.empty()) { LOGE("out_image empty"); goto done_sr; }
        cv::imwrite(outpath, out_image);
        result = JNI_TRUE;
        write_step("SR: done");
    }

done_sr:
    env->ReleaseStringUTFChars(in_path,   imagepath);
    env->ReleaseStringUTFChars(out_path,  outpath);
    env->ReleaseStringUTFChars(model_dir, mdir);
    return result;
}

// ── Colourisation ─────────────────────────────────────────────────────────────
JNIEXPORT jboolean JNICALL
Java_com_ernesto_pictureupgrader_MainActivity_imgColouration(
        JNIEnv* env, jobject,
        jstring in_path, jstring out_path, jstring model_dir) {

    const char* imagepath = env->GetStringUTFChars(in_path,   nullptr);
    const char* outpath   = env->GetStringUTFChars(out_path,  nullptr);
    const char* mdir      = env->GetStringUTFChars(model_dir, nullptr);
    jboolean result = JNI_FALSE;

    write_step("COL: imread");
    cv::Mat img = cv::imread(imagepath, cv::IMREAD_COLOR);
    if (img.empty()) { LOGE("imread failed"); goto done_col; }

    {
        // Colourisation model (256x256 input) is lightweight,
        // but output blending needs original size — cap at 1000px
        write_step("COL: resize");
        cv::Mat resized;
        const int LIMIT = 1000;
        int w = img.cols, h = img.rows;
        if (w >= h && w > LIMIT) {
            cv::resize(img, resized, cv::Size(LIMIT, (int)((double)LIMIT/w*h)));
        } else if (h > w && h > LIMIT) {
            cv::resize(img, resized, cv::Size((int)((double)LIMIT/h*w), LIMIT));
        } else {
            resized = img.clone();
        }
        img.release();
        LOGI("COL: resized to %dx%d", resized.cols, resized.rows);

        write_step("COL: colorization()");
        cv::Mat out_image;
        int ret = colorization(resized, out_image, std::string(mdir));
        resized.release();
        if (ret < 0) { LOGE("colorization failed: %d", ret); goto done_col; }
        if (out_image.empty()) { LOGE("out_image empty"); goto done_col; }

        write_step("COL: imwrite");
        cv::imwrite(outpath, out_image);
        result = JNI_TRUE;
        write_step("COL: done");
    }

done_col:
    env->ReleaseStringUTFChars(in_path,   imagepath);
    env->ReleaseStringUTFChars(out_path,  outpath);
    env->ReleaseStringUTFChars(model_dir, mdir);
    return result;
}

} // extern "C"
