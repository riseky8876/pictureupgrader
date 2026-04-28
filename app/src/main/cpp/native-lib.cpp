#include <jni.h>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <include/pipeline.h>
#include <include/colornet.h>

#define TAG "PictureUpgrader"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)

#define LIMIT_PIX 1000  // reduced from 1500 to lower memory pressure

// ── crash log file path (written by signal handler, read by Kotlin) ──────────
static char g_crash_log_path[512] = {0};

static void crash_handler(int sig) {
    if (g_crash_log_path[0]) {
        FILE* f = fopen(g_crash_log_path, "w");
        if (f) {
            fprintf(f, "CRASH: signal %d (%s)\n", sig, strsignal(sig));
            fprintf(f, "This usually means: out-of-memory, null pointer, or stack overflow\n");
            fclose(f);
        }
    }
    // re-raise so Android gets the real crash dump too
    signal(sig, SIG_DFL);
    raise(sig);
}

extern "C" {

JNIEXPORT void JNICALL
Java_com_ernesto_pictureupgrader_MainActivity_initCrashHandler(
        JNIEnv* env, jobject, jstring crash_log_path) {
    const char* path = env->GetStringUTFChars(crash_log_path, nullptr);
    strncpy(g_crash_log_path, path, sizeof(g_crash_log_path) - 1);
    env->ReleaseStringUTFChars(crash_log_path, path);
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    signal(SIGBUS,  crash_handler);
    signal(SIGILL,  crash_handler);
    LOGI("Crash handler installed, log: %s", g_crash_log_path);
}

// ── Super Resolution ─────────────────────────────────────────────────────────
JNIEXPORT jboolean JNICALL
Java_com_ernesto_pictureupgrader_MainActivity_imgSupResolution(
        JNIEnv* env, jobject,
        jstring in_path, jstring out_path, jstring model_dir) {

    const char* imagepath = env->GetStringUTFChars(in_path,    nullptr);
    const char* outpath   = env->GetStringUTFChars(out_path,   nullptr);
    const char* mdir      = env->GetStringUTFChars(model_dir,  nullptr);

    LOGI("SR start | in=%s out=%s model=%s", imagepath, outpath, mdir);
    jboolean result = JNI_FALSE;

    try {
        // 1. read image
        cv::Mat img = cv::imread(imagepath, cv::IMREAD_COLOR);
        if (img.empty()) {
            LOGE("SR: imread failed: %s", imagepath);
            goto cleanup_sr;
        }
        LOGI("SR: image loaded %dx%d", img.cols, img.rows);

        // 2. resize if too large (in → resized, never apply in-place)
        {
            cv::Mat resized;
            int w = img.cols, h = img.rows;
            if (w >= h && w > LIMIT_PIX) {
                int nw = LIMIT_PIX, nh = (int)((double)LIMIT_PIX / w * h);
                cv::resize(img, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
                LOGI("SR: resized to %dx%d", nw, nh);
            } else if (h > w && h > LIMIT_PIX) {
                int nh = LIMIT_PIX, nw = (int)((double)LIMIT_PIX / h * w);
                cv::resize(img, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
                LOGI("SR: resized to %dx%d", nw, nh);
            } else {
                resized = img.clone();
            }

            // 3. create pipeline
            wsdsb::PipelineConfig_t cfg;
            cfg.model_path  = std::string(mdir);
            cfg.bg_upsample = true;

            wsdsb::PipeLine pipe;
            LOGI("SR: loading models...");
            if (pipe.CreatePipeLine(cfg) < 0) {
                LOGE("SR: CreatePipeLine failed (model load error)");
                goto cleanup_sr;
            }
            LOGI("SR: models loaded, running inference...");

            // 4. apply — ALWAYS separate src/dst
            cv::Mat out_image;
            pipe.Apply(resized, out_image);

            if (out_image.empty()) {
                LOGE("SR: out_image is empty after Apply");
                goto cleanup_sr;
            }

            // 5. write output
            if (!cv::imwrite(outpath, out_image)) {
                LOGE("SR: imwrite failed: %s", outpath);
                goto cleanup_sr;
            }
            LOGI("SR: done, wrote %s", outpath);
            result = JNI_TRUE;
        }

    } catch (const std::exception& e) {
        LOGE("SR: exception: %s", e.what());
    } catch (...) {
        LOGE("SR: unknown exception");
    }

cleanup_sr:
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

    const char* imagepath = env->GetStringUTFChars(in_path,    nullptr);
    const char* outpath   = env->GetStringUTFChars(out_path,   nullptr);
    const char* mdir      = env->GetStringUTFChars(model_dir,  nullptr);

    LOGI("Colour start | in=%s", imagepath);
    jboolean result = JNI_FALSE;

    try {
        cv::Mat img = cv::imread(imagepath, cv::IMREAD_COLOR);
        if (img.empty()) {
            LOGE("Colour: imread failed: %s", imagepath);
            goto cleanup_col;
        }
        LOGI("Colour: image loaded %dx%d", img.cols, img.rows);

        {
            cv::Mat out_image;
            int ret = colorization(img, out_image, std::string(mdir));
            if (ret < 0) {
                LOGE("Colour: colorization() returned %d", ret);
                goto cleanup_col;
            }
            if (out_image.empty()) {
                LOGE("Colour: out_image empty after colorization");
                goto cleanup_col;
            }
            if (!cv::imwrite(outpath, out_image)) {
                LOGE("Colour: imwrite failed: %s", outpath);
                goto cleanup_col;
            }
            LOGI("Colour: done, wrote %s", outpath);
            result = JNI_TRUE;
        }

    } catch (const std::exception& e) {
        LOGE("Colour: exception: %s", e.what());
    } catch (...) {
        LOGE("Colour: unknown exception");
    }

cleanup_col:
    env->ReleaseStringUTFChars(in_path,   imagepath);
    env->ReleaseStringUTFChars(out_path,  outpath);
    env->ReleaseStringUTFChars(model_dir, mdir);
    return result;
}

} // extern "C"
