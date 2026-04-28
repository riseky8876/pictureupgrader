#ifndef COLORNET_H_INCLUDED
#define COLORNET_H_INCLUDED

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#endif

// out_image must be non-const ref so copyTo() can write into it
int colorization(const cv::Mat& bgr, cv::Mat& out_image, const std::string& model_path);

#endif // COLORNET_H_INCLUDED
