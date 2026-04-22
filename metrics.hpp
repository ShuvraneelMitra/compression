#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <sstream>
#include <stdexcept>

namespace metrics {
    double MSE(cv::Mat& x, cv::Mat& y);
    double SNR(cv::Mat& x, cv::Mat& y);
    double PSNR(cv::Mat& x, cv::Mat& y);
}
