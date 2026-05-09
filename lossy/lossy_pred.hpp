#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <numeric>
#include <array>
#include <cmath>

#include "quantizer.hpp"

enum class Mode {
    LINEAR,
};

enum class Norm_Mode {
    NONE,
    LINEAR,
    SOFTMAX,
};

enum class Q_Mode {
    UNIFORM,
};

cv::Mat1s quantized_error(const cv::Mat1b& A, Mode mode,
                          std::array<double, 4> coeff,
                          Norm_Mode norm_mode = Norm_Mode::LINEAR,
                          Q_Mode quantization_mode = Q_Mode::UNIFORM,
                          int num_levels = 255
                         );
cv::Mat1b lossy_decode(const cv::Mat1s& e, Mode mode,
                  std::array<double, 4> coeff = {0.0, 0.0, 0.0, 0.0},
                  Norm_Mode norm_mode = Norm_Mode::LINEAR);