#pragma once

#include <opencv2/opencv.hpp>

#include <numbers>
#include <cmath>
#include <vector>
#include <complex>
#include <stdexcept>

#include "../lossy/quantizer.hpp"

using Complex = std::complex<float>;

enum class BitAlloc {
    UNIFORM,
    VARIANCE_BASED,
};

cv::Mat1s dct_opencv(const cv::Mat1b& A);
cv::Mat1s dct_slow(const cv::Mat1b& A, uint block_size);
cv::Mat1s dct_vetterli(const cv::Mat1b& A, uint block_size);
cv::Mat1s idct_fast(const cv::Mat1s& coeffs, uint block_size);

template<typename T>
int intify_and_saturate(T x, int low, int high){
    int x_int = static_cast<int>(x);
    if(x_int <= low) return low;
    if(x_int >= high) return high;
    return x_int;
}