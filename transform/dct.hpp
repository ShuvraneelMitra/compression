#pragma once

#include <opencv2/opencv.hpp>

#include <numbers>
#include <cmath>
#include <vector>
#include <complex>
#include <stdexcept>

using Complex = std::complex<float>;

cv::Mat1s dct_opencv(const cv::Mat1b& A);
cv::Mat1s dct_slow(const cv::Mat1b& A, uint block_size);
cv::Mat1s dct_vetterli(const cv::Mat1b& A, uint block_size);
cv::Mat1s idct_fast(const cv::Mat1s& coeffs, uint block_size);