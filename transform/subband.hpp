#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

std::vector<cv::Mat1d> decompose_4(const cv::Mat& A, const cv::Mat& lpf, const cv::Mat& hpf);
cv::Mat1d recombine_4(const std::vector<cv::Mat1d>& subbands, const cv::Mat& lpf, const cv::Mat& hpf);
