#pragma once

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

std::vector<cv::Mat1d> decompose(const cv::Mat& A, const cv::Mat& lpf, const cv::Mat& hpf);
std::vector<cv::Mat1d> decompose(const cv::Mat& A, const cv::Mat& lpf, const cv::Mat& hpf, uint levels);
cv::Mat1d recombine(const std::vector<cv::Mat1d>& subbands, const cv::Mat& lpf, const cv::Mat& hpf);
cv::Mat1d recombine(const std::vector<cv::Mat1d>& subbands, const cv::Mat& lpf, const cv::Mat& hpf, uint levels);

cv::Mat1d decomposeLL(const cv::Mat& A, const cv::Mat& lpf, const cv::Mat& hpf, uint levels);

