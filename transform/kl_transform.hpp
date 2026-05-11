#pragma once

#include <opencv2/opencv.hpp>

#include <vector>
#include <tuple>
#include <algorithm>

std::tuple<cv::Mat1d, cv::Mat1d, cv::Mat1d, cv::Mat1d>
kl_transform(cv::Mat& img, uint block_size, uint top_k);

cv::Mat1b kl_reconstruct(uint img_dim, cv::Mat& coeffs, 
                         cv::Mat& mu, cv::Mat& gamma_topk, uint block_size);

// It is recommended to keep the block size pretty small, <= 16, because
// finding the eigenvectors of a 32 x 32 matrix is really expensive since it
// is an O(N^3) operation and will take a huge amount of time.