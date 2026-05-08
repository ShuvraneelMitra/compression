#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <numeric>
#include <array>
#include <cmath>

enum class Mode {
    LINEAR,
};

enum class Norm_Mode {
    NONE,
    LINEAR,
    SOFTMAX,
};

/* LOSSLESS PREDICTIVE ENCODER
The currently received pixel A(i, j) is predicted as a function of the 
previously received pixels which are also spatial neighbours of the currently 
received pixel (i, j). Then the error in prediction is given as
e(i, j) = A(i, j) - A_hat(i, j).
This error is then encoded using some lossless compression technique.

Since the predictor can only consider the pixels received so far, and we
input pixels in a row-major fashion, the prediction is based on the pixels
from the current row from 0 to j - 1 and all the pixels from the previous
rows. We usually take only the spatial neighbours. Let us have the prediction
as a linear combination of these "closest" neighbours (linear predictive coding):

A_hat(i, j) = a1*A(i - 1, j - 1) + a2*A(i - 1, j) +
              a3*A(i - 1, j + 1) + a4*A(i, j - 1).

We set a1 + a2 + a3 + a4 = 1. If a4 = 1 and a1 = a2 = a3 = 0 we get a previous
pixel predictor. We always start with A_hat(0, 0) = 0.

Why do we do this? We directly take advantage of the spatial redunancy/smoothness in the image.
This is because the entropy in the error values is typically expected to be much smaller than
that in the pixel values, and hence we can achieve much better compression via 
an entropy-based encoding like Huffman/LZW.
*/

cv::Mat1b lossless_predict(const cv::Mat1b& A, Mode mode,
                  std::array<double, 4> coeff = {0.0, 0.0, 0.0, 0.0},
                  Norm_Mode norm_mode = Norm_Mode::LINEAR);
cv::Mat1s error_matrix(const cv::Mat1b& truth, const cv::Mat1b& pred);
cv::Mat1b lossless_decode(const cv::Mat1s& e, Mode mode,
                  std::array<double, 4> coeff = {0.0, 0.0, 0.0, 0.0},
                  Norm_Mode norm_mode = Norm_Mode::LINEAR);