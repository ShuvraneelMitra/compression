#include <opencv2/opencv.hpp>
#include <iostream>

#include "../metrics.hpp"
#include "../lossless/huffman.hpp"

/*
Tests for linking of OpenCV and running basic operations
*/

int main() {
    cv::Mat img = cv::imread("../test_img.bmp", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: could not load image\n";
        return -1;
    }

    std::cout << "Width: " << img.cols 
              << " Height: " << img.rows 
              << " Channels: " << img.channels() << std::endl;
    // cv::imshow("Display", img);
    // cv::waitKey(0);

    cv::Mat A = (cv::Mat_<float>(2, 2) << 
        1.0, 2.0,
        3.0, 4.0
    );

    cv::Mat B = (cv::Mat_<float>(2, 2) << 
        1.0, 1.0,
        1.0, 1.0
    );

    std::cout << "The MSE of the test matrices is " << metrics::MSE(A, B) << "\n";
    std::cout << "The SNR of the test matrices is " << metrics::SNR(A, B) << "\n";
    std::cout << "The PSNR of the test matrices is " << metrics::PSNR(A, B) << "\n";

    return 0;
}