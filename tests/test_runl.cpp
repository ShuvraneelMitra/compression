#include <opencv2/opencv.hpp>

#include <iostream>
#include <filesystem>


#include "../lossless/run_length.hpp"


int main() {
    cv::Mat img = cv::imread("../test_imgs/grayscale.bmp", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: could not load image\n";
        return -1;
    }

    std::cout << "Width: " << img.cols 
              << " Height: " << img.rows 
              << " Channels: " << img.channels() << std::endl;
    cv::imshow("Display", img);
    cv::waitKey(0);
    
    std::string s = "0001101000111110010111111111111111011100000000010000111111100000001";
    run_length_encode(s, "out.txt");

    std::vector<char> v(s.begin(), s.end());
    run_length_encode(v.begin(), v.end(), "out1.txt");
    return 0;
}