#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "../lossless/lempel_ziv.hpp"

int main() {
    cv::Mat img = cv::imread("../low_entropy.bmp", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: could not load image\n";
        return -1;
    }

    std::cout << "Width: " << img.cols 
              << " Height: " << img.rows 
              << " Channels: " << img.channels() << std::endl;
    cv::imshow("Display", img);
    cv::waitKey(0);
    
    cv::Mat A = (cv::Mat_<uchar>(2, 4) << 
        32, 32, 34, 32, 
        34, 32, 32, 33
    );

    lz_encode(img, "intermediate.txt");
    cv::Mat out = lz_decode(img.rows, img.cols, img.channels(), img.depth(), "intermediate.txt");

    cv::imshow("Decoded", out);
    cv::waitKey(0);
    
    std::filesystem::path p{"../low_entropy.bmp"};
    // Returns file size in bytes as std::uintmax_t
    auto init_size = std::filesystem::file_size(p); 
    std::cout << "Original file size: " << init_size << " bytes\n";

    std::filesystem::path q{"intermediate.txt"};  
    auto fin_size = std::filesystem::file_size(q); 
    std::cout << "Encoded file size: " << fin_size << " bytes\n";

    std::cout << "Total reduction = " << (init_size - fin_size) * 100 / init_size << "%\n"; 
    return 0;
}