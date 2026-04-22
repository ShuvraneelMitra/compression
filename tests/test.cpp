#include <opencv2/opencv.hpp>
#include <iostream>

#include "../metrics.hpp"
#include "../lossless/huffman.hpp"

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
    
    std::unique_ptr<Node> tree = std::move(huffman_init(img));
    std::unordered_map<int, std::string> table;
    huffman_codemap(tree.get(), table);
    huffman_encode(img, table, "intermediate.txt");
    cv::Mat out = huffman_decode(img.rows, img.cols, img.channels(), img.depth(), "intermediate.txt", tree.get());
    
    std::cout << "\nWidth: " << out.cols 
              << " Height: " << out.rows 
              << " Channels: " << img.channels() << std::endl;
    cv::imshow("Decoded", out);
    cv::waitKey(0);

    return 0;
}