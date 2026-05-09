#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "../metrics.hpp"
#include "../lossless/huffman.hpp"

int main() {
    cv::Mat img = cv::imread("../test_imgs/grayscale.bmp", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: could not load image\n";
        return -1;
    }

    std::cout << "Width: " << img.cols 
              << " Height: " << img.rows 
              << " Channels: " << img.channels() << std::endl;
    cv::imshow("Display", img);
    cv::waitKey(0);
    
    std::unique_ptr<Node> tree = std::move(huffman_init(img));
    std::unordered_map<int, std::string> table;
    huffman_codemap(tree.get(), table);
    huffman_encode(img, table, "intermediate.txt");
    cv::Mat out = huffman_decode(img.rows, img.cols, img.channels(), img.depth(), "intermediate.txt", tree.get());
    
    cv::imshow("Decoded", out);
    cv::waitKey(0);

    std::filesystem::path p{"../test_imgs/grayscale.bmp"};
    // Returns file size in bytes as std::uintmax_t
    auto init_size = std::filesystem::file_size(p); 
    std::cout << "Original file size: " << init_size << " bytes\n";

    std::filesystem::path q{"intermediate.txt"};  
    auto fin_size = std::filesystem::file_size(q); 
    std::cout << "Encoded file size: " << fin_size << " bytes\n";

    std::cout << "Total reduction = " << (init_size - fin_size) * 100 / init_size << "%\n"; 

    std::cout << "\nThe MSE of the original image with reference to the predicted is "
              << metrics::MSE(img, out);
    std::cout << "\nThe SNR of the original image with reference to the predicted is "
              << metrics::SNR(img, out);      
    std::cout << "\nThe PSNR of the original image with reference to the predicted is "
              << metrics::PSNR(img, out);      
    return 0;
}