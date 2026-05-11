#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "../metrics.hpp"
#include "../transform/kl_transform.hpp"
#include "../lossless/huffman.hpp"
#include "../lossy/quantizer.hpp"


int main() {
    cv::Mat1b img = cv::imread("../test_imgs/512x512.bmp", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: could not load image\n";
        return -1;
    }

    std::cout << "Width: " << img.cols 
              << " Height: " << img.rows 
              << " Channels: " << img.channels() << std::endl;
    cv::imshow("Display", img);
    cv::waitKey(0);

    const int BLOCK_SIZE = 16;

    std::tuple<cv::Mat1d, cv::Mat1d, cv::Mat1d, cv::Mat1d> tup = 
                        kl_transform(img, BLOCK_SIZE, 40);
    /*
    Don't need to worry about Huffman coding having to handle negative values of
    errors: for the algorithm it is simply another symbol having an associated
    probability value.
    */
    cv::Mat1d coeffs = std::get<0>(tup);

    UniformQuantizer<uchar, double> uq(0, 255, 100);
    cv::Mat1b quantized_coeffs = uq.quantize(coeffs);

    std::unique_ptr<Node> tree = std::move(huffman_init(coeffs));
    std::unordered_map<int, std::string> table;
    huffman_codemap(tree.get(), table);
    huffman_encode(coeffs, table, "intermediate.txt");
    
    cv::Mat out = huffman_decode(coeffs.rows, coeffs.cols, coeffs.channels(), coeffs.depth(), "intermediate.txt", tree.get());
    cv::Mat1b decoded = kl_reconstruct(img.rows, coeffs, std::get<2>(tup), std::get<3>(tup), BLOCK_SIZE);
    cv::imshow("Decoded", decoded);
    cv::waitKey(0);

    std::filesystem::path p{"../test_imgs/512x512.bmp"};
    // Returns file size in bytes as std::uintmax_t
    auto init_size = std::filesystem::file_size(p); 
    std::cout << "Original file size: " << init_size << " bytes\n";

    std::filesystem::path q{"intermediate.txt"};  
    auto fin_size = std::filesystem::file_size(q); 
    std::cout << "Encoded file size: " << fin_size << " bytes\n";

    std::cout << "Total reduction = " << (init_size - fin_size) * 100 / init_size << "%\n"; 

    std::cout << "\nThe MSE of the original image with reference to the reconstructed is "
              << metrics::MSE(img, decoded);
    std::cout << "\nThe SNR of the original image with reference to the reconstructed is "
              << metrics::SNR(img, decoded);      
    std::cout << "\nThe PSNR of the original image with reference to the reconstructed is "
              << metrics::PSNR(img, decoded);      
}