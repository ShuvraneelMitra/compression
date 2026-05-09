#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "../metrics.hpp"
#include "../lossless/huffman.hpp"
#include "../lossy/lossy_pred.hpp"


int main() {
    cv::Mat1b img = cv::imread("../test_imgs/grayscale.bmp", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: could not load image\n";
        return -1;
    }

    std::cout << "Width: " << img.cols 
              << " Height: " << img.rows 
              << " Channels: " << img.channels() << std::endl;
    cv::imshow("Display", img);
    cv::waitKey(0);

    cv::Mat1s e = quantized_error(img, Mode::LINEAR, {0.0, 1.0, 3.0, 2.0},
                                  Norm_Mode::LINEAR, Q_Mode::UNIFORM, 134
                                );

    /*
    Don't need to worry about Huffman coding having to handle negative values of
    errors: for the algorithm it is simply another symbol having an associated
    probability value.
    */
    std::unique_ptr<Node> tree = std::move(huffman_init(e));
    std::unordered_map<int, std::string> table;
    huffman_codemap(tree.get(), table);
    huffman_encode(e, table, "intermediate.txt");
    
    cv::Mat out = huffman_decode(e.rows, e.cols, e.channels(), e.depth(), "intermediate.txt", tree.get());
    cv::Mat1b decoded = lossy_decode(out, Mode::LINEAR, 
                                    {0.0, 1.0, 3.0, 2.0}, Norm_Mode::LINEAR);
    cv::imshow("Decoded", decoded);
    cv::waitKey(0);

    std::filesystem::path p{"../test_imgs/grayscale.bmp"};
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