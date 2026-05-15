#include "../transform/dct.hpp"
#include <filesystem>

int main() {
    cv::Mat1b input = (cv::Mat1b(8,8) <<
        166,162,162,160,155,163,160,155,
        166,162,162,160,155,163,160,155,
        166,162,162,160,155,163,160,155,
        166,162,162,160,155,163,160,155,
        166,162,162,160,155,163,160,155,
        161,160,155,159,154,154,156,154,
        159,163,158,163,155,155,156,152,
        159,162,162,160,153,153,153,151
    );

    cv::Mat1d shifted;
    input.convertTo(shifted, CV_64F);

    shifted -= 128.0;

    std::cout << "Shifted block:\n\n";
    std::cout << shifted << "\n\n";

    cv::Mat1s dct_coeffs(8, 8);

    dct_coeffs = dct_vetterli(shifted, 8);

    std::cout << "Computed DCT coefficients:\n\n";
    std::cout << dct_coeffs << "\n\n";

    cv::Mat1s expected = (cv::Mat1s(8,8) <<
        248, 19,  3,  4, -7,  9,  1, -7,
         11, -2,  3,  6, -3,  2,  5,  0,
         -4,  2, -2, -3,  0, -1, -1,  0,
         -1, -1,  1,  1,  2,  0, -1,  0,
          2,  1,  0,  0, -2,  0,  3,  0,
          0,  0, -1,  0,  0,  0, -1, -1,
         -3,  0,  1,  0,  1,  0,  0,  0,
          3,  0,  0,  0, -1,  0,  0,  0
    );

    std::cout << "Expected coefficients:\n\n";
    std::cout << expected << "\n\n";

    cv::Mat1s diff = dct_coeffs - expected;

    std::cout << "Difference:\n\n";
    std::cout << diff << "\n\n";

    double max_error;
    cv::minMaxLoc(cv::abs(diff), nullptr, &max_error);

    std::cout << "Maximum absolute error: "
              << max_error << "\n";
    // A maximum absolute error of 1 is acceptable due to rounding errors

    cv::Mat1f reconstructed = idct_fast(dct_coeffs, 8);
    cv::Mat1b reconstructed_u8;
    reconstructed.convertTo(reconstructed_u8, CV_8U);

    std::cout << "Reconstructed:\n\n";
    std::cout << reconstructed_u8 << "\n\n";

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

    const int BLOCK_SIZE = 512;

    dct_coeffs = dct_vetterli(img, BLOCK_SIZE);
    cv::Mat1f vis;
    dct_coeffs.convertTo(vis, CV_32F);

    cv::Mat1f abs_vis = cv::abs(vis);
    abs_vis += 1.0f;
    cv::log(abs_vis, abs_vis);

    cv::normalize(abs_vis, abs_vis, 0, 255, cv::NORM_MINMAX);
    cv::Mat1b display;
    abs_vis.convertTo(display, CV_8U);

    cv::imshow("DCT Magnitude", display);
    cv::waitKey(0);


    reconstructed = idct_fast(dct_coeffs, BLOCK_SIZE);
    reconstructed.convertTo(reconstructed_u8, CV_8U);

    cv::imshow("Reconstructed", reconstructed_u8);
    cv::waitKey(0);
    return 0;
}