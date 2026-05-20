#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "../metrics.hpp"
#include "../transform/subband.hpp"

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
    
    // HAAR
    cv::Mat haar_lpf = (cv::Mat1d(1, 2) << 
        1.0 / std::sqrt(2.0),
        1.0 / std::sqrt(2.0)
    );

    cv::Mat haar_hpf = (cv::Mat1d(1, 2) << 
        1.0 / std::sqrt(2.0),
       -1.0 / std::sqrt(2.0)
    );

    // DAUBECHIES 
    cv::Mat1d db8_lpf = (cv::Mat1d(1,16) <<
       -0.00011747678412476953,
        0.0006754494059985568,
       -0.00039174037337694705,
       -0.004870352993451574,
        0.008746094047405777,
        0.013981027917398282,
       -0.044088253930794755,
       -0.017369301001807547,
        0.12874742662018601,
        0.00047248457391233877,
       -0.2840155429615469,
       -0.015829105256349306,
        0.5853546836542067,
        0.6756307362972898,
        0.31287159091429995,
        0.05441584224308161
    );

    cv::Mat1d db8_hpf = (cv::Mat1d(1,16) <<
        0.05441584224308161,
       -0.31287159091429995,
        0.6756307362972898,
       -0.5853546836542067,
       -0.015829105256349306,
        0.2840155429615469,
        0.00047248457391233877,
       -0.12874742662018601,
       -0.017369301001807547,
        0.044088253930794755,
        0.013981027917398282,
       -0.008746094047405777,
       -0.004870352993451574,
        0.00039174037337694705,
        0.0006754494059985568,
        0.00011747678412476953
    );

    std::vector<cv::Mat1d> subbands = decompose_4(img, db8_lpf, haar_hpf);
    std::cout << "Each subband has height: " << subbands[0].rows << 
                 " and width: " << subbands[0].cols << "\n";

    cv::Mat1b LL, LH, HL, HH;
    cv::normalize(subbands[0], LL, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Subband 0: Approximation", LL);
    cv::normalize(subbands[1], LH, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Subband 1: Vertical", LH);
    cv::normalize(subbands[2], HL, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Subband 2: Horizontal", HL);
    cv::normalize(subbands[3], HH, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Subband 3: Diagonal", HH);
    cv::waitKey(0);

    cv::Mat1d reconstructed = recombine_4(subbands, db8_lpf, haar_hpf);
    cv::Mat1b reconstructed_vis;
    reconstructed.convertTo(reconstructed_vis, CV_8U);
    cv::imshow("Reconstructed", reconstructed_vis);
    cv::waitKey(0);

    std::cout << "\nThe MSE of the original image with reference to the reconstructed is "
              << metrics::MSE(img, reconstructed_vis);
    std::cout << "\nThe SNR of the original image with reference to the reconstructed is "
              << metrics::SNR(img, reconstructed_vis);      
    std::cout << "\nThe PSNR of the original image with reference to the reconstructed is "
              << metrics::PSNR(img, reconstructed_vis);      
    return 0;
}

