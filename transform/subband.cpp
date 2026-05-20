#include "subband.hpp"

cv::Mat1d downsample_cols(const cv::Mat1d& input) {
    cv::Mat1d output(input.rows, (input.cols + 1) / 2);

    for(int r = 0; r < input.rows; ++r) {
        for(int c = 0; c < output.cols; ++c) {
            output(r, c) = input(r, 2 * c);
        }
    }

    return output;
}

cv::Mat1d downsample_rows(const cv::Mat1d& input) {
    cv::Mat1d output((input.rows + 1) / 2, input.cols);

    for(int r = 0; r < output.rows; ++r) {
        for(int c = 0; c < input.cols; ++c) {
            output(r, c) = input(2 * r, c);
        }
    }

    return output;
}

cv::Mat1d upsample_cols(const cv::Mat1d& input) {
    cv::Mat1d output(input.rows, 2 * input.cols );

    for(int r = 0; r < input.rows; ++r) {
        for(int c = 0; c < output.cols; ++c) {
            if(c % 2 == 1) output(r, c) = 0;
            else output(r, c) = input(r, c / 2); 
        }
    }

    return output;
}

cv::Mat1d upsample_rows(const cv::Mat1d& input) {
    cv::Mat1d output(2 * input.rows, input.cols);

    for(int r = 0; r < output.rows; ++r) {
        for(int c = 0; c < input.cols; ++c) {
            if(r % 2 == 1) output(r, c) = 0;
            else output(r, c) = input(r / 2, c); 
        }
    }

    return output;
}

std::vector<cv::Mat1d> decompose_4(const cv::Mat& A, const cv::Mat& lpf, const cv::Mat& hpf){
    /*
    2D wavelet decomposition using separable filter banks:

    1. Apply low-pass (h0) and high-pass (h1) filters along rows
    2. Downsample by 2 horizontally
    3. Apply low-pass and high-pass filters along columns
    4. Downsample by 2 vertically

    This produces four subbands:
    LL -> approximation
    LH -> vertical details
    HL -> horizontal details
    HH -> diagonal details
    */
    cv::Mat1d img;
    A.convertTo(img, CV_64FC1);

    CV_Assert(lpf.rows == 1 || lpf.cols == 1);
    CV_Assert(hpf.rows == 1 || hpf.cols == 1);

    cv::Mat1d img64;
    img.convertTo(img64, CV_64F);

    cv::Mat1d lpf64, hpf64;
    lpf.convertTo(lpf64, CV_64F);
    hpf.convertTo(hpf64, CV_64F);

    cv::Mat row_lpf = lpf64.reshape(1, 1);
    cv::Mat row_hpf = hpf64.reshape(1, 1);

    cv::Mat col_lpf = row_lpf.t();
    cv::Mat col_hpf = row_hpf.t();

    cv::Mat1d L, H;
    cv::filter2D(img64, L, CV_64F, col_lpf);
    cv::filter2D(img64, H, CV_64F, col_hpf);

    L = downsample_rows(L);
    H = downsample_rows(H);

    cv::Mat1d LL, LH, HL, HH;
    cv::filter2D(L, LL, CV_64F, row_lpf);
    cv::filter2D(L, LH, CV_64F, row_hpf);
    cv::filter2D(H, HL, CV_64F, row_lpf);
    cv::filter2D(H, HH, CV_64F, row_hpf);

    LL = downsample_cols(LL);
    LH = downsample_cols(LH);
    HL = downsample_cols(HL);
    HH = downsample_cols(HH);

    return {LL, LH, HL, HH};
}

cv::Mat1d recombine_4(const std::vector<cv::Mat1d>& subbands, const cv::Mat& lpf, const cv::Mat& hpf){
    /*
    Reconstructs 4 subbands into the original image. For ensuring
    near-perfect reconstruction, supply the function with the lpf and 
    hpf coefficients that were used during the decomposition process
    because for perfect reconstruction the hpf and lpf values for recombination
    are cross-coupled. The required recombination filter coefficients will be 
    calculated by this function.

    The input vector of matrices MUST follow the order of LL, LH, HL, HH
    because this order is assumed during the reconstruction. We use the relationships
    g_0(n) = (-1)^n h_1(n)
    g_1(n) = (-1)^{n+1}h_0(n)
    */
    
    if(subbands.size() != 4)
        throw std::invalid_argument("Size of subband vector should be exactly 4!");

    cv::Mat a = upsample_cols(subbands[0]);
    cv::Mat dv = upsample_cols(subbands[1]);
    cv::Mat dh = upsample_cols(subbands[2]);
    cv::Mat dd = upsample_cols(subbands[3]);

    // New filter calculation
    cv::Mat1d lpf_recomb = cv::Mat1d::zeros(hpf.rows, hpf.cols); 
    cv::Mat1d hpf_recomb = cv::Mat1d::zeros(lpf.rows, lpf.cols);
    for(int i = 0; i < lpf_recomb.cols; ++i) 
        lpf_recomb(0, i) = (i % 2)? (-hpf.at<double>(0, i)) : hpf.at<double>(0, i); 
    for(int i = 0; i < hpf_recomb.cols; ++i) 
        hpf_recomb(0, i) = (i % 2)? (lpf.at<double>(0, i)) : -lpf.at<double>(0, i); 

    cv::Mat filtered_a, filtered_dv, filtered_dh, filtered_dd;
    cv::filter2D(a, filtered_a, CV_64F, lpf_recomb);
    cv::filter2D(dv, filtered_dv, CV_64F, hpf_recomb);
    cv::filter2D(dh, filtered_dh, CV_64F, lpf_recomb);
    cv::filter2D(dd, filtered_dd, CV_64F, hpf_recomb);

    cv::Mat combined_cols_for_lpf = filtered_a + filtered_dv;
    cv::Mat combined_cols_for_hpf = filtered_dh + filtered_dd;

    cv::Mat upsampled_for_lpf = upsample_rows(combined_cols_for_lpf);
    cv::Mat upsampled_for_hpf = upsample_rows(combined_cols_for_hpf);

    cv::Mat1d lpf_out, hpf_out;
    cv::filter2D(upsampled_for_lpf, lpf_out, CV_64F, lpf_recomb.t());
    cv::filter2D(upsampled_for_hpf, hpf_out, CV_64F, hpf_recomb.t());

    return lpf_out + hpf_out;
}
