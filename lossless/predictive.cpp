#include "predictive.hpp"

using PredictorFn =
    uchar(*)(const std::array<uchar,4>&,
              const std::array<double,4>&);

uchar linear_comb(const std::array<uchar, 4>& neighbours, 
                   const std::array<double, 4>& coeff) {
    double dot_product = 0.0;
    for(int i = 0; i < 4; ++i){
        dot_product += neighbours[i] * coeff[i];
    }
    return std::round(dot_product);
}

constexpr PredictorFn FUNCTIONS[] = {
    linear_comb
};

void normalize_coefficients(std::array<double,4>& coeff,
                            Norm_Mode norm_mode) {
    double sum = std::accumulate(coeff.begin(), coeff.end(), 0.0);
    if(norm_mode == Norm_Mode::NONE &&  std::fabs(sum - 1) > FLT_EPSILON)
        throw std::invalid_argument(
            "With norm_mode set to NONE, the coeff array should sum to exactly 1"
        );
    if(std::fabs(sum) < FLT_EPSILON)
        throw std::invalid_argument(
            "Coefficient sum is zero"
        );


    if(norm_mode == Norm_Mode::LINEAR){
        for(double& c : coeff) c = c / sum;
    }
    else if(norm_mode == Norm_Mode::SOFTMAX) {
        // Stable calculation of softmax: since calculations of exp(x) are
        // very likely to blow up quick, we use the fact that 
        // softmax(c1 - x, c2 - x, ..., cn - x) = softmax(c1, c2, ..., cn) 
        // for any x. Here we choose x = max(c1, c2, ..., cn).

        double maxi = *std::max_element(coeff.begin(), coeff.end());
        for(double& c : coeff) c = c - maxi;
        std::transform(coeff.begin(), coeff.end(), coeff.begin(), 
                       [](double n) { return std::exp(n); });

        double sum = std::accumulate(coeff.begin(), coeff.end(), 0.0);
        for(double& c : coeff) c = c / sum;
    }
}

cv::Mat1b lossless_predict(const cv::Mat1b& A, Mode mode,
                  std::array<double, 4> coeff,
                  Norm_Mode norm_mode) {

    // Right now let us handle only grayscale images with one channel.
    if(mode == Mode::LINEAR && 
        std::all_of(coeff.begin(), coeff.end(), [](double i) { return std::fabs(i) < FLT_EPSILON; })) {
            throw std::invalid_argument(
                "If mode is LINEAR then coefficients must be provided"
            );
    }
    // The input coefficients need not sum to 1, we will normalize them using
    // either softmax or linear normalization. If norm_mode is set to NONE, it 
    // is the user's responsibility to ensure that the coefficients sum to 1.
    normalize_coefficients(coeff, norm_mode);

    cv::Mat1b predicted(A.rows, A.cols);
    std::array<uchar, 4> neighbours{0, 0, 0, 0};
    for(int r = 0; r < A.rows; ++r){
        for(int c = 0; c < A.cols; ++c){
            neighbours[0] = (c == 0 || r == 0)?          0 : A(r - 1, c - 1);
            neighbours[1] = (r == 0)?                    0 : A(r - 1, c);
            neighbours[2] = (c == A.cols - 1 || r == 0)? 0 : A(r - 1, c + 1);
            neighbours[3] = (c == 0)?                    0 : A(r, c - 1);
            predicted(r, c) = cv::saturate_cast<uchar>(
                                FUNCTIONS[static_cast<int>(mode)](neighbours, coeff)
                              );
        }
    }
    return predicted;
}

cv::Mat1s error_matrix(const cv::Mat1b& truth,
                       const cv::Mat1b& pred) {
    // This is 1s instead of 1b to account for subtractions
    cv::Mat1s err(truth.rows, truth.cols);

    for(int r = 0; r < truth.rows; ++r){
        for(int c = 0; c < truth.cols; ++c){
            err(r, c) =
                static_cast<short>(truth(r, c)) -
                static_cast<short>(pred(r, c));
        }
    }

    return err;
}

cv::Mat1b lossless_decode(const cv::Mat1s& e, Mode mode,
                  std::array<double, 4> coeff,
                  Norm_Mode norm_mode) {
    // Decoding is done by simply using A(i, j) = A_hat(i, j) + e(i, j);
    // replicate the prediction process.
    cv::Mat1b output(e.rows, e.cols);

    if(mode == Mode::LINEAR && 
        std::all_of(coeff.begin(), coeff.end(), [](double i) { return std::fabs(i) < FLT_EPSILON; })) {
            throw std::invalid_argument(
                "If mode is LINEAR then coefficients must be provided"
            );
    }
    normalize_coefficients(coeff, norm_mode);

    std::array<uchar, 4> neighbours{0, 0, 0, 0};
    for(int r = 0; r < e.rows; ++r){
        for(int c = 0; c < e.cols; ++c){
            neighbours[0] = (c == 0 || r == 0)?          0 : output(r - 1, c - 1);
            neighbours[1] = (r == 0)?                    0 : output(r - 1, c);
            neighbours[2] = (c == e.cols - 1 || r == 0)? 0 : output(r - 1, c + 1);
            neighbours[3] = (c == 0)?                    0 : output(r, c - 1);
            uchar predicted = FUNCTIONS[static_cast<int>(mode)](neighbours, coeff);
            int reconstructed = static_cast<int>(e(r, c)) + static_cast<int>(predicted);
            output(r, c) = cv::saturate_cast<uchar>(reconstructed);
        }
    }
    return output;
}


