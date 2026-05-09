#include "lossy_pred.hpp"

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

double autocorrelation(const cv::Mat& img, int di, int dj) {
    CV_Assert(img.type() == CV_8UC1);

    double sum = 0.0;
    long long count = 0;

    for(int i = 0; i < img.rows; ++i) {
        for(int j = 0; j < img.cols; ++j) {

            int ni = i - di;
            int nj = j - dj;

            if(ni >= 0 && ni < img.rows && nj >= 0 && nj < img.cols) {
                double s1 = static_cast<double>(img.at<uchar>(i, j));
                double s2 = static_cast<double>(img.at<uchar>(ni, nj));
                sum += s1 * s2;
                ++count;
            }
        }
    }
    return (count > 0) ? sum / count : 0.0;
}

std::array<double, 4> computePredictorCoefficients(const cv::Mat& img) {
    CV_Assert(img.type() == CV_8UC1);

    auto R = [&](int i, int j) {
        return autocorrelation(img, i, j);
    };

    Matrix4x4 phi(
        R(0,0),  R(0,1),  R(0,2),  R(1,0),
        R(0,-1), R(0,0),  R(0,1),  R(1,-1),
        R(0,-2), R(0,-1), R(0,0),  R(1,-2),
        R(-1,0), R(-1,1), R(-1,2), R(0,0)
    );

    CoeffVector varphi(
        R(1,1),
        R(1,0),
        R(1,-1),
        R(0,1)
    );

    CoeffVector coeffs;

    cv::solve(
        phi,
        varphi,
        coeffs,
        cv::DECOMP_SVD
    );

    std::array<double, 4> result;
    for(int i = 0; i < 4; ++i) {
        result[i] = coeffs(i);
    }

    return result;
}

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

cv::Mat1s quantized_error(const cv::Mat1b& A, Mode mode,
                          std::array<double, 4> coeff,
                          Norm_Mode norm_mode,
                          Q_Mode quantization_mode,
                          int num_levels
                         ) {

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

    if(num_levels > 512) throw std::invalid_argument("Too many levels!");
    UniformQuantizer<short, short> uq(-256, 255, num_levels);

    cv::Mat1b predicted(A.rows, A.cols);
    cv::Mat1s recon(A.rows, A.cols);
    cv::Mat1s error_mat(A.rows, A.cols);
    std::array<uchar, 4> neighbours{0, 0, 0, 0};
    for(int r = 0; r < A.rows; ++r){
        for(int c = 0; c < A.cols; ++c){
            neighbours[0] = (c == 0 || r == 0)?          0 : recon(r - 1, c - 1);
            neighbours[1] = (r == 0)?                    0 : recon(r - 1, c);
            neighbours[2] = (c == A.cols - 1 || r == 0)? 0 : recon(r - 1, c + 1);
            neighbours[3] = (c == 0)?                    0 : recon(r, c - 1);
            predicted(r, c) = cv::saturate_cast<uchar>(
                                FUNCTIONS[static_cast<int>(mode)](neighbours, coeff)
                              );

            error_mat.at<short>(r, c) = A(r, c) - predicted(r, c);
            error_mat.at<short>(r, c) = uq.quantize(error_mat(r, c));
            recon.at<short>(r, c) = error_mat(r, c) + predicted(r, c);
        }
    }
    return error_mat;
}

cv::Mat1b lossy_decode(const cv::Mat1s& e, Mode mode,
                  std::array<double, 4> coeff,
                  Norm_Mode norm_mode) {
    // Decoding is done by simply using A(i, j) = A_hat(i, j) + e(i, j);
    // replicate the prediction and reconstruction/feedback process.

    // The input matrix e, is considered to be the quantized error matrix.
    cv::Mat1b output(e.rows, e.cols);

    if(mode == Mode::LINEAR && 
        std::all_of(coeff.begin(), coeff.end(), [](double i) { return std::fabs(i) < FLT_EPSILON; })) {
            throw std::invalid_argument(
                "If mode is LINEAR then coefficients must be provided"
            );
    }
    normalize_coefficients(coeff, norm_mode);

    cv::Mat1s recon(e.rows, e.cols);
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

            recon.at<short>(r, c) = e(r, c) + predicted;
        }
    }
    return recon;
}

cv::Mat1s delta_modulated_error(const cv::Mat1b& A, float alpha) {
    return quantized_error(
        A, Mode::LINEAR, {0.0, 0.0, 0.0, alpha}, Norm_Mode::NONE,
        Q_Mode::UNIFORM, 2
    );
}

cv::Mat1s DPCM_Error(const cv::Mat1b& A, int levels) {
    // The optimal values of the coefficients can be found by solving a matrix equation

    using CoeffVector = cv::Vec4d;
    using Matrix4x4   = cv::Matx44d;


    return quantized_error(
        A, Mode::LINEAR, computePredictorCoefficients(A), Norm_Mode::NONE,
        Q_Mode::UNIFORM, levels
    );
}