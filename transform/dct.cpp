#include "dct.hpp"

cv::Mat1s dct_opencv(const cv::Mat1b& A){
    cv::Mat1f input_float;
    A.convertTo(input_float, CV_32F);

    cv::Mat1f dct_coeffs;
    cv::dct(input_float, dct_coeffs);

    cv::Mat1s dct_short;
    dct_coeffs.convertTo(dct_short, CV_16S);

    return dct_short;
}

double compute_coeff_naive(const cv::Mat1b& A, int u, int v) {
    // takes an N x N block and computes its (u, v)-th DCT coefficient
    if(A.rows != A.cols) throw std::invalid_argument("Input image should be square and grayscale!");

    int n = A.rows;
    double out = 0;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            double c1 = cos(static_cast<double>(std::numbers::pi * (2 * j + 1) * v) / static_cast<double>(2 * n));
            double c2 = cos(static_cast<double>(std::numbers::pi * (2 * i + 1) * u) / static_cast<double>(2 * n));
            out += A(i, j) * c1 * c2;
        }
    }

    if(u == 0 && v == 0) out *= (1 /(double) n);
    else if(u == 0 || v == 0) out *= (sqrt(2) /(double) n);
    else out *= (2 /(double) n);
    return out;
}

cv::Mat1s dct_slow(const cv::Mat1b& A, uint block_size) {
    // At this point we are just focusing on grayscales
    if(A.rows != A.cols) throw std::invalid_argument("Input image should be square and grayscale!");
    if(A.rows % block_size != 0) throw std::invalid_argument("Block size should be a factor of the image size");

    cv::Mat1d result = cv::Mat1d::zeros(A.rows, A.cols);

    int n = A.rows;
    int d = n / block_size;
    int num_blocks = (n / block_size) * (n / block_size);

    for(int i = 0; i < d; i++){
        for(int j = 0; j < d; j++){

            cv::Rect block(j * block_size, i * block_size,
                            block_size, block_size);
            cv::Mat result_blk = result(block);
            cv::Mat A_blk = A(block);

            for(int u = 0; u < block_size; ++u){
                for(int v = 0; v < block_size; ++v){
                    result_blk.at<double>(u, v) = std::round(compute_coeff_naive(A_blk, u, v));
                    // first place of introduction of loss
                }
            }
        }
    }

    cv::Mat1s result_short;
    result.convertTo(result_short, CV_MAKETYPE(CV_16S, 1));
    return result;
}

//------------- Fast DCT implementation and its helper functions --------------------

cv::Mat1b rearrange(cv::Mat1b& A){
    cv::Mat1b y = cv::Mat1b::zeros(A.rows, A.cols);

    int n = A.rows;
    for(int r = 0; r < n; r++){
        int row_src = (r < n/2) ? 2*r : 2*n - 2*r - 1;
        for(int c = 0; c < n; c++) {
            int col_src = (c < n/2) ? 2*c : 2*n - 2*c - 1;
            y.at<uchar>(r, c) = A.at<uchar>(row_src, col_src);
        }
    }

    return y;
}

std::vector<Complex> fft1d(const std::vector<Complex>& x) {
    int N = x.size();

    if(N == 1)
        return x;

    if(N % 2 != 0)
        throw std::invalid_argument(
            "FFT requires power-of-2 size"
        );

    std::vector<Complex> even(N / 2);
    std::vector<Complex> odd(N / 2);

    for(int i = 0; i < N / 2; ++i) {
        even[i] = x[2 * i];
        odd[i]  = x[2 * i + 1];
    }

    auto Fe = fft1d(even);
    auto Fo = fft1d(odd);

    std::vector<Complex> X(N);

    for(int k = 0; k < N / 2; ++k) {

        Complex twiddle =
            std::polar(
                1.0f,
                -2.0f * static_cast<float>(CV_PI) * k / N
            ) * Fo[k];

        X[k]       = Fe[k] + twiddle;
        X[k+N/2]   = Fe[k] - twiddle;
    }

    return X;
}

cv::Mat_<cv::Vec2f> fft2d(const cv::Mat1f& input) {

    if(input.rows != input.cols)
        throw std::invalid_argument(
            "Input must be square"
        );

    int N = input.rows;

    if((N & (N - 1)) != 0)
        throw std::invalid_argument(
            "FFT requires power-of-2 dimensions"
        );

    std::vector<std::vector<Complex>>
        temp(N, std::vector<Complex>(N));

    for(int r = 0; r < N; ++r) {

        std::vector<Complex> row(N);

        for(int c = 0; c < N; ++c)
            row[c] = Complex(input(r, c), 0.0f);

        temp[r] = fft1d(row);
    }

    cv::Mat_<cv::Vec2f> output(N, N);

    for(int c = 0; c < N; ++c) {

        std::vector<Complex> col(N);

        for(int r = 0; r < N; ++r)
            col[r] = temp[r][c];

        auto col_fft = fft1d(col);

        for(int r = 0; r < N; ++r) {

            output(r, c)[0] = col_fft[r].real();
            output(r, c)[1] = col_fft[r].imag();
        }
    }

    return output;
}

std::vector<Complex> ifft1d(const std::vector<Complex>& X) {

    int N = X.size();

    std::vector<Complex> conjX(N);

    for(int i = 0; i < N; ++i)
        conjX[i] = std::conj(X[i]);

    auto temp = fft1d(conjX);

    std::vector<Complex> x(N);

    for(int i = 0; i < N; ++i)
        x[i] = std::conj(temp[i]) / (float)N;

    return x;
}

cv::Mat_<cv::Vec2f> ifft2d(const cv::Mat_<cv::Vec2f>& input) {

    int N = input.rows;

    std::vector<std::vector<Complex>>
        temp(N, std::vector<Complex>(N));

    for(int r = 0; r < N; ++r) {
        std::vector<Complex> row(N);
        for(int c = 0; c < N; ++c) {
            row[c] = Complex(input(r,c)[0], input(r,c)[1]);
        }

        temp[r] = ifft1d(row);
    }

    cv::Mat_<cv::Vec2f> output(N, N);

    for(int c = 0; c < N; ++c) {
        std::vector<Complex> col(N);

        for(int r = 0; r < N; ++r) col[r] = temp[r][c];
        auto col_ifft = ifft1d(col);

        for(int r = 0; r < N; ++r) {
            output(r,c)[0] = col_ifft[r].real();
            output(r,c)[1] = col_ifft[r].imag();
        }
    }

    return output;
}

cv::Mat1f inverse_rearrange(const cv::Mat1f& y) {
    int N = y.rows;

    cv::Mat1f x(N, N);
    for(int r = 0; r < N; ++r) {
        int row_dst = (r < N/2)? 2*r : 2*N - 2*r - 1;
        for(int c = 0; c < N; ++c) {
            int col_dst = (c < N/2)? 2*c : 2*N - 2*c - 1;
            x(row_dst, col_dst) = y(r,c);
        }
    }

    return x;
}

cv::Mat1f dct_block_fft(const cv::Mat1f& block) {

    int N = block.rows;
    cv::Mat1f temp(N, N);

    for(int r = 0; r < N; ++r) {
        std::vector<Complex> x(2 * N);

        for(int n = 0; n < N; ++n) {
            x[n] = Complex(block(r, n), 0.0f);
            x[2 * N - n - 1] = Complex(block(r, n), 0.0f);
        }

        auto X = fft1d(x);
        for(int k = 0; k < N; ++k) {
            float theta = -CV_PI * k / (2.0f * N);

            Complex phase = std::polar(1.0f, theta);
            float val = 0.5f * (X[k] * phase).real();

            if(k == 0) val *= std::sqrt(1.0f / N);
            else val *= std::sqrt(2.0f / N);

            temp(r, k) = val;
        }
    }

    cv::Mat1f out(N, N);

    for(int c = 0; c < N; ++c) {
        std::vector<Complex> x(2 * N);

        for(int n = 0; n < N; ++n) {
            x[n] = Complex(temp(n, c), 0.0f);
            x[2 * N - n - 1] = Complex(temp(n, c), 0.0f);
        }

        auto X = fft1d(x);
        for(int k = 0; k < N; ++k) {
            float theta = -CV_PI * k / (2.0f * N);

            Complex phase = std::polar(1.0f, theta);
            float val = 0.5f * (X[k] * phase).real();

            if(k == 0) val *= std::sqrt(1.0f / N);
            else val *= std::sqrt(2.0f / N);

            out(k, c) = val;
        }
    }

    return out;
}

cv::Mat1f idct_block_fft(const cv::Mat1f& coeffs) {
    int N = coeffs.rows;

    cv::Mat1f temp(N, N);

    for(int c = 0; c < N; ++c) {
        std::vector<Complex> X(2 * N);

        for(int k = 0; k < N; ++k) {
            float alpha = (k == 0) ? std::sqrt(1.0f / N) : std::sqrt(2.0f / N);
            float theta = CV_PI * k / (2.0f * N);

            Complex phase = std::polar(1.0f, theta);

            X[k] = 2.0f * coeffs(k, c) * phase / alpha;
        }

        for(int k = N + 1; k < 2 * N; ++k) X[k] = std::conj(X[2 * N - k]);

        auto x = ifft1d(X);
        for(int n = 0; n < N; ++n) temp(n, c) = x[n].real();
    }

    cv::Mat1f out(N, N);

    for(int r = 0; r < N; ++r) {
        std::vector<Complex> X(2 * N);

        for(int k = 0; k < N; ++k) {
            float alpha = (k == 0) ? std::sqrt(1.0f / N) : std::sqrt(2.0f / N);
            float theta = CV_PI * k / (2.0f * N);
 
            Complex phase = std::polar(1.0f, theta);
            X[k] = 2.0f * temp(r, k) * phase / alpha;
        }

        for(int k = N + 1; k < 2 * N; ++k) X[k] = std::conj(X[2 * N - k]);
        auto x = ifft1d(X);
        for(int n = 0; n < N; ++n) out(r, n) = x[n].real();
    }

    return out;
}

cv::Mat1s dct_vetterli(const cv::Mat1b& A, uint block_size) {

    int N = A.rows;
    cv::Mat1s result = cv::Mat1s::zeros(N, N);
    int blocks = N / block_size;

    for(int bi = 0; bi < blocks; ++bi) {
        for(int bj = 0; bj < blocks; ++bj) {
            cv::Rect roi(
                bj * block_size,
                bi * block_size,
                block_size,
                block_size
            );

            cv::Mat1f blk;
            A(roi).convertTo(blk, CV_32F);

            auto dct = dct_block_fft(blk);

            cv::Mat1s q;
            dct.convertTo(q, CV_16S);

            q.copyTo(result(roi));
        }
    }

    return result;
}

cv::Mat1s idct_fast(const cv::Mat1s& coeffs, uint block_size) {

    int N = coeffs.rows;

    cv::Mat1f reconstructed = cv::Mat1f::zeros(N, N);
    int blocks = N / block_size;

    for(int bi = 0; bi < blocks; ++bi) {
        for(int bj = 0; bj < blocks; ++bj) {

            cv::Rect roi(
                bj * block_size,
                bi * block_size,
                block_size,
                block_size
            );

            cv::Mat1f blk;
            coeffs(roi).convertTo(blk, CV_32F);

            auto spatial = idct_block_fft(blk);
            spatial.copyTo(reconstructed(roi));
        }
    }

    cv::Mat1s out;
    reconstructed.convertTo(out, CV_16S);

    return out;
}