#include "metrics.hpp"

double metrics::MSE(cv::Mat& x, cv::Mat& y){
    if(x.cols != y.cols || x.rows != y.rows){
        std::stringstream ss;
        ss << "The shapes of the input matrices do not match; the shapes are ("
        << x.rows << ", " << x.cols << ") and ("
        << y.rows << ", " << y.cols << ")";
        throw std::invalid_argument(ss.str());
    }

    cv::Mat x_float, y_float;
    x.convertTo(x_float, CV_64F);
    y.convertTo(y_float, CV_64F);

    cv::Mat diff;
    cv::absdiff(x_float, y_float, diff);
    diff = diff.mul(diff); 

    cv::Scalar mean_val = cv::mean(diff);

    // Now sum over all the channels
    double mse = 0.0;
    for (int i = 0; i < diff.channels(); ++i) {
        mse += mean_val[i];
    }

    return mse;
} 

double metrics::SNR(cv::Mat& x, cv::Mat& y){
    /*
    If we model SNR to be between a "ground truth" image and 
    an "approximate reconstruction" image, then input y as the 
    ground truth.
    */
    if(x.cols != y.cols || x.rows != y.rows){
        std::stringstream ss;
        ss << "The shapes of the input matrices do not match; the shapes are ("
        << x.rows << ", " << x.cols << ") and ("
        << y.rows << ", " << y.cols << ")";
        throw std::invalid_argument(ss.str());
    }

    cv::Mat x_float, y_float;
    x.convertTo(x_float, CV_64F);
    y.convertTo(y_float, CV_64F);

    // Calculate the error for the denominator
    cv::Mat diff;
    cv::absdiff(x_float, y_float, diff);
    diff = diff.mul(diff);

    cv::Scalar sq_err = cv::sum(diff);

    // Now sum over all the channels
    double denominator = 0.0;
    for (int i = 0; i < diff.channels(); ++i) {
        denominator += sq_err[i];
    }

    // Calculate the numerator
    y_float = y_float.mul(y_float);
    cv::Scalar sum_squares = cv::sum(y_float);

    double numerator = 0.0;
    for (int i = 0; i < y_float.channels(); ++i) {
        numerator += sum_squares[i];
    }

    return 20 * log10(numerator / denominator);
}

double metrics::PSNR(cv::Mat& x, cv::Mat& y){
    /*
    If we model SNR to be between a "ground truth" image and 
    an "approximate reconstruction" image, then input y as the 
    ground truth.
    */
    if(x.cols != y.cols || x.rows != y.rows){
        std::stringstream ss;
        ss << "The shapes of the input matrices do not match; the shapes are ("
        << x.rows << ", " << x.cols << ") and ("
        << y.rows << ", " << y.cols << ")";
        throw std::invalid_argument(ss.str());
    }

    cv::Mat x_float, y_float;
    x.convertTo(x_float, CV_64F);
    y.convertTo(y_float, CV_64F);

    // Calculate the error for the denominator
    cv::Mat diff;
    cv::absdiff(x_float, y_float, diff);
    diff = diff.mul(diff);

    cv::Scalar sq_err = cv::sum(diff);

    // Now sum over all the channels
    double denominator = 0.0;
    for (int i = 0; i < diff.channels(); ++i) {
        denominator += sq_err[i];
    }

    // I separated out these two values so that 255 * 255 * rows * cols does not get too large.
    return 40 * log10(255) + 20 * log10((double)(x_float.rows * x_float.cols) / denominator);
}
