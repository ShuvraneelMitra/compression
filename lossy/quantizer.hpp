#pragma once

#include <opencv2/opencv.hpp>

#include <concepts>

template<typename T, typename U>
requires std::integral<T>
class UniformQuantizer {
    T max_val;
    T min_val;
    T levels;

    public:
        UniformQuantizer(T min_val, T max_val, T levels):
            max_val(max_val), min_val(min_val), levels(levels) {}
        
        T quantize(U x) {
            // introducing U helps support float types for x

            if(x <= min_val) return min_val;
            if(x >= max_val) return max_val;

            double h = (max_val - min_val + 1) / static_cast<double>(levels);
            T idx = static_cast<T>((x - min_val) / h);

            return static_cast<T>(min_val + idx * h +  h/2); 
        }

        cv::Mat_<T> quantize(const cv::Mat_<U>& A)
        {
            cv::Mat_<T> output(A.rows, A.cols);
            const U* in_ptr = A.template ptr<U>(0);
            T* out_ptr = output.template ptr<T>(0);

            size_t total = A.total();

            for(size_t i = 0; i < total; ++i) {
                out_ptr[i] = quantize(in_ptr[i]);
            }

            return output;
        }
};

// TODO: Implement Adaptive quantization