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
};

// TODO: Implement Adaptive quantization