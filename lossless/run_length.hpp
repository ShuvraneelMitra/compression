#pragma once

#include <opencv2/opencv.hpp>

#include <fstream>
#include <string>
#include <algorithm> 
#include <cctype>
#include <type_traits>

/*
We aim to deal with only grayscale images with pixel values
between 0 to 255 here, that is, having at most 8 bits.
*/
#define MAX_BITS 8

template<typename T>
constexpr bool is_character_v =
       std::is_same_v<std::remove_cvref_t<T>, char>
    || std::is_same_v<std::remove_cvref_t<T>, wchar_t>
    || std::is_same_v<std::remove_cvref_t<T>, char8_t>
    || std::is_same_v<std::remove_cvref_t<T>, char16_t>
    || std::is_same_v<std::remove_cvref_t<T>, char32_t>;

template<typename T>
constexpr bool is_integer_v =
    std::is_integral_v<std::remove_cvref_t<T>>;

/*
Consider the problem of converting a grayscale image into its
bit-plane representation. We can simply go to each pixel (r, c) and
extract (A[r][c] >> i) & 1 to get the i-th bitplane image. 
*/
cv::Mat1b getBitPlane(const cv::Mat1b& A, int i);

void run_length_encode(std::string s, const std::string& fileName);

template<typename Iterator>
void run_length_encode(Iterator begin, Iterator end, const std::string& fileName){
    using T = std::iter_value_t<Iterator>;

    std::ofstream output(fileName);

    if (!output.is_open()) {
        throw std::ios_base::failure(
            "Could not open output file"
        );
    }

    while(begin != end){
        if constexpr(is_character_v<T>) {
            if(*begin != '0' && *begin != '1') {
                throw std::invalid_argument(
                    "String should only contain ones and zeros"
                );
            }
        }
        else if constexpr(is_integer_v<T>) {
            if(*begin != 0 && *begin != 1) {
                throw std::invalid_argument(
                    "Container should only contain 0 and 1"
                );
            }
        }

        auto ch = *begin;
        std::size_t count = 0;
        while(begin != end && *begin == ch){
            ++count;
            ++begin;
        }
        output << count;
        if(begin != end) output << ',';
    }
}

template<typename Iterator>
std::vector<int> run_length_encode(Iterator begin, Iterator end){
    using T = std::iter_value_t<Iterator>;

    if (!output.is_open()) {
        throw std::ios_base::failure(
            "Could not open output file"
        );
    }
    std::vector<int> result;

    while(begin != end){
        if constexpr(is_character_v<T>) {
            if(*begin != '0' && *begin != '1') {
                throw std::invalid_argument(
                    "String should only contain ones and zeros"
                );
            }
        }
        else if constexpr(is_integer_v<T>) {
            if(*begin != 0 && *begin != 1) {
                throw std::invalid_argument(
                    "Container should only contain 0 and 1"
                );
            }
        }

        auto ch = *begin;
        std::size_t count = 0;
        while(begin != end && *begin == ch){
            ++count;
            ++begin;
        }
        result.push_back(count);
    }
    return result;
}