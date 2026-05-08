#include "run_length.hpp"

inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

cv::Mat1b getBitPlane(const cv::Mat1b& A, int i) {
    cv::Mat1b result(A.rows, A.cols);

    for (int r = 0; r < A.rows; r++) {
        for (int c = 0; c < A.cols; c++) {
            result(r, c) = (A(r, c) >> i) & 1;
        }
    }

    return result;
}

void run_length_encode(std::string s, const std::string& fileName){
    /*
    Deals with strings containing only 1 or 0 and errors out if
    any other character is detected.

    We adopt the encoding convention:
    - The stream is assumed to start with 1.
    - If the actual stream starts with 0, prepend "0,".
    */

    std::ofstream output(fileName);
    if (!output.is_open()) {
        throw std::ios_base::failure("Could not open output file");
    }
    if(s.empty()) return;
    
    trim(s);
    if(s[0] == '0') output << "0,";

    size_t idx = 0;
    while(idx < s.length()) {
            if(s[idx] != '0' && s[idx] != '1') throw std::invalid_argument("String should only contain ones and zeros");

            char ch = s[idx];
            int last = idx;
            while(idx < s.length() && s[idx] == ch) idx++;
            output << (idx - last);
            if(idx < s.length()) output << ',';
    }
}
