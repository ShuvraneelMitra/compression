#pragma once

#include <opencv2/opencv.hpp>


#include "subband.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>

using uint = unsigned int;

enum class EZWSymbol : uint8_t {
    POS = 0, // positive significant coefficient
    NEG = 1, // negative significant coefficient
    IZ  = 2, // isolated zero
    ZTR = 3, // zerotree root
    REF0 = 4,
    REF1 = 5
};

struct EZWToken {
    EZWSymbol symbol;
};

struct EZWTree {
    int rows{};
    int cols{};
    uint levels{};

    std::vector<cv::Point> roots;
    std::unordered_map<int, std::vector<cv::Point>> children;

    int id(const cv::Point& p) const {
        return p.y * cols + p.x;
    }

    const std::vector<cv::Point>& getChildren(const cv::Point& p) const;
};

struct EZWState {
    cv::Mat1b significant;
    cv::Mat1d reconAbs;
    std::vector<cv::Point> lsp;
    std::vector<EZWToken> stream;
};

bool isPowerOfTwo(int n);

int initialThresholdFromCoeffs(const cv::Mat& C);

EZWTree buildEZWTree(int rows, int cols, uint levels);

void dominantPass(const cv::Mat1d& C,
                  int T,
                  const EZWTree& tree,
                  EZWState& state);

void dominantScan(const cv::Point& x,
                  const cv::Mat1d& C,
                  int T,
                  const EZWTree& tree,
                  EZWState& state,
                  std::vector<cv::Point>& newlySignificant);

bool hasSignificantDescendant(const cv::Point& x,
                              const cv::Mat1d& C,
                              int T,
                              const EZWTree& tree,
                              const EZWState& state);

void subordinatePass(const cv::Mat1d& C,
                     int T,
                     EZWState& state,
                     const std::vector<cv::Point>& oldLSP);

void writeEZWStream(const std::string& filename,
                    const std::vector<EZWToken>& stream,
                    int rows,
                    int cols,
                    uint levels,
                    int initialT);

void ezw_encode(const cv::Mat& A,
                const cv::Mat& lpf,
                const cv::Mat& hpf,
                uint levels,
                int init_thresh,
                const std::string& filename);

struct EZWFile {
    int rows{};
    int cols{};
    uint levels{};
    int initialT{};
    std::vector<EZWToken> stream;
};

EZWFile readEZWStream(const std::string& filename);

cv::Mat1d ezw_decode_coeffs(const std::string& filename);

void inverseDominantPass(const EZWTree& tree,
                         int T,
                         EZWState& state,
                         const std::vector<EZWToken>& stream,
                         size_t& cursor);

void inverseDominantScan(const cv::Point& x,
                         const EZWTree& tree,
                         int T,
                         EZWState& state,
                         const std::vector<EZWToken>& stream,
                         size_t& cursor);

void inverseSubordinatePass(int T,
                            EZWState& state,
                            const std::vector<cv::Point>& oldLSP,
                            const std::vector<EZWToken>& stream,
                            size_t& cursor);
