#pragma once

#include <opencv2/opencv.hpp>

#include <fstream>
#include <array>
#include <string>

#define MAX_LZ_DICT_SZ 512
#define NUM_SINGLE_SYMBOLS 256
#define CODE_BITS 9

struct TrieNode {
    std::array<std::unique_ptr<TrieNode>, NUM_SINGLE_SYMBOLS> children;
    int end_of_symbol = -1; 
    // whether the current trie node represents the end of a symbol in the encoding string;
    // if it does not, this field is -1, else the code of the symbol set that ends in this node.
};

class Trie {
    std::unique_ptr<TrieNode> root;
    int last_idx = NUM_SINGLE_SYMBOLS - 1;
    public:
        Trie() {
            root = std::make_unique<TrieNode>();
        }
        
        TrieNode* getRoot() {return root.get();}
        void insert(const cv::Mat& A, int start_idx, int end_idx);
        void insert(const int i);
        bool search(const cv::Mat& A, int start_idx, int end_idx);
        bool startsWith(const cv::Mat& A, int start_idx, int end_idx);
        int size() const { return last_idx + 1; }
};

void lz_encode(const cv::Mat& A, std::string fileName);
cv::Mat lz_decode(int rows,
                    int cols,
                    int channels,
                    int depth,
                    std::string inFile);