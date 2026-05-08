#pragma once

#include <opencv2/opencv.hpp>

#include <map>
#include <fstream>
#include <vector>

struct Node {
    int f;
    int symbol;
    std::unique_ptr<Node> left, right;

    Node(int symbol, int f): f(f), symbol(symbol), left(nullptr), right(nullptr){}
};

class HUFFMAN_COMPARE{
    public:
    bool operator()(const std::unique_ptr<Node>& A, 
                    const std::unique_ptr<Node>& B) {
        return A->f > B->f;
    }
};

std::unique_ptr<Node> huffman_init(const cv::Mat& A);
std::unique_ptr<Node> huffman_init(std::vector<const cv::Mat>& images);
void huffman_codemap(const Node* node,
                     std::unordered_map<int, std::string>& table,
                     std::string current="");
void huffman_encode(const cv::Mat& A,
                    std::unordered_map<int, std::string>& table, 
                    const std::string& fileName);
cv::Mat huffman_decode(int rows,
                    int cols,
                    int channels,
                    int depth,

                    const std::string& inFile, 
                    Node* encoding_tree);
