#include "huffman.hpp"

std::unique_ptr<Node> huffman_init(const cv::Mat& A) {
    /* Huffman coding is a popular lossless Variable Length Coding scheme, 
    based on the following principles: 
    (a) Shorter code words are assigned to more probable symbols and longer 
    code words are assigned to less probable symbols. 
    (b) No code word of a symbol is a prefix of another code word. 
    This makes Huffman coding uniquely decodable. (prefix-free code) 
    (c) Every source symbol must have a unique code word assigned to it. 
    In terms of Shannon’s noiseless coding theorem, Huffman coding is optimal 
    for a fixed alphabet size. 

    This implementation is lossy if the matrix contains floats as its elements
    */
    std::map<int, int> freq; 

    const int channels = A.channels();
    const int depth = A.depth();  

    for (int r = 0; r < A.rows; ++r) {
        switch (depth) {
            case CV_8U:{
                const uchar* row_ptr = A.ptr<uchar>(r);
                for (int c = 0; c < A.cols * channels; ++c){
                    freq[row_ptr[c]]++;
                }
                break;
            }

            case CV_32F: {
                const float* fptr = A.ptr<float>(r);
                for (int c = 0; c < A.cols * channels; ++c){
                    int symbol = static_cast<int>(fptr[c] * 255.0f);
                    freq[symbol]++; 
                }
                break;
            }

            case CV_8S: {
                const char* cptr = A.ptr<char>(r);
                for (int c = 0; c < A.cols * channels; ++c){
                    freq[cptr[c]]++;
                }
                break;
            }

            default:
                throw std::runtime_error("Unsupported Mat depth");
        }
    }

    std::vector<std::unique_ptr<Node>> heap;
    for (const auto& [symbol, f] : freq) {
        heap.push_back(std::make_unique<Node>(symbol, f));
    }

    std::make_heap(heap.begin(), heap.end(), HUFFMAN_COMPARE{});

    while (heap.size() > 1) {
        std::pop_heap(heap.begin(), heap.end(), HUFFMAN_COMPARE{});
        auto u = std::move(heap.back());
        heap.pop_back();

        std::pop_heap(heap.begin(), heap.end(), HUFFMAN_COMPARE{});
        auto v = std::move(heap.back());
        heap.pop_back();

        auto parent = std::make_unique<Node>(-1, u->f + v->f);
        parent->left = std::move(u);
        parent->right = std::move(v);

        heap.push_back(std::move(parent));
        std::push_heap(heap.begin(), heap.end(), HUFFMAN_COMPARE{});
    }

    return std::move(heap.front());
}

std::unique_ptr<Node> huffman_init(const std::vector<cv::Mat>& images) {
    /* 
    An overload that allows you to create the Huffman encoding
    table from a collection of images rather than a single 
    image.
    */
    std::map<int, int> freq; 

    const int channels = images.front().channels();
    const int depth = images.front().depth();  

    for(cv::Mat img : images){
        for (int r = 0; r < img.rows; ++r) {
            switch (depth) {
                case CV_8U: {
                    const uchar* row_ptr = img.ptr<uchar>(r);
                    for (int c = 0; c < img.cols * channels; ++c){
                        freq[row_ptr[c]]++;
                    }
                    break;
                }

                case CV_32F: {
                    const float* fptr = img.ptr<float>(r);
                    for (int c = 0; c < img.cols * channels; ++c){
                        int symbol = static_cast<int>(fptr[c] * 255.0f);
                        freq[symbol]++; 
                    }
                    break;
                }

                case CV_8S: {
                    const char* cptr = img.ptr<char>(r);
                    for (int c = 0; c < img.cols * channels; ++c){
                        freq[cptr[c]]++;
                    }
                    break;
                }

                default:
                    throw std::runtime_error("Unsupported Mat depth");
            }
        }
    }   

    std::vector<std::unique_ptr<Node>> heap;
    for (const auto& [symbol, f] : freq) {
        heap.push_back(std::make_unique<Node>(symbol, f));
    }

    std::make_heap(heap.begin(), heap.end(), HUFFMAN_COMPARE{});

    while (heap.size() > 1) {
        std::pop_heap(heap.begin(), heap.end(), HUFFMAN_COMPARE{});
        auto u = std::move(heap.back());
        heap.pop_back();

        std::pop_heap(heap.begin(), heap.end(), HUFFMAN_COMPARE{});
        auto v = std::move(heap.back());
        heap.pop_back();

        auto parent = std::make_unique<Node>(-1, u->f + v->f);
        parent->left = std::move(u);
        parent->right = std::move(v);

        heap.push_back(std::move(parent));
        std::push_heap(heap.begin(), heap.end(), HUFFMAN_COMPARE{});
    }
    return std::move(heap.front());
}

void huffman_codemap(const Node* node,
                     std::unordered_map<int, std::string>& table,
                     std::string current){
    /*
    This is an alternate approach that you would take instead
    of traversing the entire tree everytime you encounter a 
    symbol to find it in the leaves, leading to a very large encoding
    time. Instead, we pre-build the code mapping and then perform an
    O(log n) lookup for each symbol.
    */

    if (!node) return;
    if (!node->left && !node->right) {
        table[node->symbol] = current.empty() ? "0" : current;
        return;
    }

    current.push_back('0');
    huffman_codemap(node->left.get(), table, current);
    current.pop_back();

    current.push_back('1');
    huffman_codemap(node->right.get(), table, current);
    current.pop_back();
}

void huffman_encode(const cv::Mat& A,
                    std::unordered_map<int, std::string>& table, 
                    const std::string& fileName){
    /*
    For a matrix p with R rows, C columns and K channels, the memory is:

    Row 0: [p(0,0,0), p(0,0,1), ..., p(0,0,K-1),  p(0,1,0), ..., p(0,C-1,K-1)]
    Row 1: [p(1,0,0), p(1,0,1), ...,                             p(1,C-1,K-1)]

    So each row x column element enumerates all its channels first and only then 
    moves on to the next column. This might be important in decoding. 
    */
    std::ofstream output(fileName, std::ios::binary);
    if(!output.is_open()){
        throw std::ios_base::failure("Could not open output file");
    }

    const int channels = A.channels();
    const int depth = A.depth();

    uint8_t buffer = 0;
    int bit_count = 0;

    auto write_bit = [&](int bit) {
        buffer = (buffer << 1) | bit;
        bit_count++;

        if (bit_count == 8) {
            output.put(buffer);
            buffer = 0;
            bit_count = 0;
        }
    };

    for (int r = 0; r < A.rows; ++r){
        switch (depth) {
            case CV_8U: {
                const uchar* row_ptr = A.ptr<uchar>(r);
                for (int c = 0; c < A.cols * channels; ++c){
                    for (char b : table[row_ptr[c]]) {
                        write_bit(b == '1');
                    }
                }
                break;
            }

            case CV_32F: {
                const float* fptr = A.ptr<float>(r);
                for (int c = 0; c < A.cols * channels; ++c){
                    int symbol = static_cast<int>(fptr[c] * 255.0f);
                    for (char b : table[symbol]) {
                        write_bit(b == '1');
                    }
                }
                break;
            }

            case CV_8S: {
                const char* cptr = A.ptr<char>(r);
                for (int c = 0; c < A.cols * channels; ++c){
                    for (char b : table[cptr[c]]) {
                        write_bit(b == '1');
                    }
                }
                break;
            }

            default:
                throw std::runtime_error("Unsupported Mat depth");
        }
    }
    if (bit_count > 0) {
        buffer <<= (8 - bit_count); 
        output.put(buffer);
    }
    output.close();
}

cv::Mat huffman_decode(int rows,
                    int cols,
                    int channels,
                    int depth,

                    const std::string& inFile, 
                    Node* encoding_tree){
    std::ifstream input(inFile, std::ios::binary);

    if(!input.is_open()){
        throw std::ios_base::failure("Could not open file");
    }

    char ch;
    Node* current = encoding_tree;
    std::vector<int> decoded_symbols;
    decoded_symbols.reserve(rows * cols * channels);

    char byte;
    while (input.get(byte) && decoded_symbols.size() < rows * cols * channels) {
        for (int i = 7; i >= 0; --i) {
            int bit = (byte >> i) & 1;

            current = (bit == 0) ? current->left.get() : current->right.get();

            if (!current->left && !current->right) {
                decoded_symbols.push_back(current->symbol);
                current = encoding_tree;

                if (decoded_symbols.size() == rows * cols * channels)
                    break;
            }
        }
    }

    if (decoded_symbols.size() != rows * cols * channels) {
        throw std::runtime_error("Decoded size mismatch");
    }

    input.close();

    int type = CV_MAKETYPE(depth, channels);
    cv::Mat output(rows, cols, type);

    size_t idx = 0;
    for (int r = 0; r < rows; ++r) {
        switch (depth) {
            case CV_8U: {
                uchar* row_ptr = output.ptr<uchar>(r);
                for (int c = 0; c < cols * channels; ++c) {
                    row_ptr[c] = static_cast<uchar>(decoded_symbols[idx++]);
                }
                break;
            }

            case CV_32F: {
                float* row_ptr = output.ptr<float>(r);
                for (int c = 0; c < cols * channels; ++c) {
                    // inverse of (element * 255.0f)
                    row_ptr[c] = decoded_symbols[idx++] / 255.0f;
                }
                break;
            }

            case CV_8S: {
                char* row_ptr = output.ptr<char>(r);
                for (int c = 0; c < cols * channels; ++c) {
                    row_ptr[c] = static_cast<char>(decoded_symbols[idx++]);
                }
                break;
            }

            default:
                throw std::runtime_error("Unsupported Mat depth");
        }
    }

    return output;
}