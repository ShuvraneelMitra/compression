#include "lempel_ziv.hpp"

void Trie::insert(const cv::Mat& A, int start_idx, int end_idx) {
    // Treat the matrix as a 1D array while traversing it
    TrieNode* current = root.get();

    for (int i = start_idx; i <= end_idx; i++) {
        int idx = A.data[i];
        if (!current->children[idx]) {
            current->children[idx] = std::make_unique<TrieNode>();
        }
        current = current->children[idx].get();
    }

    current->end_of_symbol = ++last_idx;
}

void Trie::insert(const int i){
    root->children[i] = std::make_unique<TrieNode>();
    root->children[i]->end_of_symbol = i;
}

bool Trie::search(const cv::Mat& A, int start_idx, int end_idx) {
    TrieNode* current = root.get();
    for (int v = start_idx; v <= end_idx; v++) {
        int idx = A.data[v];
        if (!current->children[idx]) return false;
        current = current->children[idx].get();
    }

    return (current->end_of_symbol > -1);
}

bool Trie::startsWith(const cv::Mat& A, int start_idx, int end_idx) {
    TrieNode* current = root.get();
    int ans = 0;
    for (int v = start_idx; v <= end_idx; v++) {
        int idx = A.data[v];
        if (!current->children[idx]) return false;
        current = current->children[idx].get();
    }

    return true;
}

void lz_encode(const cv::Mat& A, const std::string& fileName) {
    /*
    Lempel-Ziv encoding is a form of lossless encoding which
    1. assigns a fixed length codeword to a variable length of symbols.
    2. Unlike Huffman coding and arithmetic coding, does not require a priori 
    knowledge of the probabilities of the source symbols. 

    The dictionary that we will maintain is implemented in the form of a trie.
    */

    std::ofstream output(fileName, std::ios::binary);
    if (!output.is_open()) {
        throw std::ios_base::failure("Could not open output file");
    }

    if (!A.isContinuous() || A.depth() != CV_8U) {
        throw std::runtime_error("Only continuous CV_8U Mat supported");
    }

    const int total = A.total() * A.channels();
    uchar* data = A.data;

    uint8_t buffer = 0;
    int bit_count = 0;

    auto write_bit = [&](int bit) {
        buffer = (buffer << 1) | bit;
        if (++bit_count == 8) {
            output.put(buffer);
            buffer = 0;
            bit_count = 0;
        }
    };

    auto write_code = [&](int code) {
        for (int i = CODE_BITS - 1; i >= 0; --i)
            write_bit((code >> i) & 1);
    };

    Trie trie;

    for (int i = 0; i < 256; i++) {
        trie.insert(i);
    }

    int idx = 0;

    while (idx < total) {
        TrieNode* current = trie.getRoot();
        int last_code = -1;
        int start = idx;

        while (idx < total && current->children[data[idx]]) {
            current = current->children[data[idx]].get();
            last_code = current->end_of_symbol;
            idx++;
        }

        write_code(last_code);
        if (idx < total) {
            if (trie.size() < (1 << CODE_BITS)) {
                trie.insert(A, start, idx);
            }
        }
    }

    if (bit_count > 0) {
        buffer <<= (8 - bit_count);
        output.put(buffer);
    }

    output.close();
}

cv::Mat lz_decode(int rows,
                  int cols,
                  int channels,
                  int depth,
                  const std::string& inFile) {
    std::ifstream input(inFile, std::ios::binary);
    if (!input.is_open()) {
        throw std::ios_base::failure("Could not open file");
    }

    uint8_t buffer = 0;
    int bits_left = 0;

    auto read_bit = [&]() -> int {
        if (bits_left == 0) {
            int c = input.get();
            if (c == std::char_traits<char>::eof()) return -1;
            buffer    = static_cast<uint8_t>(c);
            bits_left = 8;
        }
        return (buffer >> --bits_left) & 1;
    };

    auto read_code = [&]() -> int {
        int code = 0;
        for (int i = 0; i < CODE_BITS; ++i) {
            int bit = read_bit();
            if (bit < 0) return -1;     
            code = (code << 1) | bit;
        }
        return code;
    };


    std::vector<std::vector<uchar>> dict;
    dict.reserve(1 << CODE_BITS);
    for (int i = 0; i < 256; ++i)
        dict.push_back({static_cast<uchar>(i)});

    const size_t total = static_cast<size_t>(rows) * cols * channels;
    std::vector<uchar> output_data;
    output_data.reserve(total);

    int code = read_code();
    if (code < 0 || code >= static_cast<int>(dict.size()))
        return cv::Mat::zeros(rows, cols, CV_MAKETYPE(depth, channels));

    std::vector<uchar> prev = dict[code];
    output_data.insert(output_data.end(), prev.begin(), prev.end());

    while (output_data.size() < total) {
        code = read_code();
        if (code < 0) break;

        std::vector<uchar> entry;

        if (code < static_cast<int>(dict.size())) {
            // Normal case: code already in dictionary
            entry = dict[code];

        } else if (code == static_cast<int>(dict.size())) {
            entry = prev;
            entry.push_back(prev[0]);

        } else {
            throw std::runtime_error("Corrupt LZW stream: unexpected code " +
                                     std::to_string(code));
        }

        for (uchar b : entry) {
            if (output_data.size() >= total) break;
            output_data.push_back(b);
        }

        if (static_cast<int>(dict.size()) < (1 << CODE_BITS)) {
            std::vector<uchar> new_entry = prev;
            new_entry.push_back(entry[0]);
            dict.push_back(std::move(new_entry));
        }

        prev = std::move(entry);
    }

    cv::Mat result(rows, cols, CV_MAKETYPE(depth, channels));
    std::memcpy(result.data,
                output_data.data(),
                std::min(output_data.size(), total));

    return result;
}