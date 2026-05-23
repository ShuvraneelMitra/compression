#include "ezw.hpp"

/*
The EZW algorithm claims to be efficient and flexible in low-bitrate
channel transmission, as even if the transmission is interrupted in the middle,
we still get a decent-enough reconstruction due to the encoding of the more
significant coefficients before the less significant ones. We will have to test 
that in test_ezw.cpp

Using an embedded coding algorithm, an encoder can terminate the encoding at 
any point thereby allowing a target rate or target distortion metric to be
met exactly. Also, given a bit stream, the decoder can cease decoding at any 
point in the bit stream and still produce exactly the same image that would have 
been encoded at the bit rate corresponding to the truncated bit stream. In 
addition to producing a fully embedded bit stream, EZW consistently produces 
compression results that are competitive with virtually all known compression 
algorithms on standard test images.

THIS WEBSITE HAS BEEN A LIFESAVER!! http://www.polyvalens.com/wavelets/ezw/

Coding an image using the EZW scheme, together with some optimizations results
in a remarkably effective image compressor with the property that the compressed
data stream can have any bit rate desired. 
*/

namespace {

    constexpr int minimum_threshold = 1;

    void appendSymbol(std::vector<EZWToken>& stream, EZWSymbol s) {
        stream.push_back(EZWToken{s});
    }

    cv::Mat1d toDoubleMat(const cv::Mat& src) {
        cv::Mat dst;
        src.convertTo(dst, CV_64F);
        return dst;
    }

    void addChild(EZWTree& tree, const cv::Point& parent, const cv::Point& child) {
        tree.children[tree.id(parent)].push_back(child);
    }

    bool inside(int x, int y, int cols, int rows) {
        return x >= 0 && x < cols && y >= 0 && y < rows;
    }

} 

bool isPowerOfTwo(int n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

const std::vector<cv::Point>& EZWTree::getChildren(const cv::Point& p) const {
    static const std::vector<cv::Point> empty;
    auto it = children.find(id(p));
    if (it == children.end()) {
        return empty;
    }
    return it->second;
}

int initialThresholdFromCoeffs(const cv::Mat& C) {
    cv::Mat absC = cv::abs(C);

    double maxVal = 0.0;
    cv::minMaxLoc(absC, nullptr, &maxVal);

    if (maxVal <= 0.0) {
        return 0;
    }

    return 1 << static_cast<int>(std::floor(std::log2(maxVal)));
}

EZWTree buildEZWTree(int rows, int cols, uint levels) {
    if (rows != cols) {
        throw std::invalid_argument("EZW implementation currently expects square coefficient matrices.");
    }

    if (!isPowerOfTwo(rows)) {
        throw std::invalid_argument("Input matrix size must be a power of two.");
    }

    if (levels == 0) {
        throw std::invalid_argument("Number of wavelet decomposition levels must be positive.");
    }

    const int N = rows;

    if ((N >> levels) <= 0) {
        throw std::invalid_argument("Too many decomposition levels for the given image size.");
    }

    EZWTree tree;
    tree.rows = rows;
    tree.cols = cols;
    tree.levels = levels;

    const int coarseLLSize = N >> levels;

    for (int y = 0; y < coarseLLSize; ++y) {
        for (int x = 0; x < coarseLLSize; ++x) {
            tree.roots.emplace_back(x, y);
        }
    }

    for (const cv::Point& r : tree.roots) {
        const int s = coarseLLSize;

        addChild(tree, r, cv::Point(r.x + s, r.y));     // HL
        addChild(tree, r, cv::Point(r.x,     r.y + s)); // LH
        addChild(tree, r, cv::Point(r.x + s, r.y + s)); // HH
    }

    for (int k = static_cast<int>(levels); k >= 2; --k) {
        const int parentBandSize = N >> k;
        const int parentBlockSize = N >> (k - 1);

        const int childBandSize = N >> (k - 1);
        const int childBlockSize = N >> (k - 2);

        struct SubbandOffset {
            int px;
            int py;
            int cx;
            int cy;
        };

        const std::vector<SubbandOffset> subbands = {
            {parentBandSize, 0,              childBandSize, 0             }, // HL
            {0,              parentBandSize, 0,             childBandSize}, // LH
            {parentBandSize, parentBandSize, childBandSize, childBandSize}  // HH
        };

        for (const auto& sb : subbands) {
            for (int ly = 0; ly < parentBandSize; ++ly) {
                for (int lx = 0; lx < parentBandSize; ++lx) {
                    cv::Point parent(sb.px + lx, sb.py + ly);

                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            int childLocalX = 2 * lx + dx;
                            int childLocalY = 2 * ly + dy;

                            cv::Point child(sb.cx + childLocalX,
                                            sb.cy + childLocalY);

                            if (inside(parent.x, parent.y, cols, rows) &&
                                inside(child.x, child.y, cols, rows) &&
                                child.x < childBlockSize &&
                                child.y < childBlockSize) {
                                addChild(tree, parent, child);
                            }
                        }
                    }
                }
            }
        }

        (void)parentBlockSize;
    }

    return tree;
}

bool hasSignificantDescendant(const cv::Point& x,
                              const cv::Mat1d& C,
                              int T,
                              const EZWTree& tree,
                              const EZWState& state) {
    for (const cv::Point& y : tree.getChildren(x)) {
        if (!state.significant(y) && std::abs(C(y)) >= T) {
            return true;
        }

        if (hasSignificantDescendant(y, C, T, tree, state)) {
            return true;
        }
    }

    return false;
}

void dominantScan(const cv::Point& x,
                  const cv::Mat1d& C,
                  int T,
                  const EZWTree& tree,
                  EZWState& state,
                  std::vector<cv::Point>& newlySignificant) {
    if (state.significant(x)) {
        for (const cv::Point& y : tree.getChildren(x)) {
            dominantScan(y, C, T, tree, state, newlySignificant);
        }
        return;
    }

    const double coeff = C(x);
    const double mag = std::abs(coeff);

    if (mag >= T) {
        if (coeff >= 0.0) {
            appendSymbol(state.stream, EZWSymbol::POS);
        } else {
            appendSymbol(state.stream, EZWSymbol::NEG);
        }

        state.significant(x) = 1;
        state.reconAbs(x) = static_cast<double>(T);

        state.lsp.push_back(x);
        newlySignificant.push_back(x);

        for (const cv::Point& y : tree.getChildren(x)) {
            dominantScan(y, C, T, tree, state, newlySignificant);
        }

        return;
    }

    if (hasSignificantDescendant(x, C, T, tree, state)) {
        appendSymbol(state.stream, EZWSymbol::IZ);

        for (const cv::Point& y : tree.getChildren(x)) {
            dominantScan(y, C, T, tree, state, newlySignificant);
        }
    } else {
        appendSymbol(state.stream, EZWSymbol::ZTR);
    }
}

void dominantPass(const cv::Mat1d& C,
                  int T,
                  const EZWTree& tree,
                  EZWState& state) {
    std::vector<cv::Point> newlySignificant;

    for (const cv::Point& r : tree.roots) {
        dominantScan(r, C, T, tree, state, newlySignificant);
    }
}

void subordinatePass(const cv::Mat1d& C,
                     int T,
                     EZWState& state,
                     const std::vector<cv::Point>& oldLSP) {
    const double Tnext = static_cast<double>(T) / 2.0;

    for (const cv::Point& x : oldLSP) {
        const double mag = std::abs(C(x));

        if (mag >= state.reconAbs(x) + Tnext) {
            appendSymbol(state.stream, EZWSymbol::REF1);
            state.reconAbs(x) += Tnext;
        } else {
            appendSymbol(state.stream, EZWSymbol::REF0);
        }
    }
}

void writeEZWStream(const std::string& filename,
                    const std::vector<EZWToken>& stream,
                    int rows,
                    int cols,
                    uint levels,
                    int initialT) {
    std::ofstream output(filename, std::ios::binary);

    if (!output.is_open()) {
        throw std::ios_base::failure("Could not open output file.");
    }

    const char magic[4] = {'E', 'Z', 'W', '1'};
    output.write(magic, 4);

    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    output.write(reinterpret_cast<const char*>(&levels), sizeof(levels));
    output.write(reinterpret_cast<const char*>(&initialT), sizeof(initialT));

    uint64_t count = static_cast<uint64_t>(stream.size());
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const EZWToken& token : stream) {
        const uint8_t value = static_cast<uint8_t>(token.symbol);
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
}

void ezw_encode(const cv::Mat& A,
                const cv::Mat& lpf,
                const cv::Mat& hpf,
                uint levels,
                int init_thresh,
                const std::string& filename) {
    if (A.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }

    if (A.rows != A.cols) {
        throw std::invalid_argument("Input image must be square.");
    }

    if (!isPowerOfTwo(A.rows)) {
        throw std::invalid_argument("Input image size must be a power of two.");
    }

    cv::Mat img;
    A.convertTo(img, CV_64F);

    cv::Mat dwtRaw = decomposeLL(img, lpf, hpf, levels);
    cv::Mat1d C = toDoubleMat(dwtRaw);

    int T = init_thresh;

    if (T <= 0) {
        T = initialThresholdFromCoeffs(C);
    }

    if (T <= 0) {
        writeEZWStream(filename, {}, A.rows, A.cols, levels, 0);
        return;
    }

    EZWTree tree = buildEZWTree(C.rows, C.cols, levels);

    EZWState state;
    state.significant = cv::Mat1b::zeros(C.rows, C.cols);
    state.reconAbs = cv::Mat1d::zeros(C.rows, C.cols);

    const int initialT = T;

    while (T >= minimum_threshold) {
        std::vector<cv::Point> oldLSP = state.lsp;

        dominantPass(C, T, tree, state);
        subordinatePass(C, T, state, oldLSP);

        T >>= 1;
    }

    writeEZWStream(filename, state.stream, A.rows, A.cols, levels, initialT);
}

EZWFile readEZWStream(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);

    if (!input.is_open()) {
        throw std::ios_base::failure("Could not open EZW file.");
    }

    char magic[4]{};
    input.read(magic, 4);

    if (magic[0] != 'E' || magic[1] != 'Z' || magic[2] != 'W' || magic[3] != '1') {
        throw std::runtime_error("Invalid EZW file format.");
    }

    EZWFile file;

    input.read(reinterpret_cast<char*>(&file.rows), sizeof(file.rows));
    input.read(reinterpret_cast<char*>(&file.cols), sizeof(file.cols));
    input.read(reinterpret_cast<char*>(&file.levels), sizeof(file.levels));
    input.read(reinterpret_cast<char*>(&file.initialT), sizeof(file.initialT));

    uint64_t count = 0;
    input.read(reinterpret_cast<char*>(&count), sizeof(count));

    file.stream.resize(static_cast<size_t>(count));

    for (uint64_t i = 0; i < count; ++i) {
        uint8_t value = 0;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        file.stream[static_cast<size_t>(i)] = EZWToken{static_cast<EZWSymbol>(value)};
    }

    return file;
}

void inverseDominantScan(const cv::Point& x,
                         const EZWTree& tree,
                         int T,
                         EZWState& state,
                         const std::vector<EZWToken>& stream,
                         size_t& cursor) {
    if (state.significant(x)) {
        for (const cv::Point& y : tree.getChildren(x)) {
            inverseDominantScan(y, tree, T, state, stream, cursor);
        }
        return;
    }

    if (cursor >= stream.size()) {
        return;
    }

    const EZWSymbol symbol = stream[cursor++].symbol;

    switch (symbol) {
        case EZWSymbol::POS: {
            state.significant(x) = 1;
            state.reconAbs(x) = static_cast<double>(T);
            state.lsp.push_back(x);

            for (const cv::Point& y : tree.getChildren(x)) {
                inverseDominantScan(y, tree, T, state, stream, cursor);
            }

            break;
        }

        case EZWSymbol::NEG: {
            state.significant(x) = 1;
            state.reconAbs(x) = -static_cast<double>(T);
            state.lsp.push_back(x);

            for (const cv::Point& y : tree.getChildren(x)) {
                inverseDominantScan(y, tree, T, state, stream, cursor);
            }

            break;
        }

        case EZWSymbol::IZ: {
            for (const cv::Point& y : tree.getChildren(x)) {
                inverseDominantScan(y, tree, T, state, stream, cursor);
            }

            break;
        }

        case EZWSymbol::ZTR: {
            /*
                Entire subtree remains zero at this threshold.
                No descendant symbols are consumed.
            */
            break;
        }

        default: {
            throw std::runtime_error("Unexpected refinement symbol inside dominant pass.");
        }
    }
}

void inverseDominantPass(const EZWTree& tree,
                         int T,
                         EZWState& state,
                         const std::vector<EZWToken>& stream,
                         size_t& cursor) {
    for (const cv::Point& r : tree.roots) {
        inverseDominantScan(r, tree, T, state, stream, cursor);
    }
}

void inverseSubordinatePass(int T,
                            EZWState& state,
                            const std::vector<cv::Point>& oldLSP,
                            const std::vector<EZWToken>& stream,
                            size_t& cursor) {
    const double Tnext = static_cast<double>(T) / 2.0;

    for (const cv::Point& x : oldLSP) {
        if (cursor >= stream.size()) {
            return;
        }

        const EZWSymbol symbol = stream[cursor++].symbol;

        if (symbol == EZWSymbol::REF1) {
            if (state.reconAbs(x) >= 0.0) {
                state.reconAbs(x) += Tnext;
            } else {
                state.reconAbs(x) -= Tnext;
            }
        } else if (symbol == EZWSymbol::REF0) {
            /*
                No magnitude refinement at this bit-plane.
            */
        } else {
            throw std::runtime_error("Unexpected dominant symbol inside subordinate pass.");
        }
    }
}

cv::Mat1d ezw_decode_coeffs(const std::string& filename) {
    EZWFile file = readEZWStream(filename);

    if (file.initialT <= 0) {
        return cv::Mat1d::zeros(file.rows, file.cols);
    }

    EZWTree tree = buildEZWTree(file.rows, file.cols, file.levels);

    EZWState state;
    state.significant = cv::Mat1b::zeros(file.rows, file.cols);
    state.reconAbs = cv::Mat1d::zeros(file.rows, file.cols);

    size_t cursor = 0;
    int T = file.initialT;

    while (T >= minimum_threshold && cursor < file.stream.size()) {
        std::vector<cv::Point> oldLSP = state.lsp;

        inverseDominantPass(tree, T, state, file.stream, cursor);
        inverseSubordinatePass(T, state, oldLSP, file.stream, cursor);

        T >>= 1;
    }

    return state.reconAbs;
}

cv::Mat ezw_decode(const std::string& filename,
                   const cv::Mat& lpf,
                   const cv::Mat& hpf,
                   uint levels) {
    cv::Mat1d coeffs = ezw_decode_coeffs(filename);

    /*
        Replace this with your actual inverse wavelet reconstruction function.
        Example placeholder:
    */
    cv::Mat reconstructed = reconstructLL(coeffs, lpf, hpf, levels);

    return reconstructed;
}
