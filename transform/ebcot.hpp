#ifndef EBCOT_HPP
#define EBCOT_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace ebcot
{

struct Coord
{
    int r;
    int c;

    Coord() : r(0), c(0) {}
    Coord(int row, int col) : r(row), c(col) {}
};

struct CodeBlock
{
    int row0;
    int col0;
    int rows;
    int cols;

    std::vector<std::vector<int>> coeffs;
};

enum class CodingPassType : uint8_t
{
    SignificancePropagation = 0,
    MagnitudeRefinement     = 1,
    Cleanup                 = 2
};

struct CodedPass
{
    CodingPassType type;
    int bitplane;
    std::vector<uint8_t> bits;
    std::vector<uint8_t> contexts;
};

struct EncodedBlock
{
    int row0;
    int col0;
    int rows;
    int cols;
    int initial_bitplane;

    std::vector<CodedPass> passes;
};

struct EBCOTBitstream
{
    int image_rows;
    int image_cols;
    int block_rows;
    int block_cols;

    std::vector<EncodedBlock> blocks;
};

class EBCOT
{
public:
    EBCOT(int block_rows = 32, int block_cols = 32);

    EBCOTBitstream encode(
        const std::vector<std::vector<int>>& coeffs,
        int max_bitplanes = -1
    );

    std::vector<std::vector<int>> decode(
        const EBCOTBitstream& stream
    );

private:
    int block_rows_;
    int block_cols_;

private:
    std::vector<CodeBlock> split_into_blocks(
        const std::vector<std::vector<int>>& coeffs
    ) const;

    EncodedBlock encode_block(
        const CodeBlock& block,
        int max_bitplanes
    ) const;

    CodeBlock decode_block(
        const EncodedBlock& encoded
    ) const;

    int initial_bitplane(
        const std::vector<std::vector<int>>& coeffs
    ) const;

    bool in_bounds(
        int r,
        int c,
        int rows,
        int cols
    ) const;

    bool has_significant_neighbor(
        const std::vector<std::vector<uint8_t>>& significant,
        int r,
        int c
    ) const;

    int significance_context(
        const std::vector<std::vector<uint8_t>>& significant,
        int r,
        int c
    ) const;

    int sign_context(
        const std::vector<std::vector<int>>& reconstructed,
        const std::vector<std::vector<uint8_t>>& significant,
        int r,
        int c
    ) const;

    int refinement_context(
        const std::vector<std::vector<uint8_t>>& refined_before,
        const std::vector<std::vector<uint8_t>>& significant,
        int r,
        int c
    ) const;

    void append_bit(
        CodedPass& pass,
        uint8_t context,
        uint8_t bit
    ) const;

    uint8_t read_bit(
        const CodedPass& pass,
        size_t& ptr
    ) const;

    void encode_significance_bit(
        CodedPass& pass,
        const std::vector<std::vector<int>>& coeffs,
        std::vector<std::vector<int>>& reconstructed,
        std::vector<std::vector<uint8_t>>& significant,
        int r,
        int c,
        int threshold
    ) const;

    void decode_significance_bit(
        const CodedPass& pass,
        size_t& ptr,
        std::vector<std::vector<int>>& reconstructed,
        std::vector<std::vector<uint8_t>>& significant,
        int r,
        int c,
        int threshold
    ) const;
};

} // namespace ebcot

#endif
