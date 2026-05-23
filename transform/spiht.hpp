#ifndef SPIHT_HPP
#define SPIHT_HPP

#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace spiht
{

enum class SPIHTSymbol : uint8_t
{
    ZERO = 0,
    ONE  = 1
};

enum class SetType : uint8_t
{
    D,  
    L  
};

struct Coordinate
{
    int row;
    int col;

    Coordinate() : row(0), col(0) {}
    Coordinate(int r, int c) : row(r), col(c) {}

    bool operator==(const Coordinate& other) const
    {
        return row == other.row && col == other.col;
    }
};

struct LISNode
{
    Coordinate coord;
    SetType type;

    LISNode() = default;

    LISNode(Coordinate c, SetType t)
        : coord(c), type(t)
    {}
};

struct SPIHTBitstream
{
    int rows;
    int cols;
    int levels;

    int initial_threshold;

    std::vector<uint8_t> bits;
};

class SPIHT
{
public:

    SPIHT(int rows, int cols, int levels);

    SPIHTBitstream encode(
        const std::vector<std::vector<int>>& coeffs,
        int max_passes
    );

    std::vector<std::vector<int>> decode(
        const SPIHTBitstream& stream,
        int max_passes
    );

private:

    int rows_;
    int cols_;
    int levels_;

private:

    bool is_significant(
        int value,
        int threshold
    ) const;

    bool set_significant_D(
        const std::vector<std::vector<int>>& coeffs,
        const Coordinate& c,
        int threshold
    ) const;

    bool set_significant_L(
        const std::vector<std::vector<int>>& coeffs,
        const Coordinate& c,
        int threshold
    ) const;

    std::vector<Coordinate> offspring(
        const Coordinate& c
    ) const;

    std::vector<Coordinate> descendants(
        const Coordinate& c
    ) const;

    std::vector<Coordinate> grand_descendants(
        const Coordinate& c
    ) const;

    bool has_offspring(
        const Coordinate& c
    ) const;

    std::vector<Coordinate> initial_roots() const;

    int initial_threshold(
        const std::vector<std::vector<int>>& coeffs
    ) const;
};

} // namespace spiht

#endif
