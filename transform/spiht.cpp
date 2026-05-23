#include "spiht.hpp"

namespace spiht
{

    SPIHT::SPIHT(int rows, int cols, int levels)
        : rows_(rows),
        cols_(cols),
        levels_(levels)
    {
    }

    bool SPIHT::is_significant(
        int value,
        int threshold
    ) const
    {
        return std::abs(value) >= threshold;
    }

    int SPIHT::initial_threshold(
        const std::vector<std::vector<int>>& coeffs
    ) const
    {
        int maximum = 0;

        for(const auto& row : coeffs)
        {
            for(int v : row)
            {
                maximum = std::max(maximum, std::abs(v));
            }
        }

        if(maximum == 0)
        {
            return 1;
        }

        return static_cast<int>(
            std::pow(
                2,
                static_cast<int>(
                    std::floor(std::log2(maximum))
                )
            )
        );
    }

    std::vector<Coordinate> SPIHT::initial_roots() const
    {
        int root_rows = rows_ >> levels_;
        int root_cols = cols_ >> levels_;

        std::vector<Coordinate> roots;

        for(int r = 0; r < root_rows; ++r)
        {
            for(int c = 0; c < root_cols; ++c)
            {
                roots.emplace_back(r, c);
            }
        }

        return roots;
    }

    bool SPIHT::has_offspring(
        const Coordinate& c
    ) const
    {
        int r = c.row * 2;
        int col = c.col * 2;

        return (
            r + 1 < rows_ &&
            col + 1 < cols_
        );
    }

    std::vector<Coordinate> SPIHT::offspring(
        const Coordinate& c
    ) const
    {
        std::vector<Coordinate> children;

        int r = c.row * 2;
        int col = c.col * 2;

        if(r + 1 >= rows_ || col + 1 >= cols_)
        {
            return children;
        }

        children.emplace_back(r,     col);
        children.emplace_back(r,     col + 1);
        children.emplace_back(r + 1, col);
        children.emplace_back(r + 1, col + 1);

        return children;
    }

    std::vector<Coordinate> SPIHT::descendants(
        const Coordinate& c
    ) const
    {
        std::vector<Coordinate> result;

        std::queue<Coordinate> q;

        for(const auto& child : offspring(c))
        {
            q.push(child);
        }

        while(!q.empty())
        {
            Coordinate current = q.front();
            q.pop();

            result.push_back(current);

            for(const auto& child : offspring(current))
            {
                q.push(child);
            }
        }

        return result;
    }

    std::vector<Coordinate> SPIHT::grand_descendants(
        const Coordinate& c
    ) const
    {
        std::vector<Coordinate> result;

        for(const auto& child : offspring(c))
        {
            for(const auto& grandchild : descendants(child))
            {
                result.push_back(grandchild);
            }
        }

        return result;
    }

    bool SPIHT::set_significant_D(
        const std::vector<std::vector<int>>& coeffs,
        const Coordinate& c,
        int threshold
    ) const
    {
        auto desc = descendants(c);

        for(const auto& d : desc)
        {
            if(is_significant(coeffs[d.row][d.col], threshold))
            {
                return true;
            }
        }

        return false;
    }

    bool SPIHT::set_significant_L(
        const std::vector<std::vector<int>>& coeffs,
        const Coordinate& c,
        int threshold
    ) const
    {
        auto desc = grand_descendants(c);

        for(const auto& d : desc)
        {
            if(is_significant(coeffs[d.row][d.col], threshold))
            {
                return true;
            }
        }

        return false;
    }

    SPIHTBitstream SPIHT::encode(
        const std::vector<std::vector<int>>& coeffs,
        int max_passes
    )
    {
        SPIHTBitstream stream;

        stream.rows = rows_;
        stream.cols = cols_;
        stream.levels = levels_;

        int threshold = initial_threshold(coeffs);

        stream.initial_threshold = threshold;

        std::vector<Coordinate> LIP;
        std::vector<LISNode> LIS;
        std::vector<Coordinate> LSP;

        auto roots = initial_roots();

        for(const auto& r : roots)
        {
            LIP.push_back(r);

            if(has_offspring(r))
            {
                LIS.emplace_back(r, SetType::D);
            }
        }

        for(int pass = 0; pass < max_passes; ++pass)
        {
            if(threshold < 1) break;

            size_t lip_size = LIP.size();
            std::vector<Coordinate> new_lsp_entries;
            for(size_t i = 0; i < lip_size; )
            {
                Coordinate c = LIP[i];

                bool significant = is_significant(coeffs[c.row][c.col], threshold);

                stream.bits.push_back(significant ? 1 : 0);

                if(significant)
                {
                    int sign = coeffs[c.row][c.col] < 0 ? 1 : 0;
                    stream.bits.push_back(sign);
                    new_lsp_entries.push_back(c);
                    LIP.erase(LIP.begin() + i);
                    lip_size--;
                }
                else i++;
            }

            size_t lis_size = LIS.size();

            for(size_t i = 0; i < lis_size; )
            {
                LISNode node = LIS[i];
                bool significant = false;

                if(node.type == SetType::D) significant = set_significant_D(coeffs, node.coord, threshold);
                else significant = set_significant_L(coeffs, node.coord, threshold);
                
                stream.bits.push_back(significant ? 1 : 0);

                if(!significant)
                {
                    ++i;
                    continue;
                }

                if(node.type == SetType::D)
                {
                    auto children = offspring(node.coord);

                    for(const auto& child : children)
                    {
                        bool child_sig = is_significant(coeffs[child.row][child.col], threshold);

                        stream.bits.push_back(child_sig ? 1 : 0);

                        if(child_sig)
                        {
                            int sign = coeffs[child.row][child.col] < 0 ? 1 : 0;
                            stream.bits.push_back(sign);
                            new_lsp_entries.push_back(child);
                        }
                        else LIP.push_back(child);
                    }

                    if(has_offspring(children[0]))
                    {
                        LIS[i].type = SetType::L;
                        ++i;
                    }
                    else
                    {
                        LIS.erase(LIS.begin() + i);
                        lis_size--;
                    }

                    for(const auto& child : children)
                    {
                        if(has_offspring(child))
                            LIS.emplace_back(child, SetType::D);
                    }
                }
                else
                {
                    auto children = offspring(node.coord);

                    for(const auto& child : children)
                    {
                        if(has_offspring(child))
                        {
                            bool sig = set_significant_D(coeffs, child, threshold);
                            stream.bits.push_back(sig ? 1 : 0);

                            if(sig) LIS.emplace_back(child, SetType::D);
                            else LIS.emplace_back(child, SetType::L);
                        }
                    }

                    LIS.erase(LIS.begin() + i);
                    lis_size--;
                }
            }

            for(const auto& c : new_lsp_entries) LSP.push_back(c);
            int refinement_bit = threshold >> 1;

            if(refinement_bit > 0)
            {
                size_t old_lsp_size =
                    LSP.size() - new_lsp_entries.size();

                for(size_t i = 0; i < old_lsp_size; ++i)
                {
                    Coordinate c = LSP[i];

                    int magnitude = std::abs(coeffs[c.row][c.col]);
                    int bit = (magnitude & refinement_bit) ? 1 : 0;
                    stream.bits.push_back(bit);
                }
            }

            threshold >>= 1;
        }

        return stream;
    }

    std::vector<std::vector<int>> SPIHT::decode(const SPIHTBitstream& stream, int max_passes)
    {
        std::vector<std::vector<int>> coeffs(
            rows_,
            std::vector<int>(cols_, 0)
        );

        int threshold =
            stream.initial_threshold;

        size_t ptr = 0;

        std::vector<Coordinate> LIP;
        std::vector<LISNode> LIS;
        std::vector<Coordinate> LSP;

        auto roots = initial_roots();

        for(const auto& r : roots)
        {
            LIP.push_back(r);
            if(has_offspring(r)) LIS.emplace_back(r, SetType::D);
        }

        for(int pass = 0; pass < max_passes; ++pass)
        {
            if(threshold < 1) break;

            std::vector<Coordinate> new_lsp_entries;
            size_t lip_size = LIP.size();

            for(size_t i = 0; i < lip_size; )
            {
                Coordinate c = LIP[i];
                int significant = stream.bits[ptr++];

                if(significant)
                {
                    int sign = stream.bits[ptr++];
                    coeffs[c.row][c.col] = sign ? -threshold : threshold;
                    new_lsp_entries.push_back(c);

                    LIP.erase(LIP.begin() + i);
                    lip_size--;
                }
                else ++i;
            }
            size_t lis_size = LIS.size();

            for(size_t i = 0; i < lis_size; )
            {
                LISNode node = LIS[i];

                int significant = stream.bits[ptr++];

                if(!significant)
                {
                    ++i;
                    continue;
                }

                if(node.type == SetType::D)
                {
                    auto children = offspring(node.coord);

                    for(const auto& child : children)
                    {
                        int child_sig = stream.bits[ptr++];

                        if(child_sig)
                        {
                            int sign = stream.bits[ptr++];
                            coeffs[child.row][child.col] = sign ? -threshold : threshold;
                            new_lsp_entries.push_back(child);
                        }
                        else LIP.push_back(child);
                    }

                    if(has_offspring(children[0]))
                    {
                        LIS[i].type = SetType::L;
                        ++i;
                    }
                    else
                    {
                        LIS.erase(LIS.begin() + i);
                        lis_size--;
                    }

                    for(const auto& child : children)
                    {
                        if(has_offspring(child)) LIS.emplace_back(child, SetType::D);
                    }
                }
                else
                {
                    auto children = offspring(node.coord);

                    for(const auto& child : children)
                    {
                        if(has_offspring(child))
                        {
                            int sig = stream.bits[ptr++];

                            if(sig)
                            {
                                LIS.emplace_back(child, SetType::D);
                            }
                            else
                            {
                                LIS.emplace_back(child, SetType::L);
                            }
                        }
                    }

                    LIS.erase(LIS.begin() + i);
                    lis_size--;
                }
            }

            for(const auto& c : new_lsp_entries)
            {
                LSP.push_back(c);
            }

            int refinement_bit =
                threshold >> 1;

            if(refinement_bit > 0)
            {
                size_t old_lsp_size = LSP.size() - new_lsp_entries.size();

                for(size_t i = 0; i < old_lsp_size; ++i)
                {
                    Coordinate c = LSP[i];

                    int bit =
                        stream.bits[ptr++];

                    if(bit)
                    {
                        if(coeffs[c.row][c.col] >= 0)
                        {
                            coeffs[c.row][c.col] += refinement_bit;
                        }
                        else
                        {
                            coeffs[c.row][c.col] -= refinement_bit;
                        }
                    }
                }
            }

            threshold >>= 1;
        }

        return coeffs;
    }

}
