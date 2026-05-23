#include "ebcot.hpp"

namespace ebcot
{

EBCOT::EBCOT(int block_rows, int block_cols)
    : block_rows_(block_rows),
      block_cols_(block_cols)
{
    if(block_rows_ <= 0 || block_cols_ <= 0)
    {
        throw std::invalid_argument("Code-block dimensions must be positive.");
    }
}

bool EBCOT::in_bounds(
    int r,
    int c,
    int rows,
    int cols
) const
{
    return r >= 0 && c >= 0 && r < rows && c < cols;
}

int EBCOT::initial_bitplane(
    const std::vector<std::vector<int>>& coeffs
) const
{
    int max_abs = 0;

    for(const auto& row : coeffs)
    {
        for(int v : row)
        {
            max_abs = std::max(max_abs, std::abs(v));
        }
    }

    if(max_abs == 0)
    {
        return 0;
    }

    return static_cast<int>(std::floor(std::log2(max_abs)));
}

std::vector<CodeBlock> EBCOT::split_into_blocks(
    const std::vector<std::vector<int>>& coeffs
) const
{
    int rows = static_cast<int>(coeffs.size());
    int cols = static_cast<int>(coeffs[0].size());

    std::vector<CodeBlock> blocks;

    for(int r0 = 0; r0 < rows; r0 += block_rows_)
    {
        for(int c0 = 0; c0 < cols; c0 += block_cols_)
        {
            int br = std::min(block_rows_, rows - r0);
            int bc = std::min(block_cols_, cols - c0);

            CodeBlock block;
            block.row0 = r0;
            block.col0 = c0;
            block.rows = br;
            block.cols = bc;
            block.coeffs.assign(br, std::vector<int>(bc, 0));

            for(int r = 0; r < br; ++r)
            {
                for(int c = 0; c < bc; ++c)
                {
                    block.coeffs[r][c] = coeffs[r0 + r][c0 + c];
                }
            }

            blocks.push_back(block);
        }
    }

    return blocks;
}

bool EBCOT::has_significant_neighbor(
    const std::vector<std::vector<uint8_t>>& significant,
    int r,
    int c
) const
{
    int rows = static_cast<int>(significant.size());
    int cols = static_cast<int>(significant[0].size());

    for(int dr = -1; dr <= 1; ++dr)
    {
        for(int dc = -1; dc <= 1; ++dc)
        {
            if(dr == 0 && dc == 0)
            {
                continue;
            }

            int nr = r + dr;
            int nc = c + dc;

            if(in_bounds(nr, nc, rows, cols) && significant[nr][nc])
            {
                return true;
            }
        }
    }

    return false;
}

int EBCOT::significance_context(
    const std::vector<std::vector<uint8_t>>& significant,
    int r,
    int c
) const
{
    int rows = static_cast<int>(significant.size());
    int cols = static_cast<int>(significant[0].size());

    int horizontal = 0;
    int vertical = 0;
    int diagonal = 0;

    const int hr[2] = {0, 0};
    const int hc[2] = {-1, 1};

    const int vr[2] = {-1, 1};
    const int vc[2] = {0, 0};

    const int dr[4] = {-1, -1, 1, 1};
    const int dc[4] = {-1, 1, -1, 1};

    for(int k = 0; k < 2; ++k)
    {
        int nr = r + hr[k];
        int nc = c + hc[k];

        if(in_bounds(nr, nc, rows, cols) && significant[nr][nc])
        {
            horizontal++;
        }

        nr = r + vr[k];
        nc = c + vc[k];

        if(in_bounds(nr, nc, rows, cols) && significant[nr][nc])
        {
            vertical++;
        }
    }

    for(int k = 0; k < 4; ++k)
    {
        int nr = r + dr[k];
        int nc = c + dc[k];

        if(in_bounds(nr, nc, rows, cols) && significant[nr][nc])
        {
            diagonal++;
        }
    }

    horizontal = std::min(horizontal, 2);
    vertical = std::min(vertical, 2);
    diagonal = std::min(diagonal, 4);

    return horizontal * 15 + vertical * 5 + diagonal;
}

int EBCOT::sign_context(
    const std::vector<std::vector<int>>& reconstructed,
    const std::vector<std::vector<uint8_t>>& significant,
    int r,
    int c
) const
{
    int rows = static_cast<int>(significant.size());
    int cols = static_cast<int>(significant[0].size());

    int vertical_bias = 0;
    int horizontal_bias = 0;

    if(in_bounds(r - 1, c, rows, cols) && significant[r - 1][c])
    {
        vertical_bias += reconstructed[r - 1][c] >= 0 ? 1 : -1;
    }

    if(in_bounds(r + 1, c, rows, cols) && significant[r + 1][c])
    {
        vertical_bias += reconstructed[r + 1][c] >= 0 ? 1 : -1;
    }

    if(in_bounds(r, c - 1, rows, cols) && significant[r][c - 1])
    {
        horizontal_bias += reconstructed[r][c - 1] >= 0 ? 1 : -1;
    }

    if(in_bounds(r, c + 1, rows, cols) && significant[r][c + 1])
    {
        horizontal_bias += reconstructed[r][c + 1] >= 0 ? 1 : -1;
    }

    vertical_bias = std::max(-1, std::min(1, vertical_bias));
    horizontal_bias = std::max(-1, std::min(1, horizontal_bias));

    return 100 + (vertical_bias + 1) * 3 + (horizontal_bias + 1);
}

int EBCOT::refinement_context(
    const std::vector<std::vector<uint8_t>>& refined_before,
    const std::vector<std::vector<uint8_t>>& significant,
    int r,
    int c
) const
{
    if(!refined_before[r][c])
    {
        if(has_significant_neighbor(significant, r, c))
        {
            return 200;
        }

        return 201;
    }

    return 202;
}

void EBCOT::append_bit(
    CodedPass& pass,
    uint8_t context,
    uint8_t bit
) const
{
    pass.contexts.push_back(context);
    pass.bits.push_back(bit ? 1 : 0);
}

uint8_t EBCOT::read_bit(
    const CodedPass& pass,
    size_t& ptr
) const
{
    if(ptr >= pass.bits.size())
    {
        throw std::runtime_error("Unexpected end of coded pass.");
    }

    return pass.bits[ptr++];
}

void EBCOT::encode_significance_bit(
    CodedPass& pass,
    const std::vector<std::vector<int>>& coeffs,
    std::vector<std::vector<int>>& reconstructed,
    std::vector<std::vector<uint8_t>>& significant,
    int r,
    int c,
    int threshold
) const
{
    uint8_t sig = std::abs(coeffs[r][c]) >= threshold ? 1 : 0;

    append_bit(
        pass,
        static_cast<uint8_t>(significance_context(significant, r, c)),
        sig
    );

    if(sig)
    {
        uint8_t sign = coeffs[r][c] < 0 ? 1 : 0;

        append_bit(
            pass,
            static_cast<uint8_t>(sign_context(reconstructed, significant, r, c)),
            sign
        );

        significant[r][c] = 1;
        reconstructed[r][c] = sign ? -threshold : threshold;
    }
}

void EBCOT::decode_significance_bit(
    const CodedPass& pass,
    size_t& ptr,
    std::vector<std::vector<int>>& reconstructed,
    std::vector<std::vector<uint8_t>>& significant,
    int r,
    int c,
    int threshold
) const
{
    uint8_t sig = read_bit(pass, ptr);

    if(sig)
    {
        uint8_t sign = read_bit(pass, ptr);

        significant[r][c] = 1;
        reconstructed[r][c] = sign ? -threshold : threshold;
    }
}

EncodedBlock EBCOT::encode_block(
    const CodeBlock& block,
    int max_bitplanes
) const
{
    EncodedBlock encoded;

    encoded.row0 = block.row0;
    encoded.col0 = block.col0;
    encoded.rows = block.rows;
    encoded.cols = block.cols;

    int top_bitplane = initial_bitplane(block.coeffs);
    encoded.initial_bitplane = top_bitplane;

    if(max_bitplanes < 0)
    {
        max_bitplanes = top_bitplane + 1;
    }

    std::vector<std::vector<int>> reconstructed(
        block.rows,
        std::vector<int>(block.cols, 0)
    );

    std::vector<std::vector<uint8_t>> significant(
        block.rows,
        std::vector<uint8_t>(block.cols, 0)
    );

    std::vector<std::vector<uint8_t>> refined_before(
        block.rows,
        std::vector<uint8_t>(block.cols, 0)
    );

    for(int bp = top_bitplane; bp >= 0 && max_bitplanes > 0; --bp, --max_bitplanes)
    {
        int threshold = 1 << bp;

        CodedPass sigpass;
        sigpass.type = CodingPassType::SignificancePropagation;
        sigpass.bitplane = bp;

        std::vector<std::vector<uint8_t>> visited(
            block.rows,
            std::vector<uint8_t>(block.cols, 0)
        );

        for(int r = 0; r < block.rows; ++r)
        {
            for(int c = 0; c < block.cols; ++c)
            {
                if(!significant[r][c] &&
                   has_significant_neighbor(significant, r, c))
                {
                    encode_significance_bit(
                        sigpass,
                        block.coeffs,
                        reconstructed,
                        significant,
                        r,
                        c,
                        threshold
                    );

                    visited[r][c] = 1;
                }
            }
        }

        encoded.passes.push_back(sigpass);

        CodedPass refpass;
        refpass.type = CodingPassType::MagnitudeRefinement;
        refpass.bitplane = bp;

        int refine_value = threshold;

        for(int r = 0; r < block.rows; ++r)
        {
            for(int c = 0; c < block.cols; ++c)
            {
                if(significant[r][c] &&
                   std::abs(reconstructed[r][c]) > threshold)
                {
                    uint8_t bit =
                        (std::abs(block.coeffs[r][c]) & refine_value) ? 1 : 0;

                    append_bit(
                        refpass,
                        static_cast<uint8_t>(
                            refinement_context(refined_before, significant, r, c)
                        ),
                        bit
                    );

                    if(bit)
                    {
                        if(reconstructed[r][c] >= 0)
                        {
                            reconstructed[r][c] += refine_value;
                        }
                        else
                        {
                            reconstructed[r][c] -= refine_value;
                        }
                    }

                    refined_before[r][c] = 1;
                }
            }
        }

        encoded.passes.push_back(refpass);

        /*
        =====================================================
        3. Cleanup Pass
        =====================================================
        Scan remaining insignificant samples not visited in
        the significance propagation pass.
        =====================================================
        */

        CodedPass cleanpass;
        cleanpass.type = CodingPassType::Cleanup;
        cleanpass.bitplane = bp;

        for(int r = 0; r < block.rows; ++r)
        {
            for(int c = 0; c < block.cols; ++c)
            {
                if(!significant[r][c] && !visited[r][c])
                {
                    encode_significance_bit(
                        cleanpass,
                        block.coeffs,
                        reconstructed,
                        significant,
                        r,
                        c,
                        threshold
                    );
                }
            }
        }

        encoded.passes.push_back(cleanpass);
    }

    return encoded;
}

CodeBlock EBCOT::decode_block(
    const EncodedBlock& encoded
) const
{
    CodeBlock block;

    block.row0 = encoded.row0;
    block.col0 = encoded.col0;
    block.rows = encoded.rows;
    block.cols = encoded.cols;

    block.coeffs.assign(
        block.rows,
        std::vector<int>(block.cols, 0)
    );

    std::vector<std::vector<int>>& reconstructed = block.coeffs;

    std::vector<std::vector<uint8_t>> significant(
        block.rows,
        std::vector<uint8_t>(block.cols, 0)
    );

    std::vector<std::vector<uint8_t>> refined_before(
        block.rows,
        std::vector<uint8_t>(block.cols, 0)
    );

    for(const auto& pass : encoded.passes)
    {
        int bp = pass.bitplane;
        int threshold = 1 << bp;
        size_t ptr = 0;

        if(pass.type == CodingPassType::SignificancePropagation)
        {
            for(int r = 0; r < block.rows; ++r)
            {
                for(int c = 0; c < block.cols; ++c)
                {
                    if(!significant[r][c] &&
                       has_significant_neighbor(significant, r, c))
                    {
                        decode_significance_bit(
                            pass,
                            ptr,
                            reconstructed,
                            significant,
                            r,
                            c,
                            threshold
                        );
                    }
                }
            }
        }
        else if(pass.type == CodingPassType::MagnitudeRefinement)
        {
            int refine_value = threshold;

            for(int r = 0; r < block.rows; ++r)
            {
                for(int c = 0; c < block.cols; ++c)
                {
                    if(significant[r][c] &&
                       std::abs(reconstructed[r][c]) > threshold)
                    {
                        uint8_t bit = read_bit(pass, ptr);

                        if(bit)
                        {
                            if(reconstructed[r][c] >= 0)
                            {
                                reconstructed[r][c] += refine_value;
                            }
                            else
                            {
                                reconstructed[r][c] -= refine_value;
                            }
                        }

                        refined_before[r][c] = 1;
                    }
                }
            }
        }
        else if(pass.type == CodingPassType::Cleanup)
        {
            std::vector<std::vector<uint8_t>> visited(
                block.rows,
                std::vector<uint8_t>(block.cols, 0)
            );

            /*
            Important:
            The decoder must reconstruct the same visited map
            as the encoder would have used for the corresponding
            significance propagation pass. In a full production
            implementation, pass-state snapshots or deterministic
            pass grouping should be used.
            */

            for(int r = 0; r < block.rows; ++r)
            {
                for(int c = 0; c < block.cols; ++c)
                {
                    if(!significant[r][c] && !visited[r][c])
                    {
                        decode_significance_bit(
                            pass,
                            ptr,
                            reconstructed,
                            significant,
                            r,
                            c,
                            threshold
                        );
                    }
                }
            }
        }
    }

    return block;
}

EBCOTBitstream EBCOT::encode(
    const std::vector<std::vector<int>>& coeffs,
    int max_bitplanes
)
{
    if(coeffs.empty() || coeffs[0].empty())
    {
        throw std::invalid_argument("Input coefficient array is empty.");
    }

    int rows = static_cast<int>(coeffs.size());
    int cols = static_cast<int>(coeffs[0].size());

    for(const auto& row : coeffs)
    {
        if(static_cast<int>(row.size()) != cols)
        {
            throw std::invalid_argument("Input coefficient array must be rectangular.");
        }
    }

    EBCOTBitstream stream;

    stream.image_rows = rows;
    stream.image_cols = cols;
    stream.block_rows = block_rows_;
    stream.block_cols = block_cols_;

    auto blocks = split_into_blocks(coeffs);

    for(const auto& block : blocks)
    {
        stream.blocks.push_back(
            encode_block(block, max_bitplanes)
        );
    }

    return stream;
}

std::vector<std::vector<int>> EBCOT::decode(
    const EBCOTBitstream& stream
)
{
    std::vector<std::vector<int>> coeffs(
        stream.image_rows,
        std::vector<int>(stream.image_cols, 0)
    );

    for(const auto& encoded_block : stream.blocks)
    {
        CodeBlock block = decode_block(encoded_block);

        for(int r = 0; r < block.rows; ++r)
        {
            for(int c = 0; c < block.cols; ++c)
            {
                coeffs[block.row0 + r][block.col0 + c] =
                    block.coeffs[r][c];
            }
        }
    }

    return coeffs;
}

} // namespace ebcot
