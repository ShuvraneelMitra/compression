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
*/



