# **Multimedia Compression Techniques**

This is a C++ Library which implements a number of algorithms that 
are used for compression of multimedia objects. 

I took a course in my sixth semester called "*Multimedia Systems and Applications*" which motivated me to start this project. The algorithms implemented in this repository are:

- Lossless Compression Algorithms
    - Huffman Coding
    - Lempel-Ziv Coding
    - Run-Length Encoding and BitPlane conversion
    - Lossless Predictive Coding

## **NOTES**
1. The compression achieved via lossless predictive coding, combined with Huffman encoding of the error values, is extremely substantial, with a randomly chosen coefficient set of [0, 1, 3, 2] (before normalization) yielding 81% compression for the grayscale image `grayscale.bmp`, as compared to 8% with only Huffman encoding. 

Build with `cmake -B build -G "MinGW Makefiles" -DOpenCV_DIR=C:/opencv/build`