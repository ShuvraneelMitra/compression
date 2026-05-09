# **Multimedia Compression Techniques**

This is a C++ Library which implements a number of algorithms that 
are used for compression of multimedia objects. 

I took a course in my sixth semester called "*Multimedia Systems and Applications*" which motivated me to start this project. The algorithms implemented in this repository are:

- Lossless Compression Algorithms
    - Huffman Coding
    - Lempel-Ziv Coding
    - Run-Length Encoding and BitPlane conversion
    - Lossless Predictive Coding
- Lossy Compression Algorithms
    - Lossy Predictive Coding

We use BitMap (`.bmp`) image files for testing the algorithms because they are in the most uncompressed form.

### **Build instructions**
1. Clone the repository and `cd` into it.
2. Create the CMake build files in a directory called `build` by running the command `cmake -B build -G "MinGW Makefiles" -DOpenCV_DIR=<YOUR_OPENCV_DIRECTORY>`. The placeholder `YOUR_OPENCV_DIRECTORY` should contain the location of your opencv installation, for me it was `C:/opencv/build` since I installed it manually.
3. `cd` into the `build` folder. Build using `cmake --build .`
4. Run the executable.

## **NOTES**
1. The compression achieved via lossless predictive coding, combined with Huffman encoding of the error values, is extremely substantial, with a randomly chosen coefficient set of [0, 1, 3, 2] (before normalization) yielding 81% compression for the grayscale image `grayscale.bmp`, as compared to 8% with only Huffman encoding. 
2. Quantization improves the compression ratio, although with the tradeoff of image quality. With just 4 levels of quantization, we can get upto 93% compression, although with significantly visible deterioration of quality.