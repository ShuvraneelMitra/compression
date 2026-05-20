cmake_minimum_required(VERSION 3.15)
project(compression)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

find_package(OpenCV REQUIRED)

add_executable(test 
               tests/test_subband.cpp
               metrics.cpp
               lossless/huffman.cpp
               transform/subband.cpp
              )
target_link_libraries(test PRIVATE ${OpenCV_LIBS})
