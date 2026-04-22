cmake_minimum_required(VERSION 3.15)
project(opencv_test)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

find_package(OpenCV REQUIRED)

add_executable(test 
               tests/test_huffman.cpp
               metrics.cpp
               lossless/huffman.cpp
              )
target_link_libraries(test PRIVATE ${OpenCV_LIBS})