cmake_minimum_required(VERSION 3.15)
project(opencv_test)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

find_package(OpenCV REQUIRED)

add_executable(test 
               tests/test_lz.cpp
               metrics.cpp
               lossless/lempel_ziv.cpp
              )
target_link_libraries(test PRIVATE ${OpenCV_LIBS})