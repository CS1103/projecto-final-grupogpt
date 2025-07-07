cmake_minimum_required(VERSION 3.29)
project(projecto_final_grupogpt)

set(CMAKE_CXX_STANDARD 20)

# This creates the "train.exe" program from only train.cpp
add_executable(train train.cpp)

# This creates the "predict.exe" program from only predict.cpp
add_executable(predict predict.cpp
        common_helpers.h)
