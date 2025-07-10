cmake_minimum_required(VERSION 3.29)
project(projecto_final_grupogpt)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Ejecutables
add_executable(train train.cpp)

add_executable(predict predict.cpp utils/common_helpers.h utils/ascii_view.h)
