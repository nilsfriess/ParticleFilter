cmake_minimum_required(VERSION 3.7)

# set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks=-*,readability-*)

project(SoftwarePracticalUQ LANGUAGES CXX)

add_subdirectory(libs)
add_subdirectory(apps)
