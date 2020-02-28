cmake_minimum_required(VERSION 3.7)
project(SMCPF-apps)

set(CMAKE_BUILD_TYPE Debug)
set(CUSTOM_FLAGS -O3 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wpedantic -Wsign-conversion -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference -Wuseless-cast -Wdouble-promotion -Wformat=2 -march=native -pipe)

include_directories("./include/gcem/include" "./include/stats/include")

########################################################################
# Example 1
# Simple nonlinear non-Gaussian model
########################################################################
add_executable(example1 example1/main.cc)      
set_target_properties(example1 PROPERTIES
		      CXX_STANDARD 17
		      CXX_EXTENSIONS OFF)

target_compile_options(example1 PRIVATE ${CUSTOM_FLAGS})
target_link_libraries(example1 PRIVATE smcpf)

########################################################################
# Example 1
# Lotka volterra model with bootstrap filter
########################################################################

add_executable(example2 example2/main.cc)
set_target_properties(example2 PROPERTIES
		      CXX_STANDARD 17
		      CXX_EXTENSIONS OFF)

target_compile_options(example2 PRIVATE ${CUSTOM_FLAGS})
target_link_libraries(example2 PRIVATE lapack armadillo tbb smcpf)

########################################################################
# Example 1
# Lotka volterra model with optimal filter
########################################################################

# TODO #
