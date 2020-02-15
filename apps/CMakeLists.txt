cmake_minimum_required(VERSION 3.7)
project(SMCPF-apps)

set(CMAKE_BUILD_TYPE Debug)
set(CUSTOM_FLAGS -O3 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wpedantic -Wsign-conversion -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference -Wuseless-cast -Wdouble-promotion -Wformat=2 -march=native -pipe)

include_directories("./include/gcem/include" "./include/stats/include")


add_executable(example1 example_1/main.cc)

set_target_properties(example1 PROPERTIES
		      CXX_STANDARD 17
		      CXX_EXTENSIONS OFF)

target_compile_options(example1 PRIVATE ${CUSTOM_FLAGS})
target_include_directories(example1 PRIVATE ${YOUR_DIRECTORY})

target_link_libraries(example1 PRIVATE lapack armadillo tbb smcpf)
