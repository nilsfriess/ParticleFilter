project(SMCPF-app)
set(CMAKE_BUILD_TYPE Debug)

set(app_SOURCES main.cc)


add_executable(app ${app_SOURCES})

set_target_properties(app PROPERTIES
		    CXX_STANDARD 17
		    CXX_EXTENSIONS OFF)

target_compile_options(app PRIVATE -O3 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual -Wpedantic -Wsign-conversion -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference -Wuseless-cast -Wdouble-promotion -Wformat=2 -march=native -pipe)
target_link_libraries(app PRIVATE lapack armadillo gcem stats SMCPF tbb)
