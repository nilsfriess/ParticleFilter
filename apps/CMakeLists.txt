project(SMCPF-app)

set(app_SOURCES main.cc)


add_executable(app ${app_SOURCES})

set_target_properties(app PROPERTIES
		    CXX_STANDARD 17
		    CXX_EXTENSIONS OFF)

target_compile_options(app PRIVATE -O3 -Wall -Wextra -march=native -pipe)
target_link_libraries(app PRIVATE lapack blaze gcem stats SMCPF tbb)
