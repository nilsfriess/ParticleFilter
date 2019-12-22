project(SMCPF-app)

set(app_SOURCES main.cc)


add_executable(app ${app_SOURCES})

target_compile_options(app PRIVATE -O2 -Wall -Wextra -pipe)
target_link_libraries(app PRIVATE lapack blaze gcem stats SMCPF)
