add_executable(nbody nbody.cpp)
add_sycl_to_target(TARGET nbody SOURCES nbody.cpp)
