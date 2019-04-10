set(HOST_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "Host C++ compiler.")
set(CMAKE_CXX_COMPILER ${HCC})

execute_process(COMMAND ${HCC_CONFIG} --cxxflags OUTPUT_VARIABLE ${HCC_FLAGS})
execute_process(COMMAND ${HCC_CONFIG} --ldflags OUTPUT_VARIABLE ${HCC_LDFLAGS})

add_executable(nbody nbody.cpp)
target_compile_options(nbody PRIVATE ${HCC_FLAGS} -Wall -Wextra -pedantic)
set_property(TARGET nbody PROPERTY LINK_FLAGS "${HCC_LDFLAGS}")

set(CMAKE_CXX_COMPILER ${HOST_CXX_COMPILER})
