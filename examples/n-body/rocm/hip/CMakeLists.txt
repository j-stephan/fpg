if(HIP_HIPCC_FLAGS STREQUAL "")
    set(HIP_HIPCC_FLAGS "-O3 -std=c++14 -Wall -Wextra -pedantic")
endif()

if(HIP_NVCC_FLAGS STREQUAL "")
    set(HIP_NVCC_FLAGS "-O3 -std=c++14 -gencode arch=${FPG_ARCH},code=${FPG_CODE} -Xcompiler -Wall,-Wextra,-pedantic")
endif()

HIP_ADD_EXECUTABLE(nbody nbody.cpp)
