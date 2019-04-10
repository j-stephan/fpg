if(CUDA_NVCC_FLAGS STREQUAL "")
    set(CUDA_NVCC_FLAGS "-O3 -std=c++14 -gencode arch=${FPG_ARCH},code=${FPG_CODE} -Xcompiler -Wall,-Wextra,-pedantic")
endif()

CUDA_ADD_EXECUTABLE(nbody nbody.cu)
