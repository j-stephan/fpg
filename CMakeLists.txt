cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(fpg LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# detect CUDA
find_package(CUDA)
if(NOT CUDA_FOUND)
    message(STATUS "No CUDA package found, disabling CUDA benchmarks.")
else()
    if(FPG_ARCH STREQUAL "")
        set(FPG_ARCH "compute_35")
    endif()
    if(FPG_CODE STREQUAL "")
        set(FPG_CODE "sm_35")
    endif()
endif()

# detect HC
find_program(HCC NAMES         hcc
                 PATHS         /opt/rocm
                               /opt/rocm/hcc
                               ENV HCC_BIN_DIR
                 PATH_SUFFIXES bin)

find_program(HCC_CONFIG NAMES         hcc-config
                        PATHS         /opt/rocm
                                      /opt/rocm/hcc
                                      ENV HCC_BIN_DIR
                        PATH_SUFFIXES bin)

if(HCC STREQUAL HCC-NOTFOUND OR HCC_CONFIG STREQUAL HCC_CONFIG-NOTFOUND)
    message(STATUS "No hc compiler found, disabling hc benchmarks.")
else()
    message ("${HCC}")
    message ("${HCC_CONFIG}")
    set(HCC_FOUND TRUE)
endif() 

# detect HIP
find_package(HIP)
if(NOT HIP_FOUND)
    message(STATUS "No HIP package found, disabling HIP benchmarks.")
endif()

# detect SYCL
# You may set COMPUTECPP_USER_FLAGS to your desired compilation flags
# You must set COMPUTECPP_BITCODE to one of {spir(64),spirv(64),ptx(64)}
# You must set ComputeCpp_DIR to the root directory of your ComputeCpp installation
find_package(ComputeCpp)
if(NOT ComputeCpp_FOUND)
    message(STATUS "No ComputeCpp package found, disabling SYCL benchmarks.")
endif()
