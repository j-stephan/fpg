if(CUDA_FOUND)
    add_subdirectory(cuda)
endif()

if(HCC_FOUND)
    add_subdirectory(rocm/hc)
endif()

if(HIP_FOUND)
    add_subdirectory(rocm/hip)
endif()

if(ComputeCpp_FOUND)
    add_subdirectory(sycl)
endif()
