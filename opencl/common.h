#ifndef FPG_CL_COMMON_H
#define FPG_CL_COMMON_H

#include <string>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

auto create_program(const cl::Context& context, const std::string& path)
    -> cl::Program;

#endif
