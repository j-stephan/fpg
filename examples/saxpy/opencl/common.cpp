#include <fstream>
#include <iterator>
#include <string>
#include <utility>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "common.h"

auto create_program(const cl::Context& context, const std::string& path)
    -> cl::Program
{
    auto&& file = std::ifstream{path};
    auto code = std::string{std::istreambuf_iterator<char>{file},
                            {std::istreambuf_iterator<char>{}}};

    auto source = cl::Program::Sources{1,
                                       std::make_pair(code.c_str(),
                                                      code.length() + 1)};

    return cl::Program{context, source};
}
