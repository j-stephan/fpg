#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// get rid of nasty deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "common.h"

constexpr auto N = 1024 * 500;
constexpr auto a = 100.f;

// len(x) == N
auto algo_cpu(const float* x, float* y) -> std::chrono::duration<double>
{
    auto start = std::chrono::steady_clock::now();
    for(auto i = 0; i < N; ++i)
        y[i] = a * x[i] + y[i];
    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>{stop - start};
}

auto algo_gpu(float* x, float* y) -> std::chrono::duration<double>
{

    auto platforms = std::vector<cl::Platform>{};
    cl::Platform::get(&platforms);
    auto platform = platforms[0];

    auto devices = std::vector<cl::Device>{};
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices); 
    auto device = devices[0];
    
    auto context = cl::Context(devices);

    auto program = create_program(context, "saxpy.cl");
    program.build(devices);

    auto kernel = cl::Kernel{program, "saxpy"};

    auto start = std::chrono::steady_clock::now();
    auto buf_x = cl::Buffer{context, CL_MEM_READ_ONLY, N * sizeof(float)};
    auto buf_y = cl::Buffer{context, CL_MEM_READ_WRITE, N * sizeof(float)};

    auto queue = cl::CommandQueue{context, device};

    cl::copy(queue, x, x + N, buf_x);
    cl::copy(queue, y, y + N, buf_y);

    kernel.setArg(0, buf_x);
    kernel.setArg(1, buf_y);
    kernel.setArg(2, a);

    auto global= cl::NDRange(N);
    auto local = cl::NDRange(256);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    
    cl::copy(queue, buf_y, y, y + N);

    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>{stop - start};
}

auto main() -> int
{
    float x[N];
    float y[N];

    // initialize input data
    auto random_gen = std::default_random_engine{};
    auto distribution = std::uniform_real_distribution<float>{-N, N};
    
    std::generate_n(x, N, [&]() { return distribution(random_gen); });
    std::generate_n(y, N, [&]() { return distribution(random_gen); });

    // make a copy for use on GPU
    float y_gpu[N];
    std::copy_n(y, N, y_gpu);

    auto dur_cpu = algo_cpu(x, y);
    auto dur_gpu = algo_gpu(x, y_gpu);

    // verify
    auto errors = 0;
    for(auto i = 0; i < N; ++i) {
        if(std::abs(y[i] - y_gpu[i]) > std::abs(y[i] * 0.0001f))
            ++errors;
    }

    std::cout << errors << " errors" << std::endl;
    std::cout << "CPU version took " << dur_cpu.count() << " s" << std::endl;
    std::cout << "GPU version took " << dur_gpu.count() << " s" << std::endl;

    return 0;
}
