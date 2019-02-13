#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip/hip_runtime.h>
#pragma clang diagnostic pop

#define CHECK(cmd) \
{ \
    auto error = cmd; \
    if(error != hipSuccess) \
    { \
        std::cerr << "Error: '" << hipGetErrorString(error) \
                  << "' (" << error << ") at " << __FILE__ << ":" \
                  << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} \

__global__ void read_write(const float4* __restrict__ A,
                                 float4* __restrict__ B,
                           std::size_t elems)
{
    auto stride = hipGridDim_x * hipBlockDim_x;
    for(auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
             i < elems;
             i += stride)
    {
        B[i] = A[i];
    } 
}

__global__ void write(float4* __restrict__ B, std::size_t elems)
{
    auto stride = gridDim.x * blockDim.x;
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += stride)
    {
        B[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

auto do_benchmark(int sms, int max_blocks, std::ofstream& file,
                  int start_size, int stop_size) -> void
{
    std::cout << "Benchmarking size " << sizeof(float4) << std::endl;

    constexpr auto iters = 10;
    constexpr auto max_mem = 1u << 31; // mem per vector
    constexpr auto max_elems = static_cast<int>(max_mem / sizeof(float4));

    for(auto block_size = start_size; block_size <= stop_size; block_size *= 2)
    {
        for(auto elems = block_size * sms; elems <= max_elems; elems *= 2)
        {
            // Allocate memory on device
            auto A_d = static_cast<float4*>(nullptr);
            auto B_d = static_cast<float4*>(nullptr);

            CHECK(hipMalloc(&A_d, sizeof(float4) * elems));
            CHECK(hipMalloc(&B_d, sizeof(float4) * elems));

            for(auto block_num = sms;
                     block_num <= std::min(elems / block_size, max_blocks);
                     block_num *= 2)
            {
                // Initialize device memory
                CHECK(hipMemset(A_d, 0x00, sizeof(float4) * elems)); // zero
                CHECK(hipMemset(B_d, 0xff, sizeof(float4) * elems)); // NaN

                auto mintime = std::numeric_limits<float>::max();
                for(auto k = 0; k < iters; ++k)
                {
                    auto start_event = hipEvent_t{};
                    auto stop_event = hipEvent_t{};
                    CHECK(hipDeviceSynchronize());
                    CHECK(hipEventCreate(&start_event));
                    CHECK(hipEventCreate(&stop_event));

                    CHECK(hipEventRecord(start_event, 0));
                    hipLaunchKernelGGL(read_write,
                                       dim3(block_num), dim3(block_size), 0, 0,
                                       A_d, B_d, elems);
                    CHECK(hipEventRecord(stop_event, 0));

                    CHECK(hipEventSynchronize(stop_event));

                    auto elapsed = float{};
                    CHECK(hipGetLastError());
                    CHECK(hipEventElapsedTime(&elapsed,
                                              start_event, stop_event));

                    mintime = std::min(mintime, elapsed);
                }

                file << "RW;" << block_size << ";" << block_num << ";"
                     << sizeof(float4) << ";" << elems << ";"
                     << mintime << ";"
                     << (2.0e-9 * sizeof(float4) * elems) / (mintime / 1e3)
                     << std::endl;
            }

            file << std::endl;

            for(auto block_num = sms;
                     block_num <= std::min(elems / block_size, max_blocks);
                     block_num *= 2)
            {
                // Initialize device memory
                CHECK(hipMemset(B_d, 0xff, sizeof(float4) * elems)); // NaN

                auto mintime = std::numeric_limits<float>::max();
                for(auto k = 0; k < iters; ++k)
                {
                    auto start_event = hipEvent_t{};
                    auto stop_event = hipEvent_t{};
                    CHECK(hipDeviceSynchronize());
                    CHECK(hipEventCreate(&start_event));
                    CHECK(hipEventCreate(&stop_event));

                    CHECK(hipEventRecord(start_event, 0));
                    hipLaunchKernelGGL(write,
                                       dim3(block_num), dim3(block_size), 0, 0,
                                       B_d, elems);
                    CHECK(hipEventRecord(stop_event, 0));

                    CHECK(hipEventSynchronize(stop_event));

                    auto elapsed = float{};
                    CHECK(hipGetLastError());
                    CHECK(hipEventElapsedTime(&elapsed, start_event, stop_event));

                    mintime = std::min(mintime, elapsed);
                }

                file << "WO;" << block_size << ";" << block_num << ";"
                     << sizeof(float4) << ";" << elems << ";"
                     << mintime << ";"
                     << (1.0e-9 * sizeof(float4) * elems) / (mintime / 1e3)
                     << std::endl;
            }

            file << std::endl;

            CHECK(hipFree(B_d));
            CHECK(hipFree(A_d));
        }
    }
}

auto main() -> int
{
    // set up devices
    auto dev_count = int{};
    CHECK(hipGetDeviceCount(&dev_count));

    std::cout << "Available accelerators: " << std::endl;
    for(auto i = 0; i < dev_count; ++i)
    {
        auto prop = hipDeviceProp_t{};
        CHECK(hipGetDeviceProperties(&prop, i));

        std::cout << "\t[" << i << "] " << prop.name << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Select accelerator: ";
    auto index = 0;
    std::cin >> index;

    if(index >= dev_count)
    {
        std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                  << std::endl;
        return EXIT_FAILURE;
    }

    CHECK(hipSetDevice(index));
    CHECK(hipFree(nullptr));

    auto prop = hipDeviceProp_t{};
    CHECK(hipGetDeviceProperties(&prop, index));
    
    auto sms = prop.multiProcessorCount;
    auto max_blocks = prop.maxGridSize[0];

    auto now = std::chrono::system_clock::now();
    auto cnow = std::chrono::system_clock::to_time_t(now);

    auto filename = std::stringstream{};
    filename << std::put_time(std::localtime(&cnow), "%Y-%m-%d %X");
    filename << ".csv";

    auto file = std::ofstream{filename.str()};

    file << "type;block_size;block_num;elem_size;elem_num;mintime;throughput"
         << std::endl;

    do_benchmark(sms, max_blocks, file, 64, 1024);
#ifdef KEPLER
    do_benchmark(sms, max_blocks, file, 192, 768);
#endif

    return EXIT_SUCCESS;
}
