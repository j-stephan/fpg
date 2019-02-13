#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#define CHECK(cmd) \
{ \
    auto error = cmd; \
    if(error != cudaSuccess) \
    { \
        std::cerr << "Error: '" << cudaGetErrorString(error) \
                  << "' (" << error << ") at " << __FILE__ << ":" \
                  << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} \

__global__ void read_write(const float4* __restrict__ A,
                                 float4* __restrict__ B,
                           std::size_t elems)
{
    auto stride = gridDim.x * blockDim.x;
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += stride)
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

            CHECK(cudaMalloc(&A_d, sizeof(float4) * elems));
            CHECK(cudaMalloc(&B_d, sizeof(float4) * elems));

            for(auto block_num = sms;
                     block_num <= std::min(elems / block_size, max_blocks);
                     block_num *= 2)
            {
                // Initialize device memory
                CHECK(cudaMemset(A_d, 0x00, sizeof(float4) * elems)); // zero
                CHECK(cudaMemset(B_d, 0xff, sizeof(float4) * elems)); // NaN

                auto mintime = std::numeric_limits<float>::max();
                for(auto k = 0; k < iters; ++k)
                {
                    auto start_event = cudaEvent_t{};
                    auto stop_event = cudaEvent_t{};
                    CHECK(cudaDeviceSynchronize());
                    CHECK(cudaEventCreate(&start_event));
                    CHECK(cudaEventCreate(&stop_event));

                    CHECK(cudaEventRecord(start_event, 0));
                    read_write<<<block_num, block_size>>>(A_d, B_d, elems);
                    CHECK(cudaEventRecord(stop_event, 0));

                    CHECK(cudaEventSynchronize(stop_event));

                    auto elapsed = float{};
                    CHECK(cudaGetLastError());
                    CHECK(cudaEventElapsedTime(&elapsed,
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
                CHECK(cudaMemset(B_d, 0xff, sizeof(float4) * elems)); // NaN

                auto mintime = std::numeric_limits<float>::max();
                for(auto k = 0; k < iters; ++k)
                {
                    auto start_event = cudaEvent_t{};
                    auto stop_event = cudaEvent_t{};
                    CHECK(cudaDeviceSynchronize());
                    CHECK(cudaEventCreate(&start_event));
                    CHECK(cudaEventCreate(&stop_event));

                    CHECK(cudaEventRecord(start_event, 0));
                    write<<<block_num, block_size>>>(B_d, elems);
                    CHECK(cudaEventRecord(stop_event, 0));

                    CHECK(cudaEventSynchronize(stop_event));

                    auto elapsed = float{};
                    CHECK(cudaGetLastError());
                    CHECK(cudaEventElapsedTime(&elapsed, start_event,
                                               stop_event));

                    mintime = std::min(mintime, elapsed);
                }

                file << "WO;" << block_size << ";" << block_num << ";"
                     << sizeof(float4) << ";" << elems << ";"
                     << mintime << ";"
                     << (1.0e-9 * sizeof(float4) * elems) / (mintime / 1e3)
                     << std::endl;
            }

            file << std::endl;

            CHECK(cudaFree(B_d));
            CHECK(cudaFree(A_d));
        }
    }
}

auto main() -> int
{
    // set up devices
    auto dev_count = int{};
    CHECK(cudaGetDeviceCount(&dev_count));

    std::cout << "Available accelerators: " << std::endl;
    for(auto i = 0; i < dev_count; ++i)
    {
        auto prop = cudaDeviceProp{};
        CHECK(cudaGetDeviceProperties(&prop, i));

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

    CHECK(cudaSetDevice(index));
    CHECK(cudaFree(nullptr));

    auto prop = cudaDeviceProp{};
    CHECK(cudaGetDeviceProperties(&prop, index));
    
    auto sms = prop.multiProcessorCount;
    auto max_blocks = prop.maxGridSize[0];

    auto now = std::chrono::system_clock::now();
    auto cnow = std::chrono::system_clock::to_time_t(now);

    auto filename = std::stringstream{};
    filename << "CUDA-";
    filename << std::put_time(std::localtime(&cnow), "%Y-%m-%d-%X");
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
