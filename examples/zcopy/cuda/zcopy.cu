#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <limits>

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

constexpr auto elems = 1 << 25;
constexpr auto iters = 10;

__global__ void read_write(const double2* __restrict__ A,
                                 double2* __restrict__ B)
{
    auto stride = gridDim.x * blockDim.x;
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += stride)
    {
        B[i] = A[i];
    } 
}

__global__ void write(double2* __restrict__ B)
{
    auto stride = gridDim.x * blockDim.x;
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < elems; i += stride)
    {
        B[i] = make_double2(0.0, 0.0);
    }
}

auto main() -> int
{
    // set up devices
    auto dev_count = int{};
    CHECK(cudaGetDeviceCount(&dev_count));

    std::cout << "Available accelerators: " << std::endl;
    for(auto i = 0u; i < dev_count; ++i)
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

    // Allocate memory on device
    auto A_d = static_cast<double2*>(nullptr);
    auto B_d = static_cast<double2*>(nullptr);

    CHECK(cudaMalloc(&A_d, sizeof(double2) * elems));
    CHECK(cudaMalloc(&B_d, sizeof(double2) * elems));

    // Initialize device memory
    CHECK(cudaMemset(A_d, 0x00, sizeof(double2) * elems)); // zero
    CHECK(cudaMemset(B_d, 0xff, sizeof(double2) * elems)); // NaN

    constexpr auto block_size = 128;
    constexpr auto num_blocks = (elems + (block_size - 1)) / block_size;
    constexpr auto blocks = num_blocks > 65520 ? 65520 : num_blocks;

    std::cout << "zcopy: operating on vectors of " << elems << " double2s"
              << " = " << sizeof(double2) * elems << " bytes" << std::endl;

    std::cout << "zcopy: using " << block_size << " threads per block, "
              << blocks << " blocks" << std::endl;

    auto mintime = std::numeric_limits<float>::max();
    for(auto k = 0; k < iters; ++k)
    {
        auto start_event = cudaEvent_t{};
        auto stop_event = cudaEvent_t{};
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventCreate(&start_event));
        CHECK(cudaEventCreate(&stop_event));

        CHECK(cudaEventRecord(start_event, 0));
        read_write<<<blocks, block_size>>>(A_d, B_d);
        CHECK(cudaEventRecord(stop_event, 0));

        CHECK(cudaEventSynchronize(stop_event));

        auto elapsed = float{};
        CHECK(cudaGetLastError());
        CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));

        mintime = std::min(mintime, elapsed);
    }

    std::cout << "RW: mintime = " << mintime << " msec  "
              << "throughput = "
              << (2.0e-9 * sizeof(double2) * elems) / (mintime / 1e3)
              << " GB/sec" << std::endl;

    for(auto k = 0; k < iters; ++k)
    {
        auto start_event = cudaEvent_t{};
        auto stop_event = cudaEvent_t{};
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventCreate(&start_event));
        CHECK(cudaEventCreate(&stop_event));

        CHECK(cudaEventRecord(start_event, 0));
        write<<<blocks, block_size>>>(B_d);
        CHECK(cudaEventRecord(stop_event, 0));

        CHECK(cudaEventSynchronize(stop_event));

        auto elapsed = float{};
        CHECK(cudaGetLastError());
        CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));

        mintime = std::min(mintime, elapsed);
    }

    std::cout << "WO: mintime = " << mintime << " msec  "
              << "throughput = "
              << (1.0e-9 * sizeof(double2) * elems) / (mintime / 1e3)
              << " GB/sec" << std::endl;

    return EXIT_SUCCESS;
}
