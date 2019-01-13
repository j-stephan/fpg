#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <limits>

#include <hip/hip_runtime.h>

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

constexpr auto elems = 1 << 28;
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
    CHECK(hipGetDeviceCount(&dev_count));

    std::cout << "Available accelerators: " << std::endl;
    for(auto i = 0u; i < dev_count; ++i)
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

    // Allocate memory on device
    auto A_d = static_cast<double2*>(nullptr);
    auto B_d = static_cast<double2*>(nullptr);

    CHECK(hipMalloc(&A_d, sizeof(double2) * elems));
    CHECK(hipMalloc(&B_d, sizeof(double2) * elems));

    // Initialize device memory
    CHECK(hipMemset(A_d, 0x00, sizeof(double2) * elems)); // zero
    CHECK(hipMemset(B_d, 0xff, sizeof(double2) * elems)); // NaN

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
        auto start_event = hipEvent_t{};
        auto stop_event = hipEvent_t{};
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventCreate(&start_event));
        CHECK(hipEventCreate(&stop_event));

        CHECK(hipEventRecord(start_event, 0));
        hipLaunchKernelGGL(read_write, dim3(blocks), dim3(block_size), 0, 0,
                           A_d, B_d);
        CHECK(hipEventRecord(stop_event, 0));

        CHECK(hipEventSynchronize(stop_event));

        auto elapsed = float{};
        CHECK(hipGetLastError());
        CHECK(hipEventElapsedTime(&elapsed, start_event, stop_event));

        mintime = std::min(mintime, elapsed);
    }

    std::cout << "RW: mintime = " << mintime << " msec  "
              << "throughput = "
              << (2.0e-9 * sizeof(double2) * elems) / (mintime / 1e3)
              << " GB/sec" << std::endl;

    for(auto k = 0; k < iters; ++k)
    {
        auto start_event = hipEvent_t{};
        auto stop_event = hipEvent_t{};
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventCreate(&start_event));
        CHECK(hipEventCreate(&stop_event));

        CHECK(hipEventRecord(start_event, 0));
        hipLaunchKernelGGL(write, dim3(blocks), dim3(block_size), 0, 0, B_d);
        CHECK(hipEventRecord(stop_event, 0));

        CHECK(hipEventSynchronize(stop_event));

        auto elapsed = float{};
        CHECK(hipGetLastError());
        CHECK(hipEventElapsedTime(&elapsed, start_event, stop_event));

        mintime = std::min(mintime, elapsed);
    }

    std::cout << "WO: mintime = " << mintime << " msec  "
              << "throughput = "
              << (2.0e-9 * sizeof(double2) * elems) / (mintime / 1e3)
              << " GB/sec" << std::endl;

    return EXIT_SUCCESS;
}
