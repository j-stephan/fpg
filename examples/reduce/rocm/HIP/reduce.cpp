// based on Justin Luitjen's (NVIDIA) "Faster Parallel Reductions on Kepler"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#define CHECK(cmd) \
{ \
    hipError_t error = cmd; \
    if(error != hipSuccess) \
    { \
        std::cerr << "Error: '" << hipGetErrorString(error) \
                  << "' (" << error << ") at " << __FILE__ << ":" << __LINE__ \
                  << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

__inline__ __device__ auto warp_reduce_sum(int val)
{
    for(auto offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

__inline__ __device__ auto block_reduce_sum(int val)
{
    constexpr auto shared_size = 1024 / warpSize;
    // shared memory for 16 / 32 partial sums
    static __shared__ int shared[shared_size];
    auto lane_id = hipThreadIdx_x % warpSize;
    auto warp_id = hipThreadIdx_x / warpSize;

    // each warp performs partial reduction
    val = warp_reduce_sum(val);

    // write reduced value to shared memory
    if(lane_id == 0)
        shared[warp_id] = val;

    // wait for all partial reductions
    __syncthreads();

    // read from shared memory only if that warp existed
    val = (hipThreadIdx_x < hipBlockDim_x / warpSize) ? shared[lane_id] : 0;

    // final reduce within first warp
    if(warp_id == 0)
        val = warp_reduce_sum(val);

    return val;
}

// ============================================================================
// Stable
// ============================================================================

__global__ void device_reduce_stable(const int* data, int* result, int dim)
{
    auto sum = 0;
    // reduce multiple elements per thread
    for(auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
             i < dim;
             i += hipBlockDim_x * hipGridDim_x)
    {
        sum += data[i];
    }
    sum = block_reduce_sum(sum);
    
    if(hipThreadIdx_x == 0)
        result[hipBlockIdx_x] = sum;
}

auto reduce_stable(const int* data, int* result, int dim, int threads, int blocks)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(device_reduce_stable),
                       dim3(blocks), dim3(threads), 0, 0,
                       data, result, dim);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(device_reduce_stable),
                       dim3(1), dim3(1024), 0, 0,
                       result, result, dim);
}

// ============================================================================
// Host
// ============================================================================
template <int array_dim, typename Func>
auto do_reduce(const std::string& label, Func f)
{
    static_assert(array_dim % 64 == 0, "array_dim % 64 != 0");

    constexpr auto iterations = 5;
    constexpr auto threads = 512;
    constexpr auto blocks = std::min((array_dim + threads - 1) / threads, 1024);
    
    // create data
    auto data = std::vector<int>{};
    data.resize(array_dim);

    // initialize data
    auto rd = std::random_device{};
    auto rng = std::mt19937{rd()};
    auto uid = std::uniform_int_distribution{0, 3};
    std::generate(std::begin(data), std::end(data), [&]() { return uid(rng); });

    auto result = std::vector<int>{};
    constexpr auto result_dim = blocks * iterations + blocks;
    result.resize(result_dim);
    std::fill(std::begin(result), std::end(result), 0);

    // copy to GPU
    auto data_gpu = static_cast<int*>(nullptr);
    auto result_gpu = static_cast<int*>(nullptr);

    constexpr auto data_bytes = array_dim * sizeof(int);
    constexpr auto result_bytes = result_dim * sizeof(int);

    CHECK(hipMalloc(&data_gpu, data_bytes));
    CHECK(hipMalloc(&result_gpu, result_bytes));

    CHECK(hipMemcpy(data_gpu, data.data(), data_bytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(result_gpu, result.data(), result_bytes, hipMemcpyHostToDevice));

    // warm up
    auto offset = blocks * iterations;
    f(data_gpu, result_gpu + offset, array_dim, threads, blocks);
    CHECK(hipDeviceSynchronize());

    // reduce
    auto start = std::chrono::steady_clock::now();
    for(auto i = 0; i < iterations; ++i)
    {
        offset = i * blocks;
        f(data_gpu, result_gpu + offset, array_dim, threads, blocks);
    }
    CHECK(hipDeviceSynchronize());
    auto stop = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration<double>{stop - start};

    CHECK(hipMemcpy(result.data(), result_gpu, result_bytes, hipMemcpyDeviceToHost));

    // verify
    auto verify = std::accumulate(std::begin(data), std::end(data), 0);
    if(verify != result[0])
    {
        std::cerr << "Mismatch: expected " << verify
                  << ", got " << result[0] << std::endl;
    }

    auto bandwidth = data_bytes * iterations / dur.count();

    std::cout << label << ";" << array_dim << ";" << bandwidth << std::endl;

    CHECK(hipFree(result_gpu));
    CHECK(hipFree(data_gpu));
}

auto main() -> int
{
    std::cout << "type;array_dim;bandwidth" << std::endl;
    do_reduce<1024>("stable", reduce_stable);
    do_reduce<2048>("stable", reduce_stable);
    do_reduce<4096>("stable", reduce_stable);
    do_reduce<8192>("stable", reduce_stable);
    do_reduce<16384>("stable", reduce_stable);
    do_reduce<32768>("stable", reduce_stable);
    do_reduce<65536>("stable", reduce_stable);
    do_reduce<131072>("stable", reduce_stable);
    do_reduce<262144>("stable", reduce_stable);
    do_reduce<524288>("stable", reduce_stable);
    do_reduce<1048576>("stable", reduce_stable);
    do_reduce<2097152>("stable", reduce_stable);
    do_reduce<4194304>("stable", reduce_stable);
    do_reduce<8388608>("stable", reduce_stable);
    do_reduce<16777216>("stable", reduce_stable);
    do_reduce<33554432>("stable", reduce_stable);
    do_reduce<67108864>("stable", reduce_stable);
    do_reduce<134217728>("stable", reduce_stable);

    return 1;

}

// vi: set syntax=hip
