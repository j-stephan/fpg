/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

// based on Justin Luitjen's (NVIDIA) "Faster Parallel Reductions on Kepler"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#define CHECK(cmd) \
{ \
    auto error = cmd; \
    if(error != cudaSuccess) \
    { \
        std::cerr << "Error: '" << cudaGetErrorString(error) \
                  << "' (" << error << ") at " << __FILE__ << ":" << __LINE__ \
                  << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

__inline__ __device__ auto warp_reduce_sum(int val)
{
    for(auto offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ auto block_reduce_sum(int val)
{
    // this is really dumb - CUDA's builtin warpSize is not constexpr
    constexpr auto warp_size = 32;
    constexpr auto shared_size = 1024 / warp_size;
    // shared memory for 16 / 32 partial sums
    static __shared__ int shared[shared_size];
    auto lane_id = threadIdx.x % warpSize;
    auto warp_id = threadIdx.x / warpSize;

    // each warp performs partial reduction
    val = warp_reduce_sum(val);

    // write reduced value to shared memory
    if(lane_id == 0)
        shared[warp_id] = val;

    // wait for all partial reductions
    __syncthreads();

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane_id] : 0;

    // final reduce within first warp
    if(warp_id == 0)
        val = warp_reduce_sum(val);

    return val;
}

// =============================================================================
// Stable
// =============================================================================

__global__ void device_reduce_stable(const int* data, int* result, int dim)
{
    auto sum = 0;

    // reduce multiple elements per thread
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x;
             i < dim;
             i += blockDim.x * gridDim.x)
    {
        sum += data[i];
    }
    sum = block_reduce_sum(sum);
    
    if(threadIdx.x == 0)
        result[blockIdx.x] = sum;
}

auto reduce_stable(const int* data, int* result, int dim, int blocks,
                   int threads)
{
    device_reduce_stable<<<blocks, threads>>>(data, result, dim);
    device_reduce_stable<<<1, 1024>>>(result, result, blocks);
}

// =============================================================================
// Stable - vec2
// =============================================================================

__global__ void device_reduce_stable_vec2(const int* data, int* result, int dim)
{
    auto sum = 0;

    // reduce multiple elements per thread
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x;
             i < dim / 2;
             i += blockDim.x * gridDim.x)
    {
        auto val = reinterpret_cast<const int2*>(data)[i];
        sum += val.x + val.y;
    }
    sum = block_reduce_sum(sum);
    
    if(threadIdx.x == 0)
        result[blockIdx.x] = sum;
}

auto reduce_stable_vec2(const int* data, int* result, int dim, int blocks,
                   int threads)
{
    blocks = std::min(((dim / 2) + threads - 1) / threads, 1024);

    device_reduce_stable_vec2<<<blocks, threads>>>(data, result, dim);
    device_reduce_stable<<<1, 1024>>>(result, result, blocks);
}

// =============================================================================
// Stable - vec4
// =============================================================================

__global__ void device_reduce_stable_vec4(const int* data, int* result, int dim)
{
    auto sum = 0;

    // reduce multiple elements per thread
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x;
             i < dim / 4;
             i += blockDim.x * gridDim.x)
    {
        auto val = reinterpret_cast<const int4*>(data)[i];
        sum += (val.x + val.y) + (val.z + val.w);
    }
    sum = block_reduce_sum(sum);
    
    if(threadIdx.x == 0)
        result[blockIdx.x] = sum;
}

auto reduce_stable_vec4(const int* data, int* result, int dim, int blocks,
                   int threads)
{
    blocks = std::min(((dim / 4) + threads - 1) / threads, 1024);

    device_reduce_stable_vec4<<<blocks, threads>>>(data, result, dim);
    device_reduce_stable<<<1, 1024>>>(result, result, blocks);
}

// =============================================================================
// Warp + Atomic
// =============================================================================

__global__ void device_reduce_warp_atomic(const int* data, int* result,
                                          int dim)
{
    auto sum = 0;

    // reduce multiple elements per thread
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x;
             i < dim;
             i += blockDim.x * gridDim.x)
    {
        sum += data[i];
    }
    sum = warp_reduce_sum(sum);

    if((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(result, sum);
}

inline auto reduce_warp_atomic(const int* data, int* result, int dim,
                               int blocks, int threads)
{
    threads = 256;
    blocks = std::min((dim + threads - 1) / threads, 2048);

    CHECK(cudaMemsetAsync(result, 0, sizeof(int)));
    device_reduce_warp_atomic<<<blocks, threads>>>(data, result, dim);
}

// ============================================================================
// Warp + Atomic - vec2
// ============================================================================

__global__ void device_reduce_warp_atomic_vec2(const int* data, int* result,
                                               int dim)
{
    auto sum = 0;

    // reduce multiple elements per thread
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x;
             i < dim / 2;
             i += blockDim.x * gridDim.x)
    {
        auto val = reinterpret_cast<const int2*>(data)[i];
        sum += val.x + val.y;
    }
    sum = warp_reduce_sum(sum);

    if((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(result, sum);
}

inline auto reduce_warp_atomic_vec2(const int* data, int* result, int dim,
                               int blocks, int threads)
{
    threads = 256;
    blocks = std::min(((dim / 2) + threads - 1) / threads, 2048);

    CHECK(cudaMemsetAsync(result, 0, sizeof(int)));
    device_reduce_warp_atomic_vec2<<<blocks, threads>>>(data, result, dim);
}

// ============================================================================
// Warp + Atomic - vec4
// ============================================================================

__global__ void device_reduce_warp_atomic_vec4(const int* data, int* result,
                                               int dim)
{
    auto sum = 0;

    // reduce multiple elements per thread
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x;
             i < dim / 4;
             i += blockDim.x * gridDim.x)
    {
        auto val = reinterpret_cast<const int4*>(data)[i];
        sum += (val.x + val.y) + (val.z + val.w);
    }
    sum = warp_reduce_sum(sum);

    if((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(result, sum);
}

inline auto reduce_warp_atomic_vec4(const int* data, int* result, int dim,
                               int blocks, int threads)
{
    threads = 256;
    blocks = std::min(((dim / 4) + threads - 1) / threads, 2048);

    CHECK(cudaMemsetAsync(result, 0, sizeof(int)));
    device_reduce_warp_atomic_vec4<<<blocks, threads>>>(data, result, dim);
}

// ============================================================================
// Block + Atomic
// ============================================================================

__global__ void device_reduce_block_atomic(const int* data, int* result,
                                           int dim)
{
    auto sum = 0;
    // reduce multiple elements per thread
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x;
             i < dim;
             i += blockDim.x * gridDim.x)
    {
        sum += data[i];
    }
    sum = block_reduce_sum(sum);

    if(threadIdx.x == 0)
        atomicAdd(result, sum);
}

inline auto reduce_block_atomic(const int* data, int* result, int dim,
                                int blocks, int threads)
{
    threads = 256;
    blocks = std::min((dim + threads - 1) / threads, 2048);

    CHECK(cudaMemsetAsync(result, 0, sizeof(int)));
    device_reduce_block_atomic<<<blocks, threads>>>(data, result, dim);
}

// ============================================================================
// Block + Atomic - vec2
// ============================================================================

__global__ void device_reduce_block_atomic_vec2(const int* data, int* result,
                                           int dim)
{
    auto sum = 0;

    // reduce multiple elements per thread
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x;
             i < dim / 2;
             i += blockDim.x * gridDim.x)
    {
        auto val = reinterpret_cast<const int2*>(data)[i];
        sum += val.x + val.y;
    }
    sum = block_reduce_sum(sum);

    if(threadIdx.x == 0)
        atomicAdd(result, sum);
}

inline auto reduce_block_atomic_vec2(const int* data, int* result, int dim,
                                int blocks, int threads)
{
    threads = 256;
    blocks = std::min(((dim / 2) + threads - 1) / threads, 2048);

    CHECK(cudaMemsetAsync(result, 0, sizeof(int)));
    device_reduce_block_atomic_vec2<<<blocks, threads>>>(data, result, dim);
}

// ============================================================================
// Block + Atomic - vec4
// ============================================================================

__global__ void device_reduce_block_atomic_vec4(const int* data, int* result,
                                           int dim)
{
    auto sum = 0;

    // reduce multiple elements per thread
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x;
             i < dim / 4;
             i += blockDim.x * gridDim.x)
    {
        auto val = reinterpret_cast<const int4*>(data)[i];
        sum += (val.x + val.y) + (val.z + val.w);
    }
    sum = block_reduce_sum(sum);

    if(threadIdx.x == 0)
        atomicAdd(result, sum);
}

inline auto reduce_block_atomic_vec4(const int* data, int* result, int dim,
                                int blocks, int threads)
{
    threads = 256;
    blocks = std::min(((dim / 4) + threads - 1) / threads, 2048);

    CHECK(cudaMemsetAsync(result, 0, sizeof(int)));
    device_reduce_block_atomic_vec4<<<blocks, threads>>>(data, result, dim);
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
    auto&& rd = std::random_device{};
    auto rng = std::mt19937{rd()};
    auto uid = std::uniform_int_distribution<>(0, 3);
    std::generate(std::begin(data), std::end(data), [&]() { return uid(rng); });

    auto result = std::vector<int>{};
    result.resize(blocks); // only stable needs multiple results
    std::fill(std::begin(result), std::end(result), 0);

    // copy to GPU
    auto data_gpu = static_cast<int*>(nullptr);
    auto result_gpu = static_cast<int*>(nullptr);

    constexpr auto data_bytes = array_dim * sizeof(int);
    constexpr auto result_bytes = blocks * sizeof(int);

    CHECK(cudaMalloc(&data_gpu, data_bytes));
    CHECK(cudaMalloc(&result_gpu, result_bytes));

    CHECK(cudaMemcpy(data_gpu, data.data(),
                     data_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(result_gpu, result.data(),
                     result_bytes, cudaMemcpyHostToDevice));

    // warm up
    f(data_gpu, result_gpu, array_dim, blocks, threads);
    CHECK(cudaDeviceSynchronize());

    // reduce
    auto start = std::chrono::steady_clock::now();
    for(auto i = 0; i < iterations; ++i)
    {
        f(data_gpu, result_gpu, array_dim, blocks, threads);
    }
    CHECK(cudaDeviceSynchronize());
    auto stop = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration<double>{stop - start};

    CHECK(cudaMemcpy(result.data(), result_gpu,
                     result_bytes, cudaMemcpyDeviceToHost));

    // verify
    auto verify = std::accumulate(std::begin(data), std::end(data), 0);
    if(verify != result[0])
    {
        std::cerr << "Mismatch: expected " << verify
                  << ", got " << result[0] << std::endl;
    }

    auto bandwidth = data_bytes * iterations / dur.count();

    std::cout << label << ";" << array_dim << ";" << bandwidth << std::endl;

    CHECK(cudaFree(result_gpu));
    CHECK(cudaFree(data_gpu));
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

    do_reduce<1024>("stable_vec2", reduce_stable_vec2);
    do_reduce<2048>("stable_vec2", reduce_stable_vec2);
    do_reduce<4096>("stable_vec2", reduce_stable_vec2);
    do_reduce<8192>("stable_vec2", reduce_stable_vec2);
    do_reduce<16384>("stable_vec2", reduce_stable_vec2);
    do_reduce<32768>("stable_vec2", reduce_stable_vec2);
    do_reduce<65536>("stable_vec2", reduce_stable_vec2);
    do_reduce<131072>("stable_vec2", reduce_stable_vec2);
    do_reduce<262144>("stable_vec2", reduce_stable_vec2);
    do_reduce<524288>("stable_vec2", reduce_stable_vec2);
    do_reduce<1048576>("stable_vec2", reduce_stable_vec2);
    do_reduce<2097152>("stable_vec2", reduce_stable_vec2);
    do_reduce<4194304>("stable_vec2", reduce_stable_vec2);
    do_reduce<8388608>("stable_vec2", reduce_stable_vec2);
    do_reduce<16777216>("stable_vec2", reduce_stable_vec2);
    do_reduce<33554432>("stable_vec2", reduce_stable_vec2);
    do_reduce<67108864>("stable_vec2", reduce_stable_vec2);
    do_reduce<134217728>("stable_vec2", reduce_stable_vec2);

    do_reduce<1024>("stable_vec4", reduce_stable_vec4);
    do_reduce<2048>("stable_vec4", reduce_stable_vec4);
    do_reduce<4096>("stable_vec4", reduce_stable_vec4);
    do_reduce<8192>("stable_vec4", reduce_stable_vec4);
    do_reduce<16384>("stable_vec4", reduce_stable_vec4);
    do_reduce<32768>("stable_vec4", reduce_stable_vec4);
    do_reduce<65536>("stable_vec4", reduce_stable_vec4);
    do_reduce<131072>("stable_vec4", reduce_stable_vec4);
    do_reduce<262144>("stable_vec4", reduce_stable_vec4);
    do_reduce<524288>("stable_vec4", reduce_stable_vec4);
    do_reduce<1048576>("stable_vec4", reduce_stable_vec4);
    do_reduce<2097152>("stable_vec4", reduce_stable_vec4);
    do_reduce<4194304>("stable_vec4", reduce_stable_vec4);
    do_reduce<8388608>("stable_vec4", reduce_stable_vec4);
    do_reduce<16777216>("stable_vec4", reduce_stable_vec4);
    do_reduce<33554432>("stable_vec4", reduce_stable_vec4);
    do_reduce<67108864>("stable_vec4", reduce_stable_vec4);
    do_reduce<134217728>("stable_vec4", reduce_stable_vec4);

    do_reduce<1024>("warp_atomic", reduce_warp_atomic);
    do_reduce<2048>("warp_atomic", reduce_warp_atomic);
    do_reduce<4096>("warp_atomic", reduce_warp_atomic);
    do_reduce<8192>("warp_atomic", reduce_warp_atomic);
    do_reduce<16384>("warp_atomic", reduce_warp_atomic);
    do_reduce<32768>("warp_atomic", reduce_warp_atomic);
    do_reduce<65536>("warp_atomic", reduce_warp_atomic);
    do_reduce<131072>("warp_atomic", reduce_warp_atomic);
    do_reduce<262144>("warp_atomic", reduce_warp_atomic);
    do_reduce<524288>("warp_atomic", reduce_warp_atomic);
    do_reduce<1048576>("warp_atomic", reduce_warp_atomic);
    do_reduce<2097152>("warp_atomic", reduce_warp_atomic);
    do_reduce<4194304>("warp_atomic", reduce_warp_atomic);
    do_reduce<8388608>("warp_atomic", reduce_warp_atomic);
    do_reduce<16777216>("warp_atomic", reduce_warp_atomic);
    do_reduce<33554432>("warp_atomic", reduce_warp_atomic);
    do_reduce<67108864>("warp_atomic", reduce_warp_atomic);
    do_reduce<134217728>("warp_atomic", reduce_warp_atomic);

    do_reduce<1024>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<2048>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<4096>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<8192>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<16384>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<32768>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<65536>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<131072>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<262144>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<524288>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<1048576>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<2097152>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<4194304>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<8388608>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<16777216>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<33554432>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<67108864>("warp_atomic_vec2", reduce_warp_atomic_vec2);
    do_reduce<134217728>("warp_atomic_vec2", reduce_warp_atomic_vec2);

    do_reduce<1024>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<2048>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<4096>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<8192>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<16384>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<32768>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<65536>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<131072>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<262144>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<524288>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<1048576>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<2097152>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<4194304>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<8388608>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<16777216>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<33554432>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<67108864>("warp_atomic_vec4", reduce_warp_atomic_vec4);
    do_reduce<134217728>("warp_atomic_vec4", reduce_warp_atomic_vec4);

    do_reduce<1024>("block_atomic", reduce_block_atomic);
    do_reduce<2048>("block_atomic", reduce_block_atomic);
    do_reduce<4096>("block_atomic", reduce_block_atomic);
    do_reduce<8192>("block_atomic", reduce_block_atomic);
    do_reduce<16384>("block_atomic", reduce_block_atomic);
    do_reduce<32768>("block_atomic", reduce_block_atomic);
    do_reduce<65536>("block_atomic", reduce_block_atomic);
    do_reduce<131072>("block_atomic", reduce_block_atomic);
    do_reduce<262144>("block_atomic", reduce_block_atomic);
    do_reduce<524288>("block_atomic", reduce_block_atomic);
    do_reduce<1048576>("block_atomic", reduce_block_atomic);
    do_reduce<2097152>("block_atomic", reduce_block_atomic);
    do_reduce<4194304>("block_atomic", reduce_block_atomic);
    do_reduce<8388608>("block_atomic", reduce_block_atomic);
    do_reduce<16777216>("block_atomic", reduce_block_atomic);
    do_reduce<33554432>("block_atomic", reduce_block_atomic);
    do_reduce<67108864>("block_atomic", reduce_block_atomic);
    do_reduce<134217728>("block_atomic", reduce_block_atomic);

    do_reduce<1024>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<2048>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<4096>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<8192>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<16384>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<32768>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<65536>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<131072>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<262144>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<524288>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<1048576>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<2097152>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<4194304>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<8388608>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<16777216>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<33554432>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<67108864>("block_atomic_vec2", reduce_block_atomic_vec2);
    do_reduce<134217728>("block_atomic_vec2", reduce_block_atomic_vec2);

    do_reduce<1024>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<2048>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<4096>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<8192>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<16384>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<32768>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<65536>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<131072>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<262144>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<524288>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<1048576>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<2097152>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<4194304>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<8388608>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<16777216>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<33554432>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<67108864>("block_atomic_vec4", reduce_block_atomic_vec4);
    do_reduce<134217728>("block_atomic_vec4", reduce_block_atomic_vec4);

    return 1;
}

