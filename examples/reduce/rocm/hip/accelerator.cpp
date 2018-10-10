/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <cstdint>
#include <cstdlib>
#include <cwchar>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#pragma clang diagnostic ignored "-Wextra-semi"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip/hip_runtime.h>
#pragma clang diagnostic pop

#include "accelerator.h"

#define CHECK(cmd) \
{ \
        auto error = cmd; \
        if(error != hipSuccess) \
        { \
                    std::cerr << "Error: '" << hipGetErrorString(error) \
                              << "' (" << error << ") at " << __FILE__ << ":" \
                              << __LINE__  << std::endl; \
                    std::exit(EXIT_FAILURE); \
                } \
}

namespace acc
{
    namespace
    {
        auto hstream_ = hipStream_t{};
    }

    struct dev_ptr_impl
    {
        int* ptr;
        std::size_t size;
    };

    dev_ptr::~dev_ptr()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    struct dev_clock_impl
    {
        hipEvent_t event;
    };

    dev_clock::~dev_clock()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    auto init() -> void
    {
        // set up default device
        auto dev_count = int{};
        CHECK(hipGetDeviceCount(&dev_count));

        std::cout << "Available accelerators: " << std::endl;
        for(auto d = 0; d < dev_count; ++d)
        {
            auto prop = hipDeviceProp_t{};
            CHECK(hipGetDeviceProperties(&prop, d));

            std::cout << "\t[" << d << "] " << prop.name << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Select accelerator: " << std::endl;
        auto index = 0;
        std::cin >> index;

        if(index >= dev_count)
        {
            std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        CHECK(hipSetDevice(index));
        CHECK(hipFree(nullptr)); // force context init

        CHECK(hipStreamCreate(&hstream_));
    }

    auto get_info() -> info
    {
        auto id = int{};
        auto name = std::string{};
        auto cc_major = 0;
        auto cc_minor = 0;
        auto mem_clock = 0;
        auto clock = 0;
        auto num_sm = 0;
        return info{id, name, cc_major, cc_minor, mem_clock, clock, num_sm};
    }

    auto make_array(std::size_t size) -> dev_ptr
    {
        auto d_ptr = new dev_ptr_impl{nullptr, size};
        CHECK(hipMalloc(&d_ptr->ptr, size * sizeof(int)));
        return dev_ptr{d_ptr};
    }

    auto copy_h2d(const std::vector<int>& src, dev_ptr& dst) -> void
    {
        CHECK(hipMemcpy(dst.p_impl->ptr, src.data(), src.size() * sizeof(int),
                        hipMemcpyHostToDevice));
    }

    auto copy_d2h(const dev_ptr& src, std::vector<int>& dst) -> void
    {
        CHECK(hipMemcpy(dst.data(), src.p_impl->ptr,
                        src.p_impl->size * sizeof(int),
                        hipMemcpyDeviceToHost));
    }

    __global__ void block_reduce(const int* data, int* result, std::size_t size)
    {
        __shared__ int scratch[1024]; 

        auto i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        if(i >= size)
            return;

        // avoid neutral element
        auto tsum = data[i];

        auto grid_size = hipGridDim_x * hipBlockDim_x;
        i += grid_size;

        // GRID, read from global memory
        while((i + 3 * grid_size) < size)
        {
            tsum += data[i] + data[i + grid_size] + data[i + 2 * grid_size] +
                    data[i + 3 * grid_size];
            i += 4 * grid_size;
        }

        // tail
        while(i < size)
        {
            tsum += data[i];
            i += grid_size;
        }

        scratch[i] = tsum;
        __syncthreads();


        // BLOCK + WARP, read from shared memory
        #pragma unroll
        for(auto bs = hipBlockDim_x, bsup = (hipBlockDim_x + 1) / 2;
            bs > 1;
            bs /= 2, bsup = (bs + 1) / 2)
        {
            auto cond = i < bsup // first half of block
                        && (i + bsup) < hipBlockDim_x
                        && static_cast<unsigned>(i * hipBlockDim_x + i + bsup)
                           < size;

            if(cond)
            {
                scratch[i] += scratch[i + bsup];
            }
            __syncthreads();

            // store to global memory
            if(i == 0)
                result[i] = scratch[0];
        }
    }

    auto do_benchmark(const dev_ptr& data, dev_ptr& result, std::size_t size,
                      int blocks, int block_size) -> void
    {
        hipLaunchKernelGGL(block_reduce,
                           dim3{static_cast<std::uint32_t>(blocks), 1u, 1u},
                           dim3{static_cast<std::uint32_t>(block_size), 1u, 1u},
                           0, hstream_, data.p_impl->ptr, result.p_impl->ptr,
                           size);
        hipLaunchKernelGGL(block_reduce,
                           dim3{1u, 1u, 1u},
                           dim3{static_cast<std::uint32_t>(block_size), 1u, 1u},
                           0, hstream_, result.p_impl->ptr, result.p_impl->ptr,
                           blocks);
    }

    auto start_clock() -> dev_clock
    {
        auto clock_impl = new dev_clock_impl{};
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(clock_impl->event, hstream_));
        return dev_clock{clock_impl};
    }

    auto stop_clock() -> dev_clock
    {
        auto clock_impl = new dev_clock_impl{};
        CHECK(hipEventRecord(clock_impl->event, hstream_));
        CHECK(hipEventSynchronize(clock_impl->event));
        return dev_clock{clock_impl};
    }

    auto get_duration(const dev_clock& start, const dev_clock& stop) -> float
    {
        auto ms = float{};
        CHECK(hipGetLastError());
        CHECK(hipEventElapsedTime(&ms, start.p_impl->event,
                                  stop.p_impl->event));
        return ms;
    }
}
