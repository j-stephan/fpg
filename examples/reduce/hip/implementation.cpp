/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <cstddef>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#pragma clang diagnostic ignored "-Wextra-semi"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#pragma clang diagnostic pop

#include "../implementation.h"

namespace impl
{
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

        scratch[hipThreadIdx_x] = tsum;
        __syncthreads();


        // BLOCK + WARP, read from shared memory
        #pragma unroll
        for(auto bs = hipBlockDim_x, bsup = (hipBlockDim_x + 1) / 2;
            bs > 1;
            bs /= 2, bsup = (bs + 1) / 2)
        {
            auto cond = hipThreadIdx_x < bsup // first half of block
                        && (hipThreadIdx_x + bsup) < hipBlockDim_x
                        && (hipBlockIdx_x * hipBlockDim_x +
                            hipThreadIdx_x + bsup) < size;

            if(cond)
            {
                scratch[hipThreadIdx_x] += scratch[hipThreadIdx_x + bsup];
            }
            __syncthreads();

            // store to global memory
            if(hipThreadIdx_x == 0)
                result[hipBlockIdx_x] = scratch[0];
        }
    }

    auto reduce(device_handle& handle, const dev_ptr& data, dev_ptr& result,
                std::size_t size, int blocks, int block_size) -> void
    {
        hipLaunchKernelGGL(block_reduce,
                           dim3(blocks), dim3(block_size), 0,
                           handle.p_impl->stream,
                           data.p_impl->ptr, result.p_impl->ptr, size);
        hipLaunchKernelGGL(block_reduce,
                           dim3(1), dim3(block_size), 0,
                           handle.p_impl->stream,
                           result.p_impl->ptr, result.p_impl->ptr, blocks);
    }
}

// vim:ft=hip
