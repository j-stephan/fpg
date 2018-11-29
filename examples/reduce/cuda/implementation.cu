/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <cstddef>
#include <cstdlib>
#include <cwchar>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

#include "common/common.h"
#include "../implementation.h"

namespace impl
{
    __global__ void block_reduce(const int* data, int* result, std::size_t size)
    {
        __shared__ int scratch[1024]; 

        auto i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= size)
            return;

        // avoid neutral element
        auto tsum = data[i];

        auto grid_size = gridDim.x * blockDim.x;
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

        scratch[threadIdx.x] = tsum;
        __syncthreads();

        // BLOCK + WARP, read from shared memory
        #pragma unroll
        for(auto bs = blockDim.x, bsup = (blockDim.x + 1) / 2;
            bs > 1;
            bs /= 2, bsup = (bs + 1) / 2)
        {
            auto cond = threadIdx.x < bsup // first half of block
                        && (threadIdx.x + bsup) < blockDim.x
                        && (blockIdx.x * blockDim.x +
                            threadIdx.x + bsup) < size;

            if(cond)
            {
                scratch[threadIdx.x] += scratch[threadIdx.x + bsup];
            }
            __syncthreads();

            // store to global memory
            if(threadIdx.x == 0)
                result[blockIdx.x] = scratch[0];
        }
    }

    auto reduce(dev_handle& handle, const dev_ptr& data, dev_ptr& result,
                std::size_t size, int blocks, int block_size) -> void
    {
        block_reduce<<<blocks, block_size, 0, dev_handle.p_impl->stream>>>(
                data.p_impl->ptr, result.p_impl->ptr, size);
        
        block_reduce<<<1, block_size, 0, dev_handle.p_impl->stream>>>(
                result.p_impl->ptr, result.p_impl->ptr, blocks);
    }

}

// vim:ft=cuda
