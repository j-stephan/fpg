/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <cstddef>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wc++17-extensions"
#include <hc.hpp>
#pragma GCC diagnostic pop

#include "../implementation.h"

namespace acc
{
    auto block_reduce(hc::tiled_index<1> idx,
                      hc::array_view<int, 1> data,
                      hc::array_view<int, 1> result,
                      std::size_t size, int blocks, int block_size) [[hc]]
    -> void
    {
        tile_static int scratch[1024]; 

        auto i = static_cast<unsigned>(idx.tile[0] * block_size
                                       + idx.local[0]);

        if(i >= size)
            return;

        // avoid neutral element
        auto tsum = data[i];

        auto grid_size = blocks * block_size;
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

        scratch[idx.local[0]] = tsum;
        idx.barrier.wait_with_tile_static_memory_fence();


        // BLOCK + WARP, read from shared memory
        #pragma unroll
        for(auto bs = block_size, bsup = (block_size + 1) / 2;
            bs > 1;
            bs /= 2, bsup = (bs + 1) / 2)
        {
            auto cond = idx.local[0] < bsup // first half of block
                        && (idx.local[0] + bsup) < block_size
                        && static_cast<unsigned>(idx.tile[0] * block_size
                                                 + idx.local[0] + bsup)
                           < size;

            if(cond)
            {
                scratch[idx.local[0]] += scratch[idx.local[0] + bsup];
            }
            idx.barrier.wait_with_tile_static_memory_fence();

            // store to global memory
            if(idx.local[0] == 0)
                result[idx.tile[0]] = scratch[0];
        }
    }

    auto reduce(const dev_ptr& data, dev_ptr& result, std::size_t size,
                int blocks, int block_size) -> void
    {
        auto data_view = hc::array_view<int, 1>(data.p_impl->data);
        auto result_view = hc::array_view<int, 1>(result.p_impl->data);

        auto global_extent = hc::extent<1>(blocks * block_size);
        hc::parallel_for_each(global_extent.tile(block_size),
                                       [=] (hc::tiled_index<1> idx) [[hc]]
        {
            block_reduce(idx, data_view, result_view, size, blocks, block_size);
        });

        global_extent = hc::extent<1>(block_size);
        hc::parallel_for_each(global_extent.tile(block_size),
                              [=] (hc::tiled_index<1> idx) [[hc]]
        {
            block_reduce(idx, result_view, result_view, blocks, 1, block_size);
        });
    }
}
