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

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wc++17-extensions"
#pragma GCC diagnostic push
#include <hc.hpp>
#pragma GCC diagnostic pop

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "accelerator.h"

namespace acc
{
    namespace
    {
        // deal with HC's ridiculous wstrings
        auto sys_loc = std::locale{""};

        template <class I, class E, class S>
        struct convert_type : std::codecvt_byname<I, E, S>
        {
            using std::codecvt_byname<I, E, S>::codecvt_byname;
            ~convert_type() {}
        };
        using cvt = convert_type<wchar_t, char, std::mbstate_t>;
        auto&& converter = std::wstring_convert<cvt>{
            new cvt{sys_loc.name()}};
    }

    struct dev_ptr_impl
    {
        hc::array<int, 1> data;
    };

    dev_ptr::~dev_ptr()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    struct dev_clock_impl
    {
        std::uint64_t ticks;
    };

    dev_clock::~dev_clock()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    auto init() -> void
    {
        // set up default accelerator
        auto accelerators = hc::accelerator::get_all();
        std::cout << "Available accelerators: " << std::endl;
        for(auto i = 0u; i < accelerators.size(); ++i)
        {
            auto&& acc = accelerators[i];
            auto desc = converter.to_bytes(acc.get_description());
            auto path = converter.to_bytes(acc.get_device_path());

            std::cout << "\t[" << i << "] " << desc << " at " << path
                      << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Select accelerator: ";
        auto index = 0u;
        std::cin >> index;

        if(index >= accelerators.size())
        {
            std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        hc::accelerator::set_default(accelerators[index].get_device_path());
    }

    auto get_info() -> info
    {
        // select default accelerator
        auto accelerator = hc::accelerator();
        auto agent = static_cast<hsa_agent_t*>(accelerator.get_hsa_agent());

        auto mem_clock = int{};
        auto status = hsa_agent_get_info(
                        *agent,
                        static_cast<hsa_agent_info_t>(
                            HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY),
                        &mem_clock);
        if(status != HSA_STATUS_SUCCESS)
        {
            std::cerr << "hsa_agent_get_info("
                      << "HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY)"
                      << " returned an error." << std::endl;
            std::exit(status);
        }

        auto clock = int{};
        status = hsa_agent_get_info(
                    *agent,
                    static_cast<hsa_agent_info_t>(
                        HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY),
                    &clock);
        if(status != HSA_STATUS_SUCCESS)
        {
            std::cerr << "hsa_agent_get_info("
                      << "HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY)"
                      << " returned an error." << std::endl;
            std::exit(status);
        }

        auto num_sm = int{};
        status = hsa_agent_get_info(
                    *agent,
                    static_cast<hsa_agent_info_t>(
                        HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
                    &num_sm);
        if(status != HSA_STATUS_SUCCESS)
        {
            std::cerr << "hsa_agent_get_info("
                      << "HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT)"
                      << " returned an error." << std::endl;
            std::exit(status);
        }

        return info{accelerator.get_seqnum(),
                    converter.to_bytes(accelerator.get_description()),
                    1, 3, mem_clock, clock, num_sm};
    }

    auto make_array(std::size_t size) -> dev_ptr
    {
        using namespace hc;
        return dev_ptr{new dev_ptr_impl{array<int, 1>{extent<1>{size}}}};
    }

    auto copy_h2d(const std::vector<int>& src, dev_ptr& dst) -> void
    {
        hc::copy(std::begin(src), std::end(src), dst.p_impl->data);
    }

    auto copy_d2h(const dev_ptr& src, std::vector<int>& dst) -> void
    {
        hc::copy(src.p_impl->data, std::begin(dst));
    }

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

    auto do_benchmark(const dev_ptr& data, dev_ptr& result, std::size_t size,
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

    auto start_clock() -> dev_clock
    {
        return dev_clock{new dev_clock_impl{hc::get_system_ticks()}};
    }

    auto stop_clock() -> dev_clock
    {
        return dev_clock{new dev_clock_impl{hc::get_system_ticks()}};
    }

    auto get_duration(const dev_clock& start, const dev_clock& stop) -> float
    {
        auto start_ticks = start.p_impl->ticks;
        auto stop_ticks = stop.p_impl->ticks;
        auto elapsed = stop_ticks - start_ticks;

        auto tps = hc::get_tick_frequency();
        auto tpms = static_cast<float>(tps) / 1000.f;

        return static_cast<float>(elapsed) / tpms;
    }
}
