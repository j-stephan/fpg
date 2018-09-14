/* This file is part of fpg.
 *
 * fpg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * fpg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with fpg. If not, see <http://www.gnu.org/licenses/>
 */

// based on Justin Luitjen's (NVIDIA) "Faster Parallel Reductions on Kepler"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include <hc.hpp>

struct device
{
    hc::accelerator_view av;
    unsigned int cu_count;
};

// TODO: shuffle-reduce, vec (int3)

auto wavefront_reduce_sum(int val) [[hc]] -> int
{
    for(auto offset = (__HSA_WAVEFRONT_SIZE__ / 2); offset > 0; offset /= 2)
        val += hc::__shfl_down(val, offset);

    return val;
} 

auto tile_reduce_sum(int val, hc::tiled_index<1> idx) [[hc]] -> int
{
    tile_static int scratch[16]; // scratch memory for 16 partial sums
    const auto lane_id = hc::__lane_id();
    const auto wavefront_id = idx.local[0] / __HSA_WAVEFRONT_SIZE__;

    // each wavefront performs partial reduction
    val = wavefront_reduce_sum(val);
    if(lane_id == 0)
        scratch[wavefront_id] = val;

    // wait for all partial reductions
    idx.barrier.wait_with_tile_static_memory_fence();

    // read from scratch memory only if that wavefront existed
    val = (idx.local[0] < std::max(idx.tile_dim[0] / __HSA_WAVEFRONT_SIZE__, 1))
            ? scratch[lane_id] : 0;

    if(wavefront_id == 0)
        // final reduce within first wavefront
        val = wavefront_reduce_sum(val);

    return val;
}

// =============================================================================
// Stable
// =============================================================================

auto device_reduce_stable(hc::tiled_index<1> idx,
                          hc::array_view<int, 1> data,
                          hc::array_view<int, 1> result) [[hc]]
{
    auto sum = 0;
    auto global_extent = data.get_extent();
    // reduce multiple elements per lane
    for(auto i = idx.tile[0] * idx.tile_dim[0] + idx.local[0];
        i < global_extent[0];
        i += idx.tile_dim[0] * (global_extent[0] / idx.tile_dim[0]))
    {
        sum += data[i];
    }
    // tile_reduce_sum returns correct result only on first thread
    sum = tile_reduce_sum(sum, idx);
    if(idx.local[0] == 0)
        result[idx.tile[0]] = sum;
}

inline auto reduce_stable(hc::array_view<int, 1> data,
                          hc::array_view<int, 1> result,
                          std::size_t lanes, std::size_t tiles, device& dev)
{
    hc::parallel_for_each(dev.av,
                          data.get_extent().tile(lanes),
                          [=] (hc::tiled_index<1> idx) [[hc]]
    {
        device_reduce_stable(idx, data, result);
    }
    );

    hc::parallel_for_each(dev.av,
                          result.get_extent().tile(std::min(tiles, 1024ul)),
                          [=] (hc::tiled_index<1> idx) [[hc]]
    {
        device_reduce_stable(idx, result, result);
    }
    ); 
}

// =============================================================================
// Wavefront + Atomic
// =============================================================================
auto device_reduce_wavefront_atomic(hc::tiled_index<1> idx,
                                    hc::array_view<int, 1> data,
                                    hc::array_view<int, 1> result) [[hc]]
{
    auto sum = 0;
    auto global_extent = data.get_extent();
    // reduce multiple elements per lane
    for(auto i = idx.tile[0] * idx.tile_dim[0] + idx.local[0];
        i < global_extent[0];
        i += idx.tile_dim[0] * (global_extent[0] / idx.tile_dim[0]))
    {
        sum += data[i];
    }
    sum = wavefront_reduce_sum(sum);
    if(hc::__lane_id() == 0)
        hc::atomic_fetch_add(result.accelerator_pointer(), sum);
}

inline auto reduce_wavefront_atomic(const hc::array_view<int, 1>& data,
                                    hc::array_view<int, 1>& result,
                                    std::size_t lanes, std::size_t tiles,
                                    device& dev)
{
    hc::parallel_for_each(dev.av,
                          data.get_extent().tile(lanes),
                          [=] (hc::tiled_index<1> idx) [[hc]]
    {
        device_reduce_wavefront_atomic(idx, data, result);
    }
    );
}

// =============================================================================
// Tile + Atomic
// =============================================================================
auto device_reduce_tile_atomic(hc::tiled_index<1> idx,
                               hc::array_view<int, 1> data,
                               hc::array_view<int, 1> result) [[hc]]
{
    auto sum = 0;
    auto global_extent = data.get_extent();
    // reduce multiple elements per lane
    for(auto i = idx.tile[0] * idx.tile_dim[0] + idx.local[0];
        i < global_extent[0];
        i += idx.tile_dim[0] * (global_extent[0] / idx.tile_dim[0]))
    {
        sum += data[i];
    }
    sum = tile_reduce_sum(sum, idx);
    if(idx.local[0] == 0)
        hc::atomic_fetch_add(result.accelerator_pointer(), sum);
}

inline auto reduce_tile_atomic(hc::array_view<int, 1> data,
                               hc::array_view<int, 1> result,
                               std::size_t lanes, std::size_t tiles,
                               device& dev)
{
    hc::parallel_for_each(dev.av,
                          data.get_extent().tile(lanes),
                          [=] (hc::tiled_index<1> idx) [[hc]]
    {
        device_reduce_tile_atomic(idx, data, result);
    }
    );
}

// =============================================================================
// Host
// =============================================================================

template <int array_dim, typename Func>
auto do_reduce(const std::wstring& label, device& dev, Func f) -> void
{
    static_assert(array_dim % 64 == 0, "array_dim % 64 != 0");

    constexpr auto iterations = 5;
    constexpr auto lanes = std::min(512, array_dim);
    constexpr auto tiles = array_dim / lanes;

    // create data
    auto data = std::vector<int>{};
    data.resize(array_dim);
    
    // initialize data
    auto rd = std::random_device{};
    auto rng = std::mt19937{rd()};
    auto uid = std::uniform_int_distribution{0, 3};
    std::generate(std::begin(data), std::end(data), [&]() { return uid(rng); });

    auto result = std::vector<int>{};
    auto result_size = tiles * iterations + tiles;
    result.resize(result_size);
    std::fill(std::begin(result), std::end(result), 0);

    // copy to GPU
    auto data_gpu = hc::array<int, 1>{hc::extent<1>{array_dim},
                                      std::begin(data), std::end(data),
                                      dev.av};
    auto result_gpu = hc::array<int, 1>{hc::extent<1>{result_size},
                                        std::begin(result), std::end(result),
                                        dev.av};
    auto data_view = hc::array_view<int, 1>{data_gpu};

    // warm up
    auto res_warmup_view = result_gpu.section(hc::index<1>{tiles * iterations},
                                              hc::extent<1>{tiles});
    f(data_view, res_warmup_view, lanes, tiles, dev);

    // reduce
    auto start = std::chrono::steady_clock::now();
    for(auto i = 0; i < iterations; ++i)
    {
        // iterate through different buffers
        auto res_view = result_gpu.section(hc::index<1>{i * tiles},
                                           hc::extent<1>{tiles});
        f(data_view, res_view, lanes, tiles, dev);
    }
    auto stop = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration<double>{stop - start};
 
    hc::copy(result_gpu, std::begin(result)); 

    // verify
    auto verify = std::accumulate(std::begin(data), std::end(data), 0);
    if(verify != result[0])
    {
        std::cerr << "Mismatch: expected " << verify
                  << ", got " << result[0] << std::endl;
    }

    constexpr auto bytes = array_dim * sizeof(int);
    // auto throughput = array_dim * iterations / dur.count();
    auto bandwidth = bytes * iterations / dur.count();

    std::wcout << label << ";" << array_dim << ";" << bandwidth << std::endl;
}

auto main() -> int
{
    // set up accelerator
    auto accelerators = hc::accelerator::get_all();

    std::wcout << "Available accelerators: " << std::endl;
    for(auto i = 0; i < accelerators.size(); ++i)
    {
        auto&& acc = accelerators[i];
        std::wcout << "\t[" << i << "] " << acc.get_description()
                   << " at " << acc.get_device_path() << std::endl;
    }

    std::wcout << std::endl;
    std::wcout << "Select accelerator: ";
    auto index = 0;
    std::cin >> index;

    if(index >= accelerators.size())
    {
        std::wcout << "I'm sorry, Dave. I'm afraid I can't do that."
                   << std::endl;
        std::exit(-1);
    }

    auto dev = device{accelerators[index].get_default_view(),
                      accelerators[index].get_cu_count()};

    std::wcout << "type;array_dim;bandwidth" << std::endl;
    do_reduce<1024>(L"stable", dev, reduce_stable);
    do_reduce<2048>(L"stable", dev, reduce_stable);
    do_reduce<4096>(L"stable", dev, reduce_stable);
    do_reduce<8192>(L"stable", dev, reduce_stable);
    do_reduce<16384>(L"stable", dev, reduce_stable);
    do_reduce<32768>(L"stable", dev, reduce_stable);
    do_reduce<65536>(L"stable", dev, reduce_stable);
    do_reduce<131072>(L"stable", dev, reduce_stable);
    do_reduce<262144>(L"stable", dev, reduce_stable);
    do_reduce<524288>(L"stable", dev, reduce_stable);
    do_reduce<1048576>(L"stable", dev, reduce_stable);
    do_reduce<2097152>(L"stable", dev, reduce_stable);
    do_reduce<4194304>(L"stable", dev, reduce_stable);
    do_reduce<8388608>(L"stable", dev, reduce_stable);
    do_reduce<16777216>(L"stable", dev, reduce_stable);
    do_reduce<33554432>(L"stable", dev, reduce_stable);
    do_reduce<67108864>(L"stable", dev, reduce_stable);
    do_reduce<134217728>(L"stable", dev, reduce_stable);

    do_reduce<1024>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<2048>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<4096>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<8192>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<16384>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<32768>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<65536>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<131072>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<262144>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<524288>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<1048576>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<2097152>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<4194304>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<8388608>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<16777216>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<33554432>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<67108864>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
    do_reduce<134217728>(L"wavefront_atomic", dev, reduce_wavefront_atomic);

    do_reduce<1024>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<2048>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<4096>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<8192>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<16384>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<32768>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<65536>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<131072>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<262144>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<524288>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<1048576>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<2097152>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<4194304>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<8388608>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<16777216>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<33554432>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<67108864>(L"tile_atomic", dev, reduce_tile_atomic);
    do_reduce<134217728>(L"tile_atomic", dev, reduce_tile_atomic);
}
