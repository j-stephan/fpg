/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
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

#include <CL/sycl.hpp>

// TODO: shuffle-reduce, vec (int3)

/*auto tile_reduce_sum(int val, cl::sycl::nd_item<1> item) -> int
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
}*/

// =============================================================================
// Stable
// =============================================================================

/*auto device_reduce_stable(hc::tiled_index<1> idx,
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
}*/

inline auto reduce_stable(cl::sycl::buffer<int, 1>& data,
                          cl::sycl::buffer<int, 1>& result,
                          cl::sycl::queue& queue,
                          std::size_t local_size)
{
    queue.submit([&] (cl::sycl::handler& cgh)
    {
        auto data_acc = data.get_access<cl::sycl::access::mode::read>(cgh);
        auto res_acc = result.get_access<cl::sycl::access::mode::discard_write>(cgh);

        // local_size elements in local memory
        auto local_mem = cl::sycl::accessor<int, 1,
                                            cl::sycl::access::mode::read_write,
                                            cl::sycl::access::target::local>
                                            {cl::sycl::range<1>{local_size}, cgh}; 
        // TODO: actual kernel
        cgh.parallel_for<class reduction_kernel>(cl::sycl::nd_range<1>{1024, 512},
        [=] (cl::sycl::nd_item<1> item)
        {
            auto local_id = item.get_local_linear_id();
            auto global_id = item.get_global_linear_id();

            local_mem[local_id] = 0;
            item.barrier(cl::sycl::access::fence_space::local_space);
        }
        );
    }
    );

    queue.submit([&] (cl::sycl::handler& cgh)
    {
        auto res_acc = result.get_access<cl::sycl::access::mode::read_write>(cgh);

        // TODO: actual kernel
    }
    );
}

// =============================================================================
// Wavefront + Atomic
// =============================================================================
/*auto device_reduce_wavefront_atomic(hc::tiled_index<1> idx,
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
}*/

// =============================================================================
// Tile + Atomic
// =============================================================================
/*auto device_reduce_tile_atomic(hc::tiled_index<1> idx,
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
}*/

// =============================================================================
// Host
// =============================================================================

template <int array_dim, typename Func>
auto do_reduce(const std::string& label, cl::sycl::device& dev, Func f) -> void
{
    static_assert(array_dim % 64 == 0, "array_dim % 64 != 0");

    constexpr auto iterations = 5;
    constexpr auto lanes = std::min(512, array_dim);
    constexpr auto tiles = array_dim / lanes;

    // create queue
    auto queue = cl::sycl::queue{dev};

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
    auto data_gpu = cl::sycl::buffer<int, 1>{std::begin(data), std::end(data)};
    auto result_gpu = cl::sycl::buffer<int, 1>{std::begin(result), std::end(result)};

    // this is needed for local memory allocation
    auto&& device = queue.get_device();
    auto wg_size = device.get_info<cl::sycl::info::device::max_work_group_size>();

    // warm up
    f(data_gpu, result_gpu, queue, wg_size);

    // reduce
    /*auto start = std::chrono::steady_clock::now();
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

    std::cout << label << ";" << array_dim << ";" << bandwidth << std::endl;*/
}

auto main() -> int
{
    try
    {
        // set up platform
        auto platforms = cl::sycl::platform::get_platforms(); 
        std::cout << "Available platforms: " << std::endl;
        for(auto i = 0; i < platforms.size(); ++i)
        {
            auto&& platform = platforms[i];
            std::cout << "\t[" << i << "] "
                      << platform.get_info<cl::sycl::info::platform::name>()
                      << ", vendor: "
                      << platform.get_info<cl::sycl::info::platform::vendor>()
                      << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Select platform: ";
        auto index = 0;
        std::cin >> index;

        if(index >= platforms.size())
        {
            std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                      << std::endl;
            std::exit(-1);
        }

        auto&& platform = platforms[index];

        // set up context
        auto context = cl::sycl::context{platform};

        // set up device
        auto devices = context.get_devices();

        std::cout << std::endl;
        std::cout << "Available devices: " << std::endl;
        for(auto i = 0; i < devices.size(); ++i)
        {
            auto&& device = devices[i];

            std::cout << "\t[" << i << "] "
                      << device.get_info<cl::sycl::info::device::name>()
                      << " at "
                      << device.get_info<cl::sycl::info::device::vendor_id>()
                      << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Select platform: ";
        index = 0;
        std::cin >> index;

        if(index >= devices.size())
        {
            std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                      << std::endl;
            std::exit(-1);
        }

        auto&& dev = devices[index];

        std::cout << "type;array_dim;bandwidth" << std::endl;
        do_reduce<1024>("stable", dev, reduce_stable);
        do_reduce<2048>("stable", dev, reduce_stable);
        do_reduce<4096>("stable", dev, reduce_stable);
        do_reduce<8192>("stable", dev, reduce_stable);
        do_reduce<16384>("stable", dev, reduce_stable);
        do_reduce<32768>("stable", dev, reduce_stable);
        do_reduce<65536>("stable", dev, reduce_stable);
        do_reduce<131072>("stable", dev, reduce_stable);
        do_reduce<262144>("stable", dev, reduce_stable);
        do_reduce<524288>("stable", dev, reduce_stable);
        do_reduce<1048576>("stable", dev, reduce_stable);
        do_reduce<2097152>("stable", dev, reduce_stable);
        do_reduce<4194304>("stable", dev, reduce_stable);
        do_reduce<8388608>("stable", dev, reduce_stable);
        do_reduce<16777216>("stable", dev, reduce_stable);
        do_reduce<33554432>("stable", dev, reduce_stable);
        do_reduce<67108864>("stable", dev, reduce_stable);
        do_reduce<134217728>("stable", dev, reduce_stable);

        /*do_reduce<1024>(L"wavefront_atomic", dev, reduce_wavefront_atomic);
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
        do_reduce<134217728>(L"tile_atomic", dev, reduce_tile_atomic);*/
        return 1;
    }
    catch(const cl::sycl::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
