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

// TODO: shuffle-reduce, atomic-reduce (wavefront), atomic-reduce (tile), vec

constexpr auto wavefront_dim = 64;

auto wavefront_reduce_sum(int val) [[hc]] -> int
{
    for(auto offset = wavefront_dim / 2; offset > 0; offset /= 2)
        val += hc::__shfl_down(val, offset);

    return val;
} 

auto tile_reduce_sum(int val, hc::tiled_index<1> idx) [[hc]] -> int
{
    tile_static int scratch[wavefront_dim]; // scratch memory for 64 sums
    auto lane_id = hc::__lane_id();
    auto wavefront_id = idx.local[0] / wavefront_dim;

    // each wavefront performs partial reduction
    val = wavefront_reduce_sum(val);
    if(lane_id == 0)
        scratch[wavefront_id] = val;

    // wait for all partial reductions
    idx.barrier.wait();

    // read from shared memory only if that wavefront existed
    val = (idx.local[0] < idx.tile_dim[0] / wavefront_dim)
            ? scratch[lane_id] : 0;

    if(wavefront_id == 0)
        // final reduce within first wavefront
        val = wavefront_reduce_sum(val);

    return val;
}

auto device_reduce(hc::tiled_index<1> idx,
                   const hc::array<int, 1>& data,
                   hc::array<int, 1>& result) [[hc]] -> void
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
    {
        result[idx.tile[0]] = sum;
        //printf("%i\n", sum);
    }
}

template<std::size_t size>
auto reduce_stable(std::vector<int>& data, device& dev)
    -> int
{
    constexpr auto lanes = std::min(512ul, size);
    constexpr auto tiles = size / lanes;
    auto result = std::vector<int>{};
    result.resize(tiles);
    
    auto data_gpu = hc::array<int, 1>{hc::extent<1>{size},
                                      std::begin(data), std::end(data),
                                      dev.av};
    auto result_gpu = hc::array<int, 1>{hc::extent<1>{tiles}, dev.av};

    hc::parallel_for_each(dev.av,
                          data_gpu.get_extent().tile(lanes),
                          [&] (hc::tiled_index<1> idx) [[hc]]
    {
        device_reduce(idx, data_gpu, result_gpu);
    }
    );

/*    hc::parallel_for_each(dev.av,
                          result_gpu.get_extent().tile(std::min(tiles, 1024ul)),
                          [&] (hc::tiled_index<1> idx) [[hc]]
    {
        device_reduce(idx, result_gpu, result_gpu);
    }
    );*/

    hc::copy(result_gpu, std::begin(result)); 

    return result[0];
}

template <int array_dim, typename Func>
auto do_reduce(const std::wstring& label, device& dev, Func f) -> void
{
    static_assert(array_dim % 64 == 0, "array_dim % 64 != 0");

    constexpr auto iterations = 5;

    // create data
    auto data = std::vector<int>{};
    data.resize(array_dim);
    
    // initialize data
    auto rd = std::random_device{};
    auto rng = std::mt19937{rd()};
    auto uid = std::uniform_int_distribution{0, 3};
    std::generate(std::begin(data), std::end(data), [&]() { return uid(rng); });

    // reduce
    auto start = std::chrono::steady_clock::now();
    auto result = 0;
    for(auto i = 0; i < iterations; ++i)
    {
        result = f(data, dev);
    }
    auto stop = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration<double>{stop - start};

    // verify
    auto verify = std::accumulate(std::begin(data), std::end(data), 0);
    if(verify != result)
    {
        std::cerr << "Mismatch: expected " << verify
                  << ", got " << result << std::endl;
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

    auto dev = device{accelerators[index].get_default_view(),
                      accelerators[index].get_cu_count()};

    std::wcout << "type;array_dim;bandwidth" << std::endl;
    do_reduce<64>(L"stable", dev, reduce_stable<64>);
    do_reduce<128>(L"stable", dev, reduce_stable<128>);
    do_reduce<256>(L"stable", dev, reduce_stable<256>);
    do_reduce<512>(L"stable", dev, reduce_stable<512>);
    /*
    do_reduce<1024>(L"stable", dev, reduce_stable<1024>);
    do_reduce<2048>(L"stable", dev, reduce_stable<2048>);
    do_reduce<4096>(L"stable", dev, reduce_stable<4096>);
    do_reduce<8192>(L"stable", dev, reduce_stable<8192>);
    do_reduce<16384>(L"stable", dev, reduce_stable<16384>);
    do_reduce<32768>(L"stable", dev, reduce_stable<32768>);
    do_reduce<65536>(L"stable", dev, reduce_stable<65536>);
    do_reduce<131072>(L"stable", dev, reduce_stable<131072>);
    do_reduce<262144>(L"stable", dev, reduce_stable<262144>);
    do_reduce<524288>(L"stable", dev, reduce_stable<524288>);
    do_reduce<1048576>(L"stable", dev, reduce_stable<1048576>);
    do_reduce<2097152>(L"stable", dev, reduce_stable<2097152>);
    do_reduce<4194304>(L"stable", dev, reduce_stable<4194304>);
    do_reduce<8388608>(L"stable", dev, reduce_stable<8388608>);
    do_reduce<16777216>(L"stable", dev, reduce_stable<16777216>);
    do_reduce<33554432>(L"stable", dev, reduce_stable<33554432>);
    do_reduce<67108864>(L"stable", dev, reduce_stable<67108864>);
    do_reduce<134217728>(L"stable", dev, reduce_stable<134217728>);
    */
}
