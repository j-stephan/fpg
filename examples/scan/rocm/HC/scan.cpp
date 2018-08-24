// based on NVIDIA's Modern GPU scan benchmark

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include <hc.hpp>

struct device
{
    hc::accelerator_view av;
    unsigned int cu_count;
};

template<typename T>
auto scan(int count, int iterations, T identity)
{
    // create data
    auto cpu_input = std::vector<T>(count);
    auto cpu_result = std::vector<T>(count);

    auto gpu_input = std::vector<T>(count);
    auto gpu_result = std::vector<T>(count);

    // initialize data
    std::fill(std::begin(cpu_input), std::end(cpu_input), static_cast<T>(1));
    std::fill(std::begin(gpu_input), std::end(gpu_input), static_cast<T>(1));

    // launch kernel
    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < iterations; ++i)
    {
        // params: gpu_input, count, identity, op, 0, 0, gpu_result
    }
    auto stop = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration<double>{stop - start};

    auto bytes = (2 * sizeof(T) + sizeof(T)) * count;
    auto throughput = count * iterations / dur.count();
    auto bandwidth = bytes * iterations / dur.count();
    std::cout << count << ":\t"
              << throughput / 1.0e6 << " M/s\t"
              << bandwidth / 1.0e9 << " GB/s" << std::endl;

    // verify
    auto x = identity;
    for(auto i = 0; i < count; ++i)
    {
        if(x != gpu_result[i])
        {
            std::cerr << "ERROR AT " << i << std::endl;
            std::exit(0);
        }
        x += cpu_input[i];
    }
}

template<typename T, int tile_dim>
auto reduce(const std::vector<T>& data, const device& dev) -> T
{
    auto count = static_cast<int>(data.size());
    auto tile_num = std::max(count / tile_dim, 1);

    // reduce tail
    auto stride = tile_dim * tile_num;
    auto tail_length = count % stride;
    auto tail_sum = 0;
    if(tail_length != 0)
    {
        tail_sum = std::accumulate(std::end(data) - tail_length,
                                   std::end(data), 0);
        count -= tail_length;
        if(count == 0)
            return tail_sum;
    }

    // copy data to GPU
    auto data_arr_cpu = hc::array<T, 1>{hc::extent<1>{count}, std::begin(data)};
    auto data_arr_gpu = hc::array<T, 1>{hc::extent<1>{count}, dev.av};
    data_arr_cpu.copy_to(data_arr_gpu);

    auto reduce_device = hc::array<T, 1>{hc::extent<1>{tile_num}, dev.av};

    // reduce along tiles
    hc::parallel_for_each(dev.av,
                          data_arr_gpu.get_extent().tile(tile_dim),
                          [=, &data_arr_gpu, &reduce_device]
                          (hc::tiled_index<1> tile_index) [[hc]]
    {
        // use local memory
        tile_static T tile_data[tile_dim];

        auto local_idx = tile_index.local[0];

        // Reduce thread data into tile_static memory
        auto input_idx = tile_index.tile[0] * tile_dim + local_idx;
        tile_data[local_idx] = 0;
        do
        {
            tile_data[local_idx] += data_arr_gpu[input_idx]
                                    + data_arr_gpu[input_idx + tile_dim];
            input_idx += stride;
        } while(input_idx < count);
        tile_index.barrier.wait();

        // Reduce tile data into tile_static memory
        for(auto offset = tile_dim / 2; offset > 0; offset /= 2)
        {
            if(local_idx < offset)
                tile_data[local_idx] += tile_data[local_idx + offset];

            tile_index.barrier.wait();
        }

        // Store tile result in global memory
        if(local_idx == 0)
            reduce_device[tile_index.tile[0]] = tile_data[0];
    });
    
    // global reduce
    auto reduce_host = std::vector<T>(tile_num);
    hc::copy(reduce_device, std::begin(reduce_host));

    return std::accumulate(std::begin(reduce_host),
                           std::end(reduce_host),
                           tail_sum);
}

template<typename T, int tile_dim>
auto max_reduce(int count, int iterations, const device& dev)
    -> void
{
    static_assert(tile_dim % 64 == 0);
    // create data
    auto data = std::vector<T>(count);
    auto result = std::vector<T>(1);

    // initialize data
    auto rd = std::random_device{};
    auto rng = std::mt19937{rd()};
    auto uid = std::uniform_int_distribution{0, count};
    std::generate(std::begin(data), std::end(data), [&]() { return uid(rng); });


    // reduce
    auto start = std::chrono::steady_clock::now();
    for(auto i = 0; i < iterations; ++i)
        reduce<T, tile_dim>(data, dev);
    auto stop = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration<double>{stop - start};

    auto bytes = count * sizeof(T);
    auto throughput = count * iterations / dur.count();
    auto bandwidth = bytes * iterations / dur.count();

    std::wcout << typeid(T).name() << ";" << tile_dim << ";" << count << ";"
              << throughput << ";" << bandwidth << std::endl;
}

constexpr auto tests = std::array<std::array<int, 2>, 10>{{
    {{10000, 1000}},
    {{50000, 1000}},
    {{100000, 1000}},
    {{200000, 500}},
    {{500000, 200}},
    {{1000000, 200}},
    {{2000000, 200}},
    {{5000000, 200}},
    {{10000000, 100}},
    {{20000000, 100}}
}};

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

    using T1 = int;
    using T2 = std::int64_t;

    {
        /*
        std::cout << "Benchmarking scan on type "
                  << typeid(T1).name() << std::endl;
        for(const auto& test_set : tests)
            scan(test_set[0], test_set[1], static_cast<T1>(0));

        std::cout << "Benchmarking scan on type "
                  << typeid(T2).name() << std::endl;
        for(const auto& test_set : tests)
            scan(test_set[0], test_set[1], static_cast<T2>(0));
        */
    }

    {
        std::wcout << "type;tile_dim;count;throughput;bandwidth" << std::endl;
        // 1024 is the maximum tile_dim
        for(const auto& test_set : tests)
            max_reduce<T1, 64>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 64>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 128>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 128>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 192>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 192>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 256>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 256>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 320>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 320>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 384>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 384>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 448>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 448>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 512>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 512>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 576>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 576>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 640>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 640>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 704>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 704>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 768>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 768>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 832>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 832>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 896>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 896>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 960>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 960>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T1, 1024>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;

        for(const auto& test_set : tests)
            max_reduce<T2, 1024>(test_set[0], test_set[1], dev);
        std::wcout << std::endl;
    }
}
