#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hc.hpp>
#pragma clang diagnostic pop

#include <hc_math.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#include <hc_short_vector.hpp>
#pragma clang diagnostic pop

using double2 = hc::short_vector::double_2;
using double4 = hc::short_vector::double_4;

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

template <typename DataT>
auto initialize(hc::tiled_index<1> idx, hc::array_view<DataT, 1> A,
                hc::array_view<DataT, 1> B) [[hc]] -> void;

template <>
auto initialize<float>(hc::tiled_index<1> idx, hc::array_view<float, 1> A,
                       hc::array_view<float, 1> B) [[hc]] -> void
{
    auto i = idx.global[0];
    A[i] = float{};
    B[i] = std::numeric_limits<float>::quiet_NaN();
}

template <>
auto initialize<double>(hc::tiled_index<1> idx, hc::array_view<double, 1> A,
                       hc::array_view<double, 1> B) [[hc]] -> void
{
    auto i = idx.global[0];
    A[i] = double{};
    B[i] = std::numeric_limits<double>::quiet_NaN();
}

template <>
auto initialize<double2>(hc::tiled_index<1> idx, hc::array_view<double2, 1> A,
                         hc::array_view<double2, 1> B) [[hc]] -> void
{
    auto i = idx.global[0];
    A[i] = double2{};
    B[i] = double2{std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::quiet_NaN()};
}

template <>
auto initialize<double4>(hc::tiled_index<1> idx, hc::array_view<double4, 1> A,
                         hc::array_view<double4, 1> B) [[hc]] -> void
{
    auto i = idx.global[0];
    A[i] = double4{};
    B[i] = double4{std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::quiet_NaN()};
}

template <typename DataT>
auto read_write(hc::tiled_index<1> idx, hc::array_view<DataT, 1> A,
                hc::array_view<DataT, 1> B, std::size_t elems,
                unsigned int tiles, unsigned int tile_size) [[hc]] -> void
{
    auto stride = tiles * tile_size;
    for(auto i = idx.tile[0] * tile_size + idx.local[0];
             i < elems;
             i += stride)
    {
        B[i] = A[i];
    } 
}

template <typename DataT>
auto write(hc::tiled_index<1> idx, hc::array_view<DataT, 1> B,
           std::size_t elems, unsigned int tiles, unsigned int tile_size) [[hc]]
-> void
{
    auto stride = tiles * tile_size;
    for(auto i = idx.tile[0] * tile_size + idx.local[0];
             i < elems;
             i += stride)
    {
        B[i] = DataT{};
    } 
}

template <typename DataT>
auto do_benchmark(unsigned int cus, std::ofstream& file) -> void
{
    std::cout << "Benchmarking size " << sizeof(DataT) << std::endl;
    constexpr auto iters = 10;
    constexpr auto max_mem = 1u << 31; // memory per vector
    constexpr auto max_elems = max_mem / sizeof(DataT);
    constexpr auto max_tiles = 1u << 22; // determined experimentally 

    for(auto tile_size = 64u; tile_size <= 1024u; tile_size *= 2u)
    {

        for(auto elems = tile_size * cus; elems <= max_elems; elems *= 2u)
        {
            // Allocate memory on device
            auto A_d = hc::array_view<DataT, 1>{hc::extent<1>{elems}};
            auto B_d = hc::array_view<DataT, 1>{hc::extent<1>{elems}};

            for(auto tile_num = cus;
                     tile_num <= std::min(elems / tile_size, max_tiles);
                     tile_num *= 2u)
            {
                // Initialize device memory
                hc::parallel_for_each(hc::extent<1>{elems}.tile(1024),
                [=] (hc::tiled_index<1> idx) [[hc]]
                {
                    initialize(idx, A_d, B_d);
                });

                auto mintime = std::numeric_limits<float>::max();
                for(auto k = 0; k < iters; ++k)
                {
                    auto start = hc::get_system_ticks();

                    hc::parallel_for_each(
                    hc::extent<1>{tile_num * tile_size}.tile(tile_size),
                    [=] (hc::tiled_index<1> idx) [[hc]]
                    {
                        read_write(idx, A_d, B_d, elems, tile_num, tile_size);
                    });
                    
                    auto stop = hc::get_system_ticks();
                    auto elapsed_ticks = stop - start;

                    auto tps = hc::get_tick_frequency();
                    auto tpms = static_cast<float>(tps) / 1000.f;

                    auto elapsed = static_cast<float>(elapsed_ticks) / tpms;

                    mintime = std::min(mintime, elapsed);
                }

                file << "RW;" << tile_size << ";" << tile_num << ";"
                     << sizeof(DataT) << ";" << elems << ";"
                     << mintime << ";"
                     << (2.0e-9 * sizeof(DataT) * elems) / (mintime / 1e3)
                     << std::endl;
            }

            file << std::endl;

            for(auto tile_num = cus;
                     tile_num <= std::min(elems / tile_size, max_tiles);
                     tile_num *= 2u)
            {
                // Initialize device memory
                hc::parallel_for_each(hc::extent<1>{elems}.tile(1024),
                [=] (hc::tiled_index<1> idx) [[hc]]
                {
                    initialize(idx, A_d, B_d);
                });

                auto mintime = std::numeric_limits<float>::max();
                for(auto k = 0; k < iters; ++k)
                {
                    auto start = hc::get_system_ticks();

                    hc::parallel_for_each(
                    hc::extent<1>{tile_num * tile_size}.tile(tile_size),
                    [=] (hc::tiled_index<1> idx) [[hc]]
                    {
                        write(idx, B_d, elems, tile_num, tile_size);
                    });

                    auto stop = hc::get_system_ticks();
                    auto elapsed_ticks = stop - start;

                    auto tps = hc::get_tick_frequency();
                    auto tpms = static_cast<float>(tps) / 1000.f;

                    auto elapsed = static_cast<float>(elapsed_ticks) / tpms;

                    mintime = std::min(mintime, elapsed);
                }

                file << "WO;" << tile_size << ";" << tile_num << ";"
                     << sizeof(DataT) << ";" << elems << ";"
                     << mintime << ";"
                     << (1.0e-9 * sizeof(DataT) * elems) / (mintime / 1e3)
                     << std::endl;
            }

            file << std::endl;
        }
    }
}

auto main() -> int
{
    // set up devices
    auto accelerators = hc::accelerator::get_all();
    std::cout << "Available accelerators: " << std::endl;
    for(auto i = 0u; i < accelerators.size(); ++i)
    {
        auto&& acc = accelerators[i];
        std::cout << "\t[" << i << "] "
                  << converter.to_bytes(acc.get_description())
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

    auto&& my_accelerator = accelerators[index];
    hc::accelerator::set_default(my_accelerator.get_device_path());

    auto cus = my_accelerator.get_cu_count();

    auto now = std::chrono::system_clock::now(); 
    auto cnow = std::chrono::system_clock::to_time_t(now);

    auto filename = std::stringstream{};
    filename << std::put_time(std::localtime(&cnow), "%Y-%m-%d %X");
    filename << ".csv";

    auto file = std::ofstream{filename.str()};

    file << "type;tile_size;tile_num;elem_size;elem_num;mintime;throughput"
         << std::endl;

    do_benchmark<float>(cus, file);
    do_benchmark<double>(cus, file);
    do_benchmark<double2>(cus, file);
    do_benchmark<double4>(cus, file);

    return EXIT_SUCCESS;
}
