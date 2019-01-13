#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <limits>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wc++17-extensions"
#include <hc.hpp>
#pragma clang diagnostic pop

#include <hc_math.hpp>
#include <hc_short_vector.hpp>

constexpr auto elems = 1 << 28;
constexpr auto iters = 10;

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

auto initialize(hc::tiled_index<1> idx, hc::tiled_extent<1> extent,
                hc::array_view<hc::double_2, 1> A,
                hc::array_view<hc::double_2, 1> B) [[hc]] -> void
{
    auto i = idx.global[0];
    A[i] = hc::double_2{0.0, 0.0};
    B[i] = hc::double_2{std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN()};
}

auto read_write(hc::tiled_index<1> idx, hc::tiled_extent<1> extent,
                const hc::array_view<const hc::double_2, 1> A,
                      hc::array_view<      hc::double_2, 1> B) [[hc]] -> void
{
    auto stride = extent[0] * extent.tile_dim[0];
    for(auto i = idx.tile[0] * extent.tile_dim[0] + idx.local[0];
             i < elems;
             i += stride)
    {
        B[i] = A[i];
    } 
}

auto write(hc::tiled_index<1> idx, hc::tile_extent<1> extent,
           hc::array_view<hc::double_2, 1>B) [[hc]] -> void
{
    auto stride = extent[0] * extent.tile_dim[0];
    for(auto i = idx.tile[0] * extent.tile_dim[0] + idx.local[0];
             i < elems;
             i += stride)
    {
        B[i] = hc::double_2{0.0, 0.0};
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

    hc::accelerator::set_default(accelerators[index].get_device_path());

    // Allocate memory on device
    auto A_d = hc::array_view<hc::double_2, 1>{hc::extent<1>{elems}};
    auto B_d = hc::array_view<hc::double_2, 1>{hc::extent<1>{elems}};

    // Initialize device memory
    {
        auto global_extent = hc::extent<1>{elems};
        auto tiled_extent = global_extent.tile(1024);
        hc::parallel_for_each(tiled_extent, [=] (hc::tiled_index<1> idx)
        {
            initialize(idx, tiled_extent, A_d, B_d);
        });
    }

    constexpr auto tile_size = 128;
    constexpr auto num_tiles = (elems + (tile_size - 1)) / tile_size;
    constexpr auto tiles = num_tiles > 65520 ? 65520 : num_tiles;

    std::cout << "zcopy: operating on vectors of " << elems << " double2s"
              << " = " << sizeof(hc::double_2) * elems << " bytes" << std::endl;

    std::cout << "zcopy: using " << tile_size << " tiles per block, "
              << tiles << " tiles" << std::endl;

    auto mintime = std::numeric_limits<float>::max();
    for(auto k = 0; k < iters; ++k)
    {
        auto start = hc::get_system_ticks();

        auto global_extent = hc::extent<1>{tiles * tile_size};
        auto tiled_extent = global_extent.tile(tile_size);
        hc::parallel_for_each(tiled_extent, [=] (hc::tiled_index<1> idx)
        {
            read_write(idx, tiled_extent, A_d, B_d);
        });
        

        auto stop = hc::get_system_ticks();
        auto elapsed_ticks = stop - start;

        auto tps = hc::get_tick_frequency();
        auto tpms = static_cast<float>(tps) / 1000.f;

        auto elapsed = static_cast<float>(elapsed_ticks) / tpms;

        mintime = std::min(mintime, elapsed);
    }

    std::cout << "RW: mintime = " << mintime << " msec  "
              << "throughput = "
              << (2.0e-9 * sizeof(hc::double_2) * elems) / (mintime / 1e3)
              << " GB/sec" << std::endl;

    for(auto k = 0; k < iters; ++k)
    {
        auto start = hc::get_system_ticks();

        auto global_extent = hc::extent<1>{tiles * tile_size};
        auto tiled_extent = global_extent.tile(tile_size);
        hc::parallel_for_each(tiled_extent, [=] (hc::tiled_index<1> idx)
        {
            write(idx, tiled_extent, B_d);
        });
        

        auto stop = hc::get_system_ticks();
        auto elapsed_ticks = stop - start;

        auto tps = hc::get_tick_frequency();
        auto tpms = static_cast<float>(tps) / 1000.f;

        auto elapsed = static_cast<float>(elapsed_ticks) / tpms;

        mintime = std::min(mintime, elapsed);
    }

    std::cout << "WO: mintime = " << mintime << " msec  "
              << "throughput = "
              << (2.0e-9 * sizeof(hc::double_2) * elems) / (mintime / 1e3)
              << " GB/sec" << std::endl;

    return EXIT_SUCCESS;
}
