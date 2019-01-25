#include <algorithm>
#include <cstdlib>
#include <cwchar>
#include <iostream>
#include <iterator>
#include <locale>
#include <random>
#include <string>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wc++17-extensions"
#include <hc.hpp>
#pragma clang diagnostic pop

#include <hc_math.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include <hc_short_vector.hpp>
#pragma clang diagnostic pop

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

using float3 = hc::short_vector::float_3;
using float4 = hc::short_vector::float_4;

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
    auto&& converter = std::wstring_convert<cvt>{new cvt{sys_loc.name()}};
}

[[hc]]
auto body_body_interaction(float4 bi, float4 bj, float3 ai) -> float3
{
    constexpr auto eps = 0.001f;
    constexpr auto eps2 = eps * eps;

    // r_ij [3 FLOPS]
    auto r = bj.get_xyz() - bi.get_xyz();

    // dist_sqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    //auto dist_sqr = float{r.x * r.x + r.y * r.y + r.z * r.z} + eps2;
    auto dist_sqr = fmaf(r.x, r.x, fmaf(r.y, r.y, fmaf(r.z, r.z, eps2)));

    // inv_dist_cube = 1/dist_sqr^(3/2) [4 FLOPS]
    // NOTE: rsqrt lives in the global namespace, not the hc namespace
    auto dist_sixth = dist_sqr * dist_sqr * dist_sqr;
    auto inv_dist_cube = rsqrt(dist_sixth);

    // s = m_j * inv_dist_cube [1 FLOP]
    auto s = float{bj.w} * inv_dist_cube;

    // a_i = a_i + s * r_ij [6 FLOPS]
    //ai += r * s;
    ai.x = fmaf(r.x, s, ai.x);
    ai.y = fmaf(r.y, s, ai.y);
    ai.z = fmaf(r.z, s, ai.z);

    return ai;
}

[[hc]]
auto force_calculation(hc::tiled_index<1> idx, float4 body_pos,
                       hc::array_view<const float4, 1> positions,
                       float4* sh_position, // this is tile_static
                       unsigned tiles)
{
    auto acc = float3{0.f, 0.f, 0.f};

    for(auto tile = 0u; tile < tiles; ++tile)
    {
        auto id = tile * idx.tile_dim[0] + idx.local[0];

        sh_position[idx.local[0]] = positions[id];
        idx.barrier.wait_with_tile_static_memory_fence();

        // this loop corresponds to tile_calculation() from GPUGems 3
        #pragma unroll
        for(auto i = 0; i < idx.tile_dim[0]; ++i)
        {
            acc = body_body_interaction(body_pos, sh_position[i], acc);
        }
        idx.barrier.wait_with_tile_static_memory_fence();
    }

    return acc;
}

[[hc]]
auto body_integration(hc::tiled_index<1> idx,
                      hc::array_view<const float4, 1> old_pos,
                      hc::array_view<float4, 1> new_pos,
                      hc::array_view<float4, 1> vel,
                      std::size_t n, unsigned tiles)
-> void
{
    constexpr auto delta_time = 0.2f;
    constexpr auto damping = 0.5f;

    auto sh_position = static_cast<float4*>(
                        hc::get_dynamic_group_segment_base_pointer());

    auto index = static_cast<std::size_t>(idx.global[0]);
    if(index >= n)
        return;

    auto position = old_pos[index];
    auto accel = force_calculation(idx, position, old_pos, sh_position, tiles);

    /*
     * acceleration = force / mass
     * new velocity = old velocity + acceleration * delta_time
     * note that the body's mass is canceled out here and in
     *  body_body_interaction. Thus force == acceleration
     */
    auto velocity = vel[index];

    velocity.set_xyz(velocity.get_xyz() + accel.get_xyz() * delta_time); 
    velocity.set_xyz(velocity.get_xyz() * damping);

    position.set_xyz(position.get_xyz() + velocity.get_xyz() * delta_time);

    new_pos[index] = position;
    vel[index] = velocity;
}

auto main() -> int
{
    auto gen = std::mt19937{std::random_device{}()};
    auto dis = std::uniform_real_distribution<float>{-42.f, 42.f};

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
        return EXIT_FAILURE;
    }

    hc::accelerator::set_default(accelerators[index].get_device_path());

    std::cout << "tile_size;n;time_ms;gflops" << std::endl;

    for(auto n = 2048ul; n <= 524288ul; n *= 2ul)
    {
        auto old_positions = std::vector<float4>{};
        auto new_positions = std::vector<float4>{};
        auto velocities = std::vector<float4>{};

        old_positions.resize(n);
        new_positions.resize(n);
        velocities.resize(n);

        auto init_vec = [&]()
        {
            return float4{dis(gen), dis(gen), dis(gen), 0.f};
        };

        std::generate(begin(old_positions), end(old_positions), init_vec);
        std::fill(begin(new_positions), end(new_positions), float4{});
        std::generate(begin(velocities), end(velocities), init_vec);

        auto d_old_positions = hc::array<float4, 1>{hc::extent<1>{n},
                                                    std::begin(old_positions)};
        auto d_new_positions = hc::array<float4, 1>{hc::extent<1>{n},
                                                    std::begin(new_positions)};
        auto d_velocities = hc::array<float4, 1>{hc::extent<1>{n},
                                                 std::begin(velocities)};

        for(auto block_size = 64u; block_size <= 1024u; block_size *= 2)
        {
            constexpr auto iterations = 10;

            auto tiles = (n + block_size - 1) / block_size;
            auto global_extent = hc::extent<1>{n};

            // The only way to do task graphing in HC is to use the
            // completion_future returned by parallel_for_each. As a result
            // things get really messy when trying to loop
            hc::completion_future futures[iterations];

            auto old_pos_view = hc::array_view<float4, 1>{d_old_positions};
            auto new_pos_view = hc::array_view<float4, 1>{d_new_positions};
            auto velo_view = hc::array_view<float4, 1>{d_velocities};

            // Initial launch
            futures[0] = hc::parallel_for_each(
            global_extent.tile_with_dynamic(
                block_size, block_size * sizeof(float4)),
            [=] (hc::tiled_index<1> idx) [[hc]]
            {
                body_integration(idx, old_pos_view, new_pos_view, velo_view,
                                 n, tiles);
            });

            // Subsequent launches
            for(auto it = 1; it < iterations; ++it)
            {
                futures[it - 1].wait();
                std::swap(old_pos_view, new_pos_view);

                futures[it] = hc::parallel_for_each(
                global_extent.tile_with_dynamic(
                    block_size, block_size * sizeof(float4)),
                [=] (hc::tiled_index<1> idx) [[hc]]
                {
                    body_integration(idx, old_pos_view, new_pos_view,
                                     velo_view, n, tiles);
                });
            }

            auto start_future = futures[0];
            auto stop_future = futures[iterations - 1];

            stop_future.wait();
            
            auto start = start_future.get_begin_tick();
            auto stop = stop_future.get_end_tick();
            auto elapsed = stop - start;

            auto f = stop_future.get_tick_frequency();
            auto fms = static_cast<double>(f) / 1000.f;

            auto time_s = static_cast<double>(elapsed) / f;
            auto time_ms = static_cast<double>(elapsed) / fms;

            constexpr auto flops_per_interaction = 20.;
            auto interactions = static_cast<double>(n * n);
            auto interactions_per_second = interactions * iterations / time_s;
            auto flops = interactions_per_second * flops_per_interaction;
            auto gflops = flops / 1e9;

            std::cout << block_size << ";" << n << ";" << time_ms << ";"
                      << gflops << std::endl;
        }
    }
    return EXIT_SUCCESS;
}
