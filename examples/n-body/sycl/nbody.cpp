/* Copyright (c) 2018 - 2019, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>
#include <utility>

#define SYCL_SIMPLE_SWIZZLES
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <CL/sycl.hpp>
#pragma clang diagnostic pop

constexpr auto eps = 0.001f;
constexpr auto eps2 = eps * eps;
constexpr auto damping = 0.5f;
constexpr auto delta_time = 0.2f;
constexpr auto iterations = 10;

using float3 = cl::sycl::float3;
using float4 = cl::sycl::float4;

// FIXME: using this because there is no rsqrt for NVIDIA right now
auto Q_rsqrt(const float number) -> float
{
    // evil floating point bit level hacking, taken from Quake 3 Arena
    const auto x2 = number * 0.5f;
    auto y = number;
    auto i = *(reinterpret_cast<std::int32_t*>(&y));

    i = 0x5f3759df - (i >> 1);

    y = *(reinterpret_cast<float*>(&i));
    y *= 1.5f - (x2 * y * y);

    return y;
}

// we can use this function both on the host and the device
auto body_body_interaction(float4 bi, float4 bj, float3 ai) -> float3
{
    // r_ij [3 FLOPS]
    auto r = bj.xyz() - bi.xyz();

    // dist_sqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    const auto dist_sqr = cl::sycl::fma(r.x(), r.x(), cl::sycl::fma(
                                            r.y(), r.y(), cl::sycl::fma(
                                                r.z(), r.z(), eps2)));

    // inv_dist_cube = 1/dist_sqr^(3/2) [4 FLOPS]
    auto dist_sixth = dist_sqr * dist_sqr * dist_sqr;

    // FIXME: rsqrt currently not supported on NVIDIA devices
    auto inv_dist_cube = Q_rsqrt(dist_sixth);

    // s = m_j * inv_dist_cube [1 FLOP]
    const auto s = float{bj.w()} * inv_dist_cube;
    const auto s3 = float3{s, s, s};

    // a_i = a_i + s * r_ij [6 FLOPS]
    ai = cl::sycl::fma(r, s3, ai);

    return ai;
}

/******************************************************************************
 * Device-side N-Body
 *****************************************************************************/
auto force_calculation_d(cl::sycl::nd_item<1> my_item, float4 body_pos,
                         cl::sycl::accessor<float4, 1,
                            cl::sycl::access::mode::read,
                            cl::sycl::access::target::global_buffer,
                            cl::sycl::access::placeholder::true_t> positions,
                         cl::sycl::accessor<float4, 1,
                            cl::sycl::access::mode::read_write,
                            cl::sycl::access::target::local> sh_position,
                         unsigned tiles)
-> float3
{
    auto acc = float3{0.f, 0.f, 0.f};

    for(auto tile = 0u; tile < tiles; ++tile)
    {
        auto idx = tile * my_item.get_local_range(0) + my_item.get_local_id(0);

        sh_position[my_item.get_local_id()] = positions[idx];
        my_item.barrier(cl::sycl::access::fence_space::local_space);

        // this loop corresponds to tile_calculation() from GPUGems 3
        #pragma unroll 8
        for(auto i = 0u; i < my_item.get_local_range(0); ++i)
        {
            acc = body_body_interaction(body_pos, sh_position[i], acc);
        }
        my_item.barrier(cl::sycl::access::fence_space::local_space);
    }

    return acc;
}

struct body_integrator
{  
    cl::sycl::accessor<float4, 1, cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer,
                       cl::sycl::access::placeholder::true_t> old_pos;

    cl::sycl::accessor<float4, 1, cl::sycl::access::mode::discard_write,
                       cl::sycl::access::target::global_buffer,
                       cl::sycl::access::placeholder::true_t> new_pos;
    
    cl::sycl::accessor<float4, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer,
                       cl::sycl::access::placeholder::true_t> vel;

    cl::sycl::accessor<float4, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local> sh_position;

    std::size_t n;  // bodies
    unsigned tiles;

    auto operator()(cl::sycl::nd_item<1> my_item) -> void
    {
        auto index = my_item.get_global_id(0);
        if(index >= n)
            return;

        auto position = old_pos[index];
        auto accel = force_calculation_d(my_item, position, old_pos,
                                         sh_position, tiles);

        /*
         * acceleration = force / mass
         * new velocity = old velocity + acceleration * delta_time
         * note that the body's mass is canceled out here and in
         *  body_body_interaction. Thus force == acceleration
         */
        auto velocity = vel[index];

        velocity.xyz() += accel.xyz() * delta_time;
        velocity.xyz() *= damping;

        position.xyz() += velocity.xyz() * delta_time; 

        new_pos[index] = position;
        vel[index] = velocity;
    }
};

/******************************************************************************
 * Host-side N-Body
 *****************************************************************************/
auto force_calculation_h(const std::vector<float4>& positions, std::size_t n)
-> std::vector<float3>
{
    auto accels = std::vector<float3>{};
    accels.resize(n);

    #pragma omp parallel for
    for(auto i = 0u; i < n; ++i)
    {
        #pragma unroll 4
        for(auto j = 0u; j < n; ++j)
        {
            body_body_interaction(positions[i], positions[j], accels[i]);
        }
    }

    return accels;
}

auto body_integration_h(std::vector<float4>& old_pos,
                        std::vector<float4>& new_pos,
                        std::vector<float4>& vel,
                        std::size_t n) -> void
{
    for(auto it = 0; it < iterations; ++it)
    {
        auto accels = force_calculation_h(old_pos, n);
       
        #pragma omp parallel for
        for(auto i = 0u; i < n; ++i)
        {
            auto position = old_pos[i];
            auto accel = accels[i];;

            /*
             * acceleration = force / mass
             * new velocity = old velocity + acceleration * delta_time
             * note that the body's mass is canceled out here and in
             *  body_body_interaction. Thus force == acceleration
             */
            auto velocity = vel[i];

            velocity.xyz() += accel.xyz() * delta_time;
            velocity.xyz() *= damping;

            position.xyz() += velocity.xyz() * delta_time; 

            new_pos[i] = position;
            vel[i] = velocity;
        }

        std::swap(old_pos, new_pos);
    }
}

auto main() -> int
{
    try
    {
        auto gen = std::mt19937{std::random_device{}()};
        auto dis = std::uniform_real_distribution<float>{-42.f, 42.f};

        auto platforms = cl::sycl::platform::get_platforms();
        std::cout << "Available platforms: " << std::endl;
        for(auto i = 0u; i < platforms.size(); ++i)
        {
            auto&& p = platforms[i];
            auto vendor = p.get_info<cl::sycl::info::platform::vendor>();
            auto name = p.get_info<cl::sycl::info::platform::name>();
            std::cout << "\t[" << i << "] Vendor: " << vendor << ", "
                      << "name: " << name << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Select platform: ";
        auto index = 0u;
        std::cin >> index;

        if(index >= platforms.size())
        {
            std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                      << std::endl;
            return EXIT_FAILURE;
        }

        auto&& platform = platforms[index];

        // set up default context
        auto ctx = cl::sycl::context{platform};

        // set up default accelerator
        auto accelerators = ctx.get_devices();
        std::cout << "Available accelerators: " << std::endl;
        for(auto i = 0u; i < accelerators.size(); ++i)
        {
            auto&& acc = accelerators[i];
            auto vendor = acc.get_info<cl::sycl::info::device::vendor>(); 
            auto name = acc.get_info<cl::sycl::info::device::name>();
            std::cout << "\t[" << i << "] Vendor: " << vendor << ", "
                      << "name: " << name << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Select accelerator: ";
        std::cin >> index;

        if(index >= accelerators.size())
        {
            std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                      << std::endl;
            return EXIT_FAILURE;
        }

        auto acc = accelerators[index];

        // create queue on device
        auto exception_handler = [] (cl::sycl::exception_list exceptions)
        {
            for(std::exception_ptr e : exceptions)
            {
                try
                {
                    std::rethrow_exception(e);
                }
                catch(const cl::sycl::exception& err)
                {                    
                    std::cerr << "Caught asynchronous SYCL exception: "
                              << err.what() << std::endl;
                }
            }
        };
    
        auto queue = cl::sycl::queue{acc, exception_handler,
                        cl::sycl::property::queue::enable_profiling{}};

        // Precompile Kernel to reduce overhead
        auto program = cl::sycl::program{queue.get_context()};
        program.build_with_kernel_type<body_integrator>();

        std::cout << "Host program: " << program.is_host() << std::endl;
        std::cout << "Compile options: " << program.get_compile_options() << std::endl;
        std::cout << "Link options: " << program.get_link_options() << std::endl;
        std::cout << "Build options: " << program.get_build_options() << std::endl;
        std::cout << "Program state: ";

        auto state = program.get_state();
        if(state == cl::sycl::program_state::none)
            std::cout << " none";
        else if(state == cl::sycl::program_state::compiled)
            std::cout << " compiled";
        else if(state == cl::sycl::program_state::linked)
            std::cout << " linked";
        std::cout << std::endl;

        for(auto n = 2048ul; n <= 524288ul; n *= 2ul)
        {
            auto old_positions = std::vector<float4>{};
            auto new_positions = std::vector<float4>{};
            auto positions_cmp = std::vector<float4>{}; // for verification
            auto velocities = std::vector<float4>{};

            old_positions.resize(n);
            new_positions.resize(n);
            positions_cmp.resize(n);
            velocities.resize(n);

            auto init_vec = [&]()
            {
                return float4{dis(gen), dis(gen), dis(gen), 0.f};
            };

            std::generate(begin(old_positions), end(old_positions), init_vec);
            std::fill(begin(new_positions), end(new_positions), float4{});
            std::fill(begin(positions_cmp), end(positions_cmp), float4{});
            std::generate(begin(velocities), end(velocities), init_vec);

            auto d_old_positions = cl::sycl::buffer<float4, 1>{
                cl::sycl::range<1>{old_positions.size()}};
            d_old_positions.set_write_back(false);

            auto d_new_positions = cl::sycl::buffer<float4, 1>{
                cl::sycl::range<1>{new_positions.size()}};
            d_new_positions.set_write_back(false);

            auto d_velocities = cl::sycl::buffer<float4, 1>{
                cl::sycl::range<1>{velocities.size()}};
            d_velocities.set_write_back(false);

            queue.submit([&](cl::sycl::handler& cgh)
            {
                auto acc = d_old_positions.get_access<
                    cl::sycl::access::mode::discard_write,
                    cl::sycl::access::target::global_buffer>(cgh);

                cgh.copy(old_positions.data(), acc);
            });

            queue.submit([&](cl::sycl::handler& cgh)
            {
                auto acc = d_new_positions.get_access<
                    cl::sycl::access::mode::discard_write,
                    cl::sycl::access::target::global_buffer>(cgh);

                cgh.copy(new_positions.data(), acc);
            });

            queue.submit([&](cl::sycl::handler& cgh)
            {
                auto acc = d_velocities.get_access<
                    cl::sycl::access::mode::discard_write,
                    cl::sycl::access::target::global_buffer>(cgh);

                cgh.copy(velocities.data(), acc);
            });

            /*
               we only need the CPU result once per different n, so launch here
               and hope computation is done once we need it
            */
            auto verification = std::async(std::launch::async,
                                           body_integration_h,
                                           std::ref(old_positions),
                                           std::ref(new_positions),
                                           std::ref(velocities), n);                

            for(auto block_size = 32u; block_size <= 1024u; block_size *= 2u)
            {
                auto tiles = (static_cast<unsigned>(n) + block_size - 1)
                             / block_size;
                
                auto first_event = cl::sycl::event{};
                auto last_event = cl::sycl::event{};

                for(auto i = 0; i < iterations; ++i)
                {
                    last_event = queue.submit(
                    [&, n, tiles](cl::sycl::handler& cgh)
                    {
                        auto sh_position = cl::sycl::accessor<
                            float4, 1,
                            cl::sycl::access::mode::read_write,
                            cl::sycl::access::target::local>{
                                cl::sycl::range<1>{block_size},
                                cgh};

                        auto integrator = body_integrator{
                                .sh_position = sh_position,
                                .n = n,
                                .tiles = tiles};

                        cgh.require(d_old_positions, integrator.old_pos);
                        cgh.require(d_new_positions, integrator.new_pos);
                        cgh.require(d_velocities, integrator.vel);

                        cgh.parallel_for(
                            program.get_kernel<body_integrator>(),
                            cl::sycl::nd_range<1>{
                                cl::sycl::range<1>{n},
                                cl::sycl::range<1>{block_size}},
                                integrator
                        );
                    });

                    if(i == 0)
                        first_event = last_event;

                    std::swap(d_old_positions, d_new_positions);
                }
                queue.wait_and_throw();

                constexpr auto even_iterations = ((iterations % 2) == 0);
                auto copy_src = even_iterations ? d_old_positions :
                                                  d_new_positions;

                queue.submit([&](cl::sycl::handler& cgh)
                {
                    auto acc = copy_src.get_access<
                        cl::sycl::access::mode::read,
                        cl::sycl::access::target::global_buffer>(cgh);

                    cgh.copy(acc, positions_cmp.data());
                });
                queue.wait_and_throw();

                // verify
                verification.wait();
                auto&& cmp_vec = even_iterations ? old_positions :
                                                   new_positions;
                auto cmp = std::mismatch(std::begin(cmp_vec), std::end(cmp_vec),
                                         std::begin(positions_cmp),
                                         [](const float4& a, const float4& b)
                                         {
                                            using namespace cl::sycl;
                                            constexpr auto err = 1e-2;
                                            auto diff = fabs(a.xyz() - b.xyz());
                                            auto x = float{diff.x()} <= err;
                                            auto y = float{diff.y()} <= err;
                                            auto z = float{diff.z()} <= err;
                                            return x & y & z;
                                         });

                if(cmp.first != std::end(cmp_vec))
                {
                    std::cerr << "Mismatch: {" << cmp.first->x() << ", "
                                               << cmp.first->y() << ", "
                                               << cmp.first->z() << "} != {"
                                               << cmp.second->x() << ", "
                                               << cmp.second->y() << ", "
                                               << cmp.second->z() << "}"
                                               << std::endl;
                    return EXIT_FAILURE;
                }

                auto start = first_event.get_profiling_info<
                    cl::sycl::info::event_profiling::command_start>();
                auto stop = last_event.get_profiling_info<
                    cl::sycl::info::event_profiling::command_end>();

                auto time_ns = stop - start;
                auto time_s = time_ns / 1e9;
                auto time_ms = time_ns / 1e6;

                constexpr auto flops_per_interaction = 20.;
                auto interactions = static_cast<double>(n * n);
                auto interactions_per_second = interactions * iterations 
                                               / time_s;
                auto flops = interactions_per_second * flops_per_interaction;
                auto gflops = flops / 1e9;

                std::cout << block_size << ";" << n << ";" << time_ms << ";"
                          << gflops << std::endl;
            }
        }
    }
    catch(const cl::sycl::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
