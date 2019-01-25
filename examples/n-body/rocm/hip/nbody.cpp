#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#pragma clang diagnostic ignored "-Wextra-semi"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#pragma clang diagnostic pop

#define CHECK(cmd) \
{ \
    auto error = cmd; \
    if(error != hipSuccess) \
    { \
        std::cerr << "Error: '" << hipGetErrorString(error) \
                  << "' (" << error << ") at " << __FILE__ << ":" \
                  << __LINE__  << std::endl; \
        return EXIT_FAILURE; \
    } \
}

constexpr auto eps = 0.001f;
constexpr auto eps2 = eps * eps;
constexpr auto damping = 0.5f;
constexpr auto delta_time = 0.2f;
constexpr auto iterations = 10;

__device__ auto body_body_interaction(float4 bi, float4 bj, float3 ai)
-> float3
{
    // r_ij
    auto r = float3{};
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // dist_sqr = dot(r_ij, r_ij) + EPS^2
    // auto dist_sqr = r.x * r.x + r.y * r.y + r.z * r.z + eps2;
    auto dist_sqr = fmaf(r.x, r.x, fmaf(r.y, r.y, fmaf(r.z, r.z, eps2)));

    // inv_dist_cube = 1/dist_sqr^(3/2)
    auto dist_sixth = dist_sqr * dist_sqr * dist_sqr;
    auto inv_dist_cube = rsqrtf(dist_sixth);

    // s = m_j * inv_dist_cube
    auto s = bj.w * inv_dist_cube;

    // a_i = a_i + s * r_ij
    //ai.x += r.x * s;
    ai.x = fmaf(r.x, s, ai.x);
    //ai.y += r.y * s;
    ai.y = fmaf(r.y, s, ai.y);
    //ai.z += r.z * s;
    ai.z = fmaf(r.z, s, ai.z);

    return ai;
}

__device__ auto force_calculation(float4 body_pos,
                                  const float4* positions,
                                  unsigned tiles)
-> float3
{
    extern __shared__ float4 sh_position[];

    auto acc = float3{0.f, 0.f, 0.f};

    for(auto tile = 0u; tile < tiles; ++tile)
    {
        auto idx = tile * hipBlockDim_x + hipThreadIdx_x;

        sh_position[hipThreadIdx_x] = positions[idx];
        __syncthreads();

        // this loop corresponds to tile_calculation() from GPUGems 3
        #pragma unroll
        for(auto i = 0u; i < hipBlockDim_x; ++i)
        {
            acc = body_body_interaction(body_pos, sh_position[i], acc);
        }
        __syncthreads();
    }

    return acc;
}

__global__ void body_integration(const float4* __restrict__ old_pos,
                                 float4* __restrict__ new_pos,
                                 float4* __restrict__ vel,
                                 std::size_t n, unsigned tiles)
{
    auto index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(index >= n)
        return;

    auto position = old_pos[index];
    auto accel = force_calculation(position, old_pos, tiles);

    /*
     * acceleration = force / mass
     * new velocity = old velocity + acceleration * delta_time
     * note that the body's mass is canceled out here and in
     *  body_body_interaction. Thus force == acceleration
     */
    auto velocity = vel[index];

    velocity.x += accel.x * delta_time;
    velocity.y += accel.y * delta_time;
    velocity.z += accel.z * delta_time;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    position.x += velocity.x * delta_time; 
    position.y += velocity.y * delta_time; 
    position.z += velocity.z * delta_time; 

    new_pos[index] = position;
    vel[index] = velocity;
}

auto main() -> int
{
    auto gen = std::mt19937{std::random_device{}()};
    auto dis = std::uniform_real_distribution<float>{-42.f, 42.f};

    auto device_num = 0;
    CHECK(hipGetDeviceCount(&device_num));

    std::cout << "Available accelerators: " << std::endl;
    for(auto i = 0; i < device_num; ++i)
    {
        auto prop = hipDeviceProp_t{};
        CHECK(hipGetDeviceProperties(&prop, i));
        std::cout << "\t[" << i << "] " << prop.name << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Select accelerator: ";
    auto index = 0;
    std::cin >> index;

    if(index >= device_num)
    {
        std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                  << std::endl;
        return EXIT_FAILURE;
    }

    CHECK(hipSetDevice(index));
    CHECK(hipDeviceReset());

    auto prop = hipDeviceProp_t{};
    CHECK(hipGetDeviceProperties(&prop, index));
    auto warp_size = static_cast<unsigned>(prop.warpSize);

    // create stream on device
    auto stream = hipStream_t{};
    CHECK(hipStreamCreate(&stream));

    auto start_event = hipEvent_t{};
    auto stop_event = hipEvent_t{};
    CHECK(hipEventCreate(&start_event));
    CHECK(hipEventCreate(&stop_event));

    std::cout << "block_size;n;time_ms;gflops" << std::endl;

    for(auto n = 1024ul; n <= 524288ul; n *= 2ul)
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

        auto d_old_positions = static_cast<float4*>(nullptr);
        auto d_new_positions = static_cast<float4*>(nullptr);
        auto d_velocities = static_cast<float4*>(nullptr);

        auto bytes = n * sizeof(float4);
        CHECK(hipMalloc(&d_old_positions, bytes));
        CHECK(hipMalloc(&d_new_positions, bytes));
        CHECK(hipMalloc(&d_velocities, bytes));

        CHECK(hipMemcpy(d_old_positions, old_positions.data(), bytes,
                        hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d_new_positions, new_positions.data(), bytes,
                        hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d_velocities, velocities.data(), bytes,
                        hipMemcpyHostToDevice));

        for(auto block_size = warp_size; block_size <= 1024u; block_size *= 2)
        {
            auto tiles = (n + block_size - 1) / block_size;

            CHECK(hipEventRecord(start_event, stream));
            for(auto i = 0; i < iterations; ++i)
            {
                hipLaunchKernelGGL(body_integration,
                                   dim3(tiles), dim3(block_size),
                                   block_size * sizeof(float4), stream,
                                   d_old_positions, d_new_positions,
                                   d_velocities, n, tiles);
                CHECK(hipEventRecord(stop_event, stream));

                CHECK(hipStreamWaitEvent(stream, stop_event, 0));
                std::swap(d_old_positions, d_new_positions);
            }
            CHECK(hipEventSynchronize(stop_event));

            auto time_ms = 0.f;
            CHECK(hipEventElapsedTime(&time_ms, start_event, stop_event));
            auto time_s = time_ms / 1e3;

            constexpr auto flops_per_interaction = 20.;
            auto interactions = static_cast<double>(n * n);
            auto interactions_per_second = interactions * iterations / time_s;
            auto flops = interactions_per_second * flops_per_interaction;
            auto gflops = flops / 1e9;

            std::cout << block_size << ";" << n << ";" << time_ms << ";"
                      << gflops << std::endl;
        }

        CHECK(hipFree(d_velocities));
        CHECK(hipFree(d_new_positions));
        CHECK(hipFree(d_old_positions));

    }

    CHECK(hipEventDestroy(stop_event));
    CHECK(hipEventDestroy(start_event));
    CHECK(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}

