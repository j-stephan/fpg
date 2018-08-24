#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <amp.h>

constexpr auto N = 1024 * 500;
constexpr auto a = 100.f;

// len(x) == N
auto algo_cpu(const float* x, float* y) -> std::chrono::duration<double>
{
    auto start = std::chrono::steady_clock::now();
    for(auto i = 0; i < N; ++i)
        y[i] = a * x[i] + y[i];
    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>{stop - start};
}

auto algo_gpu(float* x, float* y) -> std::chrono::duration<double>
{
    auto start = std::chrono::steady_clock::now();

    // create array_view == GPU memory
    auto av_x = Concurrency::array_view<float, 1>{N, x};
    auto av_y = Concurrency::array_view<float, 1>{N, y};

    // launch GPU kernel
    Concurrency::parallel_for_each(Concurrency::extent<1>(N),
                                   [=](Concurrency::index<1> i) restrict(amp) {
                av_y[i] = a * av_x[i] + av_y[i];
            }
    );

    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>{stop - start};

    // no explicit synchronization needed -- the returned future from
    // parallel_for_each synchronized upon its own destruction
}

auto main() -> int
{
    float x[N];
    float y[N];

    // initialize input data
    auto random_gen = std::default_random_engine{};
    auto distribution = std::uniform_real_distribution<float>{-N, N};
    
    std::generate_n(x, N, [&]() { return distribution(random_gen); });
    std::generate_n(y, N, [&]() { return distribution(random_gen); });

    // make a copy for use on GPU
    float y_gpu[N];
    std::copy_n(y, N, y_gpu);

    auto dur_cpu = algo_cpu(x, y);
    auto dur_gpu = algo_gpu(x, y_gpu);

    // verify
    auto errors = 0;
    for(auto i = 0; i < N; ++i) {
        if(std::abs(y[i] - y_gpu[i]) > std::abs(y[i] * 0.0001f))
            ++errors;
    }

    std::cout << errors << " errors" << std::endl;
    std::cout << "CPU version took " << dur_cpu.count() << " s" << std::endl;
    std::cout << "GPU version took " << dur_gpu.count() << " s" << std::endl;

    return 0;
}
