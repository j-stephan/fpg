#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include <hc.hpp>

constexpr auto N = 1024 * 500;

auto main() -> int
{
    constexpr auto a = 100.f;
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

    // CPU implementation
    for(auto i = 0; i < N; ++i)
        y[i] = a * x[i] + y[i];

    // create array_views for GPU memory
    auto av_x = hc::array_view<float, 1>{N, x};
    auto av_y = hc::array_view<float, 1>{N, y_gpu};

    // launch GPU kernel
    hc::parallel_for_each(hc::extent<1>(N), [=](hc::index<1> i) [[hc]] {
                av_y[i] = a * av_x[i] + av_y[i];
            }
    );

    // verify
    auto errors = 0;
    for(auto i = 0; i < N; ++i) {
        if(std::abs(y[i] - av_y[i]) > std::abs(y[i] * 0.0001f))
            ++errors;
    }

    std::cout << errors << " errors" << std::endl;

    return 0;
}
