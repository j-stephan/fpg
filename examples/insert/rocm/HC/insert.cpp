/* This file is part of fpg.
 *
 * fpg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * fpg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with fpg. If not, see <http://www.gnu.org/licenses/>
 */

// based on NVIDIA's Modern GPU benchmark

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <utility>
#include <typeinfo>
#include <vector>

#include <hc.hpp>

template <typename T>
auto bulk_remove(int count, int iterations) -> void
{
    auto remove_count = count / 2;
    auto keep_count = count - remove_count;

    // randomly remove half of the indices
    auto indices = std::vector<int>(count);
    std::iota(std::begin(indices), std::end(indices), 0);

    // randomly choose one of the unselected integers and swap it to the front
    auto rd = std::random_device{};
    auto rng = std::mt19937{rd()};
    for(auto i = 0; i < count; ++i)
    {
        auto uid = std::uniform_int_distribution{0, count - i - 1};
        auto index = uid(rng);
        if(index < 0)
            continue;
        std::swap(indices[i], indices[i + index]);
    }

    indices.resize(remove_count);
    std::sort(std::begin(indices), std::end(indices));

    // create data
    auto cpu_data = std::vector<T>(count);
    auto cpu_dest = std::vector<T>(keep_count);
    std::iota(std::begin(cpu_data), std::end(cpu_data), 0);

    auto gpu_data = std::vector<T>(count);
    auto gpu_dest = std::vector<T>(keep_count);
    std::iota(std::begin(gpu_data), std::end(gpu_data), 0);

    // create GPU view on data
    auto av_data = hc::array_view<T, 1>{hc::extent<1>{count}, gpu_data};
    auto av_indices = hc::array_view<int, 1>{hc::extent<1>{keep_count},
                                             indices};
    auto av_dest = hc::array_view<T, 1>{hc::extent<1>{keep_count}, gpu_dest};

    // launch kernel: copy the values that are not matched by an index
    auto start = std::chrono::steady_clock::now();
    for(auto i = 0; i < iterations; ++i)
    {
        hc::parallel_for_each(
                av_data.get_extent(),
                [=](hc::index<1> idx) [[hc]] {
                    // TODO
                }
        );
    }
    auto stop = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration<double>{stop - start};

    auto bytes = sizeof(T) * (count + keep_count) + remove_count * sizeof(int);
    auto throughput = count * iterations / dur.count();
    auto bandwidth = bytes * iterations / dur.count(); 

    std::cout << count << ":\t"
              << throughput / 1.0e6
              << "\t" << bandwidth / 1.0e9
              << std::endl;

    // verify
    auto index = 0;
    auto output = 0;
    for(auto input = 0; input < count; ++input)
    {
        auto p = false;
        if(index >= remove_count)
            p = true;
        else
            p = input < indices[index];

        if(p)
            cpu_dest[output++] = cpu_data[input];
        else
            ++index;
    }

    for(int i = 0; i < keep_count; ++i)
    {
        if(cpu_dest[i] != gpu_dest[i])
        {
            std::cerr << "MISMATCH ERROR AT " << i << std::endl;
            std::exit(0);
        }
    }
}

template <typename T>
auto bulk_insert(int count, int iterations) -> void
{
    auto a_count = count / 2;
    auto b_count = count - a_count;

    // create memory
    auto a_count_s = static_cast<std::size_t>(a_count);
    auto b_count_s = static_cast<std::size_t>(b_count);
    auto count_s = static_cast<std::size_t>(count);

    auto a_cpu = std::vector<T>(a_count_s);
    auto b_cpu = std::vector<T>(b_count_s);
    auto dest_cpu = std::vector<T>(count_s);
    auto dest_cpu_verify = std::vector<T>(count_s);

    auto a_gpu = std::vector<T>(a_count_s);
    auto b_gpu = std::vector<T>(b_count_s);
    auto dest_gpu = std::vector<T>(count_s);

    auto indices = std::vector<int>(a_count_s);

    // initialize memory
    auto rd = std::random_device{};
    auto rng = std::mt19937{rd()};
    auto ab_uid = std::uniform_int_distribution{0, count};
    auto id_uid = std::uniform_int_distribution{0, b_count};

    std::generate(std::begin(a_gpu), std::end(a_gpu),
                  [&](){ return ab_uid(rng); });
    std::copy(std::begin(a_gpu), std::end(a_gpu), std::begin(a_cpu));

    std::generate(std::begin(b_gpu), std::end(b_gpu),
                  [&](){ return ab_uid(rng); });
    std::copy(std::begin(b_gpu), std::end(b_gpu), std::begin(b_cpu));

    std::generate(std::begin(indices), std::end(indices),
                  [&](){ return id_uid(rng); });
    std::sort(std::begin(indices), std::end(indices));

    // create GPU view on memory
    auto av_a = hc::array_view<T, 1>{hc::extent<1>{a_count_s}, a_gpu};
    auto av_b = hc::array_view<T, 1>{hc::extent<1>{b_count_s}, b_gpu};
    auto av_indices = hc::array_view<int, 1>{hc::extent<1>{a_count_s}, indices};
    auto av_dest = hc::array_view<T, 1>{hc::extent<1>{count_s}, dest_gpu};

    // launch kernel
    auto start = std::chrono::steady_clock::now();
    for(auto i = 0; i < iterations; ++i)
    {
        hc.parallel_for_each(
                av_dest.get_extent(),
                [=](hc::index<1> idx)
                {
                    // TODO
                }
        );
    }
    auto stop = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration<double>{stop - start};

    auto bytes = (sizeof(int) + 2 * sizeof(T)) * a_count
                 + 2 * sizeof(T) * b_count;
    auto throughput = count * iterations / dur.count();
    auto bandwidth = bytes * iterations / dur.count();

    std::cout << count << ":\t"
              << throughput / 1.0e6 << " M/s\t"
              << bandwidth / 1.0e9 << " GB/s"
              << std::endl;

    // verify
    std::copy(std::begin(dest_gpu), std::end(dest_gpu),
              std::begin(dest_cpu_verify));

    auto a = 0;
    auto b = 0;
    auto output = 0;
    while(output < count)
    {
        auto p = false;
        if(a >= a_count)
            p = false;
        else if(b >= b_count)
            p = true;
        else
            p = (indices[a] <= b);

        if(p)
        {
            dest_cpu[output] = a_cpu[a];
            ++output;
            ++a;
        }
        else
        {
            dest_cpu[output] = b_cpu[b];
            ++output;
            ++b;
        }
    }

    for(auto i = 0; i < count; ++i)
    {
        if(dest_cpu[i] != dest_cpu_verify[i])
        {
            std::cerr << "bulk_insert error at " << i << std::endl;
            std::exit(0);
        }
    }
}

constexpr auto tests = std::array<std::array<int, 2>, 10> {{
                            {{10000, 2000}},
                            {{50000, 2000}},
                            {{100000, 2000}},
                            {{200000, 1000}},
                            {{500000, 500}},
                            {{1000000, 400}},
                            {{2000000, 400}},
                            {{5000000, 400}},
                            {{10000000, 300}},
                            {{20000000, 300}}}};

auto main() -> int
{
    using T1 = int;
    using T2 = std::int64_t;

/*    std::cout << "Benchmarking BulkRemove on type "
        << typeid(T1).name() << std::endl;
    for(const auto& test_set : tests)
        bulk_remove<T1>(test_set[0], test_set[1]);

    std::cout << "Benchmarking BulkRemove on type "
        << typeid(T2).name() << std::endl;
    for(const auto& test_set : tests)
        bulk_remove<T2>(test_set[0], test_set[1]);*/

    std::cout << "Benchmarking bulk_insert on type "
        << typeid(T1).name() << std::endl;
    for(const auto& test_set : tests)
        bulk_insert<T1>(test_set[0], test_set[1]);

    std::cout << "Benchmarking bulk_insert on type "
        << typeid(T2).name() << std::endl;
    for(const auto& test_set : tests)
        bulk_insert<T2>(test_set[0], test_set[1]);

    return 0;
}
