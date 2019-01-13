/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include "common/common.h"

#include "benchmark.h"
#include "implementation.h"
#include "params.h"
#include "result.h"

namespace benchmark
{
    namespace
    {
        auto handle_ = dev_handle{};
    }

    auto init() -> void
    {
        handle_ = common::init();
    }

    auto print_header() -> void
    {
        std::cout << "index,"
                  << "dev_id,"
                  << "dev_name,"
                  << "dev_CC,"
                  << "dev_memClock,"
                  << "dev_clock,"
                  << "n,"
                  << "numSMs,"
                  << "blocks_i,"
                  << "blocks_i/numSMs,"
                  << "blocks_n,"
                  << "TBlockSize,"
                  << "TRuns,"
                  << "min_time,"
                  << "max_throughput" << std::endl;
    }

    auto create_params() -> params
    {
        return params{1024u, (1024u << 17u), 5};
    }

    auto run(const params& p) -> result
    {
        auto info = common::get_info();

        auto r = result{};

        for(auto s = p.first_size; s <= p.last_size; s *= 2)
        {
            for(auto block_size = 64; block_size <= 1024; block_size *= 2)
            {
                auto blocks_i = info.num_sm;
                auto blocks_n = ((((s + 1) / 2) - 1) / block_size) + 1;
                auto i = 0;

                // GRIDSIZE LOOP
                do
                {
                    blocks_i *= 2; // starting with 2 * num_sm blocks per grid
                    if(static_cast<unsigned>(blocks_i) > blocks_n)
                        blocks_i = blocks_n;

                    // create data
                    auto data = std::vector<int>{};
                    data.resize(s);

                    // initialize data
                    auto&& rd = std::random_device{};
                    auto rng = std::mt19937{rd()};
                    auto uid = std::uniform_int_distribution<>{0, 3};
                    std::generate(std::begin(data), std::end(data), [&]()
                    {
                        return uid(rng);
                    });

                    auto result = std::vector<int>{};
                    result.resize(blocks_i); 
                    std::fill(std::begin(result), std::end(result), 0);

                    // run benchmark
                    auto min_time = std::numeric_limits<float>::max(); 

                    // own scope to satisfy SYCL
                    {
                        // copy to accelerator
                        auto data_gpu = common::make_array(data.size());
                        auto result_gpu = common::make_array(result.size());
                        common::copy_h2d(data, data_gpu);
                        common::copy_h2d(result, result_gpu);

                        // warm up
                        impl::reduce(handle_, data_gpu, result_gpu,
                                     s, blocks_i, block_size);

                        for(auto j = 0u; j < p.iterations; ++j)
                        {
                            auto start = common::start_clock();
                            impl::reduce(handle_, data_gpu, result_gpu,
                                         s, blocks_i, block_size);
                            auto stop = common::stop_clock();
                            auto dur = common::get_duration(start, stop);
                            min_time = std::min(min_time, dur);
                        }

                        // verify
                        common::copy_d2h(result_gpu, result);
                    }

                    auto verify = std::accumulate(std::begin(data),
                                                  std::end(data), 0);
                    if(verify != result[0])
                    {
                        std::cerr << "s = " << s << std::endl;
                        std::cerr << "blocks_i = " << blocks_i << std::endl;
                        std::cerr << "blocks_n = " << blocks_n << std::endl;
                        std::cerr << "block_size = " << block_size << std::endl;
                        std::cerr << "i = " << i << std::endl;
                        std::cerr << "Mismatch: expected " << verify
                                  << ", got: " << result[0] << std::endl;
                        std::exit(EXIT_FAILURE);
                    }

                    auto bytes = s * sizeof(int);
                    // This is actually the bandwidth
                    auto max_throughput = bytes / min_time * 1e-6;

                    r.tuples.emplace_back(std::make_tuple(
                        i++, info.id, info.name, info.cc_major, info.cc_minor,
                        info.mem_clock, info.clock, s, info.num_sm, blocks_i,
                        blocks_n, block_size, p.iterations, min_time,
                        max_throughput));

                } while(static_cast<unsigned>(blocks_i) < blocks_n);
            }
        }
        return r;
    }

    auto print_result(const result& r) -> void
    {
        for(const auto& t : r.tuples)
        {
            // This would look much nicer in C++17 but CUDA prevents this
            auto index = int{};
            auto dev_id = int{};
            auto dev_name = std::string{};
            auto dev_CC_major = int{};
            auto dev_CC_minor = int{};
            auto dev_memClock = int{};
            auto dev_clock = int{};
            auto n = int{};
            auto numSMs = int{};
            auto blocks_i = int{};
            auto blocks_n = int{};
            auto TBlockSize = int{};
            auto TRuns = int{};
            auto min_time = float{};
            auto max_throughput = float{};

            std::tie(index, dev_id, dev_name, dev_CC_major, dev_CC_minor,
                     dev_memClock, dev_clock, n, numSMs, blocks_i, blocks_n,
                     TBlockSize, TRuns, min_time, max_throughput) = t;
             
            std::cout << std::setw(3)
                      << index << ","
                      << dev_id << ","
                      << dev_name << ","
                      << dev_CC_major << "." << dev_CC_minor << ","
                      << dev_memClock << ","
                      << dev_clock << ","
                      << n << ","
                      << numSMs << ","
                      << blocks_i << ","
                      << blocks_i / numSMs << ","
                      << blocks_n << ","
                      << TBlockSize << ","
                      << TRuns << ","
                      << min_time << " ms,"
                      << max_throughput << " GB/s" << std::endl;
        }
    }
}

