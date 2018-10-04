/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <iomanip>
#include <iostream>
#include <tuple>

#include "benchmark.h"
#include "params.h"
#include "result.h"

namespace benchmark
{
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
        // call specific implementation from here
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
                      << min_time << ","
                      << max_throughput << " GB/s" << std::endl;
        }
    }
}

