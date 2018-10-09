/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#ifndef RESULT_H
#define RESULT_H

#include <string>
#include <tuple>
#include <vector>

namespace benchmark
{
    struct result
    {
        std::vector<std::tuple<int, // index
                               int, // dev_id
                               std::string, // dev_name
                               int, int, // dev_CC
                               int, // dev_memClock
                               int, // dev_clock
                               int, // n
                               int, // numSMs
                               int, // blocks_i
                               int, // blocks_n
                               int, // TBlockSize
                               int, // TRuns
                               float, // min_time
                               float // max_throughput
                               >> tuples;
    };
}

#endif
