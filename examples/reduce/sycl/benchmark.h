/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "params.h"
#include "result.h"

namespace benchmark
{
    auto init() -> void;
    auto print_header() -> void;
    auto create_params() -> params;
    auto run(const params& p) -> result;
    auto print_result(const result& r) -> void;
}

#endif
