/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <cstdlib>

#include "benchmark.h"

auto main() -> int
{
    try
    {
        benchmark::print_header();
        auto params = benchmark::create_params();
        auto result = benchmark::run(params);
        benchmark::print_result(result);
    }
    catch(const std::exception& e)
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
