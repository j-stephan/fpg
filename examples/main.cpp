/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <cstdlib>
#include <iostream>

#include "benchmark.h"

auto main() -> int
{
    try
    {
        benchmark::init();
        benchmark::print_header();
        auto params = benchmark::create_params();
        auto result = benchmark::run(params);
        benchmark::print_result(result);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Unhandled exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
