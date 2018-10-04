#include <cstdlib>

#include "benchmark.h"

auto main() -> int
{
    try{
        benchmark::print_header();
        auto data = benchmark::create_data();
        auto params = benchmark::create_params();
        auto result = benchmark::run(data, parms);
        benchmark::print_result(result);
    }
    catch(const std::exception& e)
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
