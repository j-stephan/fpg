/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <cstddef>

namespace benchmark
{
    struct params
    {
        std::size_t first_size;
        std::size_t last_size;
        unsigned int iterations;
    };
}
