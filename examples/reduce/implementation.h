/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#ifndef IMPLEMENTATION_H
#define IMPLEMENTATION_H

#include <cstddef>

#include "common/common.h"

namespace impl
{
    auto reduce(dev_handle& handle, const dev_ptr& data, dev_ptr& result,
                std::size_t size, int blocks, int block_size) -> void;
}

#endif
