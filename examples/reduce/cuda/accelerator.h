/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#ifndef ACCELERATOR_H
#define ACCELERATOR_H

#include <string>
#include <vector>

namespace acc
{
    struct info
    {
        int id;
        std::string name;
        int cc_major;
        int cc_minor;
        int mem_clock;
        int clock;
        int num_sm;
    };

    struct dev_ptr_impl;
    struct dev_ptr
    {
        ~dev_ptr(); 
        dev_ptr_impl* p_impl;
    };

    struct dev_clock_impl;
    struct dev_clock
    {
        ~dev_clock();
        dev_clock_impl* p_impl;
    };

    auto init() -> void;
    auto get_info() -> info;

    auto make_array(std::size_t size) -> dev_ptr;

    auto copy_h2d(const std::vector<int>& src, dev_ptr& dst) -> void;
    auto copy_d2h(const dev_ptr& src, std::vector<int>& dst) -> void;
    auto do_benchmark(const dev_ptr& data, dev_ptr& result, std::size_t size,
                      int blocks, int block_size) -> void;

    auto start_clock() -> dev_clock;
    auto stop_clock() -> dev_clock;
    auto get_duration(const dev_clock& start, const dev_clock& stop) -> float;
}

#endif
