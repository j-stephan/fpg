/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <iostream>

#include "common/common.h"

#define CHECK(cmd) \
{ \
        auto error = cmd; \
        if(error != cudaSuccess) \
        { \
                    std::cerr << "Error: '" << cudaGetErrorString(error) \
                              << "' (" << error << ") at " << __FILE__ << ":" \
                              << __LINE__  << std::endl; \
                    std::exit(EXIT_FAILURE); \
                } \
}

namespace common
{
    struct dev_handle_impl
    {
        cudaStream_t stream;

        ~dev_handle_impl()
        {
            CHECK(cudaStreamDestroy(stream));
        }
    };

    dev_handle::~dev_handle()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    struct dev_ptr_impl
    {
        void* ptr;
        std::size_t size;

        ~dev_ptr_impl()
        {
            if(ptr != nullptr)
                CHECK(cudaFree(ptr));
        }
    };

    dev_ptr::~dev_ptr()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    struct dev_clock_impl
    {
        cudaEvent_t event;
    };

    dev_clock::~dev_clock()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    auto init() -> dev_handle
    {
        // set up default device
        auto dev_count = int{};
        CHECK(cudaGetDeviceCount(&dev_count));

        std::cout << "Available accelerators: " << std::endl;
        for(auto d = 0; d < dev_count; ++d)
        {
            auto prop = cudaDeviceProp{};
            CHECK(cudaGetDeviceProperties(&prop, d));

            std::cout << "\t[" << d << "] " << prop.name << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Select accelerator: ";
        auto index = 0;
        std::cin >> index;

        if(index >= dev_count)
        {
            std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        CHECK(cudaSetDevice(index));
        CHECK(cudaFree(nullptr)); // force context init

        auto stream = cudaStream_t{};
        CHECK(cudaStreamCreate(&stream));
        return dev_handle{new dev_handle_impl{stream}};
    }

    auto get_info() -> info
    {
        auto id = int{};
        CHECK(cudaGetDevice(&id));

        auto prop = cudaDeviceProp{};
        CHECK(cudaGetDeviceProperties(&prop, id));

        auto name = std::string{prop.name};
        auto cc_major = prop.major;
        auto cc_minor = prop.minor;
        auto mem_clock = prop.memoryClockRate / 1000;
        auto clock = prop.clockRate / 1000;
        auto num_sm = prop.multiProcessorCount;
        return info{id, name, cc_major, cc_minor, mem_clock, clock, num_sm};
    }

    auto make_array(std::size_t size) -> dev_ptr
    {
        auto d_ptr = new dev_ptr_impl{nullptr, size};
        CHECK(cudaMalloc(&(d_ptr->ptr), size * sizeof(int)));
        return dev_ptr{d_ptr};
    }

    auto copy_h2d(const std::vector<int>& src, dev_ptr& dst) -> void
    {
        CHECK(cudaMemcpy(dst.p_impl->ptr, src.data(), src.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
    }

    auto copy_d2h(const dev_ptr& src, std::vector<int>& dst) -> void
    {
        CHECK(cudaMemcpy(dst.data(), src.p_impl->ptr,
                        src.p_impl->size * sizeof(int),
                        cudaMemcpyDeviceToHost));
    }

    auto start_clock(dev_handle& handle) -> dev_clock
    {
        auto clock_impl = new dev_clock_impl{};
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventCreate(&(clock_impl->event)));
        CHECK(cudaEventRecord(clock_impl->event, handle.p_impl->stream));
        return dev_clock{clock_impl};
    }

    auto stop_clock(dev_handle& handle) -> dev_clock
    {
        auto clock_impl = new dev_clock_impl{};
        CHECK(cudaEventCreate(&(clock_impl->event)));
        CHECK(cudaEventRecord(clock_impl->event, handle.p_impl->stream));
        CHECK(cudaEventSynchronize(clock_impl->event));
        return dev_clock{clock_impl};
    }

    auto get_duration(const dev_clock& start, const dev_clock& stop) -> float
    {
        auto ms = float{};
        CHECK(cudaGetLastError());
        CHECK(cudaEventElapsedTime(&ms, start.p_impl->event,
                                   stop.p_impl->event));
        return ms;
    }
}
