/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <cstdint>
#include <cstdlib>
#include <cwchar>
#include <iostream>
#include <iterator>
#include <locale>
#include <string>
#include <vector>

#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic push
#include <CL/sycl.hpp>
#pragma clang diagnostic pop

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "accelerator.h"

namespace acc
{
    namespace
    {
        auto ctx_ = cl::sycl::context{};
        auto acc_ = cl::sycl::device{};
        auto queue_ = cl::sycl::queue{};
        auto first_event_ = cl::sycl::event{};
        auto last_event_ = cl::sycl::event{};
    }

    struct dev_ptr_impl
    {
        cl::sycl::buffer<int, 1> data;
    };

    dev_ptr::~dev_ptr()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    struct dev_clock_impl
    {
    };

    dev_clock::~dev_clock()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    auto init() -> void
    {
        // set up default platform
        auto platforms = cl::sycl::platform::get_platforms();
        std::cout << "Available platforms: " << std::endl;
        for(auto i = 0u; i < platforms.size(); ++i)
        {
            auto&& p = platforms[i];
            auto vendor = p.get_info<cl::sycl::info::platform::vendor>();
            auto name = p.get_info<cl::sycl::info::platform::name>();
            std::cout << "\t[" << i << "] Vendor: " << vendor << ", "
                      << "name: " << name << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Select platform: ";
        auto index = 0u;
        std::cin >> index;

        if(index >= platforms.size())
        {
            std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        auto&& platform = platforms[index];

        // set up default context
        ctx_ = cl::sycl::context{platform};

        // set up default accelerator
        auto accelerators = ctx_.get_devices();
        std::cout << "Available accelerators: " << std::endl;
        for(auto i = 0u; i < accelerators.size(); ++i)
        {
            auto&& acc = accelerators[i];
            auto vendor = acc.get_info<cl::sycl::info::device::vendor>(); 
            auto name = acc.get_info<cl::sycl::info::device::name>();
            std::cout << "\t[" << i << "] Vendor: " << vendor << ", "
                      << "name: " << name << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Select accelerator: ";
        std::cin >> index;

        if(index >= accelerators.size())
        {
            std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        acc_ = accelerators[index];

        // create queue on device
        queue_ = cl::sycl::queue{acc_,
                                 cl::sycl::property::queue::enable_profiling{}};
    }

    auto get_info() -> info
    {
        // can't be queried with SYCL 1.2
        auto device_index = 0;

        auto device_name = std::string{
                            acc_.get_info<cl::sycl::info::device::name>()};

        auto cc_major = 1;
        auto cc_minor = 0;

        auto is_nvidia = acc_.has_extension("cl_nv_device_attribute_query");
        if(is_nvidia)
        {
            auto uint_cc_major = 0u;
            auto uint_cc_minor = 0u;

            auto opencl_device_id = acc_.get();
            auto opencl_device = cl::Device{opencl_device_id};

            opencl_device.getInfo(CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV,
                                  &uint_cc_major);
            opencl_device.getInfo(CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV,
                                  &uint_cc_minor);

            cc_major = static_cast<int>(uint_cc_major);
            cc_minor = static_cast<int>(uint_cc_minor);
        }
        else
        {
            auto has_cc11 = false;
            {
                // OpenCL 1.2 supports global atomics for int32
                // OpenCL 1.2 supports global atomic_xchg for float
                has_cc11 = true;
                cc_minor = 1;
            }

            auto has_cc12 = false;
            {
                if(has_cc11)
                {
                    // OpenCL 1.2 supports shared atomics for int32
                    // OpenCL 1.2 supports shared atomic_xchg for float
                    auto has_global_int64_atomics =
                        acc_.has_extension("cl_khr_int64_base_atomics") &&
                        acc_.has_extension("cl_khr_int64_extended_atomics");

                    // not checking for warp vote since we don't have intrinsics
                    if(has_global_int64_atomics)
                    {
                        has_cc12 = true;
                        cc_minor = 2;
                    }
                }
            }

            auto has_cc13 = false;
            {
                if(has_cc12)
                {
                    // check for double support
                    if(acc_.has_extension("cl_khr_fp64"))
                    {
                        has_cc13 = true;
                        cc_minor = 3;
                    }
                }

            }

            // nothing more than CC 1.3 is supported by OpenCL 1.2
        }

        // can't be queried with SYCL 1.2
        auto mem_clock = 0;

        auto clock = static_cast<int>(
                            acc_.get_info<
                                cl::sycl::info::device::max_clock_frequency>());

        auto num_sm = static_cast<int>(
                            acc_.get_info<
                                cl::sycl::info::device::max_compute_units>());

        return info {device_index, device_name, cc_major, cc_minor, mem_clock,
                     clock, num_sm};
    }

    auto make_array(std::size_t size) -> dev_ptr
    {
        using namespace cl::sycl;
        return dev_ptr{new dev_ptr_impl{buffer<int>{range<1>{size}}}};
    }

    auto copy_h2d(const std::vector<int>& src, dev_ptr& dst) -> void
    {
        auto&& dst_buf = dst.p_impl->data;
        queue_.submit([&] (cl::sycl::handler& cgh)
        {
            auto accessor = dst_buf.get_access<
                                cl::sycl::access::mode::discard_write>();

            cgh.copy(src.data(), accessor);
        });
    }

    auto copy_d2h(dev_ptr& src, std::vector<int>& dst) -> void
    {
        auto&& src_buf = src.p_impl->data;
        queue_.submit([&] (cl::sycl::handler& cgh)
        {
            auto accessor = src_buf.get_access<
                                cl::sycl::access::mode::read>();

            cgh.copy(accessor, dst.data());
        });
    }

    struct block_reduce
    {
        cl::sycl::accessor<int, 1,
                           cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer> data;
        cl::sycl::accessor<int, 1,
                           cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer> result;

        std::size_t size;
        int blocks;
        int block_size;

        auto operator()(cl::sycl::group<1> work_group) -> void
        {
            // shared memory is defined per work_group
            int scratch[1024];

            work_group.parallel_for_work_item(
                [&] (cl::sycl::h_item<1> work_item)
            {
                auto i = work_item.get_global_id().get(0);
                if(i > size)
                    return;

                // avoid neutral element
                auto tsum = data[i];

                auto grid_size = work_item.get_global_range().get(0);
                i += grid_size;

                // GRID, read from global memory
                while((i + 3 * grid_size) < size)
                {
                    tsum += data[i] + data[i + grid_size] +
                            data[i + 2 * grid_size] + data[i + 3 * grid_size];
                    i += 4 * grid_size;
                }

                // tail
                while(i < size)
                {
                    tsum += data[i];
                    i += grid_size;
                }

                scratch[work_item.get_local_id().get(0)] = tsum;
            });

            // implicit barrier

            auto block_id = work_group.get_id().get(0);
            work_group.parallel_for_work_item(
                [&] (cl::sycl::h_item<1> work_item)
            {
                auto block_size = work_item.get_local_range().get(0);
                auto local_id = work_item.get_local_id().get(0);
                // BLOCK + WARP, read from shared memory
                #pragma unroll
                for(auto bs = block_size, bsup = (block_size + 1) / 2;
                    bs > 1;
                    bs /= 2, bsup = (bs + 1) / 2)
                {
                    auto cond = local_id < bsup &&
                                (local_id + bsup) < block_size &&
                                (block_id * block_size + local_id + bsup)
                                    < size;
                    if(cond)
                    {
                        scratch[local_id] += scratch[local_id + bsup];
                    }
                                
                }
            });

            // implicit barrier

            work_group.parallel_for_work_item(
                [&] (cl::sycl::h_item<1> work_item)
            {
                // store to global memory
                if(work_item.get_local_id().get(0) == 0)
                    result[block_id] = scratch[0];
            });
        }
    };

    auto do_benchmark(dev_ptr& data, dev_ptr& result, std::size_t size,
                      int blocks, int block_size) -> void
    {
        try
        {
            auto& data_buf = data.p_impl->data;
            auto& result_buf = result.p_impl->data;

            first_event_ = queue_.submit([&] (cl::sycl::handler& cgh)
            {
                // first loop
                auto blocks_range = cl::sycl::range<1>{
                                        static_cast<std::size_t>(blocks)};
                auto block_size_range = cl::sycl::range<1>{
                                        static_cast<std::size_t>(block_size)};

                auto data_acc = data_buf.get_access<
                                        cl::sycl::access::mode::read,
                                        cl::sycl::access::target::global_buffer>
                                        (cgh);
                auto result_acc = result_buf.get_access<
                                        cl::sycl::access::mode::write,
                                        cl::sycl::access::target::global_buffer>
                                        (cgh);

                auto reducer = block_reduce{data_acc, result_acc, size, blocks,
                                            block_size};
                cgh.parallel_for_work_group(blocks_range, block_size_range,
                                            reducer);
            });

            /*last_event_ = queue_.submit([&] (cl::sycl::handler& cgh)
            {
                // second loop
                auto blocks_range = cl::sycl::range<1>{
                                        static_cast<std::size_t>(1)};
                auto block_size_range = cl::sycl::range<1>{
                                        static_cast<std::size_t>(block_size)};

                auto data_acc = result_buf.get_access<
                                    cl::sycl::access::mode::read,
                                    cl::sycl::access::target::global_buffer>
                                    (cgh);
                auto result_acc = result_buf.get_access<
                                    cl::sycl::access::mode::write,
                                    cl::sycl::access::target::global_buffer>
                                    (cgh);

                auto reducer = block_reduce{data_acc, result_acc,
                                            static_cast<std::size_t>(blocks), 1,
                                            block_size};
                cgh.parallel_for_work_group(blocks_range, block_size_range,
                                            reducer);
            });*/
        }
        catch(const cl::sycl::exception& err)
        {
            std::cerr << err.what() << std::endl;
            if(err.get_cl_code() != CL_SUCCESS)
                std::cerr << err.get_cl_code() << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    auto start_clock() -> dev_clock
    {
        // not needed with SYCL
        return dev_clock{nullptr};
    }

    auto stop_clock() -> dev_clock
    {
        // not needed with SYCL
        return dev_clock{nullptr};
    }

    auto get_duration(const dev_clock& /*start*/, const dev_clock& /*stop*/)
    -> float
    {
        last_event_.wait();
     
        auto start = first_event_.get_profiling_info<
                        cl::sycl::info::event_profiling::command_start>();
        auto stop = last_event_.get_profiling_info<
                        cl::sycl::info::event_profiling::command_end>();

        // the events return nanoseconds but we want milliseconds
        return static_cast<float>(stop - start) / 10e6;
    }
}
