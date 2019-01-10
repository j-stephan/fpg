/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <cstdint>
#include <cstdlib>
#include <cwchar>
#include <exception>
#include <iostream>
#include <iterator>
#include <locale>
#include <string>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
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
        auto&& queue_ = cl::sycl::queue{};
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
        auto exception_handler = [] (cl::sycl::exception_list exceptions)
        {
            for(std::exception_ptr e : exceptions)
            {
                try
                {
                    std::rethrow_exception(e);
                }
                catch(const cl::sycl::exception& err)
                {
                    std::cerr << "Caught asynchronous SYCL exception: "
                              << err.what() << std::endl;
                }
            }
        };
        queue_ = cl::sycl::queue{acc_, exception_handler,
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
        using namespace cl::sycl;

        auto dst_buf = dst.p_impl->data;

        queue_.submit([&] (handler& cgh)
        {
            auto dst_acc = dst_buf.get_access<access::mode::discard_write>(cgh);
            cgh.copy(src.data(), dst_acc);
        });
        queue_.wait_and_throw();
    }

    auto copy_d2h(dev_ptr& src, std::vector<int>& dst) -> void
    {
        using namespace cl::sycl;

        auto src_buf = src.p_impl->data;

        queue_.submit([&] (handler& cgh)
        {
            auto src_acc = src_buf.get_access<access::mode::read>(cgh);
            cgh.copy(src_acc, dst.data());
        });
        queue_.wait_and_throw();
    }

    struct block_reduce
    {
        cl::sycl::accessor<int, 1,
                           cl::sycl::access::mode::read,
                           cl::sycl::access::target::global_buffer> data;
        cl::sycl::accessor<int, 1,
                           cl::sycl::access::mode::write,
                           cl::sycl::access::target::global_buffer> result;
        cl::sycl::accessor<int, 1,
                           cl::sycl::access::mode::read_write,
                           cl::sycl::access::target::local> scratch;

        std::size_t size;

        auto operator()(cl::sycl::nd_item<1> my_item) -> void
        {
            using namespace cl::sycl;

            auto i = my_item.get_global_id(0);

            if(i >= size)
                return;

            // avoid neutral element
            auto tsum = data[i];

            auto grid_size = my_item.get_group_range(0) *
                             my_item.get_local_range(0);
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

            scratch[my_item.get_local_id(0)] = tsum;

            my_item.barrier(access::fence_space::local_space);
            
            auto block_size = my_item.get_local_range(0);
            auto global_id = my_item.get_global_id(0);
            auto local_id = my_item.get_local_id(0);

            // BLOCK + WARP, read from shared memory
            #pragma unroll
            for(auto bs = block_size, bsup = (block_size + 1) / 2;
                bs > 1;
                bs /= 2, bsup = (bs + 1) / 2)
            {
                auto cond = local_id < bsup &&
                            (local_id + bsup) < block_size &&
                            (global_id + bsup) < size;
                if(cond)
                {
                    scratch[local_id] += scratch[local_id + bsup];
                }                          
                my_item.barrier(access::fence_space::local_space);
            }

            // store to global memory
            if(my_item.get_local_id(0) == 0)
                result[my_item.get_group_linear_id()] = scratch[0];
        }
    };

    auto do_benchmark(dev_ptr& data, dev_ptr& result, std::size_t size,
                      int blocks, int block_size) -> void
    {
        using namespace cl::sycl;

        try
        {
            auto data_buf = data.p_impl->data;
            auto result_buf = result.p_impl->data;

            first_event_ = queue_.submit([&] (handler& cgh)
            {
                // first loop
                auto data_acc = data_buf.get_access<access::mode::read,
                                        access::target::global_buffer>(cgh);
                auto result_acc = result_buf.get_access<access::mode::write,
                                        access::target::global_buffer>(cgh);

                auto scratch_acc = accessor<int,
                                            1,
                                            access::mode::read_write,
                                            access::target::local>{
                                            range<1>{block_size * sizeof(int)},
                                            cgh};
                auto reducer = block_reduce{data_acc, result_acc, scratch_acc,
                                            size};

                auto global_size = static_cast<std::size_t>(blocks *
                                                            block_size);
                auto local_size = static_cast<std::size_t>(block_size);
                cgh.parallel_for(nd_range<1>{range<1>{global_size},
                                             range<1>{local_size}},
                                 reducer);
            });

            last_event_ = queue_.submit([&] (handler& cgh)
            {
                // second loop
                auto data_acc = result_buf.get_access<access::mode::read,
                                    access::target::global_buffer>(cgh);
                auto result_acc = result_buf.get_access<access::mode::write,
                                    access::target::global_buffer>(cgh);
                auto scratch_acc = accessor<int,
                                            1,
                                            access::mode::read_write,
                                            access::target::local>{
                                            range<1>{block_size * sizeof(int)},
                                            cgh};

                auto reducer = block_reduce{data_acc, result_acc, scratch_acc, 
                                            static_cast<std::size_t>(blocks)};
                auto global_size = static_cast<std::size_t>(block_size);
                auto local_size = static_cast<std::size_t>(block_size);
                cgh.parallel_for(nd_range<1>{range<1>{global_size},
                                             range<1>{local_size}},
                                 reducer);
            });
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
        last_event_.wait_and_throw();
     
        auto start = first_event_.get_profiling_info<
                        cl::sycl::info::event_profiling::command_start>();
        auto stop = last_event_.get_profiling_info<
                        cl::sycl::info::event_profiling::command_end>();

        // the events return nanoseconds but we want milliseconds
        return static_cast<float>(stop - start) / 1e6;
    }
}
