#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <CL/sycl.hpp>
#pragma clang diagnostic pop

struct reader_writer
{
    cl::sycl::accessor<cl::sycl::float4, 1,
                       cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer> A;
    cl::sycl::accessor<cl::sycl::float4, 1,
                       cl::sycl::access::mode::discard_write,
                       cl::sycl::access::target::global_buffer> B;
    std::size_t elems;

    auto operator()(cl::sycl::nd_item<1> my_item) -> void
    {
        auto stride = my_item.get_group_range(0) * my_item.get_local_range(0);
        for(auto i = my_item.get_global_id(0); i < elems; i += stride)
        {
            B[i] = A[i];
        }
    }
};

struct writer
{
    cl::sycl::accessor<cl::sycl::float4, 1,
                       cl::sycl::access::mode::discard_write,
                       cl::sycl::access::target::global_buffer> B;
    std::size_t elems;

    auto operator()(cl::sycl::nd_item<1> my_item) -> void
    {
        auto stride = my_item.get_group_range(0) * my_item.get_local_range(0);
        for(auto i = my_item.get_global_id(0); i < elems; i += stride)
        {
            B[i] = cl::sycl::float4{0.f, 0.f, 0.f, 0.f};
        }
    }
};

auto do_benchmark(cl::sycl::queue& queue, cl_uint cus, cl_uint max_groups,
                  std::ofstream& file, cl_uint start_size, cl_uint stop_size)
-> void
{
    constexpr auto iters = 10;
    constexpr auto max_mem = 1u << 31;
    constexpr auto max_elems = max_mem / sizeof(cl::sycl::float4);

    for(auto group_size = start_size; group_size <= stop_size; group_size *= 2)
    {
        for(auto elems = group_size * cus; elems <= max_elems; elems *= 2)
        {
            // Allocate memory on device
            auto A_d = cl::sycl::buffer<cl::sycl::float4, 1>{
                                                    cl::sycl::range<1>{elems}};
            auto B_d = cl::sycl::buffer<cl::sycl::float4, 1>{
                                                    cl::sycl::range<1>{elems}};
            
            for(auto group_num = cus;
                     group_num <= std::min(elems / group_size, max_groups);
                     group_num *= 2)
            {
                // Initialize device memory
                queue.submit([&] (cl::sycl::handler& cgh)
                {
                    auto A = A_d.get_access<
                                cl::sycl::access::mode::discard_write,
                                cl::sycl::access::target::global_buffer>(cgh);

                    cgh.fill(A, cl::sycl::float4{}); 
                });

                queue.submit([&] (cl::sycl::handler& cgh)
                {
                    auto B = B_d.get_access<
                                cl::sycl::access::mode::discard_write,
                                cl::sycl::access::target::global_buffer>(cgh);

                    cgh.fill(B, cl::sycl::float4{cl::sycl::nan(0u),
                                                 cl::sycl::nan(0u),
                                                 cl::sycl::nan(0u),
                                                 cl::sycl::nan(0u)});
                });
                queue.wait_and_throw();

                auto mintime = std::numeric_limits<float>::max();
                for(auto k = 0; k < iters; ++k)
                {
                    auto event = queue.submit([&] (cl::sycl::handler& cgh)
                    {
                        auto A = A_d.get_access<
                                cl::sycl::access::mode::read,        
                                cl::sycl::access::target::global_buffer>(cgh);

                        auto B = B_d.get_access<
                                cl::sycl::access::mode::discard_write,
                                cl::sycl::access::target::global_buffer>(cgh);

                        auto read_write = reader_writer{A, B, elems};
                        cgh.parallel_for(cl::sycl::nd_range<1>{
                                    cl::sycl::range<1>{group_num * group_size},
                                    cl::sycl::range<1>{group_size}},
                                         read_write);
                    });
            
                    event.wait_and_throw();
                    auto start = event.get_profiling_info<
                            cl::sycl::info::event_profiling::command_start>();
                    auto stop = event.get_profiling_info<
                            cl::sycl::info::event_profiling::command_end>();

                    auto elapsed = static_cast<float>(stop - start);
                    mintime = std::min(mintime, elapsed);
                }

                file << "RW;" << group_size << ";" << group_num << ";"
                     << sizeof(cl::sycl::float4) << ";" << elems << ";"
                     << mintime / 1e6 << ";"
                     << (2.0e-9*sizeof(cl::sycl::float4) * elems)/(mintime/1e9)
                     << std::endl;
            }

            file << std::endl;

            for(auto group_num = cus;
                     group_num <= std::min(elems / group_size, max_groups);
                     group_num *= 2)
            {
                // Initialize device memory
                queue.submit([&] (cl::sycl::handler& cgh)
                {
                    auto B = B_d.get_access<
                                cl::sycl::access::mode::discard_write,
                                cl::sycl::access::target::global_buffer>(cgh);

                    cgh.fill(B, cl::sycl::float4{cl::sycl::nan(0u),
                                                 cl::sycl::nan(0u),
                                                 cl::sycl::nan(0u),
                                                 cl::sycl::nan(0u)});
                });
                queue.wait_and_throw();

                auto mintime = std::numeric_limits<float>::max();
                for(auto k = 0; k < iters; ++k)
                {
                    auto event = queue.submit([&] (cl::sycl::handler& cgh)
                    {
                        auto B = B_d.get_access<
                                cl::sycl::access::mode::discard_write,
                                cl::sycl::access::target::global_buffer>(cgh);

                        auto write = writer{B, elems};
                        cgh.parallel_for(cl::sycl::nd_range<1>{
                                    cl::sycl::range<1>{group_num * group_size},
                                    cl::sycl::range<1>{group_size}},
                                         write);
                    });

                    event.wait_and_throw();
                    auto start = event.get_profiling_info<
                            cl::sycl::info::event_profiling::command_start>();
                    auto stop = event.get_profiling_info<
                            cl::sycl::info::event_profiling::command_end>();

                    auto elapsed = static_cast<float>(stop - start);
                    mintime = std::min(mintime, elapsed);
                }

                file << "WO;" << group_size << ";" << group_num << ";"
                     << sizeof(cl::sycl::float4) << ";" << elems << ";"
                     << mintime / 1e6 << ";"
                     << (1.0e-9*sizeof(cl::sycl::float4) * elems)/(mintime/1e9)
                     << std::endl;
            }

            file << std::endl;
        }
    }
}

auto main() -> int
{
    try
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
            return EXIT_FAILURE;
        }

        auto&& platform = platforms[index];

        // set up default context
        auto ctx = cl::sycl::context{platform};

        // set up default accelerator
        auto accelerators = ctx.get_devices();
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
            return EXIT_FAILURE;
        }

        auto&& acc = accelerators[index];

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

        auto queue = cl::sycl::queue{
            acc, exception_handler,
            cl::sycl::property::queue::enable_profiling{}};

        auto cus = acc.get_info<cl::sycl::info::device::max_compute_units>();

        // there is no way to ask for this with SYCL
        constexpr auto max_groups = (cl_uint{1} << 31) - cl_uint{1};

        auto now = std::chrono::system_clock::now();
        auto cnow = std::chrono::system_clock::to_time_t(now);

        auto filename = std::stringstream{};
        filename << "SYCL-";
        filename << std::put_time(std::localtime(&cnow), "%Y-%m-%d-%X");
        filename << ".csv";

        auto file = std::ofstream{filename.str()};

        file << "type;group_size;group_num;elem_size;elem_num;mintime;"
             << "throughput" << std::endl;

        do_benchmark(queue, cus, max_groups, file, 64, 1024);
#ifdef KEPLER
        do_benchmark(queue, cus, max_groups, file, 192, 768);
#endif
    }
    catch(const cl::sycl::exception& err)
    {
        std::cerr << err.what() << std::endl;
        if(err.get_cl_code() != CL_SUCCESS)
            std::cerr << err.get_cl_code() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
