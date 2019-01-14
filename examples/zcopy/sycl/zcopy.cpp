#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <limits>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <CL/sycl.hpp>
#pragma clang diagnostic pop

constexpr auto elems = 1 << 25;
constexpr auto iters = 10;

struct initializer
{
    cl::sycl::accessor<cl::sycl::double2, 1,
                       cl::sycl::access::mode::discard_write,
                       cl::sycl::access::target::global_buffer> A;
    cl::sycl::accessor<cl::sycl::double2, 1,
                       cl::sycl::access::mode::discard_write,
                       cl::sycl::access::target::global_buffer> B;

    auto operator()(cl::sycl::nd_item<1> my_item) -> void
    {
        auto id = my_item.get_global_id(0);
        A[id] = {0.0, 0.0};
        B[id] = {std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::quiet_NaN()};
    }
};

struct reader_writer
{
    cl::sycl::accessor<cl::sycl::double2, 1,
                       cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer> A;
    cl::sycl::accessor<cl::sycl::double2, 1,
                       cl::sycl::access::mode::discard_write,
                       cl::sycl::access::target::global_buffer> B;

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
    cl::sycl::accessor<cl::sycl::double2, 1,
                       cl::sycl::access::mode::discard_write,
                       cl::sycl::access::target::global_buffer> B;

    auto operator()(cl::sycl::nd_item<1> my_item) -> void
    {
        auto stride = my_item.get_group_range(0) * my_item.get_local_range(0);
        for(auto i = my_item.get_global_id(0); i < elems; i += stride)
        {
            B[i] = cl::sycl::double2{0.0, 0.0};
        }
    }
};

auto main() -> int
{
    try
    {
        using namespace cl::sycl;

        // set up default platform
        auto platforms = platform::get_platforms();
        std::cout << "Available platforms: " << std::endl;
        for(auto i = 0u; i < platforms.size(); ++i)
        {
            auto&& p = platforms[i];
            auto vendor = p.get_info<info::platform::vendor>();
            auto name = p.get_info<info::platform::name>();
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
        auto ctx = context{platform};

        // set up default accelerator
        auto accelerators = ctx.get_devices();
        std::cout << "Available accelerators: " << std::endl;
        for(auto i = 0u; i < accelerators.size(); ++i)
        {
            auto&& acc = accelerators[i];
            auto vendor = acc.get_info<info::device::vendor>();
            auto name = acc.get_info<info::device::name>();
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

        auto&& acc = accelerators[index];

        // create queue on device
        auto exception_handler = [] (exception_list exceptions)
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

        auto queue = cl::sycl::queue{acc, exception_handler,
                       property::queue::enable_profiling{}};


        // Allocate memory on device
        auto A_d = buffer<double2, 1>{range<1>{elems}};
        auto B_d = buffer<double2, 1>{range<1>{elems}};


        // Initialize device memory
        queue.submit([&] (handler& cgh)
        {
            auto A = A_d.get_access<access::mode::discard_write,
                                    access::target::global_buffer>(cgh);
            auto B = B_d.get_access<access::mode::discard_write,
                                    access::target::global_buffer>(cgh);

            auto initer = initializer{A, B};
            cgh.parallel_for(nd_range<1>{range<1>{elems},
                                         range<1>{1024}},
                                         initer);
        });
        queue.wait_and_throw();

        constexpr auto local_size = 128;
        constexpr auto num_groups = (elems + (local_size - 1)) / local_size;
        constexpr auto groups = num_groups > 65520 ? 65520 : num_groups;

        std::cout << "zcopy: operating on vectors of " << elems << " double2s"
                  << " = " << sizeof(double2) * elems << " bytes" << std::endl;

        std::cout << "zcopy: using " << local_size << " threads per group, "
                  << groups << " groups" << std::endl;

        auto mintime = std::numeric_limits<float>::max();
        for(auto k = 0; k < iters; ++k)
        {
            auto event = queue.submit([&] (handler& cgh)
            {
                auto A = A_d.get_access<access::mode::read,
                                        access::target::global_buffer>(cgh);
                auto B = B_d.get_access<access::mode::discard_write,
                                        access::target::global_buffer>(cgh);

                auto read_write = reader_writer{A, B};
                cgh.parallel_for(nd_range<1>{range<1>{groups * local_size},
                                             range<1>{local_size}},
                                 read_write);
            });
            
            event.wait_and_throw();
            auto start = event.get_profiling_info<
                            info::event_profiling::command_start>();
            auto stop = event.get_profiling_info<
                            info::event_profiling::command_end>();

            auto elapsed = static_cast<float>(stop - start);
            mintime = std::min(mintime, elapsed);
        }

        std::cout << "RW: mintime = " << mintime / 1e6 << " msec  "
                  << "throughput = "
                  << (2.0e-9 * sizeof(double2) * elems) / (mintime / 1e9)
                  << " GB/sec" << std::endl;

        for(auto k = 0; k < iters; ++k)
        {
            auto event = queue.submit([&] (handler& cgh)
            {
                auto B = B_d.get_access<access::mode::discard_write,
                                        access::target::global_buffer>(cgh);

                auto write = writer{B};
                cgh.parallel_for(nd_range<1>{range<1>{groups * local_size},
                                             range<1>{local_size}},
                                 write);
            });
            
            event.wait_and_throw();
            auto start = event.get_profiling_info<
                            info::event_profiling::command_start>();
            auto stop = event.get_profiling_info<
                            info::event_profiling::command_end>();

            auto elapsed = static_cast<float>(stop - start);
            mintime = std::min(mintime, elapsed);
        }

        std::cout << "WO: mintime = " << mintime / 1e6 << " msec  "
                  << "throughput = "
                  << (1e-9 * sizeof(double2) * elems) / (mintime / 1e9)
                  << " GB/sec" << std::endl;
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
