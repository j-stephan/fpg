#ifndef COMMON_H_
#define COMMON_H_

#include <string>

namespace common
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

    struct dev_handle_impl;
    struct dev_handle
    {
        ~dev_handle();
        dev_handle_impl* p_impl;
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

    // Initialize device context
    auto init() -> dev_handle;

    // Return info struct
    auto get_info() -> info;

    // Create device memory
    // size: number of elements
    auto make_array(std::size_t size) -> dev_ptr;

    // Synchronous host-to-device copy
    auto copy_h2d(const std::vector<int>& src, dev_ptr& dst) -> void;

    // Synchronous device-to-host copy
    auto copy_d2h(const dev_ptr& src, std::vector<int>& dst) -> void;

    // Start clock
    auto start_clock(dev_handle& handle) -> dev_clock;

    // Stop clock
    auto stop_clock(dev_handle& handle) -> dev_clock;

    // Returns duration in milliseconds
    auto get_duration(const dev_clock& start, const dev_clock& stop) -> float;
}

#endif

