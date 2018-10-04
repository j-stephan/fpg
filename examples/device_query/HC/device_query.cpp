/* Copyright (c) 2018, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <iostream>

#include <hc.hpp>

auto main() -> int
{
    // get all accelerators
    auto accs = hc::accelerator::get_all();

    std::cout << "Available HC accelerators: " << std::endl;
    for(auto&& acc : accs)
    {
        std::cout << "==================================================" << std::endl;
        std::wcout
            << "Description:\t\t\t" << acc.get_description() << "\n" 

            << "Device path:\t\t\t" << acc.get_device_path() << "\n"

            << "Version:\t\t\t" << acc.get_version() << "\n"

            << "Memory:\t\t\t\t" << acc.get_dedicated_memory() << " bytes\n"

            << "Double precision:\t\t"
                << (acc.get_supports_double_precision() ? "Yes\n" : "No\n")

            << "Limited double precision:\t"
                << (acc.get_supports_limited_double_precision() ? "Yes\n" :
                                                                  "No\n")

            << "Debug support:\t\t\t" << (acc.get_is_debug() ? "Yes\n" : "No\n")

            << "Is emulated:\t\t\t"
                << (acc.get_is_emulated() ? "Yes\n" : "No\n")

            << "CPU-GPU shared memory support:\t"
                << (acc.get_supports_cpu_shared_memory() ? "Yes\n" : "No\n")

            << "CPU accessible memory support:\t"
                << (acc.has_cpu_accessible_am() ? "Yes\n" : "No\n")

            << "Maximum tile static area size:\t"
                << acc.get_max_tile_static_size() << "\n"

            << "HSA accelerator:\t\t"
                << (acc.is_hsa_accelerator() ? "Yes\n" : "No\n")

            << "Number of compute units:\t" << acc.get_cu_count() << "\n"

            << "Sequence number:\t\t" << acc.get_seqnum() << std::endl;
    }

    return 1;
}
