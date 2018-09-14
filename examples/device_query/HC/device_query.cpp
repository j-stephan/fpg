/* This file is part of fpg.
 *
 * fpg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * fpg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with fpg. If not, see <http://www.gnu.org/licenses/>
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
