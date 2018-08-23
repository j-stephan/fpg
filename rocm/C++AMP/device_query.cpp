#include <iostream>

#include <amp.h>

auto main() -> int
{
    // get all accelerators
    auto accs = Concurrency::accelerator::get_all();

    std::cout << "Available C++AMP accelerators: " << std::endl;
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
                << (acc.get_supports_cpu_shared_memory() ? "Yes" : "No")
                 << std::endl;
    }

    return 1;
}
