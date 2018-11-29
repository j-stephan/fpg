#include <cstdint>
#include <cstdlib>
#include <cwchar>
#include <iostream>
#include <locale>
#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wc++17-extensions"
#include <hc.hpp>
#pragma GCC diagnostic pop

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "common/common.h"

namespace common
{
    namespace
    {
        // deal with HC's ridiculous wstrings
        auto sys_loc = std::locale{""};

        template <class I, class E, class S>
        struct convert_type : std::codecvt_byname<I, E, S>
        {
            using std::codecvt_byname<I, E, S>::codecvt_byname;
            ~convert_type() {}
        };
        using cvt = convert_type<wchar_t, char, std::mbstate_t>;
        auto&& converter = std::wstring_convert<cvt>{
            new cvt{sys_loc.name()}};
    }

    struct dev_handle_impl
    {
        // nothing to do here
    };

    dev_handle::~dev_handle()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    struct dev_ptr_impl
    {
        hc::array<int, 1> data;
    };

    dev_ptr::~dev_ptr()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    struct dev_clock_impl
    {
        std::uint64_t ticks;
    };

    dev_clock::~dev_clock()
    {
        if(p_impl != nullptr)
            delete p_impl;
    }

    auto init() -> dev_handle
    {
        // set up default accelerator
        auto accelerators = hc::accelerator::get_all();
        std::cout << "Available accelerators: " << std::endl;
        for(auto i = 0u; i < accelerators.size(); ++i)
        {
            auto&& acc = accelerators[i];
            auto desc = converter.to_bytes(acc.get_description());
            auto path = converter.to_bytes(acc.get_device_path());

            std::cout << "\t[" << i << "] " << desc << " at " << path
                      << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Select accelerator: ";
        auto index = 0u;
        std::cin >> index;

        if(index >= accelerators.size())
        {
            std::cout << "I'm sorry, Dave. I'm afraid I can't do that."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        hc::accelerator::set_default(accelerators[index].get_device_path());
        return dev_handle{nullptr};
    }

    auto get_info() -> info
    {
        // select default accelerator
        auto accelerator = hc::accelerator();
        auto agent = static_cast<hsa_agent_t*>(accelerator.get_hsa_agent());

        auto mem_clock = int{};
        auto status = hsa_agent_get_info(
                        *agent,
                        static_cast<hsa_agent_info_t>(
                            HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY),
                        &mem_clock);
        if(status != HSA_STATUS_SUCCESS)
        {
            std::cerr << "hsa_agent_get_info("
                      << "HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY)"
                      << " returned an error." << std::endl;
            std::exit(status);
        }

        auto clock = int{};
        status = hsa_agent_get_info(
                    *agent,
                    static_cast<hsa_agent_info_t>(
                        HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY),
                    &clock);
        if(status != HSA_STATUS_SUCCESS)
        {
            std::cerr << "hsa_agent_get_info("
                      << "HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY)"
                      << " returned an error." << std::endl;
            std::exit(status);
        }

        auto num_sm = int{};
        status = hsa_agent_get_info(
                    *agent,
                    static_cast<hsa_agent_info_t>(
                        HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
                    &num_sm);
        if(status != HSA_STATUS_SUCCESS)
        {
            std::cerr << "hsa_agent_get_info("
                      << "HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT)"
                      << " returned an error." << std::endl;
            std::exit(status);
        }

        return info{accelerator.get_seqnum(),
                    converter.to_bytes(accelerator.get_description()),
                    1, 3, mem_clock, clock, num_sm};
    }

    auto make_array(std::size_t size) -> dev_ptr
    {
        using namespace hc;
        return dev_ptr{new dev_ptr_impl{array<int, 1>{extent<1>{size}}}};
    }

    auto copy_h2d(const std::vector<int>& src, dev_ptr& dst) -> void
    {
        hc::copy(std::begin(src), std::end(src), dst.p_impl->data);
    }

    auto copy_d2h(const dev_ptr& src, std::vector<int>& dst) -> void
    {
        hc::copy(src.p_impl->data, std::begin(dst));
    }

    auto start_clock(dev_handle&) -> dev_clock
    {
        return dev_clock{new dev_clock_impl{hc::get_system_ticks()}};
    }

    auto stop_clock(dev_handle&) -> dev_clock
    {
        return dev_clock{new dev_clock_impl{hc::get_system_ticks()}};
    }

    auto get_duration(const dev_clock& start, const dev_clock& stop) -> float
    {
        auto start_ticks = start.p_impl->ticks;
        auto stop_ticks = stop.p_impl->ticks;
        auto elapsed = stop_ticks - start_ticks;

        auto tps = hc::get_tick_frequency();
        auto tpms = static_cast<float>(tps) / 1000.f;

        return static_cast<float>(elapsed) / tpms;
    }
}
