#ifndef __FFTM_DETAIL_RUNTIME_HARDWARE_IDENTITY_H__
#define __FFTM_DETAIL_RUNTIME_HARDWARE_IDENTITY_H__

#include <cstddef>
#include <string>

namespace fftm
{
namespace detail
{

struct runtime_hardware_identity
{
    std::string backend;
    std::string device_name;
    std::string device_uuid;
    std::string pci_bus_id;
    std::string architecture;
    int         runtime_version   = 0;
    int         driver_version    = 0;
    std::size_t total_memory_bytes = 0;
    bool        total_memory_known = false;
};

} // namespace detail
} // namespace fftm

#endif
