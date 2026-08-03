#ifndef FFTM_RUNTIME_API_TYPES_H
#define FFTM_RUNTIME_API_TYPES_H

#include <cstddef>

namespace fftm
{
namespace wrap
{

enum class memory_copy_kind
{
    device_to_device,
    device_to_host,
    host_to_device
};

struct memory_position_3d
{
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t z = 0;
};

struct pitched_memory_pointer
{
    void       *pointer = nullptr;
    std::size_t pitch = 0;
    std::size_t x_size = 0;
    std::size_t y_size = 0;
};

struct memory_extent_3d
{
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t depth = 0;
};

struct memory_copy_3d_params
{
    memory_position_3d    source_position;
    pitched_memory_pointer source;
    memory_position_3d    destination_position;
    pitched_memory_pointer destination;
    memory_extent_3d      extent;
    memory_copy_kind      kind = memory_copy_kind::device_to_device;
};

template <class T>
struct alignas( 2 * sizeof( T ) ) complex_value
{
    T x;
    T y;
};

} // namespace wrap
} // namespace fftm

#endif // FFTM_RUNTIME_API_TYPES_H
