#ifndef FFTM_FFT_PLAN_DESCRIPTOR_H
#define FFTM_FFT_PLAN_DESCRIPTOR_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "fft_direction.h"

namespace fftm
{
namespace wrap
{

struct fft_plan_descriptor
{
    direction                    dir = direction::R2C;
    int                          backend_type = 0;
    int                          rank = 0;
    bool                         default_layout = false;
    std::array<long long int, 3> n = {{ 0, 0, 0 }};
    std::array<long long int, 3> inembed = {{ 0, 0, 0 }};
    long long int                istride = 0;
    long long int                idist = 0;
    std::array<long long int, 3> onembed = {{ 0, 0, 0 }};
    long long int                ostride = 0;
    long long int                odist = 0;
    long long int                batch = 0;
    std::size_t                  work_size = 0;
    std::uintptr_t               work_area_token = 0;
    std::uintptr_t               stream_token = 0;
};

inline const char *fft_direction_name( direction dir )
{
    switch ( dir )
    {
    case direction::R2C:
        return "R2C";
    case direction::C2R:
        return "C2R";
    case direction::C2CF:
        return "C2CF";
    case direction::C2CB:
        return "C2CB";
    }
    return "unknown";
}

} // namespace wrap
} // namespace fftm

#endif // FFTM_FFT_PLAN_DESCRIPTOR_H
