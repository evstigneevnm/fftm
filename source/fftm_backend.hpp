#ifndef FFTM_BACKEND_HPP
#define FFTM_BACKEND_HPP

#include "detail/device_aware_mpi_config.h"

#if defined( FFTM_PLATFORM_HIP ) || defined( PLATFORM_HIP )
#include <external_wrap/hipfft_wrap_many.h>
#include <scfd/backend/hip.h>
#include <scfd/utils/init_hip_mpi.h>
#else
#include <external_wrap/cufft_wrap_many.h>
#include <scfd/backend/cuda.h>
#include <scfd/utils/init_cuda_mpi.h>
#endif

namespace fftm
{
namespace device_backend
{

// This is FFTM's sole public compile-time backend-selection boundary. Client
// code uses the neutral aliases below and never names vendor runtime/FFT types.

inline constexpr bool device_aware_mpi_enabled()
{
#if defined( FFTM_ENABLE_DEVICE_AWARE_MPI )
    return true;
#else
    return false;
#endif
}

#if defined( FFTM_PLATFORM_HIP ) || defined( PLATFORM_HIP )
template <class T>
using fft = ::fftm::wrap::hipfft_wrap_many<T>;
using scfd_backend = ::scfd::backend::hip;

template <class Log, class Comm>
inline int init_mpi( Log &log, const Comm &comm, int shift_index = 0, bool wrap_processes = false )
{
    return ::scfd::utils::init_hip_mpi( log, comm, shift_index, wrap_processes );
}

inline const char *name()
{
    return "hip";
}
#else
template <class T>
using fft = ::fftm::wrap::cufft_wrap_many<T>;
using scfd_backend = ::scfd::backend::cuda;

template <class Log, class Comm>
inline int init_mpi( Log &log, const Comm &comm, int shift_index = 0, bool wrap_processes = false )
{
    return ::scfd::utils::init_cuda_mpi( log, comm, shift_index, wrap_processes );
}

inline const char *name()
{
    return "cuda";
}
#endif

} // namespace device_backend
} // namespace fftm

#endif // FFTM_BACKEND_HPP
