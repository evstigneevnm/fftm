#ifndef FFTM_HIPFFT_SAFE_CALL_H
#define FFTM_HIPFFT_SAFE_CALL_H

#include <sstream>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

namespace fftm
{
namespace wrap
{
namespace detail
{

inline const char *hipfft_error_string( hipfftResult error )
{
    switch ( error )
    {
    case HIPFFT_SUCCESS:
        return "success";
    case HIPFFT_INVALID_PLAN:
        return "invalid plan";
    case HIPFFT_ALLOC_FAILED:
        return "allocation failed";
    case HIPFFT_INVALID_TYPE:
        return "invalid type";
    case HIPFFT_INVALID_VALUE:
        return "invalid value";
    case HIPFFT_INTERNAL_ERROR:
        return "internal error";
    case HIPFFT_EXEC_FAILED:
        return "execution failed";
    case HIPFFT_SETUP_FAILED:
        return "setup failed";
    case HIPFFT_INVALID_SIZE:
        return "invalid size";
    case HIPFFT_UNALIGNED_DATA:
        return "unaligned data";
    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
        return "incomplete parameter list";
    case HIPFFT_INVALID_DEVICE:
        return "invalid device";
    case HIPFFT_PARSE_ERROR:
        return "parse error";
    case HIPFFT_NO_WORKSPACE:
        return "missing workspace";
    case HIPFFT_NOT_IMPLEMENTED:
        return "not implemented";
    case HIPFFT_NOT_SUPPORTED:
        return "not supported";
    }
    return "unknown hipFFT error";
}

inline void hipfft_safe_call( hipfftResult status, const char *expression, const char *file, int line )
{
    const hipError_t synchronize_status = hipDeviceSynchronize();
    if ( status != HIPFFT_SUCCESS )
    {
        std::ostringstream message;
        message << "hipFFT call failed at " << file << ':' << line << ": " << expression << ": "
                << hipfft_error_string( status );
        throw std::runtime_error( message.str() );
    }
    if ( synchronize_status != hipSuccess )
    {
        std::ostringstream message;
        message << "HIP synchronization failed after hipFFT call at " << file << ':' << line << ": "
                << expression << ": " << hipGetErrorString( synchronize_status );
        throw std::runtime_error( message.str() );
    }
}

} // namespace detail
} // namespace wrap
} // namespace fftm

#define FFTM_HIPFFT_SAFE_CALL( expression )                                                                            \
    ::fftm::wrap::detail::hipfft_safe_call( ( expression ), #expression, __FILE__, __LINE__ )

#endif // FFTM_HIPFFT_SAFE_CALL_H
