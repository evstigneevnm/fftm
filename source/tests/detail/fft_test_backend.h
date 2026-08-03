#ifndef FFTM_TESTS_DETAIL_FFT_TEST_BACKEND_H
#define FFTM_TESTS_DETAIL_FFT_TEST_BACKEND_H

#include <cstdlib>

#include <fftm_backend.hpp>

namespace fftm
{
namespace test
{
namespace detail
{

template <class T>
using fft_test_wrap_many = ::fftm::device_backend::fft<T>;
using fft_test_backend = ::fftm::device_backend::scfd_backend;

template <class Log, class Comm>
inline int init_fft_test_mpi( Log &log, const Comm &comm )
{
    const char *wrap = std::getenv( "FFTM_WRAP_PROCS_GPUS" );
    return ::fftm::device_backend::init_mpi( log, comm, 0, wrap != nullptr && wrap[0] != '\0' && wrap[0] != '0' );
}

inline const char *fft_test_backend_name()
{
    return ::fftm::device_backend::name();
}

} // namespace detail
} // namespace test
} // namespace fftm

#endif // FFTM_TESTS_DETAIL_FFT_TEST_BACKEND_H
