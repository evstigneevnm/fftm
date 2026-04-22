#ifndef __FFTM_TESTS_DETAIL_CUDA_FFT_TEST_ENVIRONMENT_H__
#define __FFTM_TESTS_DETAIL_CUDA_FFT_TEST_ENVIRONMENT_H__

#include <scfd/backend/cuda.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/log_std.h>

#include <external_wrap/cufft_wrap_many.h>

namespace fftm
{
namespace test
{
namespace detail
{

template <class T>
struct cuda_fft_test_environment
{
    using value_type    = T;
    using backend_t     = scfd::backend::cuda;
    using base_fft_t    = fftm::wrap::cufft_wrap_many<T>;
    using runtime_api_t = typename base_fft_t::runtime_api;
    using mpi_wrap_t    = scfd::communication::mpi_wrap;
    using mpi_comm_t    = scfd::communication::mpi_comm_info;
    using log_mpi_t     = scfd::utils::log_mpi;
    using log_std_t     = scfd::utils::log_std;

    static void init_device( log_std_t &log )
    {
        scfd::utils::init_cuda_persistent( log, 0 );
    }

    static void init_device( log_mpi_t &log, const mpi_comm_t &comm_info )
    {
        scfd::utils::init_cuda_mpi( log, comm_info );
    }
};

} // namespace detail
} // namespace test
} // namespace fftm

#endif
