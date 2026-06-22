#ifndef __FFTM_TESTS_DETAIL_MPI_CUDA_TEST_INIT_H__
#define __FFTM_TESTS_DETAIL_MPI_CUDA_TEST_INIT_H__

#include <cstdlib>

#include <scfd/communication/mpi_comm_info.h>
#include <scfd/utils/init_cuda_mpi.h>

namespace fftm
{
namespace test
{
namespace detail
{

inline bool wrap_mpi_processes_over_gpus_enabled()
{
    const char *value = std::getenv( "FFTM_WRAP_PROCS_GPUS" );
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

template <class Log>
inline int init_cuda_mpi_for_tests( Log &log, const scfd::communication::mpi_comm_info &comm_info )
{
    if ( wrap_mpi_processes_over_gpus_enabled() )
    {
        return scfd::utils::init_cuda_mpi<Log, true>( log, comm_info );
    }
    return scfd::utils::init_cuda_mpi( log, comm_info );
}

} // namespace detail
} // namespace test
} // namespace fftm

#endif
