#include <cassert>
#include <stdexcept>
#include <string>

#include "detail/fft_benchmark_common.h"
#include <scfd/communication/mpi_wrap.h>

namespace
{

struct fake_comm
{
    int num_procs = 4;
    int myid      = 2;
    int reduced_failures = -1;

    int all_reduce_sum( int local_failures ) const
    {
        return reduced_failures < 0 ? local_failures : reduced_failures;
    }
};

} // namespace

int main( int argc, char **argv )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    const auto                    comm = mpi.comm_world();

    bool called = false;
    fftm::test::detail::run_collective_reporting_step(
        fake_comm{}, "successful report", [&]() { called = true; }
    );
    assert( called );

    bool local_failure_reported = false;
    try
    {
        fftm::test::detail::run_collective_reporting_step(
            fake_comm{}, "wall-time report", []() { throw std::runtime_error( "write failed" ); }
        );
    }
    catch ( const std::runtime_error &ex )
    {
        const std::string message = ex.what();
        local_failure_reported = message.find( "failed on 1 of 4 MPI ranks" ) != std::string::npos &&
                                 message.find( "rank 2: write failed" ) != std::string::npos;
    }
    assert( local_failure_reported );

    bool remote_failure_reported = false;
    try
    {
        fake_comm comm;
        comm.reduced_failures = 2;
        fftm::test::detail::run_collective_reporting_step(
            comm, "summary report", []() {}
        );
    }
    catch ( const std::runtime_error &ex )
    {
        const std::string message = ex.what();
        remote_failure_reported = message.find( "failed on 2 of 4 MPI ranks" ) != std::string::npos &&
                                  message.find( "rank 2:" ) == std::string::npos;
    }
    assert( remote_failure_reported );

    if ( comm.num_procs >= 2 )
    {
        bool collective_failure_reported = false;
        try
        {
            fftm::test::detail::run_collective_reporting_step(
                comm, "MPI wall-time report", [&]() {
                    if ( comm.myid == 1 )
                        throw std::runtime_error( "rank-local write failed" );
                }
            );
        }
        catch ( const std::runtime_error &ex )
        {
            const std::string message = ex.what();
            collective_failure_reported =
                message.find( "failed on 1 of " + std::to_string( comm.num_procs ) + " MPI ranks" ) !=
                std::string::npos;
            if ( comm.myid == 1 )
            {
                collective_failure_reported = collective_failure_reported &&
                    message.find( "rank 1: rank-local write failed" ) != std::string::npos;
            }
        }
        assert( comm.all_reduce_sum( collective_failure_reported ? 1 : 0 ) == comm.num_procs );
    }
    return 0;
}
