#include <memory>
#include <stdexcept>
#include <string>

#include <external_wrap/cufft_wrap_many.h>
#include <fftm.hpp>
#include <scfd/backend/cuda.h>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/log_mpi.h>

#include "detail/mpi_cuda_test_init.h"

namespace
{

struct log_counts
{
    int info = 0;
};

class counting_log
{
public:
    counting_log() : counts_( std::make_shared<log_counts>() )
    {
    }

    void info( const std::string & ) const
    {
        ++counts_->info;
    }

    int total() const
    {
        return counts_->info;
    }

private:
    std::shared_ptr<log_counts> counts_;
};

using value_type = double;
using base_fft_t = fftm::wrap::cufft_wrap_many<value_type>;
using backend_t  = scfd::backend::cuda;
using comm_t     = scfd::communication::mpi_comm_info;
using fftm_t     = fftm::fftm<
    base_fft_t, comm_t, backend_t,
    fftm::strategy_3d_slab_pencil<fftm::mpi_transpose_3d_mode::alltoallv>, counting_log,
    fftm::strategy_4d_slab_slab_mpi<fftm::mpi_transpose_3d_mode::alltoallv>>;

void run_3d( const comm_t &comm, const counting_log &log )
{
    fftm::processor_grid grid;
    grid.init( 1, 1 );
    fftm::global_sizes sizes;
    sizes.init( 8, 8, 8 );

    fftm_t transform( comm, log );
    transform.init<3>( grid, sizes );

    const auto input_sizes  = transform.get_local_input_sizes();
    const auto output_sizes = transform.get_local_output_sizes();
    fftm_t::real_array_t<3> input;
    fftm_t::real_array_t<3> restored;
    fftm_t::complex_array_t<3> spectrum;
    input.init( std::get<0>( input_sizes ), std::get<1>( input_sizes ), std::get<2>( input_sizes ) );
    restored.init( std::get<0>( input_sizes ), std::get<1>( input_sizes ), std::get<2>( input_sizes ) );
    spectrum.init( std::get<0>( output_sizes ), std::get<1>( output_sizes ), std::get<2>( output_sizes ) );

    transform.forward( input, spectrum );
    transform.backward( spectrum, restored );
}

void run_4d( const comm_t &comm, const counting_log &log )
{
    fftm::processor_grid grid;
    grid.init( 1, 1, 1 );
    fftm::global_sizes sizes;
    sizes.init( 4, 4, 4, 8 );

    fftm_t transform( comm, log );
    transform.init<4>( grid, sizes );

    const auto input_sizes  = transform.get_local_input_sizes_4d();
    const auto output_sizes = transform.get_local_output_sizes_4d();
    fftm_t::real_array_t<4> input;
    fftm_t::real_array_t<4> restored;
    fftm_t::complex_array_t<4> spectrum;
    input.init(
        std::get<0>( input_sizes ), std::get<1>( input_sizes ), std::get<2>( input_sizes ),
        std::get<3>( input_sizes )
    );
    restored.init(
        std::get<0>( input_sizes ), std::get<1>( input_sizes ), std::get<2>( input_sizes ),
        std::get<3>( input_sizes )
    );
    spectrum.init(
        std::get<0>( output_sizes ), std::get<1>( output_sizes ), std::get<2>( output_sizes ),
        std::get<3>( output_sizes )
    );

    transform.forward( input, spectrum );
    transform.backward( spectrum, restored );
}

} // namespace

int main( int argc, char *argv[] )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    const comm_t                  comm = mpi.comm_world();
    if ( comm.num_procs != 1 )
        throw std::logic_error( "test_fftm_quiet_default requires one MPI rank" );

    scfd::utils::log_mpi setup_log;
    fftm::test::detail::init_cuda_mpi_for_tests( setup_log, comm );

    counting_log fftm_log;
    run_3d( comm, fftm_log );
    run_4d( comm, fftm_log );
    if ( fftm_log.total() != 0 )
        throw std::runtime_error( "default FFTM options emitted library log messages" );
    return 0;
}
