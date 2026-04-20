#include <memory>

#define SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI

#include <external_wrap/cufft_wrap_many.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/backend/cuda.h>

#include <scfd/utils/log_mpi.h>

#include <fftm.hpp>

#include <complex>

int main( int argc, char *argv[] )
{
    using T              = double;
    using C              = std::complex<T>;
    using comm_info_type = scfd::communication::mpi_comm_info;
    using base_fft_t     = fftm::wrap::cufft_wrap_many<T>;
    using backend_t      = scfd::backend::cuda;
    using fftm_t         = fftm::fftm<
        base_fft_t, comm_info_type, backend_t, fftm::strategy_3d_pencil_pencil<fftm::mpi_transpose_3d_mode::alltoallv>,
        scfd::utils::log_mpi>;

    using array_R_type = typename fftm_t::template real_array_t<3>;
    using array_C_type = typename fftm_t::template complex_array_t<3>;

    scfd::communication::mpi_wrap mpi( argc, argv );
    comm_info_type                comm_info = mpi.comm_world();
    scfd::utils::log_mpi          log;

    fftm_t distributed_fft( comm_info, log );

    fftm::processor_grid grid;
    grid.init( 3, 3 );

    fftm::global_sizes sizes;
    sizes.init( 128, 128, 128 );

    distributed_fft.template init<3>( grid, sizes );
    auto input_size  = distributed_fft.get_local_input_sizes();
    auto output_size = distributed_fft.get_local_output_sizes();

    std::cout << comm_info.myid << " (input): " << std::get<0>( input_size ) << " " << std::get<1>( input_size ) << " "
              << std::get<2>( input_size ) << std::endl;
    std::cout << comm_info.myid << " (output): " << std::get<0>( output_size ) << " " << std::get<1>( output_size )
              << " " << std::get<2>( output_size ) << std::endl;
    array_R_type in;
    array_C_type out;
    in.init( std::get<0>( input_size ), std::get<1>( input_size ), std::get<2>( input_size ) );
    out.init( std::get<0>( output_size ), std::get<1>( output_size ), std::get<2>( output_size ) );

    distributed_fft.forward( in, out );

    return 0;
}
