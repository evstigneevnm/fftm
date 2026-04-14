#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>

#include <scfd/backend/cuda.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/system_timer_event.h>

#include <external_wrap/cufft_wrap_many.h>
#include <fftm.hpp>

#include "detail/fftm_4d_test_options.h"
#include "detail/poisson_4d_fft_common.h"
#include "detail/poisson_4d_problem.h"

namespace
{

using T            = double;
using base_fft_t   = fftm::wrap::cufft_wrap_many<T>;
using runtime_api_t = typename base_fft_t::runtime_api;
using backend_t    = scfd::backend::cuda;
using reduce_t     = backend_t::reduce_type;
using for_each_t   = backend_t::template for_each_nd_type<4, int>;
using idx_t        = scfd::static_vec::vec<int, 4>;
using rect_t       = scfd::static_vec::rect<int, 4>;
using strategy_kind = fftm::test::detail::fftm_4d_strategy_kind;
using test_options  = fftm::test::detail::fftm_4d_test_options;

template <class DistStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_tutorial_case(
    strategy_kind                               strategy,
    scfd::utils::log_mpi                       &log,
    const test_options                         &options,
    const scfd::communication::mpi_comm_info   &comm_info
)
{
    using fftm_t = fftm::fftm<
        base_fft_t,
        scfd::communication::mpi_comm_info,
        backend_t,
        fftm::strategy_3d_pencil_pencil<Mode>,
        scfd::utils::log_mpi,
        DistStrategy4D
    >;

    using real_array_t = typename fftm_t::template real_array_t<4>;
    using hat_array_t  = typename fftm_t::template complex_array_t<4>;

    std::size_t p1 = 0, p2 = 0, p3 = 0;
    std::tie( p1, p2, p3 ) = fftm::test::detail::choose_grid_4d( options, strategy, comm_info.num_procs );

    if ( p1 * p2 * p3 != static_cast<std::size_t>( comm_info.num_procs ) )
        throw std::logic_error( "P1*P2*P3 must equal the number of MPI processes." );

    fftm::processor_grid grid;
    grid.init( p1, p2, p3 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz, options.nw );

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( grid, sizes );

    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    fftm_t distributed_fft( comm_info, log );
    distributed_fft.template init<4>( grid, sizes );

    const auto in_sizes   = distributed_fft.get_local_input_sizes_4d();
    const auto out_sizes  = distributed_fft.get_local_output_sizes_4d();
    const auto &input_part  = distributed_fft.input_partition();
    const auto &output_part = distributed_fft.output_partition();

    real_array_t field;
    hat_array_t  field_hat;
    field.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    field_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );

    const T lx            = fftm::test::detail::poisson_4d_problem<T>::domain_length();
    const T hx            = lx / static_cast<T>( options.nx );
    const T hy            = lx / static_cast<T>( options.ny );
    const T hz            = lx / static_cast<T>( options.nz );
    const T hw            = lx / static_cast<T>( options.nw );
    const T cell_volume   = hx * hy * hz * hw;
    const T normalization = T( 1 ) / static_cast<T>( options.nx * options.ny * options.nz * options.nw );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    // Step 1: fill the owned real-space chunk with the manufactured Poisson RHS.
    for_each(
        fftm::test::detail::fill_poisson_4d_rhs_functor<T, idx_t, real_array_t>{
            field,
            hx,
            hy,
            hz,
            hw,
            hx * static_cast<T>( input_part.start_x[myid_i] ),
            hy * static_cast<T>( input_part.start_y[myid_j] ),
            hz * static_cast<T>( input_part.start_z[myid_k] ),
            T( 0 )
        },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( field )
    );
    for_each.wait();

    const T local_rhs_sum = reduce( field.total_size(), field.raw_ptr(), T( 0 ) );
    const T rhs_mean =
        comm_info.all_reduce_sum( local_rhs_sum ) / static_cast<T>( options.nx * options.ny * options.nz * options.nw );
    if ( comm_info.myid == 0 && std::abs( rhs_mean ) > T( 1.0e-12 ) )
        log.warning_f( "rhs_mean = %.8e", rhs_mean );

    T wall_ms_acc = T( 0 );
    for ( int iter = 0; iter < options.times; ++iter )
    {
        // Refill the RHS so each iteration solves the same problem without storing an extra copy.
        for_each(
            fftm::test::detail::fill_poisson_4d_rhs_functor<T, idx_t, real_array_t>{
                field,
                hx,
                hy,
                hz,
                hw,
                hx * static_cast<T>( input_part.start_x[myid_i] ),
                hy * static_cast<T>( input_part.start_y[myid_j] ),
                hz * static_cast<T>( input_part.start_z[myid_k] ),
                T( 0 )
            },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( field )
        );
        for_each.wait();

        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();

        // Step 2: FFT the RHS to Fourier space.
        distributed_fft.forward( field, field_hat );

        // Step 3: solve -|k|^2 u_hat = f_hat directly on the local spectral chunk.
        for_each(
            fftm::test::detail::solve_poisson_4d_in_place_functor<T, idx_t, hat_array_t>{
                field_hat,
                static_cast<int>( options.nx ),
                static_cast<int>( options.ny ),
                static_cast<int>( options.nz ),
                static_cast<int>( output_part.start_y[myid_i] ),
                static_cast<int>( output_part.start_z[myid_j] ),
                static_cast<int>( output_part.start_w[myid_k] )
            },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( field_hat )
        );
        for_each.wait();

        // Step 4: inverse FFT back to real space and apply CUFFT normalization.
        distributed_fft.backward( field_hat, field );
        for_each(
            fftm::test::detail::scale_real_4d_functor<T, idx_t, real_array_t>{ field, normalization },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( field )
        );
        for_each.wait();

        runtime_api_t::device_synchronize();
        t1.record();
        wall_ms_acc += static_cast<T>( t1.elapsed_time( t0 ) );
    }

    const T wall_ms = wall_ms_acc / static_cast<T>( options.times );

    // Step 5: reuse the solution buffer to accumulate the global L2 norm.
    for_each(
        fftm::test::detail::square_real_4d_functor<T, idx_t, real_array_t>{ field },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( field )
    );
    for_each.wait();

    const T local_l2_sq  = reduce( field.total_size(), field.raw_ptr(), T( 0 ) ) * cell_volume;
    const T global_l2_sq = comm_info.all_reduce_sum( local_l2_sq );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "strategy=%s, mode=%s, Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, times=%d: solution_l2=%.8e, wall_ms=%.8e",
            fftm_t::strategy_name_4d(),
            fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ),
            options.nx,
            options.ny,
            options.nz,
            options.nw,
            options.times,
            std::sqrt( global_l2_sq ),
            wall_ms
        );
    }

    return 0;
}

template <fftm::mpi_transpose_3d_mode Mode>
int run_for_strategy_kind(
    strategy_kind                               strategy,
    scfd::utils::log_mpi                       &log,
    const test_options                         &options,
    const scfd::communication::mpi_comm_info   &comm_info
)
{
    switch ( strategy )
    {
        case strategy_kind::pencil_pencil:
            return run_tutorial_case<fftm::strategy_4d_pencil_pencil_mpi<Mode>, Mode>( strategy, log, options, comm_info );
        case strategy_kind::slab_slab:
            return run_tutorial_case<fftm::strategy_4d_slab_slab_mpi<Mode>, Mode>( strategy, log, options, comm_info );
    }

    return 1;
}

int dispatch_mode(
    strategy_kind                               strategy,
    scfd::utils::log_mpi                       &log,
    const test_options                         &options,
    const scfd::communication::mpi_comm_info   &comm_info
)
{
    switch ( options.mode )
    {
        case fftm::mpi_transpose_3d_mode::p2p_waitall:
            return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::p2p_waitall>( strategy, log, options, comm_info );
        case fftm::mpi_transpose_3d_mode::p2p_waitany:
            return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::p2p_waitany>( strategy, log, options, comm_info );
        case fftm::mpi_transpose_3d_mode::alltoallv:
            return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::alltoallv>( strategy, log, options, comm_info );
        case fftm::mpi_transpose_3d_mode::alltoallw:
            return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::alltoallw>( strategy, log, options, comm_info );
    }

    return 1;
}

} // namespace

int main( int argc, char *argv[] )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    auto                          comm_info = mpi.comm_world();
    scfd::utils::log_mpi          log;

    try
    {
        scfd::utils::init_cuda_mpi( log, comm_info );

        const test_options options = fftm::test::detail::parse_fftm_4d_test_options(
            argc,
            argv,
            "test_4D_poisson_mpi_tutorial.bin",
            false,
            false,
            true
        );

        return dispatch_mode( options.strategy, log, options, comm_info );
    }
    catch ( const std::exception &e )
    {
        if ( comm_info.myid == 0 )
            log.error( e.what() );
        return 1;
    }
}
