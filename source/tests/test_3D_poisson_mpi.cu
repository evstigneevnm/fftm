#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <scfd/backend/cuda.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/scalar_traits.h>
#include <scfd/utils/system_timer_event.h>

#include <external_wrap/cufft_wrap_many.h>
#include <fftm.hpp>

#include "detail/fftm_3d_test_options.h"
#include "detail/poisson_3d_fft_common.h"
#include "detail/poisson_3d_problem.h"

namespace
{

using T             = double;
using base_fft_t    = fftm::wrap::cufft_wrap_many<T>;
using runtime_api_t = typename base_fft_t::runtime_api;
using backend_t     = scfd::backend::cuda;
using memory_t      = backend_t::memory_type;
using reduce_t      = backend_t::reduce_type;
using for_each_t    = backend_t::template for_each_nd_type<3, int>;
using idx_t         = scfd::static_vec::vec<int, 3>;
using rect_t        = scfd::static_vec::rect<int, 3>;
using strategy_kind = fftm::test::detail::fftm_3d_strategy_kind;
using test_options  = fftm::test::detail::fftm_3d_test_options;

template <class Strategy>
int run_poisson(
    scfd::utils::log_mpi &log, const test_options &options, const scfd::communication::mpi_comm_info &comm_info
)
{
    using fftm_t =
        fftm::fftm<base_fft_t, scfd::communication::mpi_comm_info, backend_t, Strategy, scfd::utils::log_mpi>;
    using real_array_t = typename fftm_t::template real_array_t<3>;
    using hat_array_t  = typename fftm_t::template complex_array_t<3>;
    using err_array_t  = real_array_t;

    fftm::processor_grid grid;
    grid.init( options.p1, options.p2 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( grid, sizes );
    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    fftm_t distributed_fft( comm_info, log );
    distributed_fft.template init<3>( grid, sizes, fftm::test::detail::make_fftm_init_options( options ) );

    const auto in_sizes  = distributed_fft.get_local_input_sizes();
    const auto out_sizes = distributed_fft.get_local_output_sizes();

    real_array_t rhs;
    real_array_t exact_solution;
    real_array_t exact_dx;
    real_array_t exact_dy;
    real_array_t exact_dz;
    real_array_t numerical_solution;
    real_array_t numerical_dx;
    real_array_t numerical_dy;
    real_array_t numerical_dz;
    err_array_t  solution_error_sq;
    err_array_t  gradient_error_sq;
    hat_array_t  rhs_hat;
    hat_array_t  solution_hat;
    hat_array_t  dx_hat;
    hat_array_t  dy_hat;
    hat_array_t  dz_hat;

    rhs.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    exact_solution.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    exact_dx.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    exact_dy.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    exact_dz.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    numerical_solution.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    numerical_dx.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    numerical_dy.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    numerical_dz.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    solution_error_sq.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    gradient_error_sq.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    rhs_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ) );
    solution_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ) );
    dx_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ) );
    dy_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ) );
    dz_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ) );

    const T lx            = fftm::test::detail::poisson_3d_problem<T>::domain_length();
    const T hx            = lx / static_cast<T>( options.nx );
    const T hy            = lx / static_cast<T>( options.ny );
    const T hz            = lx / static_cast<T>( options.nz );
    const T cell_volume   = hx * hy * hz;
    const T normalization = T( 1 ) / static_cast<T>( options.nx * options.ny * options.nz );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    const auto &input_part  = distributed_fft.input_partition();
    const auto &output_part = distributed_fft.output_partition();

    for_each(
        fftm::test::detail::fill_poisson_3d_problem_functor<T, idx_t, real_array_t>{
            rhs, exact_solution, exact_dx, exact_dy, exact_dz, hx, hy, hz,
            hx * static_cast<T>( input_part.start_x[myid_i] ), hy * static_cast<T>( input_part.start_y[myid_j] ),
            T( 0 ) },
        fftm::test::detail::make_range_3d<idx_t, rect_t>( rhs )
    );
    for_each.wait();

    const T local_rhs_sum = reduce( rhs.size(), rhs.raw_ptr(), T( 0 ) );
    const T rhs_mean =
        comm_info.all_reduce_sum( local_rhs_sum ) / static_cast<T>( options.nx * options.ny * options.nz );
    if ( comm_info.myid == 0 && std::abs( rhs_mean ) > T( 1.0e-12 ) )
    {
        log.warning_f( "rhs_mean = %.8e", rhs_mean );
    }

    scfd::utils::system_timer_event t0, t1;
    runtime_api_t::device_synchronize();
    t0.record();

    distributed_fft.forward( rhs, rhs_hat );
    for_each(
        fftm::test::detail::solve_poisson_3d_functor<T, idx_t, hat_array_t>{
            rhs_hat, solution_hat, static_cast<int>( options.nx ), static_cast<int>( options.ny ),
            static_cast<int>( output_part.start_y[myid_i] ), static_cast<int>( output_part.start_z[myid_j] ) },
        fftm::test::detail::make_range_3d<idx_t, rect_t>( rhs_hat )
    );
    for_each.wait();
    distributed_fft.backward( solution_hat, numerical_solution );
    for_each(
        fftm::test::detail::scale_real_3d_functor<T, idx_t, real_array_t>{ numerical_solution, normalization },
        fftm::test::detail::make_range_3d<idx_t, rect_t>( numerical_solution )
    );
    for_each.wait();

    runtime_api_t::device_synchronize();
    t1.record();
    const T wall_ms = static_cast<T>( t1.elapsed_time( t0 ) );

    for_each(
        fftm::test::detail::poisson_3d_derivative_spectra_functor<T, idx_t, hat_array_t>{
            solution_hat, dx_hat, dy_hat, dz_hat, static_cast<int>( options.nx ), static_cast<int>( options.ny ),
            static_cast<int>( output_part.start_y[myid_i] ), static_cast<int>( output_part.start_z[myid_j] ) },
        fftm::test::detail::make_range_3d<idx_t, rect_t>( solution_hat )
    );
    for_each.wait();

    distributed_fft.backward( dx_hat, numerical_dx );
    distributed_fft.backward( dy_hat, numerical_dy );
    distributed_fft.backward( dz_hat, numerical_dz );
    for_each(
        fftm::test::detail::scale_real_3d_functor<T, idx_t, real_array_t>{ numerical_dx, normalization },
        fftm::test::detail::make_range_3d<idx_t, rect_t>( numerical_dx )
    );
    for_each(
        fftm::test::detail::scale_real_3d_functor<T, idx_t, real_array_t>{ numerical_dy, normalization },
        fftm::test::detail::make_range_3d<idx_t, rect_t>( numerical_dy )
    );
    for_each(
        fftm::test::detail::scale_real_3d_functor<T, idx_t, real_array_t>{ numerical_dz, normalization },
        fftm::test::detail::make_range_3d<idx_t, rect_t>( numerical_dz )
    );
    for_each.wait();

    for_each(
        fftm::test::detail::poisson_3d_error_fields_functor<T, idx_t, real_array_t>{
            numerical_solution, numerical_dx, numerical_dy, numerical_dz, exact_solution, exact_dx, exact_dy, exact_dz,
            solution_error_sq, gradient_error_sq },
        fftm::test::detail::make_range_3d<idx_t, rect_t>( rhs )
    );
    for_each.wait();

    const T local_l2_sq  = reduce( solution_error_sq.size(), solution_error_sq.raw_ptr(), T( 0 ) ) * cell_volume;
    const T local_h1_sq  = reduce( gradient_error_sq.size(), gradient_error_sq.raw_ptr(), T( 0 ) ) * cell_volume;
    const T global_l2_sq = comm_info.all_reduce_sum( local_l2_sq );
    const T global_h1_sq = comm_info.all_reduce_sum( local_h1_sq );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "strategy=%s, mode=%s, Nx=%zu, Ny=%zu, Nz=%zu: L2=%.8e, H1=%.8e, wall_ms=%.8e", fftm_t::strategy_name(),
            fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_3d ), options.nx, options.ny, options.nz,
            std::sqrt( global_l2_sq ), std::sqrt( global_h1_sq ), wall_ms
        );
    }

    return 0;
}

template <fftm::mpi_transpose_3d_mode Mode>
int run_for_strategy_kind(
    strategy_kind strategy, scfd::utils::log_mpi &log, const test_options &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    switch ( strategy )
    {
    case strategy_kind::slab_pencil:
        return run_poisson<fftm::strategy_3d_slab_pencil<Mode>>( log, options, comm_info );
    case strategy_kind::pencil_slab:
        return run_poisson<fftm::strategy_3d_pencil_slab<Mode>>( log, options, comm_info );
    case strategy_kind::pencil_pencil:
        return run_poisson<fftm::strategy_3d_pencil_pencil<Mode>>( log, options, comm_info );
    }
    return 1;
}

int dispatch_mode(
    strategy_kind strategy, scfd::utils::log_mpi &log, const test_options &options,
    const scfd::communication::mpi_comm_info &comm_info
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

        const test_options options = fftm::test::detail::parse_fftm_3d_test_options(
            argc, argv, comm_info.num_procs, "test_3D_poisson_mpi.bin", true, false
        );
        if ( options.p1 * options.p2 != static_cast<std::size_t>( comm_info.num_procs ) )
            throw std::logic_error( "P1*P2 must equal the number of MPI processes" );

        int failed = 0;
        if ( options.run_all )
        {
            failed = std::max( failed, dispatch_mode( strategy_kind::slab_pencil, log, options, comm_info ) );
            failed = std::max( failed, dispatch_mode( strategy_kind::pencil_slab, log, options, comm_info ) );
            failed = std::max( failed, dispatch_mode( strategy_kind::pencil_pencil, log, options, comm_info ) );
        }
        else
        {
            failed = dispatch_mode( options.strategy, log, options, comm_info );
        }

        return failed;
    }
    catch ( const std::exception &e )
    {
        log.error( scfd::utils::nested_exception_to_multistring( e ) );
        return 1;
    }
}
