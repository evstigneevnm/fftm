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
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/system_timer_event.h>

#include <external_wrap/cufft_wrap_many.h>
#include <fftm.hpp>

#include "detail/fftm_4d_test_options.h"
#include "detail/mpi_cuda_test_init.h"
#include "detail/poisson_4d_fft_common.h"
#include "detail/poisson_4d_problem.h"

namespace
{

using T             = double;
using base_fft_t    = fftm::wrap::cufft_wrap_many<T>;
using runtime_api_t = typename base_fft_t::runtime_api;
using backend_t     = scfd::backend::cuda;
using reduce_t      = backend_t::reduce_type;
using for_each_t    = backend_t::template for_each_nd_type<4, int>;
using idx_t         = scfd::static_vec::vec<int, 4>;
using rect_t        = scfd::static_vec::rect<int, 4>;
using strategy_kind = fftm::test::detail::fftm_4d_strategy_kind;
using test_options  = fftm::test::detail::fftm_4d_test_options;

template <class DistStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_poisson(
    strategy_kind strategy, scfd::utils::log_mpi &log, const test_options &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    using fftm_t = fftm::fftm<
        base_fft_t, scfd::communication::mpi_comm_info, backend_t, fftm::strategy_3d_pencil_pencil<Mode>,
        scfd::utils::log_mpi, DistStrategy4D>;

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
    distributed_fft.template init<4>( grid, sizes, fftm::test::detail::make_fftm_init_options( options ) );

    const auto  in_sizes        = distributed_fft.get_local_input_sizes_4d();
    const auto  spectral_sizes  = distributed_fft.get_local_spectral_sizes_4d();
    const auto  spectral_starts = distributed_fft.get_local_spectral_starts_4d();
    const bool  use_native_spectral_layout = distributed_fft.uses_native_spectral_layout_4d();
    const auto &input_part      = distributed_fft.input_partition();

    real_array_t rhs;
    real_array_t exact_solution;
    real_array_t exact_dx;
    real_array_t exact_dy;
    real_array_t exact_dz;
    real_array_t exact_dw;
    real_array_t numerical_solution;
    real_array_t numerical_dx;
    real_array_t numerical_dy;
    real_array_t numerical_dz;
    real_array_t numerical_dw;
    real_array_t solution_error_sq;
    real_array_t gradient_error_sq;
    hat_array_t  rhs_hat;
    hat_array_t  solution_hat;
    hat_array_t  dx_hat;
    hat_array_t  dy_hat;
    hat_array_t  dz_hat;
    hat_array_t  dw_hat;

    rhs.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    exact_solution.init(
        std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes )
    );
    exact_dx.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    exact_dy.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    exact_dz.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    exact_dw.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    numerical_solution.init(
        std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes )
    );
    numerical_dx.init(
        std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes )
    );
    numerical_dy.init(
        std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes )
    );
    numerical_dz.init(
        std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes )
    );
    numerical_dw.init(
        std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes )
    );
    solution_error_sq.init(
        std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes )
    );
    gradient_error_sq.init(
        std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes )
    );
    rhs_hat.init(
        std::get<0>( spectral_sizes ), std::get<1>( spectral_sizes ), std::get<2>( spectral_sizes ),
        std::get<3>( spectral_sizes )
    );
    solution_hat.init(
        std::get<0>( spectral_sizes ), std::get<1>( spectral_sizes ), std::get<2>( spectral_sizes ),
        std::get<3>( spectral_sizes )
    );
    dx_hat.init(
        std::get<0>( spectral_sizes ), std::get<1>( spectral_sizes ), std::get<2>( spectral_sizes ),
        std::get<3>( spectral_sizes )
    );
    dy_hat.init(
        std::get<0>( spectral_sizes ), std::get<1>( spectral_sizes ), std::get<2>( spectral_sizes ),
        std::get<3>( spectral_sizes )
    );
    dz_hat.init(
        std::get<0>( spectral_sizes ), std::get<1>( spectral_sizes ), std::get<2>( spectral_sizes ),
        std::get<3>( spectral_sizes )
    );
    dw_hat.init(
        std::get<0>( spectral_sizes ), std::get<1>( spectral_sizes ), std::get<2>( spectral_sizes ),
        std::get<3>( spectral_sizes )
    );

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

    for_each(
        fftm::test::detail::fill_poisson_4d_problem_functor<T, idx_t, real_array_t>{
            rhs, exact_solution, exact_dx, exact_dy, exact_dz, exact_dw, hx, hy, hz, hw,
            hx * static_cast<T>( input_part.start_x[myid_i] ), hy * static_cast<T>( input_part.start_y[myid_j] ),
            hz * static_cast<T>( input_part.start_z[myid_k] ), T( 0 ) },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs )
    );
    for_each.wait();

    const T local_rhs_sum = reduce( rhs.total_size(), rhs.raw_ptr(), T( 0 ) );
    const T rhs_mean =
        comm_info.all_reduce_sum( local_rhs_sum ) / static_cast<T>( options.nx * options.ny * options.nz * options.nw );
    if ( comm_info.myid == 0 && std::abs( rhs_mean ) > T( 1.0e-12 ) )
        log.warning_f( "rhs_mean = %.8e", rhs_mean );

    scfd::utils::system_timer_event t0, t1;
    runtime_api_t::device_synchronize();
    t0.record();

    if ( use_native_spectral_layout )
    {
        auto rhs_hat_native      = distributed_fft.make_native_spectral_view_4d( rhs_hat );
        auto solution_hat_native = distributed_fft.make_native_spectral_view_4d( solution_hat );
        auto dx_hat_native       = distributed_fft.make_native_spectral_view_4d( dx_hat );
        auto dy_hat_native       = distributed_fft.make_native_spectral_view_4d( dy_hat );
        auto dz_hat_native       = distributed_fft.make_native_spectral_view_4d( dz_hat );
        auto dw_hat_native       = distributed_fft.make_native_spectral_view_4d( dw_hat );

        distributed_fft.forward_native_spectral_4d( rhs, rhs_hat_native );
        for_each(
            fftm::test::detail::solve_poisson_4d_xzwy_functor<T, idx_t, decltype( rhs_hat_native )>{
                rhs_hat_native, solution_hat_native, static_cast<int>( options.nx ), static_cast<int>( options.ny ),
                static_cast<int>( options.nz ), static_cast<int>( std::get<0>( spectral_starts ) ),
                static_cast<int>( std::get<1>( spectral_starts ) ),
                static_cast<int>( std::get<2>( spectral_starts ) ),
                static_cast<int>( std::get<3>( spectral_starts ) ) },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs_hat_native )
        );
        for_each.wait();

        for_each(
            fftm::test::detail::poisson_4d_xzwy_derivative_spectra_functor<T, idx_t, decltype( solution_hat_native )>{
                solution_hat_native, dx_hat_native, dy_hat_native, dz_hat_native, dw_hat_native,
                static_cast<int>( options.nx ), static_cast<int>( options.ny ), static_cast<int>( options.nz ),
                static_cast<int>( std::get<0>( spectral_starts ) ),
                static_cast<int>( std::get<1>( spectral_starts ) ),
                static_cast<int>( std::get<2>( spectral_starts ) ),
                static_cast<int>( std::get<3>( spectral_starts ) ) },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( solution_hat_native )
        );
        for_each.wait();

        distributed_fft.backward_native_spectral_4d( solution_hat_native, numerical_solution );
    }
    else
    {
        distributed_fft.forward( rhs, rhs_hat );
        for_each(
            fftm::test::detail::solve_poisson_4d_functor<T, idx_t, hat_array_t>{
                rhs_hat, solution_hat, static_cast<int>( options.nx ), static_cast<int>( options.ny ),
                static_cast<int>( options.nz ), static_cast<int>( std::get<0>( spectral_starts ) ),
                static_cast<int>( std::get<1>( spectral_starts ) ),
                static_cast<int>( std::get<2>( spectral_starts ) ) },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs_hat )
        );
        for_each.wait();

        for_each(
            fftm::test::detail::poisson_4d_derivative_spectra_functor<T, idx_t, hat_array_t>{
                solution_hat, dx_hat, dy_hat, dz_hat, dw_hat, static_cast<int>( options.nx ),
                static_cast<int>( options.ny ), static_cast<int>( options.nz ),
                static_cast<int>( std::get<0>( spectral_starts ) ),
                static_cast<int>( std::get<1>( spectral_starts ) ),
                static_cast<int>( std::get<2>( spectral_starts ) ) },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( solution_hat )
        );
        for_each.wait();

        distributed_fft.backward( solution_hat, numerical_solution );
    }
    for_each(
        fftm::test::detail::scale_real_4d_functor<T, idx_t, real_array_t>{ numerical_solution, normalization },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_solution )
    );
    for_each.wait();

    runtime_api_t::device_synchronize();
    t1.record();
    const T wall_ms = static_cast<T>( t1.elapsed_time( t0 ) );

    if ( use_native_spectral_layout )
    {
        auto dx_hat_native = distributed_fft.make_native_spectral_view_4d( dx_hat );
        auto dy_hat_native = distributed_fft.make_native_spectral_view_4d( dy_hat );
        auto dz_hat_native = distributed_fft.make_native_spectral_view_4d( dz_hat );
        auto dw_hat_native = distributed_fft.make_native_spectral_view_4d( dw_hat );

        distributed_fft.backward_native_spectral_4d( dx_hat_native, numerical_dx );
        distributed_fft.backward_native_spectral_4d( dy_hat_native, numerical_dy );
        distributed_fft.backward_native_spectral_4d( dz_hat_native, numerical_dz );
        distributed_fft.backward_native_spectral_4d( dw_hat_native, numerical_dw );
    }
    else
    {
        distributed_fft.backward( dx_hat, numerical_dx );
        distributed_fft.backward( dy_hat, numerical_dy );
        distributed_fft.backward( dz_hat, numerical_dz );
        distributed_fft.backward( dw_hat, numerical_dw );
    }

    for_each(
        fftm::test::detail::scale_real_4d_functor<T, idx_t, real_array_t>{ numerical_dx, normalization },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_dx )
    );
    for_each(
        fftm::test::detail::scale_real_4d_functor<T, idx_t, real_array_t>{ numerical_dy, normalization },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_dy )
    );
    for_each(
        fftm::test::detail::scale_real_4d_functor<T, idx_t, real_array_t>{ numerical_dz, normalization },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_dz )
    );
    for_each(
        fftm::test::detail::scale_real_4d_functor<T, idx_t, real_array_t>{ numerical_dw, normalization },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_dw )
    );
    for_each.wait();

    for_each(
        fftm::test::detail::poisson_4d_error_fields_functor<T, idx_t, real_array_t>{
            numerical_solution, numerical_dx, numerical_dy, numerical_dz, numerical_dw, exact_solution, exact_dx,
            exact_dy, exact_dz, exact_dw, solution_error_sq, gradient_error_sq },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs )
    );
    for_each.wait();

    const T local_l2_sq  = reduce( solution_error_sq.total_size(), solution_error_sq.raw_ptr(), T( 0 ) ) * cell_volume;
    const T local_h1_sq  = reduce( gradient_error_sq.total_size(), gradient_error_sq.raw_ptr(), T( 0 ) ) * cell_volume;
    const T global_l2_sq = comm_info.all_reduce_sum( local_l2_sq );
    const T global_h1_sq = comm_info.all_reduce_sum( local_h1_sq );
    const T l2           = std::sqrt( global_l2_sq );
    const T h1           = std::sqrt( global_h1_sq );
    const T l2_threshold = static_cast<T>( options.l2_threshold >= 0.0 ? options.l2_threshold : options.threshold );
    const T h1_threshold = static_cast<T>( options.h1_threshold >= 0.0 ? options.h1_threshold : options.threshold );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "strategy=%s, mode=%s, spectral_layout=%s, Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu: "
            "L2=%.8e, H1=%.8e, l2_threshold=%.8e, h1_threshold=%.8e, wall_ms=%.8e",
            fftm_t::strategy_name_4d(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ),
            fftm::fftm_4d_spectral_layout_name( distributed_fft.spectral_layout_4d() ), options.nx, options.ny,
            options.nz, options.nw, l2, h1, l2_threshold, h1_threshold, wall_ms
        );
        if ( l2 > l2_threshold || h1 > h1_threshold )
        {
            log.error_f(
                "4D Poisson validation failed: L2=%.8e, H1=%.8e, l2_threshold=%.8e, h1_threshold=%.8e",
                l2, h1, l2_threshold, h1_threshold
            );
        }
    }

    return ( l2 > l2_threshold || h1 > h1_threshold ) ? 1 : 0;
}

template <fftm::mpi_transpose_3d_mode Mode>
int run_for_strategy_kind(
    strategy_kind strategy, scfd::utils::log_mpi &log, const test_options &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    switch ( strategy )
    {
    case strategy_kind::pencil_pencil:
        return run_poisson<fftm::strategy_4d_pencil_pencil_mpi<Mode>, Mode>( strategy, log, options, comm_info );
    case strategy_kind::slab_slab:
        return run_poisson<fftm::strategy_4d_slab_slab_mpi<Mode>, Mode>( strategy, log, options, comm_info );
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
        fftm::test::detail::init_cuda_mpi_for_tests( log, comm_info );

        const test_options options =
            fftm::test::detail::parse_fftm_4d_test_options( argc, argv, "test_4D_poisson_mpi.bin", true, true, false );

        int failed = 0;
        if ( options.run_all )
        {
            failed = std::max( failed, dispatch_mode( strategy_kind::pencil_pencil, log, options, comm_info ) );
            failed = std::max( failed, dispatch_mode( strategy_kind::slab_slab, log, options, comm_info ) );
        }
        else
        {
            failed = dispatch_mode( options.strategy, log, options, comm_info );
        }

        return failed;
    }
    catch ( const std::exception &e )
    {
        if ( comm_info.myid == 0 )
            log.error( scfd::utils::nested_exception_to_multistring( e ) );
        return 1;
    }
}
