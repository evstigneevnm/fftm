#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>


#include <cuda_runtime.h>

#include <scfd/backend/cuda.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/scalar_traits.h>

#include <external_wrap/cufft_wrap_many.h>
#include <fftm.hpp>
#include <ffts.hpp>

#include "detail/fftm_4d_test_options.h"
#include "detail/mpi_cuda_test_init.h"

namespace
{

using T             = double;
using base_fft_t    = fftm::wrap::cufft_wrap_many<T>;
using backend_t     = scfd::backend::cuda;
using memory_t      = backend_t::memory_type;
using reduce_t      = backend_t::reduce_type;
using for_each_t    = backend_t::template for_each_nd_type<4, int>;
using idx_t         = scfd::static_vec::vec<int, 4>;
using rect_t        = scfd::static_vec::rect<int, 4>;
using strategy_kind = fftm::test::detail::fftm_4d_strategy_kind;
using test_options  = fftm::test::detail::fftm_4d_test_options;

template <class Array>
rect_t make_range( const Array &array )
{
    const auto sz = array.size_nd();
    return rect_t(
        idx_t( 0, 0, 0, 0 ),
        idx_t(
            static_cast<int>( sz[0] ), static_cast<int>( sz[1] ), static_cast<int>( sz[2] ), static_cast<int>( sz[3] )
        )
    );
}

__DEVICE_TAG__ T sample_value( T x, T y, T z, T w )
{
    return scfd::utils::scalar_traits<T>::sin( x ) + T( 0.5 ) * scfd::utils::scalar_traits<T>::cos( y ) -
           T( 0.25 ) * scfd::utils::scalar_traits<T>::sin( T( 2 ) * z ) +
           T( 0.125 ) * scfd::utils::scalar_traits<T>::cos( w ) +
           T( 0.0625 ) * scfd::utils::scalar_traits<T>::sin( x + y - z + w );
}

template <class Array>
struct fill_input_functor
{
    Array array;
    T     hx;
    T     hy;
    T     hz;
    T     hw;
    T     x0;
    T     y0;
    T     z0;
    T     w0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const T x    = x0 + hx * static_cast<T>( idx[0] );
        const T y    = y0 + hy * static_cast<T>( idx[1] );
        const T z    = z0 + hz * static_cast<T>( idx[2] );
        const T w    = w0 + hw * static_cast<T>( idx[3] );
        array( idx ) = sample_value( x, y, z, w );
    }
};

template <class LocalArray, class RefArray, class ErrorArray>
struct compare_output_functor
{
    LocalArray local;
    RefArray   reference;
    ErrorArray diff_sq;
    ErrorArray ref_sq;
    int        y_start;
    int        z_start;
    int        w_start;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const auto actual   = local( idx );
        const auto expected = reference( y_start + idx[0], z_start + idx[1], w_start + idx[2], idx[3] );

        const T diff_re = actual.x - expected.x;
        const T diff_im = actual.y - expected.y;
        const T ref_re  = expected.x;
        const T ref_im  = expected.y;

        const std::size_t lin = local.calc_lin_index( idx[0], idx[1], idx[2], idx[3] );
        diff_sq( lin )        = diff_re * diff_re + diff_im * diff_im;
        ref_sq( lin )         = ref_re * ref_re + ref_im * ref_im;
    }
};

template <class LocalArray, class RefArray, class ErrorArray>
struct compare_real_output_functor
{
    LocalArray local;
    RefArray   reference;
    ErrorArray diff_sq;
    ErrorArray ref_sq;
    int        x_start;
    int        y_start;
    int        z_start;
    int        w_start;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const T actual   = local( idx );
        const T expected = reference( x_start + idx[0], y_start + idx[1], z_start + idx[2], w_start + idx[3] );

        const T diff = actual - expected;

        const std::size_t lin = local.calc_lin_index( idx[0], idx[1], idx[2], idx[3] );
        diff_sq( lin )        = diff * diff;
        ref_sq( lin )         = expected * expected;
    }
};

template <class DistStrategy4D, class RefStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_compare(
    strategy_kind strategy, scfd::utils::log_mpi &log, const test_options &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    using fftm_t = fftm::fftm<
        base_fft_t, scfd::communication::mpi_comm_info, backend_t, fftm::strategy_3d_pencil_pencil<Mode>,
        scfd::utils::log_mpi, DistStrategy4D>;
    using ref_ffts_t = fftm::ffts<base_fft_t, backend_t, RefStrategy4D>;

    using local_real_t  = typename fftm_t::template real_array_t<4>;
    using local_hat_t   = typename fftm_t::template complex_array_t<4>;
    using ref_real_t    = typename ref_ffts_t::template real_array_t<4>;
    using ref_hat_t     = typename ref_ffts_t::template complex_array_t<4>;
    using error_array_t = scfd::arrays::array_nd<T, 1, memory_t>;

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

    fftm_t     distributed_fft( comm_info, log );
    ref_ffts_t reference_fft;

    distributed_fft.template init<4>( grid, sizes, fftm::test::detail::make_fftm_init_options( options ) );
    reference_fft.init( options.nx, options.ny, options.nz, options.nw );

    const auto &input_part  = distributed_fft.input_partition();
    const auto &output_part = distributed_fft.output_partition();

    local_real_t local_in;
    local_real_t local_back;
    local_hat_t  local_out;
    ref_real_t   ref_in;
    ref_real_t   ref_back;
    ref_hat_t    ref_out;

    local_in.init(
        input_part.size_x[myid_i], input_part.size_y[myid_j], input_part.size_z[myid_k], input_part.size_w[0]
    );
    local_back.init(
        input_part.size_x[myid_i], input_part.size_y[myid_j], input_part.size_z[myid_k], input_part.size_w[0]
    );
    local_out.init(
        output_part.size_y[myid_i], output_part.size_z[myid_j], output_part.size_w[myid_k], output_part.size_x[0]
    );
    ref_in.init( options.nx, options.ny, options.nz, options.nw );
    ref_back.init( options.nx, options.ny, options.nz, options.nw );
    ref_out.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );

    const T l  = T( 2 ) * scfd::utils::scalar_traits<T>::pi();
    const T hx = l / static_cast<T>( options.nx );
    const T hy = l / static_cast<T>( options.ny );
    const T hz = l / static_cast<T>( options.nz );
    const T hw = l / static_cast<T>( options.nw );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    for_each(
        fill_input_functor<ref_real_t>{ ref_in, hx, hy, hz, hw, T( 0 ), T( 0 ), T( 0 ), T( 0 ) }, make_range( ref_in )
    );
    for_each.wait();

    for_each(
        fill_input_functor<local_real_t>{
            local_in, hx, hy, hz, hw, hx * static_cast<T>( input_part.start_x[myid_i] ),
            hy * static_cast<T>( input_part.start_y[myid_j] ), hz * static_cast<T>( input_part.start_z[myid_k] ),
            T( 0 ) },
        make_range( local_in )
    );
    for_each.wait();

    reference_fft.forward( ref_in, ref_out );
    distributed_fft.forward( local_in, local_out );

    error_array_t forward_diff_sq;
    error_array_t forward_ref_sq;
    forward_diff_sq.init( local_out.total_size() );
    forward_ref_sq.init( local_out.total_size() );

    for_each(
        compare_output_functor<local_hat_t, ref_hat_t, error_array_t>{
            local_out, ref_out, forward_diff_sq, forward_ref_sq, static_cast<int>( output_part.start_y[myid_i] ),
            static_cast<int>( output_part.start_z[myid_j] ), static_cast<int>( output_part.start_w[myid_k] ) },
        make_range( local_out )
    );
    for_each.wait();

    const T local_forward_diff_sq  = reduce( forward_diff_sq.size(), forward_diff_sq.raw_ptr(), T( 0 ) );
    const T local_forward_ref_sq   = reduce( forward_ref_sq.size(), forward_ref_sq.raw_ptr(), T( 0 ) );
    const T global_forward_diff_sq = comm_info.all_reduce_sum( local_forward_diff_sq );
    const T global_forward_ref_sq  = comm_info.all_reduce_sum( local_forward_ref_sq );
    const T forward_relative_error = std::sqrt( global_forward_diff_sq / global_forward_ref_sq );

    reference_fft.backward( ref_out, ref_back );
    distributed_fft.backward( local_out, local_back );

    error_array_t backward_diff_sq;
    error_array_t backward_ref_sq;
    backward_diff_sq.init( local_in.total_size() );
    backward_ref_sq.init( local_in.total_size() );

    for_each(
        compare_real_output_functor<local_real_t, ref_real_t, error_array_t>{
            local_back, ref_back, backward_diff_sq, backward_ref_sq, static_cast<int>( input_part.start_x[myid_i] ),
            static_cast<int>( input_part.start_y[myid_j] ), static_cast<int>( input_part.start_z[myid_k] ), 0 },
        make_range( local_back )
    );
    for_each.wait();

    const T local_backward_diff_sq  = reduce( backward_diff_sq.size(), backward_diff_sq.raw_ptr(), T( 0 ) );
    const T local_backward_ref_sq   = reduce( backward_ref_sq.size(), backward_ref_sq.raw_ptr(), T( 0 ) );
    const T global_backward_diff_sq = comm_info.all_reduce_sum( local_backward_diff_sq );
    const T global_backward_ref_sq  = comm_info.all_reduce_sum( local_backward_ref_sq );
    const T backward_relative_error = std::sqrt( global_backward_diff_sq / global_backward_ref_sq );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "strategy=%s, mode=%s, grid=(%zu,%zu,%zu), sizes=(%zu,%zu,%zu,%zu), forward_rel_l2=%.8e, "
            "backward_rel_l2=%.8e",
            fftm_t::strategy_name_4d(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ), p1, p2, p3,
            options.nx, options.ny, options.nz, options.nw, forward_relative_error, backward_relative_error
        );
    }

    return ( forward_relative_error <= options.threshold && backward_relative_error <= options.threshold ) ? 0 : 1;
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
        return run_compare<
            fftm::strategy_4d_pencil_pencil_mpi<Mode>, fftm::strategy_4d_pencil_pencil<fftm::transpose_backend::direct>,
            Mode>( strategy, log, options, comm_info );
    case strategy_kind::slab_slab:
        return run_compare<
            fftm::strategy_4d_slab_slab_mpi<Mode>, fftm::strategy_4d_slab_slab<fftm::transpose_backend::direct>, Mode>(
            strategy, log, options, comm_info
        );
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
            fftm::test::detail::parse_fftm_4d_test_options( argc, argv, "test_fftm_4D_compare.bin", true, true, false );

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

        if ( comm_info.myid == 0 )
        {
            if ( failed == 0 )
                log.info( "PASSED" );
            else
                log.error( "FAILED" );
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
