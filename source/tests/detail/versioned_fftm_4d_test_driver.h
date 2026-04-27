#ifndef __FFTM_TESTS_DETAIL_VERSIONED_FFTM_4D_TEST_DRIVER_H__
#define __FFTM_TESTS_DETAIL_VERSIONED_FFTM_4D_TEST_DRIVER_H__

#ifndef FFTM_EGGER_TESTCASE
#error "FFTM_EGGER_TESTCASE must be defined before including versioned_fftm_4d_test_driver.h"
#endif

#ifndef FFTM_VERSIONED_TEST_BINARY
#error "FFTM_VERSIONED_TEST_BINARY must be defined before including versioned_fftm_4d_test_driver.h"
#endif

#ifndef FFTM_TEST_ENV
#error "FFTM_TEST_ENV must be defined before including versioned_fftm_4d_test_driver.h"
#endif

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/scalar_traits.h>
#include <scfd/utils/system_timer_event.h>

#include <fftm.hpp>
#include <ffts.hpp>

#include "fftm_4d_test_options.h"
#include "poisson_4d_fft_common.h"
#include "test_memory_profile_helpers.h"

namespace
{

using test_env_t    = FFTM_TEST_ENV;
using fftm_base_t   = fftm::fftm<typename test_env_t::base_fft_t, typename test_env_t::mpi_comm_t, typename test_env_t::backend_t>;
using T             = typename fftm_base_t::real;
using base_fft_t    = typename fftm_base_t::base_fft_type;
using backend_t     = typename fftm_base_t::backend_type;
using mpi_comm_t    = typename fftm_base_t::mpi_comm_type;
using runtime_api_t = typename fftm_base_t::runtime_api_t;
using log_mpi_t     = typename test_env_t::log_mpi_t;
using mpi_wrap_t    = typename test_env_t::mpi_wrap_t;
using memory_t      = typename fftm_base_t::memory_t;
using reduce_t      = typename backend_t::reduce_type;
using for_each_t    = typename backend_t::template for_each_nd_type<4, int>;
using idx_t         = scfd::static_vec::vec<int, 4>;
using rect_t        = scfd::static_vec::rect<int, 4>;
using strategy_kind = fftm::test::detail::fftm_4d_strategy_kind;
using options_t     = fftm::test::detail::fftm_4d_test_options;

template <class T_>
struct timing_statistics
{
    T_ mean   = T_( 0 );
    T_ stddev = T_( 0 );
};

template <class T_>
timing_statistics<T_> compute_timing_statistics( const std::vector<T_> &samples )
{
    timing_statistics<T_> stats;
    if ( samples.empty() )
        return stats;

    T_ sum = T_( 0 );
    for ( std::size_t i = 0; i < samples.size(); ++i )
        sum += samples[i];
    stats.mean = sum / static_cast<T_>( samples.size() );

    T_ sq_sum = T_( 0 );
    for ( std::size_t i = 0; i < samples.size(); ++i )
    {
        const T_ diff = samples[i] - stats.mean;
        sq_sum += diff * diff;
    }
    stats.stddev = std::sqrt( sq_sum / static_cast<T_>( samples.size() ) );
    return stats;
}

__DEVICE_TAG__ unsigned long long splitmix64( unsigned long long x )
{
    x += 0x9E3779B97F4A7C15ull;
    x = ( x ^ ( x >> 30 ) ) * 0xBF58476D1CE4E5B9ull;
    x = ( x ^ ( x >> 27 ) ) * 0x94D049BB133111EBull;
    return x ^ ( x >> 31 );
}

template <class T_>
__DEVICE_TAG__ T_ unit_random_from_state( unsigned long long state )
{
    const unsigned long long bits = splitmix64( state );
    const double             unit = static_cast<double>( bits >> 11 ) * ( 1.0 / 9007199254740992.0 );
    return static_cast<T_>( T_( 2 ) * static_cast<T_>( unit ) - T_( 1 ) );
}

template <class RealArray>
struct fill_random_real_functor
{
    RealArray           array;
    unsigned long long  seed;
    int                 start_x;
    int                 start_y;
    int                 start_z;
    int                 start_w;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const unsigned long long gx    = static_cast<unsigned long long>( start_x + idx[0] );
        const unsigned long long gy    = static_cast<unsigned long long>( start_y + idx[1] );
        const unsigned long long gz    = static_cast<unsigned long long>( start_z + idx[2] );
        const unsigned long long gw    = static_cast<unsigned long long>( start_w + idx[3] );
        const unsigned long long state = seed ^ ( gx * 0xD6E8FEB86659FD93ull ) ^ ( gy * 0xA5A3564E27F886A5ull ) ^
                                         ( gz * 0x9E3779B97F4A7C15ull ) ^ ( gw * 0x94D049BB133111EBull );
        array( idx ) = unit_random_from_state<T>( state );
    }
};

template <class RealArray>
struct scale_real_functor
{
    RealArray array;
    T         scale;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        array( idx ) *= scale;
    }
};

template <class RealArray>
struct overwrite_with_random_diff_square_functor
{
    RealArray           actual;
    unsigned long long  seed;
    int                 start_x;
    int                 start_y;
    int                 start_z;
    int                 start_w;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const unsigned long long gx    = static_cast<unsigned long long>( start_x + idx[0] );
        const unsigned long long gy    = static_cast<unsigned long long>( start_y + idx[1] );
        const unsigned long long gz    = static_cast<unsigned long long>( start_z + idx[2] );
        const unsigned long long gw    = static_cast<unsigned long long>( start_w + idx[3] );
        const unsigned long long state = seed ^ ( gx * 0xD6E8FEB86659FD93ull ) ^ ( gy * 0xA5A3564E27F886A5ull ) ^
                                         ( gz * 0x9E3779B97F4A7C15ull ) ^ ( gw * 0x94D049BB133111EBull );
        const T expected = unit_random_from_state<T>( state );
        const T diff     = actual( idx ) - expected;
        actual( idx )    = diff * diff;
    }
};

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

template <class ComplexArray>
struct fill_random_complex_functor
{
    ComplexArray       array;
    unsigned long long seed;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const unsigned long long i0    = static_cast<unsigned long long>( idx[0] );
        const unsigned long long i1    = static_cast<unsigned long long>( idx[1] );
        const unsigned long long i2    = static_cast<unsigned long long>( idx[2] );
        const unsigned long long i3    = static_cast<unsigned long long>( idx[3] );
        const unsigned long long state = seed ^ ( i0 * 0xD6E8FEB86659FD93ull ) ^ ( i1 * 0xA5A3564E27F886A5ull ) ^
                                         ( i2 * 0x9E3779B97F4A7C15ull ) ^ ( i3 * 0x94D049BB133111EBull );
        array( idx ).x = unit_random_from_state<T>( state );
        array( idx ).y = unit_random_from_state<T>( state ^ 0x369DEA0F31A53F85ull );
    }
};

template <class LocalArray, class RefArray, class ErrorArray>
struct compare_forward_output_functor
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

template <class T_>
struct periodic_poisson_4d_problem
{
    __DEVICE_TAG__ static T_ domain_length()
    {
        return T_( 2 ) * scfd::utils::scalar_traits<T_>::pi();
    }

    __DEVICE_TAG__ static void evaluate(
        T_ x, T_ y, T_ z, T_ w, T_ &rhs, T_ &exact_solution, T_ &exact_dx, T_ &exact_dy, T_ &exact_dz, T_ &exact_dw
    )
    {
        const T_ sin_x = scfd::utils::scalar_traits<T_>::sin( x );
        const T_ cos_x = scfd::utils::scalar_traits<T_>::cos( x );
        const T_ sin_y = scfd::utils::scalar_traits<T_>::sin( y );
        const T_ cos_y = scfd::utils::scalar_traits<T_>::cos( y );
        const T_ sin_z = scfd::utils::scalar_traits<T_>::sin( z );
        const T_ cos_z = scfd::utils::scalar_traits<T_>::cos( z );
        const T_ sin_w = scfd::utils::scalar_traits<T_>::sin( w );
        const T_ cos_w = scfd::utils::scalar_traits<T_>::cos( w );

        exact_solution = sin_x * sin_y * sin_z * sin_w;
        exact_dx       = cos_x * sin_y * sin_z * sin_w;
        exact_dy       = sin_x * cos_y * sin_z * sin_w;
        exact_dz       = sin_x * sin_y * cos_z * sin_w;
        exact_dw       = sin_x * sin_y * sin_z * cos_w;
        rhs            = -T_( 4 ) * exact_solution;
    }
};

template <class RealArray>
struct fill_periodic_poisson_4d_problem_functor
{
    RealArray rhs;
    RealArray exact_solution;
    RealArray exact_dx;
    RealArray exact_dy;
    RealArray exact_dz;
    RealArray exact_dw;
    T         hx;
    T         hy;
    T         hz;
    T         hw;
    T         x0;
    T         y0;
    T         z0;
    T         w0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const T x = x0 + hx * static_cast<T>( idx[0] );
        const T y = y0 + hy * static_cast<T>( idx[1] );
        const T z = z0 + hz * static_cast<T>( idx[2] );
        const T w = w0 + hw * static_cast<T>( idx[3] );

        T rhs_value    = T( 0 );
        T exact_value  = T( 0 );
        T exact_dx_val = T( 0 );
        T exact_dy_val = T( 0 );
        T exact_dz_val = T( 0 );
        T exact_dw_val = T( 0 );

        periodic_poisson_4d_problem<T>::evaluate(
            x, y, z, w, rhs_value, exact_value, exact_dx_val, exact_dy_val, exact_dz_val, exact_dw_val
        );

        rhs( idx )            = rhs_value;
        exact_solution( idx ) = exact_value;
        exact_dx( idx )       = exact_dx_val;
        exact_dy( idx )       = exact_dy_val;
        exact_dz( idx )       = exact_dz_val;
        exact_dw( idx )       = exact_dw_val;
    }
};

template <class RealArray>
struct periodic_poisson_4d_error_fields_functor
{
    RealArray numerical_solution;
    RealArray numerical_dx;
    RealArray numerical_dy;
    RealArray numerical_dz;
    RealArray numerical_dw;
    RealArray exact_solution;
    RealArray exact_dx;
    RealArray exact_dy;
    RealArray exact_dz;
    RealArray exact_dw;
    RealArray solution_error_sq;
    RealArray gradient_error_sq;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const T solution_diff = numerical_solution( idx ) - exact_solution( idx );
        const T dx_diff       = numerical_dx( idx ) - exact_dx( idx );
        const T dy_diff       = numerical_dy( idx ) - exact_dy( idx );
        const T dz_diff       = numerical_dz( idx ) - exact_dz( idx );
        const T dw_diff       = numerical_dw( idx ) - exact_dw( idx );

        solution_error_sq( idx ) = solution_diff * solution_diff;
        gradient_error_sq( idx ) = dx_diff * dx_diff + dy_diff * dy_diff + dz_diff * dz_diff + dw_diff * dw_diff;
    }
};

template <class DistStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_forward_random_benchmark(
    strategy_kind strategy, log_mpi_t &log, const options_t &options, const mpi_comm_t &comm_info
)
{
    using fftm_t = fftm::fftm<
        base_fft_t, mpi_comm_t, backend_t, fftm::strategy_3d_pencil_pencil<Mode>, log_mpi_t, DistStrategy4D>;
    using real_array_t = typename fftm_t::template real_array_t<4>;
    using hat_array_t  = typename fftm_t::template complex_array_t<4>;

    std::size_t p1 = 0, p2 = 0, p3 = 0;
    std::tie( p1, p2, p3 ) = fftm::test::detail::choose_grid_4d( options, strategy, comm_info.num_procs );

    fftm::processor_grid grid;
    grid.init( p1, p2, p3 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz, options.nw );

    fftm_t distributed_fft( comm_info, log );
    distributed_fft.template init<4>( grid, sizes );

    const auto  in_sizes   = distributed_fft.get_local_input_sizes_4d();
    const auto  out_sizes  = distributed_fft.get_local_output_sizes_4d();
    const auto &input_part = distributed_fft.input_partition();

    int myid_i = 0, myid_j = 0, myid_k = 0;
    {
        fftm::fft_partitioning<mpi_comm_t> partitioning( comm_info );
        partitioning.init( grid, sizes );
        std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();
    }

    real_array_t work;
    hat_array_t  hat;
    work.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );

    for_each_t for_each;
    for_each.block_size = 128;

    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );

    for ( int iter = 0; iter < options.warmup; ++iter )
    {
        const unsigned long long seed = 0xABCDEF0123456789ull + static_cast<unsigned long long>( iter );
        for_each(
            fill_random_real_functor<real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), static_cast<int>( input_part.start_z[myid_k] ), 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        runtime_api_t::device_synchronize();
        distributed_fft.forward( work, hat );
        runtime_api_t::device_synchronize();
    }

    for ( int iter = 0; iter < options.times; ++iter )
    {
        const unsigned long long seed = 0xABCDEF0123456789ull + static_cast<unsigned long long>( iter );
        for_each(
            fill_random_real_functor<real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), static_cast<int>( input_part.start_z[myid_k] ), 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        distributed_fft.forward( work, hat );
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );
    }

    const auto stats = compute_timing_statistics( wall_times );
    fftm::test::detail::log_tracked_memory_with_external_mpi(
        log, comm_info, distributed_fft, "test=fftm_v0_4d",
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( work ), fftm::test::detail::array_bytes( hat ) ) ),
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( work ), fftm::test::detail::array_bytes( hat ) ) )
    );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "test=fftm_v0_4d, strategy=%s, mode=%s, grid=(%zu,%zu,%zu), Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, "
            "warmup=%d, times=%d: avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
            fftm_t::strategy_name_4d(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ), p1, p2, p3,
            options.nx, options.ny, options.nz, options.nw, options.warmup, options.times, stats.mean, stats.stddev
        );
    }

    return 0;
}

template <class DistStrategy4D, class RefStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_forward_reference_compare(
    strategy_kind strategy, log_mpi_t &log, const options_t &options, const mpi_comm_t &comm_info
)
{
    using fftm_t = fftm::fftm<
        base_fft_t, mpi_comm_t, backend_t, fftm::strategy_3d_pencil_pencil<Mode>, log_mpi_t, DistStrategy4D>;
    using ref_ffts_t = fftm::ffts<base_fft_t, backend_t, RefStrategy4D>;

    using local_real_t  = typename fftm_t::template real_array_t<4>;
    using local_hat_t   = typename fftm_t::template complex_array_t<4>;
    using ref_real_t    = typename ref_ffts_t::template real_array_t<4>;
    using ref_hat_t     = typename ref_ffts_t::template complex_array_t<4>;
    using error_array_t = scfd::arrays::array_nd<T, 1, memory_t>;

    std::size_t p1 = 0, p2 = 0, p3 = 0;
    std::tie( p1, p2, p3 ) = fftm::test::detail::choose_grid_4d( options, strategy, comm_info.num_procs );

    fftm::processor_grid grid;
    grid.init( p1, p2, p3 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz, options.nw );

    fftm::fft_partitioning<mpi_comm_t> partitioning( comm_info );
    partitioning.init( grid, sizes );

    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    fftm_t     distributed_fft( comm_info, log );
    ref_ffts_t reference_fft;
    distributed_fft.template init<4>( grid, sizes );
    reference_fft.init( options.nx, options.ny, options.nz, options.nw );

    const auto &input_part  = distributed_fft.input_partition();
    const auto &output_part = distributed_fft.output_partition();
    const auto  in_sizes    = distributed_fft.get_local_input_sizes_4d();
    const auto  out_sizes   = distributed_fft.get_local_output_sizes_4d();

    local_real_t local_in;
    local_hat_t  local_out;
    ref_real_t   ref_in;
    ref_hat_t    ref_out;

    local_in.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    local_out.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );
    ref_in.init( options.nx, options.ny, options.nz, options.nw );
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

    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );
    T max_forward_rel_l2 = T( 0 );

    for ( int iter = 0; iter < options.warmup; ++iter )
    {
        runtime_api_t::device_synchronize();
        distributed_fft.forward( local_in, local_out );
        runtime_api_t::device_synchronize();
    }

    for ( int iter = 0; iter < options.times; ++iter )
    {
        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        distributed_fft.forward( local_in, local_out );
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );

        error_array_t diff_sq;
        error_array_t ref_sq;
        diff_sq.init( local_out.total_size() );
        ref_sq.init( local_out.total_size() );

        for_each(
            compare_forward_output_functor<local_hat_t, ref_hat_t, error_array_t>{
                local_out, ref_out, diff_sq, ref_sq, static_cast<int>( output_part.start_y[myid_i] ),
                static_cast<int>( output_part.start_z[myid_j] ), static_cast<int>( output_part.start_w[myid_k] ) },
            make_range( local_out )
        );
        for_each.wait();

        const T local_forward_diff_sq  = reduce( diff_sq.size(), diff_sq.raw_ptr(), T( 0 ) );
        const T local_forward_ref_sq   = reduce( ref_sq.size(), ref_sq.raw_ptr(), T( 0 ) );
        const T global_forward_diff_sq = comm_info.all_reduce_sum( local_forward_diff_sq );
        const T global_forward_ref_sq  = comm_info.all_reduce_sum( local_forward_ref_sq );
        const T forward_rel_l2         = std::sqrt( global_forward_diff_sq / global_forward_ref_sq );
        if ( forward_rel_l2 > max_forward_rel_l2 )
            max_forward_rel_l2 = forward_rel_l2;
    }

    const auto stats = compute_timing_statistics( wall_times );
    if ( distributed_fft.is_memory_profiling_enabled() || reference_fft.is_memory_profiling_enabled() )
    {
        using bytes_t = typename fftm_t::memory_profile_bytes_t;
        const auto distributed_buckets = distributed_fft.get_memory_profile_buckets();
        const auto reference_buckets   = reference_fft.get_memory_profile_buckets();
        const bytes_t internal_device_current = static_cast<bytes_t>(
            distributed_buckets.get( ::fftm::detail::memory_profile_bucket::device ).current +
            reference_buckets.get( ::fftm::detail::memory_profile_bucket::device ).current
        );
        const bytes_t internal_device_peak = static_cast<bytes_t>(
            distributed_buckets.get( ::fftm::detail::memory_profile_bucket::device ).peak +
            reference_buckets.get( ::fftm::detail::memory_profile_bucket::device ).peak
        );
        const bytes_t internal_host_current = static_cast<bytes_t>(
            distributed_buckets.get( ::fftm::detail::memory_profile_bucket::host_pinned ).current +
            reference_buckets.get( ::fftm::detail::memory_profile_bucket::host_pinned ).current
        );
        const bytes_t internal_host_peak = static_cast<bytes_t>(
            distributed_buckets.get( ::fftm::detail::memory_profile_bucket::host_pinned ).peak +
            reference_buckets.get( ::fftm::detail::memory_profile_bucket::host_pinned ).peak
        );
        const bytes_t internal_other_current = static_cast<bytes_t>(
            distributed_buckets.get( ::fftm::detail::memory_profile_bucket::other ).current +
            reference_buckets.get( ::fftm::detail::memory_profile_bucket::other ).current
        );
        const bytes_t internal_other_peak = static_cast<bytes_t>(
            distributed_buckets.get( ::fftm::detail::memory_profile_bucket::other ).peak +
            reference_buckets.get( ::fftm::detail::memory_profile_bucket::other ).peak
        );
        const bytes_t external_device_current = static_cast<bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( local_in ), fftm::test::detail::array_bytes( local_out ),
            fftm::test::detail::array_bytes( ref_in ), fftm::test::detail::array_bytes( ref_out ) ) );
        const bytes_t external_device_peak = static_cast<bytes_t>(
            external_device_current + fftm::test::detail::sum_bytes(
                                          fftm::test::detail::bytes_of_elems<T>( local_out.total_size() ),
                                          fftm::test::detail::bytes_of_elems<T>( local_out.total_size() ) )
        );

        fftm::test::detail::log_tracked_memory_categories_mpi<fftm_t>(
            log, comm_info, "test=fftm_v1_4d", internal_device_current, internal_device_peak, internal_host_current,
            internal_host_peak, external_device_current, external_device_peak, internal_other_current,
            internal_other_peak
        );
    }

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "test=fftm_v1_4d, strategy=%s, mode=%s, grid=(%zu,%zu,%zu), Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, "
            "warmup=%d, times=%d: forward_rel_l2=%.8e, avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
            fftm_t::strategy_name_4d(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ), p1, p2, p3,
            options.nx, options.ny, options.nz, options.nw, options.warmup, options.times, max_forward_rel_l2,
            stats.mean, stats.stddev
        );
    }

    return max_forward_rel_l2 <= static_cast<T>( options.threshold ) ? 0 : 1;
}

template <class DistStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_backward_random_benchmark(
    strategy_kind strategy, log_mpi_t &log, const options_t &options, const mpi_comm_t &comm_info
)
{
    using fftm_t = fftm::fftm<
        base_fft_t, mpi_comm_t, backend_t, fftm::strategy_3d_pencil_pencil<Mode>, log_mpi_t, DistStrategy4D>;
    using real_array_t = typename fftm_t::template real_array_t<4>;
    using hat_array_t  = typename fftm_t::template complex_array_t<4>;

    std::size_t p1 = 0, p2 = 0, p3 = 0;
    std::tie( p1, p2, p3 ) = fftm::test::detail::choose_grid_4d( options, strategy, comm_info.num_procs );

    fftm::processor_grid grid;
    grid.init( p1, p2, p3 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz, options.nw );

    fftm_t distributed_fft( comm_info, log );
    distributed_fft.template init<4>( grid, sizes );

    const auto out_sizes = distributed_fft.get_local_output_sizes_4d();
    const auto in_sizes  = distributed_fft.get_local_input_sizes_4d();

    hat_array_t  hat;
    real_array_t work;
    hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );
    work.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );

    for_each_t for_each;
    for_each.block_size = 128;

    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );

    for ( int iter = 0; iter < options.warmup; ++iter )
    {
        const unsigned long long seed = 0x1020304050607080ull + static_cast<unsigned long long>( iter );
        for_each( fill_random_complex_functor<hat_array_t>{ hat, seed }, make_range( hat ) );
        for_each.wait();

        runtime_api_t::device_synchronize();
        distributed_fft.backward( hat, work );
        runtime_api_t::device_synchronize();
    }

    for ( int iter = 0; iter < options.times; ++iter )
    {
        const unsigned long long seed = 0x1020304050607080ull + static_cast<unsigned long long>( iter );
        for_each( fill_random_complex_functor<hat_array_t>{ hat, seed }, make_range( hat ) );
        for_each.wait();

        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        distributed_fft.backward( hat, work );
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );
    }

    const auto stats = compute_timing_statistics( wall_times );
    fftm::test::detail::log_tracked_memory_with_external_mpi(
        log, comm_info, distributed_fft, "test=fftm_v2_4d",
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( hat ), fftm::test::detail::array_bytes( work ) ) ),
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( hat ), fftm::test::detail::array_bytes( work ) ) )
    );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "test=fftm_v2_4d, strategy=%s, mode=%s, grid=(%zu,%zu,%zu), Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, "
            "warmup=%d, times=%d: avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
            fftm_t::strategy_name_4d(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ), p1, p2, p3,
            options.nx, options.ny, options.nz, options.nw, options.warmup, options.times, stats.mean, stats.stddev
        );
    }

    return 0;
}

template <class DistStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_roundtrip_random_test(
    strategy_kind strategy, log_mpi_t &log, const options_t &options, const mpi_comm_t &comm_info
)
{
    using fftm_t = fftm::fftm<
        base_fft_t, mpi_comm_t, backend_t, fftm::strategy_3d_pencil_pencil<Mode>, log_mpi_t, DistStrategy4D>;
    using real_array_t = typename fftm_t::template real_array_t<4>;
    using hat_array_t  = typename fftm_t::template complex_array_t<4>;

    std::size_t p1 = 0, p2 = 0, p3 = 0;
    std::tie( p1, p2, p3 ) = fftm::test::detail::choose_grid_4d( options, strategy, comm_info.num_procs );

    fftm::processor_grid grid;
    grid.init( p1, p2, p3 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz, options.nw );

    fftm_t distributed_fft( comm_info, log );
    distributed_fft.template init<4>( grid, sizes );

    const auto  in_sizes   = distributed_fft.get_local_input_sizes_4d();
    const auto  out_sizes  = distributed_fft.get_local_output_sizes_4d();
    const auto &input_part = distributed_fft.input_partition();

    int myid_i = 0, myid_j = 0, myid_k = 0;
    {
        fftm::fft_partitioning<mpi_comm_t> partitioning( comm_info );
        partitioning.init( grid, sizes );
        std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();
    }

    real_array_t work;
    hat_array_t  hat;
    work.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    const T normalization = T( 1 ) / static_cast<T>( options.nx * options.ny * options.nz * options.nw );
    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );
    T max_l2 = T( 0 );

    for ( int iter = 0; iter < options.warmup; ++iter )
    {
        const unsigned long long seed = 0xABCDEF0123456789ull + static_cast<unsigned long long>( iter );

        for_each(
            fill_random_real_functor<real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), static_cast<int>( input_part.start_z[myid_k] ), 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        runtime_api_t::device_synchronize();
        distributed_fft.forward( work, hat );
        distributed_fft.backward( hat, work );
        runtime_api_t::device_synchronize();
    }

    for ( int iter = 0; iter < options.times; ++iter )
    {
        const unsigned long long seed = 0xABCDEF0123456789ull + static_cast<unsigned long long>( iter );

        for_each(
            fill_random_real_functor<real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), static_cast<int>( input_part.start_z[myid_k] ), 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        distributed_fft.forward( work, hat );
        distributed_fft.backward( hat, work );
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );

        for_each(
            scale_real_functor<real_array_t>{ work, normalization },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        for_each(
            overwrite_with_random_diff_square_functor<real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), static_cast<int>( input_part.start_z[myid_k] ), 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        const T local_diff_sq  = reduce( work.total_size(), work.raw_ptr(), T( 0 ) );
        const T global_diff_sq = comm_info.all_reduce_sum( local_diff_sq );
        const T diff_l2 =
            std::sqrt( global_diff_sq / static_cast<T>( options.nx * options.ny * options.nz * options.nw ) );
        if ( diff_l2 > max_l2 )
            max_l2 = diff_l2;
    }

    const auto stats = compute_timing_statistics( wall_times );
    fftm::test::detail::log_tracked_memory_with_external_mpi(
        log, comm_info, distributed_fft, "test=fftm_v3_4d",
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( work ), fftm::test::detail::array_bytes( hat ) ) ),
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( work ), fftm::test::detail::array_bytes( hat ) ) )
    );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "test=fftm_v3_4d, strategy=%s, mode=%s, grid=(%zu,%zu,%zu), Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, "
            "warmup=%d, times=%d: max_l2=%.8e, avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
            fftm_t::strategy_name_4d(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ), p1, p2, p3,
            options.nx, options.ny, options.nz, options.nw, options.warmup, options.times, max_l2, stats.mean,
            stats.stddev
        );
    }

    return max_l2 <= static_cast<T>( options.threshold ) ? 0 : 1;
}

template <class DistStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_periodic_laplacian_test(
    strategy_kind strategy, log_mpi_t &log, const options_t &options, const mpi_comm_t &comm_info
)
{
    using fftm_t = fftm::fftm<
        base_fft_t, mpi_comm_t, backend_t, fftm::strategy_3d_pencil_pencil<Mode>, log_mpi_t, DistStrategy4D>;
    using real_array_t = typename fftm_t::template real_array_t<4>;
    using hat_array_t  = typename fftm_t::template complex_array_t<4>;

    std::size_t p1 = 0, p2 = 0, p3 = 0;
    std::tie( p1, p2, p3 ) = fftm::test::detail::choose_grid_4d( options, strategy, comm_info.num_procs );

    fftm::processor_grid grid;
    grid.init( p1, p2, p3 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz, options.nw );

    fftm::fft_partitioning<mpi_comm_t> partitioning( comm_info );
    partitioning.init( grid, sizes );

    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    fftm_t distributed_fft( comm_info, log );
    distributed_fft.template init<4>( grid, sizes );

    const auto  in_sizes    = distributed_fft.get_local_input_sizes_4d();
    const auto  out_sizes   = distributed_fft.get_local_output_sizes_4d();
    const auto &input_part  = distributed_fft.input_partition();
    const auto &output_part = distributed_fft.output_partition();

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
    exact_solution.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    exact_dx.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    exact_dy.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    exact_dz.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    exact_dw.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    numerical_solution.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    numerical_dx.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    numerical_dy.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    numerical_dz.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    numerical_dw.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    solution_error_sq.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    gradient_error_sq.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    rhs_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );
    solution_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );
    dx_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );
    dy_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );
    dz_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );
    dw_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );

    const T l             = periodic_poisson_4d_problem<T>::domain_length();
    const T hx            = l / static_cast<T>( options.nx );
    const T hy            = l / static_cast<T>( options.ny );
    const T hz            = l / static_cast<T>( options.nz );
    const T hw            = l / static_cast<T>( options.nw );
    const T cell_volume   = hx * hy * hz * hw;
    const T normalization = T( 1 ) / static_cast<T>( options.nx * options.ny * options.nz * options.nw );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    for_each(
        fill_periodic_poisson_4d_problem_functor<real_array_t>{
            rhs, exact_solution, exact_dx, exact_dy, exact_dz, exact_dw, hx, hy, hz, hw,
            hx * static_cast<T>( input_part.start_x[myid_i] ), hy * static_cast<T>( input_part.start_y[myid_j] ),
            hz * static_cast<T>( input_part.start_z[myid_k] ), T( 0 ) },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs )
    );
    for_each.wait();

    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );

    for ( int iter = 0; iter < options.warmup; ++iter )
    {
        runtime_api_t::device_synchronize();
        distributed_fft.forward( rhs, rhs_hat );
        for_each(
            fftm::test::detail::solve_poisson_4d_functor<T, idx_t, hat_array_t>{
                rhs_hat, solution_hat, static_cast<int>( options.nx ), static_cast<int>( options.ny ),
                static_cast<int>( options.nz ), static_cast<int>( output_part.start_y[myid_i] ),
                static_cast<int>( output_part.start_z[myid_j] ), static_cast<int>( output_part.start_w[myid_k] ) },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs_hat )
        );
        for_each.wait();
        distributed_fft.backward( solution_hat, numerical_solution );
        for_each(
            fftm::test::detail::scale_real_4d_functor<T, idx_t, real_array_t>{ numerical_solution, normalization },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_solution )
        );
        for_each.wait();
        runtime_api_t::device_synchronize();
    }

    for ( int iter = 0; iter < options.times; ++iter )
    {
        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        distributed_fft.forward( rhs, rhs_hat );
        for_each(
            fftm::test::detail::solve_poisson_4d_functor<T, idx_t, hat_array_t>{
                rhs_hat, solution_hat, static_cast<int>( options.nx ), static_cast<int>( options.ny ),
                static_cast<int>( options.nz ), static_cast<int>( output_part.start_y[myid_i] ),
                static_cast<int>( output_part.start_z[myid_j] ), static_cast<int>( output_part.start_w[myid_k] ) },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs_hat )
        );
        for_each.wait();
        distributed_fft.backward( solution_hat, numerical_solution );
        for_each(
            fftm::test::detail::scale_real_4d_functor<T, idx_t, real_array_t>{ numerical_solution, normalization },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_solution )
        );
        for_each.wait();
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );
    }

    distributed_fft.forward( rhs, rhs_hat );
    for_each(
        fftm::test::detail::solve_poisson_4d_functor<T, idx_t, hat_array_t>{
            rhs_hat, solution_hat, static_cast<int>( options.nx ), static_cast<int>( options.ny ),
            static_cast<int>( options.nz ), static_cast<int>( output_part.start_y[myid_i] ),
            static_cast<int>( output_part.start_z[myid_j] ), static_cast<int>( output_part.start_w[myid_k] ) },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs_hat )
    );
    for_each.wait();

    for_each(
        fftm::test::detail::poisson_4d_derivative_spectra_functor<T, idx_t, hat_array_t>{
            solution_hat, dx_hat, dy_hat, dz_hat, dw_hat, static_cast<int>( options.nx ),
            static_cast<int>( options.ny ), static_cast<int>( options.nz ),
            static_cast<int>( output_part.start_y[myid_i] ), static_cast<int>( output_part.start_z[myid_j] ),
            static_cast<int>( output_part.start_w[myid_k] ) },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( solution_hat )
    );
    for_each.wait();

    distributed_fft.backward( dx_hat, numerical_dx );
    distributed_fft.backward( dy_hat, numerical_dy );
    distributed_fft.backward( dz_hat, numerical_dz );
    distributed_fft.backward( dw_hat, numerical_dw );
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
        periodic_poisson_4d_error_fields_functor<real_array_t>{
            numerical_solution, numerical_dx, numerical_dy, numerical_dz, numerical_dw, exact_solution, exact_dx,
            exact_dy, exact_dz, exact_dw, solution_error_sq, gradient_error_sq },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs )
    );
    for_each.wait();

    const T local_l2_sq  = reduce( solution_error_sq.total_size(), solution_error_sq.raw_ptr(), T( 0 ) ) * cell_volume;
    const T local_h1_sq  = reduce( gradient_error_sq.total_size(), gradient_error_sq.raw_ptr(), T( 0 ) ) * cell_volume;
    const T global_l2_sq = comm_info.all_reduce_sum( local_l2_sq );
    const T global_h1_sq = comm_info.all_reduce_sum( local_h1_sq );
    const T l2_norm      = std::sqrt( global_l2_sq );
    const T h1_norm      = std::sqrt( global_h1_sq );
    const auto stats     = compute_timing_statistics( wall_times );
    fftm::test::detail::log_tracked_memory_with_external_mpi(
        log, comm_info, distributed_fft, "test=fftm_v4_4d",
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( rhs ), fftm::test::detail::array_bytes( exact_solution ),
            fftm::test::detail::array_bytes( exact_dx ), fftm::test::detail::array_bytes( exact_dy ),
            fftm::test::detail::array_bytes( exact_dz ), fftm::test::detail::array_bytes( exact_dw ),
            fftm::test::detail::array_bytes( numerical_solution ), fftm::test::detail::array_bytes( numerical_dx ),
            fftm::test::detail::array_bytes( numerical_dy ), fftm::test::detail::array_bytes( numerical_dz ),
            fftm::test::detail::array_bytes( numerical_dw ), fftm::test::detail::array_bytes( solution_error_sq ),
            fftm::test::detail::array_bytes( gradient_error_sq ), fftm::test::detail::array_bytes( rhs_hat ),
            fftm::test::detail::array_bytes( solution_hat ), fftm::test::detail::array_bytes( dx_hat ),
            fftm::test::detail::array_bytes( dy_hat ), fftm::test::detail::array_bytes( dz_hat ),
            fftm::test::detail::array_bytes( dw_hat ) ) ),
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( rhs ), fftm::test::detail::array_bytes( exact_solution ),
            fftm::test::detail::array_bytes( exact_dx ), fftm::test::detail::array_bytes( exact_dy ),
            fftm::test::detail::array_bytes( exact_dz ), fftm::test::detail::array_bytes( exact_dw ),
            fftm::test::detail::array_bytes( numerical_solution ), fftm::test::detail::array_bytes( numerical_dx ),
            fftm::test::detail::array_bytes( numerical_dy ), fftm::test::detail::array_bytes( numerical_dz ),
            fftm::test::detail::array_bytes( numerical_dw ), fftm::test::detail::array_bytes( solution_error_sq ),
            fftm::test::detail::array_bytes( gradient_error_sq ), fftm::test::detail::array_bytes( rhs_hat ),
            fftm::test::detail::array_bytes( solution_hat ), fftm::test::detail::array_bytes( dx_hat ),
            fftm::test::detail::array_bytes( dy_hat ), fftm::test::detail::array_bytes( dz_hat ),
            fftm::test::detail::array_bytes( dw_hat ) ) )
    );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "test=fftm_v4_4d, strategy=%s, mode=%s, grid=(%zu,%zu,%zu), Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, "
            "warmup=%d, times=%d: L2=%.8e, H1=%.8e, avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
            fftm_t::strategy_name_4d(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ), p1, p2, p3,
            options.nx, options.ny, options.nz, options.nw, options.warmup, options.times, l2_norm, h1_norm,
            stats.mean, stats.stddev
        );
    }

    return std::max( l2_norm, h1_norm ) <= static_cast<T>( options.threshold ) ? 0 : 1;
}

template <class DistStrategy4D, class RefStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_case(
    std::integral_constant<int, 1>, strategy_kind strategy, log_mpi_t &log, const options_t &options,
    const mpi_comm_t &comm_info
)
{
    return run_forward_reference_compare<DistStrategy4D, RefStrategy4D, Mode>( strategy, log, options, comm_info );
}

template <class DistStrategy4D, class RefStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_case(
    std::integral_constant<int, 0>, strategy_kind strategy, log_mpi_t &log, const options_t &options,
    const mpi_comm_t &comm_info
)
{
    return run_forward_random_benchmark<DistStrategy4D, Mode>( strategy, log, options, comm_info );
}

template <class DistStrategy4D, class RefStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_case(
    std::integral_constant<int, 2>, strategy_kind strategy, log_mpi_t &log, const options_t &options,
    const mpi_comm_t &comm_info
)
{
    return run_backward_random_benchmark<DistStrategy4D, Mode>( strategy, log, options, comm_info );
}

template <class DistStrategy4D, class RefStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_case(
    std::integral_constant<int, 3>, strategy_kind strategy, log_mpi_t &log, const options_t &options,
    const mpi_comm_t &comm_info
)
{
    return run_roundtrip_random_test<DistStrategy4D, Mode>( strategy, log, options, comm_info );
}

template <class DistStrategy4D, class RefStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_case(
    std::integral_constant<int, 4>, strategy_kind strategy, log_mpi_t &log, const options_t &options,
    const mpi_comm_t &comm_info
)
{
    return run_periodic_laplacian_test<DistStrategy4D, Mode>( strategy, log, options, comm_info );
}

template <fftm::mpi_transpose_3d_mode Mode>
int run_for_strategy_kind(
    strategy_kind strategy, log_mpi_t &log, const options_t &options, const mpi_comm_t &comm_info
)
{
    switch ( strategy )
    {
    case strategy_kind::pencil_pencil:
        return run_case<
            fftm::strategy_4d_pencil_pencil_mpi<Mode>, fftm::strategy_4d_pencil_pencil<fftm::transpose_backend::direct>,
            Mode>( std::integral_constant<int, FFTM_EGGER_TESTCASE>(), strategy, log, options, comm_info );
    case strategy_kind::slab_slab:
        return run_case<
            fftm::strategy_4d_slab_slab_mpi<Mode>, fftm::strategy_4d_slab_slab<fftm::transpose_backend::direct>, Mode>(
            std::integral_constant<int, FFTM_EGGER_TESTCASE>(), strategy, log, options, comm_info
        );
    }

    return 1;
}

int dispatch_mode(
    strategy_kind strategy, log_mpi_t &log, const options_t &options, const mpi_comm_t &comm_info
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
    mpi_wrap_t mpi( argc, argv );
    auto       comm_info = mpi.comm_world();
    log_mpi_t  log;

    try
    {
        test_env_t::init_device( log, comm_info );

        const options_t options =
            fftm::test::detail::parse_fftm_4d_test_options( argc, argv, FFTM_VERSIONED_TEST_BINARY, true, true, true );

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

        if ( comm_info.myid == 0 && FFTM_EGGER_TESTCASE != 0 && FFTM_EGGER_TESTCASE != 2 )
        {
            if ( failed == 0 )
                log.info( "PASSED" );
            else
                log.error( "FAILED" );
        }

        return failed;
    }
    catch ( const std::exception &ex )
    {
        if ( comm_info.myid == 0 )
            log.error( scfd::utils::nested_exception_to_multistring( ex ) );
        return 1;
    }
}

#endif
