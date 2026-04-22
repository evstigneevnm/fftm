#ifndef __FFTM_TESTS_DETAIL_VERSIONED_FFTS_4D_TEST_DRIVER_H__
#define __FFTM_TESTS_DETAIL_VERSIONED_FFTS_4D_TEST_DRIVER_H__

#ifndef FFTM_EGGER_TESTCASE
#error "FFTM_EGGER_TESTCASE must be defined before including versioned_ffts_4d_test_driver.h"
#endif

#ifndef FFTM_VERSIONED_TEST_BINARY
#error "FFTM_VERSIONED_TEST_BINARY must be defined before including versioned_ffts_4d_test_driver.h"
#endif

#ifndef FFTM_TEST_ENV
#error "FFTM_TEST_ENV must be defined before including versioned_ffts_4d_test_driver.h"
#endif

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/scalar_traits.h>
#include <scfd/utils/system_timer_event.h>

#include <ffts.hpp>

#include "fft_benchmark_common.h"
#include "fft_benchmark_options.h"

namespace
{

using test_env_t    = FFTM_TEST_ENV;
using ffts_base_t   = fftm::ffts<typename test_env_t::base_fft_t, typename test_env_t::backend_t>;
using T             = typename ffts_base_t::real;
using base_fft_t    = typename ffts_base_t::base_fft_type;
using backend_t     = typename ffts_base_t::backend_type;
using runtime_api_t = typename ffts_base_t::runtime_api;
using log_std_t     = typename test_env_t::log_std_t;
using memory_t      = typename ffts_base_t::memory_t;
using reduce_t      = typename backend_t::reduce_type;
using for_each_t    = typename backend_t::template for_each_nd_type<4, int>;
using idx_t         = scfd::static_vec::vec<int, 4>;
using rect_t        = scfd::static_vec::rect<int, 4>;
using options_t     = fftm::test::detail::ffts_4d_benchmark_options<T>;
using strategy_kind = fftm::test::detail::ffts_4d_strategy_kind;

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

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const T x    = hx * static_cast<T>( idx[0] );
        const T y    = hy * static_cast<T>( idx[1] );
        const T z    = hz * static_cast<T>( idx[2] );
        const T w    = hw * static_cast<T>( idx[3] );
        array( idx ) = sample_value( x, y, z, w );
    }
};

template <class LocalArray, class RefArray, class ErrorArray>
struct compare_forward_output_functor
{
    LocalArray local;
    RefArray   reference;
    ErrorArray diff_sq;
    ErrorArray ref_sq;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const auto actual   = local( idx );
        const auto expected = reference( idx );

        const T diff_re = actual.x - expected.x;
        const T diff_im = actual.y - expected.y;
        const T ref_re  = expected.x;
        const T ref_im  = expected.y;

        const std::size_t lin = local.calc_lin_index( idx[0], idx[1], idx[2], idx[3] );
        diff_sq( lin )        = diff_re * diff_re + diff_im * diff_im;
        ref_sq( lin )         = ref_re * ref_re + ref_im * ref_im;
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
        array( idx ).x = fftm::test::detail::unit_random_from_state<T>( state );
        array( idx ).y = fftm::test::detail::unit_random_from_state<T>( state ^ 0x369DEA0F31A53F85ull );
    }
};

template <class ComplexArray>
struct solve_poisson_4d_functor
{
    ComplexArray rhs_hat;
    ComplexArray solution_hat;
    int          nx;
    int          ny;
    int          nz;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const int gy = idx[0];
        const int gz = idx[1];
        const int gw = idx[2];
        const int kx = idx[3] <= nx / 2 ? idx[3] : idx[3] - nx;
        const int ky = gy <= ny / 2 ? gy : gy - ny;
        const int kz = gz <= nz / 2 ? gz : gz - nz;
        const int kw = gw;

        const T k2 = static_cast<T>( kx * kx + ky * ky + kz * kz + kw * kw );
        if ( k2 == T( 0 ) )
        {
            solution_hat( idx ).x = T( 0 );
            solution_hat( idx ).y = T( 0 );
            return;
        }

        const T scale         = -T( 1 ) / k2;
        solution_hat( idx ).x = scale * rhs_hat( idx ).x;
        solution_hat( idx ).y = scale * rhs_hat( idx ).y;
    }
};

template <class ComplexArray>
struct poisson_4d_derivative_spectra_functor
{
    ComplexArray solution_hat;
    ComplexArray dx_hat;
    ComplexArray dy_hat;
    ComplexArray dz_hat;
    ComplexArray dw_hat;
    int          nx;
    int          ny;
    int          nz;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const int gy = idx[0];
        const int gz = idx[1];
        const int gw = idx[2];
        const int kx = idx[3] <= nx / 2 ? idx[3] : idx[3] - nx;
        const int ky = gy <= ny / 2 ? gy : gy - ny;
        const int kz = gz <= nz / 2 ? gz : gz - nz;
        const int kw = gw;

        const T real_part = solution_hat( idx ).x;
        const T imag_part = solution_hat( idx ).y;

        dx_hat( idx ).x = -static_cast<T>( kx ) * imag_part;
        dx_hat( idx ).y = static_cast<T>( kx ) * real_part;

        dy_hat( idx ).x = -static_cast<T>( ky ) * imag_part;
        dy_hat( idx ).y = static_cast<T>( ky ) * real_part;

        dz_hat( idx ).x = -static_cast<T>( kz ) * imag_part;
        dz_hat( idx ).y = static_cast<T>( kz ) * real_part;

        dw_hat( idx ).x = -static_cast<T>( kw ) * imag_part;
        dw_hat( idx ).y = static_cast<T>( kw ) * real_part;
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

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const T x = hx * static_cast<T>( idx[0] );
        const T y = hy * static_cast<T>( idx[1] );
        const T z = hz * static_cast<T>( idx[2] );
        const T w = hw * static_cast<T>( idx[3] );

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

template <class Strategy4D>
int run_forward_random_benchmark( log_std_t &log, const options_t &options )
{
    using ffts_t       = fftm::ffts<base_fft_t, backend_t, Strategy4D>;
    using real_array_t = typename ffts_t::template real_array_t<4>;
    using hat_array_t  = typename ffts_t::template complex_array_t<4>;

    ffts_t fft;
    fft.init( options.nx, options.ny, options.nz, options.nw );

    real_array_t work;
    hat_array_t  hat;
    work.init( options.nx, options.ny, options.nz, options.nw );
    hat.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );

    for_each_t for_each;
    for_each.block_size = 128;

    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );

    for ( int iter = 0; iter < options.times; ++iter )
    {
        const unsigned long long seed = 0xABCDEF0123456789ull + static_cast<unsigned long long>( iter );
        for_each(
            fftm::test::detail::fill_random_real_4d_functor<T, idx_t, real_array_t>{ work, seed, 0, 0, 0, 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        fft.forward( work, hat );
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );
    }

    const auto stats = fftm::test::detail::compute_timing_statistics( wall_times );

    log.info_f(
        "test=ffts_v0_4d, strategy=%s, Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, times=%d: avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
        ffts_t::strategy_name(), options.nx, options.ny, options.nz, options.nw, options.times, stats.mean, stats.stddev
    );

    return 0;
}

template <class Strategy4D>
int run_forward_reference_compare( log_std_t &log, const options_t &options )
{
    using ffts_t       = fftm::ffts<base_fft_t, backend_t, Strategy4D>;
    using ref_ffts_t   = fftm::ffts<base_fft_t, backend_t, fftm::strategy_4d_pencil_pencil<fftm::transpose_backend::direct>>;
    using real_array_t = typename ffts_t::template real_array_t<4>;
    using hat_array_t  = typename ffts_t::template complex_array_t<4>;
    using ref_real_t   = typename ref_ffts_t::template real_array_t<4>;
    using ref_hat_t    = typename ref_ffts_t::template complex_array_t<4>;
    using error_array_t = scfd::arrays::array_nd<T, 1, memory_t>;

    ffts_t     fft;
    ref_ffts_t reference_fft;
    fft.init( options.nx, options.ny, options.nz, options.nw );
    reference_fft.init( options.nx, options.ny, options.nz, options.nw );

    real_array_t input;
    hat_array_t  output;
    ref_real_t   ref_input;
    ref_hat_t    ref_output;
    input.init( options.nx, options.ny, options.nz, options.nw );
    output.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );
    ref_input.init( options.nx, options.ny, options.nz, options.nw );
    ref_output.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );

    const T l  = T( 2 ) * scfd::utils::scalar_traits<T>::pi();
    const T hx = l / static_cast<T>( options.nx );
    const T hy = l / static_cast<T>( options.ny );
    const T hz = l / static_cast<T>( options.nz );
    const T hw = l / static_cast<T>( options.nw );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    for_each( fill_input_functor<real_array_t>{ input, hx, hy, hz, hw }, make_range( input ) );
    for_each( fill_input_functor<ref_real_t>{ ref_input, hx, hy, hz, hw }, make_range( ref_input ) );
    for_each.wait();

    reference_fft.forward( ref_input, ref_output );

    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );
    T max_forward_rel_l2 = T( 0 );

    for ( int iter = 0; iter < options.times; ++iter )
    {
        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        fft.forward( input, output );
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );

        error_array_t diff_sq;
        error_array_t ref_sq;
        diff_sq.init( output.total_size() );
        ref_sq.init( output.total_size() );

        for_each(
            compare_forward_output_functor<hat_array_t, ref_hat_t, error_array_t>{ output, ref_output, diff_sq, ref_sq },
            make_range( output )
        );
        for_each.wait();

        const T forward_diff_sq = reduce( diff_sq.size(), diff_sq.raw_ptr(), T( 0 ) );
        const T forward_ref_sq  = reduce( ref_sq.size(), ref_sq.raw_ptr(), T( 0 ) );
        const T forward_rel_l2  = std::sqrt( forward_diff_sq / forward_ref_sq );
        if ( forward_rel_l2 > max_forward_rel_l2 )
            max_forward_rel_l2 = forward_rel_l2;
    }

    const auto stats = fftm::test::detail::compute_timing_statistics( wall_times );

    log.info_f(
        "test=ffts_v1_4d, strategy=%s, Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, times=%d: forward_rel_l2=%.8e, avg_wall_ms=%.8e, "
        "stddev_wall_ms=%.8e",
        ffts_t::strategy_name(), options.nx, options.ny, options.nz, options.nw, options.times, max_forward_rel_l2,
        stats.mean, stats.stddev
    );

    return max_forward_rel_l2 <= options.epsilon ? 0 : 1;
}

template <class Strategy4D>
int run_backward_random_benchmark( log_std_t &log, const options_t &options )
{
    using ffts_t       = fftm::ffts<base_fft_t, backend_t, Strategy4D>;
    using real_array_t = typename ffts_t::template real_array_t<4>;
    using hat_array_t  = typename ffts_t::template complex_array_t<4>;

    ffts_t fft;
    fft.init( options.nx, options.ny, options.nz, options.nw );

    hat_array_t  hat;
    real_array_t work;
    hat.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );
    work.init( options.nx, options.ny, options.nz, options.nw );

    for_each_t for_each;
    for_each.block_size = 128;

    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );

    for ( int iter = 0; iter < options.times; ++iter )
    {
        const unsigned long long seed = 0x1020304050607080ull + static_cast<unsigned long long>( iter );
        for_each( fill_random_complex_functor<hat_array_t>{ hat, seed }, make_range( hat ) );
        for_each.wait();

        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        fft.backward( hat, work );
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );
    }

    const auto stats = fftm::test::detail::compute_timing_statistics( wall_times );

    log.info_f(
        "test=ffts_v2_4d, strategy=%s, Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, times=%d: avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
        ffts_t::strategy_name(), options.nx, options.ny, options.nz, options.nw, options.times, stats.mean, stats.stddev
    );

    return 0;
}

template <class Strategy4D>
int run_roundtrip_random_test( log_std_t &log, const options_t &options )
{
    using ffts_t       = fftm::ffts<base_fft_t, backend_t, Strategy4D>;
    using real_array_t = typename ffts_t::template real_array_t<4>;
    using hat_array_t  = typename ffts_t::template complex_array_t<4>;

    ffts_t fft;
    fft.init( options.nx, options.ny, options.nz, options.nw );

    real_array_t work;
    hat_array_t  hat;
    work.init( options.nx, options.ny, options.nz, options.nw );
    hat.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    const T normalization = T( 1 ) / static_cast<T>( options.nx * options.ny * options.nz * options.nw );
    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );
    T max_l2 = T( 0 );

    for ( int iter = 0; iter < options.times; ++iter )
    {
        const unsigned long long seed = 0xABCDEF0123456789ull + static_cast<unsigned long long>( iter );
        for_each(
            fftm::test::detail::fill_random_real_4d_functor<T, idx_t, real_array_t>{ work, seed, 0, 0, 0, 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        fft.forward( work, hat );
        fft.backward( hat, work );
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );

        for_each(
            fftm::test::detail::scale_real_functor<T, idx_t, real_array_t>{ work, normalization },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        for_each(
            fftm::test::detail::overwrite_with_random_diff_square_4d_functor<T, idx_t, real_array_t>{ work, seed, 0, 0, 0, 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        const T diff_l2 =
            std::sqrt( reduce( work.size(), work.raw_ptr(), T( 0 ) ) / static_cast<T>( work.size() ) );
        if ( diff_l2 > max_l2 )
            max_l2 = diff_l2;
    }

    const auto stats = fftm::test::detail::compute_timing_statistics( wall_times );

    log.info_f(
        "test=ffts_v3_4d, strategy=%s, Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, times=%d: max_l2=%.8e, avg_wall_ms=%.8e, "
        "stddev_wall_ms=%.8e",
        ffts_t::strategy_name(), options.nx, options.ny, options.nz, options.nw, options.times, max_l2, stats.mean,
        stats.stddev
    );

    return max_l2 <= options.epsilon ? 0 : 1;
}

template <class Strategy4D>
int run_periodic_laplacian_test( log_std_t &log, const options_t &options )
{
    using ffts_t       = fftm::ffts<base_fft_t, backend_t, Strategy4D>;
    using real_array_t = typename ffts_t::template real_array_t<4>;
    using hat_array_t  = typename ffts_t::template complex_array_t<4>;

    ffts_t fft;
    fft.init( options.nx, options.ny, options.nz, options.nw );

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

    rhs.init( options.nx, options.ny, options.nz, options.nw );
    exact_solution.init( options.nx, options.ny, options.nz, options.nw );
    exact_dx.init( options.nx, options.ny, options.nz, options.nw );
    exact_dy.init( options.nx, options.ny, options.nz, options.nw );
    exact_dz.init( options.nx, options.ny, options.nz, options.nw );
    exact_dw.init( options.nx, options.ny, options.nz, options.nw );
    numerical_solution.init( options.nx, options.ny, options.nz, options.nw );
    numerical_dx.init( options.nx, options.ny, options.nz, options.nw );
    numerical_dy.init( options.nx, options.ny, options.nz, options.nw );
    numerical_dz.init( options.nx, options.ny, options.nz, options.nw );
    numerical_dw.init( options.nx, options.ny, options.nz, options.nw );
    solution_error_sq.init( options.nx, options.ny, options.nz, options.nw );
    gradient_error_sq.init( options.nx, options.ny, options.nz, options.nw );
    rhs_hat.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );
    solution_hat.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );
    dx_hat.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );
    dy_hat.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );
    dz_hat.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );
    dw_hat.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );

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
            rhs, exact_solution, exact_dx, exact_dy, exact_dz, exact_dw, hx, hy, hz, hw },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs )
    );
    for_each.wait();

    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );

    for ( int iter = 0; iter < options.times; ++iter )
    {
        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        fft.forward( rhs, rhs_hat );
        for_each(
            solve_poisson_4d_functor<hat_array_t>{
                rhs_hat, solution_hat, static_cast<int>( options.nx ), static_cast<int>( options.ny ),
                static_cast<int>( options.nz ) },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs_hat )
        );
        for_each.wait();
        fft.backward( solution_hat, numerical_solution );
        for_each(
            fftm::test::detail::scale_real_functor<T, idx_t, real_array_t>{ numerical_solution, normalization },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_solution )
        );
        for_each.wait();
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );
    }

    fft.forward( rhs, rhs_hat );
    for_each(
        solve_poisson_4d_functor<hat_array_t>{
            rhs_hat, solution_hat, static_cast<int>( options.nx ), static_cast<int>( options.ny ),
            static_cast<int>( options.nz ) },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( rhs_hat )
    );
    for_each.wait();

    for_each(
        poisson_4d_derivative_spectra_functor<hat_array_t>{
            solution_hat, dx_hat, dy_hat, dz_hat, dw_hat, static_cast<int>( options.nx ),
            static_cast<int>( options.ny ), static_cast<int>( options.nz ) },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( solution_hat )
    );
    for_each.wait();

    fft.backward( dx_hat, numerical_dx );
    fft.backward( dy_hat, numerical_dy );
    fft.backward( dz_hat, numerical_dz );
    fft.backward( dw_hat, numerical_dw );
    for_each(
        fftm::test::detail::scale_real_functor<T, idx_t, real_array_t>{ numerical_dx, normalization },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_dx )
    );
    for_each(
        fftm::test::detail::scale_real_functor<T, idx_t, real_array_t>{ numerical_dy, normalization },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_dy )
    );
    for_each(
        fftm::test::detail::scale_real_functor<T, idx_t, real_array_t>{ numerical_dz, normalization },
        fftm::test::detail::make_range_4d<idx_t, rect_t>( numerical_dz )
    );
    for_each(
        fftm::test::detail::scale_real_functor<T, idx_t, real_array_t>{ numerical_dw, normalization },
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

    const T l2_norm  = std::sqrt( reduce( solution_error_sq.size(), solution_error_sq.raw_ptr(), T( 0 ) ) * cell_volume );
    const T h1_norm  = std::sqrt( reduce( gradient_error_sq.size(), gradient_error_sq.raw_ptr(), T( 0 ) ) * cell_volume );
    const auto stats = fftm::test::detail::compute_timing_statistics( wall_times );

    log.info_f(
        "test=ffts_v4_4d, strategy=%s, Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, times=%d: L2=%.8e, H1=%.8e, avg_wall_ms=%.8e, "
        "stddev_wall_ms=%.8e",
        ffts_t::strategy_name(), options.nx, options.ny, options.nz, options.nw, options.times, l2_norm, h1_norm,
        stats.mean, stats.stddev
    );

    return std::max( l2_norm, h1_norm ) <= options.epsilon ? 0 : 1;
}

template <class Strategy4D>
int run_case( std::integral_constant<int, 0>, log_std_t &log, const options_t &options )
{
    return run_forward_random_benchmark<Strategy4D>( log, options );
}

template <class Strategy4D>
int run_case( std::integral_constant<int, 1>, log_std_t &log, const options_t &options )
{
    return run_forward_reference_compare<Strategy4D>( log, options );
}

template <class Strategy4D>
int run_case( std::integral_constant<int, 2>, log_std_t &log, const options_t &options )
{
    return run_backward_random_benchmark<Strategy4D>( log, options );
}

template <class Strategy4D>
int run_case( std::integral_constant<int, 3>, log_std_t &log, const options_t &options )
{
    return run_roundtrip_random_test<Strategy4D>( log, options );
}

template <class Strategy4D>
int run_case( std::integral_constant<int, 4>, log_std_t &log, const options_t &options )
{
    return run_periodic_laplacian_test<Strategy4D>( log, options );
}

template <int Testcase>
int dispatch_strategy( log_std_t &log, const options_t &options, strategy_kind strategy )
{
    switch ( strategy )
    {
    case strategy_kind::pencil_direct:
        return run_case<fftm::strategy_4d_pencil_pencil<fftm::transpose_backend::direct>>(
            std::integral_constant<int, Testcase>(), log, options
        );
    case strategy_kind::pencil_memcpy:
        return run_case<fftm::strategy_4d_pencil_pencil<fftm::transpose_backend::memcpy>>(
            std::integral_constant<int, Testcase>(), log, options
        );
    case strategy_kind::slab_direct:
        return run_case<fftm::strategy_4d_slab_slab<fftm::transpose_backend::direct>>(
            std::integral_constant<int, Testcase>(), log, options
        );
    case strategy_kind::slab_memcpy:
        return run_case<fftm::strategy_4d_slab_slab<fftm::transpose_backend::memcpy>>(
            std::integral_constant<int, Testcase>(), log, options
        );
    }
    return 1;
}

} // namespace

int main( int argc, char *argv[] )
{
    log_std_t log;

    try
    {
        test_env_t::init_device( log );
        const options_t options =
            fftm::test::detail::parse_ffts_4d_benchmark_options<T>( argc, argv, FFTM_VERSIONED_TEST_BINARY );

        if ( options.run_all )
        {
            int failed = 0;
            failed |= dispatch_strategy<FFTM_EGGER_TESTCASE>( log, options, strategy_kind::pencil_direct );
            failed |= dispatch_strategy<FFTM_EGGER_TESTCASE>( log, options, strategy_kind::pencil_memcpy );
            failed |= dispatch_strategy<FFTM_EGGER_TESTCASE>( log, options, strategy_kind::slab_direct );
            failed |= dispatch_strategy<FFTM_EGGER_TESTCASE>( log, options, strategy_kind::slab_memcpy );
            return failed;
        }

        return dispatch_strategy<FFTM_EGGER_TESTCASE>( log, options, options.strategy );
    }
    catch ( const std::exception &ex )
    {
        log.error( scfd::utils::nested_exception_to_multistring( ex ) );
        return 1;
    }
}

#endif
