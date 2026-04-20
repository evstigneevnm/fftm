#ifndef __FFTM_TESTS_DETAIL_POISSON_4D_PROBLEM_H__
#define __FFTM_TESTS_DETAIL_POISSON_4D_PROBLEM_H__

#include <scfd/utils/device_tag.h>
#include <scfd/utils/scalar_traits.h>

namespace fftm
{
namespace test
{
namespace detail
{

template <class T>
struct poisson_4d_problem
{
    __DEVICE_TAG__ static T domain_length()
    {
        return T( 2 ) * scfd::utils::scalar_traits<T>::pi();
    }

    __DEVICE_TAG__ static void
    evaluate( T x, T y, T z, T w, T &rhs, T &exact_solution, T &exact_dx, T &exact_dy, T &exact_dz, T &exact_dw )
    {
        const T pi = scfd::utils::scalar_traits<T>::pi();

        const T dx       = x - pi;
        const T dy       = y - pi;
        const T dz       = z - pi;
        const T dw       = w - pi;
        const T exponent = -( dx * dx + dy * dy + dz * dz + dw * dw );

        const T gaussian = scfd::utils::scalar_traits<T>::exp( exponent );

        const T sin_x = scfd::utils::scalar_traits<T>::sin( x );
        const T cos_x = scfd::utils::scalar_traits<T>::cos( x );
        const T sin_y = scfd::utils::scalar_traits<T>::sin( y );
        const T cos_y = scfd::utils::scalar_traits<T>::cos( y );
        const T sin_z = scfd::utils::scalar_traits<T>::sin( z );
        const T cos_z = scfd::utils::scalar_traits<T>::cos( z );
        const T sin_w = scfd::utils::scalar_traits<T>::sin( w );
        const T cos_w = scfd::utils::scalar_traits<T>::cos( w );

        exact_solution = T( 100 ) * gaussian * sin_x * sin_y * sin_z * sin_w;
        exact_dx       = T( 100 ) * gaussian * ( cos_x - T( 2 ) * dx * sin_x ) * sin_y * sin_z * sin_w;
        exact_dy       = T( 100 ) * gaussian * sin_x * ( cos_y - T( 2 ) * dy * sin_y ) * sin_z * sin_w;
        exact_dz       = T( 100 ) * gaussian * sin_x * sin_y * ( cos_z - T( 2 ) * dz * sin_z ) * sin_w;
        exact_dw       = T( 100 ) * gaussian * sin_x * sin_y * sin_z * ( cos_w - T( 2 ) * dw * sin_w );

        rhs = T( 100 ) * gaussian *
              ( ( T( 4 ) * dx * dx + T( 4 ) * dy * dy + T( 4 ) * dz * dz + T( 4 ) * dw * dw - T( 12 ) ) * sin_x *
                    sin_y * sin_z * sin_w -
                T( 4 ) * dx * cos_x * sin_y * sin_z * sin_w - T( 4 ) * dy * sin_x * cos_y * sin_z * sin_w -
                T( 4 ) * dz * sin_x * sin_y * cos_z * sin_w - T( 4 ) * dw * sin_x * sin_y * sin_z * cos_w );
    }
};

template <class T, class Idx, class RealArray>
struct fill_poisson_4d_rhs_functor
{
    RealArray rhs;
    T         hx;
    T         hy;
    T         hz;
    T         hw;
    T         x0;
    T         y0;
    T         z0;
    T         w0;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
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

        poisson_4d_problem<T>::evaluate(
            x, y, z, w, rhs_value, exact_value, exact_dx_val, exact_dy_val, exact_dz_val, exact_dw_val
        );

        rhs( idx ) = rhs_value;
    }
};

template <class T, class Idx, class RealArray>
struct fill_poisson_4d_problem_functor
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

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
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

        poisson_4d_problem<T>::evaluate(
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

template <class T, class Idx, class RealArray>
struct poisson_4d_error_fields_functor
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

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
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

} // namespace detail
} // namespace test
} // namespace fftm

#endif
