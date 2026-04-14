#ifndef __FFTM_TESTS_DETAIL_POISSON_3D_FFT_COMMON_H__
#define __FFTM_TESTS_DETAIL_POISSON_3D_FFT_COMMON_H__

#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>

namespace fftm
{
namespace test
{
namespace detail
{

template <class Idx, class Rect, class Array>
Rect make_range_3d( const Array &array )
{
    const auto sz = array.size_nd();
    return Rect(
        Idx( 0, 0, 0 ),
        Idx(
            static_cast<int>( sz[0] ),
            static_cast<int>( sz[1] ),
            static_cast<int>( sz[2] )
        )
    );
}

template <class T, class Idx, class ComplexArray>
struct solve_poisson_3d_functor
{
    ComplexArray rhs_hat;
    ComplexArray solution_hat;
    int          nx;
    int          ny;
    int          y_start;
    int          z_start;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const int kx = idx[0] <= nx / 2 ? idx[0] : idx[0] - nx;
        const int gy = y_start + idx[2];
        const int ky = gy <= ny / 2 ? gy : gy - ny;
        const int kz = z_start + idx[1];

        const T k2 = static_cast<T>( kx * kx + ky * ky + kz * kz );
        if ( k2 == T( 0 ) )
        {
            solution_hat( idx ).x = T( 0 );
            solution_hat( idx ).y = T( 0 );
            return;
        }

        const T scale = -T( 1 ) / k2;
        solution_hat( idx ).x = scale * rhs_hat( idx ).x;
        solution_hat( idx ).y = scale * rhs_hat( idx ).y;
    }
};

template <class T, class Idx, class ComplexArray>
struct solve_poisson_3d_in_place_functor
{
    ComplexArray spectrum;
    int          nx;
    int          ny;
    int          y_start;
    int          z_start;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const int kx = idx[0] <= nx / 2 ? idx[0] : idx[0] - nx;
        const int gy = y_start + idx[2];
        const int ky = gy <= ny / 2 ? gy : gy - ny;
        const int kz = z_start + idx[1];

        const T k2 = static_cast<T>( kx * kx + ky * ky + kz * kz );
        if ( k2 == T( 0 ) )
        {
            spectrum( idx ).x = T( 0 );
            spectrum( idx ).y = T( 0 );
            return;
        }

        const T scale = -T( 1 ) / k2;
        spectrum( idx ).x *= scale;
        spectrum( idx ).y *= scale;
    }
};

template <class T, class Idx, class ComplexArray>
struct poisson_3d_derivative_spectra_functor
{
    ComplexArray solution_hat;
    ComplexArray dx_hat;
    ComplexArray dy_hat;
    ComplexArray dz_hat;
    int          nx;
    int          ny;
    int          y_start;
    int          z_start;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const int kx = idx[0] <= nx / 2 ? idx[0] : idx[0] - nx;
        const int gy = y_start + idx[2];
        const int ky = gy <= ny / 2 ? gy : gy - ny;
        const int kz = z_start + idx[1];

        const T real_part = solution_hat( idx ).x;
        const T imag_part = solution_hat( idx ).y;

        dx_hat( idx ).x = -static_cast<T>( kx ) * imag_part;
        dx_hat( idx ).y =  static_cast<T>( kx ) * real_part;

        dy_hat( idx ).x = -static_cast<T>( ky ) * imag_part;
        dy_hat( idx ).y =  static_cast<T>( ky ) * real_part;

        dz_hat( idx ).x = -static_cast<T>( kz ) * imag_part;
        dz_hat( idx ).y =  static_cast<T>( kz ) * real_part;
    }
};

template <class T, class Idx, class RealArray>
struct scale_real_3d_functor
{
    RealArray field;
    T         scale;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        field( idx ) *= scale;
    }
};

template <class T, class Idx, class RealArray>
struct square_real_3d_functor
{
    RealArray field;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const T value = field( idx );
        field( idx ) = value * value;
    }
};

} // namespace detail
} // namespace test
} // namespace fftm

#endif
