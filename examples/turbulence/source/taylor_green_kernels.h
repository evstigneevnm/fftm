#ifndef FFTM_EXAMPLES_TURBULENCE_TAYLOR_GREEN_KERNELS_H
#define FFTM_EXAMPLES_TURBULENCE_TAYLOR_GREEN_KERNELS_H

#include <cmath>

#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>

namespace fftm
{
namespace examples
{
namespace turbulence
{

using index3_t = scfd::static_vec::vec<int, 3>;

struct spectral_coordinates
{
    int  nx;
    int  ny;
    int  nz;
    int  y_start;
    int  z_start;
    bool xyz_layout;

    __DEVICE_TAG__ static int signed_wave_number( int index, int size )
    {
        return index <= size / 2 ? index : index - size;
    }

    __DEVICE_TAG__ void wave_numbers( const index3_t &idx, int &kx, int &ky, int &kz ) const
    {
        kx = signed_wave_number( idx[0], nx );
        const int global_y = y_start + ( xyz_layout ? idx[1] : idx[2] );
        const int global_z = z_start + ( xyz_layout ? idx[2] : idx[1] );
        ky = signed_wave_number( global_y, ny );
        kz = global_z;
    }

    __DEVICE_TAG__ bool retained_by_two_thirds_rule( int kx, int ky, int kz ) const
    {
        return abs( kx ) <= nx / 3 && abs( ky ) <= ny / 3 && kz <= nz / 3;
    }
};

template <class Real, class RealArray>
struct fill_taylor_green_functor
{
    RealArray ux;
    RealArray uy;
    RealArray uz;
    Real      hx;
    Real      hy;
    Real      hz;
    int       x_start;
    int       y_start;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        const Real x = hx * static_cast<Real>( x_start + idx[0] );
        const Real y = hy * static_cast<Real>( y_start + idx[1] );
        const Real z = hz * static_cast<Real>( idx[2] );
        const Real cz = cos( z );
        ux( idx ) = sin( x ) * cos( y ) * cz;
        uy( idx ) = -cos( x ) * sin( y ) * cz;
        uz( idx ) = Real( 0 );
    }
};

template <class Real, class RealArray>
struct scale_real_functor
{
    RealArray field;
    Real      scale;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        field( idx ) *= scale;
    }
};

template <class ComplexArray>
struct zero_complex_functor
{
    ComplexArray field;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        field( idx ).x = 0;
        field( idx ).y = 0;
    }
};

template <class ComplexArray>
struct vorticity_spectrum_functor
{
    ComplexArray        ux;
    ComplexArray        uy;
    ComplexArray        uz;
    ComplexArray        omega;
    spectral_coordinates coordinates;
    int                 component;
    int                 output_cutoff;

    template <class Complex>
    __DEVICE_TAG__ static Complex multiply_i_wave( const Complex &value, int wave )
    {
        Complex result;
        result.x = -static_cast<decltype( result.x )>( wave ) * value.y;
        result.y = static_cast<decltype( result.y )>( wave ) * value.x;
        return result;
    }

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        int kx = 0, ky = 0, kz = 0;
        coordinates.wave_numbers( idx, kx, ky, kz );

        auto value = omega( idx );
        if ( component == 0 )
        {
            const auto first  = multiply_i_wave( uz( idx ), ky );
            const auto second = multiply_i_wave( uy( idx ), kz );
            value.x = first.x - second.x;
            value.y = first.y - second.y;
        }
        else if ( component == 1 )
        {
            const auto first  = multiply_i_wave( ux( idx ), kz );
            const auto second = multiply_i_wave( uz( idx ), kx );
            value.x = first.x - second.x;
            value.y = first.y - second.y;
        }
        else
        {
            const auto first  = multiply_i_wave( uy( idx ), kx );
            const auto second = multiply_i_wave( ux( idx ), ky );
            value.x = first.x - second.x;
            value.y = first.y - second.y;
        }

        const bool keep_dealiased = coordinates.retained_by_two_thirds_rule( kx, ky, kz );
        const bool keep_output =
            output_cutoff <= 0 ||
            ( abs( kx ) <= output_cutoff && abs( ky ) <= output_cutoff && kz <= output_cutoff );
        if ( !keep_dealiased || !keep_output )
        {
            value.x = 0;
            value.y = 0;
        }
        omega( idx ) = value;
    }
};

template <class ComplexArray>
struct spectral_derivative_functor
{
    ComplexArray         input;
    ComplexArray         derivative;
    spectral_coordinates coordinates;
    int                  direction;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        int kx = 0, ky = 0, kz = 0;
        coordinates.wave_numbers( idx, kx, ky, kz );
        const int wave = direction == 0 ? kx : ( direction == 1 ? ky : kz );
        const auto input_value = input( idx );
        auto       value       = derivative( idx );
        if ( !coordinates.retained_by_two_thirds_rule( kx, ky, kz ) )
        {
            value.x = 0;
            value.y = 0;
        }
        else
        {
            value.x = -static_cast<decltype( value.x )>( wave ) * input_value.y;
            value.y = static_cast<decltype( value.y )>( wave ) * input_value.x;
        }
        derivative( idx ) = value;
    }
};

template <class ComplexArray>
struct scalar_spectral_filter_functor
{
    ComplexArray         field;
    spectral_coordinates coordinates;
    int                  cutoff;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        int kx = 0, ky = 0, kz = 0;
        coordinates.wave_numbers( idx, kx, ky, kz );
        if ( cutoff <= 0 || ( abs( kx ) <= cutoff && abs( ky ) <= cutoff && kz <= cutoff ) )
            return;
        field( idx ).x = 0;
        field( idx ).y = 0;
    }
};

template <class RealArray>
struct rotational_nonlinearity_functor
{
    RealArray ux;
    RealArray uy;
    RealArray uz;
    RealArray wx;
    RealArray wy;
    RealArray wz;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        const auto ux_value = ux( idx );
        const auto uy_value = uy( idx );
        const auto uz_value = uz( idx );
        const auto wx_value = wx( idx );
        const auto wy_value = wy( idx );
        const auto wz_value = wz( idx );

        wx( idx ) = uy_value * wz_value - uz_value * wy_value;
        wy( idx ) = uz_value * wx_value - ux_value * wz_value;
        wz( idx ) = ux_value * wy_value - uy_value * wx_value;
    }
};

template <class ComplexArray>
struct project_and_dealias_functor
{
    ComplexArray         qx;
    ComplexArray         qy;
    ComplexArray         qz;
    spectral_coordinates coordinates;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        int kx = 0, ky = 0, kz = 0;
        coordinates.wave_numbers( idx, kx, ky, kz );

        auto x = qx( idx );
        auto y = qy( idx );
        auto z = qz( idx );
        const int k2 = kx * kx + ky * ky + kz * kz;

        if ( k2 == 0 || !coordinates.retained_by_two_thirds_rule( kx, ky, kz ) )
        {
            x.x = x.y = 0;
            y.x = y.y = 0;
            z.x = z.y = 0;
        }
        else
        {
            const auto dot_real = kx * x.x + ky * y.x + kz * z.x;
            const auto dot_imag = kx * x.y + ky * y.y + kz * z.y;
            const auto inv_k2   = decltype( x.x )( 1 ) / static_cast<decltype( x.x )>( k2 );
            x.x -= static_cast<decltype( x.x )>( kx ) * dot_real * inv_k2;
            x.y -= static_cast<decltype( x.y )>( kx ) * dot_imag * inv_k2;
            y.x -= static_cast<decltype( y.x )>( ky ) * dot_real * inv_k2;
            y.y -= static_cast<decltype( y.y )>( ky ) * dot_imag * inv_k2;
            z.x -= static_cast<decltype( z.x )>( kz ) * dot_real * inv_k2;
            z.y -= static_cast<decltype( z.y )>( kz ) * dot_imag * inv_k2;
        }

        qx( idx ) = x;
        qy( idx ) = y;
        qz( idx ) = z;
    }
};

template <class RealArray>
struct fill_real_functor
{
    RealArray field;
    using real_t = typename RealArray::value_type;
    real_t value;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        field( idx ) = value;
    }
};

template <class RealArray>
struct accumulate_q_square_functor
{
    RealArray q;
    RealArray gradient;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        const auto value = gradient( idx );
        q( idx ) -= decltype( value )( 0.5 ) * value * value;
    }
};

template <class RealArray>
struct accumulate_q_pair_functor
{
    RealArray q;
    RealArray gradient_ij;
    RealArray gradient_ji;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        q( idx ) -= gradient_ij( idx ) * gradient_ji( idx );
    }
};

template <class Real, class ComplexArray>
struct low_storage_rk3_update_functor
{
    ComplexArray         ux;
    ComplexArray         uy;
    ComplexArray         uz;
    ComplexArray         rx;
    ComplexArray         ry;
    ComplexArray         rz;
    ComplexArray         nx;
    ComplexArray         ny;
    ComplexArray         nz;
    spectral_coordinates coordinates;
    Real                 viscosity;
    Real                 dt;
    Real                 stage_a;
    Real                 stage_b;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        int kx = 0, ky = 0, kz = 0;
        coordinates.wave_numbers( idx, kx, ky, kz );
        const int k2 = kx * kx + ky * ky + kz * kz;

        auto u0 = ux( idx );
        auto u1 = uy( idx );
        auto u2 = uz( idx );
        auto r0 = rx( idx );
        auto r1 = ry( idx );
        auto r2 = rz( idx );
        const auto n0 = nx( idx );
        const auto n1 = ny( idx );
        const auto n2 = nz( idx );

        if ( k2 == 0 || !coordinates.retained_by_two_thirds_rule( kx, ky, kz ) )
        {
            u0.x = u0.y = 0;
            u1.x = u1.y = 0;
            u2.x = u2.y = 0;
            r0.x = r0.y = 0;
            r1.x = r1.y = 0;
            r2.x = r2.y = 0;
        }
        else
        {
            const Real diffusion = viscosity * static_cast<Real>( k2 );
            r0.x = stage_a * r0.x + dt * ( n0.x - diffusion * u0.x );
            r0.y = stage_a * r0.y + dt * ( n0.y - diffusion * u0.y );
            r1.x = stage_a * r1.x + dt * ( n1.x - diffusion * u1.x );
            r1.y = stage_a * r1.y + dt * ( n1.y - diffusion * u1.y );
            r2.x = stage_a * r2.x + dt * ( n2.x - diffusion * u2.x );
            r2.y = stage_a * r2.y + dt * ( n2.y - diffusion * u2.y );

            u0.x += stage_b * r0.x;
            u0.y += stage_b * r0.y;
            u1.x += stage_b * r1.x;
            u1.y += stage_b * r1.y;
            u2.x += stage_b * r2.x;
            u2.y += stage_b * r2.y;
        }

        ux( idx ) = u0;
        uy( idx ) = u1;
        uz( idx ) = u2;
        rx( idx ) = r0;
        ry( idx ) = r1;
        rz( idx ) = r2;
    }
};

template <class RealArray>
struct kinetic_energy_density_functor
{
    RealArray ux;
    RealArray uy;
    RealArray uz;
    RealArray result;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        const auto x = ux( idx );
        const auto y = uy( idx );
        const auto z = uz( idx );
        result( idx ) = decltype( x )( 0.5 ) * ( x * x + y * y + z * z );
    }
};

template <class RealArray>
struct enstrophy_density_functor
{
    RealArray wx;
    RealArray wy;
    RealArray wz;
    RealArray result;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        const auto x = wx( idx );
        const auto y = wy( idx );
        const auto z = wz( idx );
        result( idx ) = decltype( x )( 0.5 ) * ( x * x + y * y + z * z );
    }
};

template <class RealArray>
struct speed_functor
{
    RealArray ux;
    RealArray uy;
    RealArray uz;
    RealArray result;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        const auto x = ux( idx );
        const auto y = uy( idx );
        const auto z = uz( idx );
        result( idx ) = sqrt( x * x + y * y + z * z );
    }
};

template <class ComplexArray>
struct divergence_spectrum_functor
{
    ComplexArray         ux;
    ComplexArray         uy;
    ComplexArray         uz;
    ComplexArray         divergence;
    spectral_coordinates coordinates;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        int kx = 0, ky = 0, kz = 0;
        coordinates.wave_numbers( idx, kx, ky, kz );
        const auto x = ux( idx );
        const auto y = uy( idx );
        const auto z = uz( idx );
        auto       result = divergence( idx );
        const auto dot_real = kx * x.x + ky * y.x + kz * z.x;
        const auto dot_imag = kx * x.y + ky * y.y + kz * z.y;
        result.x = -dot_imag;
        result.y = dot_real;
        divergence( idx ) = result;
    }
};

template <class RealArray>
struct square_functor
{
    RealArray input;
    RealArray result;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        const auto value = input( idx );
        result( idx ) = value * value;
    }
};

template <class RealArray, class SnapshotArray>
struct snapshot_scalar_functor
{
    RealArray     input;
    SnapshotArray output;
    int           offset_x;
    int           offset_y;
    int           offset_z;
    int           stride_x;
    int           stride_y;
    int           stride_z;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        output( idx ) = static_cast<float>(
            input(
                offset_x + idx[0] * stride_x, offset_y + idx[1] * stride_y,
                offset_z + idx[2] * stride_z
            )
        );
    }
};

template <class SourceArray, class SnapshotArray>
struct snapshot_omega_z_functor
{
    SourceArray   omega_z;
    SnapshotArray output;
    int           x_offset;
    int           y_offset;
    int           z_offset;
    int           stride_x;
    int           stride_y;
    int           stride_z;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        const index3_t src(
            x_offset + idx[0] * stride_x, y_offset + idx[1] * stride_y, z_offset + idx[2] * stride_z
        );
        output( idx ) = static_cast<float>( omega_z( src ) );
    }
};

template <class SourceArray, class SnapshotArray>
struct snapshot_vorticity_magnitude_functor
{
    SourceArray   omega_x;
    SourceArray   omega_y;
    SourceArray   omega_z;
    SnapshotArray output;
    int           x_offset;
    int           y_offset;
    int           z_offset;
    int           stride_x;
    int           stride_y;
    int           stride_z;

    __DEVICE_TAG__ void operator()( const index3_t &idx ) const
    {
        const index3_t src(
            x_offset + idx[0] * stride_x, y_offset + idx[1] * stride_y, z_offset + idx[2] * stride_z
        );
        const auto x = omega_x( src );
        const auto y = omega_y( src );
        const auto z = omega_z( src );
        output( idx ) = static_cast<float>( sqrt( x * x + y * y + z * z ) );
    }
};

} // namespace turbulence
} // namespace examples
} // namespace fftm

#endif
