#ifndef FFTM_EXAMPLES_TURBULENCE_SPACETIME_KERNELS_H
#define FFTM_EXAMPLES_TURBULENCE_SPACETIME_KERNELS_H

#include <cmath>

#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>

namespace fftm
{
namespace examples
{
namespace turbulence
{

using spacetime_index3_t = scfd::static_vec::vec<int, 3>;
using spacetime_index4_t = scfd::static_vec::vec<int, 4>;

template <class SnapshotArray, class Field4D>
struct insert_snapshot_functor
{
    SnapshotArray snapshot;
    Field4D       field;
    int           frame;

    __DEVICE_TAG__ void operator()( const spacetime_index3_t &idx ) const
    {
        field( idx[0], idx[1], idx[2], frame ) = snapshot( idx );
    }
};

template <class Real, class Field4D>
struct temporal_preprocess_functor
{
    Field4D field;
    int     frames;
    bool    subtract_mean;
    bool    hann_window;
    Real    window_scale;

    __DEVICE_TAG__ void operator()( const spacetime_index3_t &idx ) const
    {
        Real mean = Real( 0 );
        if ( subtract_mean )
        {
            for ( int frame = 0; frame < frames; ++frame )
                mean += field( idx[0], idx[1], idx[2], frame );
            mean /= static_cast<Real>( frames );
        }

        const Real two_pi = Real( 6.283185307179586476925286766559 );
        for ( int frame = 0; frame < frames; ++frame )
        {
            Real value = field( idx[0], idx[1], idx[2], frame ) - mean;
            if ( hann_window )
            {
                const Real phase = two_pi * static_cast<Real>( frame ) / static_cast<Real>( frames - 1 );
                value *= Real( 0.5 ) * ( Real( 1 ) - cos( phase ) ) * window_scale;
            }
            field( idx[0], idx[1], idx[2], frame ) = value;
        }
    }
};

template <class Real, class NativeSpectrum, class MetricArray>
struct spectral_power_functor
{
    NativeSpectrum spectrum;
    MetricArray    metric;
    int            nx;
    int            ny;
    int            nz;
    int            nw;
    int            start_x;
    int            start_z;
    int            start_w;
    int            start_y;
    int            mode_min;
    int            mode_max;
    int            spatial_cutoff;

    __DEVICE_TAG__ static int signed_wave_number( int value, int size )
    {
        return value <= size / 2 ? value : value - size;
    }

    __DEVICE_TAG__ void operator()( const spacetime_index4_t &idx ) const
    {
        const int kx = signed_wave_number( start_x + idx[0], nx );
        const int kz = signed_wave_number( start_z + idx[1], nz );
        const int kw = start_w + idx[2];
        const int ky = signed_wave_number( start_y + idx[3], ny );
        const bool temporal_keep = kw >= mode_min && kw <= mode_max;
        const bool spatial_keep =
            spatial_cutoff <= 0 ||
            ( abs( kx ) <= spatial_cutoff && abs( ky ) <= spatial_cutoff && abs( kz ) <= spatial_cutoff );
        if ( !temporal_keep || !spatial_keep )
        {
            metric( idx ) = Real( 0 );
            return;
        }

        const auto value = spectrum( idx );
        const Real hermitian_weight = kw == 0 || ( nw % 2 == 0 && kw == nw / 2 ) ? Real( 1 ) : Real( 2 );
        metric( idx ) = hermitian_weight * ( value.x * value.x + value.y * value.y );
    }
};

template <class NativeSpectrum>
struct spacetime_bandpass_functor
{
    NativeSpectrum spectrum;
    int            nx;
    int            ny;
    int            nz;
    int            start_x;
    int            start_z;
    int            start_w;
    int            start_y;
    int            mode_min;
    int            mode_max;
    int            spatial_cutoff;

    __DEVICE_TAG__ static int signed_wave_number( int value, int size )
    {
        return value <= size / 2 ? value : value - size;
    }

    __DEVICE_TAG__ void operator()( const spacetime_index4_t &idx ) const
    {
        const int kx = signed_wave_number( start_x + idx[0], nx );
        const int kz = signed_wave_number( start_z + idx[1], nz );
        const int kw = start_w + idx[2];
        const int ky = signed_wave_number( start_y + idx[3], ny );
        const bool temporal_keep = kw >= mode_min && kw <= mode_max;
        const bool spatial_keep =
            spatial_cutoff <= 0 ||
            ( abs( kx ) <= spatial_cutoff && abs( ky ) <= spatial_cutoff && abs( kz ) <= spatial_cutoff );
        if ( temporal_keep && spatial_keep )
            return;
        spectrum( idx ).x = 0;
        spectrum( idx ).y = 0;
    }
};

template <class Real, class Field4D>
struct scale_spacetime_functor
{
    Field4D field;
    Real    scale;

    __DEVICE_TAG__ void operator()( const spacetime_index4_t &idx ) const
    {
        field( idx ) *= scale;
    }
};

template <class Field4D, class SnapshotArray>
struct extract_snapshot_functor
{
    Field4D       field;
    SnapshotArray snapshot;
    int           frame;

    __DEVICE_TAG__ void operator()( const spacetime_index3_t &idx ) const
    {
        snapshot( idx ) = static_cast<float>( field( idx[0], idx[1], idx[2], frame ) );
    }
};

} // namespace turbulence
} // namespace examples
} // namespace fftm

#endif
