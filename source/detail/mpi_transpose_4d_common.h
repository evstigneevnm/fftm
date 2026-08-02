#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_4D_COMMON_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_4D_COMMON_H__

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/system_timer_event.h>

#include "../fft_partitioning.h"
#include "../profiling.h"
#include "mpi_transpose_3d.h"

namespace fftm
{
namespace detail
{

template <class Idx>
__DEVICE_TAG__ inline std::size_t packed_index_4d( const Idx &idx, std::size_t d0, std::size_t d1, std::size_t d2 )
{
    return static_cast<std::size_t>( idx[0] ) +
           d0 * ( static_cast<std::size_t>( idx[1] ) +
                  d1 * ( static_cast<std::size_t>( idx[2] ) + d2 * static_cast<std::size_t>( idx[3] ) ) );
}

template <class Idx>
__DEVICE_TAG__ inline std::size_t
packed_index_4d( int i0, int i1, int i2, int i3, std::size_t d0, std::size_t d1, std::size_t d2 )
{
    return static_cast<std::size_t>( i0 ) +
           d0 * ( static_cast<std::size_t>( i1 ) +
                  d1 * ( static_cast<std::size_t>( i2 ) + d2 * static_cast<std::size_t>( i3 ) ) );
}

template <class Idx>
using rect_4d_t = scfd::static_vec::rect<int, 4>;

template <class Idx>
inline rect_4d_t<Idx> make_range_4d( std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3 )
{
    return rect_4d_t<Idx>(
        Idx( 0, 0, 0, 0 ),
        Idx( static_cast<int>( d0 ), static_cast<int>( d1 ), static_cast<int>( d2 ), static_cast<int>( d3 ) )
    );
}

struct mpi_transpose_4d_stage_timer
{
    using callback_t = void ( * )( void *, const char *, double );

    void set( void *context_, callback_t callback_, const std::string &prefix_ )
    {
        context  = context_;
        callback = callback_;
        prefix   = prefix_;
    }

    bool enabled() const
    {
        return context != nullptr && callback != nullptr && !prefix.empty();
    }

    void record( const char *stage, double ms ) const
    {
        if ( !enabled() )
            return;
        const std::string full_stage = prefix + "/" + stage;
        callback( context, full_stage.c_str(), ms );
    }

    void      *context  = nullptr;
    callback_t callback = nullptr;
    std::string prefix;
};

template <class RuntimeAPI, class Fn>
void time_mpi_transpose_4d_substage(
    const mpi_transpose_4d_stage_timer &timer, const char *direction, const char *phase, Fn fn
)
{
    if ( !timer.enabled() )
    {
        fn();
        return;
    }

    const std::string stage = std::string( direction ) + "/" + phase;
    RuntimeAPI::device_synchronize();
    scfd::utils::system_timer_event begin, end;
    begin.record();
    fn();
    RuntimeAPI::device_synchronize();
    end.record();
    timer.record( stage.c_str(), end.elapsed_time( begin ) );
}

} // namespace detail
} // namespace fftm

#endif // __FFTM_DETAIL_MPI_TRANSPOSE_4D_COMMON_H__
