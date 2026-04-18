#ifndef __FFTM_TESTS_DETAIL_FFT_BENCHMARK_COMMON_H__
#define __FFTM_TESTS_DETAIL_FFT_BENCHMARK_COMMON_H__

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>

namespace fftm
{
namespace test
{
namespace detail
{

template <class T>
inline T default_benchmark_epsilon()
{
    return T( 10 ) * std::numeric_limits<T>::epsilon();
}

template <class T>
struct timing_statistics
{
    T mean   = T( 0 );
    T stddev = T( 0 );
};

template <class T>
inline timing_statistics<T> compute_timing_statistics( const std::vector<T> &samples )
{
    timing_statistics<T> stats;
    if ( samples.empty() )
        return stats;

    T sum = T( 0 );
    for ( std::size_t i = 0; i < samples.size(); ++i )
        sum += samples[i];
    stats.mean = sum / static_cast<T>( samples.size() );

    T sq_sum = T( 0 );
    for ( std::size_t i = 0; i < samples.size(); ++i )
    {
        const T diff = samples[i] - stats.mean;
        sq_sum += diff * diff;
    }
    stats.stddev = std::sqrt( sq_sum / static_cast<T>( samples.size() ) );
    return stats;
}

inline bool path_exists( const std::string &path )
{
    struct stat st;
    return ::stat( path.c_str(), &st ) == 0;
}

inline void ensure_directory_exists( const std::string &path )
{
    if ( path.empty() )
        throw std::logic_error( "CSV output directory cannot be empty" );

    if ( path_exists( path ) )
        return;

    std::string current;
    if ( path[0] == '/' )
        current = "/";

    std::size_t pos = 0;
    while ( pos <= path.size() )
    {
        const std::size_t next = path.find( '/', pos );
        const std::string part = path.substr( pos, next == std::string::npos ? std::string::npos : next - pos );

        if ( !part.empty() && part != "." )
        {
            if ( !current.empty() && current[current.size() - 1] != '/' )
                current += "/";
            current += part;

            if ( ::mkdir( current.c_str(), 0755 ) != 0 && errno != EEXIST )
            {
                throw std::runtime_error(
                    "Failed to create directory '" + current + "': errno=" + std::to_string( errno )
                );
            }
        }

        if ( next == std::string::npos )
            break;
        pos = next + 1;
    }
}

inline std::string join_path( const std::string &directory, const std::string &filename )
{
    if ( directory.empty() )
        return filename;
    if ( directory[directory.size() - 1] == '/' )
        return directory + filename;
    return directory + "/" + filename;
}

inline std::string csv_quote( const std::string &value )
{
    std::string escaped;
    escaped.reserve( value.size() + 2 );
    escaped += '"';
    for ( std::size_t i = 0; i < value.size(); ++i )
    {
        if ( value[i] == '"' )
            escaped += "\"\"";
        else
            escaped += value[i];
    }
    escaped += '"';
    return escaped;
}

inline void append_csv_row(
    const std::string &directory,
    const std::string &filename,
    const std::string &header,
    const std::string &row
)
{
    ensure_directory_exists( directory );
    const std::string path = join_path( directory, filename );
    const bool write_header = !path_exists( path );

    std::ofstream out( path.c_str(), std::ios::out | std::ios::app );
    if ( !out )
        throw std::runtime_error( "Failed to open CSV output file '" + path + "'" );

    if ( write_header )
        out << header << '\n';
    out << row << '\n';
}

__DEVICE_TAG__ inline unsigned long long splitmix64( unsigned long long x )
{
    x += 0x9E3779B97F4A7C15ull;
    x = ( x ^ ( x >> 30 ) ) * 0xBF58476D1CE4E5B9ull;
    x = ( x ^ ( x >> 27 ) ) * 0x94D049BB133111EBull;
    return x ^ ( x >> 31 );
}

template <class T>
__DEVICE_TAG__ inline T unit_random_from_state( unsigned long long state )
{
    const unsigned long long bits = splitmix64( state );
    const double unit = static_cast<double>( bits >> 11 ) * ( 1.0 / 9007199254740992.0 );
    return static_cast<T>( T( 2 ) * static_cast<T>( unit ) - T( 1 ) );
}

template <class T, class Idx, class Array>
struct fill_random_real_3d_functor
{
    Array              array;
    unsigned long long seed;
    int                start_x;
    int                start_y;
    int                start_z;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const unsigned long long gx = static_cast<unsigned long long>( start_x + idx[0] );
        const unsigned long long gy = static_cast<unsigned long long>( start_y + idx[1] );
        const unsigned long long gz = static_cast<unsigned long long>( start_z + idx[2] );
        const unsigned long long state =
            seed ^
            ( gx * 0xD6E8FEB86659FD93ull ) ^
            ( gy * 0xA5A3564E27F886A5ull ) ^
            ( gz * 0x9E3779B97F4A7C15ull );
        array( idx ) = unit_random_from_state<T>( state );
    }
};

template <class T, class Idx, class Array>
struct fill_random_real_4d_functor
{
    Array              array;
    unsigned long long seed;
    int                start_x;
    int                start_y;
    int                start_z;
    int                start_w;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const unsigned long long gx = static_cast<unsigned long long>( start_x + idx[0] );
        const unsigned long long gy = static_cast<unsigned long long>( start_y + idx[1] );
        const unsigned long long gz = static_cast<unsigned long long>( start_z + idx[2] );
        const unsigned long long gw = static_cast<unsigned long long>( start_w + idx[3] );
        const unsigned long long state =
            seed ^
            ( gx * 0xD6E8FEB86659FD93ull ) ^
            ( gy * 0xA5A3564E27F886A5ull ) ^
            ( gz * 0x9E3779B97F4A7C15ull ) ^
            ( gw * 0x94D049BB133111EBull );
        array( idx ) = unit_random_from_state<T>( state );
    }
};

template <class Idx, class ArrayIn, class ArrayOut>
struct copy_same_indices_functor
{
    ArrayIn  src;
    ArrayOut dst;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        dst( idx ) = src( idx );
    }
};

template <class T, class Idx, class Array>
struct scale_real_functor
{
    Array array;
    T     scale;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        array( idx ) *= scale;
    }
};

template <class T, class Idx, class ArrayIn, class ArrayRef, class ErrorArray>
struct diff_square_real_3d_functor
{
    ArrayIn    actual;
    ArrayRef   reference;
    ErrorArray diff_sq;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const T diff = actual( idx ) - reference( idx );
        diff_sq( actual.calc_lin_index( idx[0], idx[1], idx[2] ) ) = diff * diff;
    }
};

template <class T, class Idx, class ArrayIn, class ArrayRef, class ErrorArray>
struct diff_square_real_4d_functor
{
    ArrayIn    actual;
    ArrayRef   reference;
    ErrorArray diff_sq;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const T diff = actual( idx ) - reference( idx );
        diff_sq( actual.calc_lin_index( idx[0], idx[1], idx[2], idx[3] ) ) = diff * diff;
    }
};

template <class Idx, class Rect, class Array>
inline Rect make_range_3d( const Array &array )
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

template <class Idx, class Rect, class Array>
inline Rect make_range_4d( const Array &array )
{
    const auto sz = array.size_nd();
    return Rect(
        Idx( 0, 0, 0, 0 ),
        Idx(
            static_cast<int>( sz[0] ),
            static_cast<int>( sz[1] ),
            static_cast<int>( sz[2] ),
            static_cast<int>( sz[3] )
        )
    );
}

} // namespace detail
} // namespace test
} // namespace fftm

#endif
