#ifndef __FFTM_TESTS_DETAIL_POISSON_FFT_TEST_COMMON_H__
#define __FFTM_TESTS_DETAIL_POISSON_FFT_TEST_COMMON_H__

#include <cstdio>
#include <stdexcept>
#include <string>

#include <scfd/arrays/tensor_array_nd.h>

#include <contrib/scfd/test/arrays/custom_index_fast_arranger.h>

namespace scfd
{
namespace arrays
{

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_10_t = scfd::arrays::custom_index_fast_arranger<1, 0>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_210_t = scfd::arrays::custom_index_fast_arranger<2, 1, 0>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_0213_t = scfd::arrays::custom_index_fast_arranger<0, 2, 1, 3>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_3012_t = scfd::arrays::custom_index_fast_arranger<3, 0, 1, 2>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_1023_t = scfd::arrays::custom_index_fast_arranger<1, 0, 2, 3>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_3120_t = scfd::arrays::custom_index_fast_arranger<3, 1, 2, 0>::type<Dims...>;

}
}

namespace io
{

template <class T, class Array>
void write_out_pos_file_scal_2D_quad( const std::string &filename, const Array &u_in, T lx, T ly )
{
    using view_t = typename Array::view_type;

    view_t u( u_in, true );
    auto   sz = u.size_nd();

    const std::size_t nx = static_cast<std::size_t>( sz[0] );
    const std::size_t ny = static_cast<std::size_t>( sz[1] );

    FILE *stream = std::fopen( filename.c_str(), "w" );
    if ( stream == NULL )
    {
        throw std::runtime_error( "error creating file: " + filename );
    }

    std::fprintf( stream, "View '%s %i' {\n", filename.c_str(), 0 );
    std::fprintf( stream, "TIME{0};\n" );

    for ( std::size_t i = 0; i < nx; ++i )
    {
        const std::size_t ip = ( i + 1 ) % nx;
        const T           x0 = lx * static_cast<T>( i ) / static_cast<T>( nx );
        const T           x1 = lx * static_cast<T>( i + 1 ) / static_cast<T>( nx );

        for ( std::size_t j = 0; j < ny; ++j )
        {
            const std::size_t jp = ( j + 1 ) % ny;
            const T           y0 = ly * static_cast<T>( j ) / static_cast<T>( ny );
            const T           y1 = ly * static_cast<T>( j + 1 ) / static_cast<T>( ny );

            std::fprintf(
                stream,
                "SQ(%le, %le, 0, %le, %le, 0, %le, %le, 0, %le, %le, 0){%le, %le, %le, %le};\n",
                static_cast<double>( x0 ),
                static_cast<double>( y0 ),
                static_cast<double>( x1 ),
                static_cast<double>( y0 ),
                static_cast<double>( x1 ),
                static_cast<double>( y1 ),
                static_cast<double>( x0 ),
                static_cast<double>( y1 ),
                static_cast<double>( u( i, j ) ),
                static_cast<double>( u( ip, j ) ),
                static_cast<double>( u( ip, jp ) ),
                static_cast<double>( u( i, jp ) )
            );
        }
    }

    std::fprintf( stream, "};\n" );
    std::fclose( stream );
    u.release( false );
}

template <class T, class Array>
void write_out_pos_file_scal_3D_hex( const std::string &filename, const Array &u_in, T lx, T ly, T lz )
{
    using view_t = typename Array::view_type;

    view_t u( u_in, true );
    auto   sz = u.size_nd();

    const std::size_t nx = static_cast<std::size_t>( sz[0] );
    const std::size_t ny = static_cast<std::size_t>( sz[1] );
    const std::size_t nz = static_cast<std::size_t>( sz[2] );

    FILE *stream = std::fopen( filename.c_str(), "w" );
    if ( stream == NULL )
    {
        throw std::runtime_error( "error creating file: " + filename );
    }

    std::fprintf( stream, "View '%s %i' {\n", filename.c_str(), 0 );
    std::fprintf( stream, "TIME{0};\n" );

    for ( std::size_t i = 0; i < nx; ++i )
    {
        const std::size_t ip = ( i + 1 ) % nx;
        const T           x0 = lx * static_cast<T>( i ) / static_cast<T>( nx );
        const T           x1 = lx * static_cast<T>( i + 1 ) / static_cast<T>( nx );

        for ( std::size_t j = 0; j < ny; ++j )
        {
            const std::size_t jp = ( j + 1 ) % ny;
            const T           y0 = ly * static_cast<T>( j ) / static_cast<T>( ny );
            const T           y1 = ly * static_cast<T>( j + 1 ) / static_cast<T>( ny );

            for ( std::size_t k = 0; k < nz; ++k )
            {
                const std::size_t kp = ( k + 1 ) % nz;
                const T           z0 = lz * static_cast<T>( k ) / static_cast<T>( nz );
                const T           z1 = lz * static_cast<T>( k + 1 ) / static_cast<T>( nz );

                std::fprintf(
                    stream,
                    "SH(%le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, "
                    "%le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le)"
                    "{%le, %le, %le, %le, %le, %le, %le, %le};\n",
                    static_cast<double>( x0 ),
                    static_cast<double>( y0 ),
                    static_cast<double>( z0 ),
                    static_cast<double>( x1 ),
                    static_cast<double>( y0 ),
                    static_cast<double>( z0 ),
                    static_cast<double>( x1 ),
                    static_cast<double>( y1 ),
                    static_cast<double>( z0 ),
                    static_cast<double>( x0 ),
                    static_cast<double>( y1 ),
                    static_cast<double>( z0 ),
                    static_cast<double>( x0 ),
                    static_cast<double>( y0 ),
                    static_cast<double>( z1 ),
                    static_cast<double>( x1 ),
                    static_cast<double>( y0 ),
                    static_cast<double>( z1 ),
                    static_cast<double>( x1 ),
                    static_cast<double>( y1 ),
                    static_cast<double>( z1 ),
                    static_cast<double>( x0 ),
                    static_cast<double>( y1 ),
                    static_cast<double>( z1 ),
                    static_cast<double>( u( i, j, k ) ),
                    static_cast<double>( u( ip, j, k ) ),
                    static_cast<double>( u( ip, jp, k ) ),
                    static_cast<double>( u( i, jp, k ) ),
                    static_cast<double>( u( i, j, kp ) ),
                    static_cast<double>( u( ip, j, kp ) ),
                    static_cast<double>( u( ip, jp, kp ) ),
                    static_cast<double>( u( i, jp, kp ) )
                );
            }
        }
    }

    std::fprintf( stream, "};\n" );
    std::fclose( stream );
    u.release( false );
}

}

#endif // __FFTM_TESTS_DETAIL_POISSON_FFT_TEST_COMMON_H__
