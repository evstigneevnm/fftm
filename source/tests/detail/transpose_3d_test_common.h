#ifndef __FFTM_TESTS_DETAIL_TRANSPOSE_3D_TEST_COMMON_H__
#define __FFTM_TESTS_DETAIL_TRANSPOSE_3D_TEST_COMMON_H__

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <scfd/arrays/tensor_array_nd.h>

#include <contrib/scfd/test/arrays/custom_index_fast_arranger.h>

namespace fftm
{
namespace tests
{
namespace detail
{

template <scfd::arrays::ordinal_type... Dims>
using arranger_012_t = scfd::arrays::custom_index_fast_arranger<0, 1, 2>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using arranger_021_t = scfd::arrays::custom_index_fast_arranger<0, 2, 1>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using arranger_102_t = scfd::arrays::custom_index_fast_arranger<1, 0, 2>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using arranger_120_t = scfd::arrays::custom_index_fast_arranger<1, 2, 0>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using arranger_201_t = scfd::arrays::custom_index_fast_arranger<2, 0, 1>::type<Dims...>;

template <scfd::arrays::ordinal_type... Dims>
using arranger_210_t = scfd::arrays::custom_index_fast_arranger<2, 1, 0>::type<Dims...>;

enum class permutation_3d
{
    xyz,
    xzy,
    zxy,
    zyx,
    yzx,
    yxz
};

template <permutation_3d Perm>
struct permutation_3d_traits;

template <>
struct permutation_3d_traits<permutation_3d::xyz>
{
    enum
    {
        axis0 = 0,
        axis1 = 1,
        axis2 = 2
    };

    template <scfd::arrays::ordinal_type... Dims>
    using arranger = arranger_102_t<Dims...>;

    static const char *label()
    {
        return "XYZ";
    }
};

template <>
struct permutation_3d_traits<permutation_3d::xzy>
{
    enum
    {
        axis0 = 0,
        axis1 = 2,
        axis2 = 1
    };

    template <scfd::arrays::ordinal_type... Dims>
    using arranger = arranger_201_t<Dims...>;

    static const char *label()
    {
        return "XZY";
    }
};

template <>
struct permutation_3d_traits<permutation_3d::zxy>
{
    enum
    {
        axis0 = 2,
        axis1 = 0,
        axis2 = 1
    };

    template <scfd::arrays::ordinal_type... Dims>
    using arranger = arranger_210_t<Dims...>;

    static const char *label()
    {
        return "ZXY";
    }
};

template <>
struct permutation_3d_traits<permutation_3d::zyx>
{
    enum
    {
        axis0 = 2,
        axis1 = 1,
        axis2 = 0
    };

    template <scfd::arrays::ordinal_type... Dims>
    using arranger = arranger_120_t<Dims...>;

    static const char *label()
    {
        return "ZYX";
    }
};

template <>
struct permutation_3d_traits<permutation_3d::yzx>
{
    enum
    {
        axis0 = 1,
        axis1 = 2,
        axis2 = 0
    };

    template <scfd::arrays::ordinal_type... Dims>
    using arranger = arranger_021_t<Dims...>;

    static const char *label()
    {
        return "YZX";
    }
};

template <>
struct permutation_3d_traits<permutation_3d::yxz>
{
    enum
    {
        axis0 = 1,
        axis1 = 0,
        axis2 = 2
    };

    template <scfd::arrays::ordinal_type... Dims>
    using arranger = arranger_012_t<Dims...>;

    static const char *label()
    {
        return "YXZ";
    }
};

template <class T, class Memory, permutation_3d Perm>
using permuted_tensor_3d_t =
    scfd::arrays::tensor_array_nd<T, 3, Memory, permutation_3d_traits<Perm>::template arranger>;

template <permutation_3d Perm>
inline std::array<std::size_t, 3> dims_for_permutation( std::size_t nx, std::size_t ny, std::size_t nz )
{
    const std::size_t logical_dims[3] = { nx, ny, nz };

    return { {
        logical_dims[permutation_3d_traits<Perm>::axis0],
        logical_dims[permutation_3d_traits<Perm>::axis1],
        logical_dims[permutation_3d_traits<Perm>::axis2],
    } };
}

template <permutation_3d Perm, class Array>
inline void init_permuted_array( Array &array, std::size_t nx, std::size_t ny, std::size_t nz )
{
    const auto dims = dims_for_permutation<Perm>( nx, ny, nz );
    array.init( dims[0], dims[1], dims[2] );
}

template <permutation_3d Perm, class Array>
inline void
verify_array_shape( const Array &array, std::size_t nx, std::size_t ny, std::size_t nz, const std::string &array_name )
{
    const auto expected = dims_for_permutation<Perm>( nx, ny, nz );
    const auto actual   = array.size_nd();

    const bool shape_ok = ( static_cast<std::size_t>( actual[0] ) == expected[0] ) &&
                          ( static_cast<std::size_t>( actual[1] ) == expected[1] ) &&
                          ( static_cast<std::size_t>( actual[2] ) == expected[2] );

    if ( !shape_ok )
    {
        throw std::runtime_error(
            array_name + " shape mismatch for permutation " + permutation_3d_traits<Perm>::label() + ": expected (" +
            std::to_string( expected[0] ) + ", " + std::to_string( expected[1] ) + ", " +
            std::to_string( expected[2] ) + "), got (" + std::to_string( actual[0] ) + ", " +
            std::to_string( actual[1] ) + ", " + std::to_string( actual[2] ) + ")"
        );
    }
}

template <class T, class Array>
void write_out_pos_file_scal_3D_point(
    const std::string &filename, const Array &u_in, std::size_t nx, std::size_t ny, std::size_t nz
)
{
    using view_t = typename Array::view_type;

    view_t u( u_in );

    FILE *stream = std::fopen( filename.c_str(), "w" );
    if ( stream == NULL )
    {
        throw std::runtime_error( "error creating file: " + filename );
    }
    std::fclose( stream );

    stream = std::fopen( filename.c_str(), "a" );
    if ( stream == NULL )
    {
        throw std::runtime_error( "error opening file: " + filename );
    }

    std::fprintf( stream, "View '%s %i' {\n", filename.c_str(), 0 );
    std::fprintf( stream, "TIME{0};\n" );

    for ( std::size_t i = 0; i < nx; ++i )
    {
        const double x0 = static_cast<double>( i );
        const double x1 = static_cast<double>( i + 1 );

        for ( std::size_t j = 0; j < ny; ++j )
        {
            const double y0 = static_cast<double>( j );
            const double y1 = static_cast<double>( j + 1 );

            for ( std::size_t k = 0; k < nz; ++k )
            {
                const double z0 = static_cast<double>( k );
                const double z1 = static_cast<double>( k + 1 );

                const T value = u( i, j, k ).real();

                std::fprintf(
                    stream,
                    "SH(%le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, "
                    "%le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le)"
                    "{%le, %le, %le, %le, %le, %le, %le, %le};\n",
                    x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
                    static_cast<double>( value ), static_cast<double>( value ), static_cast<double>( value ),
                    static_cast<double>( value ), static_cast<double>( value ), static_cast<double>( value ),
                    static_cast<double>( value ), static_cast<double>( value )
                );
            }
        }
    }

    std::fprintf( stream, "};\n" );
    std::fclose( stream );
    u.release( false );
}

} // namespace detail
} // namespace tests
} // namespace fftm

#endif // __FFTM_TESTS_DETAIL_TRANSPOSE_3D_TEST_COMMON_H__
