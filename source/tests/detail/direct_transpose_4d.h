#ifndef __FFTM_TESTS_DETAIL_DIRECT_TRANSPOSE_4D_H__
#define __FFTM_TESTS_DETAIL_DIRECT_TRANSPOSE_4D_H__

#include <cstddef>
#include <stdexcept>
#include <string>

#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>

namespace fftm
{
namespace tests
{
namespace detail
{

template <class Idx, class ArrayIn, class ArrayOut, int DstAxis0, int DstAxis1, int DstAxis2, int DstAxis3>
struct direct_transpose_4d_functor
{
    direct_transpose_4d_functor( const ArrayIn &_in, ArrayOut &_out )
        : in( _in )
        , out( _out )
    {
    }

    ArrayIn  in;
    ArrayOut out;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const int coords[4] = { idx[0], idx[1], idx[2], idx[3] };
        out( coords[DstAxis0], coords[DstAxis1], coords[DstAxis2], coords[DstAxis3] ) = in( idx );
    }
};

class direct_transpose_4d
{
public:
    direct_transpose_4d( std::size_t nx, std::size_t ny, std::size_t nz, std::size_t nw_half )
        : nx_( nx )
        , ny_( ny )
        , nz_( nz )
        , nw_half_( nw_half )
    {
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xyzw_to_xywz( const ForEach &for_each, const SrcArray &src, DstArray &dst ) const
    {
        transpose<0, 1, 3, 2>(
            for_each,
            src,
            dst,
            dims_xyzw(),
            dims_xywz(),
            "xyzw_to_xywz source",
            "xyzw_to_xywz destination"
        );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xywz_to_xyzw( const ForEach &for_each, const SrcArray &src, DstArray &dst ) const
    {
        transpose<0, 1, 3, 2>(
            for_each,
            src,
            dst,
            dims_xywz(),
            dims_xyzw(),
            "xywz_to_xyzw source",
            "xywz_to_xyzw destination"
        );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xywz_to_xzwy( const ForEach &for_each, const SrcArray &src, DstArray &dst ) const
    {
        transpose<0, 3, 2, 1>(
            for_each,
            src,
            dst,
            dims_xywz(),
            dims_xzwy(),
            "xywz_to_xzwy source",
            "xywz_to_xzwy destination"
        );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xzwy_to_xywz( const ForEach &for_each, const SrcArray &src, DstArray &dst ) const
    {
        transpose<0, 3, 2, 1>(
            for_each,
            src,
            dst,
            dims_xzwy(),
            dims_xywz(),
            "xzwy_to_xywz source",
            "xzwy_to_xywz destination"
        );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xzwy_to_yzwx( const ForEach &for_each, const SrcArray &src, DstArray &dst ) const
    {
        transpose<3, 1, 2, 0>(
            for_each,
            src,
            dst,
            dims_xzwy(),
            dims_yzwx(),
            "xzwy_to_yzwx source",
            "xzwy_to_yzwx destination"
        );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void yzwx_to_xzwy( const ForEach &for_each, const SrcArray &src, DstArray &dst ) const
    {
        transpose<3, 1, 2, 0>(
            for_each,
            src,
            dst,
            dims_yzwx(),
            dims_xzwy(),
            "yzwx_to_xzwy source",
            "yzwx_to_xzwy destination"
        );
    }

private:
    using idx_t   = scfd::static_vec::vec<int, 4>;
    using range_t = scfd::static_vec::rect<int, 4>;

    struct dims_4d_t
    {
        std::size_t d0;
        std::size_t d1;
        std::size_t d2;
        std::size_t d3;
    };

    dims_4d_t dims_xyzw() const
    {
        return { nx_, ny_, nz_, nw_half_ };
    }

    dims_4d_t dims_xywz() const
    {
        return { nx_, ny_, nw_half_, nz_ };
    }

    dims_4d_t dims_xzwy() const
    {
        return { nx_, nz_, nw_half_, ny_ };
    }

    dims_4d_t dims_yzwx() const
    {
        return { ny_, nz_, nw_half_, nx_ };
    }

    template <class Array>
    static void verify_shape( const Array &array, const dims_4d_t &expected, const std::string &name )
    {
        const auto actual = array.size_nd();

        const bool shape_ok = ( static_cast<std::size_t>( actual[0] ) == expected.d0 ) &&
                              ( static_cast<std::size_t>( actual[1] ) == expected.d1 ) &&
                              ( static_cast<std::size_t>( actual[2] ) == expected.d2 ) &&
                              ( static_cast<std::size_t>( actual[3] ) == expected.d3 );

        if ( !shape_ok )
        {
            throw std::runtime_error(
                name + " shape mismatch: expected (" + std::to_string( expected.d0 ) + ", " +
                std::to_string( expected.d1 ) + ", " + std::to_string( expected.d2 ) + ", " +
                std::to_string( expected.d3 ) + "), got (" + std::to_string( actual[0] ) + ", " +
                std::to_string( actual[1] ) + ", " + std::to_string( actual[2] ) + ", " +
                std::to_string( actual[3] ) + ")"
            );
        }
    }

    static range_t make_range( const dims_4d_t &dims )
    {
        return range_t(
            idx_t( 0, 0, 0, 0 ),
            idx_t(
                static_cast<int>( dims.d0 ),
                static_cast<int>( dims.d1 ),
                static_cast<int>( dims.d2 ),
                static_cast<int>( dims.d3 )
            )
        );
    }

    template <int DstAxis0, int DstAxis1, int DstAxis2, int DstAxis3, class ForEach, class SrcArray, class DstArray>
    static void transpose(
        const ForEach      &for_each,
        const SrcArray     &src,
        DstArray           &dst,
        const dims_4d_t    &src_dims,
        const dims_4d_t    &dst_dims,
        const std::string  &src_name,
        const std::string  &dst_name
    )
    {
        verify_shape( src, src_dims, src_name );
        verify_shape( dst, dst_dims, dst_name );

        for_each(
            direct_transpose_4d_functor<idx_t, SrcArray, DstArray, DstAxis0, DstAxis1, DstAxis2, DstAxis3>(
                src,
                dst
            ),
            make_range( src_dims )
        );
        for_each.wait();
    }

    std::size_t nx_;
    std::size_t ny_;
    std::size_t nz_;
    std::size_t nw_half_;
};

} // namespace detail
} // namespace tests
} // namespace fftm

#endif // __FFTM_TESTS_DETAIL_DIRECT_TRANSPOSE_4D_H__
