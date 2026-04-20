#ifndef __FFTM_TESTS_DETAIL_MANUAL_TRANSPOSE_3D_H__
#define __FFTM_TESTS_DETAIL_MANUAL_TRANSPOSE_3D_H__

#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>

#include "transpose_3d_test_common.h"

namespace fftm
{
namespace tests
{
namespace detail
{

template <permutation_3d SrcPerm, permutation_3d DstPerm, class Idx, class ArrayIn, class ArrayOut>
struct manual_transpose_3d_functor
{
    manual_transpose_3d_functor( const ArrayIn &_in, ArrayOut &_out ) : in( _in ), out( _out )
    {
    }

    ArrayIn  in;
    ArrayOut out;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        int coords[3];

        coords[permutation_3d_traits<SrcPerm>::axis0] = idx[0];
        coords[permutation_3d_traits<SrcPerm>::axis1] = idx[1];
        coords[permutation_3d_traits<SrcPerm>::axis2] = idx[2];

        out( coords[permutation_3d_traits<DstPerm>::axis0], coords[permutation_3d_traits<DstPerm>::axis1],
             coords[permutation_3d_traits<DstPerm>::axis2] ) = in( idx );
    }
};

template <permutation_3d SrcPerm, permutation_3d DstPerm>
class manual_transpose_3d
{
public:
    manual_transpose_3d( std::size_t nx, std::size_t ny, std::size_t nz ) : nx_( nx ), ny_( ny ), nz_( nz )
    {
    }

    template <class ForEach, class ArrayIn, class ArrayOut>
    void transpose( const ForEach &for_each, const ArrayIn &in, ArrayOut &out ) const
    {
        using idx_t = scfd::static_vec::vec<int, 3>;

        verify_array_shape<SrcPerm>( in, nx_, ny_, nz_, "manual transpose source" );
        verify_array_shape<DstPerm>( out, nx_, ny_, nz_, "manual transpose destination" );

        const auto                           src_dims = dims_for_permutation<SrcPerm>( nx_, ny_, nz_ );
        const scfd::static_vec::rect<int, 3> range(
            idx_t( 0, 0, 0 ),
            idx_t( static_cast<int>( src_dims[0] ), static_cast<int>( src_dims[1] ), static_cast<int>( src_dims[2] ) )
        );

        for_each( manual_transpose_3d_functor<SrcPerm, DstPerm, idx_t, ArrayIn, ArrayOut>( in, out ), range );
        for_each.wait();
    }

private:
    std::size_t nx_;
    std::size_t ny_;
    std::size_t nz_;
};

} // namespace detail
} // namespace tests
} // namespace fftm

#endif // __FFTM_TESTS_DETAIL_MANUAL_TRANSPOSE_3D_H__
