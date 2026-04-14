#ifndef __FFTM_TESTS_DETAIL_CUDA_MEMCPY_3D_TRANSPOSER_H__
#define __FFTM_TESTS_DETAIL_CUDA_MEMCPY_3D_TRANSPOSER_H__

#include "../../external_wrap/cufft_wrap.h"

#include "transpose_3d_test_common.h"

namespace fftm
{
namespace tests
{
namespace detail
{

template <class ValueType, class RuntimeAPI = ::fftm::wrap::cuda_runtime_api>
class cuda_memcpy_3d_transposer
{
public:
    using runtime_api_t = RuntimeAPI;
    using stream_t      = typename runtime_api_t::stream_t;
    using memcpy_kind_t = typename runtime_api_t::memcpy_kind_t;

    cuda_memcpy_3d_transposer(
        std::size_t    nx,
        std::size_t    ny,
        std::size_t    nz,
        memcpy_kind_t  kind = runtime_api_t::device_to_device_kind()
    )
        : nx_( nx )
        , ny_( ny )
        , nz_( nz )
        , kind_( kind )
    {
    }

    std::size_t nx() const
    {
        return nx_;
    }

    std::size_t ny() const
    {
        return ny_;
    }

    std::size_t nz() const
    {
        return nz_;
    }

    template <permutation_3d SrcPerm, permutation_3d DstPerm, class ArrayIn, class ArrayOut>
    void transpose_async( const ArrayIn &in, ArrayOut &out, stream_t stream = 0 ) const
    {
        verify_array_shape<SrcPerm>( in, nx_, ny_, nz_, "transpose source" );
        verify_array_shape<DstPerm>( out, nx_, ny_, nz_, "transpose destination" );

        if ( static_cast<std::size_t>( in.total_size() ) != nx_ * ny_ * nz_ )
        {
            throw std::runtime_error( "transpose source total size mismatch" );
        }
        if ( static_cast<std::size_t>( out.total_size() ) != nx_ * ny_ * nz_ )
        {
            throw std::runtime_error( "transpose destination total size mismatch" );
        }

        auto params = make_params( in.raw_ptr(), out.raw_ptr() );
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    template <permutation_3d SrcPerm, permutation_3d DstPerm, class ArrayIn, class ArrayOut>
    void transpose( const ArrayIn &in, ArrayOut &out, stream_t stream = 0 ) const
    {
        transpose_async<SrcPerm, DstPerm>( in, out, stream );
        runtime_api_t::stream_synchronize( stream );
    }

private:
    typename runtime_api_t::memcpy_3d_params_t make_params( const ValueType *src_ptr, ValueType *dst_ptr ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};

        params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( const_cast<ValueType *>( src_ptr ), ny_ * sizeof( ValueType ), ny_, nx_ );

        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPtr = runtime_api_t::make_pitched_ptr( dst_ptr, ny_ * sizeof( ValueType ), ny_, nx_ );

        // All compatible permutation-aware arrays preserve the same physical Y-X-Z packing.
        params.extent = runtime_api_t::make_extent( ny_ * sizeof( ValueType ), nx_, nz_ );
        params.kind   = kind_;

        return params;
    }

private:
    std::size_t    nx_;
    std::size_t    ny_;
    std::size_t    nz_;
    memcpy_kind_t  kind_;
};

} // namespace detail
} // namespace tests
} // namespace fftm

#endif // __FFTM_TESTS_DETAIL_CUDA_MEMCPY_3D_TRANSPOSER_H__
