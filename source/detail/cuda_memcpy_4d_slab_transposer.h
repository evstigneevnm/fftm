#ifndef __FFTM_DETAIL_CUDA_MEMCPY_4D_SLAB_TRANSPOSER_H__
#define __FFTM_DETAIL_CUDA_MEMCPY_4D_SLAB_TRANSPOSER_H__

#include <cstddef>

#include "../external_wrap/cufft_wrap.h"

namespace fftm
{
namespace detail
{

template <class ValueType, class RuntimeAPI = ::fftm::wrap::cuda_runtime_api>
class cuda_memcpy_4d_slab_transposer
{
public:
    using runtime_api_t = RuntimeAPI;
    using stream_t      = typename runtime_api_t::stream_t;

    cuda_memcpy_4d_slab_transposer( std::size_t nx, std::size_t ny, std::size_t nz, std::size_t nw_half )
        : nx_( nx )
        , ny_( ny )
        , nz_( nz )
        , nw_half_( nw_half )
    {
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xyzw_to_xywz( const ForEach &, const SrcArray &src, DstArray &dst, stream_t stream = 0 ) const
    {
        copy_step_z_slabs( src, dst, stream );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xywz_to_xyzw( const ForEach &, const SrcArray &src, DstArray &dst, stream_t stream = 0 ) const
    {
        copy_step_z_slabs_inverse( src, dst, stream );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xywz_to_xzwy( const ForEach &, const SrcArray &src, DstArray &dst, stream_t stream = 0 ) const
    {
        copy_step_y_slabs( src, dst, stream );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xzwy_to_xywz( const ForEach &, const SrcArray &src, DstArray &dst, stream_t stream = 0 ) const
    {
        copy_step_y_slabs_inverse( src, dst, stream );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xzwy_to_yzwx( const ForEach &, const SrcArray &src, DstArray &dst, stream_t stream = 0 ) const
    {
        copy_step_full_buffer( src, dst, stream );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void yzwx_to_xzwy( const ForEach &, const SrcArray &src, DstArray &dst, stream_t stream = 0 ) const
    {
        copy_step_full_buffer( src, dst, stream );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void xyzw_to_zwxy( const ForEach &, const SrcArray &src, DstArray &dst, stream_t stream = 0 ) const
    {
        copy_async_full_buffer( src, dst, stream );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void zwxy_to_xyzw( const ForEach &, const SrcArray &src, DstArray &dst, stream_t stream = 0 ) const
    {
        copy_async_full_buffer( src, dst, stream );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void zwxy_to_yzwx( const ForEach &, const SrcArray &src, DstArray &dst, stream_t stream = 0 ) const
    {
        copy_async_full_buffer( src, dst, stream );
    }

    template <class ForEach, class SrcArray, class DstArray>
    void yzwx_to_zwxy( const ForEach &, const SrcArray &src, DstArray &dst, stream_t stream = 0 ) const
    {
        copy_async_full_buffer( src, dst, stream );
    }

private:
    template <class SrcArray, class DstArray>
    void copy_step_z_slabs( const SrcArray &src, DstArray &dst, stream_t stream ) const
    {
        for ( std::size_t z = 0; z < nz_; ++z )
        {
            typename runtime_api_t::memcpy_3d_params_t params = {};
            params.srcPos = runtime_api_t::make_pos( 0, z * ny_, 0 );
            params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
            params.srcPtr = runtime_api_t::make_pitched_ptr(
                const_cast<ValueType *>( src.raw_ptr() ),
                nx_ * sizeof( ValueType ),
                nx_,
                ny_ * nz_
            );
            params.dstPtr = runtime_api_t::make_pitched_ptr(
                dst.raw_ptr() + dst.calc_lin_index( 0, 0, 0, z ),
                nx_ * sizeof( ValueType ),
                nx_,
                ny_
            );
            params.extent = runtime_api_t::make_extent( nx_ * sizeof( ValueType ), ny_, nw_half_ );
            params.kind   = runtime_api_t::device_to_device_kind();

            runtime_api_t::memcpy_3d_async( &params, stream );
        }
    }

    template <class SrcArray, class DstArray>
    void copy_step_z_slabs_inverse( const SrcArray &src, DstArray &dst, stream_t stream ) const
    {
        for ( std::size_t z = 0; z < nz_; ++z )
        {
            typename runtime_api_t::memcpy_3d_params_t params = {};
            params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
            params.dstPos = runtime_api_t::make_pos( 0, z * ny_, 0 );
            params.srcPtr = runtime_api_t::make_pitched_ptr(
                const_cast<ValueType *>( src.raw_ptr() + src.calc_lin_index( 0, 0, 0, z ) ),
                nx_ * sizeof( ValueType ),
                nx_,
                ny_
            );
            params.dstPtr = runtime_api_t::make_pitched_ptr(
                dst.raw_ptr(),
                nx_ * sizeof( ValueType ),
                nx_,
                ny_ * nz_
            );
            params.extent = runtime_api_t::make_extent( nx_ * sizeof( ValueType ), ny_, nw_half_ );
            params.kind   = runtime_api_t::device_to_device_kind();

            runtime_api_t::memcpy_3d_async( &params, stream );
        }
    }

    template <class SrcArray, class DstArray>
    void copy_step_y_slabs( const SrcArray &src, DstArray &dst, stream_t stream ) const
    {
        for ( std::size_t y = 0; y < ny_; ++y )
        {
            typename runtime_api_t::memcpy_3d_params_t params = {};
            params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
            params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
            params.srcPtr = runtime_api_t::make_pitched_ptr(
                const_cast<ValueType *>( src.raw_ptr() + src.calc_lin_index( 0, y, 0, 0 ) ),
                nx_ * ny_ * sizeof( ValueType ),
                nx_,
                nw_half_
            );
            params.dstPtr = runtime_api_t::make_pitched_ptr(
                dst.raw_ptr() + dst.calc_lin_index( 0, 0, 0, y ),
                nx_ * sizeof( ValueType ),
                nx_,
                nw_half_
            );
            params.extent = runtime_api_t::make_extent( nx_ * sizeof( ValueType ), nw_half_, nz_ );
            params.kind   = runtime_api_t::device_to_device_kind();

            runtime_api_t::memcpy_3d_async( &params, stream );
        }
    }

    template <class SrcArray, class DstArray>
    void copy_step_y_slabs_inverse( const SrcArray &src, DstArray &dst, stream_t stream ) const
    {
        for ( std::size_t y = 0; y < ny_; ++y )
        {
            typename runtime_api_t::memcpy_3d_params_t params = {};
            params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
            params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
            params.srcPtr = runtime_api_t::make_pitched_ptr(
                const_cast<ValueType *>( src.raw_ptr() + src.calc_lin_index( 0, 0, 0, y ) ),
                nx_ * sizeof( ValueType ),
                nx_,
                nw_half_
            );
            params.dstPtr = runtime_api_t::make_pitched_ptr(
                dst.raw_ptr() + dst.calc_lin_index( 0, y, 0, 0 ),
                nx_ * ny_ * sizeof( ValueType ),
                nx_,
                nw_half_
            );
            params.extent = runtime_api_t::make_extent( nx_ * sizeof( ValueType ), nw_half_, nz_ );
            params.kind   = runtime_api_t::device_to_device_kind();

            runtime_api_t::memcpy_3d_async( &params, stream );
        }
    }

    template <class SrcArray, class DstArray>
    void copy_step_full_buffer( const SrcArray &src, DstArray &dst, stream_t stream ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr = runtime_api_t::make_pitched_ptr(
            const_cast<ValueType *>( src.raw_ptr() ),
            nx_ * sizeof( ValueType ),
            nx_,
            nw_half_
        );
        params.dstPtr = runtime_api_t::make_pitched_ptr(
            dst.raw_ptr(),
            nx_ * sizeof( ValueType ),
            nx_,
            nw_half_
        );
        params.extent = runtime_api_t::make_extent( nx_ * sizeof( ValueType ), nw_half_, nz_ * ny_ );
        params.kind   = runtime_api_t::device_to_device_kind();

        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    template <class SrcArray, class DstArray>
    void copy_async_full_buffer( const SrcArray &src, DstArray &dst, stream_t stream ) const
    {
        runtime_api_t::memcpy_async(
            dst.raw_ptr(),
            src.raw_ptr(),
            sizeof( ValueType ) * static_cast<std::size_t>( src.total_size() ),
            runtime_api_t::device_to_device_kind(),
            stream
        );
    }

    std::size_t nx_;
    std::size_t ny_;
    std::size_t nz_;
    std::size_t nw_half_;
};

} // namespace detail
} // namespace fftm

#endif // __FFTM_DETAIL_CUDA_MEMCPY_4D_SLAB_TRANSPOSER_H__
