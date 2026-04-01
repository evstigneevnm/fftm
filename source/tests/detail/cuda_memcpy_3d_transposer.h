#ifndef __FFTM_TESTS_DETAIL_CUDA_MEMCPY_3D_TRANSPOSER_H__
#define __FFTM_TESTS_DETAIL_CUDA_MEMCPY_3D_TRANSPOSER_H__

#include <cuda_runtime.h>

#include <scfd/utils/cuda_safe_call.h>

#include "transpose_3d_test_common.h"

namespace fftm
{
namespace tests
{
namespace detail
{

template <class ValueType>
class cuda_memcpy_3d_transposer
{
public:
    cuda_memcpy_3d_transposer(
        std::size_t    nx,
        std::size_t    ny,
        std::size_t    nz,
        cudaMemcpyKind kind = cudaMemcpyDeviceToDevice
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
    void transpose_async( const ArrayIn &in, ArrayOut &out, cudaStream_t stream = 0 ) const
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
        CUDA_SAFE_CALL( cudaMemcpy3DAsync( &params, stream ) );
    }

    template <permutation_3d SrcPerm, permutation_3d DstPerm, class ArrayIn, class ArrayOut>
    void transpose( const ArrayIn &in, ArrayOut &out, cudaStream_t stream = 0 ) const
    {
        transpose_async<SrcPerm, DstPerm>( in, out, stream );
        CUDA_SAFE_CALL( cudaStreamSynchronize( stream ) );
    }

private:
    cudaMemcpy3DParms make_params( const ValueType *src_ptr, ValueType *dst_ptr ) const
    {
        cudaMemcpy3DParms params = { 0 };

        params.srcPos = make_cudaPos( 0, 0, 0 );
        params.srcPtr =
            make_cudaPitchedPtr( const_cast<ValueType *>( src_ptr ), ny_ * sizeof( ValueType ), ny_, nx_ );

        params.dstPos = make_cudaPos( 0, 0, 0 );
        params.dstPtr = make_cudaPitchedPtr( dst_ptr, ny_ * sizeof( ValueType ), ny_, nx_ );

        // All compatible permutation-aware arrays preserve the same physical Y-X-Z packing.
        params.extent = make_cudaExtent( ny_ * sizeof( ValueType ), nx_, nz_ );
        params.kind   = kind_;

        return params;
    }

private:
    std::size_t    nx_;
    std::size_t    ny_;
    std::size_t    nz_;
    cudaMemcpyKind kind_;
};

} // namespace detail
} // namespace tests
} // namespace fftm

#endif // __FFTM_TESTS_DETAIL_CUDA_MEMCPY_3D_TRANSPOSER_H__
