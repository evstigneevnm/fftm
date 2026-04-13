#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_SAME_Z_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_SAME_Z_H__

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/cuda_stream_wrap.h>
#include <scfd/utils/log_mpi.h>

#include "../fft_partitioning.h"
#include "mpi_transpose_3d.h"

namespace fftm
{

template <class ValueType, class Backend, class MPIComm, class Log = scfd::utils::log_mpi>
class mpi_transpose_3d_same_z
{
public:
    using value_type       = ValueType;
    using backend_t        = Backend;
    using memory_t         = typename backend_t::memory_type;
    using partition_t      = ::fftm::partition;
    using mpi_comm_t       = scfd::communication::mpi_comm;
    using contiguous_buf_t = scfd::arrays::array_nd<value_type, 1, memory_t>;

    mpi_transpose_3d_same_z( const MPIComm &mpi, const Log &log = Log() )
        : mpi_( mpi )
        , log_( log )
    {
        static_assert(
            std::is_same<memory_t, scfd::memory::cuda_device>::value,
            "mpi_transpose_3d_same_z currently requires a CUDA backend memory type"
        );
    }

    void init( const partition_t &input_dim, const partition_t &output_dim, int myid_i, int myid_j )
    {
        input_dim_  = input_dim;
        output_dim_ = output_dim;
        myid_i_     = myid_i;
        myid_j_     = myid_j;

        if ( input_dim_.size_y.size() != 1 )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z expects the source layout to keep Y undistributed" );
        }
        if ( output_dim_.size_x.size() != 1 )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z expects the destination layout to keep X undistributed" );
        }
        if ( input_dim_.size_x.size() != output_dim_.size_y.size() )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z requires matching source-X and destination-Y partition counts" );
        }
        if ( input_dim_.size_z != output_dim_.size_z )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z requires Z ownership to stay unchanged" );
        }

        nx_local_  = input_dim_.size_x.at( myid_i_ );
        nx_global_ = output_dim_.size_x.at( 0 );
        ny_global_ = input_dim_.size_y.at( 0 );
        ny_local_  = output_dim_.size_y.at( myid_i_ );
        nz_local_  = input_dim_.size_z.at( myid_j_ );

        line_comm_      = std::make_unique<mpi_comm_t>( std::move( mpi_.split( myid_j_, myid_i_ ) ) );
        line_comm_info_ = line_comm_->info();
        if ( line_comm_info_.num_procs != static_cast<int>( input_dim_.size_x.size() ) )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z communicator size does not match X partition count" );
        }

        const std::size_t total_send_elems =
            std::max( nx_local_ * ny_global_ * nz_local_, nx_global_ * ny_local_ * nz_local_ );
        const std::size_t total_recv_elems =
            std::max( nx_local_ * ny_global_ * nz_local_, nx_global_ * ny_local_ * nz_local_ );

        send_buffer_.init( total_send_elems );
        recv_buffer_.init( total_recv_elems );

        send_requests_.assign( line_comm_info_.num_procs, MPI_REQUEST_NULL );
        recv_requests_.assign( line_comm_info_.num_procs, MPI_REQUEST_NULL );

        streams_.clear();
        streams_.reserve( line_comm_info_.num_procs );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            streams_.emplace_back( true );
        }

        init_forward_layout_();
        init_backward_layout_();
        is_inited_ = true;
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_x_to_y( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        ensure_is_inited_();
        verify_forward_shapes_( in, out );
        reset_requests_();
        pack_forward_( in );

        switch ( mode )
        {
            case mpi_transpose_3d_mode::p2p_waitall:
                forward_p2p_waitall_( out );
                break;
            case mpi_transpose_3d_mode::p2p_waitany:
                forward_p2p_waitany_( out );
                break;
            case mpi_transpose_3d_mode::alltoallv:
                forward_alltoallv_( out );
                break;
            case mpi_transpose_3d_mode::alltoallw:
                forward_alltoallw_( out );
                break;
        }
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_y_to_x( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        ensure_is_inited_();
        verify_backward_shapes_( in, out );
        reset_requests_();

        switch ( mode )
        {
            case mpi_transpose_3d_mode::p2p_waitall:
                backward_p2p_waitall_( in, out );
                break;
            case mpi_transpose_3d_mode::p2p_waitany:
                backward_p2p_waitany_( in, out );
                break;
            case mpi_transpose_3d_mode::alltoallv:
                backward_alltoallv_( in, out );
                break;
            case mpi_transpose_3d_mode::alltoallw:
                backward_alltoallw_( in, out );
                break;
        }
    }

private:
    void ensure_is_inited_() const
    {
        if ( !is_inited_ )
            throw std::logic_error( "mpi_transpose_3d_same_z::init must be called before transpose" );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_forward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != ny_global_ )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z forward input shape mismatch" );
        }
        if ( static_cast<std::size_t>( out_size[0] ) != nx_global_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != ny_local_ )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z forward output shape mismatch" );
        }
    }

    template <class ArrayIn, class ArrayOut>
    void verify_backward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_global_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != ny_local_ )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z backward input shape mismatch" );
        }
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != ny_global_ )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z backward output shape mismatch" );
        }
    }

    void reset_requests_()
    {
        std::fill( send_requests_.begin(), send_requests_.end(), MPI_REQUEST_NULL );
        std::fill( recv_requests_.begin(), recv_requests_.end(), MPI_REQUEST_NULL );
    }

    std::size_t bytes_from_elems_( std::size_t elems ) const
    {
        return elems * sizeof( value_type );
    }

    std::size_t forward_chunk_elems_( int target_i ) const
    {
        return nx_local_ * output_dim_.size_y[target_i] * nz_local_;
    }

    std::size_t forward_recv_offset_elems_( int source_i ) const
    {
        return input_dim_.start_x[source_i] * ny_local_;
    }

    std::size_t forward_recv_pack_offset_elems_( int source_i ) const
    {
        return forward_recv_offsets_[source_i];
    }

    std::size_t backward_send_offset_elems_( int target_i ) const
    {
        return input_dim_.start_x[target_i] * ny_local_;
    }

    std::size_t backward_chunk_elems_( int source_i ) const
    {
        return nx_local_ * output_dim_.size_y[source_i] * nz_local_;
    }

    std::size_t backward_recv_pack_offset_elems_( int source_i ) const
    {
        return backward_recv_offsets_[source_i];
    }

    void init_forward_layout_()
    {
        const int comm_size = line_comm_info_.num_procs;
        forward_recv_offsets_.resize( comm_size );
        forward_send_offsets_.resize( comm_size );
        forward_sendcounts_.resize( comm_size );
        forward_sdispls_.resize( comm_size );
        forward_recvcounts_.resize( comm_size );
        forward_rdispls_.resize( comm_size );
        forward_sendtypes_w_.assign( comm_size, MPI_BYTE );
        forward_recvtypes_w_.assign( comm_size, MPI_BYTE );

        std::size_t packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            forward_send_offsets_[p] = packed_offset;
            packed_offset += forward_chunk_elems_( p );
        }

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            forward_recv_offsets_[p] = packed_offset;
            packed_offset += input_dim_.size_x[p] * ny_local_ * nz_local_;
        }

        for ( int p = 0; p < comm_size; ++p )
        {
            forward_sendcounts_[p] = detail::mpi_int_cast( bytes_from_elems_( forward_chunk_elems_( p ) ), "same_z forward sendcounts" );
            forward_sdispls_[p]    = detail::mpi_int_cast( bytes_from_elems_( forward_send_offsets_[p] ), "same_z forward sdispls" );
            forward_recvcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( input_dim_.size_x[p] * ny_local_ * nz_local_ ),
                "same_z forward recvcounts"
            );
            forward_rdispls_[p]    = detail::mpi_int_cast(
                bytes_from_elems_( forward_recv_pack_offset_elems_( p ) ),
                "same_z forward rdispls"
            );
        }
    }

    void init_backward_layout_()
    {
        const int comm_size = line_comm_info_.num_procs;
        backward_send_offsets_.resize( comm_size );
        backward_recv_offsets_.resize( comm_size );
        backward_sendcounts_.resize( comm_size );
        backward_sdispls_.resize( comm_size );
        backward_recvcounts_.resize( comm_size );
        backward_rdispls_.resize( comm_size );
        backward_sendtypes_w_.assign( comm_size, MPI_BYTE );
        backward_recvtypes_w_.assign( comm_size, MPI_BYTE );

        std::size_t packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            backward_send_offsets_[p] = packed_offset;
            packed_offset += input_dim_.size_x[p] * ny_local_ * nz_local_;
        }

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            backward_recv_offsets_[p] = packed_offset;
            packed_offset += backward_chunk_elems_( p );
        }

        for ( int p = 0; p < comm_size; ++p )
        {
            backward_sendcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( input_dim_.size_x[p] * ny_local_ * nz_local_ ),
                "same_z backward sendcounts"
            );
            backward_sdispls_[p] = detail::mpi_int_cast(
                bytes_from_elems_( backward_send_offsets_[p] ),
                "same_z backward sdispls"
            );
            backward_recvcounts_[p] = detail::mpi_int_cast( bytes_from_elems_( backward_chunk_elems_( p ) ), "same_z backward recvcounts" );
            backward_rdispls_[p] = detail::mpi_int_cast(
                bytes_from_elems_( backward_recv_pack_offset_elems_( p ) ),
                "same_z backward rdispls"
            );
        }
    }

    void synchronize_streams_()
    {
        for ( std::size_t i = 0; i < streams_.size(); ++i )
        {
            CUDA_SAFE_CALL( cudaStreamSynchronize( streams_[i].stream() ) );
        }
    }

    void pack_forward_chunk_async_(
        const value_type *src_ptr,
        std::size_t       src_y_offset,
        std::size_t       packed_y_size,
        value_type       *dst_ptr,
        cudaStream_t      stream
    ) const
    {
        cudaMemcpy3DParms params = {};
        params.srcPos            = make_cudaPos( src_y_offset * sizeof( value_type ), 0, 0 );
        params.srcPtr            = make_cudaPitchedPtr(
            const_cast<value_type *>( src_ptr ),
            ny_global_ * sizeof( value_type ),
            ny_global_,
            nx_local_
        );
        params.dstPos = make_cudaPos( 0, 0, 0 );
        params.dstPtr = make_cudaPitchedPtr(
            dst_ptr,
            packed_y_size * sizeof( value_type ),
            packed_y_size,
            nx_local_
        );
        params.extent = make_cudaExtent( packed_y_size * sizeof( value_type ), nx_local_, nz_local_ );
        params.kind   = cudaMemcpyDeviceToDevice;
        CUDA_SAFE_CALL( cudaMemcpy3DAsync( &params, stream ) );
    }

    void unpack_forward_chunk_async_(
        const value_type *src_ptr,
        std::size_t       packed_x_size,
        std::size_t       dst_x_offset,
        value_type       *dst_ptr,
        cudaStream_t      stream
    ) const
    {
        cudaMemcpy3DParms params = {};
        params.srcPos            = make_cudaPos( 0, 0, 0 );
        params.srcPtr            = make_cudaPitchedPtr(
            const_cast<value_type *>( src_ptr ),
            ny_local_ * sizeof( value_type ),
            ny_local_,
            packed_x_size
        );
        params.dstPos = make_cudaPos( 0, dst_x_offset, 0 );
        params.dstPtr = make_cudaPitchedPtr(
            dst_ptr,
            ny_local_ * sizeof( value_type ),
            ny_local_,
            nx_global_
        );
        params.extent = make_cudaExtent( ny_local_ * sizeof( value_type ), packed_x_size, nz_local_ );
        params.kind   = cudaMemcpyDeviceToDevice;
        CUDA_SAFE_CALL( cudaMemcpy3DAsync( &params, stream ) );
    }

    void unpack_backward_chunk_async_(
        const value_type *src_ptr,
        std::size_t       packed_y_size,
        std::size_t       dst_y_offset,
        value_type       *dst_ptr,
        cudaStream_t      stream
    ) const
    {
        cudaMemcpy3DParms params = {};
        params.srcPos            = make_cudaPos( 0, 0, 0 );
        params.srcPtr            = make_cudaPitchedPtr(
            const_cast<value_type *>( src_ptr ),
            packed_y_size * sizeof( value_type ),
            packed_y_size,
            nx_local_
        );
        params.dstPos = make_cudaPos( dst_y_offset * sizeof( value_type ), 0, 0 );
        params.dstPtr = make_cudaPitchedPtr(
            dst_ptr,
            ny_global_ * sizeof( value_type ),
            ny_global_,
            nx_local_
        );
        params.extent = make_cudaExtent( packed_y_size * sizeof( value_type ), nx_local_, nz_local_ );
        params.kind   = cudaMemcpyDeviceToDevice;
        CUDA_SAFE_CALL( cudaMemcpy3DAsync( &params, stream ) );
    }

    void pack_backward_chunk_async_(
        const value_type *src_ptr,
        std::size_t       src_x_offset,
        std::size_t       packed_x_size,
        value_type       *dst_ptr,
        cudaStream_t      stream
    ) const
    {
        cudaMemcpy3DParms params = {};
        params.srcPos            = make_cudaPos( 0, src_x_offset, 0 );
        params.srcPtr            = make_cudaPitchedPtr(
            const_cast<value_type *>( src_ptr ),
            ny_local_ * sizeof( value_type ),
            ny_local_,
            nx_global_
        );
        params.dstPos = make_cudaPos( 0, 0, 0 );
        params.dstPtr = make_cudaPitchedPtr(
            dst_ptr,
            ny_local_ * sizeof( value_type ),
            ny_local_,
            packed_x_size
        );
        params.extent = make_cudaExtent( ny_local_ * sizeof( value_type ), packed_x_size, nz_local_ );
        params.kind   = cudaMemcpyDeviceToDevice;
        CUDA_SAFE_CALL( cudaMemcpy3DAsync( &params, stream ) );
    }

    template <class ArrayIn>
    void pack_forward_( const ArrayIn &in )
    {
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            pack_forward_chunk_async_(
                in.raw_ptr(),
                output_dim_.start_y[p],
                output_dim_.size_y[p],
                send_buffer_.raw_ptr() + forward_send_offsets_[p],
                streams_[p].stream()
            );
        }
        synchronize_streams_();
    }

    template <class ArrayOut>
    void unpack_forward_chunk_async_( int source_i, ArrayOut &out )
    {
        unpack_forward_chunk_async_(
            recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( source_i ),
            input_dim_.size_x[source_i],
            input_dim_.start_x[source_i],
            out.raw_ptr(),
            streams_[source_i].stream()
        );
    }

    template <class ArrayOut>
    void unpack_backward_chunk_async_( int source_i, ArrayOut &out )
    {
        unpack_backward_chunk_async_(
            recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( source_i ),
            output_dim_.size_y[source_i],
            output_dim_.start_y[source_i],
            out.raw_ptr(),
            streams_[source_i].stream()
        );
    }

    template <class ArrayIn>
    void pack_backward_( const ArrayIn &in )
    {
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            pack_backward_chunk_async_(
                in.raw_ptr(),
                input_dim_.start_x[p],
                input_dim_.size_x[p],
                send_buffer_.raw_ptr() + backward_send_offsets_[p],
                streams_[p].stream()
            );
        }
        synchronize_streams_();
    }

    template <class ArrayOut>
    void forward_p2p_waitall_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d_same_z currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        const int comm_size = line_comm_info_.num_procs;
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_i_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( input_dim_.size_x[p] * ny_local_ * nz_local_ ), "same_z forward recv count" ),
                MPI_BYTE,
                p,
                p,
                recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + forward_send_offsets_[p],
                detail::mpi_int_cast( bytes_from_elems_( forward_chunk_elems_( p ) ), "same_z forward send count" ),
                MPI_BYTE,
                p,
                myid_i_,
                send_requests_[p]
            );
        }

        CUDA_SAFE_CALL( cudaMemcpyAsync(
            recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( myid_i_ ),
            send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
            bytes_from_elems_( forward_chunk_elems_( myid_i_ ) ),
            cudaMemcpyDeviceToDevice,
            streams_[myid_i_].stream()
        ) );

        line_comm_info_.waitall( comm_size, recv_requests_.data() );
        for ( int p = 0; p < comm_size; ++p )
        {
            unpack_forward_chunk_async_( p, out );
        }
        synchronize_streams_();
        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayOut>
    void forward_p2p_waitany_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d_same_z currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        const int comm_size = line_comm_info_.num_procs;
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_i_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( input_dim_.size_x[p] * ny_local_ * nz_local_ ), "same_z forward recv count" ),
                MPI_BYTE,
                p,
                p,
                recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + forward_send_offsets_[p],
                detail::mpi_int_cast( bytes_from_elems_( forward_chunk_elems_( p ) ), "same_z forward send count" ),
                MPI_BYTE,
                p,
                myid_i_,
                send_requests_[p]
            );
        }

        CUDA_SAFE_CALL( cudaMemcpyAsync(
            recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( myid_i_ ),
            send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
            bytes_from_elems_( forward_chunk_elems_( myid_i_ ) ),
            cudaMemcpyDeviceToDevice,
            streams_[myid_i_].stream()
        ) );

        unpack_forward_chunk_async_( myid_i_, out );

        int completed = 0;
        while ( completed < comm_size - 1 )
        {
            const int p = line_comm_info_.waitany( comm_size, recv_requests_.data() );
            if ( p == MPI_UNDEFINED )
                break;
            unpack_forward_chunk_async_( p, out );
            completed++;
        }

        synchronize_streams_();
        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayOut>
    void forward_alltoallv_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d_same_z currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        line_comm_info_.alltoallv(
            static_cast<const void *>( send_buffer_.raw_ptr() ),
            forward_sendcounts_.data(),
            forward_sdispls_.data(),
            MPI_BYTE,
            static_cast<void *>( recv_buffer_.raw_ptr() ),
            forward_recvcounts_.data(),
            forward_rdispls_.data(),
            MPI_BYTE
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            unpack_forward_chunk_async_( p, out );
        }
        synchronize_streams_();
#endif
    }

    template <class ArrayOut>
    void forward_alltoallw_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d_same_z currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        line_comm_info_.alltoallw(
            static_cast<const void *>( send_buffer_.raw_ptr() ),
            forward_sendcounts_.data(),
            forward_sdispls_.data(),
            forward_sendtypes_w_.data(),
            static_cast<void *>( recv_buffer_.raw_ptr() ),
            forward_recvcounts_.data(),
            forward_rdispls_.data(),
            forward_recvtypes_w_.data()
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            unpack_forward_chunk_async_( p, out );
        }
        synchronize_streams_();
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_p2p_waitall_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d_same_z currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        const int comm_size = line_comm_info_.num_procs;
        pack_backward_( in );

        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_i_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( backward_chunk_elems_( p ) ), "same_z backward recv count" ),
                MPI_BYTE,
                p,
                myid_i_,
                recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + backward_send_offsets_[p],
                detail::mpi_int_cast( bytes_from_elems_( input_dim_.size_x[p] * ny_local_ * nz_local_ ), "same_z backward send count" ),
                MPI_BYTE,
                p,
                p,
                send_requests_[p]
            );
        }

        CUDA_SAFE_CALL( cudaMemcpyAsync(
            recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( myid_i_ ),
            send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
            bytes_from_elems_( backward_chunk_elems_( myid_i_ ) ),
            cudaMemcpyDeviceToDevice,
            streams_[myid_i_].stream()
        ) );

        line_comm_info_.waitall( comm_size, recv_requests_.data() );
        for ( int p = 0; p < comm_size; ++p )
        {
            unpack_backward_chunk_async_( p, out );
        }
        synchronize_streams_();
        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_p2p_waitany_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d_same_z currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        const int comm_size = line_comm_info_.num_procs;
        pack_backward_( in );

        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_i_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( backward_chunk_elems_( p ) ), "same_z backward recv count" ),
                MPI_BYTE,
                p,
                myid_i_,
                recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + backward_send_offsets_[p],
                detail::mpi_int_cast( bytes_from_elems_( input_dim_.size_x[p] * ny_local_ * nz_local_ ), "same_z backward send count" ),
                MPI_BYTE,
                p,
                p,
                send_requests_[p]
            );
        }

        CUDA_SAFE_CALL( cudaMemcpyAsync(
            recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( myid_i_ ),
            send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
            bytes_from_elems_( backward_chunk_elems_( myid_i_ ) ),
            cudaMemcpyDeviceToDevice,
            streams_[myid_i_].stream()
        ) );

        unpack_backward_chunk_async_( myid_i_, out );

        int completed = 0;
        while ( completed < comm_size - 1 )
        {
            const int p = line_comm_info_.waitany( comm_size, recv_requests_.data() );
            if ( p == MPI_UNDEFINED )
                break;
            unpack_backward_chunk_async_( p, out );
            completed++;
        }

        synchronize_streams_();
        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_alltoallv_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d_same_z currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        pack_backward_( in );

        line_comm_info_.alltoallv(
            static_cast<const void *>( send_buffer_.raw_ptr() ),
            backward_sendcounts_.data(),
            backward_sdispls_.data(),
            MPI_BYTE,
            static_cast<void *>( recv_buffer_.raw_ptr() ),
            backward_recvcounts_.data(),
            backward_rdispls_.data(),
            MPI_BYTE
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            unpack_backward_chunk_async_( p, out );
        }
        synchronize_streams_();
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_alltoallw_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d_same_z currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        pack_backward_( in );

        line_comm_info_.alltoallw(
            static_cast<const void *>( send_buffer_.raw_ptr() ),
            backward_sendcounts_.data(),
            backward_sdispls_.data(),
            backward_sendtypes_w_.data(),
            static_cast<void *>( recv_buffer_.raw_ptr() ),
            backward_recvcounts_.data(),
            backward_rdispls_.data(),
            backward_recvtypes_w_.data()
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            unpack_backward_chunk_async_( p, out );
        }
        synchronize_streams_();
#endif
    }

private:
    MPIComm mpi_;
    Log     log_;

    bool is_inited_ = false;
    int  myid_i_    = 0;
    int  myid_j_    = 0;

    std::size_t nx_local_  = 0;
    std::size_t nx_global_ = 0;
    std::size_t ny_local_  = 0;
    std::size_t ny_global_ = 0;
    std::size_t nz_local_  = 0;

    partition_t input_dim_;
    partition_t output_dim_;

    std::unique_ptr<mpi_comm_t>             line_comm_;
    scfd::communication::mpi_comm_info      line_comm_info_;
    contiguous_buf_t                        send_buffer_;
    contiguous_buf_t                        recv_buffer_;
    std::vector<MPI_Request>                send_requests_;
    std::vector<MPI_Request>                recv_requests_;
    std::vector<scfd::utils::cuda_stream_wrap> streams_;

    std::vector<std::size_t> forward_send_offsets_;
    std::vector<std::size_t> forward_recv_offsets_;
    std::vector<int>         forward_sendcounts_;
    std::vector<int>         forward_sdispls_;
    std::vector<int>         forward_recvcounts_;
    std::vector<int>         forward_rdispls_;
    std::vector<MPI_Datatype> forward_sendtypes_w_;
    std::vector<MPI_Datatype> forward_recvtypes_w_;

    std::vector<std::size_t> backward_send_offsets_;
    std::vector<std::size_t> backward_recv_offsets_;
    std::vector<int>         backward_sendcounts_;
    std::vector<int>         backward_sdispls_;
    std::vector<int>         backward_recvcounts_;
    std::vector<int>         backward_rdispls_;
    std::vector<MPI_Datatype> backward_sendtypes_w_;
    std::vector<MPI_Datatype> backward_recvtypes_w_;
};

} // namespace fftm

#endif
