#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_H__

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/cuda_stream_wrap.h>
#include <scfd/utils/log_mpi.h>

#include "../fft_partitioning.h"

namespace fftm
{

enum class mpi_transpose_3d_mode
{
    p2p_waitall,
    p2p_waitany,
    alltoallv,
    alltoallw
};

inline const char *mpi_transpose_3d_mode_name( mpi_transpose_3d_mode mode )
{
    switch ( mode )
    {
        case mpi_transpose_3d_mode::p2p_waitall:
            return "p2p-waitall";
        case mpi_transpose_3d_mode::p2p_waitany:
            return "p2p-waitany";
        case mpi_transpose_3d_mode::alltoallv:
            return "alltoallv";
        case mpi_transpose_3d_mode::alltoallw:
            return "alltoallw";
    }
    return "unknown";
}

namespace detail
{

inline int mpi_int_cast( std::size_t value, const std::string &what )
{
    if ( value > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
    {
        throw std::logic_error( what + " exceeds MPI int range" );
    }
    return static_cast<int>( value );
}

} // namespace detail

template <class ValueType, class Backend, class MPIComm, class Log = scfd::utils::log_mpi>
class mpi_transpose_3d
{
public:
    using value_type       = ValueType;
    using backend_t        = Backend;
    using memory_t         = typename backend_t::memory_type;
    using partition_t      = ::fftm::partition;
    using mpi_comm_t       = scfd::communication::mpi_comm;
    using contiguous_buf_t = scfd::arrays::array_nd<value_type, 1, memory_t>;
    using mpi_request_t    = scfd::communication::detail::mpi_request;
    using mpi_dtype_t      = scfd::communication::detail::mpi_data_type<>;

    mpi_transpose_3d( const MPIComm &mpi, const Log &log = Log() )
        : mpi_( mpi )
        , log_( log )
    {
        static_assert(
            std::is_same<memory_t, scfd::memory::cuda_device>::value,
            "mpi_transpose_3d currently requires a CUDA backend memory type"
        );
    }

    ~mpi_transpose_3d()
    {
        free_forward_types_();
    }

    void init( const partition_t &input_dim, const partition_t &output_dim, int myid_i, int myid_j )
    {
        free_forward_types_();

        input_dim_  = input_dim;
        output_dim_ = output_dim;
        myid_i_     = myid_i;
        myid_j_     = myid_j;

        if ( input_dim_.size_z.size() != 1 )
        {
            throw std::logic_error( "mpi_transpose_3d expects the source layout to keep Z undistributed" );
        }
        if ( output_dim_.size_y.size() != 1 )
        {
            throw std::logic_error( "mpi_transpose_3d expects the destination layout to keep Y undistributed" );
        }
        if ( output_dim_.size_x.at( myid_i_ ) != input_dim_.size_x.at( myid_i_ ) )
        {
            throw std::logic_error( "mpi_transpose_3d requires X ownership to stay unchanged" );
        }
        if ( input_dim_.size_y.size() != output_dim_.size_z.size() )
        {
            throw std::logic_error( "mpi_transpose_3d requires matching source-Y and destination-Z partition counts" );
        }

        nx_local_  = input_dim_.size_x.at( myid_i_ );
        ny_local_  = input_dim_.size_y.at( myid_j_ );
        ny_global_ = output_dim_.size_y.at( 0 );
        nz_global_ = input_dim_.size_z.at( 0 );
        nz_local_  = output_dim_.size_z.at( myid_j_ );

        row_comm_      = std::make_unique<mpi_comm_t>( std::move( mpi_.split( myid_i_, myid_j_ ) ) );
        row_comm_info_ = row_comm_->info();
        if ( row_comm_info_.num_procs != static_cast<int>( input_dim_.size_y.size() ) )
        {
            throw std::logic_error( "mpi_transpose_3d row communicator size does not match Y partition count" );
        }

        const std::size_t max_send_elems =
            std::max( nx_local_ * ny_local_ * nz_global_, nx_local_ * ny_global_ * nz_local_ );
        const std::size_t max_recv_elems =
            std::max( nx_local_ * ny_global_ * nz_local_, nx_local_ * ny_local_ * nz_global_ );

        send_buffer_.init( max_send_elems );
        recv_buffer_.init( max_recv_elems );

        send_requests_.assign( row_comm_info_.num_procs, mpi_request_t() );
        recv_requests_.assign( row_comm_info_.num_procs, mpi_request_t() );

        streams_.clear();
        streams_.reserve( row_comm_info_.num_procs );
        for ( int p = 0; p < row_comm_info_.num_procs; ++p )
        {
            streams_.emplace_back( true );
        }

        init_forward_layout_();
        init_backward_layout_();
        is_inited_ = true;
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xyz_to_xzy( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        ensure_is_inited_();
        verify_forward_shapes_( in, out );
        reset_requests_();

        switch ( mode )
        {
            case mpi_transpose_3d_mode::p2p_waitall:
                forward_p2p_waitall_( in, out );
                break;
            case mpi_transpose_3d_mode::p2p_waitany:
                forward_p2p_waitany_( in, out );
                break;
            case mpi_transpose_3d_mode::alltoallv:
                forward_alltoallv_( in, out );
                break;
            case mpi_transpose_3d_mode::alltoallw:
                forward_alltoallw_( in, out );
                break;
        }
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xzy_to_xyz( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
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
        {
            throw std::logic_error( "mpi_transpose_3d::init must be called before transpose" );
        }
    }

    template <class ArrayIn, class ArrayOut>
    void verify_forward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size = in.size_nd();
        const auto out_size = out.size_nd();

        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nz_global_ )
        {
            throw std::logic_error( "mpi_transpose_3d forward input shape mismatch" );
        }
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != ny_global_ )
        {
            throw std::logic_error( "mpi_transpose_3d forward output shape mismatch" );
        }
    }

    template <class ArrayIn, class ArrayOut>
    void verify_backward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size = in.size_nd();
        const auto out_size = out.size_nd();

        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != ny_global_ )
        {
            throw std::logic_error( "mpi_transpose_3d backward input shape mismatch" );
        }
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nz_global_ )
        {
            throw std::logic_error( "mpi_transpose_3d backward output shape mismatch" );
        }
    }

    void reset_requests_()
    {
        std::fill( send_requests_.begin(), send_requests_.end(), mpi_request_t() );
        std::fill( recv_requests_.begin(), recv_requests_.end(), mpi_request_t() );
    }

    std::size_t bytes_from_elems_( std::size_t elements ) const
    {
        return elements * sizeof( value_type );
    }

    std::size_t forward_send_chunk_elems_( int target_j ) const
    {
        return nx_local_ * ny_local_ * output_dim_.size_z[target_j];
    }

    std::size_t forward_send_offset_elems_( int target_j ) const
    {
        return output_dim_.start_z[target_j] * nx_local_ * ny_local_;
    }

    std::size_t forward_recv_chunk_elems_( int source_j ) const
    {
        return nx_local_ * input_dim_.size_y[source_j] * nz_local_;
    }

    std::size_t forward_recv_offset_elems_( int source_j ) const
    {
        return nx_local_ * nz_local_ * input_dim_.start_y[source_j];
    }

    std::size_t backward_send_chunk_elems_( int target_j ) const
    {
        return nx_local_ * input_dim_.size_y[target_j] * nz_local_;
    }

    std::size_t backward_send_pack_offset_elems_( int target_j ) const
    {
        return backward_send_offsets_[target_j];
    }

    std::size_t backward_recv_chunk_elems_( int source_j ) const
    {
        return nx_local_ * ny_local_ * output_dim_.size_z[source_j];
    }

    std::size_t backward_recv_offset_elems_( int source_j ) const
    {
        return output_dim_.start_z[source_j] * nx_local_ * ny_local_;
    }

    void init_forward_layout_()
    {
        const int row_size = row_comm_info_.num_procs;

        forward_sendcounts_.resize( row_size );
        forward_sdispls_.resize( row_size );
        forward_recvcounts_.resize( row_size );
        forward_rdispls_.resize( row_size );
        forward_sendcounts_w_.assign( row_size, 0 );
        forward_sdispls_w_.assign( row_size, 0 );
        forward_sendtypes_w_.assign( row_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );
        forward_recvcounts_w_.assign( row_size, 1 );
        forward_rdispls_w_.assign( row_size, 0 );
        forward_recvtypes_w_.assign( row_size, mpi_dtype_t() );

        for ( int p = 0; p < row_size; ++p )
        {
            forward_sendcounts_[p]   = detail::mpi_int_cast( bytes_from_elems_( forward_send_chunk_elems_( p ) ), "forward_sendcounts" );
            forward_sdispls_[p]      = detail::mpi_int_cast( bytes_from_elems_( forward_send_offset_elems_( p ) ), "forward_sdispls" );
            forward_recvcounts_[p]   = detail::mpi_int_cast( bytes_from_elems_( forward_recv_chunk_elems_( p ) ), "forward_recvcounts" );
            forward_rdispls_[p]      = detail::mpi_int_cast( bytes_from_elems_( forward_recv_offset_elems_( p ) ), "forward_rdispls" );
            forward_sendcounts_w_[p] = forward_sendcounts_[p];
            forward_sdispls_w_[p]    = forward_sdispls_[p];
            forward_rdispls_w_[p]    = detail::mpi_int_cast( input_dim_.start_y[p] * sizeof( value_type ), "forward_rdispls_w" );

            forward_recvtypes_w_[p] = scfd::communication::detail::type_vector(
                detail::mpi_int_cast( nx_local_ * nz_local_, "forward recv type count" ),
                detail::mpi_int_cast( input_dim_.size_y[p] * sizeof( value_type ), "forward recv type blocklength" ),
                detail::mpi_int_cast( ny_global_ * sizeof( value_type ), "forward recv type stride" ),
                scfd::communication::detail::mpi_data_type<char>::mpi_type()
            );
            scfd::communication::detail::type_commit( forward_recvtypes_w_[p] );
        }
    }

    void init_backward_layout_()
    {
        const int row_size = row_comm_info_.num_procs;

        backward_send_offsets_.resize( row_size );
        backward_sendcounts_.resize( row_size );
        backward_sdispls_.resize( row_size );
        backward_recvcounts_.resize( row_size );
        backward_rdispls_.resize( row_size );
        backward_sendcounts_w_.assign( row_size, 0 );
        backward_sdispls_w_.assign( row_size, 0 );
        backward_sendtypes_w_.assign( row_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );
        backward_recvcounts_w_.assign( row_size, 0 );
        backward_rdispls_w_.assign( row_size, 0 );
        backward_recvtypes_w_.assign( row_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );

        std::size_t packed_offset = 0;
        for ( int p = 0; p < row_size; ++p )
        {
            backward_send_offsets_[p] = packed_offset;
            packed_offset += backward_send_chunk_elems_( p );
        }

        for ( int p = 0; p < row_size; ++p )
        {
            backward_sendcounts_[p]   = detail::mpi_int_cast( bytes_from_elems_( backward_send_chunk_elems_( p ) ), "backward_sendcounts" );
            backward_sdispls_[p]      = detail::mpi_int_cast( bytes_from_elems_( backward_send_pack_offset_elems_( p ) ), "backward_sdispls" );
            backward_recvcounts_[p]   = detail::mpi_int_cast( bytes_from_elems_( backward_recv_chunk_elems_( p ) ), "backward_recvcounts" );
            backward_rdispls_[p]      = detail::mpi_int_cast( bytes_from_elems_( backward_recv_offset_elems_( p ) ), "backward_rdispls" );
            backward_sendcounts_w_[p] = backward_sendcounts_[p];
            backward_sdispls_w_[p]    = backward_sdispls_[p];
            backward_recvcounts_w_[p] = backward_recvcounts_[p];
            backward_rdispls_w_[p]    = backward_rdispls_[p];
        }
    }

    void free_forward_types_()
    {
        for ( std::size_t i = 0; i < forward_recvtypes_w_.size(); ++i )
        {
            scfd::communication::detail::type_free( forward_recvtypes_w_[i] );
        }
        forward_recvtypes_w_.clear();
    }

    void synchronize_streams_()
    {
        for ( std::size_t i = 0; i < streams_.size(); ++i )
        {
            CUDA_SAFE_CALL( cudaStreamSynchronize( streams_[i].stream() ) );
        }
    }

    void copy_chunk_to_output_async_(
        const value_type *src_ptr,
        std::size_t       src_y_size,
        std::size_t       dst_y_offset,
        value_type       *dst_ptr,
        cudaStream_t      stream
    ) const
    {
        cudaMemcpy3DParms params = {};
        params.srcPos            = make_cudaPos( 0, 0, 0 );
        params.srcPtr            = make_cudaPitchedPtr(
            const_cast<value_type *>( src_ptr ),
            src_y_size * sizeof( value_type ),
            src_y_size,
            nx_local_
        );

        params.dstPos = make_cudaPos( dst_y_offset * sizeof( value_type ), 0, 0 );
        params.dstPtr = make_cudaPitchedPtr(
            dst_ptr,
            ny_global_ * sizeof( value_type ),
            ny_global_,
            nx_local_
        );

        params.extent = make_cudaExtent( src_y_size * sizeof( value_type ), nx_local_, nz_local_ );
        params.kind   = cudaMemcpyDeviceToDevice;

        CUDA_SAFE_CALL( cudaMemcpy3DAsync( &params, stream ) );
    }

    void pack_backward_chunk_async_(
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

    template <class ArrayIn, class ArrayOut>
    void copy_forward_self_block_async_( const ArrayIn &in, ArrayOut &out )
    {
        copy_chunk_to_output_async_(
            in.raw_ptr() + forward_send_offset_elems_( myid_j_ ),
            ny_local_,
            input_dim_.start_y[myid_j_],
            out.raw_ptr(),
            streams_[myid_j_].stream()
        );
    }

    template <class ArrayOut>
    void unpack_forward_received_chunk_async_( int source_j, ArrayOut &out )
    {
        copy_chunk_to_output_async_(
            recv_buffer_.raw_ptr() + forward_recv_offset_elems_( source_j ),
            input_dim_.size_y[source_j],
            input_dim_.start_y[source_j],
            out.raw_ptr(),
            streams_[source_j].stream()
        );
    }

    template <class ArrayIn>
    void pack_backward_chunk_async_( int target_j, const ArrayIn &in )
    {
        pack_backward_chunk_async_(
            in.raw_ptr(),
            input_dim_.start_y[target_j],
            input_dim_.size_y[target_j],
            send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( target_j ),
            streams_[target_j].stream()
        );
    }

    template <class ArrayIn, class ArrayOut>
    void forward_p2p_waitall_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        const int row_size = row_comm_info_.num_procs;

        copy_forward_self_block_async_( in, out );

        for ( int p = 0; p < row_size; ++p )
        {
            if ( p == myid_j_ )
                continue;

            row_comm_info_.irecv(
                recv_buffer_.raw_ptr() + forward_recv_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( forward_recv_chunk_elems_( p ) ), "forward p2p recv count" ),
                scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                p,
                p,
                recv_requests_[p]
            );

            row_comm_info_.isend(
                in.raw_ptr() + forward_send_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( forward_send_chunk_elems_( p ) ), "forward p2p send count" ),
                scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                p,
                myid_j_,
                send_requests_[p]
            );
        }

        row_comm_info_.waitall( row_size, recv_requests_.data() );
        for ( int p = 0; p < row_size; ++p )
        {
            if ( p != myid_j_ )
                unpack_forward_received_chunk_async_( p, out );
        }

        synchronize_streams_();
        row_comm_info_.waitall( row_size, send_requests_.data() );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void forward_p2p_waitany_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        const int row_size = row_comm_info_.num_procs;

        copy_forward_self_block_async_( in, out );

        for ( int p = 0; p < row_size; ++p )
        {
            if ( p == myid_j_ )
                continue;

            row_comm_info_.irecv(
                recv_buffer_.raw_ptr() + forward_recv_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( forward_recv_chunk_elems_( p ) ), "forward p2p recv count" ),
                scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                p,
                p,
                recv_requests_[p]
            );

            row_comm_info_.isend(
                in.raw_ptr() + forward_send_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( forward_send_chunk_elems_( p ) ), "forward p2p send count" ),
                scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                p,
                myid_j_,
                send_requests_[p]
            );
        }

        int completed = 0;
        while ( completed < row_size - 1 )
        {
            const int p = row_comm_info_.waitany( row_size, recv_requests_.data() );
            if ( p == MPI_UNDEFINED )
                break;
            unpack_forward_received_chunk_async_( p, out );
            completed++;
        }

        synchronize_streams_();
        row_comm_info_.waitall( row_size, send_requests_.data() );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void forward_alltoallv_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        row_comm_info_.alltoallv(
            static_cast<const void *>( in.raw_ptr() ),
            forward_sendcounts_.data(),
            forward_sdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type(),
            static_cast<void *>( recv_buffer_.raw_ptr() ),
            forward_recvcounts_.data(),
            forward_rdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type()
        );

        for ( int p = 0; p < row_comm_info_.num_procs; ++p )
        {
            unpack_forward_received_chunk_async_( p, out );
        }
        synchronize_streams_();
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void forward_alltoallw_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        row_comm_info_.alltoallw(
            static_cast<const void *>( in.raw_ptr() ),
            forward_sendcounts_w_.data(),
            forward_sdispls_w_.data(),
            forward_sendtypes_w_.data(),
            static_cast<void *>( out.raw_ptr() ),
            forward_recvcounts_w_.data(),
            forward_rdispls_w_.data(),
            forward_recvtypes_w_.data()
        );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_p2p_waitall_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        const int row_size = row_comm_info_.num_procs;

        for ( int p = 0; p < row_size; ++p )
        {
            if ( p == myid_j_ )
                continue;

            pack_backward_chunk_async_( p, in );

            row_comm_info_.irecv(
                out.raw_ptr() + backward_recv_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( backward_recv_chunk_elems_( p ) ), "backward p2p recv count" ),
                scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                p,
                myid_j_,
                recv_requests_[p]
            );
        }

        synchronize_streams_();

        for ( int p = 0; p < row_size; ++p )
        {
            if ( p == myid_j_ )
                continue;

            row_comm_info_.isend(
                send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( backward_send_chunk_elems_( p ) ), "backward p2p send count" ),
                scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                p,
                p,
                send_requests_[p]
            );
        }

        CUDA_SAFE_CALL( cudaMemcpyAsync(
            out.raw_ptr() + backward_recv_offset_elems_( myid_j_ ),
            send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( myid_j_ ),
            bytes_from_elems_( backward_send_chunk_elems_( myid_j_ ) ),
            cudaMemcpyDeviceToDevice
        ) );

        row_comm_info_.waitall( row_size, recv_requests_.data() );
        row_comm_info_.waitall( row_size, send_requests_.data() );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_p2p_waitany_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        const int row_size = row_comm_info_.num_procs;

        for ( int p = 0; p < row_size; ++p )
        {
            if ( p == myid_j_ )
                continue;

            pack_backward_chunk_async_( p, in );

            row_comm_info_.irecv(
                out.raw_ptr() + backward_recv_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( backward_recv_chunk_elems_( p ) ), "backward p2p recv count" ),
                scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                p,
                myid_j_,
                recv_requests_[p]
            );
        }

        synchronize_streams_();

        for ( int p = 0; p < row_size; ++p )
        {
            if ( p == myid_j_ )
                continue;
            row_comm_info_.isend(
                send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( backward_send_chunk_elems_( p ) ), "backward p2p send count" ),
                scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                p,
                p,
                send_requests_[p]
            );
        }

        CUDA_SAFE_CALL( cudaMemcpyAsync(
            out.raw_ptr() + backward_recv_offset_elems_( myid_j_ ),
            send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( myid_j_ ),
            bytes_from_elems_( backward_send_chunk_elems_( myid_j_ ) ),
            cudaMemcpyDeviceToDevice
        ) );

        int completed = 0;
        while ( completed < row_size - 1 )
        {
            const int p = row_comm_info_.waitany( row_size, recv_requests_.data() );
            if ( p == MPI_UNDEFINED )
                break;
            completed++;
        }

        row_comm_info_.waitall( row_size, send_requests_.data() );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_alltoallv_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        for ( int p = 0; p < row_comm_info_.num_procs; ++p )
        {
            pack_backward_chunk_async_( p, in );
        }
        synchronize_streams_();

        row_comm_info_.alltoallv(
            static_cast<const void *>( send_buffer_.raw_ptr() ),
            backward_sendcounts_.data(),
            backward_sdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type(),
            static_cast<void *>( out.raw_ptr() ),
            backward_recvcounts_.data(),
            backward_rdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type()
        );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_alltoallw_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        for ( int p = 0; p < row_comm_info_.num_procs; ++p )
        {
            pack_backward_chunk_async_( p, in );
        }
        synchronize_streams_();

        row_comm_info_.alltoallw(
            static_cast<const void *>( send_buffer_.raw_ptr() ),
            backward_sendcounts_w_.data(),
            backward_sdispls_w_.data(),
            backward_sendtypes_w_.data(),
            static_cast<void *>( out.raw_ptr() ),
            backward_recvcounts_w_.data(),
            backward_rdispls_w_.data(),
            backward_recvtypes_w_.data()
        );
#endif
    }

private:
    MPIComm mpi_;
    Log     log_;

    bool is_inited_ = false;
    int  myid_i_    = 0;
    int  myid_j_    = 0;

    std::size_t nx_local_  = 0;
    std::size_t ny_local_  = 0;
    std::size_t ny_global_ = 0;
    std::size_t nz_global_ = 0;
    std::size_t nz_local_  = 0;

    partition_t input_dim_;
    partition_t output_dim_;

    std::unique_ptr<mpi_comm_t>             row_comm_;
    scfd::communication::mpi_comm_info      row_comm_info_;
    contiguous_buf_t                        send_buffer_;
    contiguous_buf_t                        recv_buffer_;
    std::vector<mpi_request_t>              send_requests_;
    std::vector<mpi_request_t>              recv_requests_;
    std::vector<scfd::utils::cuda_stream_wrap> streams_;

    std::vector<int>           forward_sendcounts_;
    std::vector<int>           forward_sdispls_;
    std::vector<int>           forward_recvcounts_;
    std::vector<int>           forward_rdispls_;
    std::vector<int>           forward_sendcounts_w_;
    std::vector<int>           forward_sdispls_w_;
    std::vector<mpi_dtype_t>   forward_sendtypes_w_;
    std::vector<int>           forward_recvcounts_w_;
    std::vector<int>           forward_rdispls_w_;
    std::vector<mpi_dtype_t>   forward_recvtypes_w_;

    std::vector<std::size_t>   backward_send_offsets_;
    std::vector<int>           backward_sendcounts_;
    std::vector<int>           backward_sdispls_;
    std::vector<int>           backward_recvcounts_;
    std::vector<int>           backward_rdispls_;
    std::vector<int>           backward_sendcounts_w_;
    std::vector<int>           backward_sdispls_w_;
    std::vector<mpi_dtype_t>   backward_sendtypes_w_;
    std::vector<int>           backward_recvcounts_w_;
    std::vector<int>           backward_rdispls_w_;
    std::vector<mpi_dtype_t>   backward_recvtypes_w_;
};

} // namespace fftm

#endif // __FFTM_DETAIL_MPI_TRANSPOSE_3D_H__
