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
    using value_type      = ValueType;
    using backend_t       = Backend;
    using memory_t        = typename backend_t::memory_type;
    using partition_t     = ::fftm::partition;
    using mpi_comm_t      = scfd::communication::mpi_comm;
    using contiguous_buf_t = scfd::arrays::array_nd<value_type, 1, memory_t>;

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
        free_derived_types_();
    }

    void init( const partition_t &input_dim, const partition_t &output_dim, int myid_i, int myid_j )
    {
        free_derived_types_();

        input_dim_ = input_dim;
        output_dim_ = output_dim;
        myid_i_ = myid_i;
        myid_j_ = myid_j;

        if ( input_dim_.size_z.size() != 1 )
        {
            throw std::logic_error( "mpi_transpose_3d expects the source layout to keep Z undistributed" );
        }
        if ( output_dim_.size_y.size() != 1 )
        {
            throw std::logic_error( "mpi_transpose_3d expects the destination layout to keep Y undistributed" );
        }

        nx_local_ = input_dim_.size_x.at( myid_i_ );
        ny_local_ = input_dim_.size_y.at( myid_j_ );
        nz_global_ = input_dim_.size_z.at( 0 );
        ny_global_ = output_dim_.size_y.at( 0 );
        nz_local_ = output_dim_.size_z.at( myid_j_ );

        if ( output_dim_.size_x.at( myid_i_ ) != nx_local_ )
        {
            throw std::logic_error( "mpi_transpose_3d requires X ownership to stay unchanged" );
        }
        if ( input_dim_.size_y.size() != output_dim_.size_z.size() )
        {
            throw std::logic_error( "mpi_transpose_3d requires matching source-Y and destination-Z partition counts" );
        }

        row_comm_ = std::make_unique<mpi_comm_t>( std::move( mpi_.split( myid_i_, myid_j_ ) ) );
        row_comm_info_ = row_comm_->info();
        const int row_size = row_comm_info_.num_procs;

        if ( row_size != static_cast<int>( input_dim_.size_y.size() ) )
        {
            throw std::logic_error( "mpi_transpose_3d row communicator size does not match Y partition count" );
        }

        recv_buffer_.init( total_recv_elems_() );

        recv_requests_.assign( row_size, MPI_REQUEST_NULL );
        send_requests_.assign( row_size, MPI_REQUEST_NULL );

        streams_.clear();
        streams_.reserve( row_size );
        for ( int p = 0; p < row_size; ++p )
        {
            streams_.emplace_back( true );
        }

        init_alltoallv_layout_();
        init_alltoallw_layout_();
        is_inited_ = true;
    }

    std::size_t local_nx() const
    {
        return nx_local_;
    }
    std::size_t local_ny() const
    {
        return ny_local_;
    }
    std::size_t local_nz() const
    {
        return nz_global_;
    }
    std::size_t output_local_z() const
    {
        return nz_local_;
    }
    std::size_t output_global_y() const
    {
        return ny_global_;
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xyz_to_xzy( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        ensure_is_inited_();
        verify_input_output_shapes_( in, out );
        reset_requests_();

        switch ( mode )
        {
            case mpi_transpose_3d_mode::p2p_waitall:
                transpose_p2p_waitall_( in, out );
                break;
            case mpi_transpose_3d_mode::p2p_waitany:
                transpose_p2p_waitany_( in, out );
                break;
            case mpi_transpose_3d_mode::alltoallv:
                transpose_alltoallv_( in, out );
                break;
            case mpi_transpose_3d_mode::alltoallw:
                transpose_alltoallw_( in, out );
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
    void verify_input_output_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size = in.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nz_global_ )
        {
            throw std::logic_error(
                "mpi_transpose_3d input shape mismatch: expected (" + std::to_string( nx_local_ ) + ", " +
                std::to_string( ny_local_ ) + ", " + std::to_string( nz_global_ ) + ")"
            );
        }

        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != ny_global_ )
        {
            throw std::logic_error(
                "mpi_transpose_3d output shape mismatch: expected (" + std::to_string( nx_local_ ) + ", " +
                std::to_string( nz_local_ ) + ", " + std::to_string( ny_global_ ) + ")"
            );
        }
    }

    void reset_requests_()
    {
        std::fill( recv_requests_.begin(), recv_requests_.end(), MPI_REQUEST_NULL );
        std::fill( send_requests_.begin(), send_requests_.end(), MPI_REQUEST_NULL );
    }

    std::size_t send_chunk_elems_( int target_j ) const
    {
        return nx_local_ * ny_local_ * output_dim_.size_z[target_j];
    }

    std::size_t recv_chunk_elems_( int source_j ) const
    {
        return nx_local_ * input_dim_.size_y[source_j] * nz_local_;
    }

    std::size_t send_offset_elems_( int target_j ) const
    {
        return output_dim_.start_z[target_j] * nx_local_ * ny_local_;
    }

    std::size_t recv_offset_elems_( int source_j ) const
    {
        return nx_local_ * nz_local_ * input_dim_.start_y[source_j];
    }

    std::size_t total_recv_elems_() const
    {
        return nx_local_ * ny_global_ * nz_local_;
    }

    std::size_t bytes_from_elems_( std::size_t elements ) const
    {
        return elements * sizeof( value_type );
    }

    void init_alltoallv_layout_()
    {
        const int row_size = row_comm_info_.num_procs;

        sendcounts_.resize( row_size );
        sdispls_.resize( row_size );
        recvcounts_.resize( row_size );
        rdispls_.resize( row_size );

        for ( int p = 0; p < row_size; ++p )
        {
            sendcounts_[p] = detail::mpi_int_cast( bytes_from_elems_( send_chunk_elems_( p ) ), "sendcounts" );
            sdispls_[p]    = detail::mpi_int_cast( bytes_from_elems_( send_offset_elems_( p ) ), "sdispls" );
            recvcounts_[p] = detail::mpi_int_cast( bytes_from_elems_( recv_chunk_elems_( p ) ), "recvcounts" );
            rdispls_[p]    = detail::mpi_int_cast( bytes_from_elems_( recv_offset_elems_( p ) ), "rdispls" );
        }
    }

    void init_alltoallw_layout_()
    {
        const int row_size = row_comm_info_.num_procs;

        sendcounts_w_.resize( row_size );
        sdispls_w_.resize( row_size );
        sendtypes_w_.resize( row_size, MPI_BYTE );
        recvcounts_w_.resize( row_size, 1 );
        rdispls_w_.resize( row_size );
        recvtypes_w_.resize( row_size, MPI_DATATYPE_NULL );

        for ( int p = 0; p < row_size; ++p )
        {
            sendcounts_w_[p] = detail::mpi_int_cast( bytes_from_elems_( send_chunk_elems_( p ) ), "sendcounts_w" );
            sdispls_w_[p]    = detail::mpi_int_cast( bytes_from_elems_( send_offset_elems_( p ) ), "sdispls_w" );
            rdispls_w_[p]    = detail::mpi_int_cast(
                input_dim_.start_y[p] * sizeof( value_type ),
                "rdispls_w"
            );

            recvtypes_w_[p] = scfd::communication::detail::type_vector(
                detail::mpi_int_cast( nx_local_ * nz_local_, "recv type count" ),
                detail::mpi_int_cast( input_dim_.size_y[p] * sizeof( value_type ), "recv type blocklength" ),
                detail::mpi_int_cast( ny_global_ * sizeof( value_type ), "recv type stride" ),
                MPI_BYTE
            );
            scfd::communication::detail::type_commit( recvtypes_w_[p] );
        }
    }

    void free_derived_types_()
    {
        for ( std::size_t i = 0; i < recvtypes_w_.size(); ++i )
        {
            scfd::communication::detail::type_free( recvtypes_w_[i] );
        }
        recvtypes_w_.clear();
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

        params.srcPos = make_cudaPos( 0, 0, 0 );
        params.srcPtr = make_cudaPitchedPtr(
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

    template <class ArrayIn, class ArrayOut>
    void copy_self_block_async_( const ArrayIn &in, ArrayOut &out )
    {
        copy_chunk_to_output_async_(
            in.raw_ptr() + send_offset_elems_( myid_j_ ),
            ny_local_,
            input_dim_.start_y[myid_j_],
            out.raw_ptr(),
            streams_[myid_j_].stream()
        );
    }

    template <class ArrayOut>
    void unpack_received_chunk_async_( int source_j, ArrayOut &out )
    {
        copy_chunk_to_output_async_(
            recv_buffer_.raw_ptr() + recv_offset_elems_( source_j ),
            input_dim_.size_y[source_j],
            input_dim_.start_y[source_j],
            out.raw_ptr(),
            streams_[source_j].stream()
        );
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_p2p_waitall_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        const int row_size = row_comm_info_.num_procs;

        copy_self_block_async_( in, out );

        for ( int p = 0; p < row_size; ++p )
        {
            if ( p == myid_j_ )
                continue;

            row_comm_info_.irecv(
                recv_buffer_.raw_ptr() + recv_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( recv_chunk_elems_( p ) ), "p2p recv count" ),
                MPI_BYTE,
                p,
                p,
                recv_requests_[p]
            );

            row_comm_info_.isend(
                in.raw_ptr() + send_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( send_chunk_elems_( p ) ), "p2p send count" ),
                MPI_BYTE,
                p,
                myid_j_,
                send_requests_[p]
            );
        }

        row_comm_info_.waitall( row_size, recv_requests_.data() );

        for ( int p = 0; p < row_size; ++p )
        {
            if ( p == myid_j_ )
                continue;
            unpack_received_chunk_async_( p, out );
        }

        synchronize_streams_();
        row_comm_info_.waitall( row_size, send_requests_.data() );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_p2p_waitany_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        const int row_size = row_comm_info_.num_procs;

        copy_self_block_async_( in, out );

        for ( int p = 0; p < row_size; ++p )
        {
            if ( p == myid_j_ )
                continue;

            row_comm_info_.irecv(
                recv_buffer_.raw_ptr() + recv_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( recv_chunk_elems_( p ) ), "p2p recv count" ),
                MPI_BYTE,
                p,
                p,
                recv_requests_[p]
            );

            row_comm_info_.isend(
                in.raw_ptr() + send_offset_elems_( p ),
                detail::mpi_int_cast( bytes_from_elems_( send_chunk_elems_( p ) ), "p2p send count" ),
                MPI_BYTE,
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

            unpack_received_chunk_async_( p, out );
            completed++;
        }

        synchronize_streams_();
        row_comm_info_.waitall( row_size, send_requests_.data() );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_alltoallv_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        row_comm_info_.alltoallv(
            static_cast<const void *>( in.raw_ptr() ),
            sendcounts_.data(),
            sdispls_.data(),
            MPI_BYTE,
            static_cast<void *>( recv_buffer_.raw_ptr() ),
            recvcounts_.data(),
            rdispls_.data(),
            MPI_BYTE
        );

        for ( int p = 0; p < row_comm_info_.num_procs; ++p )
        {
            unpack_received_chunk_async_( p, out );
        }

        synchronize_streams_();
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_alltoallw_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_3d currently requires SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        row_comm_info_.alltoallw(
            static_cast<const void *>( in.raw_ptr() ),
            sendcounts_w_.data(),
            sdispls_w_.data(),
            sendtypes_w_.data(),
            static_cast<void *>( out.raw_ptr() ),
            recvcounts_w_.data(),
            rdispls_w_.data(),
            recvtypes_w_.data()
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

    std::unique_ptr<mpi_comm_t> row_comm_;
    scfd::communication::mpi_comm_info row_comm_info_;

    contiguous_buf_t recv_buffer_;

    std::vector<MPI_Request> send_requests_;
    std::vector<MPI_Request> recv_requests_;
    std::vector<scfd::utils::cuda_stream_wrap> streams_;

    std::vector<int> sendcounts_;
    std::vector<int> sdispls_;
    std::vector<int> recvcounts_;
    std::vector<int> rdispls_;

    std::vector<int> sendcounts_w_;
    std::vector<int> sdispls_w_;
    std::vector<MPI_Datatype> sendtypes_w_;
    std::vector<int> recvcounts_w_;
    std::vector<int> rdispls_w_;
    std::vector<MPI_Datatype> recvtypes_w_;
};

} // namespace fftm

#endif // __FFTM_DETAIL_MPI_TRANSPOSE_3D_H__
