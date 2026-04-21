#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_4D_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_4D_H__

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_mpi.h>

#include "../fft_partitioning.h"
#include "../profiling.h"
#include "mpi_transpose_3d.h"

namespace fftm
{
namespace detail
{

template <class Idx>
__DEVICE_TAG__ inline std::size_t packed_index_4d( const Idx &idx, std::size_t d0, std::size_t d1, std::size_t d2 )
{
    return static_cast<std::size_t>( idx[0] ) +
           d0 * ( static_cast<std::size_t>( idx[1] ) +
                  d1 * ( static_cast<std::size_t>( idx[2] ) + d2 * static_cast<std::size_t>( idx[3] ) ) );
}

template <class Idx>
__DEVICE_TAG__ inline std::size_t
packed_index_4d( int i0, int i1, int i2, int i3, std::size_t d0, std::size_t d1, std::size_t d2 )
{
    return static_cast<std::size_t>( i0 ) +
           d0 * ( static_cast<std::size_t>( i1 ) +
                  d1 * ( static_cast<std::size_t>( i2 ) + d2 * static_cast<std::size_t>( i3 ) ) );
}

template <class Idx>
using rect_4d_t = scfd::static_vec::rect<int, 4>;

template <class Idx>
inline rect_4d_t<Idx> make_range_4d( std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3 )
{
    return rect_4d_t<Idx>(
        Idx( 0, 0, 0, 0 ),
        Idx( static_cast<int>( d0 ), static_cast<int>( d1 ), static_cast<int>( d2 ), static_cast<int>( d3 ) )
    );
}

template <
    class ValueType, class Backend, class MPIComm, class Log = scfd::utils::log_mpi,
    class RuntimeAPI = ::fftm::wrap::cuda_runtime_api>
class mpi_transpose_4d_same_xy
{
public:
    using value_type        = ValueType;
    using backend_t         = Backend;
    using memory_t          = typename backend_t::memory_type;
    using partition_t       = ::fftm::partition;
    using mpi_comm_t        = scfd::communication::mpi_comm;
    using contiguous_buf_t  = scfd::arrays::array_nd<value_type, 1, memory_t>;
    using mpi_request_t     = scfd::communication::detail::mpi_request;
    using mpi_dtype_t       = scfd::communication::detail::mpi_data_type<>;
    using idx_t             = scfd::static_vec::vec<int, 4>;
    using range_t           = rect_4d_t<idx_t>;
    using for_each_t        = typename backend_t::template for_each_nd_type<4, int>;
    using runtime_api_t     = RuntimeAPI;
    using profiler_t        = ::fftm::fftm_profiler;
    using profiler_scope_t  = ::fftm::profile_scope<profiler_t>;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

    mpi_transpose_4d_same_xy( const MPIComm &mpi, const Log &log = Log() ) : mpi_( mpi ), log_( log )
    {
        static_assert(
            std::is_same<memory_t, typename runtime_api_t::memory_type>::value,
            "mpi_transpose_4d_same_xy currently requires a CUDA backend memory "
            "type"
        );
        for_each_.block_size = 128;
    }

    void set_profiler( profiler_t *profiler )
    {
        profiler_ = profiler;
    }

    void set_memory_profiler( memory_profiler_t *profiler, const std::string &prefix )
    {
        memory_profiler_       = profiler;
        memory_profile_prefix_ = prefix;
        if ( is_inited_ )
        {
            update_memory_profile_();
        }
    }

    void use_external_work_area()
    {
        use_external_work_area_ = true;
    }

    std::size_t get_work_size_bytes() const
    {
        return bytes_from_elems_( send_buffer_elems_ ) + bytes_from_elems_( recv_buffer_elems_ );
    }

    void set_external_work_area( void *external_work_area )
    {
        if ( !use_external_work_area_ )
        {
            throw std::logic_error(
                "mpi_transpose_4d_same_xy::set_external_work_area: external work area was not enabled."
            );
        }
        external_work_area_ = external_work_area;
        bind_external_work_area_();
        update_memory_profile_();
    }

    void init( const partition_t &input_dim, const partition_t &output_dim, int myid_i, int myid_j, int myid_k )
    {
        auto scope  = profile_scope_( "mpi_transpose_4d_same_xy::init" );
        input_dim_  = input_dim;
        output_dim_ = output_dim;
        myid_i_     = myid_i;
        myid_j_     = myid_j;
        myid_k_     = myid_k;

        if ( input_dim_.size_w.size() != 1 )
            throw std::logic_error( "mpi_transpose_4d_same_xy expects the "
                                    "source layout to keep W undistributed" );
        if ( output_dim_.size_z.size() != 1 )
            throw std::logic_error( "mpi_transpose_4d_same_xy expects the destination layout to "
                                    "keep Z undistributed" );
        if ( input_dim_.size_x != output_dim_.size_x || input_dim_.size_y != output_dim_.size_y )
            throw std::logic_error( "mpi_transpose_4d_same_xy requires X/Y "
                                    "ownership to stay unchanged" );
        if ( input_dim_.size_z.size() != output_dim_.size_w.size() )
            throw std::logic_error( "mpi_transpose_4d_same_xy requires matching source-Z and "
                                    "destination-W partition counts" );

        nx_local_  = input_dim_.size_x.at( myid_i_ );
        ny_local_  = input_dim_.size_y.at( myid_j_ );
        nz_local_  = input_dim_.size_z.at( myid_k_ );
        nz_global_ = output_dim_.size_z.at( 0 );
        nw_global_ = input_dim_.size_w.at( 0 );
        nw_local_  = output_dim_.size_w.at( myid_k_ );

        const int color = myid_i_ * static_cast<int>( input_dim_.size_y.size() ) + myid_j_;
        line_comm_      = std::make_unique<mpi_comm_t>( std::move( mpi_.split( color, myid_k_ ) ) );
        line_comm_info_ = line_comm_->info();

        if ( line_comm_info_.num_procs != static_cast<int>( input_dim_.size_z.size() ) )
            throw std::logic_error( "mpi_transpose_4d_same_xy communicator size mismatch" );

        init_layouts_();
        send_buffer_elems_ = max_buffer_elems_;
        recv_buffer_elems_ = max_buffer_elems_;
        if ( !use_external_work_area_ )
        {
            send_buffer_.init( send_buffer_elems_ );
            recv_buffer_.init( recv_buffer_elems_ );
        }
        send_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        recv_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        is_inited_ = true;
        update_memory_profile_();
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xyzw_to_xywz( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_xy::transpose_xyzw_to_xywz/" ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_forward_shapes_( in, out );
        reset_requests_();
        pack_forward_all_( in );

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
    void transpose_xywz_to_xyzw( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_xy::transpose_xywz_to_xyzw/" ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_backward_shapes_( in, out );
        reset_requests_();
        pack_backward_all_( in );

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            backward_p2p_waitall_( out );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            backward_p2p_waitany_( out );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            backward_alltoallv_( out );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            backward_alltoallw_( out );
            break;
        }
    }

public:
    template <class Buffer, class ArrayIn>
    struct pack_forward_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t w_offset;
        std::size_t nx;
        std::size_t ny;
        std::size_t nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer( offset + packed_index_4d( idx, nx, ny, nz ) ) =
                in( idx[0], idx[1], idx[2], static_cast<int>( w_offset ) + idx[3] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_forward_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t z_offset;
        std::size_t nx;
        std::size_t ny;
        std::size_t nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[0], idx[1], idx[3], static_cast<int>( z_offset ) + idx[2] ) =
                buffer( offset + packed_index_4d( idx, nx, ny, nz ) );
        }
    };

    template <class Buffer, class ArrayIn>
    struct pack_backward_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t z_offset;
        std::size_t nx;
        std::size_t ny;
        std::size_t nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer( offset + packed_index_4d( idx, nx, ny, nz ) ) =
                in( idx[0], idx[1], idx[3], static_cast<int>( z_offset ) + idx[2] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_backward_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t w_offset;
        std::size_t nx;
        std::size_t ny;
        std::size_t nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[0], idx[1], idx[2], static_cast<int>( w_offset ) + idx[3] ) =
                buffer( offset + packed_index_4d( idx, nx, ny, nz ) );
        }
    };

private:
    profiler_scope_t profile_scope_( const std::string &name )
    {
        return profiler_scope_t( profiler_, name );
    }

    void ensure_is_inited_() const
    {
        if ( !is_inited_ )
            throw std::logic_error( "mpi_transpose_4d_same_xy::init must be "
                                    "called before transpose" );
        if ( use_external_work_area_ && external_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_4d_same_xy: external work area was not bound." );
        }
    }

    void update_memory_profile_()
    {
        if ( memory_profiler_ == nullptr || memory_profile_prefix_.empty() )
        {
            return;
        }

        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/send_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<memory_profiler_t::bytes_type>( send_buffer_.size() ) *
                                          static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/recv_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<memory_profiler_t::bytes_type>( recv_buffer_.size() ) *
                                          static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
    }

    void bind_external_work_area_()
    {
        if ( external_work_area_ == nullptr )
        {
            return;
        }
        char *raw = static_cast<char *>( external_work_area_ );
        send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), send_buffer_elems_ );
        recv_buffer_.init_by_raw_data(
            reinterpret_cast<value_type *>( raw + bytes_from_elems_( send_buffer_elems_ ) ), recv_buffer_elems_
        );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_forward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[3] ) != nw_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_xy forward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != nz_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_xy forward output shape mismatch" );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_backward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != nz_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_xy backward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[3] ) != nw_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_xy backward output shape mismatch" );
    }

    void reset_requests_()
    {
        std::fill( send_requests_.begin(), send_requests_.end(), mpi_request_t() );
        std::fill( recv_requests_.begin(), recv_requests_.end(), mpi_request_t() );
    }

    std::size_t bytes_from_elems_( std::size_t elems ) const
    {
        return elems * sizeof( value_type );
    }

    std::size_t forward_send_chunk_elems_( int target_k ) const
    {
        return nx_local_ * ny_local_ * nz_local_ * output_dim_.size_w[target_k];
    }

    std::size_t forward_recv_chunk_elems_( int source_k ) const
    {
        return nx_local_ * ny_local_ * input_dim_.size_z[source_k] * nw_local_;
    }

    std::size_t backward_send_chunk_elems_( int target_k ) const
    {
        return nx_local_ * ny_local_ * input_dim_.size_z[target_k] * nw_local_;
    }

    std::size_t backward_recv_chunk_elems_( int source_k ) const
    {
        return nx_local_ * ny_local_ * nz_local_ * output_dim_.size_w[source_k];
    }

    void init_layouts_()
    {
        const int comm_size = line_comm_info_.num_procs;

        forward_send_offsets_.resize( comm_size );
        forward_recv_offsets_.resize( comm_size );
        forward_sendcounts_.resize( comm_size );
        forward_sdispls_.resize( comm_size );
        forward_recvcounts_.resize( comm_size );
        forward_rdispls_.resize( comm_size );
        forward_sendcounts_w_.resize( comm_size );
        forward_sdispls_w_.resize( comm_size );
        forward_sendtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );
        forward_recvcounts_w_.resize( comm_size );
        forward_rdispls_w_.resize( comm_size );
        forward_recvtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );

        backward_send_offsets_.resize( comm_size );
        backward_recv_offsets_.resize( comm_size );
        backward_sendcounts_.resize( comm_size );
        backward_sdispls_.resize( comm_size );
        backward_recvcounts_.resize( comm_size );
        backward_rdispls_.resize( comm_size );
        backward_sendcounts_w_.resize( comm_size );
        backward_sdispls_w_.resize( comm_size );
        backward_sendtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );
        backward_recvcounts_w_.resize( comm_size );
        backward_rdispls_w_.resize( comm_size );
        backward_recvtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );

        std::size_t packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            forward_send_offsets_[p] = packed_offset;
            packed_offset += forward_send_chunk_elems_( p );
        }
        max_buffer_elems_ = packed_offset;

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            forward_recv_offsets_[p] = packed_offset;
            packed_offset += forward_recv_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            backward_send_offsets_[p] = packed_offset;
            packed_offset += backward_send_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            backward_recv_offsets_[p] = packed_offset;
            packed_offset += backward_recv_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        for ( int p = 0; p < comm_size; ++p )
        {
            forward_sendcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( forward_send_chunk_elems_( p ) ), "same_xy forward sendcount"
            );
            forward_sdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( forward_send_offsets_[p] ), "same_xy forward sdispl" );
            forward_recvcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( forward_recv_chunk_elems_( p ) ), "same_xy forward recvcount"
            );
            forward_rdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( forward_recv_offsets_[p] ), "same_xy forward rdispl" );
            forward_sendcounts_w_[p] = forward_sendcounts_[p];
            forward_sdispls_w_[p]    = forward_sdispls_[p];
            forward_recvcounts_w_[p] = forward_recvcounts_[p];
            forward_rdispls_w_[p]    = forward_rdispls_[p];

            backward_sendcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( backward_send_chunk_elems_( p ) ), "same_xy backward sendcount"
            );
            backward_sdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( backward_send_offsets_[p] ), "same_xy backward sdispl" );
            backward_recvcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( backward_recv_chunk_elems_( p ) ), "same_xy backward recvcount"
            );
            backward_rdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( backward_recv_offsets_[p] ), "same_xy backward rdispl" );
            backward_sendcounts_w_[p] = backward_sendcounts_[p];
            backward_sdispls_w_[p]    = backward_sdispls_[p];
            backward_recvcounts_w_[p] = backward_recvcounts_[p];
            backward_rdispls_w_[p]    = backward_rdispls_[p];
        }
    }

    template <class ArrayIn>
    void pack_forward_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_forward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            for_each_(
                pack_forward_functor<contiguous_buf_t, ArrayIn>{
                    send_buffer_, in, forward_send_offsets_[p], output_dim_.start_w[p], nx_local_, ny_local_,
                    nz_local_ },
                make_range_4d<idx_t>( nx_local_, ny_local_, nz_local_, output_dim_.size_w[p] )
            );
            for_each_.wait();
        }
    }

    template <class ArrayOut>
    void unpack_forward_chunk_( int source_k, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_forward_chunk" );
        for_each_(
            unpack_forward_functor<contiguous_buf_t, ArrayOut>{
                recv_buffer_, out, forward_recv_offsets_[source_k], input_dim_.start_z[source_k], nx_local_, ny_local_,
                input_dim_.size_z[source_k] },
            make_range_4d<idx_t>( nx_local_, ny_local_, input_dim_.size_z[source_k], nw_local_ )
        );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_backward_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_backward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            for_each_(
                pack_backward_functor<contiguous_buf_t, ArrayIn>{
                    send_buffer_, in, backward_send_offsets_[p], input_dim_.start_z[p], nx_local_, ny_local_,
                    input_dim_.size_z[p] },
                make_range_4d<idx_t>( nx_local_, ny_local_, input_dim_.size_z[p], nw_local_ )
            );
            for_each_.wait();
        }
    }

    template <class ArrayOut>
    void unpack_backward_chunk_( int source_k, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_backward_chunk" );
        for_each_(
            unpack_backward_functor<contiguous_buf_t, ArrayOut>{
                recv_buffer_, out, backward_recv_offsets_[source_k], output_dim_.start_w[source_k], nx_local_,
                ny_local_, nz_local_ },
            make_range_4d<idx_t>( nx_local_, ny_local_, nz_local_, output_dim_.size_w[source_k] )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void forward_p2p_waitall_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xy currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "forward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_k_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                    scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                    scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_k_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_k_],
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_k_],
                bytes_from_elems_( forward_recv_chunk_elems_( myid_k_ ) ), runtime_api_t::device_to_device_kind()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                unpack_forward_chunk_( p, out );
        }
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayOut>
    void forward_p2p_waitany_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xy currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "forward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_k_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                    scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                    scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_k_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_k_],
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_k_],
                bytes_from_elems_( forward_recv_chunk_elems_( myid_k_ ) ), runtime_api_t::device_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_forward_chunk_( myid_k_, out );
        }

        {
            auto phase     = profile_scope_( "waitany_unpack" );
            int  completed = 0;
            while ( completed < comm_size - 1 )
            {
                const int p = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                if ( p == MPI_UNDEFINED )
                    break;
                unpack_forward_chunk_( p, out );
                completed++;
            }
        }

        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayOut>
    void forward_alltoallv_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xy currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "forward_alltoallv" );
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            line_comm_info_.alltoallv(
                static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                forward_sdispls_.data(), scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                static_cast<void *>( recv_buffer_.raw_ptr() ), forward_recvcounts_.data(), forward_rdispls_.data(),
                scfd::communication::detail::mpi_data_type<char>::mpi_type()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_forward_chunk_( p, out );
        }
#endif
    }

    template <class ArrayOut>
    void forward_alltoallw_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xy currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "forward_alltoallw" );
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_w_.data(),
                forward_sdispls_w_.data(), forward_sendtypes_w_.data(), static_cast<void *>( recv_buffer_.raw_ptr() ),
                forward_recvcounts_w_.data(), forward_rdispls_w_.data(), forward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_forward_chunk_( p, out );
        }
#endif
    }

    template <class ArrayOut>
    void backward_p2p_waitall_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xy currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "backward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_k_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                    scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_k_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                    scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_k_],
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_k_],
                bytes_from_elems_( backward_recv_chunk_elems_( myid_k_ ) ), runtime_api_t::device_to_device_kind()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                unpack_backward_chunk_( p, out );
        }
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayOut>
    void backward_p2p_waitany_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xy currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "backward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_k_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                    scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_k_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                    scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_k_],
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_k_],
                bytes_from_elems_( backward_recv_chunk_elems_( myid_k_ ) ), runtime_api_t::device_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_backward_chunk_( myid_k_, out );
        }

        {
            auto phase     = profile_scope_( "waitany_unpack" );
            int  completed = 0;
            while ( completed < comm_size - 1 )
            {
                const int p = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                if ( p == MPI_UNDEFINED )
                    break;
                unpack_backward_chunk_( p, out );
                completed++;
            }
        }

        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayOut>
    void backward_alltoallv_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xy currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "backward_alltoallv" );
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            line_comm_info_.alltoallv(
                static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                backward_sdispls_.data(), scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                static_cast<void *>( recv_buffer_.raw_ptr() ), backward_recvcounts_.data(), backward_rdispls_.data(),
                scfd::communication::detail::mpi_data_type<char>::mpi_type()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_backward_chunk_( p, out );
        }
#endif
    }

    template <class ArrayOut>
    void backward_alltoallw_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xy currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "backward_alltoallw" );
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
                backward_sdispls_w_.data(), backward_sendtypes_w_.data(), static_cast<void *>( recv_buffer_.raw_ptr() ),
                backward_recvcounts_w_.data(), backward_rdispls_w_.data(), backward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_backward_chunk_( p, out );
        }
#endif
    }

private:
    MPIComm            mpi_;
    Log                log_;
    profiler_t        *profiler_        = nullptr;
    memory_profiler_t *memory_profiler_ = nullptr;
    std::string        memory_profile_prefix_;

    bool is_inited_ = false;
    int  myid_i_    = 0;
    int  myid_j_    = 0;
    int  myid_k_    = 0;

    std::size_t nx_local_  = 0;
    std::size_t ny_local_  = 0;
    std::size_t nz_local_  = 0;
    std::size_t nz_global_ = 0;
    std::size_t nw_local_  = 0;
    std::size_t nw_global_ = 0;

    partition_t input_dim_;
    partition_t output_dim_;

    std::unique_ptr<mpi_comm_t>        line_comm_;
    scfd::communication::mpi_comm_info line_comm_info_;
    contiguous_buf_t                   send_buffer_;
    contiguous_buf_t                   recv_buffer_;
    std::size_t                        send_buffer_elems_      = 0;
    std::size_t                        recv_buffer_elems_      = 0;
    bool                               use_external_work_area_ = false;
    void                              *external_work_area_     = nullptr;
    for_each_t                         for_each_;
    std::vector<mpi_request_t>         send_requests_;
    std::vector<mpi_request_t>         recv_requests_;

    std::vector<std::size_t> forward_send_offsets_;
    std::vector<std::size_t> forward_recv_offsets_;
    std::vector<int>         forward_sendcounts_;
    std::vector<int>         forward_sdispls_;
    std::vector<int>         forward_recvcounts_;
    std::vector<int>         forward_rdispls_;
    std::vector<int>         forward_sendcounts_w_;
    std::vector<int>         forward_sdispls_w_;
    std::vector<mpi_dtype_t> forward_sendtypes_w_;
    std::vector<int>         forward_recvcounts_w_;
    std::vector<int>         forward_rdispls_w_;
    std::vector<mpi_dtype_t> forward_recvtypes_w_;

    std::vector<std::size_t> backward_send_offsets_;
    std::vector<std::size_t> backward_recv_offsets_;
    std::vector<int>         backward_sendcounts_;
    std::vector<int>         backward_sdispls_;
    std::vector<int>         backward_recvcounts_;
    std::vector<int>         backward_rdispls_;
    std::vector<int>         backward_sendcounts_w_;
    std::vector<int>         backward_sdispls_w_;
    std::vector<mpi_dtype_t> backward_sendtypes_w_;
    std::vector<int>         backward_recvcounts_w_;
    std::vector<int>         backward_rdispls_w_;
    std::vector<mpi_dtype_t> backward_recvtypes_w_;

    std::size_t max_buffer_elems_ = 0;
};

template <
    class ValueType, class Backend, class MPIComm, class Log = scfd::utils::log_mpi,
    class RuntimeAPI = ::fftm::wrap::cuda_runtime_api>
class mpi_transpose_4d_same_xw
{
public:
    using value_type        = ValueType;
    using backend_t         = Backend;
    using memory_t          = typename backend_t::memory_type;
    using partition_t       = ::fftm::partition;
    using mpi_comm_t        = scfd::communication::mpi_comm;
    using contiguous_buf_t  = scfd::arrays::array_nd<value_type, 1, memory_t>;
    using mpi_request_t     = scfd::communication::detail::mpi_request;
    using mpi_dtype_t       = scfd::communication::detail::mpi_data_type<>;
    using idx_t             = scfd::static_vec::vec<int, 4>;
    using range_t           = rect_4d_t<idx_t>;
    using for_each_t        = typename backend_t::template for_each_nd_type<4, int>;
    using runtime_api_t     = RuntimeAPI;
    using profiler_t        = ::fftm::fftm_profiler;
    using profiler_scope_t  = ::fftm::profile_scope<profiler_t>;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

    mpi_transpose_4d_same_xw( const MPIComm &mpi, const Log &log = Log() ) : mpi_( mpi ), log_( log )
    {
        static_assert(
            std::is_same<memory_t, typename runtime_api_t::memory_type>::value,
            "mpi_transpose_4d_same_xw currently requires a CUDA backend memory "
            "type"
        );
        for_each_.block_size = 128;
    }

    void set_profiler( profiler_t *profiler )
    {
        profiler_ = profiler;
    }

    void set_memory_profiler( memory_profiler_t *profiler, const std::string &prefix )
    {
        memory_profiler_       = profiler;
        memory_profile_prefix_ = prefix;
        if ( is_inited_ )
        {
            update_memory_profile_();
        }
    }

    void use_external_work_area()
    {
        use_external_work_area_ = true;
    }

    std::size_t get_work_size_bytes() const
    {
        return bytes_from_elems_( send_buffer_elems_ ) + bytes_from_elems_( recv_buffer_elems_ );
    }

    void set_external_work_area( void *external_work_area )
    {
        if ( !use_external_work_area_ )
        {
            throw std::logic_error(
                "mpi_transpose_4d_same_xw::set_external_work_area: external work area was not enabled."
            );
        }
        external_work_area_ = external_work_area;
        bind_external_work_area_();
        update_memory_profile_();
    }

    void init( const partition_t &input_dim, const partition_t &output_dim, int myid_i, int myid_j, int myid_k )
    {
        auto scope  = profile_scope_( "mpi_transpose_4d_same_xw::init" );
        input_dim_  = input_dim;
        output_dim_ = output_dim;
        myid_i_     = myid_i;
        myid_j_     = myid_j;
        myid_k_     = myid_k;

        if ( input_dim_.size_z.size() != 1 )
            throw std::logic_error( "mpi_transpose_4d_same_xw expects the "
                                    "source layout to keep Z undistributed" );
        if ( output_dim_.size_y.size() != 1 )
            throw std::logic_error( "mpi_transpose_4d_same_xw expects the destination layout to "
                                    "keep Y undistributed" );
        if ( input_dim_.size_x != output_dim_.size_x || input_dim_.size_w != output_dim_.size_w )
            throw std::logic_error( "mpi_transpose_4d_same_xw requires X/W "
                                    "ownership to stay unchanged" );
        if ( input_dim_.size_y.size() != output_dim_.size_z.size() )
            throw std::logic_error( "mpi_transpose_4d_same_xw requires matching source-Y and "
                                    "destination-Z partition counts" );

        nx_local_  = input_dim_.size_x.at( myid_i_ );
        ny_local_  = input_dim_.size_y.at( myid_j_ );
        ny_global_ = output_dim_.size_y.at( 0 );
        nz_global_ = input_dim_.size_z.at( 0 );
        nz_local_  = output_dim_.size_z.at( myid_j_ );
        nw_local_  = input_dim_.size_w.at( myid_k_ );

        const int color = myid_i_ * static_cast<int>( input_dim_.size_w.size() ) + myid_k_;
        line_comm_      = std::make_unique<mpi_comm_t>( std::move( mpi_.split( color, myid_j_ ) ) );
        line_comm_info_ = line_comm_->info();

        if ( line_comm_info_.num_procs != static_cast<int>( input_dim_.size_y.size() ) )
            throw std::logic_error( "mpi_transpose_4d_same_xw communicator size mismatch" );

        init_layouts_();
        send_buffer_elems_ = max_buffer_elems_;
        recv_buffer_elems_ = max_buffer_elems_;
        if ( !use_external_work_area_ )
        {
            send_buffer_.init( send_buffer_elems_ );
            recv_buffer_.init( recv_buffer_elems_ );
        }
        send_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        recv_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        is_inited_ = true;
        update_memory_profile_();
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xywz_to_xzwy( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_xw::transpose_xywz_to_xzwy/" ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_forward_shapes_( in, out );
        reset_requests_();
        pack_forward_all_( in );

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
    void transpose_xzwy_to_xywz( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_xw::transpose_xzwy_to_xywz/" ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_backward_shapes_( in, out );
        reset_requests_();
        pack_backward_all_( in );

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            backward_p2p_waitall_( out );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            backward_p2p_waitany_( out );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            backward_alltoallv_( out );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            backward_alltoallw_( out );
            break;
        }
    }

public:
    template <class Buffer, class ArrayIn>
    struct pack_forward_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t z_offset;
        std::size_t nx;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer( offset + packed_index_4d( idx, nx, nz, nw ) ) =
                in( idx[0], idx[3], idx[2], static_cast<int>( z_offset ) + idx[1] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_forward_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t y_offset;
        std::size_t nx;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[0], idx[1], idx[2], static_cast<int>( y_offset ) + idx[3] ) =
                buffer( offset + packed_index_4d( idx, nx, nz, nw ) );
        }
    };

    template <class Buffer, class ArrayIn>
    struct pack_backward_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t y_offset;
        std::size_t nx;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer( offset + packed_index_4d( idx, nx, nz, nw ) ) =
                in( idx[0], idx[1], idx[2], static_cast<int>( y_offset ) + idx[3] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_backward_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t z_offset;
        std::size_t nx;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[0], idx[3], idx[2], static_cast<int>( z_offset ) + idx[1] ) =
                buffer( offset + packed_index_4d( idx, nx, nz, nw ) );
        }
    };

private:
    profiler_scope_t profile_scope_( const std::string &name )
    {
        return profiler_scope_t( profiler_, name );
    }

    void ensure_is_inited_() const
    {
        if ( !is_inited_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw::init must be "
                                    "called before transpose" );
        if ( use_external_work_area_ && external_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_4d_same_xw: external work area was not bound." );
        }
    }

    void update_memory_profile_()
    {
        if ( memory_profiler_ == nullptr || memory_profile_prefix_.empty() )
        {
            return;
        }

        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/send_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<memory_profiler_t::bytes_type>( send_buffer_.size() ) *
                                          static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/recv_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<memory_profiler_t::bytes_type>( recv_buffer_.size() ) *
                                          static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
    }

    void bind_external_work_area_()
    {
        if ( external_work_area_ == nullptr )
        {
            return;
        }
        char *raw = static_cast<char *>( external_work_area_ );
        send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), send_buffer_elems_ );
        recv_buffer_.init_by_raw_data(
            reinterpret_cast<value_type *>( raw + bytes_from_elems_( send_buffer_elems_ ) ), recv_buffer_elems_
        );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_forward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != nz_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw forward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != ny_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw forward output shape mismatch" );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_backward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != ny_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw backward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != nz_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw backward output shape mismatch" );
    }

    void reset_requests_()
    {
        std::fill( send_requests_.begin(), send_requests_.end(), mpi_request_t() );
        std::fill( recv_requests_.begin(), recv_requests_.end(), mpi_request_t() );
    }

    std::size_t bytes_from_elems_( std::size_t elems ) const
    {
        return elems * sizeof( value_type );
    }

    std::size_t forward_send_chunk_elems_( int target_j ) const
    {
        return nx_local_ * output_dim_.size_z[target_j] * nw_local_ * ny_local_;
    }

    std::size_t forward_recv_chunk_elems_( int source_j ) const
    {
        return nx_local_ * nz_local_ * nw_local_ * input_dim_.size_y[source_j];
    }

    std::size_t backward_send_chunk_elems_( int target_j ) const
    {
        return nx_local_ * nz_local_ * nw_local_ * input_dim_.size_y[target_j];
    }

    std::size_t backward_recv_chunk_elems_( int source_j ) const
    {
        return nx_local_ * output_dim_.size_z[source_j] * nw_local_ * ny_local_;
    }

    void init_layouts_()
    {
        const int comm_size = line_comm_info_.num_procs;

        forward_send_offsets_.resize( comm_size );
        forward_recv_offsets_.resize( comm_size );
        forward_sendcounts_.resize( comm_size );
        forward_sdispls_.resize( comm_size );
        forward_recvcounts_.resize( comm_size );
        forward_rdispls_.resize( comm_size );
        forward_sendcounts_w_.resize( comm_size );
        forward_sdispls_w_.resize( comm_size );
        forward_sendtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );
        forward_recvcounts_w_.resize( comm_size );
        forward_rdispls_w_.resize( comm_size );
        forward_recvtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );

        backward_send_offsets_.resize( comm_size );
        backward_recv_offsets_.resize( comm_size );
        backward_sendcounts_.resize( comm_size );
        backward_sdispls_.resize( comm_size );
        backward_recvcounts_.resize( comm_size );
        backward_rdispls_.resize( comm_size );
        backward_sendcounts_w_.resize( comm_size );
        backward_sdispls_w_.resize( comm_size );
        backward_sendtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );
        backward_recvcounts_w_.resize( comm_size );
        backward_rdispls_w_.resize( comm_size );
        backward_recvtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );

        std::size_t packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            forward_send_offsets_[p] = packed_offset;
            packed_offset += forward_send_chunk_elems_( p );
        }
        max_buffer_elems_ = packed_offset;

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            forward_recv_offsets_[p] = packed_offset;
            packed_offset += forward_recv_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            backward_send_offsets_[p] = packed_offset;
            packed_offset += backward_send_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            backward_recv_offsets_[p] = packed_offset;
            packed_offset += backward_recv_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        for ( int p = 0; p < comm_size; ++p )
        {
            forward_sendcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( forward_send_chunk_elems_( p ) ), "same_xw forward sendcount"
            );
            forward_sdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( forward_send_offsets_[p] ), "same_xw forward sdispl" );
            forward_recvcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( forward_recv_chunk_elems_( p ) ), "same_xw forward recvcount"
            );
            forward_rdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( forward_recv_offsets_[p] ), "same_xw forward rdispl" );
            forward_sendcounts_w_[p] = forward_sendcounts_[p];
            forward_sdispls_w_[p]    = forward_sdispls_[p];
            forward_recvcounts_w_[p] = forward_recvcounts_[p];
            forward_rdispls_w_[p]    = forward_rdispls_[p];

            backward_sendcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( backward_send_chunk_elems_( p ) ), "same_xw backward sendcount"
            );
            backward_sdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( backward_send_offsets_[p] ), "same_xw backward sdispl" );
            backward_recvcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( backward_recv_chunk_elems_( p ) ), "same_xw backward recvcount"
            );
            backward_rdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( backward_recv_offsets_[p] ), "same_xw backward rdispl" );
            backward_sendcounts_w_[p] = backward_sendcounts_[p];
            backward_sdispls_w_[p]    = backward_sdispls_[p];
            backward_recvcounts_w_[p] = backward_recvcounts_[p];
            backward_rdispls_w_[p]    = backward_rdispls_[p];
        }
    }

    template <class ArrayIn>
    void pack_forward_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_forward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            for_each_(
                pack_forward_functor<contiguous_buf_t, ArrayIn>{
                    send_buffer_, in, forward_send_offsets_[p], output_dim_.start_z[p], nx_local_,
                    output_dim_.size_z[p], nw_local_ },
                make_range_4d<idx_t>( nx_local_, output_dim_.size_z[p], nw_local_, ny_local_ )
            );
            for_each_.wait();
        }
    }

    template <class ArrayOut>
    void unpack_forward_chunk_( int source_j, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_forward_chunk" );
        for_each_(
            unpack_forward_functor<contiguous_buf_t, ArrayOut>{
                recv_buffer_, out, forward_recv_offsets_[source_j], input_dim_.start_y[source_j], nx_local_, nz_local_,
                nw_local_ },
            make_range_4d<idx_t>( nx_local_, nz_local_, nw_local_, input_dim_.size_y[source_j] )
        );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_backward_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_backward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            for_each_(
                pack_backward_functor<contiguous_buf_t, ArrayIn>{
                    send_buffer_, in, backward_send_offsets_[p], input_dim_.start_y[p], nx_local_, nz_local_,
                    nw_local_ },
                make_range_4d<idx_t>( nx_local_, nz_local_, nw_local_, input_dim_.size_y[p] )
            );
            for_each_.wait();
        }
    }

    template <class ArrayOut>
    void unpack_backward_chunk_( int source_j, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_backward_chunk" );
        for_each_(
            unpack_backward_functor<contiguous_buf_t, ArrayOut>{
                recv_buffer_, out, backward_recv_offsets_[source_j], output_dim_.start_z[source_j], nx_local_,
                output_dim_.size_z[source_j], nw_local_ },
            make_range_4d<idx_t>( nx_local_, output_dim_.size_z[source_j], nw_local_, ny_local_ )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void forward_p2p_waitall_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "forward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_j_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_j_, send_requests_[p]
            );
        }

        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_j_],
            send_buffer_.raw_ptr() + forward_send_offsets_[myid_j_],
            bytes_from_elems_( forward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
        );

        line_comm_info_.waitall( comm_size, recv_requests_.data() );
        for ( int p = 0; p < comm_size; ++p )
            unpack_forward_chunk_( p, out );
        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayOut>
    void forward_p2p_waitany_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "forward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_j_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_j_, send_requests_[p]
            );
        }

        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_j_],
            send_buffer_.raw_ptr() + forward_send_offsets_[myid_j_],
            bytes_from_elems_( forward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
        );
        unpack_forward_chunk_( myid_j_, out );

        int completed = 0;
        while ( completed < comm_size - 1 )
        {
            const int p = line_comm_info_.waitany( comm_size, recv_requests_.data() );
            if ( p == MPI_UNDEFINED )
                break;
            unpack_forward_chunk_( p, out );
            completed++;
        }

        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayOut>
    void forward_alltoallv_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "forward_alltoallv" );
        line_comm_info_.alltoallv(
            static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_.data(), forward_sdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type(), static_cast<void *>( recv_buffer_.raw_ptr() ),
            forward_recvcounts_.data(), forward_rdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type()
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            unpack_forward_chunk_( p, out );
#endif
    }

    template <class ArrayOut>
    void forward_alltoallw_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "forward_alltoallw" );
        line_comm_info_.alltoallw(
            static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_w_.data(),
            forward_sdispls_w_.data(), forward_sendtypes_w_.data(), static_cast<void *>( recv_buffer_.raw_ptr() ),
            forward_recvcounts_w_.data(), forward_rdispls_w_.data(), forward_recvtypes_w_.data()
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            unpack_forward_chunk_( p, out );
#endif
    }

    template <class ArrayOut>
    void backward_p2p_waitall_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "backward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_j_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_j_, recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, send_requests_[p]
            );
        }

        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_j_],
            send_buffer_.raw_ptr() + backward_send_offsets_[myid_j_],
            bytes_from_elems_( backward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
        );

        line_comm_info_.waitall( comm_size, recv_requests_.data() );
        for ( int p = 0; p < comm_size; ++p )
            unpack_backward_chunk_( p, out );
        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayOut>
    void backward_p2p_waitany_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "backward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_j_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_j_, recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, send_requests_[p]
            );
        }

        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_j_],
            send_buffer_.raw_ptr() + backward_send_offsets_[myid_j_],
            bytes_from_elems_( backward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
        );
        unpack_backward_chunk_( myid_j_, out );

        int completed = 0;
        while ( completed < comm_size - 1 )
        {
            const int p = line_comm_info_.waitany( comm_size, recv_requests_.data() );
            if ( p == MPI_UNDEFINED )
                break;
            unpack_backward_chunk_( p, out );
            completed++;
        }

        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayOut>
    void backward_alltoallv_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "backward_alltoallv" );
        line_comm_info_.alltoallv(
            static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_.data(), backward_sdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type(), static_cast<void *>( recv_buffer_.raw_ptr() ),
            backward_recvcounts_.data(), backward_rdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type()
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            unpack_backward_chunk_( p, out );
#endif
    }

    template <class ArrayOut>
    void backward_alltoallw_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_xw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "backward_alltoallw" );
        line_comm_info_.alltoallw(
            static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
            backward_sdispls_w_.data(), backward_sendtypes_w_.data(), static_cast<void *>( recv_buffer_.raw_ptr() ),
            backward_recvcounts_w_.data(), backward_rdispls_w_.data(), backward_recvtypes_w_.data()
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            unpack_backward_chunk_( p, out );
#endif
    }

private:
    MPIComm            mpi_;
    Log                log_;
    profiler_t        *profiler_        = nullptr;
    memory_profiler_t *memory_profiler_ = nullptr;
    std::string        memory_profile_prefix_;

    bool is_inited_ = false;
    int  myid_i_    = 0;
    int  myid_j_    = 0;
    int  myid_k_    = 0;

    std::size_t nx_local_  = 0;
    std::size_t ny_local_  = 0;
    std::size_t ny_global_ = 0;
    std::size_t nz_local_  = 0;
    std::size_t nz_global_ = 0;
    std::size_t nw_local_  = 0;

    partition_t input_dim_;
    partition_t output_dim_;

    std::unique_ptr<mpi_comm_t>        line_comm_;
    scfd::communication::mpi_comm_info line_comm_info_;
    contiguous_buf_t                   send_buffer_;
    contiguous_buf_t                   recv_buffer_;
    std::size_t                        send_buffer_elems_      = 0;
    std::size_t                        recv_buffer_elems_      = 0;
    bool                               use_external_work_area_ = false;
    void                              *external_work_area_     = nullptr;
    for_each_t                         for_each_;
    std::vector<mpi_request_t>         send_requests_;
    std::vector<mpi_request_t>         recv_requests_;

    std::vector<std::size_t> forward_send_offsets_;
    std::vector<std::size_t> forward_recv_offsets_;
    std::vector<int>         forward_sendcounts_;
    std::vector<int>         forward_sdispls_;
    std::vector<int>         forward_recvcounts_;
    std::vector<int>         forward_rdispls_;
    std::vector<int>         forward_sendcounts_w_;
    std::vector<int>         forward_sdispls_w_;
    std::vector<mpi_dtype_t> forward_sendtypes_w_;
    std::vector<int>         forward_recvcounts_w_;
    std::vector<int>         forward_rdispls_w_;
    std::vector<mpi_dtype_t> forward_recvtypes_w_;

    std::vector<std::size_t> backward_send_offsets_;
    std::vector<std::size_t> backward_recv_offsets_;
    std::vector<int>         backward_sendcounts_;
    std::vector<int>         backward_sdispls_;
    std::vector<int>         backward_recvcounts_;
    std::vector<int>         backward_rdispls_;
    std::vector<int>         backward_sendcounts_w_;
    std::vector<int>         backward_sdispls_w_;
    std::vector<mpi_dtype_t> backward_sendtypes_w_;
    std::vector<int>         backward_recvcounts_w_;
    std::vector<int>         backward_rdispls_w_;
    std::vector<mpi_dtype_t> backward_recvtypes_w_;

    std::size_t max_buffer_elems_ = 0;
};

template <
    class ValueType, class Backend, class MPIComm, class Log = scfd::utils::log_mpi,
    class RuntimeAPI = ::fftm::wrap::cuda_runtime_api>
class mpi_transpose_4d_same_zw
{
public:
    using value_type        = ValueType;
    using backend_t         = Backend;
    using memory_t          = typename backend_t::memory_type;
    using partition_t       = ::fftm::partition;
    using mpi_comm_t        = scfd::communication::mpi_comm;
    using contiguous_buf_t  = scfd::arrays::array_nd<value_type, 1, memory_t>;
    using mpi_request_t     = scfd::communication::detail::mpi_request;
    using mpi_dtype_t       = scfd::communication::detail::mpi_data_type<>;
    using idx_t             = scfd::static_vec::vec<int, 4>;
    using range_t           = rect_4d_t<idx_t>;
    using for_each_t        = typename backend_t::template for_each_nd_type<4, int>;
    using runtime_api_t     = RuntimeAPI;
    using profiler_t        = ::fftm::fftm_profiler;
    using profiler_scope_t  = ::fftm::profile_scope<profiler_t>;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

    mpi_transpose_4d_same_zw( const MPIComm &mpi, const Log &log = Log() ) : mpi_( mpi ), log_( log )
    {
        static_assert(
            std::is_same<memory_t, typename runtime_api_t::memory_type>::value,
            "mpi_transpose_4d_same_zw currently requires a CUDA backend memory "
            "type"
        );
        for_each_.block_size = 128;
    }

    void set_profiler( profiler_t *profiler )
    {
        profiler_ = profiler;
    }

    void set_memory_profiler( memory_profiler_t *profiler, const std::string &prefix )
    {
        memory_profiler_       = profiler;
        memory_profile_prefix_ = prefix;
        if ( is_inited_ )
        {
            update_memory_profile_();
        }
    }

    void use_external_work_area()
    {
        use_external_work_area_ = true;
    }

    std::size_t get_work_size_bytes() const
    {
        return bytes_from_elems_( send_buffer_elems_ ) + bytes_from_elems_( recv_buffer_elems_ );
    }

    void set_external_work_area( void *external_work_area )
    {
        if ( !use_external_work_area_ )
        {
            throw std::logic_error(
                "mpi_transpose_4d_same_zw::set_external_work_area: external work area was not enabled."
            );
        }
        external_work_area_ = external_work_area;
        bind_external_work_area_();
        update_memory_profile_();
    }

    void init( const partition_t &input_dim, const partition_t &output_dim, int myid_i, int myid_j, int myid_k )
    {
        auto scope  = profile_scope_( "mpi_transpose_4d_same_zw::init" );
        input_dim_  = input_dim;
        output_dim_ = output_dim;
        myid_i_     = myid_i;
        myid_j_     = myid_j;
        myid_k_     = myid_k;

        if ( input_dim_.size_y.size() != 1 )
            throw std::logic_error( "mpi_transpose_4d_same_zw expects the "
                                    "source layout to keep Y undistributed" );
        if ( output_dim_.size_x.size() != 1 )
            throw std::logic_error( "mpi_transpose_4d_same_zw expects the destination layout to "
                                    "keep X undistributed" );
        if ( input_dim_.size_z != output_dim_.size_z || input_dim_.size_w != output_dim_.size_w )
            throw std::logic_error( "mpi_transpose_4d_same_zw requires Z/W "
                                    "ownership to stay unchanged" );
        if ( input_dim_.size_x.size() != output_dim_.size_y.size() )
            throw std::logic_error( "mpi_transpose_4d_same_zw requires matching source-X and "
                                    "destination-Y partition counts" );

        nx_local_  = input_dim_.size_x.at( myid_i_ );
        nx_global_ = output_dim_.size_x.at( 0 );
        ny_local_  = output_dim_.size_y.at( myid_i_ );
        ny_global_ = input_dim_.size_y.at( 0 );
        nz_local_  = input_dim_.size_z.at( myid_j_ );
        nw_local_  = input_dim_.size_w.at( myid_k_ );

        const int color = myid_j_ * static_cast<int>( input_dim_.size_w.size() ) + myid_k_;
        line_comm_      = std::make_unique<mpi_comm_t>( std::move( mpi_.split( color, myid_i_ ) ) );
        line_comm_info_ = line_comm_->info();

        if ( line_comm_info_.num_procs != static_cast<int>( input_dim_.size_x.size() ) )
            throw std::logic_error( "mpi_transpose_4d_same_zw communicator size mismatch" );

        init_layouts_();
        send_buffer_elems_ = max_buffer_elems_;
        recv_buffer_elems_ = max_buffer_elems_;
        if ( !use_external_work_area_ )
        {
            send_buffer_.init( send_buffer_elems_ );
            recv_buffer_.init( recv_buffer_elems_ );
        }
        send_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        recv_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        is_inited_ = true;
        update_memory_profile_();
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xzwy_to_yzwx( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_zw::transpose_xzwy_to_yzwx/" ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_forward_shapes_( in, out );
        reset_requests_();
        pack_forward_all_( in );

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
    void transpose_yzwx_to_xzwy( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_zw::transpose_yzwx_to_xzwy/" ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_backward_shapes_( in, out );
        reset_requests_();
        pack_backward_all_( in );

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            backward_p2p_waitall_( out );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            backward_p2p_waitany_( out );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            backward_alltoallv_( out );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            backward_alltoallw_( out );
            break;
        }
    }

public:
    template <class Buffer, class ArrayIn>
    struct pack_forward_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t y_offset;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer( offset + packed_index_4d( idx, ny, nz, nw ) ) =
                in( idx[3], idx[1], idx[2], static_cast<int>( y_offset ) + idx[0] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_forward_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t x_offset;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[0], idx[1], idx[2], static_cast<int>( x_offset ) + idx[3] ) =
                buffer( offset + packed_index_4d( idx, ny, nz, nw ) );
        }
    };

    template <class Buffer, class ArrayIn>
    struct pack_backward_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t x_offset;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer( offset + packed_index_4d( idx, ny, nz, nw ) ) =
                in( idx[0], idx[1], idx[2], static_cast<int>( x_offset ) + idx[3] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_backward_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t y_offset;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[3], idx[1], idx[2], static_cast<int>( y_offset ) + idx[0] ) =
                buffer( offset + packed_index_4d( idx, ny, nz, nw ) );
        }
    };

private:
    profiler_scope_t profile_scope_( const std::string &name )
    {
        return profiler_scope_t( profiler_, name );
    }

    void ensure_is_inited_() const
    {
        if ( !is_inited_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw::init must be "
                                    "called before transpose" );
        if ( use_external_work_area_ && external_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_4d_same_zw: external work area was not bound." );
        }
    }

    void update_memory_profile_()
    {
        if ( memory_profiler_ == nullptr || memory_profile_prefix_.empty() )
        {
            return;
        }

        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/send_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<memory_profiler_t::bytes_type>( send_buffer_.size() ) *
                                          static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/recv_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<memory_profiler_t::bytes_type>( recv_buffer_.size() ) *
                                          static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
    }

    void bind_external_work_area_()
    {
        if ( external_work_area_ == nullptr )
        {
            return;
        }
        char *raw = static_cast<char *>( external_work_area_ );
        send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), send_buffer_elems_ );
        recv_buffer_.init_by_raw_data(
            reinterpret_cast<value_type *>( raw + bytes_from_elems_( send_buffer_elems_ ) ), recv_buffer_elems_
        );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_forward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != ny_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw forward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != ny_local_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != nx_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw forward output shape mismatch" );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_backward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != ny_local_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != nx_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw backward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != ny_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_zw backward output shape mismatch" );
    }

    void reset_requests_()
    {
        std::fill( send_requests_.begin(), send_requests_.end(), mpi_request_t() );
        std::fill( recv_requests_.begin(), recv_requests_.end(), mpi_request_t() );
    }

    std::size_t bytes_from_elems_( std::size_t elems ) const
    {
        return elems * sizeof( value_type );
    }

    std::size_t forward_send_chunk_elems_( int target_i ) const
    {
        return output_dim_.size_y[target_i] * nz_local_ * nw_local_ * nx_local_;
    }

    std::size_t forward_recv_chunk_elems_( int source_i ) const
    {
        return ny_local_ * nz_local_ * nw_local_ * input_dim_.size_x[source_i];
    }

    std::size_t backward_send_chunk_elems_( int target_i ) const
    {
        return ny_local_ * nz_local_ * nw_local_ * input_dim_.size_x[target_i];
    }

    std::size_t backward_recv_chunk_elems_( int source_i ) const
    {
        return output_dim_.size_y[source_i] * nz_local_ * nw_local_ * nx_local_;
    }

    void init_layouts_()
    {
        const int comm_size = line_comm_info_.num_procs;

        forward_send_offsets_.resize( comm_size );
        forward_recv_offsets_.resize( comm_size );
        forward_sendcounts_.resize( comm_size );
        forward_sdispls_.resize( comm_size );
        forward_recvcounts_.resize( comm_size );
        forward_rdispls_.resize( comm_size );
        forward_sendcounts_w_.resize( comm_size );
        forward_sdispls_w_.resize( comm_size );
        forward_sendtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );
        forward_recvcounts_w_.resize( comm_size );
        forward_rdispls_w_.resize( comm_size );
        forward_recvtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );

        backward_send_offsets_.resize( comm_size );
        backward_recv_offsets_.resize( comm_size );
        backward_sendcounts_.resize( comm_size );
        backward_sdispls_.resize( comm_size );
        backward_recvcounts_.resize( comm_size );
        backward_rdispls_.resize( comm_size );
        backward_sendcounts_w_.resize( comm_size );
        backward_sdispls_w_.resize( comm_size );
        backward_sendtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );
        backward_recvcounts_w_.resize( comm_size );
        backward_rdispls_w_.resize( comm_size );
        backward_recvtypes_w_.assign( comm_size, scfd::communication::detail::mpi_data_type<char>::mpi_type() );

        std::size_t packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            forward_send_offsets_[p] = packed_offset;
            packed_offset += forward_send_chunk_elems_( p );
        }
        max_buffer_elems_ = packed_offset;

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            forward_recv_offsets_[p] = packed_offset;
            packed_offset += forward_recv_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            backward_send_offsets_[p] = packed_offset;
            packed_offset += backward_send_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        packed_offset = 0;
        for ( int p = 0; p < comm_size; ++p )
        {
            backward_recv_offsets_[p] = packed_offset;
            packed_offset += backward_recv_chunk_elems_( p );
        }
        max_buffer_elems_ = std::max( max_buffer_elems_, packed_offset );

        for ( int p = 0; p < comm_size; ++p )
        {
            forward_sendcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( forward_send_chunk_elems_( p ) ), "same_zw forward sendcount"
            );
            forward_sdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( forward_send_offsets_[p] ), "same_zw forward sdispl" );
            forward_recvcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( forward_recv_chunk_elems_( p ) ), "same_zw forward recvcount"
            );
            forward_rdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( forward_recv_offsets_[p] ), "same_zw forward rdispl" );
            forward_sendcounts_w_[p] = forward_sendcounts_[p];
            forward_sdispls_w_[p]    = forward_sdispls_[p];
            forward_recvcounts_w_[p] = forward_recvcounts_[p];
            forward_rdispls_w_[p]    = forward_rdispls_[p];

            backward_sendcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( backward_send_chunk_elems_( p ) ), "same_zw backward sendcount"
            );
            backward_sdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( backward_send_offsets_[p] ), "same_zw backward sdispl" );
            backward_recvcounts_[p] = detail::mpi_int_cast(
                bytes_from_elems_( backward_recv_chunk_elems_( p ) ), "same_zw backward recvcount"
            );
            backward_rdispls_[p] =
                detail::mpi_int_cast( bytes_from_elems_( backward_recv_offsets_[p] ), "same_zw backward rdispl" );
            backward_sendcounts_w_[p] = backward_sendcounts_[p];
            backward_sdispls_w_[p]    = backward_sdispls_[p];
            backward_recvcounts_w_[p] = backward_recvcounts_[p];
            backward_rdispls_w_[p]    = backward_rdispls_[p];
        }
    }

    template <class ArrayIn>
    void pack_forward_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_forward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            for_each_(
                pack_forward_functor<contiguous_buf_t, ArrayIn>{
                    send_buffer_, in, forward_send_offsets_[p], output_dim_.start_y[p], output_dim_.size_y[p],
                    nz_local_, nw_local_ },
                make_range_4d<idx_t>( output_dim_.size_y[p], nz_local_, nw_local_, nx_local_ )
            );
            for_each_.wait();
        }
    }

    template <class ArrayOut>
    void unpack_forward_chunk_( int source_i, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_forward_chunk" );
        for_each_(
            unpack_forward_functor<contiguous_buf_t, ArrayOut>{
                recv_buffer_, out, forward_recv_offsets_[source_i], input_dim_.start_x[source_i], ny_local_, nz_local_,
                nw_local_ },
            make_range_4d<idx_t>( ny_local_, nz_local_, nw_local_, input_dim_.size_x[source_i] )
        );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_backward_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_backward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            for_each_(
                pack_backward_functor<contiguous_buf_t, ArrayIn>{
                    send_buffer_, in, backward_send_offsets_[p], input_dim_.start_x[p], ny_local_, nz_local_,
                    nw_local_ },
                make_range_4d<idx_t>( ny_local_, nz_local_, nw_local_, input_dim_.size_x[p] )
            );
            for_each_.wait();
        }
    }

    template <class ArrayOut>
    void unpack_backward_chunk_( int source_i, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_backward_chunk" );
        for_each_(
            unpack_backward_functor<contiguous_buf_t, ArrayOut>{
                recv_buffer_, out, backward_recv_offsets_[source_i], output_dim_.start_y[source_i],
                output_dim_.size_y[source_i], nz_local_, nw_local_ },
            make_range_4d<idx_t>( output_dim_.size_y[source_i], nz_local_, nw_local_, nx_local_ )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void forward_p2p_waitall_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_zw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "forward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_i_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_i_, send_requests_[p]
            );
        }

        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_i_],
            send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
            bytes_from_elems_( forward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
        );

        line_comm_info_.waitall( comm_size, recv_requests_.data() );
        for ( int p = 0; p < comm_size; ++p )
            unpack_forward_chunk_( p, out );
        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayOut>
    void forward_p2p_waitany_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_zw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "forward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_i_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_i_, send_requests_[p]
            );
        }

        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_i_],
            send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
            bytes_from_elems_( forward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
        );
        unpack_forward_chunk_( myid_i_, out );

        int completed = 0;
        while ( completed < comm_size - 1 )
        {
            const int p = line_comm_info_.waitany( comm_size, recv_requests_.data() );
            if ( p == MPI_UNDEFINED )
                break;
            unpack_forward_chunk_( p, out );
            completed++;
        }

        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayOut>
    void forward_alltoallv_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_zw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "forward_alltoallv" );
        line_comm_info_.alltoallv(
            static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_.data(), forward_sdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type(), static_cast<void *>( recv_buffer_.raw_ptr() ),
            forward_recvcounts_.data(), forward_rdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type()
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            unpack_forward_chunk_( p, out );
#endif
    }

    template <class ArrayOut>
    void forward_alltoallw_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_zw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "forward_alltoallw" );
        line_comm_info_.alltoallw(
            static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_w_.data(),
            forward_sdispls_w_.data(), forward_sendtypes_w_.data(), static_cast<void *>( recv_buffer_.raw_ptr() ),
            forward_recvcounts_w_.data(), forward_rdispls_w_.data(), forward_recvtypes_w_.data()
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            unpack_forward_chunk_( p, out );
#endif
    }

    template <class ArrayOut>
    void backward_p2p_waitall_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_zw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "backward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_i_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_i_, recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, send_requests_[p]
            );
        }

        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_i_],
            send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
            bytes_from_elems_( backward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
        );

        line_comm_info_.waitall( comm_size, recv_requests_.data() );
        for ( int p = 0; p < comm_size; ++p )
            unpack_backward_chunk_( p, out );
        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayOut>
    void backward_p2p_waitany_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_zw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto      scope     = profile_scope_( "backward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_i_ )
                continue;
            line_comm_info_.irecv(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, myid_i_, recv_requests_[p]
            );
            line_comm_info_.isend(
                send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                scfd::communication::detail::mpi_data_type<char>::mpi_type(), p, p, send_requests_[p]
            );
        }

        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_i_],
            send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
            bytes_from_elems_( backward_recv_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind()
        );
        unpack_backward_chunk_( myid_i_, out );

        int completed = 0;
        while ( completed < comm_size - 1 )
        {
            const int p = line_comm_info_.waitany( comm_size, recv_requests_.data() );
            if ( p == MPI_UNDEFINED )
                break;
            unpack_backward_chunk_( p, out );
            completed++;
        }

        line_comm_info_.waitall( comm_size, send_requests_.data() );
#endif
    }

    template <class ArrayOut>
    void backward_alltoallv_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_zw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "backward_alltoallv" );
        line_comm_info_.alltoallv(
            static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_.data(), backward_sdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type(), static_cast<void *>( recv_buffer_.raw_ptr() ),
            backward_recvcounts_.data(), backward_rdispls_.data(),
            scfd::communication::detail::mpi_data_type<char>::mpi_type()
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            unpack_backward_chunk_( p, out );
#endif
    }

    template <class ArrayOut>
    void backward_alltoallw_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "mpi_transpose_4d_same_zw currently requires "
                                "SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI" );
#else
        auto scope = profile_scope_( "backward_alltoallw" );
        line_comm_info_.alltoallw(
            static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
            backward_sdispls_w_.data(), backward_sendtypes_w_.data(), static_cast<void *>( recv_buffer_.raw_ptr() ),
            backward_recvcounts_w_.data(), backward_rdispls_w_.data(), backward_recvtypes_w_.data()
        );

        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            unpack_backward_chunk_( p, out );
#endif
    }

private:
    MPIComm            mpi_;
    Log                log_;
    profiler_t        *profiler_        = nullptr;
    memory_profiler_t *memory_profiler_ = nullptr;
    std::string        memory_profile_prefix_;

    bool is_inited_ = false;
    int  myid_i_    = 0;
    int  myid_j_    = 0;
    int  myid_k_    = 0;

    std::size_t nx_local_  = 0;
    std::size_t nx_global_ = 0;
    std::size_t ny_local_  = 0;
    std::size_t ny_global_ = 0;
    std::size_t nz_local_  = 0;
    std::size_t nw_local_  = 0;

    partition_t input_dim_;
    partition_t output_dim_;

    std::unique_ptr<mpi_comm_t>        line_comm_;
    scfd::communication::mpi_comm_info line_comm_info_;
    contiguous_buf_t                   send_buffer_;
    contiguous_buf_t                   recv_buffer_;
    std::size_t                        send_buffer_elems_      = 0;
    std::size_t                        recv_buffer_elems_      = 0;
    bool                               use_external_work_area_ = false;
    void                              *external_work_area_     = nullptr;
    for_each_t                         for_each_;
    std::vector<mpi_request_t>         send_requests_;
    std::vector<mpi_request_t>         recv_requests_;

    std::vector<std::size_t> forward_send_offsets_;
    std::vector<std::size_t> forward_recv_offsets_;
    std::vector<int>         forward_sendcounts_;
    std::vector<int>         forward_sdispls_;
    std::vector<int>         forward_recvcounts_;
    std::vector<int>         forward_rdispls_;
    std::vector<int>         forward_sendcounts_w_;
    std::vector<int>         forward_sdispls_w_;
    std::vector<mpi_dtype_t> forward_sendtypes_w_;
    std::vector<int>         forward_recvcounts_w_;
    std::vector<int>         forward_rdispls_w_;
    std::vector<mpi_dtype_t> forward_recvtypes_w_;

    std::vector<std::size_t> backward_send_offsets_;
    std::vector<std::size_t> backward_recv_offsets_;
    std::vector<int>         backward_sendcounts_;
    std::vector<int>         backward_sdispls_;
    std::vector<int>         backward_recvcounts_;
    std::vector<int>         backward_rdispls_;
    std::vector<int>         backward_sendcounts_w_;
    std::vector<int>         backward_sdispls_w_;
    std::vector<mpi_dtype_t> backward_sendtypes_w_;
    std::vector<int>         backward_recvcounts_w_;
    std::vector<int>         backward_rdispls_w_;
    std::vector<mpi_dtype_t> backward_recvtypes_w_;

    std::size_t max_buffer_elems_ = 0;
};

} // namespace detail
} // namespace fftm

#endif // __FFTM_DETAIL_MPI_TRANSPOSE_4D_H__
