#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_H__

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/utils/log_mpi.h>

#include "../fft_partitioning.h"
#include "../profiling.h"

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

template <
    class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI>
class mpi_transpose_3d
{
public:
    using value_type        = ValueType;
    using backend_t         = Backend;
    using memory_t          = typename backend_t::memory_type;
    using host_memory_t     = typename memory_t::host_memory_type;
    using partition_t       = ::fftm::partition;
    using mpi_comm_t        = scfd::communication::mpi_comm;
    using contiguous_buf_t  = scfd::arrays::array_nd<value_type, 1, memory_t>;
    using host_buf_t        = scfd::arrays::array_nd<value_type, 1, host_memory_t>;
    using mpi_request_t     = scfd::communication::detail::mpi_request;
    using mpi_dtype_t       = scfd::communication::detail::mpi_data_type<>;
    using runtime_api_t     = RuntimeAPI;
    using profiler_t        = ::fftm::fftm_profiler;
    using profiler_scope_t  = ::fftm::profile_scope<profiler_t>;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

    mpi_transpose_3d( const MPIComm &mpi, const Log &log = Log() ) : mpi_( mpi ), log_( log )
    {
        static_assert(
            std::is_same<memory_t, typename runtime_api_t::memory_type>::value,
            "mpi_transpose_3d currently requires a CUDA backend memory type"
        );
    }

    ~mpi_transpose_3d()
    {
        free_forward_types_();
        free_value_type_();
    }

    void set_profiler( profiler_t *profiler )
    {
        profiler_ = profiler;
    }

    void set_direct_transfer_options( bool use_direct_backward_receive, bool direct_p2p_cuda_aware )
    {
        use_direct_backward_receive_ = use_direct_backward_receive;
        direct_p2p_cuda_aware_       = direct_p2p_cuda_aware;
    }

    void set_p2p_send_thread_enabled( bool enabled )
    {
        use_p2p_send_thread_ = enabled;
    }

    void set_p2p_byte_transfer_enabled( bool enabled )
    {
        use_p2p_byte_transfer_ = enabled;
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
            throw std::logic_error( "mpi_transpose_3d::set_external_work_area: external work area was not enabled." );
        }
        external_work_area_ = external_work_area;
        bind_external_work_area_();
        update_memory_profile_();
    }

    void use_external_host_work_area()
    {
        use_external_host_work_area_ = true;
    }

    std::size_t get_host_work_size_bytes() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return 0;
#else
        return bytes_from_elems_( host_send_buffer_elems_ ) + bytes_from_elems_( host_recv_buffer_elems_ );
#endif
    }

    void set_external_host_work_area( void *external_host_work_area )
    {
        if ( !use_external_host_work_area_ )
        {
            throw std::logic_error(
                "mpi_transpose_3d::set_external_host_work_area: external host work area was not enabled."
            );
        }
        external_host_work_area_ = external_host_work_area;
        bind_external_host_work_area_();
        update_memory_profile_();
    }

    void init( const partition_t &input_dim, const partition_t &output_dim, int myid_i, int myid_j )
    {
        auto scope = profile_scope_( "mpi_transpose_3d::init" );
        free_forward_types_();
        free_value_type_();

        input_dim_  = input_dim;
        output_dim_ = output_dim;
        myid_i_     = myid_i;
        myid_j_     = myid_j;

        if ( input_dim_.size_z.size() != 1 )
        {
            throw std::logic_error( "mpi_transpose_3d expects the source layout "
                                    "to keep Z undistributed" );
        }
        if ( output_dim_.size_y.size() != 1 )
        {
            throw std::logic_error( "mpi_transpose_3d expects the destination "
                                    "layout to keep Y undistributed" );
        }
        if ( output_dim_.size_x.at( myid_i_ ) != input_dim_.size_x.at( myid_i_ ) )
        {
            throw std::logic_error( "mpi_transpose_3d requires X ownership to stay unchanged" );
        }
        if ( input_dim_.size_y.size() != output_dim_.size_z.size() )
        {
            throw std::logic_error( "mpi_transpose_3d requires matching source-Y and destination-Z "
                                    "partition counts" );
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
            throw std::logic_error( "mpi_transpose_3d row communicator size "
                                    "does not match Y partition count" );
        }

        const std::size_t max_send_elems =
            std::max( nx_local_ * ny_local_ * nz_global_, nx_local_ * ny_global_ * nz_local_ );
        const std::size_t max_recv_elems =
            std::max( nx_local_ * ny_global_ * nz_local_, nx_local_ * ny_local_ * nz_global_ );

        send_buffer_elems_ = max_send_elems;
        recv_buffer_elems_ = max_recv_elems;
        host_send_buffer_elems_ = send_buffer_elems_;
        host_recv_buffer_elems_ = recv_buffer_elems_;
        if ( !use_external_work_area_ )
        {
            send_buffer_.init( send_buffer_elems_ );
            recv_buffer_.init( recv_buffer_elems_ );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( !use_external_host_work_area_ )
        {
            host_send_buffer_.init( host_send_buffer_elems_ );
            host_recv_buffer_.init( host_recv_buffer_elems_ );
        }
#endif

        send_requests_.assign( row_comm_info_.num_procs, mpi_request_t() );
        recv_requests_.assign( row_comm_info_.num_procs, mpi_request_t() );

        streams_.clear();
        streams_.reserve( row_comm_info_.num_procs );
        for ( int p = 0; p < row_comm_info_.num_procs; ++p )
        {
            streams_.emplace_back( true );
        }

        init_value_type_();
        init_forward_layout_();
        init_backward_layout_();
        is_inited_ = true;
        update_memory_profile_();
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xyz_to_xzy( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_3d::transpose_xyz_to_xzy/" ) + mpi_transpose_3d_mode_name( mode )
        );
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
        auto scope = profile_scope_(
            std::string( "mpi_transpose_3d::transpose_xzy_to_xyz/" ) + mpi_transpose_3d_mode_name( mode )
        );
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
    profiler_scope_t profile_scope_( const std::string &name )
    {
        return profiler_scope_t( profiler_, name );
    }

    void ensure_is_inited_() const
    {
        if ( !is_inited_ )
        {
            throw std::logic_error( "mpi_transpose_3d::init must be called before transpose" );
        }
        if ( use_external_work_area_ && external_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_3d: external work area was not bound." );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( use_external_host_work_area_ && get_host_work_size_bytes() != 0 && external_host_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_3d: external host work area was not bound." );
        }
#endif
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
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_send_buffer",
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            0
#else
            use_external_host_work_area_ ? 0
                                         : static_cast<memory_profiler_t::bytes_type>( host_send_buffer_.size() ) *
                                               static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_recv_buffer",
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            0
#else
            use_external_host_work_area_ ? 0
                                         : static_cast<memory_profiler_t::bytes_type>( host_recv_buffer_.size() ) *
                                               static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
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

    void bind_external_host_work_area_()
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return;
#else
        if ( external_host_work_area_ == nullptr )
        {
            return;
        }
        char *raw = static_cast<char *>( external_host_work_area_ );
        host_send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), host_send_buffer_elems_ );
        host_recv_buffer_.init_by_raw_data(
            reinterpret_cast<value_type *>( raw + bytes_from_elems_( host_send_buffer_elems_ ) ),
            host_recv_buffer_elems_
        );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void verify_forward_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
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
        const auto in_size  = in.size_nd();
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

    struct send_ready_state_t
    {
        std::mutex              mutex;
        std::condition_variable cv;
        std::vector<int>        ready_peers;
        bool                    cancel = false;
    };

    struct send_ready_callback_t
    {
        send_ready_state_t *state = nullptr;
        int                 peer  = 0;
    };

    static void send_ready_callback_( void *data )
    {
        send_ready_callback_t *callback = static_cast<send_ready_callback_t *>( data );
        {
            std::lock_guard<std::mutex> lock( callback->state->mutex );
            callback->state->ready_peers.push_back( callback->peer );
        }
        callback->state->cv.notify_one();
    }

    void enqueue_send_ready_callback_( int peer, send_ready_state_t &state, std::vector<send_ready_callback_t> &callbacks )
    {
        callbacks[peer].state = &state;
        callbacks[peer].peer  = peer;
        runtime_api_t::launch_host_func( streams_[peer].stream(), &send_ready_callback_, &callbacks[peer] );
    }

    template <class StartSend>
    void wait_ready_and_post_sends_( int expected_sends, send_ready_state_t &state, StartSend start_send )
    {
        auto scope = profile_scope_( "wait_pack_ready_post_send" );
        for ( int posted = 0; posted < expected_sends; ++posted )
        {
            int peer = 0;
            {
                std::unique_lock<std::mutex> lock( state.mutex );
                state.cv.wait( lock, [&state] { return !state.ready_peers.empty(); } );
                peer = state.ready_peers.back();
                state.ready_peers.pop_back();
            }
            start_send( peer );
        }
    }

    bool p2p_send_thread_enabled_( int expected_sends ) const
    {
        return use_p2p_send_thread_ && expected_sends > 0 && mpi_.provided_threads >= MPI_THREAD_MULTIPLE;
    }

    template <class StartSend>
    std::thread start_ready_send_thread_(
        int expected_sends, send_ready_state_t &state, StartSend start_send, std::exception_ptr &send_exception
    )
    {
        return std::thread( [expected_sends, &state, start_send, &send_exception]() mutable {
            try
            {
                for ( int posted = 0; posted < expected_sends; ++posted )
                {
                    int peer = 0;
                    {
                        std::unique_lock<std::mutex> lock( state.mutex );
                        state.cv.wait( lock, [&state] { return state.cancel || !state.ready_peers.empty(); } );
                        if ( state.ready_peers.empty() )
                            break;
                        peer = state.ready_peers.back();
                        state.ready_peers.pop_back();
                    }
                    start_send( peer );
                }
            }
            catch ( ... )
            {
                send_exception = std::current_exception();
            }
        } );
    }

    void cancel_ready_send_thread_( send_ready_state_t &state, std::thread &send_thread )
    {
        if ( !send_thread.joinable() )
            return;
        {
            std::lock_guard<std::mutex> lock( state.mutex );
            state.cancel = true;
        }
        state.cv.notify_all();
        synchronize_streams_( "send_thread_cancel_stream_sync" );
        send_thread.join();
    }

    void join_ready_send_thread_( send_ready_state_t &state, std::thread &send_thread, std::exception_ptr &send_exception )
    {
        if ( !send_thread.joinable() )
            return;
        {
            auto phase = profile_scope_( "join_send_thread" );
            send_thread.join();
        }
        if ( send_exception )
        {
            std::rethrow_exception( send_exception );
        }
        (void)state;
    }

    std::size_t bytes_from_elems_( std::size_t elements ) const
    {
        return elements * sizeof( value_type );
    }

    std::size_t forward_local_input_elems_() const
    {
        return nx_local_ * ny_local_ * nz_global_;
    }

    template <class ArrayIn>
    void copy_forward_input_to_host_send_( const ArrayIn &in )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy(
            host_send_buffer_.raw_ptr(), in.raw_ptr(), bytes_from_elems_( forward_local_input_elems_() ),
            runtime_api_t::device_to_host_kind()
        );
#else
        (void)in;
#endif
    }

    void copy_send_buffer_to_host_()
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy(
            host_send_buffer_.raw_ptr(), send_buffer_.raw_ptr(), bytes_from_elems_( send_buffer_elems_ ),
            runtime_api_t::device_to_host_kind()
        );
#endif
    }

    void copy_host_recv_buffer_to_device_()
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr(), host_recv_buffer_.raw_ptr(), bytes_from_elems_( recv_buffer_elems_ ),
            runtime_api_t::host_to_device_kind()
        );
#endif
    }

    void copy_host_forward_chunk_to_device_async_( int source_j )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy_async(
            recv_buffer_.raw_ptr() + forward_recv_offset_elems_( source_j ),
            host_recv_buffer_.raw_ptr() + forward_recv_offset_elems_( source_j ),
            bytes_from_elems_( forward_recv_chunk_elems_( source_j ) ), runtime_api_t::host_to_device_kind(),
            streams_[source_j].stream()
        );
#endif
    }

    void copy_host_backward_chunk_to_device_async_( int source_j )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy_async(
            recv_buffer_.raw_ptr() + backward_recv_offset_elems_( source_j ),
            host_recv_buffer_.raw_ptr() + backward_recv_offset_elems_( source_j ),
            bytes_from_elems_( backward_recv_chunk_elems_( source_j ) ), runtime_api_t::host_to_device_kind(),
            streams_[source_j].stream()
        );
#endif
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
        forward_sendtypes_w_.assign( row_size, mpi_value_type_ );
        forward_recvcounts_w_.assign( row_size, 1 );
        forward_rdispls_w_.assign( row_size, 0 );
        forward_recvtypes_w_.assign( row_size, mpi_dtype_t() );

        for ( int p = 0; p < row_size; ++p )
        {
            forward_sendcounts_[p]   = detail::mpi_int_cast( forward_send_chunk_elems_( p ), "forward_sendcounts" );
            forward_sdispls_[p]      = detail::mpi_int_cast( forward_send_offset_elems_( p ), "forward_sdispls" );
            forward_recvcounts_[p]   = detail::mpi_int_cast( forward_recv_chunk_elems_( p ), "forward_recvcounts" );
            forward_rdispls_[p]      = detail::mpi_int_cast( forward_recv_offset_elems_( p ), "forward_rdispls" );
            forward_sendcounts_w_[p] = forward_sendcounts_[p];

            forward_recvtypes_w_[p] = scfd::communication::detail::type_vector(
                detail::mpi_int_cast( nx_local_ * nz_local_, "forward recv type count" ),
                detail::mpi_int_cast( input_dim_.size_y[p], "forward recv type blocklength" ),
                detail::mpi_int_cast( ny_global_, "forward recv type stride" ), mpi_value_type_
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
        backward_sendtypes_w_.assign( row_size, mpi_value_type_ );
        backward_recvcounts_w_.assign( row_size, 0 );
        backward_rdispls_w_.assign( row_size, 0 );
        backward_recvtypes_w_.assign( row_size, mpi_value_type_ );

        std::size_t packed_offset = 0;
        for ( int p = 0; p < row_size; ++p )
        {
            backward_send_offsets_[p] = packed_offset;
            packed_offset += backward_send_chunk_elems_( p );
        }

        for ( int p = 0; p < row_size; ++p )
        {
            backward_sendcounts_[p] = detail::mpi_int_cast( backward_send_chunk_elems_( p ), "backward_sendcounts" );
            backward_sdispls_[p]    = detail::mpi_int_cast( backward_send_pack_offset_elems_( p ), "backward_sdispls" );
            backward_recvcounts_[p] = detail::mpi_int_cast( backward_recv_chunk_elems_( p ), "backward_recvcounts" );
            backward_rdispls_[p]    = detail::mpi_int_cast( backward_recv_offset_elems_( p ), "backward_rdispls" );
            backward_sendcounts_w_[p] = backward_sendcounts_[p];
            backward_recvcounts_w_[p] = backward_recvcounts_[p];
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

    void init_value_type_()
    {
        mpi_value_type_ = scfd::communication::detail::type_contiguous(
            detail::mpi_int_cast( sizeof( value_type ), "mpi_transpose_3d value type extent" ),
            scfd::communication::detail::mpi_data_type<char>::mpi_type()
        );
        scfd::communication::detail::type_commit( mpi_value_type_ );
    }

    void free_value_type_()
    {
        scfd::communication::detail::type_free( mpi_value_type_ );
    }

    bool cuda_aware_byte_p2p_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return use_p2p_byte_transfer_;
#else
        return false;
#endif
    }

    bool cuda_aware_direct_p2p_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return direct_p2p_cuda_aware_ && use_direct_backward_receive_;
#else
        return false;
#endif
    }

    std::size_t max_mpi_byte_chunk_bytes_() const
    {
        return static_cast<std::size_t>( std::numeric_limits<int>::max() );
    }

    void post_byte_irecv_(
        void *ptr, std::size_t bytes, int source, int tag, std::vector<mpi_request_t> &requests,
        std::vector<int> *request_peers = nullptr, std::vector<int> *remaining_by_peer = nullptr
    )
    {
        char              *base      = static_cast<char *>( ptr );
        const mpi_dtype_t  byte_type = scfd::communication::detail::mpi_data_type<char>::mpi_type();
        const std::size_t  max_bytes = max_mpi_byte_chunk_bytes_();
        int                chunks    = 0;
        for ( std::size_t offset = 0; offset < bytes; offset += max_bytes )
        {
            const std::size_t this_bytes = std::min( max_bytes, bytes - offset );
            requests.push_back( mpi_request_t() );
            row_comm_info_.irecv(
                base + offset, detail::mpi_int_cast( this_bytes, "same_x byte p2p recv count" ), byte_type, source,
                tag, requests.back()
            );
            if ( request_peers != nullptr )
            {
                request_peers->push_back( source );
            }
            ++chunks;
        }
        if ( remaining_by_peer != nullptr )
        {
            ( *remaining_by_peer )[source] += chunks;
        }
    }

    void post_byte_isend_(
        const void *ptr, std::size_t bytes, int dest, int tag, std::vector<mpi_request_t> &requests
    )
    {
        const char        *base      = static_cast<const char *>( ptr );
        const mpi_dtype_t  byte_type = scfd::communication::detail::mpi_data_type<char>::mpi_type();
        const std::size_t  max_bytes = max_mpi_byte_chunk_bytes_();
        for ( std::size_t offset = 0; offset < bytes; offset += max_bytes )
        {
            const std::size_t this_bytes = std::min( max_bytes, bytes - offset );
            requests.push_back( mpi_request_t() );
            row_comm_info_.isend(
                base + offset, detail::mpi_int_cast( this_bytes, "same_x byte p2p send count" ), byte_type, dest, tag,
                requests.back()
            );
        }
    }

    void waitall_vector_( std::vector<mpi_request_t> &requests, const char *what )
    {
        if ( requests.empty() )
        {
            return;
        }
        row_comm_info_.waitall( detail::mpi_int_cast( requests.size(), what ), requests.data() );
    }

    void synchronize_streams_( const char *label = "stream_synchronize" )
    {
        auto scope = profile_scope_( label );
        for ( std::size_t i = 0; i < streams_.size(); ++i )
        {
            runtime_api_t::stream_synchronize( streams_[i].stream() );
        }
    }

    void copy_chunk_to_output_async_(
        const value_type *src_ptr, std::size_t src_y_size, std::size_t dst_y_offset, value_type *dst_ptr,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos                                     = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, src_y_size * sizeof( value_type ), src_y_size, nx_local_ );

        params.dstPos = runtime_api_t::make_pos( dst_y_offset * sizeof( value_type ), 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, ny_global_ * sizeof( value_type ), ny_global_, nx_local_ );

        params.extent = runtime_api_t::make_extent( src_y_size * sizeof( value_type ), nx_local_, nz_local_ );
        params.kind   = runtime_api_t::device_to_device_kind();

        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void pack_backward_chunk_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t packed_y_size, value_type *dst_ptr,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( src_y_offset * sizeof( value_type ), 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, ny_global_ * sizeof( value_type ), ny_global_, nx_local_ );

        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, packed_y_size * sizeof( value_type ), packed_y_size, nx_local_ );

        params.extent = runtime_api_t::make_extent( packed_y_size * sizeof( value_type ), nx_local_, nz_local_ );
        params.kind   = runtime_api_t::device_to_device_kind();

        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    template <class ArrayIn, class ArrayOut>
    void copy_forward_self_block_async_( const ArrayIn &in, ArrayOut &out )
    {
        copy_chunk_to_output_async_(
            in.raw_ptr() + forward_send_offset_elems_( myid_j_ ), ny_local_, input_dim_.start_y[myid_j_], out.raw_ptr(),
            streams_[myid_j_].stream()
        );
    }

    template <class ArrayOut>
    void unpack_forward_received_chunk_async_( int source_j, ArrayOut &out )
    {
        copy_chunk_to_output_async_(
            recv_buffer_.raw_ptr() + forward_recv_offset_elems_( source_j ), input_dim_.size_y[source_j],
            input_dim_.start_y[source_j], out.raw_ptr(), streams_[source_j].stream()
        );
    }

    template <class ArrayIn>
    void pack_backward_chunk_async_( int target_j, const ArrayIn &in )
    {
        pack_backward_chunk_async_(
            in.raw_ptr(), input_dim_.start_y[target_j], input_dim_.size_y[target_j],
            send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( target_j ), streams_[target_j].stream()
        );
    }

    template <class ArrayOut>
    void copy_backward_recv_chunk_to_output_async_( int source_j, ArrayOut &out )
    {
        runtime_api_t::memcpy_async(
            out.raw_ptr() + backward_recv_offset_elems_( source_j ),
            recv_buffer_.raw_ptr() + backward_recv_offset_elems_( source_j ),
            bytes_from_elems_( backward_recv_chunk_elems_( source_j ) ), runtime_api_t::device_to_device_kind(),
            streams_[source_j].stream()
        );
    }

    template <class ArrayIn, class ArrayOut>
    void forward_p2p_waitall_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto      scope    = profile_scope_( "forward_p2p_waitall" );
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        {
            auto phase = profile_scope_( "self_copy" );
            copy_forward_self_block_async_( in, out );
        }
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_forward_input_to_host_send_( in );
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;

                row_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + forward_recv_offset_elems_( p ),
                    detail::mpi_int_cast( forward_recv_chunk_elems_( p ), "forward p2p recv count" ), mpi_value_type_,
                    p, p, recv_requests_[p]
                );

                row_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + forward_send_offset_elems_( p ),
                    detail::mpi_int_cast( forward_send_chunk_elems_( p ), "forward p2p send count" ), mpi_value_type_,
                    p, myid_j_, send_requests_[p]
                );
            }
        }
        {
            auto phase = profile_scope_( "wait_recv" );
            row_comm_info_.waitall( row_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p != myid_j_ )
                {
                    copy_host_forward_chunk_to_device_async_( p );
                }
            }
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p != myid_j_ )
                    unpack_forward_received_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            row_comm_info_.waitall( row_size, send_requests_.data() );
        }
#else
        auto      scope    = profile_scope_( "forward_p2p_waitall" );
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        {
            auto phase = profile_scope_( "self_copy" );
            copy_forward_self_block_async_( in, out );
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;

                if ( cuda_aware_byte_p2p_enabled_() )
                {
                    post_byte_irecv_(
                        recv_buffer_.raw_ptr() + forward_recv_offset_elems_( p ),
                        bytes_from_elems_( forward_recv_chunk_elems_( p ) ), p, p, byte_recv_requests
                    );
                    post_byte_isend_(
                        in.raw_ptr() + forward_send_offset_elems_( p ),
                        bytes_from_elems_( forward_send_chunk_elems_( p ) ), p, myid_j_, byte_send_requests
                    );
                }
                else
                {
                    row_comm_info_.irecv(
                        recv_buffer_.raw_ptr() + forward_recv_offset_elems_( p ),
                        detail::mpi_int_cast( forward_recv_chunk_elems_( p ), "forward p2p recv count" ),
                        mpi_value_type_, p, p, recv_requests_[p]
                    );

                    row_comm_info_.isend(
                        in.raw_ptr() + forward_send_offset_elems_( p ),
                        detail::mpi_int_cast( forward_send_chunk_elems_( p ), "forward p2p send count" ),
                        mpi_value_type_, p, myid_j_, send_requests_[p]
                    );
                }
            }
        }
        {
            auto phase = profile_scope_( "wait_recv" );
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_recv_requests, "same_x byte forward recv request count" );
            else
                row_comm_info_.waitall( row_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p != myid_j_ )
                    unpack_forward_received_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_send_requests, "same_x byte forward send request count" );
            else
                row_comm_info_.waitall( row_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void forward_p2p_waitany_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto      scope    = profile_scope_( "forward_p2p_waitany" );
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( row_size, 0 );
        {
            auto phase = profile_scope_( "self_copy" );
            copy_forward_self_block_async_( in, out );
        }
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_forward_input_to_host_send_( in );
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;

                row_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + forward_recv_offset_elems_( p ),
                    detail::mpi_int_cast( forward_recv_chunk_elems_( p ), "forward p2p recv count" ), mpi_value_type_,
                    p, p, recv_requests_[p]
                );

                row_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + forward_send_offset_elems_( p ),
                    detail::mpi_int_cast( forward_send_chunk_elems_( p ), "forward p2p send count" ), mpi_value_type_,
                    p, myid_j_, send_requests_[p]
                );
            }
        }

        {
            int completed = 0;
            while ( completed < row_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = row_comm_info_.waitany( row_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                {
                    auto phase = profile_scope_( "stage_recv_chunk_to_device" );
                    copy_host_forward_chunk_to_device_async_( p );
                }
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_forward_received_chunk_async_( p, out );
                }
                completed++;
            }
        }

        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            row_comm_info_.waitall( row_size, send_requests_.data() );
        }
#else
        auto      scope    = profile_scope_( "forward_p2p_waitany" );
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( row_size, 0 );
        {
            auto phase = profile_scope_( "self_copy" );
            copy_forward_self_block_async_( in, out );
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;

                if ( cuda_aware_byte_p2p_enabled_() )
                {
                    post_byte_irecv_(
                        recv_buffer_.raw_ptr() + forward_recv_offset_elems_( p ),
                        bytes_from_elems_( forward_recv_chunk_elems_( p ) ), p, p, byte_recv_requests,
                        &byte_recv_peers, &byte_recv_remaining
                    );
                    post_byte_isend_(
                        in.raw_ptr() + forward_send_offset_elems_( p ),
                        bytes_from_elems_( forward_send_chunk_elems_( p ) ), p, myid_j_, byte_send_requests
                    );
                }
                else
                {
                    row_comm_info_.irecv(
                        recv_buffer_.raw_ptr() + forward_recv_offset_elems_( p ),
                        detail::mpi_int_cast( forward_recv_chunk_elems_( p ), "forward p2p recv count" ),
                        mpi_value_type_, p, p, recv_requests_[p]
                    );

                    row_comm_info_.isend(
                        in.raw_ptr() + forward_send_offset_elems_( p ),
                        detail::mpi_int_cast( forward_send_chunk_elems_( p ), "forward p2p send count" ),
                        mpi_value_type_, p, myid_j_, send_requests_[p]
                    );
                }
            }
        }

        {
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                int completed = 0;
                const int total_requests =
                    detail::mpi_int_cast( byte_recv_requests.size(), "same_x byte forward waitany request count" );
                while ( completed < total_requests )
                {
                    int request_index = MPI_UNDEFINED;
                    {
                        auto phase   = profile_scope_( "wait_recv_any" );
                        request_index = row_comm_info_.waitany( total_requests, byte_recv_requests.data() );
                    }
                    if ( request_index == MPI_UNDEFINED )
                        break;
                    const int p = byte_recv_peers[request_index];
                    --byte_recv_remaining[p];
                    if ( byte_recv_remaining[p] == 0 )
                    {
                        auto phase = profile_scope_( "unpack_recv_chunk" );
                        unpack_forward_received_chunk_async_( p, out );
                    }
                    completed++;
                }
            }
            else
            {
                int completed = 0;
                while ( completed < row_size - 1 )
                {
                    int p = MPI_UNDEFINED;
                    {
                        auto phase = profile_scope_( "wait_recv_any" );
                        p          = row_comm_info_.waitany( row_size, recv_requests_.data() );
                    }
                    if ( p == MPI_UNDEFINED )
                        break;
                    {
                        auto phase = profile_scope_( "unpack_recv_chunk" );
                        unpack_forward_received_chunk_async_( p, out );
                    }
                    completed++;
                }
            }
        }

        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_send_requests, "same_x byte forward send request count" );
            else
                row_comm_info_.waitall( row_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void forward_alltoallv_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto scope = profile_scope_( "forward_alltoallv" );
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_forward_input_to_host_send_( in );
        }
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            row_comm_info_.alltoallv(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                forward_sdispls_.data(), mpi_value_type_, static_cast<void *>( host_recv_buffer_.raw_ptr() ),
                forward_recvcounts_.data(), forward_rdispls_.data(), mpi_value_type_
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            copy_host_recv_buffer_to_device_();
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < row_comm_info_.num_procs; ++p )
            {
                unpack_forward_received_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#else
        auto scope = profile_scope_( "forward_alltoallv" );
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            row_comm_info_.alltoallv(
                static_cast<const void *>( in.raw_ptr() ), forward_sendcounts_.data(), forward_sdispls_.data(),
                mpi_value_type_, static_cast<void *>( recv_buffer_.raw_ptr() ), forward_recvcounts_.data(),
                forward_rdispls_.data(), mpi_value_type_
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < row_comm_info_.num_procs; ++p )
            {
                unpack_forward_received_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void forward_alltoallw_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto      scope    = profile_scope_( "forward_alltoallw" );
        const int row_size = row_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_forward_input_to_host_send_( in );
        }
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            for ( int p = 0; p < row_size; ++p )
            {
                forward_sdispls_w_[p] =
                    detail::mpi_int_cast( bytes_from_elems_( forward_send_offset_elems_( p ) ), "forward_sdispls_w" );
                forward_rdispls_w_[p] =
                    detail::mpi_int_cast( input_dim_.start_y[p] * sizeof( value_type ), "forward_rdispls_w" );
            }
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            row_comm_info_.alltoallw(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), forward_sendcounts_w_.data(),
                forward_sdispls_w_.data(), forward_sendtypes_w_.data(),
                static_cast<void *>( host_recv_buffer_.raw_ptr() ), forward_recvcounts_w_.data(),
                forward_rdispls_w_.data(), forward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            copy_host_recv_buffer_to_device_();
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < row_comm_info_.num_procs; ++p )
            {
                unpack_forward_received_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#else
        auto      scope    = profile_scope_( "forward_alltoallw" );
        const int row_size = row_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            for ( int p = 0; p < row_size; ++p )
            {
                forward_sdispls_w_[p] =
                    detail::mpi_int_cast( bytes_from_elems_( forward_send_offset_elems_( p ) ), "forward_sdispls_w" );
                forward_rdispls_w_[p] =
                    detail::mpi_int_cast( input_dim_.start_y[p] * sizeof( value_type ), "forward_rdispls_w" );
            }
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            row_comm_info_.alltoallw(
                static_cast<const void *>( in.raw_ptr() ), forward_sendcounts_w_.data(), forward_sdispls_w_.data(),
                forward_sendtypes_w_.data(), static_cast<void *>( out.raw_ptr() ), forward_recvcounts_w_.data(),
                forward_rdispls_w_.data(), forward_recvtypes_w_.data()
            );
        }
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_p2p_waitall_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto      scope    = profile_scope_( "backward_p2p_waitall" );
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;

        {
            auto phase = profile_scope_( "pack_and_post_recv" );
            for ( int p = 0; p < row_size; ++p )
            {
                pack_backward_chunk_async_( p, in );

                if ( p == myid_j_ )
                    continue;

                row_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + backward_recv_offset_elems_( p ),
                    detail::mpi_int_cast( backward_recv_chunk_elems_( p ), "backward p2p recv count" ), mpi_value_type_,
                    p, myid_j_, recv_requests_[p]
                );
            }
        }

        synchronize_streams_( "pack_complete" );
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }

        {
            auto phase = profile_scope_( "post_send" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;

                row_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( p ),
                    detail::mpi_int_cast( backward_send_chunk_elems_( p ), "backward p2p send count" ), mpi_value_type_,
                    p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy_async(
                out.raw_ptr() + backward_recv_offset_elems_( myid_j_ ),
                send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( myid_j_ ),
                bytes_from_elems_( backward_send_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_j_].stream()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            row_comm_info_.waitall( row_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p != myid_j_ )
                {
                    runtime_api_t::memcpy_async(
                        out.raw_ptr() + backward_recv_offset_elems_( p ),
                        host_recv_buffer_.raw_ptr() + backward_recv_offset_elems_( p ),
                        bytes_from_elems_( backward_recv_chunk_elems_( p ) ), runtime_api_t::host_to_device_kind(),
                        streams_[p].stream()
                    );
                }
            }
        }
        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            row_comm_info_.waitall( row_size, send_requests_.data() );
        }
#else
        auto      scope    = profile_scope_( "backward_p2p_waitall" );
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;

        {
            auto phase = profile_scope_( "post_recv" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;

                value_type *recv_ptr = cuda_aware_direct_p2p_enabled_()
                                          ? out.raw_ptr() + backward_recv_offset_elems_( p )
                                          : recv_buffer_.raw_ptr() + backward_recv_offset_elems_( p );
                if ( cuda_aware_byte_p2p_enabled_() )
                {
                    post_byte_irecv_(
                        recv_ptr, bytes_from_elems_( backward_recv_chunk_elems_( p ) ), p, myid_j_, byte_recv_requests
                    );
                }
                else
                {
                    row_comm_info_.irecv(
                        recv_ptr, detail::mpi_int_cast( backward_recv_chunk_elems_( p ), "backward p2p recv count" ),
                        mpi_value_type_, p, myid_j_, recv_requests_[p]
                    );
                }
            }
        }

        send_ready_state_t                 send_ready;
        std::vector<send_ready_callback_t> callbacks( row_size );
        send_ready.ready_peers.reserve( row_size );
        auto start_send = [this, &byte_send_requests]( int p ) {
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                post_byte_isend_(
                    send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( p ),
                    bytes_from_elems_( backward_send_chunk_elems_( p ) ), p, p, byte_send_requests
                );
            }
            else
            {
                row_comm_info_.isend(
                    send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( p ),
                    detail::mpi_int_cast( backward_send_chunk_elems_( p ), "backward p2p send count" ),
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        };
        const bool         use_send_thread = p2p_send_thread_enabled_( row_size - 1 );
        std::exception_ptr send_exception;
        std::thread        send_thread;
        try
        {
        {
            auto phase = profile_scope_( "pack_send_chunks" );
            for ( int p = 0; p < row_size; ++p )
            {
                pack_backward_chunk_async_( p, in );
                if ( p != myid_j_ )
                    enqueue_send_ready_callback_( p, send_ready, callbacks );
            }
        }
        if ( use_send_thread )
        {
            auto phase  = profile_scope_( "start_send_thread" );
            send_thread = start_ready_send_thread_( row_size - 1, send_ready, start_send, send_exception );
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::stream_synchronize( streams_[myid_j_].stream() );
            runtime_api_t::memcpy_async(
                out.raw_ptr() + backward_recv_offset_elems_( myid_j_ ),
                send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( myid_j_ ),
                bytes_from_elems_( backward_send_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_j_].stream()
            );
        }
        if ( !use_send_thread )
            wait_ready_and_post_sends_( row_size - 1, send_ready, start_send );

        {
            auto phase = profile_scope_( "wait_recv" );
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_recv_requests, "same_x byte backward recv request count" );
            else
                row_comm_info_.waitall( row_size, recv_requests_.data() );
        }
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( !cuda_aware_direct_p2p_enabled_() )
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < row_size; ++p )
                if ( p != myid_j_ )
                    copy_backward_recv_chunk_to_output_async_( p, out );
        }
#endif
        synchronize_streams_( "recv_copy_complete" );
        join_ready_send_thread_( send_ready, send_thread, send_exception );
        {
            auto phase = profile_scope_( "wait_send" );
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_send_requests, "same_x byte backward send request count" );
            else
                row_comm_info_.waitall( row_size, send_requests_.data() );
        }
        }
        catch ( ... )
        {
            cancel_ready_send_thread_( send_ready, send_thread );
            throw;
        }
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_p2p_waitany_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto      scope    = profile_scope_( "backward_p2p_waitany" );
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( row_size, 0 );

        {
            auto phase = profile_scope_( "pack_and_post_recv" );
            for ( int p = 0; p < row_size; ++p )
            {
                pack_backward_chunk_async_( p, in );

                if ( p == myid_j_ )
                    continue;

                row_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + backward_recv_offset_elems_( p ),
                    detail::mpi_int_cast( backward_recv_chunk_elems_( p ), "backward p2p recv count" ), mpi_value_type_,
                    p, myid_j_, recv_requests_[p]
                );
            }
        }

        synchronize_streams_( "pack_complete" );
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }

        {
            auto phase = profile_scope_( "post_send" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                row_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( p ),
                    detail::mpi_int_cast( backward_send_chunk_elems_( p ), "backward p2p send count" ), mpi_value_type_,
                    p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy_async(
                out.raw_ptr() + backward_recv_offset_elems_( myid_j_ ),
                send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( myid_j_ ),
                bytes_from_elems_( backward_send_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_j_].stream()
            );
        }

        {
            int completed = 0;
            while ( completed < row_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = row_comm_info_.waitany( row_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                {
                    auto phase = profile_scope_( "stage_recv_chunk_to_device" );
                    runtime_api_t::memcpy_async(
                        out.raw_ptr() + backward_recv_offset_elems_( p ),
                        host_recv_buffer_.raw_ptr() + backward_recv_offset_elems_( p ),
                        bytes_from_elems_( backward_recv_chunk_elems_( p ) ), runtime_api_t::host_to_device_kind(),
                        streams_[p].stream()
                    );
                }
                completed++;
            }
        }

        {
            auto phase = profile_scope_( "wait_send" );
            row_comm_info_.waitall( row_size, send_requests_.data() );
        }
        synchronize_streams_( "recv_copy_complete" );
#else
        auto      scope    = profile_scope_( "backward_p2p_waitany" );
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( row_size, 0 );

        {
            auto phase = profile_scope_( "post_recv" );
            for ( int p = 0; p < row_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;

                value_type *recv_ptr = cuda_aware_direct_p2p_enabled_()
                                          ? out.raw_ptr() + backward_recv_offset_elems_( p )
                                          : recv_buffer_.raw_ptr() + backward_recv_offset_elems_( p );
                if ( cuda_aware_byte_p2p_enabled_() )
                {
                    post_byte_irecv_(
                        recv_ptr, bytes_from_elems_( backward_recv_chunk_elems_( p ) ), p, myid_j_,
                        byte_recv_requests, &byte_recv_peers, &byte_recv_remaining
                    );
                }
                else
                {
                    row_comm_info_.irecv(
                        recv_ptr, detail::mpi_int_cast( backward_recv_chunk_elems_( p ), "backward p2p recv count" ),
                        mpi_value_type_, p, myid_j_, recv_requests_[p]
                    );
                }
            }
        }

        send_ready_state_t                 send_ready;
        std::vector<send_ready_callback_t> callbacks( row_size );
        send_ready.ready_peers.reserve( row_size );
        auto start_send = [this, &byte_send_requests]( int p ) {
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                post_byte_isend_(
                    send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( p ),
                    bytes_from_elems_( backward_send_chunk_elems_( p ) ), p, p, byte_send_requests
                );
            }
            else
            {
                row_comm_info_.isend(
                    send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( p ),
                    detail::mpi_int_cast( backward_send_chunk_elems_( p ), "backward p2p send count" ),
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        };
        const bool         use_send_thread = p2p_send_thread_enabled_( row_size - 1 );
        std::exception_ptr send_exception;
        std::thread        send_thread;
        try
        {
        {
            auto phase = profile_scope_( "pack_send_chunks" );
            for ( int p = 0; p < row_size; ++p )
            {
                pack_backward_chunk_async_( p, in );
                if ( p != myid_j_ )
                    enqueue_send_ready_callback_( p, send_ready, callbacks );
            }
        }
        if ( use_send_thread )
        {
            auto phase  = profile_scope_( "start_send_thread" );
            send_thread = start_ready_send_thread_( row_size - 1, send_ready, start_send, send_exception );
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::stream_synchronize( streams_[myid_j_].stream() );
            runtime_api_t::memcpy_async(
                out.raw_ptr() + backward_recv_offset_elems_( myid_j_ ),
                send_buffer_.raw_ptr() + backward_send_pack_offset_elems_( myid_j_ ),
                bytes_from_elems_( backward_send_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_j_].stream()
            );
        }
        if ( !use_send_thread )
            wait_ready_and_post_sends_( row_size - 1, send_ready, start_send );

        {
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                int completed = 0;
                const int total_requests =
                    detail::mpi_int_cast( byte_recv_requests.size(), "same_x byte backward waitany request count" );
                while ( completed < total_requests )
                {
                    int request_index = MPI_UNDEFINED;
                    {
                        auto phase   = profile_scope_( "wait_recv_any" );
                        request_index = row_comm_info_.waitany( total_requests, byte_recv_requests.data() );
                    }
                    if ( request_index == MPI_UNDEFINED )
                        break;
                    const int p = byte_recv_peers[request_index];
                    --byte_recv_remaining[p];
                    if ( byte_recv_remaining[p] == 0 && !cuda_aware_direct_p2p_enabled_() )
                    {
                        auto phase = profile_scope_( "stage_recv_chunk_to_device" );
                        copy_backward_recv_chunk_to_output_async_( p, out );
                    }
                    completed++;
                }
            }
            else
            {
                int completed = 0;
                while ( completed < row_size - 1 )
                {
                    int p = MPI_UNDEFINED;
                    {
                        auto phase = profile_scope_( "wait_recv_any" );
                        p          = row_comm_info_.waitany( row_size, recv_requests_.data() );
                    }
                    if ( p == MPI_UNDEFINED )
                        break;
                    if ( !cuda_aware_direct_p2p_enabled_() )
                    {
                        auto phase = profile_scope_( "stage_recv_chunk_to_device" );
                        copy_backward_recv_chunk_to_output_async_( p, out );
                    }
                    completed++;
                }
            }
        }

        synchronize_streams_( "recv_copy_complete" );
        join_ready_send_thread_( send_ready, send_thread, send_exception );
        {
            auto phase = profile_scope_( "wait_send" );
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_send_requests, "same_x byte backward send request count" );
            else
                row_comm_info_.waitall( row_size, send_requests_.data() );
        }
        }
        catch ( ... )
        {
            cancel_ready_send_thread_( send_ready, send_thread );
            throw;
        }
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_alltoallv_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        {
            auto scope = profile_scope_( "backward_alltoallv" );
            {
                auto phase = profile_scope_( "pack_backward" );
                for ( int p = 0; p < row_comm_info_.num_procs; ++p )
                {
                    pack_backward_chunk_async_( p, in );
                }
            }
            synchronize_streams_( "pack_complete" );

            {
                auto phase = profile_scope_( "stage_send_to_host" );
                copy_send_buffer_to_host_();
            }

            {
                auto phase = profile_scope_( "mpi_alltoallv" );
                row_comm_info_.alltoallv(
                    static_cast<const void *>( host_send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                    backward_sdispls_.data(), mpi_value_type_, static_cast<void *>( host_recv_buffer_.raw_ptr() ),
                    backward_recvcounts_.data(), backward_rdispls_.data(), mpi_value_type_
                );
            }
            {
                auto phase = profile_scope_( "stage_recv_to_device" );
                runtime_api_t::memcpy(
                    out.raw_ptr(), host_recv_buffer_.raw_ptr(), bytes_from_elems_( nx_local_ * ny_local_ * nz_global_ ),
                    runtime_api_t::host_to_device_kind()
                );
            }
        }
#else
        {
            auto scope = profile_scope_( "backward_alltoallv" );
            {
                auto phase = profile_scope_( "pack_backward" );
                for ( int p = 0; p < row_comm_info_.num_procs; ++p )
                {
                    pack_backward_chunk_async_( p, in );
                }
            }
            synchronize_streams_( "pack_complete" );

            {
                auto phase = profile_scope_( "mpi_alltoallv" );
                row_comm_info_.alltoallv(
                    static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                    backward_sdispls_.data(), mpi_value_type_, static_cast<void *>( out.raw_ptr() ),
                    backward_recvcounts_.data(), backward_rdispls_.data(), mpi_value_type_
                );
            }
        }
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_alltoallw_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        {
            auto scope = profile_scope_( "backward_alltoallw" );
            {
                auto phase = profile_scope_( "pack_backward" );
                for ( int p = 0; p < row_comm_info_.num_procs; ++p )
                {
                    pack_backward_chunk_async_( p, in );
                }
            }
            synchronize_streams_( "pack_complete" );

            {
                auto phase = profile_scope_( "stage_send_to_host" );
                copy_send_buffer_to_host_();
            }

            const int row_size = row_comm_info_.num_procs;
            {
                auto phase = profile_scope_( "prepare_alltoallw_layout" );
                for ( int p = 0; p < row_size; ++p )
                {
                    backward_sdispls_w_[p] = detail::mpi_int_cast(
                        bytes_from_elems_( backward_send_pack_offset_elems_( p ) ), "backward_sdispls_w"
                    );
                    backward_rdispls_w_[p] = detail::mpi_int_cast(
                        bytes_from_elems_( backward_recv_offset_elems_( p ) ), "backward_rdispls_w"
                    );
                }
            }

            {
                auto phase = profile_scope_( "mpi_alltoallw" );
                row_comm_info_.alltoallw(
                    static_cast<const void *>( host_send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
                    backward_sdispls_w_.data(), backward_sendtypes_w_.data(),
                    static_cast<void *>( host_recv_buffer_.raw_ptr() ), backward_recvcounts_w_.data(),
                    backward_rdispls_w_.data(), backward_recvtypes_w_.data()
                );
            }
            {
                auto phase = profile_scope_( "stage_recv_to_device" );
                runtime_api_t::memcpy(
                    out.raw_ptr(), host_recv_buffer_.raw_ptr(), bytes_from_elems_( nx_local_ * ny_local_ * nz_global_ ),
                    runtime_api_t::host_to_device_kind()
                );
            }
        }
#else
        {
            auto scope = profile_scope_( "backward_alltoallw" );
            {
                auto phase = profile_scope_( "pack_backward" );
                for ( int p = 0; p < row_comm_info_.num_procs; ++p )
                {
                    pack_backward_chunk_async_( p, in );
                }
            }
            synchronize_streams_( "pack_complete" );

            const int row_size = row_comm_info_.num_procs;
            {
                auto phase = profile_scope_( "prepare_alltoallw_layout" );
                for ( int p = 0; p < row_size; ++p )
                {
                    backward_sdispls_w_[p] = detail::mpi_int_cast(
                        bytes_from_elems_( backward_send_pack_offset_elems_( p ) ), "backward_sdispls_w"
                    );
                    backward_rdispls_w_[p] = detail::mpi_int_cast(
                        bytes_from_elems_( backward_recv_offset_elems_( p ) ), "backward_rdispls_w"
                    );
                }
            }

            {
                auto phase = profile_scope_( "mpi_alltoallw" );
                row_comm_info_.alltoallw(
                    static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
                    backward_sdispls_w_.data(), backward_sendtypes_w_.data(), static_cast<void *>( out.raw_ptr() ),
                    backward_recvcounts_w_.data(), backward_rdispls_w_.data(), backward_recvtypes_w_.data()
                );
            }
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

    std::size_t nx_local_  = 0;
    std::size_t ny_local_  = 0;
    std::size_t ny_global_ = 0;
    std::size_t nz_global_ = 0;
    std::size_t nz_local_  = 0;

    partition_t input_dim_;
    partition_t output_dim_;

    std::unique_ptr<mpi_comm_t>                      row_comm_;
    scfd::communication::mpi_comm_info               row_comm_info_;
    contiguous_buf_t                                 send_buffer_;
    contiguous_buf_t                                 recv_buffer_;
    std::size_t                                      send_buffer_elems_      = 0;
    std::size_t                                      recv_buffer_elems_      = 0;
    host_buf_t                                       host_send_buffer_;
    host_buf_t                                       host_recv_buffer_;
    std::size_t                                      host_send_buffer_elems_ = 0;
    std::size_t                                      host_recv_buffer_elems_ = 0;
    bool                                             use_external_work_area_ = false;
    void                                            *external_work_area_     = nullptr;
    bool                                             use_external_host_work_area_ = false;
    void                                            *external_host_work_area_     = nullptr;
    bool                                             use_direct_backward_receive_ = false;
    bool                                             direct_p2p_cuda_aware_       = true;
    bool                                             use_p2p_send_thread_         = false;
    bool                                             use_p2p_byte_transfer_       = false;
    std::vector<mpi_request_t>                       send_requests_;
    std::vector<mpi_request_t>                       recv_requests_;
    std::vector<typename runtime_api_t::stream_wrap> streams_;
    mpi_dtype_t                                      mpi_value_type_;

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
};

template <
    class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI>
class mpi_transpose_3d_same_z
{
public:
    using value_type        = ValueType;
    using backend_t         = Backend;
    using memory_t          = typename backend_t::memory_type;
    using host_memory_t     = typename memory_t::host_memory_type;
    using partition_t       = ::fftm::partition;
    using mpi_comm_t        = scfd::communication::mpi_comm;
    using contiguous_buf_t  = scfd::arrays::array_nd<value_type, 1, memory_t>;
    using host_buf_t        = scfd::arrays::array_nd<value_type, 1, host_memory_t>;
    using mpi_request_t     = scfd::communication::detail::mpi_request;
    using mpi_dtype_t       = scfd::communication::detail::mpi_data_type<>;
    using runtime_api_t     = RuntimeAPI;
    using profiler_t        = ::fftm::fftm_profiler;
    using profiler_scope_t  = ::fftm::profile_scope<profiler_t>;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

    mpi_transpose_3d_same_z( const MPIComm &mpi, const Log &log = Log() ) : mpi_( mpi ), log_( log )
    {
        static_assert(
            std::is_same<memory_t, typename runtime_api_t::memory_type>::value,
            "mpi_transpose_3d_same_z currently requires a CUDA backend memory "
            "type"
        );
    }

    ~mpi_transpose_3d_same_z()
    {
        free_direct_backward_recv_types_();
        free_value_type_();
    }

    void set_profiler( profiler_t *profiler )
    {
        profiler_ = profiler;
    }

    void set_direct_transfer_options( bool use_direct_backward_receive, bool direct_p2p_cuda_aware )
    {
        use_direct_backward_receive_ = use_direct_backward_receive;
        direct_p2p_cuda_aware_       = direct_p2p_cuda_aware;
    }

    void set_p2p_send_thread_enabled( bool enabled )
    {
        use_p2p_send_thread_ = enabled;
    }

    void set_p2p_byte_transfer_enabled( bool enabled )
    {
        use_p2p_byte_transfer_ = enabled;
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
                "mpi_transpose_3d_same_z::set_external_work_area: external work area was not enabled."
            );
        }
        external_work_area_ = external_work_area;
        bind_external_work_area_();
        update_memory_profile_();
    }

    void use_external_host_work_area()
    {
        use_external_host_work_area_ = true;
    }

    std::size_t get_host_work_size_bytes() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return 0;
#else
        return bytes_from_elems_( host_send_buffer_elems_ ) + bytes_from_elems_( host_recv_buffer_elems_ );
#endif
    }

    void set_external_host_work_area( void *external_host_work_area )
    {
        if ( !use_external_host_work_area_ )
        {
            throw std::logic_error(
                "mpi_transpose_3d_same_z::set_external_host_work_area: external host work area was not enabled."
            );
        }
        external_host_work_area_ = external_host_work_area;
        bind_external_host_work_area_();
        update_memory_profile_();
    }

    void init( const partition_t &input_dim, const partition_t &output_dim, int myid_i, int myid_j )
    {
        auto scope = profile_scope_( "mpi_transpose_3d_same_z::init" );
        free_direct_backward_recv_types_();
        free_value_type_();

        input_dim_  = input_dim;
        output_dim_ = output_dim;
        myid_i_     = myid_i;
        myid_j_     = myid_j;

        if ( input_dim_.size_y.size() != 1 )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z expects the source "
                                    "layout to keep Y undistributed" );
        }
        if ( output_dim_.size_x.size() != 1 )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z expects the destination layout to "
                                    "keep X undistributed" );
        }
        if ( input_dim_.size_x.size() != output_dim_.size_y.size() )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z requires matching source-X and "
                                    "destination-Y partition counts" );
        }
        if ( input_dim_.size_z != output_dim_.size_z )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z requires Z "
                                    "ownership to stay unchanged" );
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
            throw std::logic_error( "mpi_transpose_3d_same_z communicator size "
                                    "does not match X partition count" );
        }

        const std::size_t total_send_elems =
            std::max( nx_local_ * ny_global_ * nz_local_, nx_global_ * ny_local_ * nz_local_ );
        const std::size_t total_recv_elems =
            std::max( nx_local_ * ny_global_ * nz_local_, nx_global_ * ny_local_ * nz_local_ );

        send_buffer_elems_ = total_send_elems;
        recv_buffer_elems_ = total_recv_elems;
        host_send_buffer_elems_ = send_buffer_elems_;
        host_recv_buffer_elems_ = recv_buffer_elems_;
        if ( !use_external_work_area_ )
        {
            send_buffer_.init( send_buffer_elems_ );
            recv_buffer_.init( recv_buffer_elems_ );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( !use_external_host_work_area_ )
        {
            host_send_buffer_.init( host_send_buffer_elems_ );
            host_recv_buffer_.init( host_recv_buffer_elems_ );
        }
#endif

        send_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        recv_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );

        streams_.clear();
        streams_.reserve( line_comm_info_.num_procs );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            streams_.emplace_back( true );
        }

        init_value_type_();
        init_forward_layout_();
        init_backward_layout_();
        init_direct_backward_recv_types_();
        is_inited_ = true;
        update_memory_profile_();
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_x_to_y( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_3d_same_z::transpose_x_to_y/" ) + mpi_transpose_3d_mode_name( mode )
        );
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
        auto scope = profile_scope_(
            std::string( "mpi_transpose_3d_same_z::transpose_y_to_x/" ) + mpi_transpose_3d_mode_name( mode )
        );
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

    template <class ArrayIn, class ArrayOut>
    void transpose_x_to_y_optimized_layout( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_3d_same_z::transpose_x_to_y_optimized/" ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_forward_shapes_( in, out );
        reset_requests_();

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            forward_optimized_p2p_waitall_( in, out );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            forward_optimized_p2p_waitany_( in, out );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            pack_forward_optimized_layout_( in );
            forward_optimized_alltoallv_( out );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            pack_forward_optimized_layout_( in );
            forward_optimized_alltoallw_( out );
            break;
        }
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_y_to_x_optimized_layout( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_3d_same_z::transpose_y_to_x_optimized/" ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_backward_shapes_( in, out );
        reset_requests_();

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            backward_optimized_p2p_waitall_( in, out );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            backward_optimized_p2p_waitany_( in, out );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            backward_optimized_alltoallv_( in, out );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            backward_optimized_alltoallw_( in, out );
            break;
        }
    }

private:
    profiler_scope_t profile_scope_( const std::string &name )
    {
        return profiler_scope_t( profiler_, name );
    }

    void ensure_is_inited_() const
    {
        if ( !is_inited_ )
            throw std::logic_error( "mpi_transpose_3d_same_z::init must be "
                                    "called before transpose" );
        if ( use_external_work_area_ && external_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z: external work area was not bound." );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( use_external_host_work_area_ && get_host_work_size_bytes() != 0 && external_host_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_3d_same_z: external host work area was not bound." );
        }
#endif
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
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_send_buffer",
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            0
#else
            use_external_host_work_area_ ? 0
                                         : static_cast<memory_profiler_t::bytes_type>( host_send_buffer_.size() ) *
                                               static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_recv_buffer",
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            0
#else
            use_external_host_work_area_ ? 0
                                         : static_cast<memory_profiler_t::bytes_type>( host_recv_buffer_.size() ) *
                                               static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
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

    void bind_external_host_work_area_()
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return;
#else
        if ( external_host_work_area_ == nullptr )
        {
            return;
        }
        char *raw = static_cast<char *>( external_host_work_area_ );
        host_send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), host_send_buffer_elems_ );
        host_recv_buffer_.init_by_raw_data(
            reinterpret_cast<value_type *>( raw + bytes_from_elems_( host_send_buffer_elems_ ) ),
            host_recv_buffer_elems_
        );
#endif
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
        std::fill( send_requests_.begin(), send_requests_.end(), mpi_request_t() );
        std::fill( recv_requests_.begin(), recv_requests_.end(), mpi_request_t() );
    }

    struct send_ready_state_t
    {
        std::mutex              mutex;
        std::condition_variable cv;
        std::vector<int>        ready_peers;
        bool                    cancel = false;
    };

    struct send_ready_callback_t
    {
        send_ready_state_t *state = nullptr;
        int                 peer  = 0;
    };

    static void send_ready_callback_( void *data )
    {
        send_ready_callback_t *callback = static_cast<send_ready_callback_t *>( data );
        {
            std::lock_guard<std::mutex> lock( callback->state->mutex );
            callback->state->ready_peers.push_back( callback->peer );
        }
        callback->state->cv.notify_one();
    }

    void enqueue_send_ready_callback_( int peer, send_ready_state_t &state, std::vector<send_ready_callback_t> &callbacks )
    {
        callbacks[peer].state = &state;
        callbacks[peer].peer  = peer;
        runtime_api_t::launch_host_func( streams_[peer].stream(), &send_ready_callback_, &callbacks[peer] );
    }

    template <class StartSend>
    void wait_ready_and_post_sends_( int expected_sends, send_ready_state_t &state, StartSend start_send )
    {
        auto scope = profile_scope_( "wait_pack_ready_post_send" );
        for ( int posted = 0; posted < expected_sends; ++posted )
        {
            int peer = 0;
            {
                std::unique_lock<std::mutex> lock( state.mutex );
                state.cv.wait( lock, [&state] { return !state.ready_peers.empty(); } );
                peer = state.ready_peers.back();
                state.ready_peers.pop_back();
            }
            start_send( peer );
        }
    }

    bool p2p_send_thread_enabled_( int expected_sends ) const
    {
        return use_p2p_send_thread_ && expected_sends > 0 && mpi_.provided_threads >= MPI_THREAD_MULTIPLE;
    }

    template <class StartSend>
    std::thread start_ready_send_thread_(
        int expected_sends, send_ready_state_t &state, StartSend start_send, std::exception_ptr &send_exception
    )
    {
        return std::thread( [expected_sends, &state, start_send, &send_exception]() mutable {
            try
            {
                for ( int posted = 0; posted < expected_sends; ++posted )
                {
                    int peer = 0;
                    {
                        std::unique_lock<std::mutex> lock( state.mutex );
                        state.cv.wait( lock, [&state] { return state.cancel || !state.ready_peers.empty(); } );
                        if ( state.ready_peers.empty() )
                            break;
                        peer = state.ready_peers.back();
                        state.ready_peers.pop_back();
                    }
                    start_send( peer );
                }
            }
            catch ( ... )
            {
                send_exception = std::current_exception();
            }
        } );
    }

    void cancel_ready_send_thread_( send_ready_state_t &state, std::thread &send_thread )
    {
        if ( !send_thread.joinable() )
            return;
        {
            std::lock_guard<std::mutex> lock( state.mutex );
            state.cancel = true;
        }
        state.cv.notify_all();
        synchronize_streams_( "send_thread_cancel_stream_sync" );
        send_thread.join();
    }

    void join_ready_send_thread_( send_ready_state_t &state, std::thread &send_thread, std::exception_ptr &send_exception )
    {
        if ( !send_thread.joinable() )
            return;
        {
            auto phase = profile_scope_( "join_send_thread" );
            send_thread.join();
        }
        if ( send_exception )
        {
            std::rethrow_exception( send_exception );
        }
        (void)state;
    }

    std::size_t bytes_from_elems_( std::size_t elems ) const
    {
        return elems * sizeof( value_type );
    }

    void copy_send_buffer_to_host_()
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy(
            host_send_buffer_.raw_ptr(), send_buffer_.raw_ptr(), bytes_from_elems_( send_buffer_elems_ ),
            runtime_api_t::device_to_host_kind()
        );
#endif
    }

    void copy_host_recv_buffer_to_device_()
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr(), host_recv_buffer_.raw_ptr(), bytes_from_elems_( recv_buffer_elems_ ),
            runtime_api_t::host_to_device_kind()
        );
#endif
    }

    void copy_host_forward_chunk_to_device_async_( int source_i )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy_async(
            recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( source_i ),
            host_recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( source_i ),
            bytes_from_elems_( input_dim_.size_x[source_i] * ny_local_ * nz_local_ ),
            runtime_api_t::host_to_device_kind(), streams_[source_i].stream()
        );
#endif
    }

    void copy_host_backward_chunk_to_device_async_( int source_i )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy_async(
            recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( source_i ),
            host_recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( source_i ),
            bytes_from_elems_( backward_chunk_elems_( source_i ) ), runtime_api_t::host_to_device_kind(),
            streams_[source_i].stream()
        );
#endif
    }

    std::size_t forward_chunk_elems_( int target_i ) const
    {
        return nx_local_ * output_dim_.size_y[target_i] * nz_local_;
    }

    std::size_t forward_recv_elems_( int source_i ) const
    {
        return input_dim_.size_x[source_i] * ny_local_ * nz_local_;
    }

    std::size_t forward_recv_offset_elems_( int source_i ) const
    {
        return input_dim_.start_x[source_i] * ny_local_;
    }

    std::size_t forward_recv_pack_offset_elems_( int source_i ) const
    {
        return forward_recv_offsets_[source_i];
    }

    std::size_t forward_output_offset_elems_( int source_i ) const
    {
        return input_dim_.start_x[source_i] * ny_local_ * nz_local_;
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

    bool cuda_aware_direct_p2p_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return direct_p2p_cuda_aware_;
#else
        return false;
#endif
    }

    bool cuda_aware_byte_p2p_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return use_p2p_byte_transfer_;
#else
        return false;
#endif
    }

    bool direct_backward_receive_requested_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return direct_p2p_cuda_aware_ && use_direct_backward_receive_;
#else
        return false;
#endif
    }

    bool direct_backward_value_receive_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return direct_backward_receive_requested_() && !cuda_aware_byte_p2p_enabled_();
#else
        return false;
#endif
    }

    bool direct_backward_byte_receive_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return direct_backward_receive_requested_() && cuda_aware_byte_p2p_enabled_();
#else
        return false;
#endif
    }

    bool direct_backward_receive_enabled_() const
    {
        return direct_backward_value_receive_enabled_() || direct_backward_byte_receive_enabled_();
    }

    std::size_t max_mpi_byte_chunk_bytes_() const
    {
        return static_cast<std::size_t>( std::numeric_limits<int>::max() );
    }

    int mpi_byte_chunk_count_( std::size_t bytes ) const
    {
        const std::size_t chunk = max_mpi_byte_chunk_bytes_();
        return detail::mpi_int_cast( ( bytes + chunk - 1 ) / chunk, "same_z byte p2p chunk count" );
    }

    void post_byte_irecv_(
        void *ptr, std::size_t bytes, int source, int tag, std::vector<mpi_request_t> &requests,
        std::vector<int> *request_peers = nullptr, std::vector<int> *remaining_by_peer = nullptr
    )
    {
        char              *base      = static_cast<char *>( ptr );
        const mpi_dtype_t  byte_type = scfd::communication::detail::mpi_data_type<char>::mpi_type();
        const std::size_t  max_bytes = max_mpi_byte_chunk_bytes_();
        int                chunks    = 0;
        for ( std::size_t offset = 0; offset < bytes; offset += max_bytes )
        {
            const std::size_t this_bytes = std::min( max_bytes, bytes - offset );
            requests.push_back( mpi_request_t() );
            line_comm_info_.irecv(
                base + offset, detail::mpi_int_cast( this_bytes, "same_z byte p2p recv count" ), byte_type, source,
                tag, requests.back()
            );
            if ( request_peers != nullptr )
            {
                request_peers->push_back( source );
            }
            ++chunks;
        }
        if ( remaining_by_peer != nullptr )
        {
            ( *remaining_by_peer )[source] += chunks;
        }
    }

    void post_byte_isend_(
        const void *ptr, std::size_t bytes, int dest, int tag, std::vector<mpi_request_t> &requests
    )
    {
        const char        *base      = static_cast<const char *>( ptr );
        const mpi_dtype_t  byte_type = scfd::communication::detail::mpi_data_type<char>::mpi_type();
        const std::size_t  max_bytes = max_mpi_byte_chunk_bytes_();
        for ( std::size_t offset = 0; offset < bytes; offset += max_bytes )
        {
            const std::size_t this_bytes = std::min( max_bytes, bytes - offset );
            requests.push_back( mpi_request_t() );
            line_comm_info_.isend(
                base + offset, detail::mpi_int_cast( this_bytes, "same_z byte p2p send count" ), byte_type, dest, tag,
                requests.back()
            );
        }
    }

    void waitall_vector_( std::vector<mpi_request_t> &requests, const char *what )
    {
        if ( requests.empty() )
        {
            return;
        }
        line_comm_info_.waitall( detail::mpi_int_cast( requests.size(), what ), requests.data() );
    }

    std::size_t direct_backward_recv_base_offset_elems_( int source_i ) const
    {
        return output_dim_.start_y[source_i] * nz_local_;
    }

    std::size_t direct_backward_recv_base_offset_bytes_( int source_i ) const
    {
        return bytes_from_elems_( direct_backward_recv_base_offset_elems_( source_i ) );
    }

    std::size_t direct_backward_row_bytes_( int source_i ) const
    {
        return bytes_from_elems_( output_dim_.size_y[source_i] * nz_local_ );
    }

    std::size_t direct_backward_stride_bytes_() const
    {
        return bytes_from_elems_( ny_global_ * nz_local_ );
    }

    std::size_t direct_backward_byte_rows_per_chunk_( int source_i ) const
    {
        const std::size_t row_bytes = direct_backward_row_bytes_( source_i );
        if ( row_bytes == 0 )
            return 1;
        return std::max<std::size_t>( 1, max_mpi_byte_chunk_bytes_() / row_bytes );
    }

    std::size_t direct_backward_send_row_bytes_() const
    {
        return bytes_from_elems_( ny_local_ * nz_local_ );
    }

    std::size_t direct_backward_byte_send_rows_per_chunk_() const
    {
        const std::size_t row_bytes = direct_backward_send_row_bytes_();
        if ( row_bytes == 0 )
            return 1;
        return std::max<std::size_t>( 1, max_mpi_byte_chunk_bytes_() / row_bytes );
    }

    struct direct_byte_recv_chunk_t
    {
        int         peer         = 0;
        std::size_t offset_bytes = 0;
        mpi_dtype_t datatype;
    };

    void post_direct_backward_byte_irecv_(
        void *base_ptr, int source, std::vector<mpi_request_t> &requests,
        std::vector<int> *request_peers = nullptr, std::vector<int> *remaining_by_peer = nullptr
    )
    {
        char *base = static_cast<char *>( base_ptr );
        for ( std::size_t i = 0; i < direct_backward_byte_recv_chunks_[source].size(); ++i )
        {
            const direct_byte_recv_chunk_t &chunk = direct_backward_byte_recv_chunks_[source][i];
            requests.push_back( mpi_request_t() );
            line_comm_info_.irecv( base + chunk.offset_bytes, 1, chunk.datatype, source, source, requests.back() );
            if ( request_peers != nullptr )
                request_peers->push_back( source );
            if ( remaining_by_peer != nullptr )
                ++( *remaining_by_peer )[source];
        }
    }

    void post_direct_backward_byte_isend_( const void *ptr, int dest, std::vector<mpi_request_t> &requests )
    {
        const char       *base      = static_cast<const char *>( ptr );
        const mpi_dtype_t byte_type = scfd::communication::detail::mpi_data_type<char>::mpi_type();
        const std::size_t row_bytes = direct_backward_send_row_bytes_();
        const std::size_t rows_step = direct_backward_byte_send_rows_per_chunk_();
        const std::size_t rows_total = input_dim_.size_x[dest];
        for ( std::size_t row = 0; row < rows_total; row += rows_step )
        {
            const std::size_t rows       = std::min( rows_step, rows_total - row );
            const std::size_t this_bytes = rows * row_bytes;
            requests.push_back( mpi_request_t() );
            line_comm_info_.isend(
                base + row * row_bytes, detail::mpi_int_cast( this_bytes, "same_z direct byte backward send count" ),
                byte_type, dest, myid_i_, requests.back()
            );
        }
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
        forward_sendtypes_w_.assign( comm_size, mpi_value_type_ );
        forward_recvtypes_w_.assign( comm_size, mpi_value_type_ );

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
            forward_sendcounts_[p] = detail::mpi_int_cast( forward_chunk_elems_( p ), "same_z forward sendcounts" );
            forward_sdispls_[p]    = detail::mpi_int_cast( forward_send_offsets_[p], "same_z forward sdispls" );
            forward_recvcounts_[p] =
                detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z forward recvcounts" );
            forward_rdispls_[p] =
                detail::mpi_int_cast( forward_recv_pack_offset_elems_( p ), "same_z forward rdispls" );
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
        backward_sendtypes_w_.assign( comm_size, mpi_value_type_ );
        backward_recvtypes_w_.assign( comm_size, mpi_value_type_ );

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
            backward_sendcounts_[p] =
                detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z backward sendcounts" );
            backward_sdispls_[p]    = detail::mpi_int_cast( backward_send_offsets_[p], "same_z backward sdispls" );
            backward_recvcounts_[p] = detail::mpi_int_cast( backward_chunk_elems_( p ), "same_z backward recvcounts" );
            backward_rdispls_[p] =
                detail::mpi_int_cast( backward_recv_pack_offset_elems_( p ), "same_z backward rdispls" );
        }
    }

    void init_direct_backward_recv_types_()
    {
        const int comm_size = line_comm_info_.num_procs;
        free_direct_backward_recv_types_();
        if ( !direct_backward_receive_requested_() )
            return;

        if ( direct_backward_value_receive_enabled_() )
            direct_backward_recvtypes_.assign( comm_size, mpi_dtype_t() );
        if ( direct_backward_byte_receive_enabled_() )
            direct_backward_byte_recv_chunks_.assign( comm_size, std::vector<direct_byte_recv_chunk_t>() );

        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_i_ )
                continue;

            const int count = detail::mpi_int_cast( nx_local_, "same_z direct backward recv count" );

            if ( direct_backward_value_receive_enabled_() )
            {
                const int blocklength = detail::mpi_int_cast(
                    output_dim_.size_y[p] * nz_local_, "same_z direct backward recv blocklength"
                );
                const int stride =
                    detail::mpi_int_cast( ny_global_ * nz_local_, "same_z direct backward recv stride" );

                direct_backward_recvtypes_[p] =
                    scfd::communication::detail::type_vector( count, blocklength, stride, mpi_value_type_ );
                scfd::communication::detail::type_commit( direct_backward_recvtypes_[p] );
            }
            if ( direct_backward_byte_receive_enabled_() )
            {
                const int         blocklength = detail::mpi_int_cast(
                    direct_backward_row_bytes_( p ), "same_z direct byte backward recv blocklength"
                );
                const int         stride =
                    detail::mpi_int_cast( direct_backward_stride_bytes_(), "same_z direct byte backward recv stride" );
                const mpi_dtype_t byte_type = scfd::communication::detail::mpi_data_type<char>::mpi_type();
                const std::size_t rows_step = direct_backward_byte_rows_per_chunk_( p );
                for ( std::size_t row = 0; row < nx_local_; row += rows_step )
                {
                    const std::size_t rows = std::min( rows_step, nx_local_ - row );
                    direct_byte_recv_chunk_t chunk;
                    chunk.peer         = p;
                    chunk.offset_bytes = direct_backward_recv_base_offset_bytes_( p ) + row * direct_backward_stride_bytes_();
                    chunk.datatype     = scfd::communication::detail::type_vector(
                        detail::mpi_int_cast( rows, "same_z direct byte backward recv rows" ), blocklength, stride,
                        byte_type
                    );
                    scfd::communication::detail::type_commit( chunk.datatype );
                    direct_backward_byte_recv_chunks_[p].push_back( chunk );
                }
            }
        }
    }

    void init_value_type_()
    {
        mpi_value_type_ = scfd::communication::detail::type_contiguous(
            detail::mpi_int_cast( sizeof( value_type ), "mpi_transpose_3d_same_z value type extent" ),
            scfd::communication::detail::mpi_data_type<char>::mpi_type()
        );
        scfd::communication::detail::type_commit( mpi_value_type_ );
    }

    void free_value_type_()
    {
        scfd::communication::detail::type_free( mpi_value_type_ );
    }

    void free_direct_backward_recv_types_()
    {
        for ( std::size_t i = 0; i < direct_backward_recvtypes_.size(); ++i )
        {
            scfd::communication::detail::type_free( direct_backward_recvtypes_[i] );
        }
        direct_backward_recvtypes_.clear();
        for ( std::size_t p = 0; p < direct_backward_byte_recv_chunks_.size(); ++p )
        {
            for ( std::size_t i = 0; i < direct_backward_byte_recv_chunks_[p].size(); ++i )
            {
                scfd::communication::detail::type_free( direct_backward_byte_recv_chunks_[p][i].datatype );
            }
        }
        direct_backward_byte_recv_chunks_.clear();
    }

    void synchronize_streams_( const char *label = "stream_synchronize" )
    {
        auto scope = profile_scope_( label );
        for ( std::size_t i = 0; i < streams_.size(); ++i )
        {
            runtime_api_t::stream_synchronize( streams_[i].stream() );
        }
    }

    void pack_forward_chunk_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t packed_y_size, value_type *dst_ptr,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( src_y_offset * sizeof( value_type ), 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, ny_global_ * sizeof( value_type ), ny_global_, nx_local_ );
        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, packed_y_size * sizeof( value_type ), packed_y_size, nx_local_ );
        params.extent = runtime_api_t::make_extent( packed_y_size * sizeof( value_type ), nx_local_, nz_local_ );
        params.kind   = runtime_api_t::device_to_device_kind();
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void unpack_forward_chunk_async_(
        const value_type *src_ptr, std::size_t packed_x_size, std::size_t dst_x_offset, value_type *dst_ptr,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos                                     = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, ny_local_ * sizeof( value_type ), ny_local_, packed_x_size );
        params.dstPos = runtime_api_t::make_pos( 0, dst_x_offset, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, ny_local_ * sizeof( value_type ), ny_local_, nx_global_ );
        params.extent = runtime_api_t::make_extent( ny_local_ * sizeof( value_type ), packed_x_size, nz_local_ );
        params.kind   = runtime_api_t::device_to_device_kind();
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void unpack_backward_chunk_async_(
        const value_type *src_ptr, std::size_t packed_y_size, std::size_t dst_y_offset, value_type *dst_ptr,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos                                     = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, packed_y_size * sizeof( value_type ), packed_y_size, nx_local_ );
        params.dstPos = runtime_api_t::make_pos( dst_y_offset * sizeof( value_type ), 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, ny_global_ * sizeof( value_type ), ny_global_, nx_local_ );
        params.extent = runtime_api_t::make_extent( packed_y_size * sizeof( value_type ), nx_local_, nz_local_ );
        params.kind   = runtime_api_t::device_to_device_kind();
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void pack_backward_chunk_async_(
        const value_type *src_ptr, std::size_t src_x_offset, std::size_t packed_x_size, value_type *dst_ptr,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos                                     = runtime_api_t::make_pos( 0, src_x_offset, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, ny_local_ * sizeof( value_type ), ny_local_, nx_global_ );
        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, ny_local_ * sizeof( value_type ), ny_local_, packed_x_size );
        params.extent = runtime_api_t::make_extent( ny_local_ * sizeof( value_type ), packed_x_size, nz_local_ );
        params.kind   = runtime_api_t::device_to_device_kind();
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    template <class ArrayIn>
    void pack_forward_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_forward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            pack_forward_chunk_async_(
                in.raw_ptr(), output_dim_.start_y[p], output_dim_.size_y[p],
                send_buffer_.raw_ptr() + forward_send_offsets_[p], streams_[p].stream()
            );
        }
        synchronize_streams_( "pack_complete" );
    }

    template <class ArrayOut>
    void unpack_forward_chunk_async_( int source_i, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_forward_chunk" );
        unpack_forward_chunk_async_(
            recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( source_i ), input_dim_.size_x[source_i],
            input_dim_.start_x[source_i], out.raw_ptr(), streams_[source_i].stream()
        );
    }

    template <class ArrayOut>
    void unpack_backward_chunk_async_( int source_i, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_backward_chunk" );
        unpack_backward_chunk_async_(
            recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( source_i ), output_dim_.size_y[source_i],
            output_dim_.start_y[source_i], out.raw_ptr(), streams_[source_i].stream()
        );
    }

    template <class ArrayIn>
    void pack_backward_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_backward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            pack_backward_chunk_async_(
                in.raw_ptr(), input_dim_.start_x[p], input_dim_.size_x[p],
                send_buffer_.raw_ptr() + backward_send_offsets_[p], streams_[p].stream()
            );
        }
        synchronize_streams_( "pack_complete" );
    }

    void pack_forward_optimized_chunk_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t packed_y_size, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        const std::size_t src_pitch_elems                 = ny_global_ * nz_local_;
        const std::size_t dst_pitch_elems                 = packed_y_size * nz_local_;
        params.srcPos = runtime_api_t::make_pos( src_y_offset * nz_local_ * sizeof( value_type ), 0, 0 );
        params.srcPtr = runtime_api_t::make_pitched_ptr(
            src_ptr, src_pitch_elems * sizeof( value_type ), src_pitch_elems, nx_local_
        );
        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPtr = runtime_api_t::make_pitched_ptr(
            dst_ptr, dst_pitch_elems * sizeof( value_type ), dst_pitch_elems, nx_local_
        );
        params.extent = runtime_api_t::make_extent( dst_pitch_elems * sizeof( value_type ), nx_local_, 1 );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void pack_forward_optimized_chunk_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t packed_y_size, value_type *dst_ptr,
        typename runtime_api_t::stream_t stream
    ) const
    {
        pack_forward_optimized_chunk_async_(
            src_ptr, src_y_offset, packed_y_size, dst_ptr, runtime_api_t::device_to_device_kind(), stream
        );
    }

    void copy_backward_optimized_chunk_to_output_async_(
        const value_type *src_ptr, std::size_t src_y_size, std::size_t dst_y_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        const std::size_t src_pitch_elems                 = src_y_size * nz_local_;
        const std::size_t dst_pitch_elems                 = ny_global_ * nz_local_;
        params.srcPos                                     = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr = runtime_api_t::make_pitched_ptr(
            src_ptr, src_pitch_elems * sizeof( value_type ), src_pitch_elems, nx_local_
        );
        params.dstPos = runtime_api_t::make_pos( dst_y_offset * nz_local_ * sizeof( value_type ), 0, 0 );
        params.dstPtr = runtime_api_t::make_pitched_ptr(
            dst_ptr, dst_pitch_elems * sizeof( value_type ), dst_pitch_elems, nx_local_
        );
        params.extent = runtime_api_t::make_extent( src_pitch_elems * sizeof( value_type ), nx_local_, 1 );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void unpack_backward_optimized_chunk_async_(
        const value_type *src_ptr, std::size_t src_y_size, std::size_t dst_y_offset, value_type *dst_ptr,
        typename runtime_api_t::stream_t stream
    ) const
    {
        copy_backward_optimized_chunk_to_output_async_(
            src_ptr, src_y_size, dst_y_offset, dst_ptr, runtime_api_t::device_to_device_kind(), stream
        );
    }

    template <class ArrayIn>
    void pack_forward_optimized_layout_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_forward_optimized_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            pack_forward_optimized_chunk_async_(
                in.raw_ptr(), output_dim_.start_y[p], output_dim_.size_y[p],
                send_buffer_.raw_ptr() + forward_send_offsets_[p], streams_[p].stream()
            );
        }
        synchronize_streams_( "pack_complete" );
    }

    template <class ArrayOut>
    void unpack_forward_optimized_chunk_async_( int source_i, ArrayOut &out )
    {
        auto              scope      = profile_scope_( "unpack_forward_optimized_chunk" );
        const std::size_t dst_offset = forward_output_offset_elems_( source_i );
        runtime_api_t::memcpy_async(
            out.raw_ptr() + dst_offset, recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( source_i ),
            bytes_from_elems_( forward_recv_elems_( source_i ) ), runtime_api_t::device_to_device_kind(),
            streams_[source_i].stream()
        );
    }

    template <class ArrayOut>
    void copy_forward_optimized_self_to_output_async_( ArrayOut &out )
    {
        runtime_api_t::memcpy_async(
            out.raw_ptr() + forward_output_offset_elems_( myid_i_ ),
            send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
            bytes_from_elems_( forward_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind(),
            streams_[myid_i_].stream()
        );
    }

    template <class ArrayIn, class ArrayOut>
    void copy_forward_optimized_self_from_input_to_output_async_( const ArrayIn &in, ArrayOut &out )
    {
        pack_forward_optimized_chunk_async_(
            in.raw_ptr(), output_dim_.start_y[myid_i_], output_dim_.size_y[myid_i_],
            out.raw_ptr() + forward_output_offset_elems_( myid_i_ ), runtime_api_t::device_to_device_kind(),
            streams_[myid_i_].stream()
        );
    }

    template <class ArrayIn>
    void pack_forward_optimized_send_chunk_for_p2p_async_( const ArrayIn &in, int peer )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        pack_forward_optimized_chunk_async_(
            in.raw_ptr(), output_dim_.start_y[peer], output_dim_.size_y[peer],
            send_buffer_.raw_ptr() + forward_send_offsets_[peer], runtime_api_t::device_to_device_kind(),
            streams_[peer].stream()
        );
#else
        pack_forward_optimized_chunk_async_(
            in.raw_ptr(), output_dim_.start_y[peer], output_dim_.size_y[peer],
            host_send_buffer_.raw_ptr() + forward_send_offsets_[peer], runtime_api_t::device_to_host_kind(),
            streams_[peer].stream()
        );
#endif
    }

    template <class ArrayOut>
    void copy_host_forward_optimized_chunk_to_output_async_( int source_i, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy_async(
            out.raw_ptr() + forward_output_offset_elems_( source_i ),
            host_recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( source_i ),
            bytes_from_elems_( forward_recv_elems_( source_i ) ), runtime_api_t::host_to_device_kind(),
            streams_[source_i].stream()
        );
#else
        (void)source_i;
        (void)out;
#endif
    }

    template <class ArrayOut>
    void unpack_backward_optimized_chunk_async_( int source_i, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_backward_optimized_chunk" );
        unpack_backward_optimized_chunk_async_(
            recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( source_i ), output_dim_.size_y[source_i],
            output_dim_.start_y[source_i], out.raw_ptr(), streams_[source_i].stream()
        );
    }

    template <class ArrayIn, class ArrayOut>
    void copy_backward_optimized_self_to_output_async_( const ArrayIn &in, ArrayOut &out )
    {
        copy_backward_optimized_chunk_to_output_async_(
            in.raw_ptr() + backward_send_offsets_[myid_i_], output_dim_.size_y[myid_i_],
            output_dim_.start_y[myid_i_], out.raw_ptr(), runtime_api_t::device_to_device_kind(),
            streams_[myid_i_].stream()
        );
    }

    template <class ArrayOut>
    void copy_host_backward_optimized_chunk_to_output_async_( int source_i, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        copy_backward_optimized_chunk_to_output_async_(
            host_recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( source_i ), output_dim_.size_y[source_i],
            output_dim_.start_y[source_i], out.raw_ptr(), runtime_api_t::host_to_device_kind(),
            streams_[source_i].stream()
        );
#else
        (void)source_i;
        (void)out;
#endif
    }

    template <class ArrayIn>
    void copy_backward_optimized_input_to_host_( const ArrayIn &in )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy(
            host_send_buffer_.raw_ptr(), in.raw_ptr(), bytes_from_elems_( nx_global_ * ny_local_ * nz_local_ ),
            runtime_api_t::device_to_host_kind()
        );
#else
        (void)in;
#endif
    }

    template <class ArrayIn>
    void copy_backward_optimized_send_chunk_to_host_async_( const ArrayIn &in, int peer )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy_async(
            host_send_buffer_.raw_ptr() + backward_send_offsets_[peer], in.raw_ptr() + backward_send_offsets_[peer],
            bytes_from_elems_( input_dim_.size_x[peer] * ny_local_ * nz_local_ ),
            runtime_api_t::device_to_host_kind(), streams_[peer].stream()
        );
#else
        (void)in;
        (void)peer;
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void forward_optimized_p2p_waitall_( const ArrayIn &in, ArrayOut &out )
    {
        auto      scope     = profile_scope_( "forward_optimized_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        {
            auto phase = profile_scope_( "post_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                value_type *recv_ptr = cuda_aware_direct_p2p_enabled_()
                                          ? out.raw_ptr() + forward_output_offset_elems_( p )
                                          : recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( p );
#else
                value_type *recv_ptr = host_recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( p );
#endif
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                if ( cuda_aware_byte_p2p_enabled_() )
                {
                    post_byte_irecv_(
                        recv_ptr, bytes_from_elems_( forward_recv_elems_( p ) ), p, p, byte_recv_requests
                    );
                }
                else
#endif
                line_comm_info_.irecv(
                    recv_ptr,
                    detail::mpi_int_cast( forward_recv_elems_( p ), "same_z opt forward recv count" ),
                    mpi_value_type_, p, p, recv_requests_[p]
                );
            }
        }
        send_ready_state_t                    send_ready;
        std::vector<send_ready_callback_t>    callbacks( comm_size );
        send_ready.ready_peers.reserve( comm_size );
        auto start_send = [this, &byte_send_requests]( int p ) {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            value_type *send_ptr = send_buffer_.raw_ptr() + forward_send_offsets_[p];
#else
            value_type *send_ptr = host_send_buffer_.raw_ptr() + forward_send_offsets_[p];
#endif
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                post_byte_isend_(
                    send_ptr, bytes_from_elems_( forward_chunk_elems_( p ) ), p, myid_i_, byte_send_requests
                );
                return;
            }
#endif
            line_comm_info_.isend(
                send_ptr, detail::mpi_int_cast( forward_chunk_elems_( p ), "same_z opt forward send count" ),
                mpi_value_type_, p, myid_i_, send_requests_[p]
            );
        };
        const bool         use_send_thread = p2p_send_thread_enabled_( comm_size - 1 );
        std::exception_ptr send_exception;
        std::thread        send_thread;
        try
        {
        {
            auto phase = profile_scope_( "pack_send_chunks" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                pack_forward_optimized_send_chunk_for_p2p_async_( in, p );
                enqueue_send_ready_callback_( p, send_ready, callbacks );
            }
        }
        if ( use_send_thread )
        {
            auto phase  = profile_scope_( "start_send_thread" );
            send_thread = start_ready_send_thread_( comm_size - 1, send_ready, start_send, send_exception );
        }
        {
            auto phase = profile_scope_( "self_copy" );
            copy_forward_optimized_self_from_input_to_output_async_( in, out );
        }
        if ( !use_send_thread )
            wait_ready_and_post_sends_( comm_size - 1, send_ready, start_send );
        {
            auto phase = profile_scope_( "wait_recv" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_recv_requests, "same_z byte forward recv request count" );
            else
#endif
                line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < comm_size; ++p )
                if ( p != myid_i_ )
                    copy_host_forward_optimized_chunk_to_output_async_( p, out );
        }
#else
        if ( !cuda_aware_direct_p2p_enabled_() )
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                if ( p != myid_i_ )
                    unpack_forward_optimized_chunk_async_( p, out );
        }
#endif
        synchronize_streams_( "recv_copy_complete" );
        join_ready_send_thread_( send_ready, send_thread, send_exception );
        {
            auto phase = profile_scope_( "wait_send" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_send_requests, "same_z byte forward send request count" );
            else
#endif
                line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
        }
        catch ( ... )
        {
            cancel_ready_send_thread_( send_ready, send_thread );
            throw;
        }
    }

    template <class ArrayIn, class ArrayOut>
    void forward_optimized_p2p_waitany_( const ArrayIn &in, ArrayOut &out )
    {
        auto      scope     = profile_scope_( "forward_optimized_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( comm_size, 0 );
        {
            auto phase = profile_scope_( "post_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                value_type *recv_ptr = cuda_aware_direct_p2p_enabled_()
                                          ? out.raw_ptr() + forward_output_offset_elems_( p )
                                          : recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( p );
#else
                value_type *recv_ptr = host_recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( p );
#endif
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                if ( cuda_aware_byte_p2p_enabled_() )
                {
                    post_byte_irecv_(
                        recv_ptr, bytes_from_elems_( forward_recv_elems_( p ) ), p, p, byte_recv_requests,
                        &byte_recv_peers, &byte_recv_remaining
                    );
                }
                else
#endif
                line_comm_info_.irecv(
                    recv_ptr,
                    detail::mpi_int_cast( forward_recv_elems_( p ), "same_z opt forward recv count" ),
                    mpi_value_type_, p, p, recv_requests_[p]
                );
            }
        }
        send_ready_state_t                    send_ready;
        std::vector<send_ready_callback_t>    callbacks( comm_size );
        send_ready.ready_peers.reserve( comm_size );
        auto start_send = [this, &byte_send_requests]( int p ) {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            value_type *send_ptr = send_buffer_.raw_ptr() + forward_send_offsets_[p];
#else
            value_type *send_ptr = host_send_buffer_.raw_ptr() + forward_send_offsets_[p];
#endif
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                post_byte_isend_(
                    send_ptr, bytes_from_elems_( forward_chunk_elems_( p ) ), p, myid_i_, byte_send_requests
                );
                return;
            }
#endif
            line_comm_info_.isend(
                send_ptr, detail::mpi_int_cast( forward_chunk_elems_( p ), "same_z opt forward send count" ),
                mpi_value_type_, p, myid_i_, send_requests_[p]
            );
        };
        const bool         use_send_thread = p2p_send_thread_enabled_( comm_size - 1 );
        std::exception_ptr send_exception;
        std::thread        send_thread;
        try
        {
        {
            auto phase = profile_scope_( "pack_send_chunks" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                pack_forward_optimized_send_chunk_for_p2p_async_( in, p );
                enqueue_send_ready_callback_( p, send_ready, callbacks );
            }
        }
        if ( use_send_thread )
        {
            auto phase  = profile_scope_( "start_send_thread" );
            send_thread = start_ready_send_thread_( comm_size - 1, send_ready, start_send, send_exception );
        }
        {
            auto phase = profile_scope_( "self_copy" );
            copy_forward_optimized_self_from_input_to_output_async_( in, out );
        }
        if ( !use_send_thread )
            wait_ready_and_post_sends_( comm_size - 1, send_ready, start_send );
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                int completed = 0;
                const int total_requests =
                    detail::mpi_int_cast( byte_recv_requests.size(), "same_z byte forward waitany request count" );
                while ( completed < total_requests )
                {
                    int request_index = MPI_UNDEFINED;
                    {
                        auto phase   = profile_scope_( "wait_recv_any" );
                        request_index = line_comm_info_.waitany( total_requests, byte_recv_requests.data() );
                    }
                    if ( request_index == MPI_UNDEFINED )
                        break;
                    const int p = byte_recv_peers[request_index];
                    --byte_recv_remaining[p];
                    if ( byte_recv_remaining[p] == 0 && !cuda_aware_direct_p2p_enabled_() )
                    {
                        auto phase = profile_scope_( "unpack_recv_chunk" );
                        unpack_forward_optimized_chunk_async_( p, out );
                    }
                    completed++;
                }
            }
            else
#endif
            {
            int completed = 0;
            while ( completed < comm_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                {
                    auto phase = profile_scope_( "stage_recv_chunk_to_device" );
                    copy_host_forward_optimized_chunk_to_output_async_( p, out );
                }
#else
                if ( !cuda_aware_direct_p2p_enabled_() )
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_forward_optimized_chunk_async_( p, out );
                }
#endif
                completed++;
            }
            }
        }
        synchronize_streams_( "recv_copy_complete" );
        join_ready_send_thread_( send_ready, send_thread, send_exception );
        {
            auto phase = profile_scope_( "wait_send" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_send_requests, "same_z byte forward send request count" );
            else
#endif
                line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
        }
        catch ( ... )
        {
            cancel_ready_send_thread_( send_ready, send_thread );
            throw;
        }
    }

    template <class ArrayOut>
    void forward_optimized_alltoallv_( ArrayOut &out )
    {
        auto scope = profile_scope_( "forward_optimized_alltoallv" );
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        line_comm_info_.alltoallv(
            host_send_buffer_.raw_ptr(), forward_sendcounts_.data(), forward_sdispls_.data(), mpi_value_type_,
            host_recv_buffer_.raw_ptr(), forward_recvcounts_.data(), forward_rdispls_.data(), mpi_value_type_
        );
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                copy_host_forward_optimized_chunk_to_output_async_( p, out );
        }
#else
        line_comm_info_.alltoallv(
            send_buffer_.raw_ptr(), forward_sendcounts_.data(), forward_sdispls_.data(), mpi_value_type_,
            out.raw_ptr(), forward_recvcounts_.data(), forward_rdispls_.data(), mpi_value_type_
        );
#endif
        synchronize_streams_( "recv_copy_complete" );
    }

    template <class ArrayOut>
    void forward_optimized_alltoallw_( ArrayOut &out )
    {
        auto             scope = profile_scope_( "forward_optimized_alltoallw" );
        std::vector<int> forward_sdispls_w( line_comm_info_.num_procs );
        std::vector<int> forward_rdispls_w( line_comm_info_.num_procs );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                forward_sdispls_w[p] = detail::mpi_int_cast(
                    bytes_from_elems_( forward_send_offsets_[p] ), "same_z opt forward sdispls_w"
                );
                forward_rdispls_w[p] = detail::mpi_int_cast(
                    bytes_from_elems_( forward_recv_pack_offset_elems_( p ) ), "same_z opt forward rdispls_w"
                );
            }
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                forward_sdispls_w.data(), forward_sendtypes_w_.data(),
                static_cast<void *>( host_recv_buffer_.raw_ptr() ), forward_recvcounts_.data(),
                forward_rdispls_w.data(), forward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                copy_host_forward_optimized_chunk_to_output_async_( p, out );
        }
#else
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                forward_sdispls_w.data(), forward_sendtypes_w_.data(), static_cast<void *>( out.raw_ptr() ),
                forward_recvcounts_.data(), forward_rdispls_w.data(), forward_recvtypes_w_.data()
            );
        }
#endif
        synchronize_streams_( "recv_copy_complete" );
    }

    template <class ArrayIn, class ArrayOut>
    void backward_optimized_p2p_waitall_( const ArrayIn &in, ArrayOut &out )
    {
        auto      scope     = profile_scope_( "backward_optimized_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                const value_type *send_ptr = in.raw_ptr() + backward_send_offsets_[p];
                if ( direct_backward_byte_receive_enabled_() )
                {
                    post_direct_backward_byte_irecv_( out.raw_ptr(), p, byte_recv_requests );
                }
                else if ( cuda_aware_byte_p2p_enabled_() )
                {
                    value_type *recv_ptr = recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p );
                    post_byte_irecv_(
                        recv_ptr, bytes_from_elems_( backward_chunk_elems_( p ) ), p, p, byte_recv_requests
                    );
                }
                else if ( direct_backward_value_receive_enabled_() )
                {
                    value_type *recv_ptr = out.raw_ptr() + direct_backward_recv_base_offset_elems_( p );
                    line_comm_info_.irecv( recv_ptr, 1, direct_backward_recvtypes_[p], p, p, recv_requests_[p] );
                }
                else
                {
                    value_type *recv_ptr = recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p );
                    line_comm_info_.irecv(
                        recv_ptr,
                        detail::mpi_int_cast( backward_chunk_elems_( p ), "same_z opt backward recv count" ),
                        mpi_value_type_, p, p, recv_requests_[p]
                    );
                }
#else
                value_type *recv_ptr = host_recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p );
                value_type *send_ptr = host_send_buffer_.raw_ptr() + backward_send_offsets_[p];
                line_comm_info_.irecv(
                    recv_ptr, detail::mpi_int_cast( backward_chunk_elems_( p ), "same_z opt backward recv count" ),
                    mpi_value_type_, p, p, recv_requests_[p]
                );
#endif
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                if ( cuda_aware_byte_p2p_enabled_() )
                {
                    if ( direct_backward_byte_receive_enabled_() )
                        post_direct_backward_byte_isend_( send_ptr, p, byte_send_requests );
                    else
                        post_byte_isend_(
                            send_ptr, bytes_from_elems_( input_dim_.size_x[p] * ny_local_ * nz_local_ ), p, myid_i_,
                            byte_send_requests
                        );
                }
                else
                {
                    line_comm_info_.isend(
                        const_cast<value_type *>( send_ptr ),
                        detail::mpi_int_cast(
                            input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z opt backward send count"
                        ),
                        mpi_value_type_, p, myid_i_, send_requests_[p]
                    );
                }
#endif
            }
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        send_ready_state_t                 send_ready;
        std::vector<send_ready_callback_t> callbacks( comm_size );
        send_ready.ready_peers.reserve( comm_size );
        auto start_send = [this]( int p ) {
            value_type *send_ptr = host_send_buffer_.raw_ptr() + backward_send_offsets_[p];
            line_comm_info_.isend(
                send_ptr,
                detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z opt backward send count" ),
                mpi_value_type_, p, myid_i_, send_requests_[p]
            );
        };
        const bool         use_send_thread = p2p_send_thread_enabled_( comm_size - 1 );
        std::exception_ptr send_exception;
        std::thread        send_thread;
        try
        {
        {
            auto phase = profile_scope_( "stage_send_to_host_chunks" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                copy_backward_optimized_send_chunk_to_host_async_( in, p );
                enqueue_send_ready_callback_( p, send_ready, callbacks );
            }
        }
        if ( use_send_thread )
        {
            auto phase  = profile_scope_( "start_send_thread" );
            send_thread = start_ready_send_thread_( comm_size - 1, send_ready, start_send, send_exception );
        }
#endif
        {
            auto phase = profile_scope_( "self_copy" );
            copy_backward_optimized_self_to_output_async_( in, out );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( !use_send_thread )
            wait_ready_and_post_sends_( comm_size - 1, send_ready, start_send );
#endif
        {
            auto phase = profile_scope_( "wait_recv" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_recv_requests, "same_z byte backward recv request count" );
            else
#endif
                line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < comm_size; ++p )
                if ( p != myid_i_ )
                    copy_host_backward_optimized_chunk_to_output_async_( p, out );
        }
#else
        if ( !direct_backward_receive_enabled_() )
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                if ( p != myid_i_ )
                    unpack_backward_optimized_chunk_async_( p, out );
        }
#endif
        synchronize_streams_( "recv_copy_complete" );
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        join_ready_send_thread_( send_ready, send_thread, send_exception );
#endif
        {
            auto phase = profile_scope_( "wait_send" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_send_requests, "same_z byte backward send request count" );
            else
#endif
                line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        }
        catch ( ... )
        {
            cancel_ready_send_thread_( send_ready, send_thread );
            throw;
        }
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_optimized_p2p_waitany_( const ArrayIn &in, ArrayOut &out )
    {
        auto      scope     = profile_scope_( "backward_optimized_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( comm_size, 0 );
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                const value_type *send_ptr = in.raw_ptr() + backward_send_offsets_[p];
                if ( direct_backward_byte_receive_enabled_() )
                {
                    post_direct_backward_byte_irecv_(
                        out.raw_ptr(), p, byte_recv_requests, &byte_recv_peers, &byte_recv_remaining
                    );
                }
                else if ( cuda_aware_byte_p2p_enabled_() )
                {
                    value_type *recv_ptr = recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p );
                    post_byte_irecv_(
                        recv_ptr, bytes_from_elems_( backward_chunk_elems_( p ) ), p, p, byte_recv_requests,
                        &byte_recv_peers, &byte_recv_remaining
                    );
                }
                else if ( direct_backward_value_receive_enabled_() )
                {
                    value_type *recv_ptr = out.raw_ptr() + direct_backward_recv_base_offset_elems_( p );
                    line_comm_info_.irecv( recv_ptr, 1, direct_backward_recvtypes_[p], p, p, recv_requests_[p] );
                }
                else
                {
                    value_type *recv_ptr = recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p );
                    line_comm_info_.irecv(
                        recv_ptr,
                        detail::mpi_int_cast( backward_chunk_elems_( p ), "same_z opt backward recv count" ),
                        mpi_value_type_, p, p, recv_requests_[p]
                    );
                }
#else
                value_type *recv_ptr = host_recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p );
                value_type *send_ptr = host_send_buffer_.raw_ptr() + backward_send_offsets_[p];
                line_comm_info_.irecv(
                    recv_ptr, detail::mpi_int_cast( backward_chunk_elems_( p ), "same_z opt backward recv count" ),
                    mpi_value_type_, p, p, recv_requests_[p]
                );
#endif
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                if ( cuda_aware_byte_p2p_enabled_() )
                {
                    if ( direct_backward_byte_receive_enabled_() )
                        post_direct_backward_byte_isend_( send_ptr, p, byte_send_requests );
                    else
                        post_byte_isend_(
                            send_ptr, bytes_from_elems_( input_dim_.size_x[p] * ny_local_ * nz_local_ ), p, myid_i_,
                            byte_send_requests
                        );
                }
                else
                {
                    line_comm_info_.isend(
                        const_cast<value_type *>( send_ptr ),
                        detail::mpi_int_cast(
                            input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z opt backward send count"
                        ),
                        mpi_value_type_, p, myid_i_, send_requests_[p]
                    );
                }
#endif
            }
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        send_ready_state_t                 send_ready;
        std::vector<send_ready_callback_t> callbacks( comm_size );
        send_ready.ready_peers.reserve( comm_size );
        auto start_send = [this]( int p ) {
            value_type *send_ptr = host_send_buffer_.raw_ptr() + backward_send_offsets_[p];
            line_comm_info_.isend(
                send_ptr,
                detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z opt backward send count" ),
                mpi_value_type_, p, myid_i_, send_requests_[p]
            );
        };
        const bool         use_send_thread = p2p_send_thread_enabled_( comm_size - 1 );
        std::exception_ptr send_exception;
        std::thread        send_thread;
        try
        {
        {
            auto phase = profile_scope_( "stage_send_to_host_chunks" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                copy_backward_optimized_send_chunk_to_host_async_( in, p );
                enqueue_send_ready_callback_( p, send_ready, callbacks );
            }
        }
        if ( use_send_thread )
        {
            auto phase  = profile_scope_( "start_send_thread" );
            send_thread = start_ready_send_thread_( comm_size - 1, send_ready, start_send, send_exception );
        }
#endif
        {
            auto phase = profile_scope_( "self_copy" );
            copy_backward_optimized_self_to_output_async_( in, out );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( !use_send_thread )
            wait_ready_and_post_sends_( comm_size - 1, send_ready, start_send );
#endif
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                int completed = 0;
                const int total_requests =
                    detail::mpi_int_cast( byte_recv_requests.size(), "same_z byte backward waitany request count" );
                while ( completed < total_requests )
                {
                    int request_index = MPI_UNDEFINED;
                    {
                        auto phase   = profile_scope_( "wait_recv_any" );
                        request_index = line_comm_info_.waitany( total_requests, byte_recv_requests.data() );
                    }
                    if ( request_index == MPI_UNDEFINED )
                        break;
                    const int p = byte_recv_peers[request_index];
                    --byte_recv_remaining[p];
                    if ( byte_recv_remaining[p] == 0 && !direct_backward_byte_receive_enabled_() )
                    {
                        auto phase = profile_scope_( "unpack_recv_chunk" );
                        unpack_backward_optimized_chunk_async_( p, out );
                    }
                    completed++;
                }
            }
            else
#endif
            {
            int completed = 0;
            while ( completed < comm_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                {
                    auto phase = profile_scope_( "stage_recv_chunk_to_device" );
                    copy_host_backward_optimized_chunk_to_output_async_( p, out );
                }
#else
                if ( !direct_backward_receive_enabled_() )
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_backward_optimized_chunk_async_( p, out );
                }
#endif
                completed++;
            }
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        join_ready_send_thread_( send_ready, send_thread, send_exception );
#endif
        {
            auto phase = profile_scope_( "wait_send" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( byte_send_requests, "same_z byte backward send request count" );
            else
#endif
                line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        }
        catch ( ... )
        {
            cancel_ready_send_thread_( send_ready, send_thread );
            throw;
        }
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_optimized_alltoallv_( const ArrayIn &in, ArrayOut &out )
    {
        auto scope = profile_scope_( "backward_optimized_alltoallv" );
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_backward_optimized_input_to_host_( in );
        }
        line_comm_info_.alltoallv(
            host_send_buffer_.raw_ptr(), backward_sendcounts_.data(), backward_sdispls_.data(), mpi_value_type_,
            host_recv_buffer_.raw_ptr(), backward_recvcounts_.data(), backward_rdispls_.data(), mpi_value_type_
        );
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                copy_host_backward_optimized_chunk_to_output_async_( p, out );
        }
#else
        line_comm_info_.alltoallv(
            const_cast<value_type *>( in.raw_ptr() ), backward_sendcounts_.data(), backward_sdispls_.data(), mpi_value_type_,
            recv_buffer_.raw_ptr(), backward_recvcounts_.data(), backward_rdispls_.data(), mpi_value_type_
        );
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_backward_optimized_chunk_async_( p, out );
        }
#endif
        synchronize_streams_( "recv_copy_complete" );
    }

    template <class ArrayIn, class ArrayOut>
    void backward_optimized_alltoallw_( const ArrayIn &in, ArrayOut &out )
    {
        auto             scope = profile_scope_( "backward_optimized_alltoallw" );
        std::vector<int> backward_sdispls_w( line_comm_info_.num_procs );
        std::vector<int> backward_rdispls_w( line_comm_info_.num_procs );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                backward_sdispls_w[p] = detail::mpi_int_cast(
                    bytes_from_elems_( backward_send_offsets_[p] ), "same_z opt backward sdispls_w"
                );
                backward_rdispls_w[p] = detail::mpi_int_cast(
                    bytes_from_elems_( backward_recv_pack_offset_elems_( p ) ), "same_z opt backward rdispls_w"
                );
            }
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_backward_optimized_input_to_host_( in );
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                backward_sdispls_w.data(), backward_sendtypes_w_.data(),
                static_cast<void *>( host_recv_buffer_.raw_ptr() ), backward_recvcounts_.data(),
                backward_rdispls_w.data(), backward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                copy_host_backward_optimized_chunk_to_output_async_( p, out );
        }
#else
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( in.raw_ptr() ), backward_sendcounts_.data(), backward_sdispls_w.data(),
                backward_sendtypes_w_.data(), static_cast<void *>( recv_buffer_.raw_ptr() ), backward_recvcounts_.data(),
                backward_rdispls_w.data(), backward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_backward_optimized_chunk_async_( p, out );
        }
#endif
        synchronize_streams_( "recv_copy_complete" );
    }

    template <class ArrayOut>
    void forward_p2p_waitall_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto      scope     = profile_scope_( "forward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( p ),
                    detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z forward recv count" ),
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + forward_send_offsets_[p],
                    detail::mpi_int_cast( forward_chunk_elems_( p ), "same_z forward send count" ), mpi_value_type_, p,
                    myid_i_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy_async(
                recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( myid_i_ ),
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
                bytes_from_elems_( forward_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_i_].stream()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p != myid_i_ )
                {
                    copy_host_forward_chunk_to_device_async_( p );
                }
            }
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                unpack_forward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#else
        auto      scope     = profile_scope_( "forward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( p ),
                    detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z forward recv count" ),
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + forward_send_offsets_[p],
                    detail::mpi_int_cast( forward_chunk_elems_( p ), "same_z forward send count" ), mpi_value_type_, p,
                    myid_i_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy_async(
                recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( myid_i_ ),
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
                bytes_from_elems_( forward_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_i_].stream()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                unpack_forward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
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
        auto      scope     = profile_scope_( "forward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( p ),
                    detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z forward recv count" ),
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + forward_send_offsets_[p],
                    detail::mpi_int_cast( forward_chunk_elems_( p ), "same_z forward send count" ), mpi_value_type_, p,
                    myid_i_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy_async(
                recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( myid_i_ ),
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
                bytes_from_elems_( forward_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_i_].stream()
            );
        }

        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_forward_chunk_async_( myid_i_, out );
        }

        {
            int completed = 0;
            while ( completed < comm_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                {
                    auto phase = profile_scope_( "stage_recv_chunk_to_device" );
                    copy_host_forward_chunk_to_device_async_( p );
                }
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_forward_chunk_async_( p, out );
                }
                completed++;
            }
        }

        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#else
        auto      scope     = profile_scope_( "forward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( p ),
                    detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z forward recv count" ),
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + forward_send_offsets_[p],
                    detail::mpi_int_cast( forward_chunk_elems_( p ), "same_z forward send count" ), mpi_value_type_, p,
                    myid_i_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy_async(
                recv_buffer_.raw_ptr() + forward_recv_pack_offset_elems_( myid_i_ ),
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_i_],
                bytes_from_elems_( forward_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_i_].stream()
            );
        }

        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_forward_chunk_async_( myid_i_, out );
        }

        {
            int completed = 0;
            while ( completed < comm_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_forward_chunk_async_( p, out );
                }
                completed++;
            }
        }

        synchronize_streams_( "recv_copy_complete" );
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
        auto scope = profile_scope_( "forward_alltoallv" );
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            line_comm_info_.alltoallv(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                forward_sdispls_.data(), mpi_value_type_, static_cast<void *>( host_recv_buffer_.raw_ptr() ),
                forward_recvcounts_.data(), forward_rdispls_.data(), mpi_value_type_
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr(), host_recv_buffer_.raw_ptr(), bytes_from_elems_( recv_buffer_elems_ ),
                runtime_api_t::host_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                unpack_forward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#else
        auto scope = profile_scope_( "forward_alltoallv" );
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            line_comm_info_.alltoallv(
                static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                forward_sdispls_.data(), mpi_value_type_, static_cast<void *>( recv_buffer_.raw_ptr() ),
                forward_recvcounts_.data(), forward_rdispls_.data(), mpi_value_type_
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                unpack_forward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#endif
    }

    template <class ArrayOut>
    void forward_alltoallw_( ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto             scope = profile_scope_( "forward_alltoallw" );
        std::vector<int> forward_sdispls_w( line_comm_info_.num_procs );
        std::vector<int> forward_rdispls_w( line_comm_info_.num_procs );
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                forward_sdispls_w[p] =
                    detail::mpi_int_cast( bytes_from_elems_( forward_send_offsets_[p] ), "same_z forward sdispls_w" );
                forward_rdispls_w[p] = detail::mpi_int_cast(
                    bytes_from_elems_( forward_recv_pack_offset_elems_( p ) ), "same_z forward rdispls_w"
                );
            }
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                forward_sdispls_w.data(), forward_sendtypes_w_.data(),
                static_cast<void *>( host_recv_buffer_.raw_ptr() ), forward_recvcounts_.data(),
                forward_rdispls_w.data(), forward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr(), host_recv_buffer_.raw_ptr(), bytes_from_elems_( recv_buffer_elems_ ),
                runtime_api_t::host_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                unpack_forward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#else
        auto             scope = profile_scope_( "forward_alltoallw" );
        std::vector<int> forward_sdispls_w( line_comm_info_.num_procs );
        std::vector<int> forward_rdispls_w( line_comm_info_.num_procs );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                forward_sdispls_w[p] =
                    detail::mpi_int_cast( bytes_from_elems_( forward_send_offsets_[p] ), "same_z forward sdispls_w" );
                forward_rdispls_w[p] = detail::mpi_int_cast(
                    bytes_from_elems_( forward_recv_pack_offset_elems_( p ) ), "same_z forward rdispls_w"
                );
            }
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                forward_sdispls_w.data(), forward_sendtypes_w_.data(), static_cast<void *>( recv_buffer_.raw_ptr() ),
                forward_recvcounts_.data(), forward_rdispls_w.data(), forward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                unpack_forward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_p2p_waitall_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto      scope     = profile_scope_( "backward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        pack_backward_( in );

        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }

        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p ),
                    detail::mpi_int_cast( backward_chunk_elems_( p ), "same_z backward recv count" ), mpi_value_type_,
                    p, myid_i_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + backward_send_offsets_[p],
                    detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z backward send count" ),
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy_async(
                recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( myid_i_ ),
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
                bytes_from_elems_( backward_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_i_].stream()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p != myid_i_ )
                {
                    copy_host_backward_chunk_to_device_async_( p );
                }
            }
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                unpack_backward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#else
        auto      scope     = profile_scope_( "backward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        pack_backward_( in );

        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p ),
                    detail::mpi_int_cast( backward_chunk_elems_( p ), "same_z backward recv count" ), mpi_value_type_,
                    p, myid_i_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + backward_send_offsets_[p],
                    detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z backward send count" ),
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy_async(
                recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( myid_i_ ),
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
                bytes_from_elems_( backward_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_i_].stream()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                unpack_backward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_p2p_waitany_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto      scope     = profile_scope_( "backward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        pack_backward_( in );

        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }

        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p ),
                    detail::mpi_int_cast( backward_chunk_elems_( p ), "same_z backward recv count" ), mpi_value_type_,
                    p, myid_i_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + backward_send_offsets_[p],
                    detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z backward send count" ),
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy_async(
                recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( myid_i_ ),
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
                bytes_from_elems_( backward_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_i_].stream()
            );
        }

        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_backward_chunk_async_( myid_i_, out );
        }

        {
            int completed = 0;
            while ( completed < comm_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                {
                    auto phase = profile_scope_( "stage_recv_chunk_to_device" );
                    copy_host_backward_chunk_to_device_async_( p );
                }
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_backward_chunk_async_( p, out );
                }
                completed++;
            }
        }

        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#else
        auto      scope     = profile_scope_( "backward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        pack_backward_( in );

        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_i_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( p ),
                    detail::mpi_int_cast( backward_chunk_elems_( p ), "same_z backward recv count" ), mpi_value_type_,
                    p, myid_i_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + backward_send_offsets_[p],
                    detail::mpi_int_cast( input_dim_.size_x[p] * ny_local_ * nz_local_, "same_z backward send count" ),
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy_async(
                recv_buffer_.raw_ptr() + backward_recv_pack_offset_elems_( myid_i_ ),
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_i_],
                bytes_from_elems_( backward_chunk_elems_( myid_i_ ) ), runtime_api_t::device_to_device_kind(),
                streams_[myid_i_].stream()
            );
        }

        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_backward_chunk_async_( myid_i_, out );
        }

        {
            int completed = 0;
            while ( completed < comm_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "wait_recv_any" );
                    p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_backward_chunk_async_( p, out );
                }
                completed++;
            }
        }

        synchronize_streams_( "recv_copy_complete" );
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_alltoallv_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto scope = profile_scope_( "backward_alltoallv" );
        pack_backward_( in );

        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }

        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            line_comm_info_.alltoallv(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                backward_sdispls_.data(), mpi_value_type_, static_cast<void *>( host_recv_buffer_.raw_ptr() ),
                backward_recvcounts_.data(), backward_rdispls_.data(), mpi_value_type_
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr(), host_recv_buffer_.raw_ptr(), bytes_from_elems_( recv_buffer_elems_ ),
                runtime_api_t::host_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                unpack_backward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#else
        auto scope = profile_scope_( "backward_alltoallv" );
        pack_backward_( in );

        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            line_comm_info_.alltoallv(
                static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                backward_sdispls_.data(), mpi_value_type_, static_cast<void *>( recv_buffer_.raw_ptr() ),
                backward_recvcounts_.data(), backward_rdispls_.data(), mpi_value_type_
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                unpack_backward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void backward_alltoallw_( const ArrayIn &in, ArrayOut &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        auto scope = profile_scope_( "backward_alltoallw" );
        pack_backward_( in );

        std::vector<int> backward_sdispls_w( line_comm_info_.num_procs );
        std::vector<int> backward_rdispls_w( line_comm_info_.num_procs );
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                backward_sdispls_w[p] =
                    detail::mpi_int_cast( bytes_from_elems_( backward_send_offsets_[p] ), "same_z backward sdispls_w" );
                backward_rdispls_w[p] = detail::mpi_int_cast(
                    bytes_from_elems_( backward_recv_pack_offset_elems_( p ) ), "same_z backward rdispls_w"
                );
            }
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                backward_sdispls_w.data(), backward_sendtypes_w_.data(),
                static_cast<void *>( host_recv_buffer_.raw_ptr() ), backward_recvcounts_.data(),
                backward_rdispls_w.data(), backward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr(), host_recv_buffer_.raw_ptr(), bytes_from_elems_( recv_buffer_elems_ ),
                runtime_api_t::host_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                unpack_backward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
#else
        auto scope = profile_scope_( "backward_alltoallw" );
        pack_backward_( in );

        std::vector<int> backward_sdispls_w( line_comm_info_.num_procs );
        std::vector<int> backward_rdispls_w( line_comm_info_.num_procs );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                backward_sdispls_w[p] =
                    detail::mpi_int_cast( bytes_from_elems_( backward_send_offsets_[p] ), "same_z backward sdispls_w" );
                backward_rdispls_w[p] = detail::mpi_int_cast(
                    bytes_from_elems_( backward_recv_pack_offset_elems_( p ) ), "same_z backward rdispls_w"
                );
            }
        }

        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                backward_sdispls_w.data(), backward_sendtypes_w_.data(), static_cast<void *>( recv_buffer_.raw_ptr() ),
                backward_recvcounts_.data(), backward_rdispls_w.data(), backward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            {
                unpack_backward_chunk_async_( p, out );
            }
        }
        synchronize_streams_( "recv_copy_complete" );
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

    std::size_t nx_local_  = 0;
    std::size_t nx_global_ = 0;
    std::size_t ny_local_  = 0;
    std::size_t ny_global_ = 0;
    std::size_t nz_local_  = 0;

    partition_t input_dim_;
    partition_t output_dim_;

    std::unique_ptr<mpi_comm_t>                      line_comm_;
    scfd::communication::mpi_comm_info               line_comm_info_;
    contiguous_buf_t                                 send_buffer_;
    contiguous_buf_t                                 recv_buffer_;
    std::size_t                                      send_buffer_elems_      = 0;
    std::size_t                                      recv_buffer_elems_      = 0;
    host_buf_t                                       host_send_buffer_;
    host_buf_t                                       host_recv_buffer_;
    std::size_t                                      host_send_buffer_elems_ = 0;
    std::size_t                                      host_recv_buffer_elems_ = 0;
    bool                                             use_external_work_area_ = false;
    void                                            *external_work_area_     = nullptr;
    bool                                             use_external_host_work_area_ = false;
    void                                            *external_host_work_area_     = nullptr;
    std::vector<mpi_request_t>                       send_requests_;
    std::vector<mpi_request_t>                       recv_requests_;
    std::vector<typename runtime_api_t::stream_wrap> streams_;
    mpi_dtype_t                                      mpi_value_type_;
    bool                                             use_direct_backward_receive_ = false;
    bool                                             direct_p2p_cuda_aware_       = true;
    bool                                             use_p2p_send_thread_         = false;
    bool                                             use_p2p_byte_transfer_       = false;

    std::vector<std::size_t> forward_send_offsets_;
    std::vector<std::size_t> forward_recv_offsets_;
    std::vector<int>         forward_sendcounts_;
    std::vector<int>         forward_sdispls_;
    std::vector<int>         forward_recvcounts_;
    std::vector<int>         forward_rdispls_;
    std::vector<mpi_dtype_t> forward_sendtypes_w_;
    std::vector<mpi_dtype_t> forward_recvtypes_w_;

    std::vector<std::size_t> backward_send_offsets_;
    std::vector<std::size_t> backward_recv_offsets_;
    std::vector<int>         backward_sendcounts_;
    std::vector<int>         backward_sdispls_;
    std::vector<int>         backward_recvcounts_;
    std::vector<int>         backward_rdispls_;
    std::vector<mpi_dtype_t> backward_sendtypes_w_;
    std::vector<mpi_dtype_t> backward_recvtypes_w_;
    std::vector<mpi_dtype_t> direct_backward_recvtypes_;
    std::vector<std::vector<direct_byte_recv_chunk_t>> direct_backward_byte_recv_chunks_;
};

} // namespace fftm

#endif // __FFTM_DETAIL_MPI_TRANSPOSE_3D_H__
