#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_EGGER_OWNED_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_EGGER_OWNED_H__

#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/utils/safe_call.h>

#include "../fft_partitioning.h"
#include "../profiling.h"
#include "mpi_transpose_3d.h"

namespace fftm
{

enum class fftm_3d_large_count_p2p_transport
{
    hindexed,
    mpi_count,
    element_count,
    chunked
};

inline const char *fftm_3d_large_count_p2p_transport_name( fftm_3d_large_count_p2p_transport transport )
{
    switch ( transport )
    {
    case fftm_3d_large_count_p2p_transport::hindexed:
        return "hindexed";
    case fftm_3d_large_count_p2p_transport::mpi_count:
        return "mpi-count";
    case fftm_3d_large_count_p2p_transport::element_count:
        return "element-count";
    case fftm_3d_large_count_p2p_transport::chunked:
        return "chunked";
    }
    return "unknown";
}

namespace detail
{

template <class BaseFFT, class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI, class OptionalProfiler>
class mpi_transpose_3d_pencil_pencil_egger_owned
{
public:
    using base_fft_t        = BaseFFT;
    using value_type        = ValueType;
    using backend_t         = Backend;
    using memory_t          = typename backend_t::memory_type;
    using host_memory_t     = typename memory_t::host_memory_type;
    using runtime_api_t     = RuntimeAPI;
    using partition_t       = ::fftm::partition;
    using mpi_comm_t        = scfd::communication::mpi_comm;
    using contiguous_buf_t  = scfd::arrays::array_nd<value_type, 1, memory_t>;
    using host_buf_t        = scfd::arrays::array_nd<value_type, 1, host_memory_t>;
    using mpi_request_t     = scfd::communication::detail::mpi_request;
    using mpi_dtype_t       = scfd::communication::detail::mpi_data_type;
    using profiler_t        = ::fftm::fftm_profiler;
    using profiler_scope_t  = ::fftm::profile_scope<profiler_t>;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

    mpi_transpose_3d_pencil_pencil_egger_owned(
        base_fft_t &base_fft, const MPIComm &mpi, const Log &log, OptionalProfiler &profiler
    )
        : base_fft_( base_fft ), mpi_( mpi ), log_( log ), optional_profiler_( profiler )
    {
        static_assert(
            std::is_same<memory_t, typename runtime_api_t::memory_type>::value,
            "mpi_transpose_3d_pencil_pencil_egger_owned requires matching backend/runtime memory types"
        );
    }

    ~mpi_transpose_3d_pencil_pencil_egger_owned()
    {
        free_persistent_send_states_();
        free_forward_direct_recvtypes_();
        free_value_type_();
    }

    void set_profiler( profiler_t *profiler )
    {
        profiler_ = profiler;
    }

    void set_memory_profiler( memory_profiler_t *profiler, const std::string &prefix )
    {
        memory_profiler_       = profiler;
        memory_profile_prefix_ = prefix;
        update_memory_profile_();
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
        if ( use_p2p_byte_transfer_ != enabled )
            free_persistent_send_states_();
        use_p2p_byte_transfer_ = enabled;
    }

    void set_persistent_p2p_enabled( bool enabled )
    {
        if ( use_persistent_p2p_ != enabled )
            free_persistent_send_states_();
        use_persistent_p2p_ = enabled;
    }

    void set_ready_p2p_send_enabled( bool enabled )
    {
        use_ready_p2p_send_ = enabled;
    }

    void set_schedule_dump_enabled( bool enabled )
    {
        print_schedule_ = enabled;
    }

    void set_direct_forward_byte_receive_enabled( bool enabled )
    {
        use_direct_forward_byte_receive_ = enabled;
    }

    void set_large_count_p2p_transport( ::fftm::fftm_3d_large_count_p2p_transport transport )
    {
        large_count_p2p_transport_ = transport;
    }

    void set_pencil_layout_selector( int selector )
    {
        pencil_layout_selector_ = selector;
    }

    void set_egger_parity_enabled( bool enabled )
    {
        egger_parity_enabled_ = enabled;
        if ( !enabled )
            return;

        free_persistent_send_states_();
        use_direct_backward_receive_ = false;
        direct_p2p_cuda_aware_       = true;
        use_p2p_send_thread_         = false;
        use_p2p_byte_transfer_       = true;
        use_persistent_p2p_          = false;
        use_ready_p2p_send_          = false;
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
                "mpi_transpose_3d_pencil_pencil_egger_owned::set_external_work_area: external work area disabled"
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
                "mpi_transpose_3d_pencil_pencil_egger_owned::set_external_host_work_area: external host work area disabled"
            );
        }
        external_host_work_area_ = external_host_work_area;
        bind_external_host_work_area_();
        update_memory_profile_();
    }

    void init(
        mpi_transpose_3d_mode mode, const partition_t &half_input_dim, const partition_t &transpose1_dim,
        const partition_t &output_dim, int myid_i, int myid_j, bool use_persistent_p2p
    )
    {
        auto scope = profile_scope_( "mpi_transpose_3d_pencil_pencil_egger_owned::init" );
        if ( mode != mpi_transpose_3d_mode::p2p_waitall && mode != mpi_transpose_3d_mode::p2p_waitany )
        {
            throw std::logic_error( "owned Egger pencil-pencil pipeline currently supports only p2p modes" );
        }

        free_persistent_send_states_();
        free_forward_direct_recvtypes_();
        free_value_type_();
        mode_               = mode;
        half_input_dim_     = half_input_dim;
        transpose1_dim_     = transpose1_dim;
        output_dim_         = output_dim;
        myid_i_             = myid_i;
        myid_j_             = myid_j;
        use_persistent_p2p_ = use_persistent_p2p;

        validate_partitions_();
        validate_large_count_p2p_transport_();
        validate_egger_parity_();

        nx_local_  = half_input_dim_.size_x.at( myid_i_ );
        ny_input_local_  = half_input_dim_.size_y.at( myid_j_ );
        ny_output_local_ = output_dim_.size_y.at( myid_i_ );
        ny_global_       = transpose1_dim_.size_y.at( 0 );
        nz_global_ = half_input_dim_.size_z.at( 0 );
        nz_local_  = transpose1_dim_.size_z.at( myid_j_ );
        nx_global_ = output_dim_.size_x.at( 0 );

        row_comm_      = std::make_unique<mpi_comm_t>( std::move( mpi_.split( myid_i_, myid_j_ ) ) );
        row_comm_info_ = row_comm_->info();
        line_comm_     = std::make_unique<mpi_comm_t>( std::move( mpi_.split( myid_j_, myid_i_ ) ) );
        line_comm_info_ = line_comm_->info();

        if ( row_comm_info_.num_procs != static_cast<int>( half_input_dim_.size_y.size() ) )
        {
            throw std::logic_error( "owned Egger pencil row communicator size mismatch" );
        }
        if ( line_comm_info_.num_procs != static_cast<int>( transpose1_dim_.size_x.size() ) )
        {
            throw std::logic_error( "owned Egger pencil line communicator size mismatch" );
        }
        build_peer_orders_();

        send_buffer_elems_ = max_redistribution_buffer_elems_();
        recv_buffer_elems_ = max_redistribution_buffer_elems_();
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

        const int max_comm = std::max( row_comm_info_.num_procs, line_comm_info_.num_procs );
        row_send_requests_.assign( row_comm_info_.num_procs, mpi_request_t() );
        row_recv_requests_.assign( row_comm_info_.num_procs, mpi_request_t() );
        line_send_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        line_recv_requests_.assign( line_comm_info_.num_procs, mpi_request_t() );
        streams_.clear();
        streams_.reserve( max_comm );
        for ( int p = 0; p < max_comm; ++p )
        {
            streams_.emplace_back( true );
        }

        init_value_type_();
        init_forward_direct_recvtypes_();
        bind_external_work_area_();
        bind_external_host_work_area_();
        is_inited_ = true;
        log_transport_if_enabled_();
        dump_schedule_if_enabled_();
        update_memory_profile_();
    }

    template <
        class RealArray3, class Stage0Complex3, class Stage1Complex3, class XFastComplex3, class XFFTComplex3,
        class ComplexArray3>
	    void forward(
	        const RealArray3 &in, Stage0Complex3 &stage0, Stage1Complex3 &stage1, XFastComplex3 &stage1_xfast,
	        XFFTComplex3 &x_fft_stage, ComplexArray3 &out, mpi_transpose_3d_mode mode
	    )
    {
        ensure_inited_( mode );
        auto scope = optional_profiler_.scoped_tic(
	            egger_parity_enabled_ ? "fftm::forward_3d_pencil_pencil_egger_parity_pipeline"
	                                  : "fftm::forward_3d_pencil_pencil_egger_owned_pipeline"
	        );
	        exec_local_fft_( "egger_owned/forward_z_fft", "forward_z", in, stage0 );
	        {
	            auto phase = profile_scope_( "egger_owned/forward_first_redistribution" );
	            SCFD_SAFE_CALL( forward_first_( stage0, stage1 ) );
	        }
	        exec_local_fft_( "egger_owned/forward_y_fft", "forward_y", stage1, stage1_xfast );
	        {
	            auto phase = profile_scope_( "egger_owned/forward_second_redistribution" );
	            SCFD_SAFE_CALL( forward_second_xfast_( stage1_xfast, x_fft_stage ) );
	        }
	        exec_local_fft_( "egger_owned/forward_x_fft", "forward_x", x_fft_stage, out );
	    }

    template <
        class ComplexArray3, class XFFTComplex3, class XFastComplex3, class Stage1Complex3, class Stage0Complex3,
        class RealArray3>
    void backward(
        ComplexArray3 &in, XFFTComplex3 &x_fft_stage, XFastComplex3 &stage1_xfast, Stage1Complex3 &stage1,
        Stage0Complex3 &stage0, RealArray3 &out, mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
        auto scope = optional_profiler_.scoped_tic(
	            egger_parity_enabled_ ? "fftm::backward_3d_pencil_pencil_egger_parity_pipeline"
	                                  : "fftm::backward_3d_pencil_pencil_egger_owned_pipeline"
	        );
	        exec_local_fft_( "egger_owned/backward_x_fft", "inverse_x", in, x_fft_stage );
	        {
	            auto phase = profile_scope_( "egger_owned/backward_second_redistribution" );
	            SCFD_SAFE_CALL( backward_second_xfast_( x_fft_stage, stage1_xfast ) );
	        }
	        exec_local_fft_( "egger_owned/backward_y_fft", "inverse_y", stage1_xfast, stage1 );
	        {
	            auto phase = profile_scope_( "egger_owned/backward_first_redistribution" );
	            SCFD_SAFE_CALL( backward_first_( stage1, stage0 ) );
	        }
	        exec_local_fft_( "egger_owned/backward_z_fft", "inverse_z", stage0, out );
	    }

	private:
	    profiler_scope_t profile_scope_( const std::string &name )
	    {
	        return profiler_scope_t( profiler_, name );
	    }

	    template <class ArrayIn, class ArrayOut>
	    void exec_local_fft_(
	        const char *profile_name, const char *plan_name, const ArrayIn &in, ArrayOut &out
	    )
	    {
	        auto phase = optional_profiler_.scoped_tic( profile_name );
	        SCFD_SAFE_CALL( base_fft_.template exec<ArrayIn, ArrayOut>( plan_name, in, out ) );
	        synchronize_after_local_fft_if_needed_();
	    }

	    std::size_t bytes_from_elems_( std::size_t elems ) const
	    {
	        return elems * sizeof( value_type );
    }

    void validate_partitions_() const
    {
        if ( half_input_dim_.size_z.size() != 1 )
        {
            throw std::logic_error( "owned Egger pencil first input must keep Z undistributed" );
        }
        if ( transpose1_dim_.size_y.size() != 1 )
        {
            throw std::logic_error( "owned Egger pencil middle layout must keep Y undistributed" );
        }
        if ( output_dim_.size_x.size() != 1 )
        {
            throw std::logic_error( "owned Egger pencil output layout must keep X undistributed" );
        }
        if ( half_input_dim_.size_y.size() != transpose1_dim_.size_z.size() )
        {
            throw std::logic_error( "owned Egger pencil first redistribution partition mismatch" );
        }
        if ( transpose1_dim_.size_x.size() != output_dim_.size_y.size() )
        {
            throw std::logic_error( "owned Egger pencil second redistribution partition mismatch" );
        }
        if ( half_input_dim_.size_x != transpose1_dim_.size_x )
        {
            throw std::logic_error( "owned Egger pencil first redistribution must preserve X ownership" );
        }
        if ( transpose1_dim_.size_z != output_dim_.size_z )
        {
            throw std::logic_error( "owned Egger pencil second redistribution must preserve Z ownership" );
        }
    }

    void ensure_inited_( mpi_transpose_3d_mode mode ) const
    {
        if ( !is_inited_ )
        {
            throw std::logic_error( "owned Egger pencil pipeline is not initialized" );
        }
        if ( mode != mode_ )
        {
            throw std::logic_error( "owned Egger pencil pipeline mode mismatch" );
        }
        if ( use_external_work_area_ && external_work_area_ == nullptr )
        {
            throw std::logic_error( "owned Egger pencil pipeline external work area is not bound" );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( use_external_host_work_area_ && get_host_work_size_bytes() != 0 && external_host_work_area_ == nullptr )
        {
            throw std::logic_error( "owned Egger pencil pipeline external host work area is not bound" );
        }
#endif
        (void)use_p2p_send_thread_;
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
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
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

    void update_memory_profile_()
    {
        if ( memory_profiler_ == nullptr || memory_profile_prefix_.empty() )
        {
            return;
        }
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/send_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<typename memory_profiler_t::bytes_type>( send_buffer_.size() ) *
                                          static_cast<typename memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/recv_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<typename memory_profiler_t::bytes_type>( recv_buffer_.size() ) *
                                          static_cast<typename memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_send_buffer",
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            0
#else
            use_external_host_work_area_
                ? 0
                : static_cast<typename memory_profiler_t::bytes_type>( host_send_buffer_.size() ) *
                      static_cast<typename memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_recv_buffer",
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            0
#else
            use_external_host_work_area_
                ? 0
                : static_cast<typename memory_profiler_t::bytes_type>( host_recv_buffer_.size() ) *
                      static_cast<typename memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
        );
    }

    void init_value_type_()
    {
        mpi_value_type_ = scfd::communication::detail::type_contiguous(
            detail::mpi_int_cast( sizeof( value_type ), "owned Egger pencil value type extent" ),
            scfd::communication::detail::mpi_data_type_trait<char>::get()
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

    bool persistent_value_p2p_enabled_() const
    {
        return use_persistent_p2p_ && !cuda_aware_byte_p2p_enabled_();
    }

    bool persistent_byte_p2p_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return use_persistent_p2p_ && cuda_aware_byte_p2p_enabled_();
#else
        return false;
#endif
    }

    bool direct_backward_receive_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return direct_p2p_cuda_aware_ && use_direct_backward_receive_;
#else
        return false;
#endif
    }

	    bool direct_forward_value_receive_enabled_() const
	    {
	#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
	        return direct_p2p_cuda_aware_ && use_direct_backward_receive_ && !cuda_aware_byte_p2p_enabled_();
	#else
	        return false;
	#endif
	    }

    bool direct_egger_opt1_forward_byte_receive_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return use_direct_forward_byte_receive_ && direct_p2p_cuda_aware_ && egger_parity_byte_path_enabled_() &&
               pencil_layout_selector_ == 2;
#else
        return false;
#endif
    }

    bool forward_receive_lands_in_stage_() const
    {
        return direct_forward_value_receive_enabled_() || direct_egger_opt1_forward_byte_receive_enabled_();
    }

    bool egger_byte_sync_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        /*
         * Baseline path intended to match Egger's stable Peer2Peer_Sync
         * communication pattern: contiguous MPI_BYTE transfers, one logical
         * request per peer, no persistent requests, and no ready-send polling.
         */
        return cuda_aware_byte_p2p_enabled_() && !use_persistent_p2p_ && !use_ready_p2p_send_ &&
               !use_direct_backward_receive_;
#else
        return false;
#endif
    }

	    bool egger_parity_forward_waitany_enabled_() const
	    {
	        return egger_parity_enabled_ && egger_byte_sync_enabled_();
	    }

	    bool egger_parity_byte_path_enabled_() const
	    {
	        return egger_parity_enabled_ && egger_byte_sync_enabled_();
	    }

	    bool egger_parity_backward_waitall_enabled_() const
	    {
	        return egger_parity_byte_path_enabled_();
	    }

	    bool egger_parity_backward_ready_post_send_enabled_() const
	    {
	        return egger_parity_byte_path_enabled_();
	    }

    bool direct_egger_byte_backward_receive_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        /*
         * The backward owned-Egger byte path receives contiguous peer blocks
         * that are already laid out for the next local FFT stage.  Receiving
         * directly into that stage removes the extra recv_buffer_ -> output
         * copy without relying on MPI derived datatypes.
         */
        return egger_byte_sync_enabled_();
#else
        return false;
#endif
    }

    void validate_large_count_p2p_transport_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( large_count_p2p_transport_ == ::fftm::fftm_3d_large_count_p2p_transport::mpi_count &&
             !mpi_count_large_count_available_() )
        {
            throw std::logic_error(
                "owned Egger requested large-count MPI count transport, but this MPI header does not expose "
                "MPI_Isend_c/MPI_Irecv_c"
            );
        }
#endif
    }

    void validate_egger_parity_() const
    {
        if ( !egger_parity_enabled_ )
            return;
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "owned Egger parity requires device-aware MPI byte transfers" );
#else
        if ( !use_p2p_byte_transfer_ || use_persistent_p2p_ || use_ready_p2p_send_ ||
             use_direct_backward_receive_ || !direct_p2p_cuda_aware_ || use_p2p_send_thread_ )
        {
            throw std::logic_error( "owned Egger parity options were not normalized before initialization" );
        }
#endif
    }

	    bool backward_receive_lands_in_output_() const
	    {
	        return direct_backward_receive_enabled_() || direct_egger_byte_backward_receive_enabled_();
	    }

    std::size_t max_mpi_byte_chunk_bytes_() const
    {
        const std::size_t int_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
        return int_limit - ( int_limit % sizeof( value_type ) );
    }

    std::size_t large_mpi_byte_datatype_block_bytes_() const
    {
        return static_cast<std::size_t>( 1 ) << 30;
    }

    bool large_count_byte_requests_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return cuda_aware_byte_p2p_enabled_() && !persistent_byte_p2p_enabled_() &&
               large_count_p2p_transport_ != ::fftm::fftm_3d_large_count_p2p_transport::chunked;
#else
        return false;
#endif
    }

    bool mpi_count_large_count_available_() const
    {
#if defined( MPI_VERSION ) && MPI_VERSION >= 4
        return true;
#else
        return false;
#endif
    }

    bool mpi_count_large_count_transport_enabled_() const
    {
#if defined( MPI_VERSION ) && MPI_VERSION >= 4
        return large_count_byte_requests_enabled_() &&
               large_count_p2p_transport_ == ::fftm::fftm_3d_large_count_p2p_transport::mpi_count;
#else
        return false;
#endif
    }

    bool element_count_large_count_transport_enabled_() const
    {
        return large_count_byte_requests_enabled_() &&
               large_count_p2p_transport_ == ::fftm::fftm_3d_large_count_p2p_transport::element_count;
    }

    bool can_use_element_count_transport_( std::size_t bytes ) const
    {
        if ( !element_count_large_count_transport_enabled_() || bytes == 0 )
            return false;
        if ( bytes % sizeof( value_type ) != 0 )
            return false;
        const std::size_t elems     = bytes / sizeof( value_type );
        const std::size_t int_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
        return elems <= int_limit;
    }

    MPI_Count mpi_count_cast_( std::size_t count, const char *context ) const
    {
        const MPI_Count max_count = ( std::numeric_limits<MPI_Count>::max )();
        if ( count > static_cast<std::size_t>( max_count ) )
            throw std::overflow_error( std::string( context ) + " exceeds MPI_Count" );
        return static_cast<MPI_Count>( count );
    }

    bool peer_paired_large_count_byte_schedule_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return large_count_byte_requests_enabled_() && egger_parity_byte_path_enabled_();
#else
        return false;
#endif
    }

    int mpi_byte_chunk_count_( std::size_t bytes ) const
    {
        const std::size_t max_bytes = max_mpi_byte_chunk_bytes_();
        if ( bytes == 0 )
            return 0;
        return detail::mpi_int_cast( ( bytes + max_bytes - 1 ) / max_bytes, "owned Egger byte p2p chunk count" );
    }

    int mpi_byte_request_count_( std::size_t bytes ) const
    {
        if ( bytes == 0 )
            return 0;
        if ( large_count_byte_requests_enabled_() )
            return 1;
        return mpi_byte_chunk_count_( bytes );
    }

    int large_byte_datatype_block_count_( std::size_t bytes ) const
    {
        const std::size_t int_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
        if ( !large_count_byte_requests_enabled_() || bytes <= int_limit ||
             large_count_p2p_transport_ != ::fftm::fftm_3d_large_count_p2p_transport::hindexed )
            return 0;
        const std::size_t block_bytes = large_mpi_byte_datatype_block_bytes_();
        return detail::mpi_int_cast(
            ( bytes + block_bytes - 1 ) / block_bytes, "owned Egger large byte datatype block count"
        );
    }

    int element_count_request_count_( std::size_t bytes ) const
    {
        return can_use_element_count_transport_( bytes ) ? 1 : 0;
    }

    int element_count_fallback_count_( std::size_t bytes ) const
    {
        return element_count_large_count_transport_enabled_() && bytes != 0 && !can_use_element_count_transport_( bytes )
                   ? 1
                   : 0;
    }

    void free_persistent_send_states_()
    {
        detail::free_persistent_send_requests( persistent_first_forward_send_ );
        detail::free_persistent_send_requests( persistent_second_forward_send_ );
        detail::free_persistent_send_requests( persistent_second_backward_send_ );
        detail::free_persistent_send_requests( persistent_first_backward_send_ );
        detail::free_persistent_recv_requests( persistent_first_forward_recv_ );
        detail::free_persistent_recv_requests( persistent_second_forward_recv_ );
        detail::free_persistent_recv_requests( persistent_second_backward_recv_ );
        detail::free_persistent_recv_requests( persistent_first_backward_recv_ );
    }

    void profile_chunk_count_( const char *label, int send_requests, int recv_requests )
    {
        {
            std::ostringstream ss;
            ss << label << "/chunk_count/send=" << send_requests << "/recv=" << recv_requests;
            auto phase = profile_scope_( ss.str() );
        }
        {
            std::ostringstream ss;
            ss << label << "/byte_requests/send=" << send_requests << "/recv=" << recv_requests;
            auto phase = profile_scope_( ss.str() );
        }
    }

	    void synchronize_before_direct_byte_sends_( const char *label )
	    {
	#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
	        if ( cuda_aware_byte_p2p_enabled_() && !egger_parity_byte_path_enabled_() )
	        {
	            auto phase = profile_scope_( label );
	            runtime_api_t::device_synchronize();
	        }
#else
        (void)label;
	#endif
	    }

	    void synchronize_after_local_fft_if_needed_()
	    {
	#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
	        if ( egger_parity_byte_path_enabled_() )
	            runtime_api_t::device_synchronize();
	#endif
	    }

    template <class ByteFunc, class ElemFunc>
	    void dump_stage_schedule_(
	        const char *stage, const char *comm_name, const std::vector<int> &order, ByteFunc byte_func,
	        ElemFunc elem_func
	    )
    {
        if ( !print_schedule_ )
            return;
        std::size_t total_bytes = 0;
        std::size_t max_bytes   = 0;
        int         requests    = 0;
        int         legacy_chunks = 0;
        int         large_blocks = 0;
        int         element_count_requests  = 0;
        int         element_count_fallbacks = 0;
        for ( const int p : order )
        {
            const std::size_t bytes = byte_func( p );
            total_bytes += bytes;
            max_bytes = std::max( max_bytes, bytes );
            requests += mpi_byte_request_count_( bytes );
            legacy_chunks += mpi_byte_chunk_count_( bytes );
            large_blocks += large_byte_datatype_block_count_( bytes );
            element_count_requests += element_count_request_count_( bytes );
            element_count_fallbacks += element_count_fallback_count_( bytes );
        }

        std::ostringstream ss;
        ss << "owned_egger_schedule stage=" << stage << " comm=" << comm_name << " rank_i=" << myid_i_
           << " rank_j=" << myid_j_ << " peers=" << order.size() << " total_bytes=" << total_bytes
	           << " max_peer_bytes=" << max_bytes << " chunk_bytes=" << max_mpi_byte_chunk_bytes_()
	           << " requests=" << requests << " legacy_chunks=" << legacy_chunks
               << " large_type_blocks=" << large_blocks << " mode=" << mpi_transpose_3d_mode_name( mode_ )
	           << " byte=" << ( cuda_aware_byte_p2p_enabled_() ? 1 : 0 )
	           << " persistent=" << ( persistent_byte_p2p_enabled_() ? 1 : 0 )
	           << " egger_parity=" << ( egger_parity_enabled_ ? 1 : 0 )
		           << " large_count_transport=" << ::fftm::fftm_3d_large_count_p2p_transport_name( large_count_p2p_transport_ )
               << " element_count_requests=" << element_count_requests
               << " element_count_fallbacks=" << element_count_fallbacks
	               << " forward_direct_byte_receive=" << ( direct_egger_opt1_forward_byte_receive_enabled_() ? 1 : 0 )
		           << " mpi_count_available=" << ( mpi_count_large_count_available_() ? 1 : 0 );
        log_.info( ss.str() );

        for ( const int p : order )
        {
            const std::size_t elems = elem_func( p );
            const std::size_t bytes = byte_func( p );
            std::ostringstream peer_ss;
	            peer_ss << "owned_egger_schedule_peer stage=" << stage << " peer=" << p << " elems=" << elems
	                    << " bytes=" << bytes << " requests=" << mpi_byte_request_count_( bytes )
	                    << " legacy_chunks=" << mpi_byte_chunk_count_( bytes )
	                    << " large_type_blocks=" << large_byte_datatype_block_count_( bytes )
                        << " element_count=" << element_count_request_count_( bytes )
                        << " element_count_fallback=" << element_count_fallback_count_( bytes );
	            log_.info( peer_ss.str() );
	        }
    }

	    void dump_schedule_if_enabled_()
    {
        dump_stage_schedule_(
            "first_forward_send", "row", row_comm_order_,
            [this]( int p ) { return bytes_from_elems_( first_forward_send_elems_( p ) ); },
            [this]( int p ) { return first_forward_send_elems_( p ); }
        );
        dump_stage_schedule_(
            "first_forward_recv", "row", row_comm_order_,
            [this]( int p ) { return bytes_from_elems_( first_forward_recv_elems_( p ) ); },
            [this]( int p ) { return first_forward_recv_elems_( p ); }
        );
        dump_stage_schedule_(
            "second_forward_send", "line", line_comm_order_,
            [this]( int p ) { return bytes_from_elems_( second_forward_send_elems_( p ) ); },
            [this]( int p ) { return second_forward_send_elems_( p ); }
        );
        dump_stage_schedule_(
            "second_forward_recv", "line", line_comm_order_,
            [this]( int p ) { return bytes_from_elems_( second_forward_recv_elems_( p ) ); },
            [this]( int p ) { return second_forward_recv_elems_( p ); }
        );
        dump_stage_schedule_(
            "second_backward_send", "line", line_comm_order_,
            [this]( int p ) { return bytes_from_elems_( second_backward_send_elems_( p ) ); },
            [this]( int p ) { return second_backward_send_elems_( p ); }
        );
        dump_stage_schedule_(
            "second_backward_recv", "line", line_comm_order_,
            [this]( int p ) { return bytes_from_elems_( second_backward_recv_elems_( p ) ); },
            [this]( int p ) { return second_backward_recv_elems_( p ); }
        );
        dump_stage_schedule_(
            "first_backward_send", "row", row_comm_order_,
            [this]( int p ) { return bytes_from_elems_( first_backward_send_elems_( p ) ); },
            [this]( int p ) { return first_backward_send_elems_( p ); }
        );
        dump_stage_schedule_(
            "first_backward_recv", "row", row_comm_order_,
            [this]( int p ) { return bytes_from_elems_( first_backward_recv_elems_( p ) ); },
            [this]( int p ) { return first_backward_recv_elems_( p ); }
        );
    }

    void log_transport_if_enabled_()
    {
        if ( !egger_parity_enabled_ && !print_schedule_ )
            return;

        std::ostringstream ss;
        ss << "owned_egger_transport rank_i=" << myid_i_ << " rank_j=" << myid_j_
           << " mode=" << mpi_transpose_3d_mode_name( mode_ )
           << " egger_parity=" << ( egger_parity_enabled_ ? 1 : 0 )
           << " byte=" << ( cuda_aware_byte_p2p_enabled_() ? 1 : 0 )
           << " persistent=" << ( use_persistent_p2p_ ? 1 : 0 )
           << " ready_send=" << ( use_ready_p2p_send_ ? 1 : 0 )
           << " send_thread=" << ( use_p2p_send_thread_ ? 1 : 0 )
	           << " direct_backward_receive=" << ( use_direct_backward_receive_ ? 1 : 0 )
	           << " forward_wait=" << ( egger_parity_forward_waitany_enabled_() ? "waitany" :
	                                    mode_ == mpi_transpose_3d_mode::p2p_waitall ? "waitall" : "waitany" )
	           << " backward_wait=" << ( egger_parity_backward_waitall_enabled_() ||
	                                             mode_ == mpi_transpose_3d_mode::p2p_waitall
	                                         ? "waitall"
	                                         : "waitany" )
	           << " backward_send=" << ( egger_parity_backward_ready_post_send_enabled_() ? "pack_ready_post"
	                                    : egger_byte_sync_enabled_()                    ? "pack_sync"
	                                                                                   : "configured" )
	           << " row_peers=" << row_comm_order_.size() << " line_peers=" << line_comm_order_.size()
	           << " layout_selector=" << pencil_layout_selector_
		               << " large_count_transport=" << ::fftm::fftm_3d_large_count_p2p_transport_name( large_count_p2p_transport_ )
                   << " element_count_value_bytes=" << sizeof( value_type )
	                   << " forward_direct_byte_receive=" << ( direct_egger_opt1_forward_byte_receive_enabled_() ? 1 : 0 )
		               << " mpi_count_available=" << ( mpi_count_large_count_available_() ? 1 : 0 );
	        log_.info( ss.str() );
    }

    void free_forward_direct_recvtypes_()
    {
        for ( auto &datatype : first_forward_direct_recvtypes_ )
            scfd::communication::detail::type_free( datatype );
        for ( auto &datatype : second_forward_direct_recvtypes_ )
            scfd::communication::detail::type_free( datatype );
        first_forward_direct_recvtypes_.clear();
        second_forward_direct_recvtypes_.clear();
    }

    void init_forward_direct_recvtypes_()
    {
        free_forward_direct_recvtypes_();
	        if ( !forward_receive_lands_in_stage_() )
	            return;

        first_forward_direct_recvtypes_.assign( row_comm_info_.num_procs, mpi_dtype_t() );
        for ( const int p : row_comm_order_ )
        {
            first_forward_direct_recvtypes_[p] = scfd::communication::detail::type_vector(
                detail::mpi_int_cast( nx_local_ * nz_local_, "owned first direct forward recv type count" ),
                detail::mpi_int_cast(
                    half_input_dim_.size_y[p], "owned first direct forward recv type blocklength"
                ),
                detail::mpi_int_cast( ny_global_, "owned first direct forward recv type stride" ), mpi_value_type_
            );
            scfd::communication::detail::type_commit( first_forward_direct_recvtypes_[p] );
        }

        second_forward_direct_recvtypes_.assign( line_comm_info_.num_procs, mpi_dtype_t() );
        for ( const int p : line_comm_order_ )
        {
            second_forward_direct_recvtypes_[p] = scfd::communication::detail::type_vector(
                detail::mpi_int_cast(
                    nz_local_ * ny_output_local_, "owned second direct forward recv type count"
                ),
                detail::mpi_int_cast(
                    transpose1_dim_.size_x[p], "owned second direct forward recv type blocklength"
                ),
                detail::mpi_int_cast( nx_global_, "owned second direct forward recv type stride" ), mpi_value_type_
            );
            scfd::communication::detail::type_commit( second_forward_direct_recvtypes_[p] );
        }
    }

    mpi_dtype_t make_large_byte_datatype_( std::size_t bytes, const char *profile_label )
    {
        auto phase = profile_scope_( profile_label );
        const std::size_t block_bytes = large_mpi_byte_datatype_block_bytes_();
        const std::size_t blocks      = ( bytes + block_bytes - 1 ) / block_bytes;
        if ( blocks > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
        {
            throw std::runtime_error( "owned Egger large byte datatype block count exceeds MPI int range" );
        }

        std::vector<int>      block_lengths( blocks );
        std::vector<MPI_Aint> displacements( blocks );
        std::size_t           offset = 0;
        for ( std::size_t i = 0; i < blocks; ++i )
        {
            const std::size_t block = std::min( block_bytes, bytes - offset );
            block_lengths[i]       = detail::mpi_int_cast( block, "owned Egger large byte datatype block length" );
            displacements[i]       = static_cast<MPI_Aint>( offset );
            offset += block;
        }

        mpi_dtype_t byte_type = scfd::communication::detail::mpi_data_type_trait<char>::get();
        mpi_dtype_t large_type;
        SCFD_MPI_SAFE_CALL( MPI_Type_create_hindexed(
            detail::mpi_int_cast( blocks, "owned Egger large byte datatype block count" ), block_lengths.data(),
            displacements.data(), byte_type.native(), large_type.native_ptr()
        ) );
        scfd::communication::detail::type_commit( large_type );
        return large_type;
    }

    template <class CommInfo>
    void post_byte_irecv_(
        CommInfo &comm_info, void *ptr, std::size_t bytes, int source, int tag, std::vector<mpi_request_t> &requests,
        std::vector<int> *request_peers = nullptr, std::vector<int> *remaining_by_peer = nullptr,
        detail::persistent_recv_requests<mpi_request_t> *persistent_state = nullptr, int *persistent_index = nullptr
    )
    {
        char             *base      = static_cast<char *>( ptr );
        const mpi_dtype_t byte_type = scfd::communication::detail::mpi_data_type_trait<char>::get();
        if ( bytes == 0 )
            return;
        if ( large_count_byte_requests_enabled_() )
        {
            requests.push_back( mpi_request_t() );
            const std::size_t int_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
            if ( can_use_element_count_transport_( bytes ) )
            {
                const std::size_t elems = bytes / sizeof( value_type );
                auto              phase = profile_scope_( "element_count/recv_post_value_count" );
                comm_info.irecv(
                    static_cast<value_type *>( ptr ),
                    detail::mpi_int_cast( elems, "owned Egger element-count p2p recv" ), mpi_value_type_, source, tag,
                    requests.back()
                );
            }
            else if ( bytes <= int_limit )
            {
                auto phase = profile_scope_( "large_byte/recv_post_int_count" );
                comm_info.irecv(
                    base, detail::mpi_int_cast( bytes, "owned Egger byte p2p recv count" ), byte_type, source, tag,
                    requests.back()
                );
            }
            else
            {
#if defined( MPI_VERSION ) && MPI_VERSION >= 4
                if ( mpi_count_large_count_transport_enabled_() )
                {
                    auto phase = profile_scope_( "large_byte/recv_post_mpi_count" );
                    SCFD_MPI_SAFE_CALL( MPI_Irecv_c(
                        base, mpi_count_cast_( bytes, "owned Egger MPI count byte p2p recv" ), byte_type.native(),
                        source, tag, comm_info.comm, requests.back().native_ptr()
                    ) );
                }
                else
#endif
                {
                mpi_dtype_t large_type = make_large_byte_datatype_( bytes, "large_byte/recv_type_create" );
                {
                    auto phase = profile_scope_( "large_byte/recv_post_large_type" );
                    comm_info.irecv( base, 1, large_type, source, tag, requests.back() );
                }
                scfd::communication::detail::type_free( large_type );
                }
            }
            if ( request_peers != nullptr )
                request_peers->push_back( source );
            if ( remaining_by_peer != nullptr )
                ( *remaining_by_peer )[source] += 1;
            return;
        }
        const std::size_t max_bytes = max_mpi_byte_chunk_bytes_();
        int               chunks    = 0;
        for ( std::size_t offset = 0; offset < bytes; offset += max_bytes )
        {
            const std::size_t this_bytes = std::min( max_bytes, bytes - offset );
            const int this_count = detail::mpi_int_cast( this_bytes, "owned Egger byte p2p recv count" );
            if ( persistent_state != nullptr )
            {
                if ( persistent_index == nullptr )
                    throw std::logic_error( "owned Egger persistent byte p2p recv index is null" );
                detail::start_persistent_recv(
                    *persistent_state, *persistent_index, base + offset, this_count, byte_type.native(), source, tag,
                    comm_info.comm
                );
                requests.push_back( persistent_state->requests[*persistent_index] );
                ++( *persistent_index );
            }
            else
            {
                requests.push_back( mpi_request_t() );
                comm_info.irecv( base + offset, this_count, byte_type, source, tag, requests.back() );
            }
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

    template <class CommInfo>
    void post_value_irecv_(
        CommInfo &comm_info, void *ptr, std::size_t elems, int source, int tag, mpi_request_t &active_request,
        detail::persistent_recv_requests<mpi_request_t> &persistent_state, int persistent_index
    )
    {
        const int count = detail::mpi_int_cast( elems, "owned Egger value p2p recv" );
        if ( use_persistent_p2p_ )
        {
            detail::start_persistent_recv(
                persistent_state, persistent_index, ptr, count, mpi_value_type_.native(), source, tag, comm_info.comm
            );
            active_request = persistent_state.requests[persistent_index];
        }
        else
        {
            comm_info.irecv( ptr, count, mpi_value_type_, source, tag, active_request );
        }
    }

	    template <class CommInfo>
	    void post_typed_irecv_(
	        CommInfo &comm_info, void *ptr, int count, const mpi_dtype_t &datatype, int source, int tag,
	        mpi_request_t &active_request, detail::persistent_recv_requests<mpi_request_t> &persistent_state,
	        int persistent_index
    )
    {
        if ( use_persistent_p2p_ )
        {
            detail::start_persistent_recv(
                persistent_state, persistent_index, ptr, count, datatype.native(), source, tag, comm_info.comm
            );
            active_request = persistent_state.requests[persistent_index];
        }
        else
        {
	            comm_info.irecv( ptr, count, datatype, source, tag, active_request );
	        }
	    }

    template <class CommInfo>
    void post_direct_forward_byte_irecv_(
        CommInfo &comm_info, void *ptr, const mpi_dtype_t &datatype, int source, int tag,
        std::vector<mpi_request_t> &requests, std::vector<int> *request_peers = nullptr,
        std::vector<int> *remaining_by_peer = nullptr
    )
    {
        requests.push_back( mpi_request_t() );
        {
            auto phase = profile_scope_( "direct_byte/forward_recv_post_strided_type" );
            comm_info.irecv( ptr, 1, datatype, source, tag, requests.back() );
        }
        if ( request_peers != nullptr )
            request_peers->push_back( source );
        if ( remaining_by_peer != nullptr )
            ( *remaining_by_peer )[source] += 1;
    }

	    template <class CommInfo>
	    void post_byte_isend_(
	        CommInfo &comm_info, const void *ptr, std::size_t bytes, int dest, int tag,
        std::vector<mpi_request_t> &requests,
        detail::persistent_send_requests<mpi_request_t> *persistent_state = nullptr, int *persistent_index = nullptr
    )
    {
        const char       *base      = static_cast<const char *>( ptr );
        const mpi_dtype_t byte_type = scfd::communication::detail::mpi_data_type_trait<char>::get();
        if ( bytes == 0 )
            return;
        if ( large_count_byte_requests_enabled_() )
        {
            requests.push_back( mpi_request_t() );
            const std::size_t int_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
            if ( can_use_element_count_transport_( bytes ) )
            {
                const std::size_t elems = bytes / sizeof( value_type );
                auto              phase = profile_scope_( "element_count/send_post_value_count" );
                comm_info.isend(
                    static_cast<const value_type *>( ptr ),
                    detail::mpi_int_cast( elems, "owned Egger element-count p2p send" ), mpi_value_type_, dest, tag,
                    requests.back()
                );
            }
            else if ( bytes <= int_limit )
            {
                auto phase = profile_scope_( "large_byte/send_post_int_count" );
                comm_info.isend(
                    base, detail::mpi_int_cast( bytes, "owned Egger byte p2p send count" ), byte_type, dest, tag,
                    requests.back()
                );
            }
            else
            {
#if defined( MPI_VERSION ) && MPI_VERSION >= 4
                if ( mpi_count_large_count_transport_enabled_() )
                {
                    auto phase = profile_scope_( "large_byte/send_post_mpi_count" );
                    SCFD_MPI_SAFE_CALL( MPI_Isend_c(
                        base, mpi_count_cast_( bytes, "owned Egger MPI count byte p2p send" ), byte_type.native(), dest,
                        tag, comm_info.comm, requests.back().native_ptr()
                    ) );
                }
                else
#endif
                {
                mpi_dtype_t large_type = make_large_byte_datatype_( bytes, "large_byte/send_type_create" );
                {
                    auto phase = profile_scope_( "large_byte/send_post_large_type" );
                    comm_info.isend( base, 1, large_type, dest, tag, requests.back() );
                }
                scfd::communication::detail::type_free( large_type );
                }
            }
            return;
        }
        const std::size_t max_bytes = max_mpi_byte_chunk_bytes_();
        for ( std::size_t offset = 0; offset < bytes; offset += max_bytes )
        {
            const std::size_t this_bytes = std::min( max_bytes, bytes - offset );
            const int this_count = detail::mpi_int_cast( this_bytes, "owned Egger byte p2p send count" );
            if ( persistent_state != nullptr )
            {
                if ( persistent_index == nullptr )
                    throw std::logic_error( "owned Egger persistent byte p2p send index is null" );
                detail::start_persistent_send(
                    *persistent_state, *persistent_index, base + offset, this_count, byte_type.native(), dest, tag,
                    comm_info.comm
                );
                ++( *persistent_index );
            }
            else
            {
                requests.push_back( mpi_request_t() );
                comm_info.isend( base + offset, this_count, byte_type, dest, tag, requests.back() );
            }
        }
    }

    template <class CommInfo>
    void post_value_isend_(
        CommInfo &comm_info, const void *ptr, std::size_t elems, int dest, int tag, mpi_request_t &request,
        detail::persistent_send_requests<mpi_request_t> &persistent_state, int persistent_index
    )
    {
        const int count = detail::mpi_int_cast( elems, "owned Egger value p2p send" );
        if ( persistent_value_p2p_enabled_() )
        {
            detail::start_persistent_send(
                persistent_state, persistent_index, ptr, count, mpi_value_type_.native(), dest, tag, comm_info.comm
            );
        }
        else
        {
            comm_info.isend( ptr, count, mpi_value_type_, dest, tag, request );
        }
    }

    template <class CommInfo>
    void waitall_vector_( CommInfo &comm_info, std::vector<mpi_request_t> &requests, const char *what )
    {
        if ( requests.empty() )
        {
            return;
        }
        comm_info.waitall( detail::mpi_int_cast( requests.size(), what ), requests.data() );
    }

    template <class CommInfo>
    void wait_send_requests_(
        CommInfo &comm_info, std::vector<mpi_request_t> &byte_send_requests,
        detail::persistent_send_requests<mpi_request_t> &persistent_send_state,
        std::vector<mpi_request_t> &value_send_requests, int value_request_count, const char *byte_what,
        const char *persistent_what
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            if ( persistent_byte_p2p_enabled_() )
            {
                if ( !persistent_send_state.requests.empty() )
                {
                    comm_info.waitall(
                        detail::mpi_int_cast( persistent_send_state.requests.size(), persistent_what ),
                        persistent_send_state.requests.data()
                    );
                }
            }
            else
            {
                waitall_vector_( comm_info, byte_send_requests, byte_what );
            }
            return;
        }
#endif
        if ( persistent_value_p2p_enabled_() )
        {
            comm_info.waitall( value_request_count, persistent_send_state.requests.data() );
        }
        else
        {
            comm_info.waitall( value_request_count, value_send_requests.data() );
        }
    }

    void null_completed_request_( std::vector<mpi_request_t> &requests, int index )
    {
        if ( index >= 0 && index < static_cast<int>( requests.size() ) )
            requests[index] = mpi_request_t();
    }

    int first_forward_byte_recv_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : row_comm_order_ )
            total_chunks += mpi_byte_request_count_( bytes_from_elems_( first_forward_recv_elems_( p ) ) );
        return total_chunks;
    }

    int second_forward_byte_recv_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : line_comm_order_ )
            total_chunks += mpi_byte_request_count_( bytes_from_elems_( second_forward_recv_elems_( p ) ) );
        return total_chunks;
    }

    int second_backward_byte_recv_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : line_comm_order_ )
            total_chunks += mpi_byte_request_count_( bytes_from_elems_( second_backward_recv_elems_( p ) ) );
        return total_chunks;
    }

    int first_backward_byte_recv_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : row_comm_order_ )
            total_chunks += mpi_byte_request_count_( bytes_from_elems_( first_backward_recv_elems_( p ) ) );
        return total_chunks;
    }

    void synchronize_streams_( int nstreams, const char *label )
    {
        auto scope = profile_scope_( label );
        for ( int i = 0; i < nstreams; ++i )
        {
            runtime_api_t::stream_synchronize( streams_[i].stream() );
        }
    }

    void reset_row_requests_()
    {
        std::fill( row_send_requests_.begin(), row_send_requests_.end(), mpi_request_t() );
        std::fill( row_recv_requests_.begin(), row_recv_requests_.end(), mpi_request_t() );
    }

    void reset_line_requests_()
    {
        std::fill( line_send_requests_.begin(), line_send_requests_.end(), mpi_request_t() );
        std::fill( line_recv_requests_.begin(), line_recv_requests_.end(), mpi_request_t() );
    }

    void build_peer_orders_()
    {
        row_comm_order_.clear();
        row_comm_order_.reserve( std::max( 0, row_comm_info_.num_procs - 1 ) );
        for ( int step = 1; step < row_comm_info_.num_procs; ++step )
        {
            row_comm_order_.push_back( ( myid_j_ + step ) % row_comm_info_.num_procs );
        }
        if ( pencil_layout_selector_ == 2 && !egger_parity_enabled_ )
            std::reverse( row_comm_order_.begin(), row_comm_order_.end() );

        line_comm_order_.clear();
        line_comm_order_.reserve( std::max( 0, line_comm_info_.num_procs - 1 ) );
        for ( int step = 1; step < line_comm_info_.num_procs; ++step )
        {
            line_comm_order_.push_back( ( myid_i_ + step ) % line_comm_info_.num_procs );
        }
        if ( pencil_layout_selector_ == 2 && !egger_parity_enabled_ )
            std::reverse( line_comm_order_.begin(), line_comm_order_.end() );
    }

    std::size_t max_first_stage_send_elems_() const
    {
        return nx_local_ * ny_input_local_ * nz_global_;
    }

    std::size_t max_first_stage_recv_elems_() const
    {
        return nx_local_ * ny_global_ * nz_local_;
    }

    std::size_t max_second_stage_send_elems_() const
    {
        return nx_local_ * ny_global_ * nz_local_;
    }

    std::size_t max_second_stage_recv_elems_() const
    {
        return nx_global_ * ny_output_local_ * nz_local_;
    }

    std::size_t max_redistribution_buffer_elems_() const
    {
        return std::max(
            std::max( max_first_stage_send_elems_(), max_first_stage_recv_elems_() ),
            std::max( max_second_stage_send_elems_(), max_second_stage_recv_elems_() )
        );
    }

    std::size_t first_forward_send_elems_( int target_j ) const
    {
        return nx_local_ * ny_input_local_ * transpose1_dim_.size_z[target_j];
    }

    std::size_t first_forward_send_offset_( int target_j ) const
    {
        return transpose1_dim_.start_z[target_j] * nx_local_ * ny_input_local_;
    }

    std::size_t first_forward_recv_elems_( int source_j ) const
    {
        return nx_local_ * half_input_dim_.size_y[source_j] * nz_local_;
    }

    std::size_t first_forward_recv_offset_( int source_j ) const
    {
        return nx_local_ * nz_local_ * half_input_dim_.start_y[source_j];
    }

    std::size_t first_backward_send_elems_( int target_j ) const
    {
        return nx_local_ * half_input_dim_.size_y[target_j] * nz_local_;
    }

    std::size_t first_backward_send_offset_( int target_j ) const
    {
        return first_forward_recv_offset_( target_j );
    }

    std::size_t first_backward_recv_elems_( int source_j ) const
    {
        return nx_local_ * ny_input_local_ * transpose1_dim_.size_z[source_j];
    }

    std::size_t first_backward_recv_offset_( int source_j ) const
    {
        return first_forward_send_offset_( source_j );
    }

    std::size_t first_backward_packed_recv_offset_( int source_j ) const
    {
        return first_backward_recv_offset_( source_j );
    }

    std::size_t second_forward_send_elems_( int target_i ) const
    {
        return nx_local_ * output_dim_.size_y[target_i] * nz_local_;
    }

    std::size_t second_forward_send_offset_( int target_i ) const
    {
        return nx_local_ * nz_local_ * output_dim_.start_y[target_i];
    }

    std::size_t second_forward_recv_elems_( int source_i ) const
    {
        return transpose1_dim_.size_x[source_i] * ny_output_local_ * nz_local_;
    }

    std::size_t second_forward_recv_offset_( int source_i ) const
    {
        std::size_t offset = 0;
        for ( int p = 0; p < source_i; ++p )
        {
            offset += second_forward_recv_elems_( p );
        }
        return offset;
    }

    std::size_t second_backward_send_elems_( int target_i ) const
    {
        return transpose1_dim_.size_x[target_i] * ny_output_local_ * nz_local_;
    }

    std::size_t second_backward_send_offset_( int target_i ) const
    {
        return second_forward_recv_offset_( target_i );
    }

    std::size_t second_backward_recv_elems_( int source_i ) const
    {
        return nx_local_ * output_dim_.size_y[source_i] * nz_local_;
    }

    std::size_t second_backward_recv_offset_( int source_i ) const
    {
        return second_forward_send_offset_( source_i );
    }

    std::size_t second_backward_packed_recv_offset_( int source_i ) const
    {
        return second_backward_recv_offset_( source_i );
    }

    void copy_first_forward_chunk_to_stage1_async_(
        const value_type *src_ptr, std::size_t src_y_size, std::size_t dst_y_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, src_y_size * sizeof( value_type ), src_y_size, nx_local_ );
        params.dstPos = runtime_api_t::make_pos( dst_y_offset * sizeof( value_type ), 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, ny_global_ * sizeof( value_type ), ny_global_, nx_local_ );
        params.extent = runtime_api_t::make_extent( src_y_size * sizeof( value_type ), nx_local_, nz_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void pack_first_backward_chunk_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t packed_y_size, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
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
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_second_forward_chunk_to_xstage_async_(
        const value_type *src_ptr, std::size_t src_x_size, std::size_t dst_x_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, src_x_size * sizeof( value_type ), src_x_size, nz_local_ );
        params.dstPos = runtime_api_t::make_pos( dst_x_offset * sizeof( value_type ), 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, nx_global_ * sizeof( value_type ), nx_global_, nz_local_ );
        params.extent = runtime_api_t::make_extent( src_x_size * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void pack_second_backward_chunk_async_(
        const value_type *src_ptr, std::size_t src_x_offset, std::size_t packed_x_size, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( src_x_offset * sizeof( value_type ), 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nx_global_ * sizeof( value_type ), nx_global_, nz_local_ );
        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, packed_x_size * sizeof( value_type ), packed_x_size, nz_local_ );
        params.extent = runtime_api_t::make_extent( packed_x_size * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    template <class Stage0Complex3, class Stage1Complex3>
    void forward_first_( const Stage0Complex3 &in, Stage1Complex3 &out )
    {
        reset_row_requests_();
        if ( row_comm_info_.num_procs == 1 )
        {
            {
                auto phase = profile_scope_( "first/degenerate_self_copy" );
                copy_first_forward_self_( in, out );
            }
            synchronize_streams_( 1, "first/degenerate_copy_complete" );
            return;
        }
        if ( mode_ == mpi_transpose_3d_mode::p2p_waitall && !egger_parity_forward_waitany_enabled_() )
            forward_first_waitall_( in, out );
        else
            forward_first_waitany_( in, out );
    }

    template <class Stage0Complex3>
    void pack_first_forward_chunk_( int p, const Stage0Complex3 &in )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *dst  = send_buffer_.raw_ptr() + first_forward_send_offset_( p );
        auto        kind = runtime_api_t::device_to_device_kind();
#else
        value_type *dst  = host_send_buffer_.raw_ptr() + first_forward_send_offset_( p );
        auto        kind = runtime_api_t::device_to_host_kind();
#endif
        runtime_api_t::memcpy_async(
            dst, in.raw_ptr() + first_forward_send_offset_( p ), bytes_from_elems_( first_forward_send_elems_( p ) ),
            kind, streams_[p].stream()
        );
    }

    int first_forward_byte_send_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : row_comm_order_ )
            total_chunks += mpi_byte_request_count_( bytes_from_elems_( first_forward_send_elems_( p ) ) );
        return total_chunks;
    }

    void prepare_first_forward_send_state_()
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( persistent_byte_p2p_enabled_() )
            detail::reset_persistent_send_requests( persistent_first_forward_send_, first_forward_byte_send_chunks_() );
        else
#endif
        if ( persistent_value_p2p_enabled_() )
            detail::reset_persistent_send_requests( persistent_first_forward_send_, row_comm_info_.num_procs );
    }

    void forward_first_post_send_(
        int p, std::vector<mpi_request_t> &byte_send_requests, int &persistent_byte_index
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const value_type *send_ptr = send_buffer_.raw_ptr() + first_forward_send_offset_( p );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            post_byte_isend_(
                row_comm_info_, send_ptr, bytes_from_elems_( first_forward_send_elems_( p ) ), p, myid_j_,
                byte_send_requests, persistent_byte_p2p_enabled_() ? &persistent_first_forward_send_ : nullptr,
                &persistent_byte_index
            );
        }
        else
        {
            post_value_isend_(
                row_comm_info_, send_ptr, first_forward_send_elems_( p ), p, myid_j_, row_send_requests_[p],
                persistent_first_forward_send_, p
            );
        }
#else
        post_value_isend_(
            row_comm_info_, host_send_buffer_.raw_ptr() + first_forward_send_offset_( p ),
            first_forward_send_elems_( p ), p, myid_j_, row_send_requests_[p], persistent_first_forward_send_, p
        );
#endif
    }

    template <class Stage0Complex3>
	    void forward_first_pack_ready_send_( const Stage0Complex3 &in, std::vector<mpi_request_t> &byte_send_requests )
	    {
	        prepare_first_forward_send_state_();
	        profile_chunk_count_( "first", first_forward_byte_send_chunks_(), first_forward_byte_recv_chunks_() );
	        int persistent_byte_index = 0;
        for ( const int p : row_comm_order_ )
        {
            {
                auto phase = profile_scope_( "first/pack_peer" );
                pack_first_forward_chunk_( p, in );
            }
        }

        std::vector<char> posted( row_comm_info_.num_procs, 0 );
        int remaining = static_cast<int>( row_comm_order_.size() );
        while ( remaining > 0 )
        {
            std::vector<int> ready_peers;
            {
                auto phase = profile_scope_( "first/wait_pack_ready" );
                while ( ready_peers.empty() )
                {
                    for ( const int p : row_comm_order_ )
                    {
                        if ( posted[p] )
                            continue;
                        if ( runtime_api_t::stream_ready( streams_[p].stream() ) )
                            ready_peers.push_back( p );
                    }
                }
            }
            for ( const int p : ready_peers )
            {
                auto phase = profile_scope_( "first/mpi_post_send" );
                forward_first_post_send_( p, byte_send_requests, persistent_byte_index );
                posted[p] = 1;
                --remaining;
            }
        }
    }

    template <class Stage0Complex3>
	    void forward_first_pack_sync_send_( const Stage0Complex3 &in, std::vector<mpi_request_t> &byte_send_requests )
	    {
	        prepare_first_forward_send_state_();
	        profile_chunk_count_( "first", first_forward_byte_send_chunks_(), first_forward_byte_recv_chunks_() );
	        int persistent_byte_index = 0;
        for ( const int p : row_comm_order_ )
        {
            {
                auto phase = profile_scope_( "first/pack_peer" );
                pack_first_forward_chunk_( p, in );
            }
            {
                auto phase = profile_scope_( "first/pre_send_stream_sync" );
                runtime_api_t::stream_synchronize( streams_[p].stream() );
            }
            {
                auto phase = profile_scope_( "first/mpi_post_send" );
                forward_first_post_send_( p, byte_send_requests, persistent_byte_index );
            }
        }
    }

    void prepare_first_forward_recv_state_()
    {
        const int row_size = row_comm_info_.num_procs;
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( persistent_byte_p2p_enabled_() )
            detail::reset_persistent_recv_requests( persistent_first_forward_recv_, first_forward_byte_recv_chunks_() );
        else if ( use_persistent_p2p_ )
            detail::reset_persistent_recv_requests( persistent_first_forward_recv_, row_size );
#else
        if ( use_persistent_p2p_ )
            detail::reset_persistent_recv_requests( persistent_first_forward_recv_, row_size );
#endif
    }

    template <class Stage1Complex3>
    void forward_first_post_recvs_(
        Stage1Complex3 &out, std::vector<mpi_request_t> &byte_recv_requests,
        std::vector<int> *byte_recv_peers = nullptr, std::vector<int> *byte_recv_remaining = nullptr
    )
    {
        prepare_first_forward_recv_state_();
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        int persistent_byte_recv_index = 0;
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state =
            persistent_byte_p2p_enabled_() ? &persistent_first_forward_recv_ : nullptr;
#endif
        for ( const int p : row_comm_order_ )
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
	            value_type *recv_ptr = forward_receive_lands_in_stage_()
	                                      ? out.raw_ptr() + half_input_dim_.start_y[p]
	                                      : recv_buffer_.raw_ptr() + first_forward_recv_offset_( p );
	            if ( cuda_aware_byte_p2p_enabled_() )
	            {
                    if ( direct_egger_opt1_forward_byte_receive_enabled_() )
                    {
                        post_direct_forward_byte_irecv_(
                            row_comm_info_, recv_ptr, first_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                            byte_recv_peers, byte_recv_remaining
                        );
                    }
                    else
                    {
	                    post_byte_irecv_(
	                        row_comm_info_, recv_ptr, bytes_from_elems_( first_forward_recv_elems_( p ) ), p, p,
	                        byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
	                        &persistent_byte_recv_index
	                    );
                    }
	            }
	            else
	            {
                if ( direct_forward_value_receive_enabled_() )
                    post_typed_irecv_(
                        row_comm_info_, recv_ptr, 1, first_forward_direct_recvtypes_[p], p, p,
                        row_recv_requests_[p], persistent_first_forward_recv_, p
                    );
                else
                    post_value_irecv_(
                        row_comm_info_, recv_ptr, first_forward_recv_elems_( p ), p, p, row_recv_requests_[p],
                        persistent_first_forward_recv_, p
                    );
            }
#else
            post_value_irecv_(
                row_comm_info_, host_recv_buffer_.raw_ptr() + first_forward_recv_offset_( p ),
                first_forward_recv_elems_( p ), p, p, row_recv_requests_[p], persistent_first_forward_recv_, p
            );
#endif
        }
    }

	    template <class Stage0Complex3, class Stage1Complex3>
	    void forward_first_post_sends_recvs_(
	        const Stage0Complex3 &in, Stage1Complex3 &out, std::vector<mpi_request_t> &byte_recv_requests,
	        std::vector<mpi_request_t> &byte_send_requests, std::vector<int> *byte_recv_peers = nullptr,
	        std::vector<int> *byte_recv_remaining = nullptr
	    )
    {
        const int row_size = row_comm_info_.num_procs;
        (void)row_size;
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        int persistent_byte_index = 0;
        detail::persistent_send_requests<mpi_request_t> *persistent_byte_state = nullptr;
        if ( persistent_byte_p2p_enabled_() )
        {
            int total_chunks = 0;
            for ( const int p : row_comm_order_ )
                total_chunks += mpi_byte_request_count_( bytes_from_elems_( first_forward_send_elems_( p ) ) );
            detail::reset_persistent_send_requests( persistent_first_forward_send_, total_chunks );
            persistent_byte_state = &persistent_first_forward_send_;
        }
        else if ( persistent_value_p2p_enabled_() )
        {
            detail::reset_persistent_send_requests( persistent_first_forward_send_, row_size );
        }
        int persistent_byte_recv_index = 0;
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state = nullptr;
        if ( persistent_byte_p2p_enabled_() )
        {
            detail::reset_persistent_recv_requests( persistent_first_forward_recv_, first_forward_byte_recv_chunks_() );
            persistent_byte_recv_state = &persistent_first_forward_recv_;
        }
        else if ( use_persistent_p2p_ )
        {
            detail::reset_persistent_recv_requests( persistent_first_forward_recv_, row_size );
        }
#else
        if ( persistent_value_p2p_enabled_() )
            detail::reset_persistent_send_requests( persistent_first_forward_send_, row_size );
        if ( use_persistent_p2p_ )
            detail::reset_persistent_recv_requests( persistent_first_forward_recv_, row_size );
#endif
        profile_chunk_count_( "first", first_forward_byte_send_chunks_(), first_forward_byte_recv_chunks_() );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( peer_paired_large_count_byte_schedule_enabled_() )
        {
            synchronize_before_direct_byte_sends_( "first/pre_send_stream_sync" );
            for ( const int p : row_comm_order_ )
            {
	                value_type *recv_ptr = forward_receive_lands_in_stage_()
	                                          ? out.raw_ptr() + half_input_dim_.start_y[p]
	                                          : recv_buffer_.raw_ptr() + first_forward_recv_offset_( p );
	                {
	                    auto phase = profile_scope_( "first/mpi_post_recv" );
                        if ( direct_egger_opt1_forward_byte_receive_enabled_() )
                        {
                            post_direct_forward_byte_irecv_(
                                row_comm_info_, recv_ptr, first_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                                byte_recv_peers, byte_recv_remaining
                            );
                        }
                        else
                        {
	                        post_byte_irecv_(
	                            row_comm_info_, recv_ptr, bytes_from_elems_( first_forward_recv_elems_( p ) ), p, p,
	                            byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
	                            &persistent_byte_recv_index
	                        );
                        }
	                }
                {
                    auto phase = profile_scope_( "first/mpi_post_send" );
                    post_byte_isend_(
                        row_comm_info_, in.raw_ptr() + first_forward_send_offset_( p ),
                        bytes_from_elems_( first_forward_send_elems_( p ) ), p, myid_j_, byte_send_requests,
                        persistent_byte_state, &persistent_byte_index
                    );
                }
            }
            return;
        }
#endif
        for ( const int p : row_comm_order_ )
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
	            value_type *recv_ptr = forward_receive_lands_in_stage_()
	                                      ? out.raw_ptr() + half_input_dim_.start_y[p]
	                                      : recv_buffer_.raw_ptr() + first_forward_recv_offset_( p );
	            if ( cuda_aware_byte_p2p_enabled_() )
	            {
	                auto phase = profile_scope_( "first/mpi_post_recv" );
                    if ( direct_egger_opt1_forward_byte_receive_enabled_() )
                    {
                        post_direct_forward_byte_irecv_(
                            row_comm_info_, recv_ptr, first_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                            byte_recv_peers, byte_recv_remaining
                        );
                    }
                    else
                    {
	                    post_byte_irecv_(
	                        row_comm_info_, recv_ptr, bytes_from_elems_( first_forward_recv_elems_( p ) ), p, p,
	                        byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
	                        &persistent_byte_recv_index
	                    );
                    }
	            }
            else
            {
                auto phase = profile_scope_( "first/mpi_post_recv" );
                post_value_irecv_(
                    row_comm_info_, recv_ptr, first_forward_recv_elems_( p ), p, p, row_recv_requests_[p],
                    persistent_first_forward_recv_, p
                );
            }
#else
            {
                auto phase = profile_scope_( "first/mpi_post_recv" );
                post_value_irecv_(
                    row_comm_info_, host_recv_buffer_.raw_ptr() + first_forward_recv_offset_( p ),
                    first_forward_recv_elems_( p ), p, p, row_recv_requests_[p], persistent_first_forward_recv_, p
                );
            }
#endif
        }

        synchronize_before_direct_byte_sends_( "first/pre_send_stream_sync" );

        for ( const int p : row_comm_order_ )
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                auto phase = profile_scope_( "first/mpi_post_send" );
                post_byte_isend_(
                    row_comm_info_, in.raw_ptr() + first_forward_send_offset_( p ),
                    bytes_from_elems_( first_forward_send_elems_( p ) ), p, myid_j_, byte_send_requests,
                    persistent_byte_state, &persistent_byte_index
                );
            }
            else
            {
                auto phase = profile_scope_( "first/mpi_post_send" );
                post_value_isend_(
                    row_comm_info_, in.raw_ptr() + first_forward_send_offset_( p ), first_forward_send_elems_( p ), p,
                    myid_j_, row_send_requests_[p], persistent_first_forward_send_, p
                );
            }
#else
            {
                auto phase = profile_scope_( "first/mpi_post_send" );
                post_value_isend_(
                    row_comm_info_, host_send_buffer_.raw_ptr() + first_forward_send_offset_( p ),
                    first_forward_send_elems_( p ), p, myid_j_, row_send_requests_[p], persistent_first_forward_send_, p
                );
            }
#endif
        }
    }

    template <class Stage0Complex3, class Stage1Complex3>
    void copy_first_forward_self_( const Stage0Complex3 &in, Stage1Complex3 &out )
    {
        copy_first_forward_chunk_to_stage1_async_(
            in.raw_ptr() + first_forward_send_offset_( myid_j_ ), ny_input_local_, half_input_dim_.start_y[myid_j_],
            out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[myid_j_].stream()
        );
    }

    template <class Stage1Complex3>
    void unpack_first_forward_recv_( int p, Stage1Complex3 &out )
    {
        copy_first_forward_chunk_to_stage1_async_(
            recv_buffer_.raw_ptr() + first_forward_recv_offset_( p ), half_input_dim_.size_y[p],
            half_input_dim_.start_y[p], out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[p].stream()
        );
    }

    template <class Stage1Complex3>
    void unpack_first_forward_host_recv_( int p, Stage1Complex3 &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        copy_first_forward_chunk_to_stage1_async_(
            host_recv_buffer_.raw_ptr() + first_forward_recv_offset_( p ), half_input_dim_.size_y[p],
            half_input_dim_.start_y[p], out.raw_ptr(), runtime_api_t::host_to_device_kind(), streams_[p].stream()
        );
#else
        (void)p;
        (void)out;
#endif
    }

    template <class Stage0Complex3, class Stage1Complex3>
    void forward_first_waitall_( const Stage0Complex3 &in, Stage1Complex3 &out )
    {
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
	        if ( egger_byte_sync_enabled_() )
	        {
	            auto phase = profile_scope_( "first/post_recv_send_direct_byte" );
	            forward_first_post_sends_recvs_( in, out, byte_recv_requests, byte_send_requests );
	        }
        else
        {
            {
                auto phase = profile_scope_( "first/mpi_post_recv" );
                forward_first_post_recvs_( out, byte_recv_requests );
            }
            {
                if ( use_ready_p2p_send_ )
                {
                    auto phase = profile_scope_( "first/pack_ready_send" );
                    forward_first_pack_ready_send_( in, byte_send_requests );
                }
                else
                {
                    auto phase = profile_scope_( "first/pack_sync_send" );
                    forward_first_pack_sync_send_( in, byte_send_requests );
                }
            }
        }
        {
            auto phase = profile_scope_( "first/self_copy" );
            copy_first_forward_self_( in, out );
        }
        {
            auto phase = profile_scope_( "first/wait_recv" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( row_comm_info_, byte_recv_requests, "owned first byte forward recv requests" );
            else
#endif
                row_comm_info_.waitall( row_size, row_recv_requests_.data() );
        }
	        if ( !forward_receive_lands_in_stage_() )
	        {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            auto phase = profile_scope_( "first/stage_recv_to_device" );
#else
            auto phase = profile_scope_( "first/unpack_recv" );
#endif
            for ( const int p : row_comm_order_ )
            {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                unpack_first_forward_host_recv_( p, out );
#else
                unpack_first_forward_recv_( p, out );
#endif
            }
        }
        synchronize_streams_( row_size, "first/recv_copy_complete" );
        {
            auto phase = profile_scope_( "first/wait_send" );
            wait_send_requests_(
                row_comm_info_, byte_send_requests, persistent_first_forward_send_, row_send_requests_, row_size,
                "owned first byte forward send requests", "owned first persistent forward send requests"
            );
        }
    }

    template <class Stage0Complex3, class Stage1Complex3>
    void forward_first_waitany_( const Stage0Complex3 &in, Stage1Complex3 &out )
    {
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( row_size, 0 );
	        if ( egger_byte_sync_enabled_() )
	        {
	            auto phase = profile_scope_( "first/post_recv_send_direct_byte" );
	            forward_first_post_sends_recvs_(
	                in, out, byte_recv_requests, byte_send_requests, &byte_recv_peers, &byte_recv_remaining
	            );
	        }
        else
        {
            {
                auto phase = profile_scope_( "first/mpi_post_recv" );
                forward_first_post_recvs_( out, byte_recv_requests, &byte_recv_peers, &byte_recv_remaining );
            }
            {
                if ( use_ready_p2p_send_ )
                {
                    auto phase = profile_scope_( "first/pack_ready_send" );
                    forward_first_pack_ready_send_( in, byte_send_requests );
                }
                else
                {
                    auto phase = profile_scope_( "first/pack_sync_send" );
                    forward_first_pack_sync_send_( in, byte_send_requests );
                }
            }
        }
        {
            auto phase = profile_scope_( "first/self_copy" );
            copy_first_forward_self_( in, out );
        }

#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const int total =
                detail::mpi_int_cast( byte_recv_requests.size(), "owned first byte forward waitany request count" );
            int completed = 0;
            while ( completed < total )
            {
                int idx = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "first/wait_recv_any" );
                    idx        = row_comm_info_.waitany( total, byte_recv_requests.data() );
                }
                if ( idx == MPI_UNDEFINED )
                    break;
                const int p = byte_recv_peers[idx];
                null_completed_request_( byte_recv_requests, idx );
                --byte_recv_remaining[p];
	                if ( byte_recv_remaining[p] == 0 && !forward_receive_lands_in_stage_() )
                {
                    auto phase = profile_scope_( "first/unpack_recv_chunk" );
                    unpack_first_forward_recv_( p, out );
                }
                ++completed;
            }
        }
        else
#endif
        {
            int completed = 0;
            while ( completed < row_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "first/wait_recv_any" );
                    p          = row_comm_info_.waitany( row_size, row_recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                row_recv_requests_[p] = mpi_request_t();
	                if ( !forward_receive_lands_in_stage_() )
                {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                    auto phase = profile_scope_( "first/stage_recv_chunk_to_device" );
                    unpack_first_forward_host_recv_( p, out );
#else
                    auto phase = profile_scope_( "first/unpack_recv_chunk" );
                    unpack_first_forward_recv_( p, out );
#endif
                }
                ++completed;
            }
        }
        synchronize_streams_( row_size, "first/recv_copy_complete" );
        {
            auto phase = profile_scope_( "first/wait_send" );
            wait_send_requests_(
                row_comm_info_, byte_send_requests, persistent_first_forward_send_, row_send_requests_, row_size,
                "owned first byte forward send requests", "owned first persistent forward send requests"
            );
        }
    }

    template <class XFastComplex3, class XFFTComplex3>
    void forward_second_xfast_( const XFastComplex3 &in, XFFTComplex3 &out )
    {
        reset_line_requests_();
        if ( line_comm_info_.num_procs == 1 )
        {
            {
                auto phase = profile_scope_( "second/degenerate_contiguous_copy" );
                copy_second_forward_degenerate_( in, out );
            }
            synchronize_streams_( 1, "second/degenerate_copy_complete" );
            return;
        }
        if ( mode_ == mpi_transpose_3d_mode::p2p_waitall && !egger_parity_forward_waitany_enabled_() )
            forward_second_xfast_waitall_( in, out );
        else
            forward_second_xfast_waitany_( in, out );
    }

    template <class XFastComplex3>
    void pack_second_forward_chunk_( int p, const XFastComplex3 &in )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *dst  = send_buffer_.raw_ptr() + second_forward_send_offset_( p );
        auto        kind = runtime_api_t::device_to_device_kind();
#else
        value_type *dst  = host_send_buffer_.raw_ptr() + second_forward_send_offset_( p );
        auto        kind = runtime_api_t::device_to_host_kind();
#endif
        runtime_api_t::memcpy_async(
            dst, in.raw_ptr() + second_forward_send_offset_( p ), bytes_from_elems_( second_forward_send_elems_( p ) ),
            kind, streams_[p].stream()
        );
    }

    int second_forward_byte_send_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : line_comm_order_ )
            total_chunks += mpi_byte_request_count_( bytes_from_elems_( second_forward_send_elems_( p ) ) );
        return total_chunks;
    }

    void prepare_second_forward_send_state_()
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( persistent_byte_p2p_enabled_() )
            detail::reset_persistent_send_requests( persistent_second_forward_send_, second_forward_byte_send_chunks_() );
        else
#endif
        if ( persistent_value_p2p_enabled_() )
            detail::reset_persistent_send_requests( persistent_second_forward_send_, line_comm_info_.num_procs );
    }

    void forward_second_post_send_(
        int p, std::vector<mpi_request_t> &byte_send_requests, int &persistent_byte_index
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const value_type *send_ptr = send_buffer_.raw_ptr() + second_forward_send_offset_( p );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            post_byte_isend_(
                line_comm_info_, send_ptr, bytes_from_elems_( second_forward_send_elems_( p ) ), p, myid_i_,
                byte_send_requests, persistent_byte_p2p_enabled_() ? &persistent_second_forward_send_ : nullptr,
                &persistent_byte_index
            );
        }
        else
        {
            post_value_isend_(
                line_comm_info_, send_ptr, second_forward_send_elems_( p ), p, myid_i_, line_send_requests_[p],
                persistent_second_forward_send_, p
            );
        }
#else
        post_value_isend_(
            line_comm_info_, host_send_buffer_.raw_ptr() + second_forward_send_offset_( p ),
            second_forward_send_elems_( p ), p, myid_i_, line_send_requests_[p], persistent_second_forward_send_, p
        );
#endif
    }

    template <class XFastComplex3>
	    void forward_second_pack_ready_send_( const XFastComplex3 &in, std::vector<mpi_request_t> &byte_send_requests )
	    {
	        prepare_second_forward_send_state_();
	        profile_chunk_count_( "second", second_forward_byte_send_chunks_(), second_forward_byte_recv_chunks_() );
	        int persistent_byte_index = 0;
        for ( const int p : line_comm_order_ )
        {
            {
                auto phase = profile_scope_( "second/pack_peer" );
                pack_second_forward_chunk_( p, in );
            }
        }

        std::vector<char> posted( line_comm_info_.num_procs, 0 );
        int remaining = static_cast<int>( line_comm_order_.size() );
        while ( remaining > 0 )
        {
            std::vector<int> ready_peers;
            {
                auto phase = profile_scope_( "second/wait_pack_ready" );
                while ( ready_peers.empty() )
                {
                    for ( const int p : line_comm_order_ )
                    {
                        if ( posted[p] )
                            continue;
                        if ( runtime_api_t::stream_ready( streams_[p].stream() ) )
                            ready_peers.push_back( p );
                    }
                }
            }
            for ( const int p : ready_peers )
            {
                auto phase = profile_scope_( "second/mpi_post_send" );
                forward_second_post_send_( p, byte_send_requests, persistent_byte_index );
                posted[p] = 1;
                --remaining;
            }
        }
    }

    template <class XFastComplex3>
	    void forward_second_pack_sync_send_( const XFastComplex3 &in, std::vector<mpi_request_t> &byte_send_requests )
	    {
	        prepare_second_forward_send_state_();
	        profile_chunk_count_( "second", second_forward_byte_send_chunks_(), second_forward_byte_recv_chunks_() );
	        int persistent_byte_index = 0;
        for ( const int p : line_comm_order_ )
        {
            {
                auto phase = profile_scope_( "second/pack_peer" );
                pack_second_forward_chunk_( p, in );
            }
            {
                auto phase = profile_scope_( "second/pre_send_stream_sync" );
                runtime_api_t::stream_synchronize( streams_[p].stream() );
            }
            {
                auto phase = profile_scope_( "second/mpi_post_send" );
                forward_second_post_send_( p, byte_send_requests, persistent_byte_index );
            }
        }
    }

    void prepare_second_forward_recv_state_()
    {
        const int line_size = line_comm_info_.num_procs;
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( persistent_byte_p2p_enabled_() )
            detail::reset_persistent_recv_requests( persistent_second_forward_recv_, second_forward_byte_recv_chunks_() );
        else if ( use_persistent_p2p_ )
            detail::reset_persistent_recv_requests( persistent_second_forward_recv_, line_size );
#else
        if ( use_persistent_p2p_ )
            detail::reset_persistent_recv_requests( persistent_second_forward_recv_, line_size );
#endif
    }

    template <class XFFTComplex3>
    void forward_second_post_recvs_(
        XFFTComplex3 &out, std::vector<mpi_request_t> &byte_recv_requests,
        std::vector<int> *byte_recv_peers = nullptr, std::vector<int> *byte_recv_remaining = nullptr
    )
    {
        prepare_second_forward_recv_state_();
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        int persistent_byte_recv_index = 0;
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state =
            persistent_byte_p2p_enabled_() ? &persistent_second_forward_recv_ : nullptr;
#endif
        for ( const int p : line_comm_order_ )
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
	            value_type *recv_ptr = forward_receive_lands_in_stage_()
	                                      ? out.raw_ptr() + transpose1_dim_.start_x[p]
	                                      : recv_buffer_.raw_ptr() + second_forward_recv_offset_( p );
	            if ( cuda_aware_byte_p2p_enabled_() )
	            {
                    if ( direct_egger_opt1_forward_byte_receive_enabled_() )
                    {
                        post_direct_forward_byte_irecv_(
                            line_comm_info_, recv_ptr, second_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                            byte_recv_peers, byte_recv_remaining
                        );
                    }
                    else
                    {
	                    post_byte_irecv_(
	                        line_comm_info_, recv_ptr, bytes_from_elems_( second_forward_recv_elems_( p ) ), p, p,
	                        byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
	                        &persistent_byte_recv_index
	                    );
                    }
	            }
            else
            {
                if ( direct_forward_value_receive_enabled_() )
                    post_typed_irecv_(
                        line_comm_info_, recv_ptr, 1, second_forward_direct_recvtypes_[p], p, p,
                        line_recv_requests_[p], persistent_second_forward_recv_, p
                    );
                else
                    post_value_irecv_(
                        line_comm_info_, recv_ptr, second_forward_recv_elems_( p ), p, p, line_recv_requests_[p],
                        persistent_second_forward_recv_, p
                    );
            }
#else
            post_value_irecv_(
                line_comm_info_, host_recv_buffer_.raw_ptr() + second_forward_recv_offset_( p ),
                second_forward_recv_elems_( p ), p, p, line_recv_requests_[p], persistent_second_forward_recv_, p
            );
#endif
        }
    }

		    template <class XFastComplex3, class XFFTComplex3>
		    void forward_second_post_sends_recvs_(
		        const XFastComplex3 &in, XFFTComplex3 &out, std::vector<mpi_request_t> &byte_recv_requests,
		        std::vector<mpi_request_t> &byte_send_requests, std::vector<int> *byte_recv_peers = nullptr,
	        std::vector<int> *byte_recv_remaining = nullptr
	    )
    {
        const int line_size = line_comm_info_.num_procs;
        (void)line_size;
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        int persistent_byte_index = 0;
        detail::persistent_send_requests<mpi_request_t> *persistent_byte_state = nullptr;
        if ( persistent_byte_p2p_enabled_() )
        {
            int total_chunks = 0;
            for ( const int p : line_comm_order_ )
                total_chunks += mpi_byte_request_count_( bytes_from_elems_( second_forward_send_elems_( p ) ) );
            detail::reset_persistent_send_requests( persistent_second_forward_send_, total_chunks );
            persistent_byte_state = &persistent_second_forward_send_;
        }
        else if ( persistent_value_p2p_enabled_() )
        {
            detail::reset_persistent_send_requests( persistent_second_forward_send_, line_size );
        }
        int persistent_byte_recv_index = 0;
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state = nullptr;
        if ( persistent_byte_p2p_enabled_() )
        {
            detail::reset_persistent_recv_requests( persistent_second_forward_recv_, second_forward_byte_recv_chunks_() );
            persistent_byte_recv_state = &persistent_second_forward_recv_;
        }
        else if ( use_persistent_p2p_ )
        {
            detail::reset_persistent_recv_requests( persistent_second_forward_recv_, line_size );
        }
#else
        if ( persistent_value_p2p_enabled_() )
            detail::reset_persistent_send_requests( persistent_second_forward_send_, line_size );
	        if ( use_persistent_p2p_ )
	            detail::reset_persistent_recv_requests( persistent_second_forward_recv_, line_size );
	#endif
	        profile_chunk_count_( "second", second_forward_byte_send_chunks_(), second_forward_byte_recv_chunks_() );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( peer_paired_large_count_byte_schedule_enabled_() )
        {
            synchronize_before_direct_byte_sends_( "second/pre_send_stream_sync" );
            for ( const int p : line_comm_order_ )
            {
	                value_type *recv_ptr = forward_receive_lands_in_stage_()
	                                          ? out.raw_ptr() + transpose1_dim_.start_x[p]
	                                          : recv_buffer_.raw_ptr() + second_forward_recv_offset_( p );
	                {
	                    auto phase = profile_scope_( "second/mpi_post_recv" );
                        if ( direct_egger_opt1_forward_byte_receive_enabled_() )
                        {
                            post_direct_forward_byte_irecv_(
                                line_comm_info_, recv_ptr, second_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                                byte_recv_peers, byte_recv_remaining
                            );
                        }
                        else
                        {
	                        post_byte_irecv_(
	                            line_comm_info_, recv_ptr, bytes_from_elems_( second_forward_recv_elems_( p ) ), p, p,
	                            byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
	                            &persistent_byte_recv_index
	                        );
                        }
	                }
                {
                    auto phase = profile_scope_( "second/mpi_post_send" );
                    post_byte_isend_(
                        line_comm_info_, in.raw_ptr() + second_forward_send_offset_( p ),
                        bytes_from_elems_( second_forward_send_elems_( p ) ), p, myid_i_, byte_send_requests,
                        persistent_byte_state, &persistent_byte_index
                    );
                }
            }
            return;
        }
#endif
	        for ( const int p : line_comm_order_ )
	        {
	#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
		            value_type *recv_ptr = forward_receive_lands_in_stage_()
		                                      ? out.raw_ptr() + transpose1_dim_.start_x[p]
		                                      : recv_buffer_.raw_ptr() + second_forward_recv_offset_( p );
		            if ( cuda_aware_byte_p2p_enabled_() )
		            {
		                auto phase = profile_scope_( "second/mpi_post_recv" );
                        if ( direct_egger_opt1_forward_byte_receive_enabled_() )
                        {
                            post_direct_forward_byte_irecv_(
                                line_comm_info_, recv_ptr, second_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                                byte_recv_peers, byte_recv_remaining
                            );
                        }
                        else
                        {
		                    post_byte_irecv_(
		                        line_comm_info_, recv_ptr, bytes_from_elems_( second_forward_recv_elems_( p ) ), p, p,
		                        byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
		                        &persistent_byte_recv_index
		                    );
                        }
		            }
	            else
	            {
	                auto phase = profile_scope_( "second/mpi_post_recv" );
	                post_value_irecv_(
	                    line_comm_info_, recv_ptr, second_forward_recv_elems_( p ), p, p, line_recv_requests_[p],
	                    persistent_second_forward_recv_, p
	                );
	            }
	#else
	            {
	                auto phase = profile_scope_( "second/mpi_post_recv" );
	                post_value_irecv_(
	                    line_comm_info_, host_recv_buffer_.raw_ptr() + second_forward_recv_offset_( p ),
	                    second_forward_recv_elems_( p ), p, p, line_recv_requests_[p], persistent_second_forward_recv_, p
	                );
	            }
	#endif
	        }

	        synchronize_before_direct_byte_sends_( "second/pre_send_stream_sync" );

	        for ( const int p : line_comm_order_ )
	        {
	#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
	            if ( cuda_aware_byte_p2p_enabled_() )
	            {
	                auto phase = profile_scope_( "second/mpi_post_send" );
	                post_byte_isend_(
	                    line_comm_info_, in.raw_ptr() + second_forward_send_offset_( p ),
	                    bytes_from_elems_( second_forward_send_elems_( p ) ), p, myid_i_, byte_send_requests,
	                    persistent_byte_state, &persistent_byte_index
	                );
	            }
	            else
	            {
	                auto phase = profile_scope_( "second/mpi_post_send" );
	                post_value_isend_(
	                    line_comm_info_, in.raw_ptr() + second_forward_send_offset_( p ), second_forward_send_elems_( p ),
	                    p, myid_i_, line_send_requests_[p], persistent_second_forward_send_, p
	                );
	            }
	#else
	            {
	                auto phase = profile_scope_( "second/mpi_post_send" );
	                post_value_isend_(
	                    line_comm_info_, host_send_buffer_.raw_ptr() + second_forward_send_offset_( p ),
	                    second_forward_send_elems_( p ), p, myid_i_, line_send_requests_[p], persistent_second_forward_send_,
	                    p
	                );
	            }
	#endif
	        }
	    }

    template <class XFastComplex3, class XFFTComplex3>
    void copy_second_forward_self_( const XFastComplex3 &in, XFFTComplex3 &out )
    {
        copy_second_forward_chunk_to_xstage_async_(
            in.raw_ptr() + second_forward_send_offset_( myid_i_ ), nx_local_, transpose1_dim_.start_x[myid_i_],
            out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[myid_i_].stream()
        );
    }

    template <class XFastComplex3, class XFFTComplex3>
    void copy_second_forward_degenerate_( const XFastComplex3 &in, XFFTComplex3 &out )
    {
        runtime_api_t::memcpy_async(
            out.raw_ptr(), in.raw_ptr(),
            bytes_from_elems_( nx_global_ * nz_local_ * ny_output_local_ ),
            runtime_api_t::device_to_device_kind(), streams_[0].stream()
        );
    }

    template <class XFFTComplex3>
    void unpack_second_forward_recv_( int p, XFFTComplex3 &out )
    {
        copy_second_forward_chunk_to_xstage_async_(
            recv_buffer_.raw_ptr() + second_forward_recv_offset_( p ), transpose1_dim_.size_x[p],
            transpose1_dim_.start_x[p], out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[p].stream()
        );
    }

    template <class XFFTComplex3>
    void unpack_second_forward_host_recv_( int p, XFFTComplex3 &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        copy_second_forward_chunk_to_xstage_async_(
            host_recv_buffer_.raw_ptr() + second_forward_recv_offset_( p ), transpose1_dim_.size_x[p],
            transpose1_dim_.start_x[p], out.raw_ptr(), runtime_api_t::host_to_device_kind(), streams_[p].stream()
        );
#else
        (void)p;
        (void)out;
#endif
    }

    template <class XFastComplex3, class XFFTComplex3>
    void forward_second_xfast_waitall_( const XFastComplex3 &in, XFFTComplex3 &out )
    {
        const int line_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
	        if ( egger_byte_sync_enabled_() )
	        {
	            auto phase = profile_scope_( "second/post_recv_send_direct_byte" );
	            forward_second_post_sends_recvs_( in, out, byte_recv_requests, byte_send_requests );
	        }
        else
        {
            {
                auto phase = profile_scope_( "second/mpi_post_recv" );
                forward_second_post_recvs_( out, byte_recv_requests );
            }
            {
                if ( use_ready_p2p_send_ )
                {
                    auto phase = profile_scope_( "second/pack_ready_send" );
                    forward_second_pack_ready_send_( in, byte_send_requests );
                }
                else
                {
                    auto phase = profile_scope_( "second/pack_sync_send" );
                    forward_second_pack_sync_send_( in, byte_send_requests );
                }
            }
        }
        {
            auto phase = profile_scope_( "second/self_copy" );
            copy_second_forward_self_( in, out );
        }
        {
            auto phase = profile_scope_( "second/wait_recv" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( line_comm_info_, byte_recv_requests, "owned second byte forward recv requests" );
            else
#endif
                line_comm_info_.waitall( line_size, line_recv_requests_.data() );
        }
	        if ( !forward_receive_lands_in_stage_() )
	        {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            auto phase = profile_scope_( "second/stage_recv_to_device" );
#else
            auto phase = profile_scope_( "second/unpack_recv" );
#endif
            for ( const int p : line_comm_order_ )
            {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                unpack_second_forward_host_recv_( p, out );
#else
                unpack_second_forward_recv_( p, out );
#endif
            }
        }
        synchronize_streams_( line_size, "second/recv_copy_complete" );
        {
            auto phase = profile_scope_( "second/wait_send" );
            wait_send_requests_(
                line_comm_info_, byte_send_requests, persistent_second_forward_send_, line_send_requests_, line_size,
                "owned second byte forward send requests", "owned second persistent forward send requests"
            );
        }
    }

    template <class XFastComplex3, class XFFTComplex3>
    void forward_second_xfast_waitany_( const XFastComplex3 &in, XFFTComplex3 &out )
    {
        const int line_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( line_size, 0 );
	        if ( egger_byte_sync_enabled_() )
	        {
	            auto phase = profile_scope_( "second/post_recv_send_direct_byte" );
	            forward_second_post_sends_recvs_(
	                in, out, byte_recv_requests, byte_send_requests, &byte_recv_peers, &byte_recv_remaining
	            );
	        }
        else
        {
            {
                auto phase = profile_scope_( "second/mpi_post_recv" );
                forward_second_post_recvs_( out, byte_recv_requests, &byte_recv_peers, &byte_recv_remaining );
            }
            {
                if ( use_ready_p2p_send_ )
                {
                    auto phase = profile_scope_( "second/pack_ready_send" );
                    forward_second_pack_ready_send_( in, byte_send_requests );
                }
                else
                {
                    auto phase = profile_scope_( "second/pack_sync_send" );
                    forward_second_pack_sync_send_( in, byte_send_requests );
                }
            }
        }
        {
            auto phase = profile_scope_( "second/self_copy" );
            copy_second_forward_self_( in, out );
        }

#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const int total =
                detail::mpi_int_cast( byte_recv_requests.size(), "owned second byte forward waitany request count" );
            int completed = 0;
            while ( completed < total )
            {
                int idx = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "second/wait_recv_any" );
                    idx        = line_comm_info_.waitany( total, byte_recv_requests.data() );
                }
                if ( idx == MPI_UNDEFINED )
                    break;
                const int p = byte_recv_peers[idx];
                null_completed_request_( byte_recv_requests, idx );
                --byte_recv_remaining[p];
	                if ( byte_recv_remaining[p] == 0 && !forward_receive_lands_in_stage_() )
                {
                    auto phase = profile_scope_( "second/unpack_recv_chunk" );
                    unpack_second_forward_recv_( p, out );
                }
                ++completed;
            }
        }
        else
#endif
        {
            int completed = 0;
            while ( completed < line_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "second/wait_recv_any" );
                    p          = line_comm_info_.waitany( line_size, line_recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                line_recv_requests_[p] = mpi_request_t();
	                if ( !forward_receive_lands_in_stage_() )
                {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                    auto phase = profile_scope_( "second/stage_recv_chunk_to_device" );
                    unpack_second_forward_host_recv_( p, out );
#else
                    auto phase = profile_scope_( "second/unpack_recv_chunk" );
                    unpack_second_forward_recv_( p, out );
#endif
                }
                ++completed;
            }
        }
        synchronize_streams_( line_size, "second/recv_copy_complete" );
        {
            auto phase = profile_scope_( "second/wait_send" );
            wait_send_requests_(
                line_comm_info_, byte_send_requests, persistent_second_forward_send_, line_send_requests_, line_size,
                "owned second byte forward send requests", "owned second persistent forward send requests"
            );
        }
    }

    template <class XFFTComplex3, class XFastComplex3>
    void backward_second_xfast_( const XFFTComplex3 &in, XFastComplex3 &out )
    {
        reset_line_requests_();
        if ( line_comm_info_.num_procs == 1 )
        {
            {
                auto phase = profile_scope_( "second_backward/degenerate_contiguous_copy" );
                copy_second_backward_degenerate_( in, out );
            }
            synchronize_streams_( 1, "second_backward/degenerate_copy_complete" );
            return;
        }
        if ( mode_ == mpi_transpose_3d_mode::p2p_waitall || egger_parity_backward_waitall_enabled_() )
            backward_second_xfast_waitall_( in, out );
        else
            backward_second_xfast_waitany_( in, out );
    }

    template <class XFFTComplex3>
    void pack_second_backward_chunk_( int p, const XFFTComplex3 &in )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *dst  = send_buffer_.raw_ptr() + second_backward_send_offset_( p );
        auto        kind = runtime_api_t::device_to_device_kind();
#else
        value_type *dst  = host_send_buffer_.raw_ptr() + second_backward_send_offset_( p );
        auto        kind = runtime_api_t::device_to_host_kind();
#endif
        pack_second_backward_chunk_async_(
            in.raw_ptr(), transpose1_dim_.start_x[p], transpose1_dim_.size_x[p], dst, kind, streams_[p].stream()
        );
    }

    template <class XFFTComplex3, class XFastComplex3>
    void copy_second_backward_self_( const XFFTComplex3 &in, XFastComplex3 &out )
    {
        pack_second_backward_chunk_async_(
            in.raw_ptr(), transpose1_dim_.start_x[myid_i_], nx_local_,
            out.raw_ptr() + second_backward_recv_offset_( myid_i_ ), runtime_api_t::device_to_device_kind(),
            streams_[myid_i_].stream()
        );
    }

    template <class XFFTComplex3, class XFastComplex3>
    void copy_second_backward_degenerate_( const XFFTComplex3 &in, XFastComplex3 &out )
    {
        runtime_api_t::memcpy_async(
            out.raw_ptr(), in.raw_ptr(),
            bytes_from_elems_( nx_global_ * nz_local_ * ny_output_local_ ),
            runtime_api_t::device_to_device_kind(), streams_[0].stream()
        );
    }

    template <class XFastComplex3>
    void copy_second_backward_recv_to_out_( int p, XFastComplex3 &out )
    {
        runtime_api_t::memcpy_async(
            out.raw_ptr() + second_backward_recv_offset_( p ),
            recv_buffer_.raw_ptr() + second_backward_packed_recv_offset_( p ),
            bytes_from_elems_( second_backward_recv_elems_( p ) ), runtime_api_t::device_to_device_kind(),
            streams_[p].stream()
        );
    }

    template <class XFastComplex3>
    void copy_second_backward_host_recv_to_out_( int p, XFastComplex3 &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy_async(
            out.raw_ptr() + second_backward_recv_offset_( p ),
            host_recv_buffer_.raw_ptr() + second_backward_packed_recv_offset_( p ),
            bytes_from_elems_( second_backward_recv_elems_( p ) ), runtime_api_t::host_to_device_kind(),
            streams_[p].stream()
        );
#else
        (void)p;
        (void)out;
#endif
    }

    template <class XFastComplex3>
    void backward_second_post_recvs_(
        XFastComplex3 &out, std::vector<mpi_request_t> &byte_recv_requests, std::vector<int> *byte_recv_peers = nullptr,
        std::vector<int> *byte_recv_remaining = nullptr
    )
    {
        const int line_size = line_comm_info_.num_procs;
        (void)line_size;
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        int persistent_byte_recv_index = 0;
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state = nullptr;
        if ( persistent_byte_p2p_enabled_() )
        {
            detail::reset_persistent_recv_requests(
                persistent_second_backward_recv_, second_backward_byte_recv_chunks_()
            );
            persistent_byte_recv_state = &persistent_second_backward_recv_;
        }
        else if ( use_persistent_p2p_ )
        {
            detail::reset_persistent_recv_requests( persistent_second_backward_recv_, line_size );
        }
#else
        if ( use_persistent_p2p_ )
            detail::reset_persistent_recv_requests( persistent_second_backward_recv_, line_size );
#endif
        for ( const int p : line_comm_order_ )
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            value_type *recv_ptr = backward_receive_lands_in_output_()
                                      ? out.raw_ptr() + second_backward_recv_offset_( p )
                                      : recv_buffer_.raw_ptr() + second_backward_packed_recv_offset_( p );
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                post_byte_irecv_(
                    line_comm_info_, recv_ptr, bytes_from_elems_( second_backward_recv_elems_( p ) ), p, p,
                    byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
                    &persistent_byte_recv_index
                );
            }
            else
            {
                post_value_irecv_(
                    line_comm_info_, recv_ptr, second_backward_recv_elems_( p ), p, p, line_recv_requests_[p],
                    persistent_second_backward_recv_, p
                );
            }
#else
            post_value_irecv_(
                line_comm_info_, host_recv_buffer_.raw_ptr() + second_backward_packed_recv_offset_( p ),
                second_backward_recv_elems_( p ), p, p, line_recv_requests_[p], persistent_second_backward_recv_, p
            );
#endif
        }
    }

    int second_backward_byte_send_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : line_comm_order_ )
            total_chunks += mpi_byte_request_count_( bytes_from_elems_( second_backward_send_elems_( p ) ) );
        return total_chunks;
    }

    void prepare_second_backward_send_state_()
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( persistent_byte_p2p_enabled_() )
            detail::reset_persistent_send_requests( persistent_second_backward_send_, second_backward_byte_send_chunks_() );
        else
#endif
        if ( persistent_value_p2p_enabled_() )
            detail::reset_persistent_send_requests( persistent_second_backward_send_, line_comm_info_.num_procs );
    }

    void backward_second_post_send_(
        int p, std::vector<mpi_request_t> &byte_send_requests, int &persistent_byte_index
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const value_type *send_ptr = send_buffer_.raw_ptr() + second_backward_send_offset_( p );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            post_byte_isend_(
                line_comm_info_, send_ptr, bytes_from_elems_( second_backward_send_elems_( p ) ), p, myid_i_,
                byte_send_requests, persistent_byte_p2p_enabled_() ? &persistent_second_backward_send_ : nullptr,
                &persistent_byte_index
            );
        }
        else
        {
            post_value_isend_(
                line_comm_info_, send_ptr, second_backward_send_elems_( p ), p, myid_i_, line_send_requests_[p],
                persistent_second_backward_send_, p
            );
        }
#else
        post_value_isend_(
            line_comm_info_, host_send_buffer_.raw_ptr() + second_backward_send_offset_( p ),
            second_backward_send_elems_( p ), p, myid_i_, line_send_requests_[p], persistent_second_backward_send_, p
        );
#endif
    }

	    void backward_second_post_sends_( std::vector<mpi_request_t> &byte_send_requests )
	    {
	        prepare_second_backward_send_state_();
	        profile_chunk_count_(
	            "second_backward", second_backward_byte_send_chunks_(), second_backward_byte_recv_chunks_()
	        );
	        int persistent_byte_index = 0;
        for ( const int p : line_comm_order_ )
            backward_second_post_send_( p, byte_send_requests, persistent_byte_index );
    }

	    void backward_second_post_sends_when_ready_( std::vector<mpi_request_t> &byte_send_requests )
	    {
	        prepare_second_backward_send_state_();
	        profile_chunk_count_(
	            "second_backward", second_backward_byte_send_chunks_(), second_backward_byte_recv_chunks_()
	        );
	        int persistent_byte_index = 0;
        std::vector<char> posted( line_comm_info_.num_procs, 0 );
        int remaining = static_cast<int>( line_comm_order_.size() );
        while ( remaining > 0 )
        {
            std::vector<int> ready_peers;
            {
                auto phase = profile_scope_( "second_backward/wait_pack_ready" );
                while ( ready_peers.empty() )
                {
                    for ( const int p : line_comm_order_ )
                    {
                        if ( posted[p] )
                            continue;
                        if ( runtime_api_t::stream_ready( streams_[p].stream() ) )
                            ready_peers.push_back( p );
                    }
                }
            }
            for ( const int p : ready_peers )
            {
                auto phase = profile_scope_( "second_backward/mpi_post_send" );
                backward_second_post_send_( p, byte_send_requests, persistent_byte_index );
                posted[p] = 1;
                --remaining;
            }
        }
    }

    template <class XFFTComplex3>
	    void backward_second_pack_sync_send_( const XFFTComplex3 &in, std::vector<mpi_request_t> &byte_send_requests )
	    {
	        prepare_second_backward_send_state_();
	        profile_chunk_count_(
	            "second_backward", second_backward_byte_send_chunks_(), second_backward_byte_recv_chunks_()
	        );
	        int persistent_byte_index = 0;
        for ( const int p : line_comm_order_ )
        {
            {
                auto phase = profile_scope_( "second_backward/pack_peer" );
                pack_second_backward_chunk_( p, in );
            }
            {
                auto phase = profile_scope_( "second_backward/pre_send_stream_sync" );
                runtime_api_t::stream_synchronize( streams_[p].stream() );
            }
            {
                auto phase = profile_scope_( "second_backward/mpi_post_send" );
                backward_second_post_send_( p, byte_send_requests, persistent_byte_index );
            }
        }
    }

    template <class XFFTComplex3, class XFastComplex3>
    void backward_second_post_recv_pack_sync_send_pairs_(
        const XFFTComplex3 &in, XFastComplex3 &out, std::vector<mpi_request_t> &byte_recv_requests,
        std::vector<mpi_request_t> &byte_send_requests, std::vector<int> *byte_recv_peers = nullptr,
        std::vector<int> *byte_recv_remaining = nullptr
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        prepare_second_backward_send_state_();
        profile_chunk_count_(
            "second_backward", second_backward_byte_send_chunks_(), second_backward_byte_recv_chunks_()
        );
        int persistent_byte_index      = 0;
        int persistent_byte_recv_index = 0;
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state = nullptr;
        for ( const int p : line_comm_order_ )
        {
            value_type *recv_ptr = backward_receive_lands_in_output_()
                                      ? out.raw_ptr() + second_backward_recv_offset_( p )
                                      : recv_buffer_.raw_ptr() + second_backward_packed_recv_offset_( p );
            {
                auto phase = profile_scope_( "second_backward/mpi_post_recv" );
                post_byte_irecv_(
                    line_comm_info_, recv_ptr, bytes_from_elems_( second_backward_recv_elems_( p ) ), p, p,
                    byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
                    &persistent_byte_recv_index
                );
            }
            {
                auto phase = profile_scope_( "second_backward/pack_peer" );
                pack_second_backward_chunk_( p, in );
            }
            {
                auto phase = profile_scope_( "second_backward/pre_send_stream_sync" );
                runtime_api_t::stream_synchronize( streams_[p].stream() );
            }
            {
                auto phase = profile_scope_( "second_backward/mpi_post_send" );
                backward_second_post_send_( p, byte_send_requests, persistent_byte_index );
            }
        }
#else
        (void)in;
        (void)out;
        (void)byte_recv_requests;
        (void)byte_send_requests;
        (void)byte_recv_peers;
        (void)byte_recv_remaining;
#endif
    }

    template <class XFFTComplex3, class XFastComplex3>
    void backward_second_xfast_waitall_( const XFFTComplex3 &in, XFastComplex3 &out )
    {
        const int line_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        const bool                 peer_paired_schedule = peer_paired_large_count_byte_schedule_enabled_();
        if ( peer_paired_schedule )
        {
            auto phase = profile_scope_( "second_backward/recv_pack_sync_send_pairs" );
            backward_second_post_recv_pack_sync_send_pairs_( in, out, byte_recv_requests, byte_send_requests );
        }
        else
        {
            {
                auto phase = profile_scope_( "second_backward/mpi_post_recv" );
                backward_second_post_recvs_( out, byte_recv_requests );
            }
	        if ( egger_byte_sync_enabled_() && !egger_parity_backward_ready_post_send_enabled_() )
	        {
	            auto phase = profile_scope_( "second_backward/pack_sync_send" );
	            backward_second_pack_sync_send_( in, byte_send_requests );
	        }
            else
            {
                {
                    auto phase = profile_scope_( "second_backward/pack_send_chunks" );
                    for ( const int p : line_comm_order_ )
                        pack_second_backward_chunk_( p, in );
                }
                {
                    auto phase = profile_scope_( "second_backward/wait_pack_ready_post_send" );
                    backward_second_post_sends_when_ready_( byte_send_requests );
                }
            }
        }
        {
            auto phase = profile_scope_( "second_backward/self_copy" );
            copy_second_backward_self_( in, out );
        }
        {
            auto phase = profile_scope_( "second_backward/wait_recv" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( line_comm_info_, byte_recv_requests, "owned second byte backward recv requests" );
            else
#endif
                line_comm_info_.waitall( line_size, line_recv_requests_.data() );
        }
        if ( !backward_receive_lands_in_output_() )
        {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            auto phase = profile_scope_( "second_backward/stage_recv_to_device" );
#else
            auto phase = profile_scope_( "second_backward/unpack_recv" );
#endif
            for ( const int p : line_comm_order_ )
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                copy_second_backward_host_recv_to_out_( p, out );
#else
                copy_second_backward_recv_to_out_( p, out );
#endif
        }
        synchronize_streams_( line_size, "second_backward/recv_copy_complete" );
        {
            auto phase = profile_scope_( "second_backward/wait_send" );
            wait_send_requests_(
                line_comm_info_, byte_send_requests, persistent_second_backward_send_, line_send_requests_, line_size,
                "owned second byte backward send requests", "owned second persistent backward send requests"
            );
        }
    }

    template <class XFFTComplex3, class XFastComplex3>
    void backward_second_xfast_waitany_( const XFFTComplex3 &in, XFastComplex3 &out )
    {
        const int line_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( line_size, 0 );
        const bool                 peer_paired_schedule = peer_paired_large_count_byte_schedule_enabled_();
        if ( peer_paired_schedule )
        {
            auto phase = profile_scope_( "second_backward/recv_pack_sync_send_pairs" );
            backward_second_post_recv_pack_sync_send_pairs_(
                in, out, byte_recv_requests, byte_send_requests, &byte_recv_peers, &byte_recv_remaining
            );
        }
        else
        {
            {
                auto phase = profile_scope_( "second_backward/mpi_post_recv" );
                backward_second_post_recvs_( out, byte_recv_requests, &byte_recv_peers, &byte_recv_remaining );
            }
	        if ( egger_byte_sync_enabled_() && !egger_parity_backward_ready_post_send_enabled_() )
	        {
	            auto phase = profile_scope_( "second_backward/pack_sync_send" );
	            backward_second_pack_sync_send_( in, byte_send_requests );
	        }
            else
            {
                {
                    auto phase = profile_scope_( "second_backward/pack_send_chunks" );
                    for ( const int p : line_comm_order_ )
                        pack_second_backward_chunk_( p, in );
                }
                {
                    auto phase = profile_scope_( "second_backward/wait_pack_ready_post_send" );
                    backward_second_post_sends_when_ready_( byte_send_requests );
                }
            }
        }
        {
            auto phase = profile_scope_( "second_backward/self_copy" );
            copy_second_backward_self_( in, out );
        }

#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const int total =
                detail::mpi_int_cast( byte_recv_requests.size(), "owned second byte backward waitany request count" );
            int completed = 0;
            while ( completed < total )
            {
                int idx = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "second_backward/wait_recv_any" );
                    idx        = line_comm_info_.waitany( total, byte_recv_requests.data() );
                }
                if ( idx == MPI_UNDEFINED )
                    break;
                const int p = byte_recv_peers[idx];
                null_completed_request_( byte_recv_requests, idx );
                --byte_recv_remaining[p];
                if ( byte_recv_remaining[p] == 0 && !backward_receive_lands_in_output_() )
                {
                    auto phase = profile_scope_( "second_backward/unpack_recv_chunk" );
                    copy_second_backward_recv_to_out_( p, out );
                }
                ++completed;
            }
        }
        else
#endif
        {
            int completed = 0;
            while ( completed < line_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "second_backward/wait_recv_any" );
                    p          = line_comm_info_.waitany( line_size, line_recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                line_recv_requests_[p] = mpi_request_t();
                if ( !backward_receive_lands_in_output_() )
                {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                    auto phase = profile_scope_( "second_backward/stage_recv_chunk_to_device" );
                    copy_second_backward_host_recv_to_out_( p, out );
#else
                    auto phase = profile_scope_( "second_backward/unpack_recv_chunk" );
                    copy_second_backward_recv_to_out_( p, out );
#endif
                }
                ++completed;
            }
        }
        synchronize_streams_( line_size, "second_backward/recv_copy_complete" );
        {
            auto phase = profile_scope_( "second_backward/wait_send" );
            wait_send_requests_(
                line_comm_info_, byte_send_requests, persistent_second_backward_send_, line_send_requests_, line_size,
                "owned second byte backward send requests", "owned second persistent backward send requests"
            );
        }
    }

    template <class Stage1Complex3, class Stage0Complex3>
    void backward_first_( const Stage1Complex3 &in, Stage0Complex3 &out )
    {
        reset_row_requests_();
        if ( row_comm_info_.num_procs == 1 )
        {
            {
                auto phase = profile_scope_( "first_backward/degenerate_self_copy" );
                copy_first_backward_self_( in, out );
            }
            synchronize_streams_( 1, "first_backward/degenerate_copy_complete" );
            return;
        }
        if ( mode_ == mpi_transpose_3d_mode::p2p_waitall || egger_parity_backward_waitall_enabled_() )
            backward_first_waitall_( in, out );
        else
            backward_first_waitany_( in, out );
    }

    template <class Stage1Complex3>
    void pack_first_backward_chunk_( int p, const Stage1Complex3 &in )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *dst  = send_buffer_.raw_ptr() + first_backward_send_offset_( p );
        auto        kind = runtime_api_t::device_to_device_kind();
#else
        value_type *dst  = host_send_buffer_.raw_ptr() + first_backward_send_offset_( p );
        auto        kind = runtime_api_t::device_to_host_kind();
#endif
        pack_first_backward_chunk_async_(
            in.raw_ptr(), half_input_dim_.start_y[p], half_input_dim_.size_y[p], dst, kind, streams_[p].stream()
        );
    }

    template <class Stage1Complex3, class Stage0Complex3>
    void copy_first_backward_self_( const Stage1Complex3 &in, Stage0Complex3 &out )
    {
        pack_first_backward_chunk_async_(
            in.raw_ptr(), half_input_dim_.start_y[myid_j_], ny_input_local_,
            out.raw_ptr() + first_backward_recv_offset_( myid_j_ ), runtime_api_t::device_to_device_kind(),
            streams_[myid_j_].stream()
        );
    }

    template <class Stage0Complex3>
    void copy_first_backward_recv_to_out_( int p, Stage0Complex3 &out )
    {
        runtime_api_t::memcpy_async(
            out.raw_ptr() + first_backward_recv_offset_( p ),
            recv_buffer_.raw_ptr() + first_backward_packed_recv_offset_( p ),
            bytes_from_elems_( first_backward_recv_elems_( p ) ), runtime_api_t::device_to_device_kind(),
            streams_[p].stream()
        );
    }

    template <class Stage0Complex3>
    void copy_first_backward_host_recv_to_out_( int p, Stage0Complex3 &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        runtime_api_t::memcpy_async(
            out.raw_ptr() + first_backward_recv_offset_( p ),
            host_recv_buffer_.raw_ptr() + first_backward_packed_recv_offset_( p ),
            bytes_from_elems_( first_backward_recv_elems_( p ) ), runtime_api_t::host_to_device_kind(),
            streams_[p].stream()
        );
#else
        (void)p;
        (void)out;
#endif
    }

    template <class Stage0Complex3>
    void backward_first_post_recvs_(
        Stage0Complex3 &out, std::vector<mpi_request_t> &byte_recv_requests, std::vector<int> *byte_recv_peers = nullptr,
        std::vector<int> *byte_recv_remaining = nullptr
    )
    {
        const int row_size = row_comm_info_.num_procs;
        (void)row_size;
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        int persistent_byte_recv_index = 0;
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state = nullptr;
        if ( persistent_byte_p2p_enabled_() )
        {
            detail::reset_persistent_recv_requests(
                persistent_first_backward_recv_, first_backward_byte_recv_chunks_()
            );
            persistent_byte_recv_state = &persistent_first_backward_recv_;
        }
        else if ( use_persistent_p2p_ )
        {
            detail::reset_persistent_recv_requests( persistent_first_backward_recv_, row_size );
        }
#else
        if ( use_persistent_p2p_ )
            detail::reset_persistent_recv_requests( persistent_first_backward_recv_, row_size );
#endif
        for ( const int p : row_comm_order_ )
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            value_type *recv_ptr = backward_receive_lands_in_output_()
                                      ? out.raw_ptr() + first_backward_recv_offset_( p )
                                      : recv_buffer_.raw_ptr() + first_backward_packed_recv_offset_( p );
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                post_byte_irecv_(
                    row_comm_info_, recv_ptr, bytes_from_elems_( first_backward_recv_elems_( p ) ), p, myid_j_,
                    byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
                    &persistent_byte_recv_index
                );
            }
            else
            {
                post_value_irecv_(
                    row_comm_info_, recv_ptr, first_backward_recv_elems_( p ), p, myid_j_, row_recv_requests_[p],
                    persistent_first_backward_recv_, p
                );
            }
#else
            post_value_irecv_(
                row_comm_info_, host_recv_buffer_.raw_ptr() + first_backward_packed_recv_offset_( p ),
                first_backward_recv_elems_( p ), p, myid_j_, row_recv_requests_[p], persistent_first_backward_recv_, p
            );
#endif
        }
    }

    int first_backward_byte_send_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : row_comm_order_ )
            total_chunks += mpi_byte_request_count_( bytes_from_elems_( first_backward_send_elems_( p ) ) );
        return total_chunks;
    }

    void prepare_first_backward_send_state_()
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( persistent_byte_p2p_enabled_() )
            detail::reset_persistent_send_requests( persistent_first_backward_send_, first_backward_byte_send_chunks_() );
        else
#endif
        if ( persistent_value_p2p_enabled_() )
            detail::reset_persistent_send_requests( persistent_first_backward_send_, row_comm_info_.num_procs );
    }

    void backward_first_post_send_(
        int p, std::vector<mpi_request_t> &byte_send_requests, int &persistent_byte_index
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const value_type *send_ptr = send_buffer_.raw_ptr() + first_backward_send_offset_( p );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            post_byte_isend_(
                row_comm_info_, send_ptr, bytes_from_elems_( first_backward_send_elems_( p ) ), p, p,
                byte_send_requests, persistent_byte_p2p_enabled_() ? &persistent_first_backward_send_ : nullptr,
                &persistent_byte_index
            );
        }
        else
        {
            post_value_isend_(
                row_comm_info_, send_ptr, first_backward_send_elems_( p ), p, p, row_send_requests_[p],
                persistent_first_backward_send_, p
            );
        }
#else
        post_value_isend_(
            row_comm_info_, host_send_buffer_.raw_ptr() + first_backward_send_offset_( p ),
            first_backward_send_elems_( p ), p, p, row_send_requests_[p], persistent_first_backward_send_, p
        );
#endif
    }

	    void backward_first_post_sends_( std::vector<mpi_request_t> &byte_send_requests )
	    {
	        prepare_first_backward_send_state_();
	        profile_chunk_count_( "first_backward", first_backward_byte_send_chunks_(), first_backward_byte_recv_chunks_() );
	        int persistent_byte_index = 0;
        for ( const int p : row_comm_order_ )
            backward_first_post_send_( p, byte_send_requests, persistent_byte_index );
    }

	    void backward_first_post_sends_when_ready_( std::vector<mpi_request_t> &byte_send_requests )
	    {
	        prepare_first_backward_send_state_();
	        profile_chunk_count_( "first_backward", first_backward_byte_send_chunks_(), first_backward_byte_recv_chunks_() );
	        int persistent_byte_index = 0;
        std::vector<char> posted( row_comm_info_.num_procs, 0 );
        int remaining = static_cast<int>( row_comm_order_.size() );
        while ( remaining > 0 )
        {
            std::vector<int> ready_peers;
            {
                auto phase = profile_scope_( "first_backward/wait_pack_ready" );
                while ( ready_peers.empty() )
                {
                    for ( const int p : row_comm_order_ )
                    {
                        if ( posted[p] )
                            continue;
                        if ( runtime_api_t::stream_ready( streams_[p].stream() ) )
                            ready_peers.push_back( p );
                    }
                }
            }
            for ( const int p : ready_peers )
            {
                auto phase = profile_scope_( "first_backward/mpi_post_send" );
                backward_first_post_send_( p, byte_send_requests, persistent_byte_index );
                posted[p] = 1;
                --remaining;
            }
        }
    }

    template <class Stage1Complex3>
	    void backward_first_pack_sync_send_( const Stage1Complex3 &in, std::vector<mpi_request_t> &byte_send_requests )
	    {
	        prepare_first_backward_send_state_();
	        profile_chunk_count_( "first_backward", first_backward_byte_send_chunks_(), first_backward_byte_recv_chunks_() );
	        int persistent_byte_index = 0;
        for ( const int p : row_comm_order_ )
        {
            {
                auto phase = profile_scope_( "first_backward/pack_peer" );
                pack_first_backward_chunk_( p, in );
            }
            {
                auto phase = profile_scope_( "first_backward/pre_send_stream_sync" );
                runtime_api_t::stream_synchronize( streams_[p].stream() );
            }
            {
                auto phase = profile_scope_( "first_backward/mpi_post_send" );
                backward_first_post_send_( p, byte_send_requests, persistent_byte_index );
            }
        }
    }

    template <class Stage1Complex3, class Stage0Complex3>
    void backward_first_post_recv_pack_sync_send_pairs_(
        const Stage1Complex3 &in, Stage0Complex3 &out, std::vector<mpi_request_t> &byte_recv_requests,
        std::vector<mpi_request_t> &byte_send_requests, std::vector<int> *byte_recv_peers = nullptr,
        std::vector<int> *byte_recv_remaining = nullptr
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        prepare_first_backward_send_state_();
        profile_chunk_count_( "first_backward", first_backward_byte_send_chunks_(), first_backward_byte_recv_chunks_() );
        int persistent_byte_index      = 0;
        int persistent_byte_recv_index = 0;
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state = nullptr;
        for ( const int p : row_comm_order_ )
        {
            value_type *recv_ptr = backward_receive_lands_in_output_()
                                      ? out.raw_ptr() + first_backward_recv_offset_( p )
                                      : recv_buffer_.raw_ptr() + first_backward_packed_recv_offset_( p );
            {
                auto phase = profile_scope_( "first_backward/mpi_post_recv" );
                post_byte_irecv_(
                    row_comm_info_, recv_ptr, bytes_from_elems_( first_backward_recv_elems_( p ) ), p, myid_j_,
                    byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
                    &persistent_byte_recv_index
                );
            }
            {
                auto phase = profile_scope_( "first_backward/pack_peer" );
                pack_first_backward_chunk_( p, in );
            }
            {
                auto phase = profile_scope_( "first_backward/pre_send_stream_sync" );
                runtime_api_t::stream_synchronize( streams_[p].stream() );
            }
            {
                auto phase = profile_scope_( "first_backward/mpi_post_send" );
                backward_first_post_send_( p, byte_send_requests, persistent_byte_index );
            }
        }
#else
        (void)in;
        (void)out;
        (void)byte_recv_requests;
        (void)byte_send_requests;
        (void)byte_recv_peers;
        (void)byte_recv_remaining;
#endif
    }

    template <class Stage1Complex3, class Stage0Complex3>
    void backward_first_waitall_( const Stage1Complex3 &in, Stage0Complex3 &out )
    {
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        const bool                 peer_paired_schedule = peer_paired_large_count_byte_schedule_enabled_();
        if ( peer_paired_schedule )
        {
            auto phase = profile_scope_( "first_backward/recv_pack_sync_send_pairs" );
            backward_first_post_recv_pack_sync_send_pairs_( in, out, byte_recv_requests, byte_send_requests );
        }
        else
        {
            {
                auto phase = profile_scope_( "first_backward/mpi_post_recv" );
                backward_first_post_recvs_( out, byte_recv_requests );
            }
	        if ( egger_byte_sync_enabled_() && !egger_parity_backward_ready_post_send_enabled_() )
	        {
	            auto phase = profile_scope_( "first_backward/pack_sync_send" );
	            backward_first_pack_sync_send_( in, byte_send_requests );
	        }
            else
            {
                {
                    auto phase = profile_scope_( "first_backward/pack_send_chunks" );
                    for ( const int p : row_comm_order_ )
                        pack_first_backward_chunk_( p, in );
                }
                {
                    auto phase = profile_scope_( "first_backward/wait_pack_ready_post_send" );
                    backward_first_post_sends_when_ready_( byte_send_requests );
                }
            }
        }
        {
            auto phase = profile_scope_( "first_backward/self_copy" );
            copy_first_backward_self_( in, out );
        }
        {
            auto phase = profile_scope_( "first_backward/wait_recv" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( row_comm_info_, byte_recv_requests, "owned first byte backward recv requests" );
            else
#endif
                row_comm_info_.waitall( row_size, row_recv_requests_.data() );
        }
        if ( !backward_receive_lands_in_output_() )
        {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            auto phase = profile_scope_( "first_backward/stage_recv_to_device" );
#else
            auto phase = profile_scope_( "first_backward/unpack_recv" );
#endif
            for ( const int p : row_comm_order_ )
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                copy_first_backward_host_recv_to_out_( p, out );
#else
                copy_first_backward_recv_to_out_( p, out );
#endif
        }
        synchronize_streams_( row_size, "first_backward/recv_copy_complete" );
        {
            auto phase = profile_scope_( "first_backward/wait_send" );
            wait_send_requests_(
                row_comm_info_, byte_send_requests, persistent_first_backward_send_, row_send_requests_, row_size,
                "owned first byte backward send requests", "owned first persistent backward send requests"
            );
        }
    }

    template <class Stage1Complex3, class Stage0Complex3>
    void backward_first_waitany_( const Stage1Complex3 &in, Stage0Complex3 &out )
    {
        const int row_size = row_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( row_size, 0 );
        const bool                 peer_paired_schedule = peer_paired_large_count_byte_schedule_enabled_();
        if ( peer_paired_schedule )
        {
            auto phase = profile_scope_( "first_backward/recv_pack_sync_send_pairs" );
            backward_first_post_recv_pack_sync_send_pairs_(
                in, out, byte_recv_requests, byte_send_requests, &byte_recv_peers, &byte_recv_remaining
            );
        }
        else
        {
            {
                auto phase = profile_scope_( "first_backward/mpi_post_recv" );
                backward_first_post_recvs_( out, byte_recv_requests, &byte_recv_peers, &byte_recv_remaining );
            }
	        if ( egger_byte_sync_enabled_() && !egger_parity_backward_ready_post_send_enabled_() )
	        {
	            auto phase = profile_scope_( "first_backward/pack_sync_send" );
	            backward_first_pack_sync_send_( in, byte_send_requests );
	        }
            else
            {
                {
                    auto phase = profile_scope_( "first_backward/pack_send_chunks" );
                    for ( const int p : row_comm_order_ )
                        pack_first_backward_chunk_( p, in );
                }
                {
                    auto phase = profile_scope_( "first_backward/wait_pack_ready_post_send" );
                    backward_first_post_sends_when_ready_( byte_send_requests );
                }
            }
        }
        {
            auto phase = profile_scope_( "first_backward/self_copy" );
            copy_first_backward_self_( in, out );
        }

#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const int total =
                detail::mpi_int_cast( byte_recv_requests.size(), "owned first byte backward waitany request count" );
            int completed = 0;
            while ( completed < total )
            {
                int idx = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "first_backward/wait_recv_any" );
                    idx        = row_comm_info_.waitany( total, byte_recv_requests.data() );
                }
                if ( idx == MPI_UNDEFINED )
                    break;
                const int p = byte_recv_peers[idx];
                null_completed_request_( byte_recv_requests, idx );
                --byte_recv_remaining[p];
                if ( byte_recv_remaining[p] == 0 && !backward_receive_lands_in_output_() )
                {
                    auto phase = profile_scope_( "first_backward/unpack_recv_chunk" );
                    copy_first_backward_recv_to_out_( p, out );
                }
                ++completed;
            }
        }
        else
#endif
        {
            int completed = 0;
            while ( completed < row_size - 1 )
            {
                int p = MPI_UNDEFINED;
                {
                    auto phase = profile_scope_( "first_backward/wait_recv_any" );
                    p          = row_comm_info_.waitany( row_size, row_recv_requests_.data() );
                }
                if ( p == MPI_UNDEFINED )
                    break;
                row_recv_requests_[p] = mpi_request_t();
                if ( !backward_receive_lands_in_output_() )
                {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                    auto phase = profile_scope_( "first_backward/stage_recv_chunk_to_device" );
                    copy_first_backward_host_recv_to_out_( p, out );
#else
                    auto phase = profile_scope_( "first_backward/unpack_recv_chunk" );
                    copy_first_backward_recv_to_out_( p, out );
#endif
                }
                ++completed;
            }
        }
        synchronize_streams_( row_size, "first_backward/recv_copy_complete" );
        {
            auto phase = profile_scope_( "first_backward/wait_send" );
            wait_send_requests_(
                row_comm_info_, byte_send_requests, persistent_first_backward_send_, row_send_requests_, row_size,
                "owned first byte backward send requests", "owned first persistent backward send requests"
            );
        }
    }

    base_fft_t        &base_fft_;
    MPIComm            mpi_;
    Log                log_;
    OptionalProfiler  &optional_profiler_;
    profiler_t        *profiler_        = nullptr;
    memory_profiler_t *memory_profiler_ = nullptr;
    std::string        memory_profile_prefix_;

    bool is_inited_                  = false;
    bool use_external_work_area_     = false;
    bool use_external_host_work_area_ = false;
    bool use_direct_backward_receive_ = false;
    bool direct_p2p_cuda_aware_       = true;
    bool use_p2p_send_thread_         = false;
	    bool use_p2p_byte_transfer_       = false;
	    bool use_persistent_p2p_          = false;
	    bool use_ready_p2p_send_          = false;
	    bool print_schedule_              = false;
	    bool egger_parity_enabled_        = false;
    bool use_direct_forward_byte_receive_ = false;
    ::fftm::fftm_3d_large_count_p2p_transport large_count_p2p_transport_ =
        ::fftm::fftm_3d_large_count_p2p_transport::hindexed;
	    int  pencil_layout_selector_      = 0;
    void *external_work_area_         = nullptr;
    void *external_host_work_area_    = nullptr;

    mpi_transpose_3d_mode mode_ = mpi_transpose_3d_mode::p2p_waitany;
    int myid_i_ = 0;
    int myid_j_ = 0;

    std::size_t nx_local_  = 0;
    std::size_t ny_input_local_  = 0;
    std::size_t ny_output_local_ = 0;
    std::size_t ny_global_       = 0;
    std::size_t nz_global_ = 0;
    std::size_t nz_local_  = 0;
    std::size_t nx_global_ = 0;

    partition_t half_input_dim_;
    partition_t transpose1_dim_;
    partition_t output_dim_;

    std::unique_ptr<mpi_comm_t> row_comm_;
    std::unique_ptr<mpi_comm_t> line_comm_;
    scfd::communication::mpi_comm_info row_comm_info_;
    scfd::communication::mpi_comm_info line_comm_info_;

    contiguous_buf_t send_buffer_;
    contiguous_buf_t recv_buffer_;
    std::size_t send_buffer_elems_ = 0;
    std::size_t recv_buffer_elems_ = 0;
    host_buf_t host_send_buffer_;
    host_buf_t host_recv_buffer_;
    std::size_t host_send_buffer_elems_ = 0;
    std::size_t host_recv_buffer_elems_ = 0;

    std::vector<mpi_request_t> row_send_requests_;
    std::vector<mpi_request_t> row_recv_requests_;
    std::vector<mpi_request_t> line_send_requests_;
    std::vector<mpi_request_t> line_recv_requests_;
    detail::persistent_send_requests<mpi_request_t> persistent_first_forward_send_;
    detail::persistent_send_requests<mpi_request_t> persistent_second_forward_send_;
    detail::persistent_send_requests<mpi_request_t> persistent_second_backward_send_;
    detail::persistent_send_requests<mpi_request_t> persistent_first_backward_send_;
    detail::persistent_recv_requests<mpi_request_t> persistent_first_forward_recv_;
    detail::persistent_recv_requests<mpi_request_t> persistent_second_forward_recv_;
    detail::persistent_recv_requests<mpi_request_t> persistent_second_backward_recv_;
    detail::persistent_recv_requests<mpi_request_t> persistent_first_backward_recv_;
    std::vector<mpi_dtype_t> first_forward_direct_recvtypes_;
    std::vector<mpi_dtype_t> second_forward_direct_recvtypes_;
    std::vector<int> row_comm_order_;
    std::vector<int> line_comm_order_;
    std::vector<typename runtime_api_t::stream_wrap> streams_;
    mpi_dtype_t mpi_value_type_;
};

} // namespace detail
} // namespace fftm

#endif
