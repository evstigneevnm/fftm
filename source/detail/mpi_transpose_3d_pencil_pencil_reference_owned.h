#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_H__

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/utils/safe_call.h>
#include <scfd/utils/system_timer_event.h>

#include "../fft_direction.h"
#include "../fft_partitioning.h"
#include "../profiling.h"
#include "mpi_transpose_3d.h"
#include "mpi_transpose_3d_pencil_pencil_plan_state.h"

namespace fftm
{

struct fftm_native_stage_timing
{
    std::string stage;
    double      ms = 0.0;
};

enum class fftm_3d_large_count_p2p_transport
{
    hindexed,
    mpi_count,
    element_count,
    chunked
};

enum class fftm_3d_contiguous_forward_send_mode
{
    single,
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

inline const char *fftm_3d_contiguous_forward_send_mode_name( fftm_3d_contiguous_forward_send_mode mode )
{
    switch ( mode )
    {
    case fftm_3d_contiguous_forward_send_mode::single:
        return "single";
    case fftm_3d_contiguous_forward_send_mode::chunked:
        return "chunked";
    }
    return "unknown";
}

namespace detail
{

template <class BaseFFT, class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI, class OptionalProfiler>
class mpi_transpose_3d_pencil_pencil_reference_owned
{
public:
    using base_fft_t        = BaseFFT;
    using value_type        = ValueType;
    using backend_t         = Backend;
    using memory_t          = typename backend_t::memory_type;
    using host_memory_t     = typename memory_t::host_memory_type;
    using runtime_api_t     = RuntimeAPI;
    using partition_t       = ::fftm::partition;
    using plan_state_t      = ::fftm::detail::mpi_transpose_3d_pencil_pencil_plan_state;
    using mpi_comm_t        = scfd::communication::mpi_comm;
    using contiguous_buf_t  = scfd::arrays::array_nd<value_type, 1, memory_t>;
    using host_buf_t        = scfd::arrays::array_nd<value_type, 1, host_memory_t>;
    using mpi_request_t     = scfd::communication::detail::mpi_request;
    using mpi_dtype_t       = scfd::communication::detail::mpi_data_type;
    using profiler_t        = ::fftm::fftm_profiler;
    using profiler_scope_t  = ::fftm::profile_scope<profiler_t>;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

private:
    enum class deferred_send_stage
    {
        none,
        forward_first,
        forward_second,
        backward_second,
        backward_first
    };

    enum class deferred_send_communicator
    {
        row,
        line
    };

    struct deferred_send_state
    {
        deferred_send_stage        stage = deferred_send_stage::none;
        deferred_send_communicator communicator = deferred_send_communicator::row;
        std::vector<mpi_request_t> requests;
        int                        remaining = 0;

        bool active() const
        {
            return stage != deferred_send_stage::none;
        }

        void reset()
        {
            stage     = deferred_send_stage::none;
            remaining = 0;
            requests.clear();
        }
    };

public:
    using plan_sequence_id_t = typename base_fft_t::plan_sequence_id_t;
    using c2c_plan_array_id_t = typename base_fft_t::c2c_plan_array_id_t;

    struct large_byte_datatype_cache
    {
        std::vector<mpi_dtype_t>   datatypes;
        std::vector<std::size_t>   bytes;
        std::vector<int>           blocks;
        int                        active_count = 0;
        int                        total_blocks = 0;
    };

    struct native_schedule_slot
    {
        std::size_t offset_elems = 0;
        std::size_t elems        = 0;
        int         mpi_peer     = 0;
        int         mpi_tag      = 0;
        bool        valid        = false;
    };

    mpi_transpose_3d_pencil_pencil_reference_owned(
        base_fft_t &base_fft, const MPIComm &mpi, const Log &log, OptionalProfiler &profiler
    )
        : base_fft_( base_fft ), mpi_( mpi ), log_( log ), optional_profiler_( profiler )
    {
        static_assert(
            std::is_same<memory_t, typename runtime_api_t::memory_type>::value,
            "mpi_transpose_3d_pencil_pencil_reference_owned requires matching backend/runtime memory types"
        );
    }

    ~mpi_transpose_3d_pencil_pencil_reference_owned()
    {
        drain_deferred_send_noexcept_();
        free_persistent_send_states_();
        free_large_byte_datatype_caches_();
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
        {
            free_persistent_send_states_();
            free_large_byte_datatype_caches_();
        }
        use_p2p_byte_transfer_ = enabled;
    }

    void set_persistent_p2p_enabled( bool enabled )
    {
        if ( use_persistent_p2p_ != enabled )
        {
            free_persistent_send_states_();
            free_large_byte_datatype_caches_();
        }
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

    void set_stable_forward_byte_send_buffer_enabled( bool enabled )
    {
        use_stable_forward_byte_send_buffer_ = enabled;
    }

    void set_ready_stable_forward_byte_send_buffer_enabled( bool enabled )
    {
        use_ready_stable_forward_byte_send_buffer_ = enabled;
    }

    void set_contiguous_forward_byte_send_enabled( bool enabled )
    {
        use_contiguous_forward_byte_send_ = enabled;
    }

    void set_physical_forward_peer_exchange_enabled( bool enabled )
    {
        use_physical_forward_peer_exchange_ = enabled;
    }

    void set_contiguous_forward_send_mode( ::fftm::fftm_3d_contiguous_forward_send_mode mode )
    {
        contiguous_forward_send_mode_ = mode;
    }

    void set_contiguous_forward_send_chunk_bytes( std::size_t bytes )
    {
        contiguous_forward_send_chunk_bytes_ = bytes;
    }

    void set_large_count_p2p_transport( ::fftm::fftm_3d_large_count_p2p_transport transport )
    {
        if ( large_count_p2p_transport_ != transport )
            free_large_byte_datatype_caches_();
        large_count_p2p_transport_ = transport;
    }

    void set_large_count_datatype_cache_enabled( bool enabled )
    {
        if ( use_large_count_datatype_cache_ != enabled )
            free_large_byte_datatype_caches_();
        use_large_count_datatype_cache_ = enabled;
    }

    void set_native_backward_second_peer_loop_enabled( bool enabled )
    {
        use_native_backward_second_peer_loop_ = enabled;
    }

    void set_deferred_send_completion_enabled( bool enabled )
    {
        use_deferred_send_completion_ = enabled;
    }

    void set_native_opt0_default_z_layout_enabled( bool enabled )
    {
        native_opt0_default_z_layout_ = enabled;
    }

    void set_native_opt0_reference_y_buffer_topology_enabled( bool enabled )
    {
        use_native_opt0_reference_y_buffer_topology_ = enabled;
    }

    void set_native_opt0_compact_y_workarea_enabled( bool enabled )
    {
        use_native_opt0_compact_y_workarea_ = enabled;
    }

    void set_native_opt0_tight_y_plan_sequence_enabled( bool enabled )
    {
        use_native_opt0_tight_y_plan_sequence_ = enabled;
    }

    void set_native_opt0_shared_y_plan_handles_enabled( bool enabled )
    {
        use_native_opt0_shared_y_plan_handles_ = enabled;
    }

    void set_native_opt0_y_group_device_sync_enabled( bool enabled )
    {
        use_native_opt0_y_group_device_sync_ = enabled;
    }

    void set_native_opt0_y_no_sync_exec_enabled( bool enabled )
    {
        use_native_opt0_y_no_sync_exec_ = enabled;
    }

    void set_native_opt0_raw_y_plan_array_executor_enabled( bool enabled )
    {
        use_native_opt0_raw_y_plan_array_executor_ = enabled;
    }

    void set_native_opt0_reference_y_plan_lifecycle_enabled( bool enabled )
    {
        use_native_opt0_reference_y_plan_lifecycle_ = enabled;
    }

    void set_native_opt0_reference_y_plan_bundle_enabled( bool enabled )
    {
        use_native_opt0_reference_y_plan_bundle_ = enabled;
    }

    void set_native_opt0_raw_y_plan_bundle_enabled( bool enabled )
    {
        use_native_opt0_raw_y_plan_bundle_ = enabled;
    }

    void set_native_opt0_y_plan_bundle_stream_first_enabled( bool enabled )
    {
        use_native_opt0_y_plan_bundle_stream_first_ = enabled;
    }

    void set_native_opt0_raw_y_plan_bundle_reference_streams_enabled( bool enabled )
    {
        use_native_opt0_raw_y_plan_bundle_reference_streams_ = enabled;
    }

    void set_native_opt0_reference_local_plan_context_enabled( bool enabled )
    {
        use_native_opt0_reference_local_plan_context_ = enabled;
    }

    template <class FFTWrap>
    std::vector<typename FFTWrap::runtime_api::stream_t> native_opt0_y_stream_handles( std::size_t plan_count )
    {
        if ( plan_count > streams_.size() )
            throw std::logic_error( "native opt0 Y stream handle query stream count mismatch" );
        std::vector<typename FFTWrap::runtime_api::stream_t> result;
        result.reserve( plan_count );
        for ( std::size_t i = 0; i < plan_count; ++i )
            result.push_back( streams_[i].stream() );
        return result;
    }

    std::size_t native_opt0_reference_domain_size_bytes() const
    {
        return native_opt0_reference_domain_size_bytes_();
    }

    void set_local_fft_diagnostics( bool enabled, const std::string &directory, const std::string &label )
    {
        local_fft_diagnostics_enabled_ = enabled;
        local_fft_diagnostics_dir_     = directory.empty() ? "." : directory;
        local_fft_diagnostics_label_   = label.empty() ? "fftm-native" : label;
    }

    void set_native_stage_timers_enabled( bool enabled )
    {
        native_stage_timers_enabled_ = enabled;
    }

    void begin_native_stage_timing_iteration( int iteration )
    {
        native_stage_timing_iteration_ = iteration;
        native_stage_timings_.clear();
        native_stage_timing_active_ = native_stage_timers_enabled_;
    }

    void end_native_stage_timing_iteration()
    {
        native_stage_timing_active_ = false;
    }

    const std::vector<::fftm::fftm_native_stage_timing> &native_stage_timings() const
    {
        return native_stage_timings_;
    }

    void dump_local_fft_plan_descriptors()
    {
        if ( !local_fft_diagnostics_enabled_ || local_fft_plan_descriptors_dumped_ )
        {
            return;
        }

        const std::string header =
            "source,run_label,rank,plan_name,direction,cufft_type,rank_fft,default_layout,n0,n1,n2,inembed0,inembed1,inembed2,"
            "istride,idist,onembed0,onembed1,onembed2,ostride,odist,batch,work_size_bytes,work_area_token,"
            "stream_token,directory";
        for ( const auto &plan : base_fft_.plan_descriptors() )
        {
            const auto &desc = plan.second;
            std::ostringstream row;
            row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << plan.first << ','
                << local_fft_direction_name_( desc.dir ) << ',' << desc.cufft_type << ',' << desc.rank << ','
                << ( desc.default_layout ? 1 : 0 );
            for ( int i = 0; i < 3; ++i )
            {
                row << ',' << desc.n[static_cast<std::size_t>( i )];
            }
            for ( int i = 0; i < 3; ++i )
            {
                row << ',' << desc.inembed[static_cast<std::size_t>( i )];
            }
            row << ',' << desc.istride << ',' << desc.idist;
            for ( int i = 0; i < 3; ++i )
            {
                row << ',' << desc.onembed[static_cast<std::size_t>( i )];
            }
            row << ',' << desc.ostride << ',' << desc.odist << ',' << desc.batch << ',' << desc.work_size << ','
                << static_cast<unsigned long long>( desc.work_area_token ) << ','
                << static_cast<unsigned long long>( desc.stream_token ) << ',' << local_fft_diagnostics_dir_;
            append_local_fft_diagnostics_row_( "local_fft_plans", header, row.str() );
        }

        local_fft_plan_descriptors_dumped_ = true;
    }

    void set_pencil_layout_selector( int selector )
    {
        pencil_layout_selector_ = selector;
    }

    void set_reference_parity_enabled( bool enabled )
    {
        reference_parity_enabled_ = enabled;
        if ( !enabled )
            return;

        free_persistent_send_states_();
        free_large_byte_datatype_caches_();
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
                "mpi_transpose_3d_pencil_pencil_reference_owned::set_external_work_area: external work area disabled"
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
                "mpi_transpose_3d_pencil_pencil_reference_owned::set_external_host_work_area: external host work area disabled"
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
        native_plan_state_bound_ = false;
        init_impl_( mode, half_input_dim, transpose1_dim, output_dim, myid_i, myid_j, use_persistent_p2p );
    }

    void init(
        mpi_transpose_3d_mode mode, const plan_state_t &plan_state, const partition_t &expected_half_input_dim,
        const partition_t &expected_transpose1_dim, const partition_t &expected_output_dim, int expected_myid_i,
        int expected_myid_j, bool use_persistent_p2p
    )
    {
        validate_native_plan_state_inputs_(
            plan_state, expected_half_input_dim, expected_transpose1_dim, expected_output_dim, expected_myid_i,
            expected_myid_j
        );
        native_plan_state_       = plan_state;
        native_plan_state_bound_ = true;
        init_impl_(
            mode, native_plan_state_.half_input_partition(), native_plan_state_.stage1_partition(),
            native_plan_state_.output_partition(), native_plan_state_.rank_i(), native_plan_state_.rank_j(),
            use_persistent_p2p
        );
    }

private:
    void init_impl_(
        mpi_transpose_3d_mode mode, const partition_t &half_input_dim, const partition_t &transpose1_dim,
        const partition_t &output_dim, int myid_i, int myid_j, bool use_persistent_p2p
    )
    {
        auto scope = profile_scope_( "mpi_transpose_3d_pencil_pencil_reference_owned::init" );
        ensure_no_deferred_send_( "initialization" );
        if ( mode != mpi_transpose_3d_mode::p2p_waitall && mode != mpi_transpose_3d_mode::p2p_waitany )
        {
            throw std::logic_error( "owned reference pencil-pencil pipeline currently supports only p2p modes" );
        }

        free_persistent_send_states_();
        free_large_byte_datatype_caches_();
        free_forward_direct_recvtypes_();
        free_value_type_();
        clear_native_plan_schedule_slots_();
        native_opt0_y_plan_array_bundle_ = base_fft_t::invalid_c2c_plan_array_id();
        mode_               = mode;
        half_input_dim_     = half_input_dim;
        transpose1_dim_     = transpose1_dim;
        output_dim_         = output_dim;
        myid_i_             = myid_i;
        myid_j_             = myid_j;
        use_persistent_p2p_ = use_persistent_p2p;

        validate_partitions_();
        validate_large_count_p2p_transport_();
        validate_reference_parity_();

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
            throw std::logic_error( "owned reference pencil row communicator size mismatch" );
        }
        if ( line_comm_info_.num_procs != static_cast<int>( transpose1_dim_.size_x.size() ) )
        {
            throw std::logic_error( "owned reference pencil line communicator size mismatch" );
        }
        build_peer_orders_();
        validate_native_plan_state_after_init_();
        build_native_plan_schedule_slots_();

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

        int max_comm = std::max( row_comm_info_.num_procs, line_comm_info_.num_procs );
        if ( native_opt0_default_z_layout_ )
        {
            const std::size_t native_y_streams = std::min( nx_local_, nz_local_ );
            if ( native_y_streams > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
            {
                throw std::overflow_error( "native opt0 y-plan stream count exceeds int range" );
            }
            max_comm = std::max( max_comm, static_cast<int>( native_y_streams ) );
        }
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
        init_large_byte_datatype_caches_();
        bind_external_work_area_();
        bind_external_host_work_area_();
        is_inited_ = true;
        log_transport_if_enabled_();
        dump_schedule_if_enabled_();
        update_memory_profile_();
    }

public:
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
	            reference_parity_enabled_ ? "fftm::forward_3d_pencil_pencil_reference_parity_pipeline"
	                                  : "fftm::forward_3d_pencil_pencil_reference_owned_pipeline"
	        );
	        exec_local_fft_( "reference_owned/forward_z_fft", "forward_z", in, stage0 );
	        SCFD_SAFE_CALL( forward_from_stage0(
	            stage0, stage1, stage1_xfast, x_fft_stage, out, mode
	        ) );
	    }

    template <class ArrayIn, class ArrayOut>
    void exec_local_fft( const char *profile_name, const char *plan_name, const ArrayIn &in, ArrayOut &out )
    {
        exec_local_fft_( profile_name, plan_name, in, out );
    }

    template <class ArrayIn, class ArrayOut>
    void exec_backward_z_with_deferred_send( const ArrayIn &in, ArrayOut &out )
    {
        execute_with_deferred_send_overlap_( deferred_send_stage::backward_first, [&]() {
            exec_local_fft_( "reference_owned/backward_z_fft", "inverse_z", in, out );
        } );
    }

    template <class ArrayIn, class ArrayOut>
    void exec_local_fft_many_offsets(
        const char *profile_name, const char *plan_label, const std::vector<std::string> &plan_names,
        const std::vector<std::size_t> &offsets, const ArrayIn &in, ArrayOut &out
    )
    {
        exec_local_fft_many_offsets_( profile_name, plan_label, plan_names, offsets, in, out );
    }

    template <class FFTWrap>
    void bind_native_opt0_y_parallel_fft_plans(
        FFTWrap &base_fft, const std::vector<std::string> &forward_plan_names,
        const std::vector<std::string> &inverse_plan_names, std::size_t minimum_work_stride_bytes = 0
    )
    {
        if ( !native_opt0_default_z_layout_ )
            return;
        if ( !is_inited_ )
            throw std::logic_error( "native opt0 y-plan binding requires initialized pencil executor" );
        if ( !use_external_work_area_ || external_work_area_ == nullptr )
            throw std::logic_error( "native opt0 y-plan binding requires an external shared work area" );
        if ( forward_plan_names.size() != inverse_plan_names.size() )
            throw std::logic_error( "native opt0 y-plan binding forward/inverse count mismatch" );
        if ( forward_plan_names.size() > streams_.size() )
            throw std::logic_error( "native opt0 y-plan binding stream count mismatch" );
        if ( use_native_opt0_shared_y_plan_handles_ )
        {
            for ( std::size_t i = 0; i < forward_plan_names.size(); ++i )
            {
                if ( forward_plan_names[i] != inverse_plan_names[i] )
                {
                    throw std::logic_error(
                        "native opt0 shared Y plan handles require forward/inverse plan-name identity"
                    );
                }
            }
        }

        std::size_t work_stride = minimum_work_stride_bytes;
        for ( std::size_t i = 0; i < forward_plan_names.size(); ++i )
        {
            const auto forward_desc = base_fft.plan_descriptor( forward_plan_names[i] );
            const auto inverse_desc = base_fft.plan_descriptor( inverse_plan_names[i] );
            work_stride = std::max( work_stride, std::max( forward_desc.work_size, inverse_desc.work_size ) );
        }

        char             *raw         = static_cast<char *>( external_work_area_ );
        const std::size_t base_offset = native_opt0_y_reference_work_base_offset_bytes();
        for ( std::size_t i = 0; i < forward_plan_names.size(); ++i )
        {
            const std::size_t work_offset = checked_add_bytes_(
                base_offset, checked_mul_bytes_( i, work_stride, "native opt0 y-plan workarea offset" ),
                "native opt0 y-plan workarea offset"
            );
            void *work_area = static_cast<void *>( raw + work_offset );
            if ( use_native_opt0_reference_y_plan_lifecycle_ )
            {
                base_fft.recreate_plan_with_stream_and_work_area(
                    forward_plan_names[i], streams_[i].stream(), work_area
                );
                if ( inverse_plan_names[i] != forward_plan_names[i] )
                {
                    base_fft.recreate_plan_with_stream_and_work_area(
                        inverse_plan_names[i], streams_[i].stream(), work_area
                    );
                }
            }
            else
            {
                base_fft.set_work_area( forward_plan_names[i], work_area );
                base_fft.set_work_area( inverse_plan_names[i], work_area );
                base_fft.set_stream( forward_plan_names[i], streams_[i].stream() );
                base_fft.set_stream( inverse_plan_names[i], streams_[i].stream() );
            }
        }

        native_opt0_forward_y_plan_sequence_ = base_fft.make_plan_sequence( forward_plan_names );
        native_opt0_inverse_y_plan_sequence_ = base_fft.make_plan_sequence( inverse_plan_names );
        native_opt0_y_parallel_plan_count_ = forward_plan_names.size();

        append_native_opt0_y_workarea_meta_( forward_plan_names.size(), base_offset, work_stride );

        const std::size_t total_span =
            native_opt0_y_reference_workspace_size_bytes( forward_plan_names.size(), work_stride );
        std::ostringstream ss;
        ss << "native_opt0_y_plan_sequence active=1 plans=" << native_opt0_y_parallel_plan_count_
           << " forward_sequence=" << native_opt0_forward_y_plan_sequence_
           << " inverse_sequence=" << native_opt0_inverse_y_plan_sequence_
           << " executor=" << ( use_native_opt0_raw_y_plan_array_executor_ ? "opaque-raw" :
                                 use_native_opt0_tight_y_plan_sequence_ ? "tight" : "cached-virtual" )
           << " exec_check=" << ( use_native_opt0_y_no_sync_exec_ ? "no-sync" : "sync" )
           << " handle_mode=" << ( use_native_opt0_shared_y_plan_handles_ ? "shared-bidir" : "separate" )
           << " lifecycle=" << ( use_native_opt0_reference_y_plan_lifecycle_ ? "reference" : "standard" )
           << " workarea_base_offset_bytes=" << base_offset
           << " workarea_stride_bytes=" << work_stride
           << " workarea_total_span_bytes=" << total_span;
        log_.info( ss.str() );
    }

    template <class FFTWrap>
    void bind_native_opt0_y_plan_array_bundle(
        FFTWrap &base_fft, typename FFTWrap::c2c_plan_array_id_t bundle_id,
        std::size_t minimum_work_stride_bytes = 0
    )
    {
        if ( !native_opt0_default_z_layout_ )
            return;
        if ( !is_inited_ )
            throw std::logic_error( "native opt0 Y plan-array binding requires initialized pencil executor" );
        if ( !use_external_work_area_ || external_work_area_ == nullptr )
            throw std::logic_error( "native opt0 Y plan-array binding requires an external shared work area" );
        if ( bundle_id == FFTWrap::invalid_c2c_plan_array_id() )
            throw std::logic_error( "native opt0 Y plan-array binding received an invalid bundle id" );

        const std::size_t plan_count  = base_fft.c2c_plan_array_size( bundle_id );
        const std::size_t work_stride =
            std::max( minimum_work_stride_bytes, base_fft.c2c_plan_array_work_size( bundle_id ) );
        char             *raw         = static_cast<char *>( external_work_area_ );
        const std::size_t base_offset = native_opt0_y_reference_work_base_offset_bytes();
        if ( use_native_opt0_raw_y_plan_bundle_ )
        {
            if ( plan_count > streams_.size() )
                throw std::logic_error( "native opt0 raw Y plan-array binding stream count mismatch" );
            std::vector<typename FFTWrap::runtime_api::stream_t> stream_handles;
            stream_handles.reserve( plan_count );
            for ( std::size_t i = 0; i < plan_count; ++i )
                stream_handles.push_back( streams_[i].stream() );
            if ( use_native_opt0_y_plan_bundle_stream_first_ && !use_native_opt0_raw_y_plan_bundle_reference_streams_ )
                base_fft.bind_c2c_plan_array_streams( bundle_id, stream_handles );
            base_fft.bind_c2c_plan_array_work_areas( bundle_id, static_cast<void *>( raw + base_offset ), work_stride );
            if ( !use_native_opt0_y_plan_bundle_stream_first_ && !use_native_opt0_raw_y_plan_bundle_reference_streams_ )
                base_fft.bind_c2c_plan_array_streams( bundle_id, stream_handles );
        }
        else
        {
            base_fft.bind_c2c_plan_array_work_areas( bundle_id, static_cast<void *>( raw + base_offset ), work_stride );
        }

        native_opt0_forward_y_plan_sequence_ = base_fft_t::invalid_plan_sequence_id();
        native_opt0_inverse_y_plan_sequence_ = base_fft_t::invalid_plan_sequence_id();
        native_opt0_y_plan_array_bundle_     = bundle_id;
        native_opt0_y_parallel_plan_count_   = plan_count;

        append_native_opt0_y_workarea_meta_( plan_count, base_offset, work_stride );

        const std::size_t total_span = native_opt0_y_reference_workspace_size_bytes( plan_count, work_stride );
        std::ostringstream ss;
        ss << "native_opt0_y_plan_bundle active=1 plans=" << native_opt0_y_parallel_plan_count_
           << " bundle_id=" << native_opt0_y_plan_array_bundle_
           << " executor=" << ( use_native_opt0_reference_local_plan_context_ ? "reference-local-context" :
                                use_native_opt0_raw_y_plan_bundle_ ? "raw-reference-bundle" : "reference-bundle" )
           << " exec_check=" << ( use_native_opt0_y_no_sync_exec_ ? "no-sync" : "sync" )
           << " handle_mode=shared-bidir"
           << " lifecycle=" << ( use_native_opt0_reference_local_plan_context_ ? "reference-local-context" :
                                 use_native_opt0_raw_y_plan_bundle_ ? "raw-reference-owned" : "bundle-owned" )
           << " stream_topology=" << ( use_native_opt0_raw_y_plan_bundle_ || use_native_opt0_reference_local_plan_context_ ?
                                       "pencil-external" : "bundle-owned" )
           << " stream_bind_order=" << ( use_native_opt0_y_plan_bundle_stream_first_ ? "stream-before-work" : "work-before-stream" )
           << " raw_reference_streams=" << ( use_native_opt0_raw_y_plan_bundle_reference_streams_ ? 1 : 0 )
           << " reference_local_context=" << ( use_native_opt0_reference_local_plan_context_ ? 1 : 0 )
           << " workarea_base_offset_bytes=" << base_offset
           << " workarea_stride_bytes=" << work_stride
           << " workarea_total_span_bytes=" << total_span;
        log_.info( ss.str() );
    }

    template <class FFTWrap>
    void bind_native_opt0_y_diagnostic_plan_array_bundle(
        FFTWrap &base_fft, typename FFTWrap::c2c_plan_array_id_t bundle_id,
        std::size_t minimum_work_stride_bytes = 0
    )
    {
        if ( !native_opt0_default_z_layout_ )
            return;
        if ( !is_inited_ )
            throw std::logic_error( "native opt0 Y diagnostic plan-array binding requires initialized pencil executor" );
        if ( !use_external_work_area_ || external_work_area_ == nullptr )
            throw std::logic_error( "native opt0 Y diagnostic plan-array binding requires an external shared work area" );
        if ( bundle_id == FFTWrap::invalid_c2c_plan_array_id() )
            throw std::logic_error( "native opt0 Y diagnostic plan-array binding received an invalid bundle id" );

        const std::size_t plan_count  = base_fft.c2c_plan_array_size( bundle_id );
        const std::size_t work_stride =
            std::max( minimum_work_stride_bytes, base_fft.c2c_plan_array_work_size( bundle_id ) );
        if ( plan_count > streams_.size() )
            throw std::logic_error( "native opt0 Y diagnostic plan-array stream count mismatch" );

        char             *raw         = static_cast<char *>( external_work_area_ );
        const std::size_t base_offset = native_opt0_y_reference_work_base_offset_bytes();
        std::vector<typename FFTWrap::runtime_api::stream_t> stream_handles;
        stream_handles.reserve( plan_count );
        for ( std::size_t i = 0; i < plan_count; ++i )
            stream_handles.push_back( streams_[i].stream() );
        if ( use_native_opt0_y_plan_bundle_stream_first_ )
            base_fft.bind_c2c_plan_array_streams( bundle_id, stream_handles );
        base_fft.bind_c2c_plan_array_work_areas( bundle_id, static_cast<void *>( raw + base_offset ), work_stride );
        if ( !use_native_opt0_y_plan_bundle_stream_first_ )
            base_fft.bind_c2c_plan_array_streams( bundle_id, stream_handles );
    }

    template <class FFTWrap, class ArrayIn, class ArrayOut>
    void run_native_opt0_y_same_buffer_microbench(
        FFTWrap &base_fft, const std::vector<std::string> &forward_plan_names,
        const std::vector<std::string> &inverse_plan_names, const std::vector<std::size_t> &offsets,
        const ArrayIn &forward_in, ArrayOut &forward_out,
        typename FFTWrap::c2c_plan_array_id_t diagnostic_raw_bundle,
        value_type *production_backward_y_output_ptr, std::size_t iterations, std::size_t warmup
    )
    {
        if ( !native_opt0_default_z_layout_ )
            throw std::logic_error( "native opt0 Y microbenchmark requires native opt0 default-Z layout" );
        if ( !is_inited_ )
            throw std::logic_error( "native opt0 Y microbenchmark requires initialized pencil executor" );
        const bool have_active_bundle = native_opt0_y_plan_array_bundle_active_();
        if ( offsets.empty() )
            throw std::logic_error( "native opt0 Y microbenchmark received empty Y plan offsets" );
        if ( forward_plan_names.empty() || inverse_plan_names.empty() )
        {
            if ( !have_active_bundle && diagnostic_raw_bundle == FFTWrap::invalid_c2c_plan_array_id() )
            {
                throw std::logic_error( "native opt0 Y microbenchmark received no Y plan handles" );
            }
        }
        else if ( forward_plan_names.size() != offsets.size() || inverse_plan_names.size() != offsets.size() )
        {
            throw std::logic_error( "native opt0 Y microbenchmark received inconsistent Y plan metadata" );
        }
        if ( iterations == 0 )
            iterations = 1;

        const bool have_forward_sequence =
            native_opt0_forward_y_plan_sequence_ != base_fft_t::invalid_plan_sequence_id();
        const bool have_inverse_sequence =
            native_opt0_inverse_y_plan_sequence_ != base_fft_t::invalid_plan_sequence_id();
        if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
        {
            bind_native_opt0_y_diagnostic_plan_array_bundle( base_fft, diagnostic_raw_bundle );
        }

        auto first_desc = typename FFTWrap::plan_descriptor_t{};
        if ( !forward_plan_names.empty() )
            first_desc = base_fft.plan_descriptor( forward_plan_names.front() );
        else if ( have_active_bundle )
            first_desc = base_fft.c2c_plan_array_descriptor( native_opt0_y_plan_array_bundle_ );
        else
            first_desc = base_fft.c2c_plan_array_descriptor( diagnostic_raw_bundle );
        std::size_t descriptor_work_stride = 0;
        if ( !forward_plan_names.empty() )
        {
            for ( std::size_t i = 0; i < offsets.size(); ++i )
            {
                descriptor_work_stride = std::max(
                    descriptor_work_stride,
                    std::max(
                        base_fft.plan_descriptor( forward_plan_names[i] ).work_size,
                        base_fft.plan_descriptor( inverse_plan_names[i] ).work_size
                    )
                );
            }
        }
        else if ( have_active_bundle )
        {
            descriptor_work_stride = base_fft.c2c_plan_array_work_size( native_opt0_y_plan_array_bundle_ );
        }
        else
        {
            descriptor_work_stride = base_fft.c2c_plan_array_work_size( diagnostic_raw_bundle );
        }

        const bool have_reference_workarea =
            native_opt0_reference_y_buffer_topology_enabled_() && external_work_area_ != nullptr;
        value_type *reference_slot0_ptr       = nullptr;
        value_type *reference_slot2_ptr       = nullptr;
        value_type *production_bwd_y_ptr  = production_backward_y_output_ptr;
        value_type *first_y_work_ptr      = nullptr;
        value_type *last_y_work_ptr       = nullptr;
        if ( have_reference_workarea )
        {
            char             *raw              = static_cast<char *>( external_work_area_ );
            const std::size_t y_work_base      = native_opt0_y_reference_work_base_offset_bytes();
            const std::size_t y_work_span      = checked_mul_bytes_(
                offsets.size(), descriptor_work_stride, "native opt0 y microbench work span"
            );
            const std::size_t y_work_end       = checked_add_bytes_(
                y_work_base, y_work_span, "native opt0 y microbench work end"
            );
            const std::size_t after_ywork      = align_up_bytes_( y_work_end, 256 );
            reference_slot0_ptr                    = native_opt0_reference_slot_ptr_( 0 );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            reference_slot2_ptr = native_opt0_reference_slot_ptr_( 2 );
#endif
            if ( production_bwd_y_ptr == nullptr )
            {
                production_bwd_y_ptr = reinterpret_cast<value_type *>( raw + after_ywork );
            }
            first_y_work_ptr      = reinterpret_cast<value_type *>( raw + y_work_base );
            last_y_work_ptr       = offsets.size() > 1
                                  ? reinterpret_cast<value_type *>(
                                        raw + y_work_base + ( offsets.size() - 1 ) * descriptor_work_stride
                                    )
                                  : first_y_work_ptr;
        }

        const auto pointer_token = []( const void *ptr ) -> unsigned long long {
            return static_cast<unsigned long long>( reinterpret_cast<std::uintptr_t>( ptr ) );
        };

	        const auto append_row = [&]( const char *direction_name, const char *variant, double total_ms,
	                                     double event_total_ms, const char *event_timer,
	                                     typename FFTWrap::plan_sequence_id_t sequence_id,
	                                     typename FFTWrap::c2c_plan_array_id_t bundle_id, const void *in_ptr,
	                                     const void *out_ptr ) {
	            const double avg_ms = total_ms / static_cast<double>( iterations );
	            const bool   have_event = event_total_ms >= 0.0;
	            const double event_avg_ms = have_event ? event_total_ms / static_cast<double>( iterations ) : -1.0;
	            const double host_overhead_total_ms = have_event ? total_ms - event_total_ms : 0.0;
	            const double host_overhead_avg_ms =
	                have_event ? host_overhead_total_ms / static_cast<double>( iterations ) : 0.0;
	            const std::string header =
	                "source,run_label,rank,direction,variant,iterations,warmup,total_ms,avg_ms,"
	                "event_total_ms,event_avg_ms,host_overhead_total_ms,host_overhead_avg_ms,event_timer,plan_count,"
	                "sequence_id,bundle_id,n,inembed,istride,idist,onembed,ostride,odist,batch,work_stride_bytes,"
	                "in_token,out_token,work_base_token,first_work_token,last_work_token,reference_slot0_token,"
	                "reference_slot2_token,production_bwd_y_token,first_offset_elems,last_offset_elems,use_tight,"
                "use_shared,use_device_sync,"
	                "use_opaque,use_ref_lifecycle,use_ref_bundle,use_raw_bundle,use_stream_first,"
	                "use_reference_streams,use_local_context,directory";
	            std::ostringstream row;
	            row << "fftm-native-y-micro," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ','
	                << direction_name << ',' << variant << ',' << iterations << ',' << warmup << ',' << total_ms << ','
	                << avg_ms << ',';
	            if ( have_event )
	            {
	                row << event_total_ms << ',' << event_avg_ms << ',' << host_overhead_total_ms << ','
	                    << host_overhead_avg_ms << ',' << event_timer << ',';
	            }
	            else
	            {
	                row << ",,,,none,";
	            }
	            row << offsets.size() << ','
	                << sequence_id << ',' << bundle_id << ',' << first_desc.n[0] << ',' << first_desc.inembed[0] << ','
	                << first_desc.istride << ',' << first_desc.idist << ',' << first_desc.onembed[0] << ','
                << first_desc.ostride << ',' << first_desc.odist << ',' << first_desc.batch << ','
                << descriptor_work_stride << ',' << pointer_token( in_ptr )
                << ',' << pointer_token( out_ptr )
                << ',' << pointer_token( external_work_area_ )
                << ',' << pointer_token( first_y_work_ptr )
                << ',' << pointer_token( last_y_work_ptr )
                << ',' << pointer_token( reference_slot0_ptr )
                << ',' << pointer_token( reference_slot2_ptr )
                << ',' << pointer_token( production_bwd_y_ptr )
                << ',' << offsets.front() << ',' << offsets.back() << ','
                << ( use_native_opt0_tight_y_plan_sequence_ ? 1 : 0 ) << ','
                << ( use_native_opt0_shared_y_plan_handles_ ? 1 : 0 ) << ','
                << ( use_native_opt0_y_group_device_sync_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_raw_y_plan_array_executor_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_reference_y_plan_lifecycle_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_reference_y_plan_bundle_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_raw_y_plan_bundle_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_y_plan_bundle_stream_first_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_raw_y_plan_bundle_reference_streams_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_reference_local_plan_context_ ? 1 : 0 ) << ','
	                << local_fft_diagnostics_dir_;
            append_local_fft_diagnostics_row_( "native_y_microbench", header, row.str() );
        };

	        struct native_y_event_pair
	        {
	            typename runtime_api_t::event_t start = nullptr;
	            typename runtime_api_t::event_t stop  = nullptr;

	            native_y_event_pair()
	            {
	                start = runtime_api_t::event_create();
	                stop  = runtime_api_t::event_create();
	            }

	            ~native_y_event_pair()
	            {
	                runtime_api_t::event_destroy( stop );
	                runtime_api_t::event_destroy( start );
	            }

	            void record_start()
	            {
	                runtime_api_t::event_record( start, runtime_api_t::default_stream() );
	            }

	            void record_stop()
	            {
	                runtime_api_t::event_record( stop, runtime_api_t::default_stream() );
	            }

	            void synchronize_stop()
	            {
	                runtime_api_t::event_synchronize( stop );
	            }

	            double elapsed_ms() const
	            {
	                return runtime_api_t::event_elapsed_time_ms( start, stop );
	            }
	        };

	        const auto run_timed = [&]( const char *direction_name, const char *variant,
	                                    typename FFTWrap::plan_sequence_id_t sequence_id,
	                                    typename FFTWrap::c2c_plan_array_id_t bundle_id, direction exec_direction,
	                                    value_type *in_ptr, value_type *out_ptr, const auto &launch ) {
	            runtime_api_t::device_synchronize();
	            for ( std::size_t i = 0; i < warmup; ++i )
	                launch( exec_direction, in_ptr, out_ptr );
	            runtime_api_t::device_synchronize();

	            native_y_event_pair event_pair;
	            scfd::utils::system_timer_event begin;
	            scfd::utils::system_timer_event end;
	            begin.record();
	            event_pair.record_start();
	            for ( std::size_t i = 0; i < iterations; ++i )
	                launch( exec_direction, in_ptr, out_ptr );
	            event_pair.record_stop();
	            runtime_api_t::device_synchronize();
	            event_pair.synchronize_stop();
	            end.record();
	            append_row(
	                direction_name, variant, end.elapsed_time( begin ), event_pair.elapsed_ms(), "default_stream",
	                sequence_id, bundle_id, in_ptr, out_ptr
	            );
	        };

	        const auto sync_native_y_streams_explicit = [&]( std::size_t count ) {
	            if ( count > streams_.size() )
	                throw std::logic_error( "native opt0 y microbench explicit stream sync count mismatch" );
	            for ( std::size_t i = 0; i < count; ++i )
	                runtime_api_t::stream_synchronize( streams_[i].stream() );
	        };

	        const auto sync_device_explicit = [&]() {
	            runtime_api_t::device_synchronize();
	        };

	        const auto sync_sequence_streams_explicit = [&]() {
	            sync_native_y_streams_explicit( offsets.size() );
	        };

	        const auto run_timed_batched = [&]( const char *direction_name, const char *variant,
	                                            typename FFTWrap::plan_sequence_id_t sequence_id,
	                                            typename FFTWrap::c2c_plan_array_id_t bundle_id,
	                                            direction exec_direction, value_type *in_ptr, value_type *out_ptr,
	                                            const auto &launch_no_sync, const auto &final_sync ) {
	            runtime_api_t::device_synchronize();
	            for ( std::size_t i = 0; i < warmup; ++i )
	                launch_no_sync( exec_direction, in_ptr, out_ptr );
	            final_sync();
	            runtime_api_t::device_synchronize();

	            native_y_event_pair event_pair;
	            scfd::utils::system_timer_event begin;
	            scfd::utils::system_timer_event end;
	            begin.record();
	            event_pair.record_start();
	            for ( std::size_t i = 0; i < iterations; ++i )
	                launch_no_sync( exec_direction, in_ptr, out_ptr );
	            event_pair.record_stop();
	            final_sync();
	            event_pair.synchronize_stop();
	            end.record();
	            append_row(
	                direction_name, variant, end.elapsed_time( begin ), event_pair.elapsed_ms(), "default_stream",
	                sequence_id, bundle_id, in_ptr, out_ptr
	            );
	        };

	        const auto run_timed_repeated = [&]( const char *direction_name, const char *variant,
	                                             typename FFTWrap::plan_sequence_id_t sequence_id,
	                                             typename FFTWrap::c2c_plan_array_id_t bundle_id,
	                                             direction exec_direction, value_type *in_ptr, value_type *out_ptr,
	                                             const auto &repeat_launch_no_sync, const auto &final_sync ) {
	            runtime_api_t::device_synchronize();
	            if ( warmup > 0 )
	                repeat_launch_no_sync( exec_direction, in_ptr, out_ptr, warmup );
	            final_sync();
	            runtime_api_t::device_synchronize();

	            native_y_event_pair event_pair;
	            scfd::utils::system_timer_event begin;
	            scfd::utils::system_timer_event end;
	            begin.record();
	            event_pair.record_start();
	            repeat_launch_no_sync( exec_direction, in_ptr, out_ptr, iterations );
	            event_pair.record_stop();
	            final_sync();
	            event_pair.synchronize_stop();
	            end.record();
	            append_row(
	                direction_name, variant, end.elapsed_time( begin ), event_pair.elapsed_ms(), "default_stream",
	                sequence_id, bundle_id, in_ptr, out_ptr
	            );
	        };

	        const auto launch_current_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                             direction exec_direction, value_type *in_ptr,
	                                                             value_type *out_ptr ) {
	            if ( use_native_opt0_raw_y_plan_array_executor_ )
	            {
	                base_fft.exec_plan_sequence_offsets_opaque_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
            }
            else if ( use_native_opt0_tight_y_plan_sequence_ )
            {
                base_fft.exec_plan_sequence_offsets_tight_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
            }
            else
	            {
	                base_fft.exec_plan_sequence_offsets_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
	            }
	        };

	        const auto repeat_current_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                             direction exec_direction, value_type *in_ptr,
	                                                             value_type *out_ptr, std::size_t repeats ) {
	            if ( use_native_opt0_raw_y_plan_array_executor_ )
	            {
	                base_fft.exec_plan_sequence_offsets_opaque_direction_repeated(
	                    sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	                );
	            }
	            else if ( use_native_opt0_tight_y_plan_sequence_ )
	            {
	                base_fft.exec_plan_sequence_offsets_tight_direction_repeated(
	                    sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	                );
	            }
	            else
	            {
	                base_fft.exec_plan_sequence_offsets_direction_repeated(
	                    sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	                );
	            }
	        };

	        const auto launch_current_sequence = [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	            const auto sequence_id = exec_direction == direction::C2CF ? native_opt0_forward_y_plan_sequence_
	                                                                       : native_opt0_inverse_y_plan_sequence_;
	            launch_current_sequence_id_no_sync( sequence_id, exec_direction, in_ptr, out_ptr );
	            synchronize_native_opt0_y_plan_streams_( offsets.size() );
	        };

	        const auto launch_virtual_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                             direction exec_direction, value_type *in_ptr,
	                                                             value_type *out_ptr ) {
	            base_fft.exec_plan_sequence_offsets_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
	        };

	        const auto repeat_virtual_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                             direction exec_direction, value_type *in_ptr,
	                                                             value_type *out_ptr, std::size_t repeats ) {
	            base_fft.exec_plan_sequence_offsets_direction_repeated(
	                sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	            );
	        };

	        const auto launch_virtual_sequence = [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	            const auto sequence_id = exec_direction == direction::C2CF ? native_opt0_forward_y_plan_sequence_
	                                                                       : native_opt0_inverse_y_plan_sequence_;
	            launch_virtual_sequence_id_no_sync( sequence_id, exec_direction, in_ptr, out_ptr );
	            synchronize_native_opt0_y_plan_streams_( offsets.size() );
	        };

	        const auto launch_tight_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                           direction exec_direction, value_type *in_ptr,
	                                                           value_type *out_ptr ) {
	            base_fft.exec_plan_sequence_offsets_tight_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
	        };

	        const auto repeat_tight_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                           direction exec_direction, value_type *in_ptr,
	                                                           value_type *out_ptr, std::size_t repeats ) {
	            base_fft.exec_plan_sequence_offsets_tight_direction_repeated(
	                sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	            );
	        };

	        const auto launch_tight_sequence = [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	            const auto sequence_id = exec_direction == direction::C2CF ? native_opt0_forward_y_plan_sequence_
	                                                                       : native_opt0_inverse_y_plan_sequence_;
	            launch_tight_sequence_id_no_sync( sequence_id, exec_direction, in_ptr, out_ptr );
	            synchronize_native_opt0_y_plan_streams_( offsets.size() );
	        };

	        const auto launch_opaque_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                            direction exec_direction, value_type *in_ptr,
	                                                            value_type *out_ptr ) {
	            base_fft.exec_plan_sequence_offsets_opaque_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
	        };

	        const auto repeat_opaque_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                            direction exec_direction, value_type *in_ptr,
	                                                            value_type *out_ptr, std::size_t repeats ) {
	            base_fft.exec_plan_sequence_offsets_opaque_direction_repeated(
	                sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	            );
	        };

	        const auto launch_opaque_sequence = [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	            const auto sequence_id = exec_direction == direction::C2CF ? native_opt0_forward_y_plan_sequence_
	                                                                       : native_opt0_inverse_y_plan_sequence_;
	            launch_opaque_sequence_id_no_sync( sequence_id, exec_direction, in_ptr, out_ptr );
	            synchronize_native_opt0_y_plan_streams_( offsets.size() );
	        };

	        const auto launch_bundle_no_sync = [&]( typename FFTWrap::c2c_plan_array_id_t bundle_id,
	                                                direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	            base_fft.exec_c2c_plan_array_direction( bundle_id, exec_direction, in_ptr, out_ptr );
	        };

	        const auto repeat_bundle_no_sync = [&]( typename FFTWrap::c2c_plan_array_id_t bundle_id,
	                                                direction exec_direction, value_type *in_ptr, value_type *out_ptr,
	                                                std::size_t repeats ) {
	            base_fft.exec_c2c_plan_array_direction_repeated( bundle_id, exec_direction, in_ptr, out_ptr, repeats );
	        };

	        const auto launch_bundle = [&]( typename FFTWrap::c2c_plan_array_id_t bundle_id, direction exec_direction,
	                                        value_type *in_ptr, value_type *out_ptr ) {
	            launch_bundle_no_sync( bundle_id, exec_direction, in_ptr, out_ptr );
	            base_fft.synchronize_c2c_plan_array_streams( bundle_id );
	        };

	        value_type *forward_in_ptr  = const_cast<value_type *>( forward_in.raw_ptr() );
	        value_type *forward_out_ptr = forward_out.raw_ptr();

	        const auto prefixed_variant = []( const std::string &prefix, const char *suffix ) {
	            return prefix + suffix;
	        };

	        const auto run_sequence_launch_sync_matrix = [&]( const char *direction_name, const std::string &prefix,
	                                                          typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                          direction exec_direction, value_type *in_ptr,
	                                                          value_type *out_ptr ) {
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-current-final-device" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_current_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-current-final-streams" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_current_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-virtual-final-device" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_virtual_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-virtual-final-streams" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_virtual_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-tight-final-device" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_tight_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-tight-final-streams" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_tight_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-opaque-final-device" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_opaque_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-opaque-final-streams" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_opaque_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-current-final-device" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_current_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-current-final-streams" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_current_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-virtual-final-device" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_virtual_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-virtual-final-streams" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_virtual_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-tight-final-device" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_tight_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-tight-final-streams" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_tight_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-opaque-final-device" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_opaque_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-opaque-final-streams" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_opaque_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	        };

	        const auto run_bundle_launch_sync_matrix = [&]( const char *direction_name, const std::string &prefix,
	                                                        typename FFTWrap::c2c_plan_array_id_t bundle_id,
	                                                        direction exec_direction, value_type *in_ptr,
	                                                        value_type *out_ptr ) {
	            const auto sync_bundle_streams_explicit = [&]() {
	                base_fft.synchronize_c2c_plan_array_streams( bundle_id );
	            };
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-bundle-final-device" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), FFTWrap::invalid_plan_sequence_id(), bundle_id,
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_bundle_no_sync( bundle_id, dir, in_raw, out_raw );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-bundle-final-streams" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), FFTWrap::invalid_plan_sequence_id(), bundle_id,
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_bundle_no_sync( bundle_id, dir, in_raw, out_raw );
	                    },
	                    sync_bundle_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-bundle-final-device" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), FFTWrap::invalid_plan_sequence_id(), bundle_id,
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_bundle_no_sync( bundle_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-bundle-final-streams" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), FFTWrap::invalid_plan_sequence_id(), bundle_id,
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_bundle_no_sync( bundle_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_bundle_streams_explicit
	                );
	            }
	        };

	        if ( have_forward_sequence )
	        {
            run_timed(
                "forward", "seq-current", native_opt0_forward_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, forward_in_ptr, forward_out_ptr,
                launch_current_sequence
            );
            run_timed(
                "forward", "seq-virtual", native_opt0_forward_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, forward_in_ptr, forward_out_ptr,
                launch_virtual_sequence
            );
            run_timed(
                "forward", "seq-tight", native_opt0_forward_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, forward_in_ptr, forward_out_ptr,
                launch_tight_sequence
            );
	            run_timed(
	                "forward", "seq-opaque", native_opt0_forward_y_plan_sequence_,
	                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, forward_in_ptr, forward_out_ptr,
	                launch_opaque_sequence
	            );
	            run_sequence_launch_sync_matrix(
	                "forward", "", native_opt0_forward_y_plan_sequence_, direction::C2CF, forward_in_ptr,
	                forward_out_ptr
	            );
	        }
        if ( native_opt0_y_plan_array_bundle_active_() )
        {
            run_timed(
                "forward", "active-bundle", FFTWrap::invalid_plan_sequence_id(), native_opt0_y_plan_array_bundle_,
                direction::C2CF, forward_in_ptr, forward_out_ptr,
	                [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                    launch_bundle( native_opt0_y_plan_array_bundle_, exec_direction, in_ptr, out_ptr );
	                }
	            );
	            run_bundle_launch_sync_matrix(
	                "forward", "active-", native_opt0_y_plan_array_bundle_, direction::C2CF, forward_in_ptr,
	                forward_out_ptr
	            );
	        }
        if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
        {
            run_timed(
                "forward", "diag-raw-bundle", FFTWrap::invalid_plan_sequence_id(), diagnostic_raw_bundle,
                direction::C2CF, forward_in_ptr, forward_out_ptr,
	                [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                    launch_bundle( diagnostic_raw_bundle, exec_direction, in_ptr, out_ptr );
	                }
	            );
	            run_bundle_launch_sync_matrix(
	                "forward", "diag-raw-", diagnostic_raw_bundle, direction::C2CF, forward_in_ptr, forward_out_ptr
	            );
	        }

        if ( have_reference_workarea )
        {
            value_type *reference_backward_output_ptr = native_opt0_reference_backward_y_output_ptr_();
            if ( have_forward_sequence )
            {
                run_timed(
                    "forward", "reference-workarea-seq-current", native_opt0_forward_y_plan_sequence_,
                    FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, reference_slot0_ptr, forward_out_ptr,
                    launch_current_sequence
                );
	                run_timed(
	                    "forward", "reference-workarea-seq-opaque", native_opt0_forward_y_plan_sequence_,
	                    FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, reference_slot0_ptr, forward_out_ptr,
	                    launch_opaque_sequence
	                );
	                run_sequence_launch_sync_matrix(
	                    "forward", "reference-workarea-", native_opt0_forward_y_plan_sequence_, direction::C2CF,
	                    reference_slot0_ptr, forward_out_ptr
	                );
	            }
            if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
            {
                run_timed(
                    "forward", "reference-workarea-diag-raw-bundle", FFTWrap::invalid_plan_sequence_id(),
                    diagnostic_raw_bundle, direction::C2CF, reference_slot0_ptr, forward_out_ptr,
	                    [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                        launch_bundle( diagnostic_raw_bundle, exec_direction, in_ptr, out_ptr );
	                    }
	                );
	                run_bundle_launch_sync_matrix(
	                    "forward", "reference-workarea-diag-raw-", diagnostic_raw_bundle, direction::C2CF,
	                    reference_slot0_ptr, forward_out_ptr
	                );
	            }

            if ( have_inverse_sequence )
            {
                run_timed(
                    "backward", "reference-workarea-seq-current", native_opt0_inverse_y_plan_sequence_,
                    FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr,
                    reference_backward_output_ptr, launch_current_sequence
                );
	                run_timed(
	                    "backward", "reference-workarea-seq-opaque", native_opt0_inverse_y_plan_sequence_,
	                    FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr,
	                    reference_backward_output_ptr, launch_opaque_sequence
	                );
	                run_sequence_launch_sync_matrix(
	                    "backward", "reference-workarea-", native_opt0_inverse_y_plan_sequence_, direction::C2CB,
	                    forward_out_ptr, reference_backward_output_ptr
	                );
	            }
            if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
            {
                run_timed(
                    "backward", "reference-workarea-diag-raw-bundle", FFTWrap::invalid_plan_sequence_id(),
                    diagnostic_raw_bundle, direction::C2CB, forward_out_ptr, reference_backward_output_ptr,
	                    [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                        launch_bundle( diagnostic_raw_bundle, exec_direction, in_ptr, out_ptr );
	                    }
	                );
	                run_bundle_launch_sync_matrix(
	                    "backward", "reference-workarea-diag-raw-", diagnostic_raw_bundle, direction::C2CB,
	                    forward_out_ptr, reference_backward_output_ptr
	                );
	            }

            if ( production_bwd_y_ptr != nullptr )
            {
                if ( have_inverse_sequence )
                {
                    run_timed(
                        "backward", "after-ywork-seq-current", native_opt0_inverse_y_plan_sequence_,
                        FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr,
                        production_bwd_y_ptr, launch_current_sequence
                    );
	                    run_timed(
	                        "backward", "after-ywork-seq-opaque", native_opt0_inverse_y_plan_sequence_,
	                        FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr,
	                        production_bwd_y_ptr, launch_opaque_sequence
	                    );
	                    run_sequence_launch_sync_matrix(
	                        "backward", "after-ywork-", native_opt0_inverse_y_plan_sequence_, direction::C2CB,
	                        forward_out_ptr, production_bwd_y_ptr
	                    );
	                }
                if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
                {
                    run_timed(
                        "backward", "after-ywork-diag-raw-bundle", FFTWrap::invalid_plan_sequence_id(),
                        diagnostic_raw_bundle, direction::C2CB, forward_out_ptr, production_bwd_y_ptr,
	                        [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                            launch_bundle( diagnostic_raw_bundle, exec_direction, in_ptr, out_ptr );
	                        }
	                    );
	                    run_bundle_launch_sync_matrix(
	                        "backward", "after-ywork-diag-raw-", diagnostic_raw_bundle, direction::C2CB,
	                        forward_out_ptr, production_bwd_y_ptr
	                    );
	                }
	            }
	        }

        if ( have_inverse_sequence )
        {
            run_timed(
                "backward", "seq-current", native_opt0_inverse_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr, forward_in_ptr,
                launch_current_sequence
            );
            run_timed(
                "backward", "seq-virtual", native_opt0_inverse_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr, forward_in_ptr,
                launch_virtual_sequence
            );
            run_timed(
                "backward", "seq-tight", native_opt0_inverse_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr, forward_in_ptr,
                launch_tight_sequence
            );
	            run_timed(
	                "backward", "seq-opaque", native_opt0_inverse_y_plan_sequence_,
	                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr, forward_in_ptr,
	                launch_opaque_sequence
	            );
	            run_sequence_launch_sync_matrix(
	                "backward", "", native_opt0_inverse_y_plan_sequence_, direction::C2CB, forward_out_ptr,
	                forward_in_ptr
	            );
	        }
        if ( native_opt0_y_plan_array_bundle_active_() )
        {
            run_timed(
                "backward", "active-bundle", FFTWrap::invalid_plan_sequence_id(), native_opt0_y_plan_array_bundle_,
                direction::C2CB, forward_out_ptr, forward_in_ptr,
	                [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                    launch_bundle( native_opt0_y_plan_array_bundle_, exec_direction, in_ptr, out_ptr );
	                }
	            );
	            run_bundle_launch_sync_matrix(
	                "backward", "active-", native_opt0_y_plan_array_bundle_, direction::C2CB, forward_out_ptr,
	                forward_in_ptr
	            );
	        }
        if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
        {
            run_timed(
                "backward", "diag-raw-bundle", FFTWrap::invalid_plan_sequence_id(), diagnostic_raw_bundle,
                direction::C2CB, forward_out_ptr, forward_in_ptr,
	                [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                    launch_bundle( diagnostic_raw_bundle, exec_direction, in_ptr, out_ptr );
	                }
	            );
	            run_bundle_launch_sync_matrix(
	                "backward", "diag-raw-", diagnostic_raw_bundle, direction::C2CB, forward_out_ptr, forward_in_ptr
	            );
	        }
    }

    std::size_t native_opt0_y_reference_work_base_offset_bytes() const
    {
        if ( !native_opt0_default_z_layout_ )
            return 0;
        if ( !is_inited_ )
            throw std::logic_error( "native opt0 y-plan workarea query requires initialized pencil executor" );
        if ( !native_opt0_reference_y_buffer_topology_enabled_() )
            return 0;
        return checked_mul_bytes_(
            native_opt0_reference_workspace_slot_offset_(), native_opt0_reference_domain_size_bytes_(),
            "native opt0 y-plan workarea base offset"
        );
    }

    std::size_t native_opt0_y_reference_workspace_size_bytes(
        std::size_t plan_count, std::size_t plan_work_stride_bytes
    ) const
    {
        if ( !native_opt0_default_z_layout_ || plan_count == 0 )
            return 0;
        return checked_add_bytes_(
            native_opt0_y_reference_work_base_offset_bytes(),
            checked_mul_bytes_( plan_count, plan_work_stride_bytes, "native opt0 y-plan workarea span" ),
            "native opt0 y-plan workspace size"
        );
    }

    template <class Stage0Complex3, class Stage1Complex3, class XFastComplex3, class XFFTComplex3, class ComplexArray3>
    void forward_from_stage0(
        Stage0Complex3 &stage0, Stage1Complex3 &stage1, XFastComplex3 &stage1_xfast, XFFTComplex3 &x_fft_stage,
        ComplexArray3 &out, mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
		        time_native_stage_( "reference_owned/forward_first_redistribution", [&]() {
		            auto phase = profile_scope_( "reference_owned/forward_first_redistribution" );
		            SCFD_SAFE_CALL( forward_first_( stage0, stage1 ) );
		        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_first, [&]() {
            exec_local_fft_( "reference_owned/forward_y_fft", "forward_y", stage1, stage1_xfast );
        } );
		        time_native_stage_( "reference_owned/forward_second_redistribution", [&]() {
		            auto phase = profile_scope_( "reference_owned/forward_second_redistribution" );
		            if ( native_forward_second_facade_active_() )
		            {
		                SCFD_SAFE_CALL( native_forward_second_xfast_( stage1_xfast, x_fft_stage ) );
		            }
		            else
		            {
		                SCFD_SAFE_CALL( forward_second_xfast_( stage1_xfast, x_fft_stage ) );
		            }
		        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_second, [&]() {
            exec_local_fft_( "reference_owned/forward_x_fft", "forward_x", x_fft_stage, out );
        } );
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
	            reference_parity_enabled_ ? "fftm::backward_3d_pencil_pencil_reference_parity_pipeline"
	                                  : "fftm::backward_3d_pencil_pencil_reference_owned_pipeline"
	        );
	        exec_local_fft_( "reference_owned/backward_x_fft", "inverse_x", in, x_fft_stage );
	        SCFD_SAFE_CALL( backward_to_stage0(
	            in, x_fft_stage, stage1_xfast, stage1, stage0, mode
	        ) );
        execute_with_deferred_send_overlap_( deferred_send_stage::backward_first, [&]() {
            exec_local_fft_( "reference_owned/backward_z_fft", "inverse_z", stage0, out );
        } );
	    }

    template <class ComplexArray3, class XFFTComplex3, class XFastComplex3, class Stage1Complex3, class Stage0Complex3>
    void backward_to_stage0(
        ComplexArray3 &in, XFFTComplex3 &x_fft_stage, XFastComplex3 &stage1_xfast, Stage1Complex3 &stage1,
        Stage0Complex3 &stage0, mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
		        time_native_stage_( "reference_owned/backward_second_redistribution", [&]() {
		            auto phase = profile_scope_( "reference_owned/backward_second_redistribution" );
		            if ( native_backward_second_facade_active_() )
		            {
		                SCFD_SAFE_CALL( native_backward_second_xfast_( x_fft_stage, stage1_xfast ) );
		            }
		            else
		            {
		                SCFD_SAFE_CALL( backward_second_xfast_( x_fft_stage, stage1_xfast ) );
		            }
		        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::backward_second, [&]() {
            exec_local_fft_( "reference_owned/backward_y_fft", "inverse_y", stage1_xfast, stage1 );
        } );
		        time_native_stage_( "reference_owned/backward_first_redistribution", [&]() {
		            auto phase = profile_scope_( "reference_owned/backward_first_redistribution" );
		            SCFD_SAFE_CALL( backward_first_( stage1, stage0 ) );
		        } );
	    }

    template <class Stage0ZFast3, class Stage1ZFast3, class Stage1YFast3, class ComplexArray3>
    void forward_native_opt0_zfast(
        Stage0ZFast3 &stage0, Stage1ZFast3 &stage1, Stage1YFast3 &stage1_yfft, ComplexArray3 &out,
        const std::vector<std::string> &forward_y_plan_names, const std::vector<std::size_t> &y_plan_offsets,
        mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
        if ( !native_opt0_default_z_layout_ )
            throw std::logic_error( "forward_native_opt0_zfast called without native opt0 layout enabled" );
        auto scope = optional_profiler_.scoped_tic( "fftm::forward_3d_pencil_pencil_native_opt0_zfast_pipeline" );
        time_native_stage_( "reference_owned/forward_first_redistribution", [&]() {
            auto phase = profile_scope_( "reference_owned/forward_first_redistribution" );
            SCFD_SAFE_CALL( forward_first_( stage0, stage1 ) );
        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_first, [&]() {
            if ( native_opt0_y_plan_array_bundle_active_() )
            {
                exec_local_fft_plan_array_bundle_(
                    "reference_owned/forward_y_fft", "forward_y_native_opt0_zfast", stage1, stage1_yfft,
                    direction::C2CF
                );
            }
            else
            {
                exec_local_fft_many_offsets_(
                    "reference_owned/forward_y_fft", "forward_y_native_opt0_zfast", forward_y_plan_names,
                    y_plan_offsets, stage1, stage1_yfft, native_opt0_forward_y_plan_sequence_,
                    use_native_opt0_shared_y_plan_handles_, direction::C2CF
                );
            }
        } );
	        time_native_stage_( "reference_owned/forward_second_redistribution", [&]() {
	            auto phase = profile_scope_( "reference_owned/forward_second_redistribution" );
	            SCFD_SAFE_CALL( forward_second_xfast_( stage1_yfft, out ) );
	        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_second, [&]() {
            exec_local_fft_( "reference_owned/forward_x_fft", "forward_x", out, out );
        } );
    }

    template <class Stage0ZFast3, class Stage1ZFast3, class Stage1YFast3, class XStage3, class ComplexArray3>
    void forward_native_opt0_zfast_reference_buffers(
        Stage0ZFast3 &stage0, Stage1ZFast3 &stage1, Stage1YFast3 &stage1_yfft, XStage3 &x_stage,
        ComplexArray3 &out, const std::vector<std::string> &forward_y_plan_names,
        const std::vector<std::size_t> &y_plan_offsets, mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
        if ( !native_opt0_reference_y_buffer_topology_enabled_() )
            throw std::logic_error(
                "forward_native_opt0_zfast_reference_buffers called without native opt0 reference Y-buffer topology enabled"
            );
        auto scope = optional_profiler_.scoped_tic(
            "fftm::forward_3d_pencil_pencil_native_opt0_reference_y_buffer_topology"
        );
	        debug_native_opt0_topology_marker_( "forward:start" );
	        append_native_opt0_forward_layout_offsets_(
	            stage0, stage1, stage1_yfft, x_stage, out, y_plan_offsets
	        );
	        time_native_stage_( "reference_owned/forward_first_redistribution", [&]() {
	            debug_native_opt0_topology_marker_( "forward:first_transpose:begin" );
	            auto phase = profile_scope_( "reference_owned/forward_first_redistribution" );
	            SCFD_SAFE_CALL( forward_first_( stage0, stage1 ) );
	            debug_native_opt0_topology_marker_( "forward:first_transpose:end" );
        } );
        debug_native_opt0_topology_marker_( "forward:y_fft:begin" );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_first, [&]() {
            if ( native_opt0_y_plan_array_bundle_active_() )
            {
                exec_local_fft_plan_array_bundle_(
                    "reference_owned/forward_y_fft", "forward_y_native_opt0_zfast", stage1, stage1_yfft,
                    direction::C2CF
                );
            }
            else
            {
                exec_local_fft_many_offsets_(
                    "reference_owned/forward_y_fft", "forward_y_native_opt0_zfast", forward_y_plan_names,
                    y_plan_offsets, stage1, stage1_yfft, native_opt0_forward_y_plan_sequence_,
                    use_native_opt0_shared_y_plan_handles_, direction::C2CF
                );
            }
        } );
        debug_native_opt0_topology_marker_( "forward:y_fft:end" );
	        time_native_stage_( "reference_owned/forward_second_redistribution", [&]() {
	            debug_native_opt0_topology_marker_( "forward:second_transpose:begin" );
	            auto phase = profile_scope_( "reference_owned/forward_second_redistribution" );
	            SCFD_SAFE_CALL( forward_second_xfast_( stage1_yfft, x_stage ) );
	            debug_native_opt0_topology_marker_( "forward:second_transpose:end" );
	        } );
        debug_native_opt0_topology_marker_( "forward:x_fft:begin" );
        execute_with_deferred_send_overlap_( deferred_send_stage::forward_second, [&]() {
            exec_local_fft_( "reference_owned/forward_x_fft", "forward_x", x_stage, out );
        } );
        debug_native_opt0_topology_marker_( "forward:x_fft:end" );
    }

    template <class ComplexArray3, class XFFTZFast3, class Stage1YFast3, class Stage1ZFast3, class Stage0ZFast3>
    void backward_native_opt0_zfast(
        ComplexArray3 &in, XFFTZFast3 &x_fft_stage, Stage1YFast3 &stage1_yfft, Stage1ZFast3 &stage1,
        Stage0ZFast3 &stage0, const std::vector<std::string> &inverse_y_plan_names,
        const std::vector<std::size_t> &y_plan_offsets, mpi_transpose_3d_mode mode
    )
    {
        ensure_inited_( mode );
	        if ( !native_opt0_default_z_layout_ )
	            throw std::logic_error( "backward_native_opt0_zfast called without native opt0 layout enabled" );
	        auto scope = optional_profiler_.scoped_tic( "fftm::backward_3d_pencil_pencil_native_opt0_zfast_pipeline" );
	        append_native_opt0_backward_layout_offsets_(
	            in, x_fft_stage, stage1_yfft, stage1, stage0, y_plan_offsets
	        );
	        exec_local_fft_( "reference_owned/backward_x_fft", "inverse_x", in, x_fft_stage );
	        time_native_stage_( "reference_owned/backward_second_redistribution", [&]() {
	            auto phase = profile_scope_( "reference_owned/backward_second_redistribution" );
	            SCFD_SAFE_CALL( backward_second_xfast_( x_fft_stage, stage1_yfft ) );
	        } );
        execute_with_deferred_send_overlap_( deferred_send_stage::backward_second, [&]() {
            if ( native_opt0_y_plan_array_bundle_active_() )
            {
                exec_local_fft_plan_array_bundle_(
                    "reference_owned/backward_y_fft", "inverse_y_native_opt0_zfast", stage1_yfft, stage1,
                    direction::C2CB
                );
            }
            else
            {
                exec_local_fft_many_offsets_(
                    "reference_owned/backward_y_fft", "inverse_y_native_opt0_zfast", inverse_y_plan_names,
                    y_plan_offsets, stage1_yfft, stage1, native_opt0_inverse_y_plan_sequence_,
                    use_native_opt0_shared_y_plan_handles_, direction::C2CB
                );
            }
        } );
	        time_native_stage_( "reference_owned/backward_first_redistribution", [&]() {
	            auto phase = profile_scope_( "reference_owned/backward_first_redistribution" );
	            SCFD_SAFE_CALL( backward_first_( stage1, stage0 ) );
	        } );
    }

		private:
		    profiler_scope_t profile_scope_( const std::string &name )
		    {
		        return profiler_scope_t( profiler_, name );
		    }

        void record_native_stage_timing_( const char *stage, double ms )
        {
            if ( !native_stage_timers_enabled_ || !native_stage_timing_active_ )
                return;
            ::fftm::fftm_native_stage_timing row;
            row.stage = stage;
            row.ms    = ms;
            native_stage_timings_.push_back( row );
        }

        template <class Fn>
        void time_native_stage_( const char *stage, Fn fn )
        {
            if ( !native_stage_timers_enabled_ || !native_stage_timing_active_ )
            {
                fn();
                return;
            }

            runtime_api_t::device_synchronize();
            scfd::utils::system_timer_event begin;
            scfd::utils::system_timer_event end;
            begin.record();
            fn();
            runtime_api_t::device_synchronize();
            end.record();
            record_native_stage_timing_( stage, end.elapsed_time( begin ) );
        }

		    template <class ArrayIn, class ArrayOut>
		    void exec_local_fft_(
		        const char *profile_name, const char *plan_name, const ArrayIn &in, ArrayOut &out
		    )
		    {
		        auto phase = optional_profiler_.scoped_tic( profile_name );
            time_native_stage_( profile_name, [&]() {
                if ( local_fft_diagnostics_enabled_ )
                {
                    using timer_event_t = typename backend_t::timer_event_type;
                    timer_event_t begin;
                    timer_event_t end;
                    begin.record();
                    SCFD_SAFE_CALL( base_fft_.template exec<ArrayIn, ArrayOut>( plan_name, in, out ) );
                    end.record();
                    const double elapsed_ms = end.elapsed_time( begin );
                    append_local_fft_timing_row_( profile_name, plan_name, elapsed_ms );
                }
                else
                {
		            SCFD_SAFE_CALL( base_fft_.template exec<ArrayIn, ArrayOut>( plan_name, in, out ) );
                }
            } );
		        synchronize_after_local_fft_if_needed_();
		    }

    template <class ArrayIn, class ArrayOut>
    void exec_local_fft_many_offsets_(
        const char *profile_name, const char *plan_label, const std::vector<std::string> &plan_names,
        const std::vector<std::size_t> &offsets, const ArrayIn &in, ArrayOut &out,
        plan_sequence_id_t plan_sequence = base_fft_t::invalid_plan_sequence_id(),
        bool use_explicit_exec_direction = false, direction exec_direction = direction::C2CF
    )
    {
        if ( plan_names.size() != offsets.size() )
            throw std::logic_error( "native opt0 z-fast local FFT plan/offset count mismatch" );
        if ( native_opt0_default_z_layout_ && plan_names.size() > streams_.size() )
            throw std::logic_error( "native opt0 z-fast local FFT stream count mismatch" );
        const bool use_cached_sequence = plan_sequence != base_fft_t::invalid_plan_sequence_id();

        auto phase = optional_profiler_.scoped_tic( profile_name );
        const bool dump_y_signature =
            should_dump_native_opt0_y_buffer_signature_( profile_name, plan_label );
        append_native_opt0_y_execution_diagnostics_(
            profile_name, plan_label, plan_names, offsets, in.raw_ptr(), out.raw_ptr(), plan_sequence,
            use_cached_sequence, use_explicit_exec_direction, exec_direction
        );
        if ( dump_y_signature )
        {
            runtime_api_t::device_synchronize();
            append_native_opt0_y_buffer_signature_(
                profile_name, plan_label, "pre", "input", plan_names, offsets, in.raw_ptr()
            );
            append_native_opt0_y_buffer_signature_(
                profile_name, plan_label, "pre", "output", plan_names, offsets, out.raw_ptr()
            );
        }
        time_native_stage_( profile_name, [&]() {
            if ( local_fft_diagnostics_enabled_ )
            {
                using timer_event_t = typename backend_t::timer_event_type;
                timer_event_t begin;
                timer_event_t end;
                begin.record();
                if ( use_cached_sequence )
                {
                    if ( use_native_opt0_y_no_sync_exec_ )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_direction_no_sync(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_no_sync(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                    else if ( use_native_opt0_raw_y_plan_array_executor_ )
                    {
                        SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_opaque_direction(
                            plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                        ) );
                    }
                    else if ( use_native_opt0_tight_y_plan_sequence_ )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_tight_direction(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_tight(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                    else
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_direction(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                }
                else
                {
                    for ( std::size_t i = 0; i < plan_names.size(); ++i )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_raw_direction(
                                plan_names[i], exec_direction, static_cast<void *>( in.raw_ptr() + offsets[i] ),
                                static_cast<void *>( out.raw_ptr() + offsets[i] )
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_raw(
                                plan_names[i], static_cast<void *>( in.raw_ptr() + offsets[i] ),
                                static_cast<void *>( out.raw_ptr() + offsets[i] )
                            ) );
                        }
                    }
                }
                synchronize_native_opt0_y_plan_streams_( plan_names.size() );
                end.record();
                const double elapsed_ms = end.elapsed_time( begin );
                append_local_fft_timing_row_( profile_name, plan_label, elapsed_ms );
            }
            else
            {
                if ( use_cached_sequence )
                {
                    if ( use_native_opt0_y_no_sync_exec_ )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_direction_no_sync(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_no_sync(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                    else if ( use_native_opt0_raw_y_plan_array_executor_ )
                    {
                        SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_opaque_direction(
                            plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                        ) );
                    }
                    else if ( use_native_opt0_tight_y_plan_sequence_ )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_tight_direction(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_tight(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                    else
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_direction(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                }
                else
                {
                    for ( std::size_t i = 0; i < plan_names.size(); ++i )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_raw_direction(
                                plan_names[i], exec_direction, static_cast<void *>( in.raw_ptr() + offsets[i] ),
                                static_cast<void *>( out.raw_ptr() + offsets[i] )
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_raw(
                                plan_names[i], static_cast<void *>( in.raw_ptr() + offsets[i] ),
                                static_cast<void *>( out.raw_ptr() + offsets[i] )
                            ) );
                        }
                    }
                }
                synchronize_native_opt0_y_plan_streams_( plan_names.size() );
            }
        } );
        if ( dump_y_signature )
        {
            runtime_api_t::device_synchronize();
            append_native_opt0_y_buffer_signature_(
                profile_name, plan_label, "post", "input", plan_names, offsets, in.raw_ptr()
            );
            append_native_opt0_y_buffer_signature_(
                profile_name, plan_label, "post", "output", plan_names, offsets, out.raw_ptr()
            );
        }
        if ( !native_opt0_y_group_device_sync_enabled_() )
        {
            synchronize_after_local_fft_if_needed_();
        }
    }

    bool native_opt0_y_plan_array_bundle_active_() const
    {
        return ( use_native_opt0_reference_y_plan_bundle_ || use_native_opt0_raw_y_plan_bundle_ ||
                 use_native_opt0_reference_local_plan_context_ ) &&
               native_opt0_y_plan_array_bundle_ != base_fft_t::invalid_c2c_plan_array_id();
    }

    template <class ArrayIn, class ArrayOut>
    void exec_local_fft_plan_array_bundle_(
        const char *profile_name, const char *plan_label, const ArrayIn &in, ArrayOut &out,
        direction exec_direction
    )
    {
        if ( !native_opt0_y_plan_array_bundle_active_() )
        {
            throw std::logic_error( "native opt0 Y plan-array bundle execution requested without an active bundle" );
        }

        auto phase = optional_profiler_.scoped_tic( profile_name );
        time_native_stage_( profile_name, [&]() {
            if ( local_fft_diagnostics_enabled_ )
            {
                using timer_event_t = typename backend_t::timer_event_type;
                timer_event_t begin;
                timer_event_t end;
                begin.record();
                if ( use_native_opt0_y_no_sync_exec_ )
                {
                    SCFD_SAFE_CALL( base_fft_.exec_c2c_plan_array_direction_no_sync(
                        native_opt0_y_plan_array_bundle_, exec_direction, in.raw_ptr(), out.raw_ptr()
                    ) );
                }
                else
                {
                    SCFD_SAFE_CALL( base_fft_.exec_c2c_plan_array_direction(
                        native_opt0_y_plan_array_bundle_, exec_direction, in.raw_ptr(), out.raw_ptr()
                    ) );
                }
                synchronize_native_opt0_y_plan_array_bundle_();
                end.record();
                const double elapsed_ms = end.elapsed_time( begin );
                append_local_fft_timing_row_( profile_name, plan_label, elapsed_ms );
            }
            else
            {
                if ( use_native_opt0_y_no_sync_exec_ )
                {
                    SCFD_SAFE_CALL( base_fft_.exec_c2c_plan_array_direction_no_sync(
                        native_opt0_y_plan_array_bundle_, exec_direction, in.raw_ptr(), out.raw_ptr()
                    ) );
                }
                else
                {
                    SCFD_SAFE_CALL( base_fft_.exec_c2c_plan_array_direction(
                        native_opt0_y_plan_array_bundle_, exec_direction, in.raw_ptr(), out.raw_ptr()
                    ) );
                }
                synchronize_native_opt0_y_plan_array_bundle_();
            }
        } );
        if ( !native_opt0_y_group_device_sync_enabled_() )
        {
            synchronize_after_local_fft_if_needed_();
        }
    }

    std::string local_fft_diagnostics_path_( const char *kind ) const
    {
        std::ostringstream path;
        path << local_fft_diagnostics_dir_;
        if ( !local_fft_diagnostics_dir_.empty() &&
             local_fft_diagnostics_dir_[local_fft_diagnostics_dir_.size() - 1] != '/' )
        {
            path << '/';
        }
        path << kind << "_r" << mpi_.myid << ".csv";
        return path.str();
    }

    static bool local_fft_file_exists_( const std::string &path )
    {
        std::ifstream in( path.c_str() );
        return static_cast<bool>( in );
    }

    void append_local_fft_diagnostics_row_(
        const char *kind, const std::string &header, const std::string &row
    ) const
    {
        const std::string path         = local_fft_diagnostics_path_( kind );
        const bool        write_header = !local_fft_file_exists_( path );
        std::ofstream     out( path.c_str(), std::ios::out | std::ios::app );
        if ( !out )
        {
            throw std::runtime_error( "failed to open local FFT diagnostics CSV '" + path + "'" );
        }
        if ( write_header )
        {
            out << header << '\n';
        }
        out << row << '\n';
    }

	    void append_local_fft_timing_row_( const char *profile_name, const char *plan_name, double elapsed_ms )
	    {
	        const std::string header = "source,run_label,rank,call_index,profile_name,plan_name,event_ms,directory";
	        std::ostringstream row;
	        row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << local_fft_call_index_++
	            << ',' << profile_name << ',' << plan_name << ',' << elapsed_ms << ',' << local_fft_diagnostics_dir_;
	        append_local_fft_diagnostics_row_( "local_fft_times", header, row.str() );
	    }

	    struct opt0_layout_coord_t
	    {
	        const char  *sample;
	        std::size_t c0;
	        std::size_t c1;
	        std::size_t c2;
	    };

	    static void add_opt0_layout_coord_(
	        std::vector<opt0_layout_coord_t> &coords, const char *sample, std::size_t c0, std::size_t c1,
	        std::size_t c2, std::size_t d0, std::size_t d1, std::size_t d2
	    )
	    {
	        if ( d0 == 0 || d1 == 0 || d2 == 0 || c0 >= d0 || c1 >= d1 || c2 >= d2 )
	            return;
	        for ( const auto &coord : coords )
	        {
	            if ( coord.c0 == c0 && coord.c1 == c1 && coord.c2 == c2 )
	                return;
	        }
	        coords.push_back( opt0_layout_coord_t{ sample, c0, c1, c2 } );
	    }

	    static std::vector<opt0_layout_coord_t> opt0_layout_coords_(
	        std::size_t d0, std::size_t d1, std::size_t d2
	    )
	    {
	        std::vector<opt0_layout_coord_t> coords;
	        add_opt0_layout_coord_( coords, "origin", 0, 0, 0, d0, d1, d2 );
	        add_opt0_layout_coord_( coords, "z1", 0, 0, std::min<std::size_t>( 1, d2 - 1 ), d0, d1, d2 );
	        add_opt0_layout_coord_( coords, "y1", 0, std::min<std::size_t>( 1, d1 - 1 ), 0, d0, d1, d2 );
	        add_opt0_layout_coord_( coords, "x1", std::min<std::size_t>( 1, d0 - 1 ), 0, 0, d0, d1, d2 );
	        add_opt0_layout_coord_( coords, "middle", d0 / 2, d1 / 2, d2 / 2, d0, d1, d2 );
	        add_opt0_layout_coord_( coords, "last", d0 - 1, d1 - 1, d2 - 1, d0, d1, d2 );
	        return coords;
	    }

	    void append_native_opt0_layout_offset_row_(
	        const char *direction, const char *role, const char *buffer_name, const char *layout_name,
	        const char *sample, std::size_t c0, std::size_t c1, std::size_t c2, std::size_t d0,
	        std::size_t d1, std::size_t d2, const void *base_ptr, std::size_t offset_elems,
	        const char *notes
	    ) const
	    {
	        const std::string header =
	            "source,run_label,rank,direction,role,buffer_name,layout_name,sample,coord0,coord1,coord2,"
	            "dim0,dim1,dim2,base_token,offset_elems,token,notes,directory";
	        const std::uintptr_t base_token = reinterpret_cast<std::uintptr_t>( base_ptr );
	        const std::uintptr_t offset_bytes =
	            checked_mul_bytes_( offset_elems, sizeof( value_type ), "native opt0 layout offset bytes" );
	        std::ostringstream row;
	        row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << direction << ','
	            << role << ',' << buffer_name << ',' << layout_name << ',' << sample << ',' << c0 << ',' << c1
	            << ',' << c2 << ',' << d0 << ',' << d1 << ',' << d2 << ','
	            << static_cast<unsigned long long>( base_token ) << ',' << offset_elems << ','
	            << static_cast<unsigned long long>( base_token + offset_bytes ) << ',' << notes << ','
	            << local_fft_diagnostics_dir_;
	        append_local_fft_diagnostics_row_( "opt0_layout_offsets", header, row.str() );
	    }

	    template <class Array>
	    void append_native_opt0_array_layout_offsets_(
	        const char *direction, const char *role, const char *buffer_name, const char *layout_name,
	        const Array &array, std::size_t d0, std::size_t d1, std::size_t d2, const char *notes
	    ) const
	    {
	        for ( const auto &coord : opt0_layout_coords_( d0, d1, d2 ) )
	        {
	            const std::size_t offset = static_cast<std::size_t>( array.calc_lin_index(
	                static_cast<int>( coord.c0 ), static_cast<int>( coord.c1 ), static_cast<int>( coord.c2 )
	            ) );
	            append_native_opt0_layout_offset_row_(
	                direction, role, buffer_name, layout_name, coord.sample, coord.c0, coord.c1, coord.c2, d0,
	                d1, d2, array.raw_ptr(), offset, notes
	            );
	        }
	    }

	    template <class Ptr>
	    void append_native_opt0_y_plan_layout_offsets_(
	        const char *direction, const char *role, const char *buffer_name, Ptr base_ptr,
	        const std::vector<std::size_t> &offsets
	    ) const
	    {
	        if ( offsets.empty() )
	            return;
	        const std::size_t x_size = transpose1_dim_.size_x[myid_i_];
	        const std::size_t y_size = ny_global_;
	        const std::size_t z_size = transpose1_dim_.size_z[myid_j_];
	        if ( x_size == 0 || y_size == 0 || z_size == 0 )
	            return;

	        const std::size_t plan_count     = offsets.size();
	        const std::size_t batches_offset = x_size <= z_size ? z_size * y_size : 1;
	        const std::size_t istride        = z_size;
	        const std::size_t idist          = x_size <= z_size ? 1 : z_size * y_size;
	        const std::size_t batch          = std::max( x_size, z_size );

	        for ( const auto &coord : opt0_layout_coords_( plan_count, batch, y_size ) )
	        {
	            const std::size_t plan_index = coord.c0;
	            const std::size_t batch_index = coord.c1;
	            const std::size_t k_index = coord.c2;
	            const std::size_t batch_offset = checked_mul_bytes_(
	                batch_index, idist, "native opt0 layout Y batch offset"
	            );
	            const std::size_t k_offset = checked_mul_bytes_(
	                k_index, istride, "native opt0 layout Y k offset"
	            );
	            const std::size_t relative_offset = checked_add_bytes_(
	                batch_offset, k_offset, "native opt0 layout Y relative offset"
	            );
	            const std::size_t offset = checked_add_bytes_(
	                offsets[plan_index], relative_offset, "native opt0 layout Y absolute offset"
	            );
	            append_native_opt0_layout_offset_row_(
	                direction, role, buffer_name, "y_plan_offsets", coord.sample, plan_index, batch_index,
	                k_index, plan_count, batch, y_size, base_ptr, offset,
	                "coord0=plan_index;coord1=batch_index;coord2=y_index"
	            );
	        }

	        append_native_opt0_layout_offset_row_(
	            direction, role, buffer_name, "y_plan_metadata", "batches_offset", 0, 0, 0, plan_count, batch,
	            y_size, base_ptr, batches_offset, "offset_elems=batches_offset;istride=z_size;idist=see-meta"
	        );
	        append_native_opt0_layout_offset_row_(
	            direction, role, buffer_name, "y_plan_metadata", "istride", 0, 0, 0, plan_count, batch,
	            y_size, base_ptr, istride, "offset_elems=istride"
	        );
	        append_native_opt0_layout_offset_row_(
	            direction, role, buffer_name, "y_plan_metadata", "idist", 0, 0, 0, plan_count, batch,
	            y_size, base_ptr, idist, "offset_elems=idist"
	        );
	    }

	    template <class Stage0ZFast3, class Stage1ZFast3, class Stage1YFast3, class XStage3, class ComplexArray3>
	    void append_native_opt0_forward_layout_offsets_(
	        const Stage0ZFast3 &stage0, const Stage1ZFast3 &stage1, const Stage1YFast3 &stage1_yfft,
	        const XStage3 &x_stage, const ComplexArray3 &out, const std::vector<std::size_t> &y_plan_offsets
	    )
	    {
	        if ( !local_fft_diagnostics_enabled_ || !native_opt0_reference_y_buffer_topology_enabled_() ||
	             native_opt0_forward_layout_offsets_dumped_ )
	            return;
	        native_opt0_forward_layout_offsets_dumped_ = true;

	        append_native_opt0_array_layout_offsets_(
	            "forward", "r2c_output_complex", "stage0_default_z", "zfast_210", stage0,
	            half_input_dim_.size_x[myid_i_], half_input_dim_.size_y[myid_j_], nz_global_, "native R2C scratch"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "forward", "first_transpose_output_temp", "stage1_zfast", "zfast_210", stage1,
	            half_input_dim_.size_x[myid_i_], ny_global_, transpose1_dim_.size_z[myid_j_], "reference mem_d[0] alias"
	        );
	        append_native_opt0_y_plan_layout_offsets_(
	            "forward", "y_input_temp", "stage1_zfast", stage1.raw_ptr(), y_plan_offsets
	        );
	        append_native_opt0_y_plan_layout_offsets_(
	            "forward", "y_output_complex", "stage1_yfft_zfast", stage1_yfft.raw_ptr(), y_plan_offsets
	        );
	        append_native_opt0_array_layout_offsets_(
	            "forward", "second_transpose_input_complex", "stage1_yfft_zfast", "zfast_210", stage1_yfft,
	            half_input_dim_.size_x[myid_i_], ny_global_, transpose1_dim_.size_z[myid_j_],
	            "public output storage interpreted as reference complex before second transpose"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "forward", "second_transpose_output_temp", "x_fft_zfast", "zfast_210", x_stage,
	            output_dim_.size_x[0], output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_],
	            "reference mem_d[0] alias before X FFT"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "forward", "final_x_output_complex", "public_complex_out", "zfast_210", out,
	            output_dim_.size_x[0], output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_],
	            "FFTM public spectral output layout"
	        );
	    }

	    template <class ComplexArray3, class XStage3, class Stage1YFast3, class Stage1ZFast3, class Stage0ZFast3>
	    void append_native_opt0_backward_layout_offsets_(
	        const ComplexArray3 &in, const XStage3 &x_stage, const Stage1YFast3 &stage1_yfft,
	        const Stage1ZFast3 &stage1, const Stage0ZFast3 &stage0,
	        const std::vector<std::size_t> &y_plan_offsets
	    )
	    {
	        if ( !local_fft_diagnostics_enabled_ || !native_opt0_reference_y_buffer_topology_enabled_() ||
	             native_opt0_backward_layout_offsets_dumped_ )
	            return;
	        native_opt0_backward_layout_offsets_dumped_ = true;

	        append_native_opt0_array_layout_offsets_(
	            "backward", "public_complex_input", "public_complex_in", "zfast_210", in,
	            output_dim_.size_x[0], output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_],
	            "FFTM public spectral input layout"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "backward", "inverse_x_output_temp", "x_fft_zfast", "zfast_210", x_stage,
	            output_dim_.size_x[0], output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_],
	            "reference mem_d[0] alias after inverse X"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "backward", "backward_second_output_y_input", "stage1_yfft_zfast", "zfast_210", stage1_yfft,
	            half_input_dim_.size_x[myid_i_], ny_global_, transpose1_dim_.size_z[myid_j_],
	            "public input storage interpreted as reference complex before inverse Y"
	        );
	        append_native_opt0_y_plan_layout_offsets_(
	            "backward", "y_input_complex", "stage1_yfft_zfast", stage1_yfft.raw_ptr(), y_plan_offsets
	        );
	        append_native_opt0_y_plan_layout_offsets_(
	            "backward", "y_output_temp", "stage1_zfast", stage1.raw_ptr(), y_plan_offsets
	        );
	        append_native_opt0_array_layout_offsets_(
	            "backward", "backward_y_output_first_transpose_input", "stage1_zfast", "zfast_210", stage1,
	            half_input_dim_.size_x[myid_i_], ny_global_, transpose1_dim_.size_z[myid_j_],
	            "reference send-safe temp slot before backward first transpose"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "backward", "final_z_input_complex", "stage0_default_z", "zfast_210", stage0,
	            half_input_dim_.size_x[myid_i_], half_input_dim_.size_y[myid_j_], nz_global_,
	            "native C2R input scratch"
	        );
	    }

	    bool should_dump_native_opt0_y_execution_diagnostics_( const char *profile_name, const char *plan_label )
	    {
        if ( !local_fft_diagnostics_enabled_ || !native_opt0_default_z_layout_ )
            return false;
        const std::string profile = profile_name == nullptr ? "" : profile_name;
        const std::string label   = plan_label == nullptr ? "" : plan_label;
        const bool        forward = profile.find( "forward_y" ) != std::string::npos ||
                             label.find( "forward_y" ) != std::string::npos;
        const bool backward = profile.find( "backward_y" ) != std::string::npos ||
                              profile.find( "inverse_y" ) != std::string::npos ||
                              label.find( "inverse_y" ) != std::string::npos;
        if ( forward )
        {
            if ( native_opt0_forward_y_execution_diagnostics_dumped_ )
                return false;
            native_opt0_forward_y_execution_diagnostics_dumped_ = true;
            return true;
        }
        if ( backward )
        {
            if ( native_opt0_backward_y_execution_diagnostics_dumped_ )
                return false;
            native_opt0_backward_y_execution_diagnostics_dumped_ = true;
            return true;
        }
        return false;
    }

    template <class InPtr, class OutPtr>
    void append_native_opt0_y_execution_diagnostics_(
        const char *profile_name, const char *plan_label, const std::vector<std::string> &plan_names,
        const std::vector<std::size_t> &offsets, InPtr in, OutPtr out, plan_sequence_id_t plan_sequence,
        bool use_cached_sequence, bool use_explicit_exec_direction, direction exec_direction
    )
    {
        if ( plan_names.empty() ||
             !should_dump_native_opt0_y_execution_diagnostics_( profile_name, plan_label ) )
        {
            return;
        }
        const std::string header =
            "source,run_label,rank,profile_name,plan_label,sequence_id,sequence_kind,executor,sync_mode,exec_check,"
            "plan_index,plan_name,offset_elems,input_base_token,output_base_token,input_token,output_token,"
            "work_area_token,work_area_offset_bytes,stream_token,handle_mode,plan_direction,exec_direction,"
            "cufft_type,rank_fft,n0,inembed0,"
            "istride,idist,onembed0,ostride,odist,batch,work_size_bytes,directory";

        const std::uintptr_t input_base  = reinterpret_cast<std::uintptr_t>( in );
        const std::uintptr_t output_base = reinterpret_cast<std::uintptr_t>( out );
        const std::uintptr_t work_base   = reinterpret_cast<std::uintptr_t>( external_work_area_ );
        const char          *executor    = use_cached_sequence
                                               ? ( use_native_opt0_raw_y_plan_array_executor_ ? "opaque-raw" :
                                                   use_native_opt0_tight_y_plan_sequence_ ? "tight" : "cached-virtual" )
                                               : "raw-loop";
        const char *sync_mode      = native_opt0_y_group_device_sync_enabled_() ? "device" : "per-stream";
        const char *exec_check     = use_native_opt0_y_no_sync_exec_ ? "no-sync" : "sync";
        const char *handle_mode    = use_native_opt0_shared_y_plan_handles_ ? "shared-bidir" : "separate";
        const char *exec_dir_label = use_explicit_exec_direction ? local_fft_direction_name_( exec_direction ) : "plan";

        for ( std::size_t i = 0; i < plan_names.size(); ++i )
        {
            const auto desc = base_fft_.plan_descriptor( plan_names[i] );
            const std::uintptr_t offset_bytes =
                checked_mul_bytes_( offsets[i], sizeof( value_type ), "native opt0 y diagnostic data offset" );
            long long work_area_offset = -1;
            if ( work_base != 0 && desc.work_area_token >= work_base )
            {
                const std::uintptr_t diff = desc.work_area_token - work_base;
                if ( diff <= static_cast<std::uintptr_t>( std::numeric_limits<long long>::max() ) )
                    work_area_offset = static_cast<long long>( diff );
            }

            std::ostringstream row;
            row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << profile_name << ','
                << plan_label << ',';
            if ( plan_sequence == base_fft_t::invalid_plan_sequence_id() )
                row << -1 << ",invalid,";
            else
                row << static_cast<unsigned long long>( plan_sequence ) << ','
                    << base_fft_.plan_sequence_kind_name( plan_sequence ) << ',';
            row << executor << ',' << sync_mode << ',' << exec_check << ',' << i << ',' << plan_names[i] << ',' << offsets[i] << ','
                << static_cast<unsigned long long>( input_base ) << ','
                << static_cast<unsigned long long>( output_base ) << ','
                << static_cast<unsigned long long>( input_base + offset_bytes ) << ','
                << static_cast<unsigned long long>( output_base + offset_bytes ) << ','
                << static_cast<unsigned long long>( desc.work_area_token ) << ',' << work_area_offset << ','
                << static_cast<unsigned long long>( desc.stream_token ) << ',' << handle_mode << ','
                << local_fft_direction_name_( desc.dir ) << ',' << exec_dir_label
                << ',' << desc.cufft_type << ',' << desc.rank << ',' << desc.n[0] << ',' << desc.inembed[0] << ','
                << desc.istride << ',' << desc.idist << ',' << desc.onembed[0] << ',' << desc.ostride << ','
                << desc.odist << ',' << desc.batch << ',' << desc.work_size << ',' << local_fft_diagnostics_dir_;
            append_local_fft_diagnostics_row_( "native_opt0_y_execution", header, row.str() );
        }
    }

    bool should_dump_native_opt0_y_buffer_signature_( const char *profile_name, const char *plan_label )
    {
        if ( !local_fft_diagnostics_enabled_ || !native_opt0_default_z_layout_ )
            return false;
        const std::string profile = profile_name == nullptr ? "" : profile_name;
        const std::string label   = plan_label == nullptr ? "" : plan_label;
        const bool        forward = profile.find( "forward_y" ) != std::string::npos ||
                             label.find( "forward_y" ) != std::string::npos;
        const bool backward = profile.find( "backward_y" ) != std::string::npos ||
                              profile.find( "inverse_y" ) != std::string::npos ||
                              label.find( "inverse_y" ) != std::string::npos;
        if ( forward )
        {
            if ( native_opt0_forward_y_buffer_signature_dumped_ )
                return false;
            native_opt0_forward_y_buffer_signature_dumped_ = true;
            return true;
        }
        if ( backward )
        {
            if ( native_opt0_backward_y_buffer_signature_dumped_ )
                return false;
            native_opt0_backward_y_buffer_signature_dumped_ = true;
            return true;
        }
        return false;
    }

    static void add_signature_index_( std::vector<std::size_t> &indices, std::size_t count, std::size_t index )
    {
        if ( count == 0 || index >= count )
            return;
        indices.push_back( index );
    }

    static std::vector<std::size_t> signature_indices_( std::size_t count )
    {
        std::vector<std::size_t> indices;
        if ( count == 0 )
            return indices;
        add_signature_index_( indices, count, 0 );
        add_signature_index_( indices, count, 1 );
        add_signature_index_( indices, count, 2 );
        add_signature_index_( indices, count, 3 );
        add_signature_index_( indices, count, count / 4 );
        add_signature_index_( indices, count, count / 2 );
        add_signature_index_( indices, count, ( 3 * count ) / 4 );
        if ( count >= 4 )
        {
            add_signature_index_( indices, count, count - 4 );
            add_signature_index_( indices, count, count - 3 );
            add_signature_index_( indices, count, count - 2 );
        }
        add_signature_index_( indices, count, count - 1 );
        std::sort( indices.begin(), indices.end() );
        indices.erase( std::unique( indices.begin(), indices.end() ), indices.end() );
        return indices;
    }

    static std::vector<std::size_t> signature_indices_from_signed_( long long int count )
    {
        if ( count <= 0 )
            return std::vector<std::size_t>();
        return signature_indices_( static_cast<std::size_t>( count ) );
    }

    static double complex_real_( const value_type &value )
    {
        return static_cast<double>( value.x );
    }

    static double complex_imag_( const value_type &value )
    {
        return static_cast<double>( value.y );
    }

    template <class Ptr>
    void append_native_opt0_y_buffer_signature_(
        const char *profile_name, const char *plan_label, const char *phase, const char *buffer_role,
        const std::vector<std::string> &plan_names, const std::vector<std::size_t> &offsets, Ptr buffer_base
    )
    {
        if ( !local_fft_diagnostics_enabled_ || plan_names.empty() || plan_names.size() != offsets.size() )
            return;

        const std::string header =
            "source,run_label,rank,profile_name,plan_label,phase,buffer_role,plan_index,plan_name,"
            "base_token,plan_token,offset_elems,n,istride,idist,batch,sample_count,sum_re,sum_im,"
            "sumsq_abs,l2_abs,min_abs,max_abs,first_re,first_im,mid_re,mid_im,last_re,last_im,directory";

        const std::vector<std::size_t> plan_indices = signature_indices_( plan_names.size() );
        const std::uintptr_t          base_token   = reinterpret_cast<std::uintptr_t>( buffer_base );

        for ( const std::size_t plan_index : plan_indices )
        {
            const auto desc = base_fft_.plan_descriptor( plan_names[plan_index] );
            const std::vector<std::size_t> batch_indices = signature_indices_from_signed_( desc.batch );
            const std::vector<std::size_t> k_indices     = signature_indices_from_signed_( desc.n[0] );
            if ( batch_indices.empty() || k_indices.empty() || desc.istride <= 0 || desc.idist <= 0 )
                continue;

            double      sum_re      = 0.0;
            double      sum_im      = 0.0;
            double      sumsq_abs   = 0.0;
            double      min_abs     = std::numeric_limits<double>::infinity();
            double      max_abs     = 0.0;
            double      first_re    = 0.0;
            double      first_im    = 0.0;
            double      mid_re      = 0.0;
            double      mid_im      = 0.0;
            double      last_re     = 0.0;
            double      last_im     = 0.0;
            std::size_t sample_count = 0;

            const std::size_t mid_batch = batch_indices[batch_indices.size() / 2];
            const std::size_t mid_k     = k_indices[k_indices.size() / 2];
            for ( const std::size_t batch_index : batch_indices )
            {
                for ( const std::size_t k_index : k_indices )
                {
                    const std::size_t batch_offset = checked_mul_bytes_(
                        batch_index, static_cast<std::size_t>( desc.idist ),
                        "native opt0 y signature batch offset"
                    );
                    const std::size_t k_offset = checked_mul_bytes_(
                        k_index, static_cast<std::size_t>( desc.istride ),
                        "native opt0 y signature element offset"
                    );
                    const std::size_t relative_offset = checked_add_bytes_(
                        batch_offset, k_offset, "native opt0 y signature relative offset"
                    );
                    const std::size_t absolute_offset = checked_add_bytes_(
                        offsets[plan_index], relative_offset, "native opt0 y signature absolute offset"
                    );

                    value_type sample;
                    runtime_api_t::memcpy(
                        &sample, buffer_base + absolute_offset, sizeof( value_type ),
                        runtime_api_t::device_to_host_kind()
                    );
                    const double re  = complex_real_( sample );
                    const double im  = complex_imag_( sample );
                    const double mag = std::sqrt( re * re + im * im );
                    sum_re += re;
                    sum_im += im;
                    sumsq_abs += re * re + im * im;
                    min_abs = std::min( min_abs, mag );
                    max_abs = std::max( max_abs, mag );
                    if ( sample_count == 0 )
                    {
                        first_re = re;
                        first_im = im;
                    }
                    if ( batch_index == mid_batch && k_index == mid_k )
                    {
                        mid_re = re;
                        mid_im = im;
                    }
                    last_re = re;
                    last_im = im;
                    ++sample_count;
                }
            }
            if ( sample_count == 0 )
            {
                min_abs = 0.0;
            }

            const std::uintptr_t offset_bytes = checked_mul_bytes_(
                offsets[plan_index], sizeof( value_type ), "native opt0 y signature data offset"
            );
            std::ostringstream row;
            row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << profile_name << ','
                << plan_label << ',' << phase << ',' << buffer_role << ',' << plan_index << ',' << plan_names[plan_index]
                << ',' << static_cast<unsigned long long>( base_token ) << ','
                << static_cast<unsigned long long>( base_token + offset_bytes ) << ',' << offsets[plan_index] << ','
                << desc.n[0] << ',' << desc.istride << ',' << desc.idist << ',' << desc.batch << ','
                << sample_count << ',' << sum_re << ',' << sum_im << ',' << sumsq_abs << ','
                << std::sqrt( sumsq_abs ) << ',' << min_abs << ',' << max_abs << ',' << first_re << ',' << first_im
                << ',' << mid_re << ',' << mid_im << ',' << last_re << ',' << last_im << ','
                << local_fft_diagnostics_dir_;
            append_local_fft_diagnostics_row_( "native_opt0_y_signature", header, row.str() );
        }
    }

    static const char *local_fft_direction_name_( direction dir )
    {
        switch ( dir )
        {
        case direction::R2C:
            return "R2C";
        case direction::C2R:
            return "C2R";
        case direction::C2CF:
            return "C2CF";
        case direction::C2CB:
            return "C2CB";
        }
        return "unknown";
    }

    std::size_t bytes_from_elems_( std::size_t elems ) const
    {
        return elems * sizeof( value_type );
    }

    static std::size_t checked_mul_bytes_( std::size_t lhs, std::size_t rhs, const char *what )
    {
        if ( rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs )
        {
            std::ostringstream ss;
            ss << what << " overflows size_t";
            throw std::overflow_error( ss.str() );
        }
        return lhs * rhs;
    }

    static std::size_t checked_add_bytes_( std::size_t lhs, std::size_t rhs, const char *what )
    {
        if ( lhs > std::numeric_limits<std::size_t>::max() - rhs )
        {
            std::ostringstream ss;
            ss << what << " overflows size_t";
            throw std::overflow_error( ss.str() );
        }
        return lhs + rhs;
    }

    std::size_t native_opt0_reference_workspace_slot_offset_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( native_opt0_reference_y_buffer_topology_enabled_() && use_native_opt0_compact_y_workarea_ )
            return 1;
        return 3;
#else
        return 1;
#endif
    }

    std::size_t native_opt0_reference_domain_size_bytes_() const
    {
        return bytes_from_elems_( max_redistribution_buffer_elems_() );
    }

    bool native_opt0_reference_y_buffer_topology_enabled_() const
    {
        return native_opt0_default_z_layout_ && use_native_opt0_reference_y_buffer_topology_;
    }

    bool native_opt0_y_group_device_sync_enabled_() const
    {
        return native_opt0_default_z_layout_ && use_native_opt0_y_group_device_sync_;
    }

    value_type *native_opt0_reference_slot_ptr_( std::size_t slot ) const
    {
        if ( external_work_area_ == nullptr )
            throw std::logic_error( "native opt0 reference Y-buffer topology requires an external work area" );
        char *raw = static_cast<char *>( external_work_area_ );
        return reinterpret_cast<value_type *>( raw + slot * native_opt0_reference_domain_size_bytes_() );
    }

    value_type *native_opt0_reference_backward_y_output_ptr_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( use_native_opt0_compact_y_workarea_ )
            return native_opt0_reference_slot_ptr_( 3 );
        return native_opt0_reference_slot_ptr_( 2 );
#else
        return native_opt0_reference_slot_ptr_( 0 );
#endif
    }

    bool native_forward_second_receive_lands_in_stage_() const
    {
        return forward_receive_lands_in_stage_();
    }

    void debug_native_opt0_topology_marker_( const char *stage )
    {
        if ( !print_schedule_ || !native_opt0_reference_y_buffer_topology_enabled_() ||
             !( local_fft_diagnostics_enabled_ || native_stage_timers_enabled_ ) )
            return;
        std::ostringstream ss;
        ss << "native_opt0_reference_y_buffer_topology_marker rank_i=" << myid_i_ << " rank_j=" << myid_j_
           << " stage=" << stage;
        log_.info( ss.str() );
    }

    static std::size_t align_up_bytes_( std::size_t value, std::size_t alignment )
    {
        return ( value + alignment - 1 ) / alignment * alignment;
    }

    void append_local_fft_meta_row_( const std::string &key, long long index, long long value )
    {
        const std::string header = "source,run_label,rank,key,index,value,directory";
        std::ostringstream row;
        row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << key << ',' << index
            << ',' << value << ',' << local_fft_diagnostics_dir_;
        append_local_fft_diagnostics_row_( "local_fft_meta", header, row.str() );
    }

    void append_native_opt0_y_workarea_meta_(
        std::size_t plan_count, std::size_t base_offset_bytes, std::size_t work_stride_bytes
    )
    {
        if ( !local_fft_diagnostics_enabled_ )
            return;
        append_local_fft_meta_row_(
            "native_opt0_workspace_slot_offset", -1,
            static_cast<long long>( native_opt0_reference_workspace_slot_offset_() )
        );
        append_local_fft_meta_row_(
            "native_opt0_domainsize_bytes", -1,
            static_cast<long long>( native_opt0_reference_domain_size_bytes_() )
        );
        append_local_fft_meta_row_(
            "native_opt0_y_workarea_stride_bytes", -1, static_cast<long long>( work_stride_bytes )
        );
        append_local_fft_meta_row_(
            "native_opt0_y_workarea_extent_bytes", -1,
            static_cast<long long>( native_opt0_y_reference_workspace_size_bytes( plan_count, work_stride_bytes ) )
        );
        for ( std::size_t i = 0; i < plan_count; ++i )
        {
            const std::size_t offset = checked_add_bytes_(
                base_offset_bytes, checked_mul_bytes_( i, work_stride_bytes, "native opt0 y diagnostic offset" ),
                "native opt0 y diagnostic offset"
            );
            append_local_fft_meta_row_(
                "native_opt0_y_workarea_offset_bytes", static_cast<long long>( i ), static_cast<long long>( offset )
            );
        }
    }

    void synchronize_native_opt0_y_plan_streams_( std::size_t count )
    {
        if ( !native_opt0_default_z_layout_ )
            return;
        if ( count > streams_.size() )
            throw std::logic_error( "native opt0 y-plan stream synchronization count mismatch" );
        if ( native_opt0_y_group_device_sync_enabled_() )
        {
            auto scope = profile_scope_( "native_opt0_y_fft_device_sync" );
            runtime_api_t::device_synchronize();
            return;
        }
        auto scope = profile_scope_( "native_opt0_y_fft_stream_sync" );
        for ( std::size_t i = 0; i < count; ++i )
        {
            runtime_api_t::stream_synchronize( streams_[i].stream() );
        }
    }

    void synchronize_native_opt0_y_plan_array_bundle_()
    {
        if ( !native_opt0_default_z_layout_ )
            return;
        if ( native_opt0_y_group_device_sync_enabled_() )
        {
            auto scope = profile_scope_( "native_opt0_y_fft_device_sync" );
            runtime_api_t::device_synchronize();
            return;
        }
        auto scope = profile_scope_( "native_opt0_y_fft_stream_sync" );
        base_fft_.synchronize_c2c_plan_array_streams( native_opt0_y_plan_array_bundle_ );
    }

    static bool partitions_equal_( const partition_t &lhs, const partition_t &rhs )
    {
        return lhs.size_x == rhs.size_x && lhs.size_y == rhs.size_y && lhs.size_z == rhs.size_z &&
               lhs.size_w == rhs.size_w && lhs.start_x == rhs.start_x && lhs.start_y == rhs.start_y &&
               lhs.start_z == rhs.start_z && lhs.start_w == rhs.start_w;
    }

    static void require_partitions_equal_(
        const partition_t &lhs, const partition_t &rhs, const std::string &context
    )
    {
        if ( !partitions_equal_( lhs, rhs ) )
            throw std::logic_error( "owned reference native pencil plan-state partition mismatch: " + context );
    }

    static void validate_peer_order_( const std::vector<int> &order, int num_procs, const std::string &context )
    {
        if ( static_cast<int>( order.size() ) != std::max( 0, num_procs - 1 ) )
        {
            throw std::logic_error(
                "owned reference native pencil plan-state peer count mismatch for " + context
            );
        }
        std::vector<int> seen( num_procs, 0 );
        for ( const int peer : order )
        {
            if ( peer < 0 || peer >= num_procs )
            {
                throw std::logic_error(
                    "owned reference native pencil plan-state peer out of range for " + context
                );
            }
            if ( seen[peer] != 0 )
            {
                throw std::logic_error(
                    "owned reference native pencil plan-state duplicate peer for " + context
                );
            }
            seen[peer] = 1;
        }
    }

    void validate_native_plan_state_inputs_(
        const plan_state_t &plan_state, const partition_t &expected_half_input_dim,
        const partition_t &expected_transpose1_dim, const partition_t &expected_output_dim, int expected_myid_i,
        int expected_myid_j
    ) const
    {
        if ( !plan_state.is_inited() )
            throw std::logic_error( "owned reference native pencil plan-state is not initialized" );
        if ( plan_state.rank_i() != expected_myid_i || plan_state.rank_j() != expected_myid_j )
            throw std::logic_error( "owned reference native pencil plan-state rank mismatch" );
        require_partitions_equal_(
            plan_state.half_input_partition(), expected_half_input_dim, "half-input partition"
        );
        require_partitions_equal_( plan_state.stage1_partition(), expected_transpose1_dim, "stage1 partition" );
        require_partitions_equal_( plan_state.output_partition(), expected_output_dim, "output partition" );
    }

    void validate_native_plan_state_after_init_() const
    {
        if ( !native_plan_state_bound_ )
            return;
        if ( native_plan_state_.rank_i() != myid_i_ || native_plan_state_.rank_j() != myid_j_ )
            throw std::logic_error( "owned reference native pencil plan-state active rank mismatch" );
        require_partitions_equal_( native_plan_state_.half_input_partition(), half_input_dim_, "active half-input" );
        require_partitions_equal_( native_plan_state_.stage1_partition(), transpose1_dim_, "active stage1" );
        require_partitions_equal_( native_plan_state_.output_partition(), output_dim_, "active output" );
        validate_peer_order_( row_comm_order_, row_comm_info_.num_procs, "row communicator" );
        validate_peer_order_( line_comm_order_, line_comm_info_.num_procs, "line communicator" );
    }

    void clear_native_plan_schedule_slots_()
    {
        first_forward_send_schedule_.clear();
        first_forward_recv_schedule_.clear();
        first_backward_send_schedule_.clear();
        first_backward_recv_schedule_.clear();
        second_forward_send_schedule_.clear();
        second_forward_recv_schedule_.clear();
        second_backward_send_schedule_.clear();
        second_backward_recv_schedule_.clear();
    }

    static std::string native_schedule_context_(
        const char *stage, int transpose, int direction, int operation, int peer
    )
    {
        std::ostringstream ss;
        ss << stage << " transpose=" << transpose << " direction=" << direction
           << " operation=" << operation << " peer=" << peer;
        return ss.str();
    }

    static std::size_t checked_nonnegative_size_( long long value, const std::string &context )
    {
        if ( value < 0 )
            throw std::logic_error( "owned reference native pencil schedule has negative value for " + context );
        return static_cast<std::size_t>( value );
    }

    std::size_t schedule_bytes_to_elems_( long long bytes, const std::string &context ) const
    {
        const std::size_t value_bytes = sizeof( value_type );
        const std::size_t byte_count  = checked_nonnegative_size_( bytes, context + " bytes" );
        if ( byte_count % value_bytes != 0 )
            throw std::logic_error( "owned reference native pencil schedule byte count is not value-aligned for " + context );
        return byte_count / value_bytes;
    }

    void set_native_schedule_slot_(
        std::vector<native_schedule_slot> &slots, const ::fftm::detail::pencil_pencil_schedule_record &record,
        int expected_transpose, int expected_direction, int expected_operation, const char *stage
    ) const
    {
        const int peer = static_cast<int>( checked_nonnegative_size_( record.peer, std::string( stage ) + " peer" ) );
        const std::string context =
            native_schedule_context_( stage, expected_transpose, expected_direction, expected_operation, peer );

        if ( record.transpose != expected_transpose || record.direction != expected_direction ||
             record.operation != expected_operation )
        {
            throw std::logic_error( "owned reference native pencil schedule routed an unexpected record for " + context );
        }
        if ( peer < 0 || peer >= static_cast<int>( slots.size() ) )
            throw std::logic_error( "owned reference native pencil schedule peer is out of range for " + context );
        if ( slots[peer].valid )
            throw std::logic_error( "owned reference native pencil schedule duplicate peer record for " + context );

        native_schedule_slot slot;
        slot.offset_elems =
            checked_nonnegative_size_( record.offset_elems, context + " offset_elems" );
        slot.elems    = schedule_bytes_to_elems_( record.bytes, context );
        slot.mpi_peer = static_cast<int>( checked_nonnegative_size_( record.mpi_peer, context + " mpi_peer" ) );
        slot.mpi_tag  = static_cast<int>( checked_nonnegative_size_( record.mpi_tag, context + " mpi_tag" ) );
        slot.valid    = true;
        if ( slot.mpi_peer < 0 || slot.mpi_peer >= static_cast<int>( slots.size() ) )
            throw std::logic_error( "owned reference native pencil schedule MPI peer is out of range for " + context );
        slots[peer] = slot;
    }

    void validate_native_schedule_slot_(
        const std::vector<native_schedule_slot> &slots, int peer, std::size_t expected_offset,
        std::size_t expected_elems, const char *stage
    ) const
    {
        if ( peer < 0 || peer >= static_cast<int>( slots.size() ) || !slots[peer].valid )
        {
            throw std::logic_error(
                "owned reference native pencil schedule missing peer metadata for " + std::string( stage )
            );
        }
        const native_schedule_slot &slot = slots[peer];
        if ( slot.offset_elems != expected_offset || slot.elems != expected_elems )
        {
            std::ostringstream ss;
            ss << "owned reference native pencil schedule count/offset mismatch for " << stage << " peer=" << peer
               << " expected_offset=" << expected_offset << " schedule_offset=" << slot.offset_elems
               << " expected_elems=" << expected_elems << " schedule_elems=" << slot.elems;
            throw std::logic_error( ss.str() );
        }
        if ( slot.mpi_peer != peer )
        {
            throw std::logic_error(
                "owned reference native pencil schedule MPI peer mismatch for " + std::string( stage )
            );
        }
    }

    void validate_native_plan_first_schedule_slots_() const
    {
        for ( const int peer : row_comm_order_ )
        {
            validate_native_schedule_slot_(
                first_forward_recv_schedule_, peer, legacy_first_forward_recv_offset_( peer ),
                legacy_first_forward_recv_elems_( peer ), "first_forward_recv"
            );
            validate_native_schedule_slot_(
                first_forward_send_schedule_, peer, legacy_first_forward_send_offset_( peer ),
                legacy_first_forward_send_elems_( peer ), "first_forward_send"
            );
            validate_native_schedule_slot_(
                first_backward_recv_schedule_, peer, legacy_first_backward_recv_offset_( peer ),
                legacy_first_backward_recv_elems_( peer ), "first_backward_recv"
            );
            validate_native_schedule_slot_(
                first_backward_send_schedule_, peer, legacy_first_backward_send_offset_( peer ),
                legacy_first_backward_send_elems_( peer ), "first_backward_send"
            );
        }
    }

    void validate_native_plan_second_schedule_slots_() const
    {
        for ( const int peer : line_comm_order_ )
        {
            validate_native_schedule_slot_(
                second_forward_recv_schedule_, peer, legacy_second_forward_recv_offset_( peer ),
                legacy_second_forward_recv_elems_( peer ), "second_forward_recv"
            );
            validate_native_schedule_slot_(
                second_forward_send_schedule_, peer, legacy_second_forward_send_offset_( peer ),
                legacy_second_forward_send_elems_( peer ), "second_forward_send"
            );
            validate_native_schedule_slot_(
                second_backward_recv_schedule_, peer, legacy_second_backward_recv_offset_( peer ),
                legacy_second_backward_recv_elems_( peer ), "second_backward_recv"
            );
            validate_native_schedule_slot_(
                second_backward_send_schedule_, peer, legacy_second_backward_send_offset_( peer ),
                legacy_second_backward_send_elems_( peer ), "second_backward_send"
            );
        }
    }

    void build_native_plan_schedule_slots_()
    {
        clear_native_plan_schedule_slots_();
        if ( !native_plan_state_bound_ )
            return;

        first_forward_send_schedule_.assign( row_comm_info_.num_procs, native_schedule_slot() );
        first_forward_recv_schedule_.assign( row_comm_info_.num_procs, native_schedule_slot() );
        first_backward_send_schedule_.assign( row_comm_info_.num_procs, native_schedule_slot() );
        first_backward_recv_schedule_.assign( row_comm_info_.num_procs, native_schedule_slot() );
        second_forward_send_schedule_.assign( line_comm_info_.num_procs, native_schedule_slot() );
        second_forward_recv_schedule_.assign( line_comm_info_.num_procs, native_schedule_slot() );
        second_backward_send_schedule_.assign( line_comm_info_.num_procs, native_schedule_slot() );
        second_backward_recv_schedule_.assign( line_comm_info_.num_procs, native_schedule_slot() );

        const std::vector<::fftm::detail::pencil_pencil_schedule_record> records =
            native_plan_state_.local_schedule_records();
        for ( const auto &record : records )
        {
            if ( record.pidx_i != myid_i_ || record.pidx_j != myid_j_ )
                throw std::logic_error( "owned reference native pencil schedule rank-index mismatch" );

            if ( record.transpose == 1 && record.direction == 0 && record.operation == 0 )
            {
                set_native_schedule_slot_( first_forward_recv_schedule_, record, 1, 0, 0, "first_forward_recv" );
            }
            else if ( record.transpose == 1 && record.direction == 0 && record.operation == 1 )
            {
                set_native_schedule_slot_( first_forward_send_schedule_, record, 1, 0, 1, "first_forward_send" );
            }
            else if ( record.transpose == 1 && record.direction == 1 && record.operation == 0 )
            {
                set_native_schedule_slot_( first_backward_recv_schedule_, record, 1, 1, 0, "first_backward_recv" );
            }
            else if ( record.transpose == 1 && record.direction == 1 && record.operation == 1 )
            {
                set_native_schedule_slot_( first_backward_send_schedule_, record, 1, 1, 1, "first_backward_send" );
            }
            else if ( record.transpose == 2 && record.direction == 0 && record.operation == 0 )
            {
                set_native_schedule_slot_( second_forward_recv_schedule_, record, 2, 0, 0, "second_forward_recv" );
            }
            else if ( record.transpose == 2 && record.direction == 0 && record.operation == 1 )
            {
                set_native_schedule_slot_( second_forward_send_schedule_, record, 2, 0, 1, "second_forward_send" );
            }
            else if ( record.transpose == 2 && record.direction == 1 && record.operation == 0 )
            {
                set_native_schedule_slot_( second_backward_recv_schedule_, record, 2, 1, 0, "second_backward_recv" );
            }
            else if ( record.transpose == 2 && record.direction == 1 && record.operation == 1 )
            {
                set_native_schedule_slot_( second_backward_send_schedule_, record, 2, 1, 1, "second_backward_send" );
            }
            else
            {
                throw std::logic_error( "owned reference native pencil schedule unsupported transpose record" );
            }
        }
        validate_native_plan_first_schedule_slots_();
        validate_native_plan_second_schedule_slots_();
    }

    bool native_first_schedule_active_() const
    {
        return native_plan_state_bound_ && !first_forward_send_schedule_.empty();
    }

    bool native_second_schedule_active_() const
    {
        return native_plan_state_bound_ && !second_forward_send_schedule_.empty();
    }

    bool native_forward_first_execution_active_() const
    {
        return native_first_schedule_active_();
    }

    bool native_forward_second_execution_active_() const
    {
        return native_second_schedule_active_();
    }

    bool native_forward_second_facade_active_() const
    {
        return native_forward_second_execution_active_();
    }

    bool native_forward_second_peer_loop_active_() const
    {
        return native_forward_second_facade_active_() && peer_paired_large_count_byte_schedule_enabled_();
    }

    bool native_backward_second_execution_active_() const
    {
        return native_second_schedule_active_();
    }

    bool native_backward_second_facade_active_() const
    {
        return native_backward_second_execution_active_();
    }

    bool native_backward_second_peer_loop_active_() const
    {
        return use_native_backward_second_peer_loop_ && native_backward_second_facade_active_() &&
               peer_paired_large_count_byte_schedule_enabled_();
    }

    bool native_backward_first_execution_active_() const
    {
        return native_first_schedule_active_();
    }

    bool native_schedule_active_( const std::vector<native_schedule_slot> &slots ) const
    {
        return native_plan_state_bound_ && !slots.empty();
    }

    const native_schedule_slot *native_schedule_slot_for_execution_(
        const std::vector<native_schedule_slot> &slots, int peer, const char *stage
    ) const
    {
        if ( !native_schedule_active_( slots ) )
            return nullptr;
        if ( peer < 0 || peer >= static_cast<int>( slots.size() ) || !slots[peer].valid )
        {
            throw std::logic_error(
                "owned reference native pencil execution missing peer metadata for " + std::string( stage )
            );
        }
        return &slots[peer];
    }

    void validate_native_execution_slot_(
        const std::vector<native_schedule_slot> &slots, int peer, int expected_tag, const char *stage
    ) const
    {
        const native_schedule_slot *slot = native_schedule_slot_for_execution_( slots, peer, stage );
        if ( slot == nullptr )
            return;
        if ( slot->mpi_peer != peer || slot->mpi_tag != expected_tag )
        {
            std::ostringstream ss;
            ss << "owned reference native pencil execution peer/tag mismatch for " << stage << " peer=" << peer
               << " schedule_peer=" << slot->mpi_peer << " expected_tag=" << expected_tag
               << " schedule_tag=" << slot->mpi_tag;
            throw std::logic_error( ss.str() );
        }
    }

    std::size_t native_or_legacy_schedule_elems_(
        const std::vector<native_schedule_slot> &slots, int peer, std::size_t legacy_elems
    ) const
    {
        if ( !native_schedule_active_( slots ) || peer < 0 || peer >= static_cast<int>( slots.size() ) ||
             !slots[peer].valid )
        {
            return legacy_elems;
        }
        return slots[peer].elems;
    }

    std::size_t native_or_legacy_schedule_offset_(
        const std::vector<native_schedule_slot> &slots, int peer, std::size_t legacy_offset
    ) const
    {
        if ( !native_schedule_active_( slots ) || peer < 0 || peer >= static_cast<int>( slots.size() ) ||
             !slots[peer].valid )
        {
            return legacy_offset;
        }
        return slots[peer].offset_elems;
    }

    void validate_partitions_() const
    {
        if ( half_input_dim_.size_z.size() != 1 )
        {
            throw std::logic_error( "owned reference pencil first input must keep Z undistributed" );
        }
        if ( transpose1_dim_.size_y.size() != 1 )
        {
            throw std::logic_error( "owned reference pencil middle layout must keep Y undistributed" );
        }
        if ( output_dim_.size_x.size() != 1 )
        {
            throw std::logic_error( "owned reference pencil output layout must keep X undistributed" );
        }
        if ( half_input_dim_.size_y.size() != transpose1_dim_.size_z.size() )
        {
            throw std::logic_error( "owned reference pencil first redistribution partition mismatch" );
        }
        if ( transpose1_dim_.size_x.size() != output_dim_.size_y.size() )
        {
            throw std::logic_error( "owned reference pencil second redistribution partition mismatch" );
        }
        if ( half_input_dim_.size_x != transpose1_dim_.size_x )
        {
            throw std::logic_error( "owned reference pencil first redistribution must preserve X ownership" );
        }
        if ( transpose1_dim_.size_z != output_dim_.size_z )
        {
            throw std::logic_error( "owned reference pencil second redistribution must preserve Z ownership" );
        }
    }

    void ensure_inited_( mpi_transpose_3d_mode mode ) const
    {
        if ( !is_inited_ )
        {
            throw std::logic_error( "owned reference pencil pipeline is not initialized" );
        }
        if ( mode != mode_ )
        {
            throw std::logic_error( "owned reference pencil pipeline mode mismatch" );
        }
        if ( use_external_work_area_ && external_work_area_ == nullptr )
        {
            throw std::logic_error( "owned reference pencil pipeline external work area is not bound" );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( use_external_host_work_area_ && get_host_work_size_bytes() != 0 && external_host_work_area_ == nullptr )
        {
            throw std::logic_error( "owned reference pencil pipeline external host work area is not bound" );
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
        if ( native_opt0_reference_y_buffer_topology_enabled_() )
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            const std::size_t domain_bytes = native_opt0_reference_domain_size_bytes_();
            send_buffer_.init_by_raw_data(
                reinterpret_cast<value_type *>( raw + 2 * domain_bytes ), send_buffer_elems_
            );
            recv_buffer_.init_by_raw_data(
                reinterpret_cast<value_type *>( raw + domain_bytes ), recv_buffer_elems_
            );
#else
            send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), send_buffer_elems_ );
            recv_buffer_.init_by_raw_data(
                reinterpret_cast<value_type *>( raw + bytes_from_elems_( send_buffer_elems_ ) ), recv_buffer_elems_
            );
#endif
            return;
        }
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
            detail::mpi_int_cast( sizeof( value_type ), "owned reference pencil value type extent" ),
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

    bool direct_reference_opt1_forward_byte_receive_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return !native_opt0_default_z_layout_ && use_direct_forward_byte_receive_ && direct_p2p_cuda_aware_ && reference_parity_byte_path_enabled_() &&
               pencil_layout_selector_ == 2;
#else
        return false;
#endif
    }

    bool forward_receive_lands_in_stage_() const
    {
        return direct_forward_value_receive_enabled_() || direct_reference_opt1_forward_byte_receive_enabled_();
    }

    bool reference_byte_sync_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        /*
         * Baseline path intended to match reference's stable Peer2Peer_Sync
         * communication pattern: contiguous MPI_BYTE transfers, one logical
         * request per peer, no persistent requests, and no ready-send polling.
         */
        return cuda_aware_byte_p2p_enabled_() && !use_persistent_p2p_ && !use_ready_p2p_send_ &&
               !use_direct_backward_receive_;
#else
        return false;
#endif
    }

	    bool reference_parity_forward_waitany_enabled_() const
	    {
	        return reference_parity_enabled_ && reference_byte_sync_enabled_();
	    }

	    bool reference_parity_byte_path_enabled_() const
	    {
	        return reference_parity_enabled_ && reference_byte_sync_enabled_();
	    }

	    bool reference_parity_backward_waitall_enabled_() const
	    {
	        return reference_parity_byte_path_enabled_();
	    }

	    bool reference_parity_backward_ready_post_send_enabled_() const
	    {
	        return reference_parity_byte_path_enabled_();
	    }

    bool direct_reference_byte_backward_receive_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        /*
         * The backward owned-reference byte path receives contiguous peer blocks
         * that are already laid out for the next local FFT stage.  Receiving
         * directly into that stage removes the extra recv_buffer_ -> output
         * copy without relying on MPI derived datatypes.
         */
        return !native_opt0_default_z_layout_ && reference_byte_sync_enabled_();
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
                "owned reference requested large-count MPI count transport, but this MPI header does not expose "
                "MPI_Isend_c/MPI_Irecv_c"
            );
        }
#endif
    }

    void validate_reference_parity_() const
    {
        if ( use_deferred_send_completion_ )
        {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            throw std::logic_error( "owned deferred send completion requires CUDA-aware MPI" );
#else
            if ( !reference_parity_enabled_ || mode_ != mpi_transpose_3d_mode::p2p_waitany ||
                 !use_p2p_byte_transfer_ || use_persistent_p2p_ )
            {
                throw std::logic_error(
                    "owned deferred send completion requires reference-parity, p2p-waitany, CUDA-aware byte "
                    "transfers, and nonpersistent requests"
                );
            }
#endif
        }
        if ( !reference_parity_enabled_ )
            return;
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "owned reference parity requires device-aware MPI byte transfers" );
#else
        if ( !use_p2p_byte_transfer_ || use_persistent_p2p_ || use_ready_p2p_send_ ||
             use_direct_backward_receive_ || !direct_p2p_cuda_aware_ || use_p2p_send_thread_ )
        {
            throw std::logic_error( "owned reference parity options were not normalized before initialization" );
        }
#endif
    }

	    bool backward_receive_lands_in_output_() const
	    {
	        return direct_backward_receive_enabled_() || direct_reference_byte_backward_receive_enabled_();
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
        return large_count_byte_requests_enabled_() && reference_parity_byte_path_enabled_();
#else
        return false;
#endif
    }

    bool ready_stable_forward_byte_send_buffer_enabled_() const
    {
        return use_stable_forward_byte_send_buffer_ && use_ready_stable_forward_byte_send_buffer_;
    }

    bool contiguous_forward_byte_send_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return use_contiguous_forward_byte_send_ && use_stable_forward_byte_send_buffer_ &&
               cuda_aware_byte_p2p_enabled_();
#else
        return false;
#endif
    }

    bool physical_forward_peer_exchange_enabled_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        /*
         * Experimental reference-like forward path:
         * - send directly from peer-contiguous forward stage blocks;
         * - use value-count MPI requests from/to contiguous buffers;
         * - keep final stage-layout reshaping as explicit runtime_api_t copies.
         *
         * This deliberately avoids direct strided CUDA-aware receives, which
         * have stalled on the A100/HPCX stack.
         */
        return !native_opt0_default_z_layout_ && use_physical_forward_peer_exchange_ && reference_parity_byte_path_enabled_() &&
               cuda_aware_byte_p2p_enabled_() && !forward_receive_lands_in_stage_();
#else
        return false;
#endif
    }

    bool can_use_physical_forward_value_count_( std::size_t elems ) const
    {
        return physical_forward_peer_exchange_enabled_() &&
               elems <= static_cast<std::size_t>( std::numeric_limits<int>::max() );
    }

    bool can_use_physical_forward_value_recv_( std::size_t bytes ) const
    {
        if ( !physical_forward_peer_exchange_enabled_() || bytes == 0 || bytes % sizeof( value_type ) != 0 )
            return false;
        return can_use_physical_forward_value_count_( bytes / sizeof( value_type ) );
    }

    bool can_use_physical_forward_value_send_( std::size_t elems ) const
    {
        return can_use_physical_forward_value_count_( elems );
    }

    bool can_use_contiguous_forward_value_send_( std::size_t elems ) const
    {
        if ( can_use_physical_forward_value_send_( elems ) )
            return true;
        if ( !contiguous_forward_byte_send_enabled_() )
            return false;
        if ( contiguous_forward_send_mode_ != ::fftm::fftm_3d_contiguous_forward_send_mode::single )
            return false;
        return elems <= static_cast<std::size_t>( std::numeric_limits<int>::max() );
    }

    bool contiguous_forward_chunked_send_enabled_() const
    {
        return contiguous_forward_byte_send_enabled_() &&
               contiguous_forward_send_mode_ == ::fftm::fftm_3d_contiguous_forward_send_mode::chunked;
    }

    bool is_forward_send_stage_( const char *stage ) const
    {
        const std::string name( stage );
        return name == "first_forward_send" || name == "second_forward_send";
    }

    bool is_forward_recv_stage_( const char *stage ) const
    {
        const std::string name( stage );
        return name == "first_forward_recv" || name == "second_forward_recv";
    }

    std::size_t effective_contiguous_forward_send_chunk_bytes_() const
    {
        const std::size_t max_bytes = max_mpi_byte_chunk_bytes_();
        std::size_t       bytes     = contiguous_forward_send_chunk_bytes_;
        if ( bytes == 0 || bytes > max_bytes )
            bytes = max_bytes;
        bytes -= bytes % sizeof( value_type );
        if ( bytes == 0 )
            bytes = max_bytes;
        return bytes;
    }

    int mpi_byte_chunk_count_( std::size_t bytes ) const
    {
        const std::size_t max_bytes = max_mpi_byte_chunk_bytes_();
        if ( bytes == 0 )
            return 0;
        return detail::mpi_int_cast( ( bytes + max_bytes - 1 ) / max_bytes, "owned reference byte p2p chunk count" );
    }

    int mpi_byte_request_count_( std::size_t bytes ) const
    {
        if ( bytes == 0 )
            return 0;
        if ( large_count_byte_requests_enabled_() )
            return 1;
        return mpi_byte_chunk_count_( bytes );
    }

    int contiguous_forward_send_request_count_( std::size_t elems ) const
    {
        const std::size_t bytes = bytes_from_elems_( elems );
        if ( bytes == 0 )
            return 0;
        if ( can_use_contiguous_forward_value_send_( elems ) )
            return 1;
        if ( contiguous_forward_chunked_send_enabled_() )
        {
            const std::size_t chunk_bytes = effective_contiguous_forward_send_chunk_bytes_();
            return detail::mpi_int_cast(
                ( bytes + chunk_bytes - 1 ) / chunk_bytes,
                "owned reference contiguous forward send chunk count"
            );
        }
        return mpi_byte_request_count_( bytes );
    }

    int contiguous_forward_byte_request_count_( std::size_t bytes ) const
    {
        if ( bytes == 0 )
            return 0;
        if ( can_use_physical_forward_value_recv_( bytes ) )
            return 1;
        if ( contiguous_forward_chunked_send_enabled_() )
        {
            const std::size_t chunk_bytes = effective_contiguous_forward_send_chunk_bytes_();
            return detail::mpi_int_cast(
                ( bytes + chunk_bytes - 1 ) / chunk_bytes,
                "owned reference contiguous forward byte chunk count"
            );
        }
        return mpi_byte_request_count_( bytes );
    }

    int large_byte_datatype_block_count_( std::size_t bytes ) const
    {
        const std::size_t int_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
        if ( !large_count_byte_requests_enabled_() || bytes <= int_limit ||
             large_count_p2p_transport_ != ::fftm::fftm_3d_large_count_p2p_transport::hindexed )
            return 0;
        const std::size_t block_bytes = large_mpi_byte_datatype_block_bytes_();
        return detail::mpi_int_cast(
            ( bytes + block_bytes - 1 ) / block_bytes, "owned reference large byte datatype block count"
        );
    }

    bool hindexed_large_byte_datatype_needed_( std::size_t bytes ) const
    {
        const std::size_t int_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
        return large_count_byte_requests_enabled_() &&
               large_count_p2p_transport_ == ::fftm::fftm_3d_large_count_p2p_transport::hindexed &&
               bytes > int_limit;
    }

    bool hindexed_large_byte_datatype_cache_enabled_() const
    {
        return use_large_count_datatype_cache_ &&
               large_count_p2p_transport_ == ::fftm::fftm_3d_large_count_p2p_transport::hindexed &&
               large_count_byte_requests_enabled_();
    }

    void free_large_byte_datatype_cache_( large_byte_datatype_cache &cache )
    {
        for ( auto &datatype : cache.datatypes )
            scfd::communication::detail::type_free( datatype );
        cache = large_byte_datatype_cache();
    }

    void free_large_byte_datatype_caches_()
    {
        free_large_byte_datatype_cache_( first_forward_send_large_byte_types_ );
        free_large_byte_datatype_cache_( first_forward_recv_large_byte_types_ );
        free_large_byte_datatype_cache_( second_forward_send_large_byte_types_ );
        free_large_byte_datatype_cache_( second_forward_recv_large_byte_types_ );
        free_large_byte_datatype_cache_( second_backward_send_large_byte_types_ );
        free_large_byte_datatype_cache_( second_backward_recv_large_byte_types_ );
        free_large_byte_datatype_cache_( first_backward_send_large_byte_types_ );
        free_large_byte_datatype_cache_( first_backward_recv_large_byte_types_ );
    }

    template <class ByteFunc>
    void init_large_byte_datatype_cache_stage_(
        large_byte_datatype_cache &cache, int num_peers, const std::vector<int> &order, ByteFunc byte_func
    )
    {
        cache.datatypes.assign( num_peers, mpi_dtype_t() );
        cache.bytes.assign( num_peers, 0 );
        cache.blocks.assign( num_peers, 0 );
        for ( const int p : order )
        {
            const std::size_t bytes = byte_func( p );
            if ( !hindexed_large_byte_datatype_needed_( bytes ) )
                continue;
            cache.datatypes[p] = make_large_byte_datatype_( bytes, "large_byte/type_cache_create_type" );
            cache.bytes[p]     = bytes;
            cache.blocks[p]    = large_byte_datatype_block_count_( bytes );
            ++cache.active_count;
            cache.total_blocks += cache.blocks[p];
        }
    }

    void init_large_byte_datatype_caches_()
    {
        free_large_byte_datatype_caches_();
        if ( !hindexed_large_byte_datatype_cache_enabled_() )
            return;

        auto phase = profile_scope_( "large_byte/type_cache_init" );
        init_large_byte_datatype_cache_stage_(
            first_forward_send_large_byte_types_, row_comm_info_.num_procs, row_comm_order_,
            [this]( int p ) { return bytes_from_elems_( first_forward_send_elems_( p ) ); }
        );
        init_large_byte_datatype_cache_stage_(
            first_forward_recv_large_byte_types_, row_comm_info_.num_procs, row_comm_order_,
            [this]( int p ) { return bytes_from_elems_( first_forward_recv_elems_( p ) ); }
        );
        init_large_byte_datatype_cache_stage_(
            second_forward_send_large_byte_types_, line_comm_info_.num_procs, line_comm_order_,
            [this]( int p ) { return bytes_from_elems_( second_forward_send_elems_( p ) ); }
        );
        init_large_byte_datatype_cache_stage_(
            second_forward_recv_large_byte_types_, line_comm_info_.num_procs, line_comm_order_,
            [this]( int p ) { return bytes_from_elems_( second_forward_recv_elems_( p ) ); }
        );
        init_large_byte_datatype_cache_stage_(
            second_backward_send_large_byte_types_, line_comm_info_.num_procs, line_comm_order_,
            [this]( int p ) { return bytes_from_elems_( second_backward_send_elems_( p ) ); }
        );
        init_large_byte_datatype_cache_stage_(
            second_backward_recv_large_byte_types_, line_comm_info_.num_procs, line_comm_order_,
            [this]( int p ) { return bytes_from_elems_( second_backward_recv_elems_( p ) ); }
        );
        init_large_byte_datatype_cache_stage_(
            first_backward_send_large_byte_types_, row_comm_info_.num_procs, row_comm_order_,
            [this]( int p ) { return bytes_from_elems_( first_backward_send_elems_( p ) ); }
        );
        init_large_byte_datatype_cache_stage_(
            first_backward_recv_large_byte_types_, row_comm_info_.num_procs, row_comm_order_,
            [this]( int p ) { return bytes_from_elems_( first_backward_recv_elems_( p ) ); }
        );
    }

    const mpi_dtype_t *cached_large_byte_datatype_(
        const large_byte_datatype_cache &cache, int peer, std::size_t bytes
    ) const
    {
        if ( peer < 0 || peer >= static_cast<int>( cache.datatypes.size() ) )
            return nullptr;
        if ( cache.datatypes[peer].is_null() || cache.bytes[peer] != bytes )
            return nullptr;
        return &cache.datatypes[peer];
    }

    const large_byte_datatype_cache *large_byte_datatype_cache_for_stage_( const char *stage ) const
    {
        const std::string name( stage );
        if ( name == "first_forward_send" )
            return &first_forward_send_large_byte_types_;
        if ( name == "first_forward_recv" )
            return &first_forward_recv_large_byte_types_;
        if ( name == "second_forward_send" )
            return &second_forward_send_large_byte_types_;
        if ( name == "second_forward_recv" )
            return &second_forward_recv_large_byte_types_;
        if ( name == "second_backward_send" )
            return &second_backward_send_large_byte_types_;
        if ( name == "second_backward_recv" )
            return &second_backward_recv_large_byte_types_;
        if ( name == "first_backward_send" )
            return &first_backward_send_large_byte_types_;
        if ( name == "first_backward_recv" )
            return &first_backward_recv_large_byte_types_;
        return nullptr;
    }

    int cached_large_byte_datatype_blocks_( const large_byte_datatype_cache *cache, int peer ) const
    {
        if ( cache == nullptr || peer < 0 || peer >= static_cast<int>( cache->blocks.size() ) )
            return 0;
        return cache->blocks[peer];
    }

    int total_cached_large_byte_datatypes_() const
    {
        return first_forward_send_large_byte_types_.active_count + first_forward_recv_large_byte_types_.active_count +
               second_forward_send_large_byte_types_.active_count + second_forward_recv_large_byte_types_.active_count +
               second_backward_send_large_byte_types_.active_count +
               second_backward_recv_large_byte_types_.active_count + first_backward_send_large_byte_types_.active_count +
               first_backward_recv_large_byte_types_.active_count;
    }

    int total_cached_large_byte_datatype_blocks_() const
    {
        return first_forward_send_large_byte_types_.total_blocks + first_forward_recv_large_byte_types_.total_blocks +
               second_forward_send_large_byte_types_.total_blocks + second_forward_recv_large_byte_types_.total_blocks +
               second_backward_send_large_byte_types_.total_blocks +
               second_backward_recv_large_byte_types_.total_blocks + first_backward_send_large_byte_types_.total_blocks +
               first_backward_recv_large_byte_types_.total_blocks;
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
	        if ( cuda_aware_byte_p2p_enabled_() && !reference_parity_byte_path_enabled_() )
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
	        if ( reference_parity_byte_path_enabled_() )
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
        int         contiguous_forward_value_requests = 0;
        int         physical_forward_value_send_requests = 0;
        int         physical_forward_value_recv_requests = 0;
        int         physical_forward_value_fallbacks = 0;
        int         contiguous_forward_chunked_requests = 0;
        int         contiguous_forward_chunked_recv_requests = 0;
        int         element_count_requests  = 0;
        int         element_count_fallbacks = 0;
        const bool  forward_send_stage = is_forward_send_stage_( stage );
        const bool  forward_recv_stage = is_forward_recv_stage_( stage );
        const large_byte_datatype_cache *large_type_cache = large_byte_datatype_cache_for_stage_( stage );
        const int cached_large_types = large_type_cache == nullptr ? 0 : large_type_cache->active_count;
        const int cached_large_type_blocks = large_type_cache == nullptr ? 0 : large_type_cache->total_blocks;
        for ( const int p : order )
        {
            const std::size_t bytes = byte_func( p );
            const std::size_t elems = elem_func( p );
            const bool        contiguous_forward_value_send =
                forward_send_stage && can_use_contiguous_forward_value_send_( elems );
            const bool physical_forward_value_send =
                forward_send_stage && can_use_physical_forward_value_send_( elems );
            const bool physical_forward_value_recv =
                forward_recv_stage && can_use_physical_forward_value_recv_( bytes );
            const bool physical_forward_value_fallback =
                ( forward_send_stage || forward_recv_stage ) && physical_forward_peer_exchange_enabled_() &&
                !physical_forward_value_send && !physical_forward_value_recv;
            const bool contiguous_forward_chunked_send =
                forward_send_stage && contiguous_forward_chunked_send_enabled_();
            const bool contiguous_forward_chunked_recv =
                forward_recv_stage && contiguous_forward_chunked_send_enabled_();
            total_bytes += bytes;
            max_bytes = std::max( max_bytes, bytes );
            requests += forward_send_stage
                            ? contiguous_forward_send_request_count_( elems )
                            : ( contiguous_forward_chunked_recv ? contiguous_forward_byte_request_count_( bytes )
                                                                : mpi_byte_request_count_( bytes ) );
            legacy_chunks += mpi_byte_chunk_count_( bytes );
            large_blocks += ( contiguous_forward_value_send || contiguous_forward_chunked_send ||
                              contiguous_forward_chunked_recv || physical_forward_value_recv )
                                ? 0
                                : large_byte_datatype_block_count_( bytes );
            contiguous_forward_value_requests += contiguous_forward_value_send ? 1 : 0;
            physical_forward_value_send_requests += physical_forward_value_send ? 1 : 0;
            physical_forward_value_recv_requests += physical_forward_value_recv ? 1 : 0;
            physical_forward_value_fallbacks += physical_forward_value_fallback ? 1 : 0;
            contiguous_forward_chunked_requests +=
                contiguous_forward_chunked_send ? contiguous_forward_send_request_count_( elems ) : 0;
            contiguous_forward_chunked_recv_requests +=
                contiguous_forward_chunked_recv ? contiguous_forward_byte_request_count_( bytes ) : 0;
            element_count_requests += element_count_request_count_( bytes );
            element_count_fallbacks += element_count_fallback_count_( bytes );
        }

        std::ostringstream ss;
        ss << "owned_reference_schedule stage=" << stage << " comm=" << comm_name << " rank_i=" << myid_i_
           << " rank_j=" << myid_j_ << " peers=" << order.size() << " total_bytes=" << total_bytes
	           << " max_peer_bytes=" << max_bytes << " chunk_bytes=" << max_mpi_byte_chunk_bytes_()
	           << " requests=" << requests << " legacy_chunks=" << legacy_chunks
               << " large_type_blocks=" << large_blocks << " mode=" << mpi_transpose_3d_mode_name( mode_ )
               << " large_type_cache_requested=" << ( use_large_count_datatype_cache_ ? 1 : 0 )
               << " large_type_cache_active=" << ( hindexed_large_byte_datatype_cache_enabled_() ? 1 : 0 )
               << " large_type_cached_peers=" << cached_large_types
               << " large_type_cached_blocks=" << cached_large_type_blocks
	               << " byte=" << ( cuda_aware_byte_p2p_enabled_() ? 1 : 0 )
		           << " persistent=" << ( persistent_byte_p2p_enabled_() ? 1 : 0 )
		           << " reference_parity=" << ( reference_parity_enabled_ ? 1 : 0 )
	               << " reference_compatible_parity=" << ( reference_parity_enabled_ ? 1 : 0 )
			           << " large_count_transport=" << ::fftm::fftm_3d_large_count_p2p_transport_name( large_count_p2p_transport_ )
               << " contiguous_forward_send_mode="
               << ::fftm::fftm_3d_contiguous_forward_send_mode_name( contiguous_forward_send_mode_ )
               << " contiguous_forward_chunk_bytes=" << effective_contiguous_forward_send_chunk_bytes_()
               << " contiguous_forward_value_requests=" << contiguous_forward_value_requests
               << " physical_forward_peer_exchange_requested="
               << ( use_physical_forward_peer_exchange_ ? 1 : 0 )
               << " physical_forward_peer_exchange_active="
               << ( physical_forward_peer_exchange_enabled_() ? 1 : 0 )
               << " physical_forward_send_source="
               << ( physical_forward_peer_exchange_enabled_() ? "stage" : "buffer" )
               << " physical_forward_value_send_requests=" << physical_forward_value_send_requests
               << " physical_forward_value_recv_requests=" << physical_forward_value_recv_requests
               << " physical_forward_value_fallbacks=" << physical_forward_value_fallbacks
               << " contiguous_forward_chunked_requests=" << contiguous_forward_chunked_requests
               << " contiguous_forward_chunked_recv_requests=" << contiguous_forward_chunked_recv_requests
               << " element_count_requests=" << element_count_requests
               << " element_count_fallbacks=" << element_count_fallbacks
	               << " forward_direct_byte_receive=" << ( direct_reference_opt1_forward_byte_receive_enabled_() ? 1 : 0 )
               << " stable_forward_byte_send_buffer=" << ( use_stable_forward_byte_send_buffer_ ? 1 : 0 )
               << " ready_stable_forward_byte_send_buffer="
               << ( ready_stable_forward_byte_send_buffer_enabled_() ? 1 : 0 )
               << " contiguous_forward_byte_send=" << ( contiguous_forward_byte_send_enabled_() ? 1 : 0 )
		           << " mpi_count_available=" << ( mpi_count_large_count_available_() ? 1 : 0 );
        log_.info( ss.str() );

        for ( const int p : order )
        {
            const std::size_t elems = elem_func( p );
            const std::size_t bytes = byte_func( p );
            const bool        contiguous_forward_value_send =
                forward_send_stage && can_use_contiguous_forward_value_send_( elems );
            const bool physical_forward_value_send =
                forward_send_stage && can_use_physical_forward_value_send_( elems );
            const bool physical_forward_value_recv =
                forward_recv_stage && can_use_physical_forward_value_recv_( bytes );
            const bool physical_forward_value_fallback =
                ( forward_send_stage || forward_recv_stage ) && physical_forward_peer_exchange_enabled_() &&
                !physical_forward_value_send && !physical_forward_value_recv;
            const bool contiguous_forward_chunked_send =
                forward_send_stage && contiguous_forward_chunked_send_enabled_();
            const bool contiguous_forward_chunked_recv =
                forward_recv_stage && contiguous_forward_chunked_send_enabled_();
            const bool cached_large_type =
                large_type_cache != nullptr && cached_large_byte_datatype_( *large_type_cache, p, bytes ) != nullptr;
            std::ostringstream peer_ss;
	            peer_ss << "owned_reference_schedule_peer stage=" << stage << " peer=" << p << " elems=" << elems
		                    << " bytes=" << bytes
                        << " requests="
                        << ( forward_send_stage
                                 ? contiguous_forward_send_request_count_( elems )
                                 : ( contiguous_forward_chunked_recv ? contiguous_forward_byte_request_count_( bytes )
                                                                     : mpi_byte_request_count_( bytes ) ) )
		                    << " legacy_chunks=" << mpi_byte_chunk_count_( bytes )
                        << " large_type_blocks="
                        << ( ( contiguous_forward_value_send || contiguous_forward_chunked_send ||
                               contiguous_forward_chunked_recv || physical_forward_value_recv )
                                 ? 0
                                 : large_byte_datatype_block_count_( bytes ) )
                        << " contiguous_forward_value_send=" << ( contiguous_forward_value_send ? 1 : 0 )
                        << " physical_forward_value_send=" << ( physical_forward_value_send ? 1 : 0 )
                        << " physical_forward_value_recv=" << ( physical_forward_value_recv ? 1 : 0 )
                        << " physical_forward_value_fallback=" << ( physical_forward_value_fallback ? 1 : 0 )
                        << " contiguous_forward_chunked_send=" << ( contiguous_forward_chunked_send ? 1 : 0 )
                        << " contiguous_forward_chunked_recv=" << ( contiguous_forward_chunked_recv ? 1 : 0 )
                        << " element_count=" << element_count_request_count_( bytes )
                        << " element_count_fallback=" << element_count_fallback_count_( bytes )
                        << " large_type_cached=" << ( cached_large_type ? 1 : 0 )
                        << " large_type_cached_blocks=" << cached_large_byte_datatype_blocks_( large_type_cache, p );
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
        if ( !reference_parity_enabled_ && !print_schedule_ )
            return;

        std::ostringstream ss;
        ss << "owned_reference_transport rank_i=" << myid_i_ << " rank_j=" << myid_j_
           << " mode=" << mpi_transpose_3d_mode_name( mode_ )
           << " reference_parity=" << ( reference_parity_enabled_ ? 1 : 0 )
           << " reference_compatible_parity=" << ( reference_parity_enabled_ ? 1 : 0 )
           << " byte=" << ( cuda_aware_byte_p2p_enabled_() ? 1 : 0 )
           << " persistent=" << ( use_persistent_p2p_ ? 1 : 0 )
           << " ready_send=" << ( use_ready_p2p_send_ ? 1 : 0 )
           << " send_thread=" << ( use_p2p_send_thread_ ? 1 : 0 )
	           << " direct_backward_receive=" << ( use_direct_backward_receive_ ? 1 : 0 )
	           << " forward_wait=" << ( reference_parity_forward_waitany_enabled_() ? "waitany" :
	                                    mode_ == mpi_transpose_3d_mode::p2p_waitall ? "waitall" : "waitany" )
	           << " backward_wait=" << ( reference_parity_backward_waitall_enabled_() ||
	                                             mode_ == mpi_transpose_3d_mode::p2p_waitall
	                                         ? "waitall"
	                                         : "waitany" )
		           << " backward_send=" << ( reference_parity_backward_ready_post_send_enabled_() ? "pack_ready_post"
		                                    : reference_byte_sync_enabled_()                    ? "pack_sync"
		                                                                                   : "configured" )
                   << " deferred_send_completion=" << ( deferred_send_completion_active_() ? 1 : 0 )
		           << " row_peers=" << row_comm_order_.size() << " line_peers=" << line_comm_order_.size()
               << " native_plan_state=" << ( native_plan_state_bound_ ? 1 : 0 )
               << " native_first_schedule=" << ( native_first_schedule_active_() ? 1 : 0 )
               << " native_second_schedule=" << ( native_second_schedule_active_() ? 1 : 0 )
               << " native_forward_first_execution=" << ( native_forward_first_execution_active_() ? 1 : 0 )
               << " native_forward_second_execution=" << ( native_forward_second_execution_active_() ? 1 : 0 )
               << " native_forward_second_facade=" << ( native_forward_second_facade_active_() ? 1 : 0 )
               << " native_forward_second_peer_loop=" << ( native_forward_second_peer_loop_active_() ? 1 : 0 )
               << " native_backward_second_execution=" << ( native_backward_second_execution_active_() ? 1 : 0 )
               << " native_backward_second_facade=" << ( native_backward_second_facade_active_() ? 1 : 0 )
               << " native_backward_second_peer_loop_requested="
               << ( use_native_backward_second_peer_loop_ ? 1 : 0 )
	               << " native_backward_second_peer_loop=" << ( native_backward_second_peer_loop_active_() ? 1 : 0 )
	               << " native_backward_first_execution=" << ( native_backward_first_execution_active_() ? 1 : 0 )
	                   << " native_opt0_reference_y_buffer_topology="
	                   << ( native_opt0_reference_y_buffer_topology_enabled_() ? 1 : 0 )
                   << " native_opt0_compact_y_workarea="
                   << ( use_native_opt0_compact_y_workarea_ ? 1 : 0 )
	                   << " native_opt0_reference_device_slots="
	                   << ( native_opt0_reference_y_buffer_topology_enabled_()
                            ? ( use_native_opt0_compact_y_workarea_
                                    ? "temp=0,recv=1/ywork,send=2,bwdtemp=3"
                                    : "temp=0,recv=1,send=2,ywork=3,bwdtemp=after_ywork" )
	                            : "standard" )
                   << " native_opt0_reference_buffer_cycle="
                   << ( native_opt0_reference_y_buffer_topology_enabled_() ? "forward_scratch-temp-io-temp-io_backward_io-temp-io-temp-scratch"
                                                                       : "standard" )
                   << " native_opt0_y_plan_sequence_executor="
                   << ( use_native_opt0_raw_y_plan_array_executor_ ? "opaque-raw" :
                        use_native_opt0_tight_y_plan_sequence_ ? "tight" : "cached-virtual" )
                   << " native_opt0_y_plan_handle_mode="
                   << ( use_native_opt0_shared_y_plan_handles_ ? "shared-bidir" : "separate" )
                   << " native_opt0_y_plan_lifecycle="
                   << ( use_native_opt0_reference_y_plan_lifecycle_ ? "reference" : "standard" )
                   << " native_opt0_y_plan_bundle="
	                   << ( use_native_opt0_reference_local_plan_context_ ? "reference-local-context" :
	                        use_native_opt0_raw_y_plan_bundle_ ? "raw-reference" :
	                        use_native_opt0_reference_y_plan_bundle_ ? "reference" : "none" )
	                   << " native_opt0_y_plan_bundle_stream_bind="
	                   << ( use_native_opt0_y_plan_bundle_stream_first_ ? "stream-before-work" : "work-before-stream" )
	                   << " native_opt0_raw_y_plan_bundle_reference_streams="
	                   << ( use_native_opt0_raw_y_plan_bundle_reference_streams_ ? 1 : 0 )
	                   << " native_opt0_reference_local_plan_context="
	                   << ( use_native_opt0_reference_local_plan_context_ ? 1 : 0 )
	                   << " native_opt0_y_sync="
	                   << ( native_opt0_y_group_device_sync_enabled_() ? "device" : "per-stream" )
	                   << " native_opt0_y_exec_check="
	                   << ( use_native_opt0_y_no_sync_exec_ ? "no-sync" : "sync" )
			           << " layout_selector=" << pencil_layout_selector_
		               << " large_count_transport=" << ::fftm::fftm_3d_large_count_p2p_transport_name( large_count_p2p_transport_ )
                   << " element_count_value_bytes=" << sizeof( value_type )
	                   << " forward_direct_byte_receive=" << ( direct_reference_opt1_forward_byte_receive_enabled_() ? 1 : 0 )
                   << " stable_forward_byte_send_buffer=" << ( use_stable_forward_byte_send_buffer_ ? 1 : 0 )
                   << " ready_stable_forward_byte_send_buffer="
                   << ( ready_stable_forward_byte_send_buffer_enabled_() ? 1 : 0 )
                   << " contiguous_forward_byte_send=" << ( contiguous_forward_byte_send_enabled_() ? 1 : 0 )
                   << " physical_forward_peer_exchange_requested="
                   << ( use_physical_forward_peer_exchange_ ? 1 : 0 )
                   << " physical_forward_peer_exchange_active="
                   << ( physical_forward_peer_exchange_enabled_() ? 1 : 0 )
                   << " physical_forward_send_source="
                   << ( physical_forward_peer_exchange_enabled_() ? "stage" : "buffer" )
                   << " contiguous_forward_send_mode="
                   << ::fftm::fftm_3d_contiguous_forward_send_mode_name( contiguous_forward_send_mode_ )
                   << " contiguous_forward_chunk_bytes=" << effective_contiguous_forward_send_chunk_bytes_()
                   << " large_type_cache_requested=" << ( use_large_count_datatype_cache_ ? 1 : 0 )
                   << " large_type_cache_active=" << ( hindexed_large_byte_datatype_cache_enabled_() ? 1 : 0 )
                   << " large_type_cached_peers=" << total_cached_large_byte_datatypes_()
                   << " large_type_cached_blocks=" << total_cached_large_byte_datatype_blocks_()
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
            throw std::runtime_error( "owned reference large byte datatype block count exceeds MPI int range" );
        }

        std::vector<int>      block_lengths( blocks );
        std::vector<MPI_Aint> displacements( blocks );
        std::size_t           offset = 0;
        for ( std::size_t i = 0; i < blocks; ++i )
        {
            const std::size_t block = std::min( block_bytes, bytes - offset );
            block_lengths[i]       = detail::mpi_int_cast( block, "owned reference large byte datatype block length" );
            displacements[i]       = static_cast<MPI_Aint>( offset );
            offset += block;
        }

        mpi_dtype_t byte_type = scfd::communication::detail::mpi_data_type_trait<char>::get();
        mpi_dtype_t large_type;
        SCFD_MPI_SAFE_CALL( MPI_Type_create_hindexed(
            detail::mpi_int_cast( blocks, "owned reference large byte datatype block count" ), block_lengths.data(),
            displacements.data(), byte_type.native(), large_type.native_ptr()
        ) );
        scfd::communication::detail::type_commit( large_type );
        return large_type;
    }

    template <class CommInfo>
    void post_byte_irecv_(
        CommInfo &comm_info, void *ptr, std::size_t bytes, int source, int tag, std::vector<mpi_request_t> &requests,
        std::vector<int> *request_peers = nullptr, std::vector<int> *remaining_by_peer = nullptr,
        detail::persistent_recv_requests<mpi_request_t> *persistent_state = nullptr, int *persistent_index = nullptr,
        const mpi_dtype_t *cached_large_byte_type = nullptr
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
                    detail::mpi_int_cast( elems, "owned reference element-count p2p recv" ), mpi_value_type_, source, tag,
                    requests.back()
                );
            }
            else if ( bytes <= int_limit )
            {
                auto phase = profile_scope_( "large_byte/recv_post_int_count" );
                comm_info.irecv(
                    base, detail::mpi_int_cast( bytes, "owned reference byte p2p recv count" ), byte_type, source, tag,
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
                        base, mpi_count_cast_( bytes, "owned reference MPI count byte p2p recv" ), byte_type.native(),
                        source, tag, comm_info.comm, requests.back().native_ptr()
                    ) );
                }
                else
#endif
                {
                    if ( cached_large_byte_type != nullptr && !cached_large_byte_type->is_null() )
                    {
                        auto phase = profile_scope_( "large_byte/recv_post_cached_type" );
                        comm_info.irecv( base, 1, *cached_large_byte_type, source, tag, requests.back() );
                    }
                    else
                    {
                        mpi_dtype_t large_type =
                            make_large_byte_datatype_( bytes, "large_byte/recv_type_create_uncached" );
                        {
                            auto phase = profile_scope_( "large_byte/recv_post_large_type_uncached" );
                            comm_info.irecv( base, 1, large_type, source, tag, requests.back() );
                        }
                        scfd::communication::detail::type_free( large_type );
                    }
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
            const int this_count = detail::mpi_int_cast( this_bytes, "owned reference byte p2p recv count" );
            if ( persistent_state != nullptr )
            {
                if ( persistent_index == nullptr )
                    throw std::logic_error( "owned reference persistent byte p2p recv index is null" );
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
        const int count = detail::mpi_int_cast( elems, "owned reference value p2p recv" );
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
    void post_contiguous_forward_value_irecv_(
        CommInfo &comm_info, void *ptr, std::size_t elems, int source, int tag,
        std::vector<mpi_request_t> &requests, std::vector<int> *request_peers,
        std::vector<int> *remaining_by_peer, const char *profile_label
    )
    {
        if ( elems == 0 )
            return;
        requests.push_back( mpi_request_t() );
        {
            auto phase = profile_scope_( profile_label );
            comm_info.irecv(
                static_cast<value_type *>( ptr ),
                detail::mpi_int_cast( elems, "owned reference physical forward value p2p recv" ), mpi_value_type_,
                source, tag, requests.back()
            );
        }
        if ( request_peers != nullptr )
            request_peers->push_back( source );
        if ( remaining_by_peer != nullptr )
            ( *remaining_by_peer )[source] += 1;
    }

    template <class CommInfo>
    void post_contiguous_forward_chunked_byte_irecv_(
        CommInfo &comm_info, void *ptr, std::size_t bytes, int source, int tag,
        std::vector<mpi_request_t> &requests, std::vector<int> *request_peers,
        std::vector<int> *remaining_by_peer, detail::persistent_recv_requests<mpi_request_t> *persistent_state,
        int *persistent_index, const char *profile_label
    )
    {
        if ( bytes == 0 )
            return;
        char             *base        = static_cast<char *>( ptr );
        const mpi_dtype_t byte_type   = scfd::communication::detail::mpi_data_type_trait<char>::get();
        const std::size_t chunk_bytes = effective_contiguous_forward_send_chunk_bytes_();
        int               chunks      = 0;
        auto              phase       = profile_scope_( profile_label );
        for ( std::size_t offset = 0; offset < bytes; offset += chunk_bytes )
        {
            const std::size_t this_bytes = std::min( chunk_bytes, bytes - offset );
            const int this_count = detail::mpi_int_cast(
                this_bytes, "owned reference contiguous forward byte chunk recv"
            );
            if ( persistent_state != nullptr )
            {
                if ( persistent_index == nullptr )
                    throw std::logic_error( "owned reference persistent contiguous forward recv index is null" );
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
                request_peers->push_back( source );
            ++chunks;
        }
        if ( remaining_by_peer != nullptr )
            ( *remaining_by_peer )[source] += chunks;
    }

    template <class CommInfo>
    void post_forward_byte_irecv_(
        CommInfo &comm_info, void *ptr, std::size_t bytes, int source, int tag,
        std::vector<mpi_request_t> &requests, std::vector<int> *request_peers,
        std::vector<int> *remaining_by_peer, detail::persistent_recv_requests<mpi_request_t> *persistent_state,
        int *persistent_index, const char *chunked_profile_label, const char *value_count_profile_label,
        const mpi_dtype_t *cached_large_byte_type = nullptr
    )
    {
        if ( can_use_physical_forward_value_recv_( bytes ) )
        {
            post_contiguous_forward_value_irecv_(
                comm_info, ptr, bytes / sizeof( value_type ), source, tag, requests, request_peers,
                remaining_by_peer, value_count_profile_label
            );
            return;
        }
        if ( contiguous_forward_chunked_send_enabled_() )
        {
            post_contiguous_forward_chunked_byte_irecv_(
                comm_info, ptr, bytes, source, tag, requests, request_peers, remaining_by_peer, persistent_state,
                persistent_index, chunked_profile_label
            );
            return;
        }
        post_byte_irecv_(
            comm_info, ptr, bytes, source, tag, requests, request_peers, remaining_by_peer, persistent_state,
            persistent_index, cached_large_byte_type
        );
    }

    template <class CommInfo>
    void post_contiguous_forward_value_isend_(
        CommInfo &comm_info, const value_type *ptr, std::size_t elems, int dest, int tag,
        std::vector<mpi_request_t> &requests, const char *profile_label
    )
    {
        if ( elems == 0 )
            return;
        requests.push_back( mpi_request_t() );
        {
            auto phase = profile_scope_( profile_label );
            comm_info.isend(
                ptr, detail::mpi_int_cast( elems, "owned reference contiguous forward value p2p send" ),
                mpi_value_type_, dest, tag, requests.back()
            );
        }
    }

    template <class CommInfo>
    void post_contiguous_forward_chunked_byte_isend_(
        CommInfo &comm_info, const void *ptr, std::size_t bytes, int dest, int tag,
        std::vector<mpi_request_t> &requests, const char *profile_label
    )
    {
        if ( bytes == 0 )
            return;
        const char       *base        = static_cast<const char *>( ptr );
        const mpi_dtype_t byte_type   = scfd::communication::detail::mpi_data_type_trait<char>::get();
        const std::size_t chunk_bytes = effective_contiguous_forward_send_chunk_bytes_();
        auto              phase       = profile_scope_( profile_label );
        for ( std::size_t offset = 0; offset < bytes; offset += chunk_bytes )
        {
            const std::size_t this_bytes = std::min( chunk_bytes, bytes - offset );
            requests.push_back( mpi_request_t() );
            comm_info.isend(
                base + offset, detail::mpi_int_cast( this_bytes, "owned reference contiguous forward byte chunk send" ),
                byte_type, dest, tag, requests.back()
            );
        }
    }

	    template <class CommInfo>
	    void post_byte_isend_(
	        CommInfo &comm_info, const void *ptr, std::size_t bytes, int dest, int tag,
        std::vector<mpi_request_t> &requests,
        detail::persistent_send_requests<mpi_request_t> *persistent_state = nullptr, int *persistent_index = nullptr,
        const mpi_dtype_t *cached_large_byte_type = nullptr
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
                    detail::mpi_int_cast( elems, "owned reference element-count p2p send" ), mpi_value_type_, dest, tag,
                    requests.back()
                );
            }
            else if ( bytes <= int_limit )
            {
                auto phase = profile_scope_( "large_byte/send_post_int_count" );
                comm_info.isend(
                    base, detail::mpi_int_cast( bytes, "owned reference byte p2p send count" ), byte_type, dest, tag,
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
                        base, mpi_count_cast_( bytes, "owned reference MPI count byte p2p send" ), byte_type.native(), dest,
                        tag, comm_info.comm, requests.back().native_ptr()
                    ) );
                }
                else
#endif
                {
                    if ( cached_large_byte_type != nullptr && !cached_large_byte_type->is_null() )
                    {
                        auto phase = profile_scope_( "large_byte/send_post_cached_type" );
                        comm_info.isend( base, 1, *cached_large_byte_type, dest, tag, requests.back() );
                    }
                    else
                    {
                        mpi_dtype_t large_type =
                            make_large_byte_datatype_( bytes, "large_byte/send_type_create_uncached" );
                        {
                            auto phase = profile_scope_( "large_byte/send_post_large_type_uncached" );
                            comm_info.isend( base, 1, large_type, dest, tag, requests.back() );
                        }
                        scfd::communication::detail::type_free( large_type );
                    }
                }
            }
            return;
        }
        const std::size_t max_bytes = max_mpi_byte_chunk_bytes_();
        for ( std::size_t offset = 0; offset < bytes; offset += max_bytes )
        {
            const std::size_t this_bytes = std::min( max_bytes, bytes - offset );
            const int this_count = detail::mpi_int_cast( this_bytes, "owned reference byte p2p send count" );
            if ( persistent_state != nullptr )
            {
                if ( persistent_index == nullptr )
                    throw std::logic_error( "owned reference persistent byte p2p send index is null" );
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
        const int count = detail::mpi_int_cast( elems, "owned reference value p2p send" );
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

    static const char *deferred_send_stage_name_( deferred_send_stage stage )
    {
        switch ( stage )
        {
        case deferred_send_stage::forward_first:
            return "forward_first";
        case deferred_send_stage::forward_second:
            return "forward_second";
        case deferred_send_stage::backward_second:
            return "backward_second";
        case deferred_send_stage::backward_first:
            return "backward_first";
        case deferred_send_stage::none:
            return "none";
        }
        return "unknown";
    }

    bool deferred_send_completion_active_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return use_deferred_send_completion_ && reference_parity_enabled_ &&
               mode_ == mpi_transpose_3d_mode::p2p_waitany && cuda_aware_byte_p2p_enabled_() &&
               !use_persistent_p2p_;
#else
        return false;
#endif
    }

    scfd::communication::mpi_comm_info &deferred_send_comm_info_()
    {
        return deferred_send_state_.communicator == deferred_send_communicator::row ? row_comm_info_
                                                                                    : line_comm_info_;
    }

    std::string deferred_send_profile_name_( deferred_send_stage stage, const char *action ) const
    {
        return std::string( "deferred_send/" ) + deferred_send_stage_name_( stage ) + "/" + action;
    }

    template <class Fn>
    void time_deferred_send_action_( deferred_send_stage stage, const char *action, Fn fn )
    {
        const std::string label = deferred_send_profile_name_( stage, action );
        auto              scope = profile_scope_( label );
        if ( !native_stage_timing_active_ )
        {
            fn();
            return;
        }

        scfd::utils::system_timer_event begin;
        scfd::utils::system_timer_event end;
        begin.record();
        fn();
        end.record();
        record_native_stage_timing_( label.c_str(), end.elapsed_time( begin ) );
    }

    void ensure_no_deferred_send_( const char *where ) const
    {
        if ( !deferred_send_state_.active() )
            return;
        throw std::logic_error(
            std::string( "owned reference deferred send still active at " ) + where + ": " +
            deferred_send_stage_name_( deferred_send_state_.stage )
        );
    }

    void defer_byte_send_requests_(
        deferred_send_stage stage, deferred_send_communicator communicator,
        std::vector<mpi_request_t> &byte_send_requests
    )
    {
        ensure_no_deferred_send_( "new send deferral" );
        time_deferred_send_action_( stage, "send_deferred", [&]() {
            deferred_send_state_.stage        = stage;
            deferred_send_state_.communicator = communicator;
            deferred_send_state_.requests.swap( byte_send_requests );
            deferred_send_state_.remaining =
                detail::mpi_int_cast( deferred_send_state_.requests.size(), "owned deferred send request count" );
            if ( deferred_send_state_.remaining == 0 )
                deferred_send_state_.reset();
        } );
    }

    void probe_deferred_send_after_fft_( deferred_send_stage expected_stage )
    {
        if ( !deferred_send_state_.active() )
            return;
        if ( deferred_send_state_.stage != expected_stage )
        {
            throw std::logic_error(
                std::string( "owned reference deferred send stage mismatch after FFT: expected " ) +
                deferred_send_stage_name_( expected_stage ) + ", active " +
                deferred_send_stage_name_( deferred_send_state_.stage )
            );
        }

        time_deferred_send_action_( expected_stage, "send_ready_after_fft", [&]() {
            auto &comm_info = deferred_send_comm_info_();
            while ( deferred_send_state_.remaining > 0 )
            {
                int flag  = 0;
                const int index = comm_info.testany(
                    detail::mpi_int_cast(
                        deferred_send_state_.requests.size(), "owned deferred send readiness request count"
                    ),
                    deferred_send_state_.requests.data(), &flag
                );
                if ( !flag )
                    break;
                if ( index == MPI_UNDEFINED )
                {
                    deferred_send_state_.remaining = 0;
                    break;
                }
                --deferred_send_state_.remaining;
            }
        } );
    }

    void drain_deferred_send_( deferred_send_stage expected_stage )
    {
        if ( !deferred_send_state_.active() )
            return;
        if ( deferred_send_state_.stage != expected_stage )
        {
            throw std::logic_error(
                std::string( "owned reference deferred send drain stage mismatch: expected " ) +
                deferred_send_stage_name_( expected_stage ) + ", active " +
                deferred_send_stage_name_( deferred_send_state_.stage )
            );
        }

        time_deferred_send_action_( expected_stage, "send_drain", [&]() {
            if ( deferred_send_state_.remaining > 0 )
            {
                auto &comm_info = deferred_send_comm_info_();
                comm_info.waitall(
                    detail::mpi_int_cast(
                        deferred_send_state_.requests.size(), "owned deferred send drain request count"
                    ),
                    deferred_send_state_.requests.data()
                );
            }
        } );
        deferred_send_state_.reset();
    }

    void finish_deferred_send_after_fft_( deferred_send_stage expected_stage )
    {
        probe_deferred_send_after_fft_( expected_stage );
        drain_deferred_send_( expected_stage );
    }

    template <class Fn>
    void execute_with_deferred_send_overlap_( deferred_send_stage stage, Fn &&fn )
    {
        try
        {
            std::forward<Fn>( fn )();
        }
        catch ( ... )
        {
            drain_deferred_send_noexcept_();
            throw;
        }
        finish_deferred_send_after_fft_( stage );
    }

    void drain_deferred_send_noexcept_() noexcept
    {
        if ( !deferred_send_state_.active() )
            return;
        try
        {
            auto &comm_info = deferred_send_comm_info_();
            if ( !deferred_send_state_.requests.empty() )
            {
                comm_info.waitall(
                    detail::mpi_int_cast(
                        deferred_send_state_.requests.size(), "owned deferred send destructor request count"
                    ),
                    deferred_send_state_.requests.data()
                );
            }
        }
        catch ( ... )
        {
        }
        deferred_send_state_.reset();
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
                waitall_vector_( comm_info, byte_send_requests, byte_what );
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

    template <class CommInfo>
    void wait_or_defer_send_requests_(
        deferred_send_stage stage, deferred_send_communicator communicator, CommInfo &comm_info,
        std::vector<mpi_request_t> &byte_send_requests,
        detail::persistent_send_requests<mpi_request_t> &persistent_send_state,
        std::vector<mpi_request_t> &value_send_requests, int value_request_count, const char *byte_what,
        const char *persistent_what
    )
    {
        if ( deferred_send_completion_active_() )
        {
            defer_byte_send_requests_( stage, communicator, byte_send_requests );
            return;
        }
        wait_send_requests_(
            comm_info, byte_send_requests, persistent_send_state, value_send_requests, value_request_count, byte_what,
            persistent_what
        );
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
            total_chunks += contiguous_forward_byte_request_count_(
                bytes_from_elems_( first_forward_recv_elems_( p ) )
            );
        return total_chunks;
    }

    int second_forward_byte_recv_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : line_comm_order_ )
            total_chunks += contiguous_forward_byte_request_count_(
                bytes_from_elems_( second_forward_recv_elems_( p ) )
            );
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
        ensure_no_deferred_send_( "row request reset" );
        std::fill( row_send_requests_.begin(), row_send_requests_.end(), mpi_request_t() );
        std::fill( row_recv_requests_.begin(), row_recv_requests_.end(), mpi_request_t() );
    }

    void reset_line_requests_()
    {
        ensure_no_deferred_send_( "line request reset" );
        std::fill( line_send_requests_.begin(), line_send_requests_.end(), mpi_request_t() );
        std::fill( line_recv_requests_.begin(), line_recv_requests_.end(), mpi_request_t() );
    }

    void build_peer_orders_()
    {
        if ( native_plan_state_bound_ )
        {
            row_comm_order_  = native_plan_state_.first_transpose_order();
            line_comm_order_ = native_plan_state_.second_transpose_order();
            return;
        }

        row_comm_order_.clear();
        row_comm_order_.reserve( std::max( 0, row_comm_info_.num_procs - 1 ) );
        for ( int step = 1; step < row_comm_info_.num_procs; ++step )
        {
            row_comm_order_.push_back( ( myid_j_ + step ) % row_comm_info_.num_procs );
        }
        if ( pencil_layout_selector_ == 2 && !reference_parity_enabled_ )
            std::reverse( row_comm_order_.begin(), row_comm_order_.end() );

        line_comm_order_.clear();
        line_comm_order_.reserve( std::max( 0, line_comm_info_.num_procs - 1 ) );
        for ( int step = 1; step < line_comm_info_.num_procs; ++step )
        {
            line_comm_order_.push_back( ( myid_i_ + step ) % line_comm_info_.num_procs );
        }
        if ( pencil_layout_selector_ == 2 && !reference_parity_enabled_ )
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

    std::size_t legacy_first_forward_send_elems_( int target_j ) const
    {
        return nx_local_ * ny_input_local_ * transpose1_dim_.size_z[target_j];
    }

    std::size_t legacy_first_forward_send_offset_( int target_j ) const
    {
        return transpose1_dim_.start_z[target_j] * nx_local_ * ny_input_local_;
    }

    std::size_t legacy_first_forward_recv_elems_( int source_j ) const
    {
        return nx_local_ * half_input_dim_.size_y[source_j] * nz_local_;
    }

    std::size_t legacy_first_forward_recv_offset_( int source_j ) const
    {
        return nx_local_ * nz_local_ * half_input_dim_.start_y[source_j];
    }

    std::size_t legacy_first_backward_send_elems_( int target_j ) const
    {
        return nx_local_ * half_input_dim_.size_y[target_j] * nz_local_;
    }

    std::size_t legacy_first_backward_send_offset_( int target_j ) const
    {
        return legacy_first_forward_recv_offset_( target_j );
    }

    std::size_t legacy_first_backward_recv_elems_( int source_j ) const
    {
        return nx_local_ * ny_input_local_ * transpose1_dim_.size_z[source_j];
    }

    std::size_t legacy_first_backward_recv_offset_( int source_j ) const
    {
        return legacy_first_forward_send_offset_( source_j );
    }

    std::size_t first_forward_send_elems_( int target_j ) const
    {
        return native_or_legacy_schedule_elems_(
            first_forward_send_schedule_, target_j, legacy_first_forward_send_elems_( target_j )
        );
    }

    std::size_t first_forward_send_offset_( int target_j ) const
    {
        return native_or_legacy_schedule_offset_(
            first_forward_send_schedule_, target_j, legacy_first_forward_send_offset_( target_j )
        );
    }

    std::size_t first_forward_recv_elems_( int source_j ) const
    {
        return native_or_legacy_schedule_elems_(
            first_forward_recv_schedule_, source_j, legacy_first_forward_recv_elems_( source_j )
        );
    }

    std::size_t first_forward_recv_offset_( int source_j ) const
    {
        return native_or_legacy_schedule_offset_(
            first_forward_recv_schedule_, source_j, legacy_first_forward_recv_offset_( source_j )
        );
    }

    template <class Stage0Complex3>
    const value_type *native_forward_first_send_source_ptr_( int target_j, const Stage0Complex3 &in ) const
    {
        validate_native_execution_slot_(
            first_forward_send_schedule_, target_j, myid_j_, "first_forward_send_source"
        );
        return in.raw_ptr() + first_forward_send_offset_( target_j );
    }

    value_type *native_forward_first_device_send_buffer_ptr_( int target_j )
    {
        validate_native_execution_slot_(
            first_forward_send_schedule_, target_j, myid_j_, "first_forward_device_send_buffer"
        );
        return send_buffer_.raw_ptr() + first_forward_send_offset_( target_j );
    }

    value_type *native_forward_first_host_send_buffer_ptr_( int target_j )
    {
        validate_native_execution_slot_(
            first_forward_send_schedule_, target_j, myid_j_, "first_forward_host_send_buffer"
        );
        return host_send_buffer_.raw_ptr() + first_forward_send_offset_( target_j );
    }

    value_type *native_forward_first_device_recv_buffer_ptr_( int source_j )
    {
        validate_native_execution_slot_(
            first_forward_recv_schedule_, source_j, source_j, "first_forward_device_recv_buffer"
        );
        return recv_buffer_.raw_ptr() + first_forward_recv_offset_( source_j );
    }

    value_type *native_forward_first_host_recv_buffer_ptr_( int source_j )
    {
        validate_native_execution_slot_(
            first_forward_recv_schedule_, source_j, source_j, "first_forward_host_recv_buffer"
        );
        return host_recv_buffer_.raw_ptr() + first_forward_recv_offset_( source_j );
    }

    template <class Stage1Complex3>
    value_type *native_forward_first_recv_target_ptr_( int source_j, Stage1Complex3 &out )
    {
        validate_native_execution_slot_(
            first_forward_recv_schedule_, source_j, source_j, "first_forward_recv_target"
        );
        if ( forward_receive_lands_in_stage_() )
            return out.raw_ptr() + half_input_dim_.start_y[source_j];
        return native_forward_first_device_recv_buffer_ptr_( source_j );
    }

    std::size_t first_backward_send_elems_( int target_j ) const
    {
        return native_or_legacy_schedule_elems_(
            first_backward_send_schedule_, target_j, legacy_first_backward_send_elems_( target_j )
        );
    }

    std::size_t first_backward_send_offset_( int target_j ) const
    {
        return native_or_legacy_schedule_offset_(
            first_backward_send_schedule_, target_j, legacy_first_backward_send_offset_( target_j )
        );
    }

    std::size_t first_backward_recv_elems_( int source_j ) const
    {
        return native_or_legacy_schedule_elems_(
            first_backward_recv_schedule_, source_j, legacy_first_backward_recv_elems_( source_j )
        );
    }

    std::size_t first_backward_recv_offset_( int source_j ) const
    {
        return native_or_legacy_schedule_offset_(
            first_backward_recv_schedule_, source_j, legacy_first_backward_recv_offset_( source_j )
        );
    }

    std::size_t first_backward_packed_recv_offset_( int source_j ) const
    {
        return first_backward_recv_offset_( source_j );
    }

    int first_backward_send_mpi_tag_( int target_j ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( first_backward_send_schedule_, target_j, "first_backward_send_tag" );
        if ( slot != nullptr )
            return slot->mpi_tag;
        return target_j;
    }

    int first_backward_recv_mpi_tag_( int source_j ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( first_backward_recv_schedule_, source_j, "first_backward_recv_tag" );
        if ( slot != nullptr )
            return slot->mpi_tag;
        return myid_j_;
    }

    value_type *native_backward_first_device_send_buffer_ptr_( int target_j )
    {
        validate_native_execution_slot_(
            first_backward_send_schedule_, target_j, first_backward_send_mpi_tag_( target_j ),
            "first_backward_device_send_buffer"
        );
        return send_buffer_.raw_ptr() + first_backward_send_offset_( target_j );
    }

    value_type *native_backward_first_host_send_buffer_ptr_( int target_j )
    {
        validate_native_execution_slot_(
            first_backward_send_schedule_, target_j, first_backward_send_mpi_tag_( target_j ),
            "first_backward_host_send_buffer"
        );
        return host_send_buffer_.raw_ptr() + first_backward_send_offset_( target_j );
    }

    template <class Stage0Complex3>
    value_type *native_backward_first_recv_output_ptr_( int source_j, Stage0Complex3 &out )
    {
        validate_native_execution_slot_(
            first_backward_recv_schedule_, source_j, first_backward_recv_mpi_tag_( source_j ),
            "first_backward_recv_output"
        );
        return out.raw_ptr() + first_backward_recv_offset_( source_j );
    }

    value_type *native_backward_first_device_recv_buffer_ptr_( int source_j )
    {
        validate_native_execution_slot_(
            first_backward_recv_schedule_, source_j, first_backward_recv_mpi_tag_( source_j ),
            "first_backward_device_recv_buffer"
        );
        return recv_buffer_.raw_ptr() + first_backward_packed_recv_offset_( source_j );
    }

    value_type *native_backward_first_host_recv_buffer_ptr_( int source_j )
    {
        validate_native_execution_slot_(
            first_backward_recv_schedule_, source_j, first_backward_recv_mpi_tag_( source_j ),
            "first_backward_host_recv_buffer"
        );
        return host_recv_buffer_.raw_ptr() + first_backward_packed_recv_offset_( source_j );
    }

    template <class Stage0Complex3>
    value_type *native_backward_first_recv_target_ptr_( int source_j, Stage0Complex3 &out )
    {
        validate_native_execution_slot_(
            first_backward_recv_schedule_, source_j, first_backward_recv_mpi_tag_( source_j ),
            "first_backward_recv_target"
        );
        if ( backward_receive_lands_in_output_() )
            return native_backward_first_recv_output_ptr_( source_j, out );
        return native_backward_first_device_recv_buffer_ptr_( source_j );
    }

    std::size_t legacy_second_forward_send_elems_( int target_i ) const
    {
        return nx_local_ * output_dim_.size_y[target_i] * nz_local_;
    }

    std::size_t legacy_second_forward_send_offset_( int target_i ) const
    {
        return nx_local_ * nz_local_ * output_dim_.start_y[target_i];
    }

    std::size_t legacy_second_forward_recv_elems_( int source_i ) const
    {
        return transpose1_dim_.size_x[source_i] * ny_output_local_ * nz_local_;
    }

    std::size_t legacy_second_forward_recv_offset_( int source_i ) const
    {
        std::size_t offset = 0;
        for ( int p = 0; p < source_i; ++p )
        {
            offset += legacy_second_forward_recv_elems_( p );
        }
        return offset;
    }

    std::size_t legacy_second_backward_send_elems_( int target_i ) const
    {
        return transpose1_dim_.size_x[target_i] * ny_output_local_ * nz_local_;
    }

    std::size_t legacy_second_backward_send_offset_( int target_i ) const
    {
        return legacy_second_forward_recv_offset_( target_i );
    }

    std::size_t legacy_second_backward_recv_elems_( int source_i ) const
    {
        return nx_local_ * output_dim_.size_y[source_i] * nz_local_;
    }

    std::size_t legacy_second_backward_recv_offset_( int source_i ) const
    {
        return legacy_second_forward_send_offset_( source_i );
    }

    std::size_t second_forward_send_elems_( int target_i ) const
    {
        return native_or_legacy_schedule_elems_(
            second_forward_send_schedule_, target_i, legacy_second_forward_send_elems_( target_i )
        );
    }

    std::size_t second_forward_send_offset_( int target_i ) const
    {
        return native_or_legacy_schedule_offset_(
            second_forward_send_schedule_, target_i, legacy_second_forward_send_offset_( target_i )
        );
    }

    std::size_t second_forward_recv_elems_( int source_i ) const
    {
        return native_or_legacy_schedule_elems_(
            second_forward_recv_schedule_, source_i, legacy_second_forward_recv_elems_( source_i )
        );
    }

    std::size_t second_forward_recv_offset_( int source_i ) const
    {
        return native_or_legacy_schedule_offset_(
            second_forward_recv_schedule_, source_i, legacy_second_forward_recv_offset_( source_i )
        );
    }

    template <class XFastComplex3>
    const value_type *native_forward_second_send_source_ptr_( int target_i, const XFastComplex3 &in ) const
    {
        validate_native_execution_slot_(
            second_forward_send_schedule_, target_i, myid_i_, "second_forward_send_source"
        );
        return in.raw_ptr() + second_forward_send_offset_( target_i );
    }

    value_type *native_forward_second_device_send_buffer_ptr_( int target_i )
    {
        validate_native_execution_slot_(
            second_forward_send_schedule_, target_i, myid_i_, "second_forward_device_send_buffer"
        );
        return send_buffer_.raw_ptr() + second_forward_send_offset_( target_i );
    }

    value_type *native_forward_second_host_send_buffer_ptr_( int target_i )
    {
        validate_native_execution_slot_(
            second_forward_send_schedule_, target_i, myid_i_, "second_forward_host_send_buffer"
        );
        return host_send_buffer_.raw_ptr() + second_forward_send_offset_( target_i );
    }

    value_type *native_forward_second_device_recv_buffer_ptr_( int source_i )
    {
        validate_native_execution_slot_(
            second_forward_recv_schedule_, source_i, source_i, "second_forward_device_recv_buffer"
        );
        return recv_buffer_.raw_ptr() + second_forward_recv_offset_( source_i );
    }

    value_type *native_forward_second_host_recv_buffer_ptr_( int source_i )
    {
        validate_native_execution_slot_(
            second_forward_recv_schedule_, source_i, source_i, "second_forward_host_recv_buffer"
        );
        return host_recv_buffer_.raw_ptr() + second_forward_recv_offset_( source_i );
    }

    template <class XFFTComplex3>
    value_type *native_forward_second_recv_target_ptr_( int source_i, XFFTComplex3 &out )
    {
        validate_native_execution_slot_(
            second_forward_recv_schedule_, source_i, source_i, "second_forward_recv_target"
        );
        if ( native_forward_second_receive_lands_in_stage_() )
            return out.raw_ptr() + transpose1_dim_.start_x[source_i];
        return native_forward_second_device_recv_buffer_ptr_( source_i );
    }

    const native_schedule_slot &native_forward_second_send_slot_( int target_i, const char *stage ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( second_forward_send_schedule_, target_i, stage );
        if ( slot == nullptr )
            throw std::logic_error( "owned reference native forward-second send slot is not active" );
        return *slot;
    }

    const native_schedule_slot &native_forward_second_recv_slot_( int source_i, const char *stage ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( second_forward_recv_schedule_, source_i, stage );
        if ( slot == nullptr )
            throw std::logic_error( "owned reference native forward-second recv slot is not active" );
        return *slot;
    }

    template <class XFastComplex3>
    const value_type *native_forward_second_send_source_ptr_(
        const native_schedule_slot &slot, const XFastComplex3 &in
    ) const
    {
        return in.raw_ptr() + slot.offset_elems;
    }

    value_type *native_forward_second_device_send_buffer_ptr_( const native_schedule_slot &slot )
    {
        return send_buffer_.raw_ptr() + slot.offset_elems;
    }

    value_type *native_forward_second_host_send_buffer_ptr_( const native_schedule_slot &slot )
    {
        return host_send_buffer_.raw_ptr() + slot.offset_elems;
    }

    value_type *native_forward_second_device_recv_buffer_ptr_( const native_schedule_slot &slot )
    {
        return recv_buffer_.raw_ptr() + slot.offset_elems;
    }

    value_type *native_forward_second_host_recv_buffer_ptr_( const native_schedule_slot &slot )
    {
        return host_recv_buffer_.raw_ptr() + slot.offset_elems;
    }

    template <class XFFTComplex3>
    value_type *native_forward_second_recv_target_ptr_(
        int source_i, const native_schedule_slot &slot, XFFTComplex3 &out
    )
    {
        if ( native_forward_second_receive_lands_in_stage_() )
            return out.raw_ptr() + transpose1_dim_.start_x[source_i];
        return native_forward_second_device_recv_buffer_ptr_( slot );
    }

    std::size_t native_forward_second_recv_x_size_(
        int source_i, const native_schedule_slot &slot, const char *stage
    ) const
    {
        const std::size_t yz = ny_output_local_ * nz_local_;
        if ( yz == 0 || slot.elems % yz != 0 )
        {
            throw std::logic_error(
                "owned reference native forward-second schedule has invalid recv extent for " + std::string( stage )
            );
        }
        const std::size_t x_size = slot.elems / yz;
        if ( x_size != transpose1_dim_.size_x[source_i] )
        {
            std::ostringstream ss;
            ss << "owned reference native forward-second recv extent mismatch for " << stage
               << " peer=" << source_i << " schedule_x=" << x_size
               << " partition_x=" << transpose1_dim_.size_x[source_i];
            throw std::logic_error( ss.str() );
        }
        return x_size;
    }

    std::size_t second_backward_send_elems_( int target_i ) const
    {
        return native_or_legacy_schedule_elems_(
            second_backward_send_schedule_, target_i, legacy_second_backward_send_elems_( target_i )
        );
    }

    std::size_t second_backward_send_offset_( int target_i ) const
    {
        return native_or_legacy_schedule_offset_(
            second_backward_send_schedule_, target_i, legacy_second_backward_send_offset_( target_i )
        );
    }

    std::size_t second_backward_recv_elems_( int source_i ) const
    {
        return native_or_legacy_schedule_elems_(
            second_backward_recv_schedule_, source_i, legacy_second_backward_recv_elems_( source_i )
        );
    }

    std::size_t second_backward_recv_offset_( int source_i ) const
    {
        return native_or_legacy_schedule_offset_(
            second_backward_recv_schedule_, source_i, legacy_second_backward_recv_offset_( source_i )
        );
    }

    std::size_t second_backward_packed_recv_offset_( int source_i ) const
    {
        return second_backward_recv_offset_( source_i );
    }

    value_type *native_backward_second_device_send_buffer_ptr_( int target_i )
    {
        validate_native_execution_slot_(
            second_backward_send_schedule_, target_i, myid_i_, "second_backward_device_send_buffer"
        );
        return send_buffer_.raw_ptr() + second_backward_send_offset_( target_i );
    }

    value_type *native_backward_second_host_send_buffer_ptr_( int target_i )
    {
        validate_native_execution_slot_(
            second_backward_send_schedule_, target_i, myid_i_, "second_backward_host_send_buffer"
        );
        return host_send_buffer_.raw_ptr() + second_backward_send_offset_( target_i );
    }

    template <class XFastComplex3>
    value_type *native_backward_second_recv_output_ptr_( int source_i, XFastComplex3 &out )
    {
        validate_native_execution_slot_(
            second_backward_recv_schedule_, source_i, source_i, "second_backward_recv_output"
        );
        return out.raw_ptr() + second_backward_recv_offset_( source_i );
    }

    value_type *native_backward_second_device_recv_buffer_ptr_( int source_i )
    {
        validate_native_execution_slot_(
            second_backward_recv_schedule_, source_i, source_i, "second_backward_device_recv_buffer"
        );
        return recv_buffer_.raw_ptr() + second_backward_packed_recv_offset_( source_i );
    }

    value_type *native_backward_second_host_recv_buffer_ptr_( int source_i )
    {
        validate_native_execution_slot_(
            second_backward_recv_schedule_, source_i, source_i, "second_backward_host_recv_buffer"
        );
        return host_recv_buffer_.raw_ptr() + second_backward_packed_recv_offset_( source_i );
    }

    template <class XFastComplex3>
    value_type *native_backward_second_recv_target_ptr_( int source_i, XFastComplex3 &out )
    {
        validate_native_execution_slot_(
            second_backward_recv_schedule_, source_i, source_i, "second_backward_recv_target"
        );
        if ( backward_receive_lands_in_output_() )
            return native_backward_second_recv_output_ptr_( source_i, out );
        return native_backward_second_device_recv_buffer_ptr_( source_i );
    }

    const native_schedule_slot &native_backward_second_send_slot_( int target_i, const char *stage ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( second_backward_send_schedule_, target_i, stage );
        if ( slot == nullptr )
            throw std::logic_error( "owned reference native backward-second send slot is not active" );
        return *slot;
    }

    const native_schedule_slot &native_backward_second_recv_slot_( int source_i, const char *stage ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( second_backward_recv_schedule_, source_i, stage );
        if ( slot == nullptr )
            throw std::logic_error( "owned reference native backward-second recv slot is not active" );
        return *slot;
    }

    value_type *native_backward_second_device_send_buffer_ptr_( const native_schedule_slot &slot )
    {
        return send_buffer_.raw_ptr() + slot.offset_elems;
    }

    value_type *native_backward_second_host_send_buffer_ptr_( const native_schedule_slot &slot )
    {
        return host_send_buffer_.raw_ptr() + slot.offset_elems;
    }

    template <class XFastComplex3>
    value_type *native_backward_second_recv_output_ptr_( const native_schedule_slot &slot, XFastComplex3 &out )
    {
        return out.raw_ptr() + slot.offset_elems;
    }

    value_type *native_backward_second_device_recv_buffer_ptr_( const native_schedule_slot &slot )
    {
        return recv_buffer_.raw_ptr() + slot.offset_elems;
    }

    value_type *native_backward_second_host_recv_buffer_ptr_( const native_schedule_slot &slot )
    {
        return host_recv_buffer_.raw_ptr() + slot.offset_elems;
    }

    template <class XFastComplex3>
    value_type *native_backward_second_recv_target_ptr_( const native_schedule_slot &slot, XFastComplex3 &out )
    {
        if ( backward_receive_lands_in_output_() )
            return native_backward_second_recv_output_ptr_( slot, out );
        return native_backward_second_device_recv_buffer_ptr_( slot );
    }

    std::size_t native_backward_second_send_x_size_(
        int target_i, const native_schedule_slot &slot, const char *stage
    ) const
    {
        const std::size_t yz = ny_output_local_ * nz_local_;
        if ( yz == 0 || slot.elems % yz != 0 )
        {
            throw std::logic_error(
                "owned reference native backward-second schedule has invalid send extent for " + std::string( stage )
            );
        }
        const std::size_t x_size = slot.elems / yz;
        if ( x_size != transpose1_dim_.size_x[target_i] )
        {
            std::ostringstream ss;
            ss << "owned reference native backward-second send extent mismatch for " << stage
               << " peer=" << target_i << " schedule_x=" << x_size
               << " partition_x=" << transpose1_dim_.size_x[target_i];
            throw std::logic_error( ss.str() );
        }
        return x_size;
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

    void copy_first_forward_zfast_source_to_send_async_(
        const value_type *src_ptr, int target_j, value_type *dst_ptr, typename runtime_api_t::memcpy_kind_t kind,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( transpose1_dim_.start_z[target_j] * sizeof( value_type ), 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_global_ * sizeof( value_type ), nz_global_, ny_input_local_ );
        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPtr = runtime_api_t::make_pitched_ptr(
            dst_ptr, transpose1_dim_.size_z[target_j] * sizeof( value_type ), transpose1_dim_.size_z[target_j],
            ny_input_local_
        );
        params.extent =
            runtime_api_t::make_extent( transpose1_dim_.size_z[target_j] * sizeof( value_type ), ny_input_local_, nx_local_ );
        params.kind = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_first_forward_zfast_source_to_stage1_async_(
        const value_type *src_ptr, std::size_t src_z_offset, std::size_t src_z_size, std::size_t dst_y_offset,
        value_type *dst_ptr, typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( src_z_offset * sizeof( value_type ), 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_global_ * sizeof( value_type ), nz_global_, ny_input_local_ );
        params.dstPos = runtime_api_t::make_pos( 0, dst_y_offset, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.extent = runtime_api_t::make_extent( src_z_size * sizeof( value_type ), ny_input_local_, nx_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_first_forward_zfast_recv_to_stage1_async_(
        const value_type *src_ptr, std::size_t src_y_size, std::size_t dst_y_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, src_y_size );
        params.dstPos = runtime_api_t::make_pos( 0, dst_y_offset, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), src_y_size, nx_local_ );
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

    void pack_first_backward_zfast_chunk_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t packed_y_size, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, src_y_offset, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, packed_y_size );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), packed_y_size, nx_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_first_backward_zfast_recv_to_default_z_async_(
        const value_type *src_ptr, std::size_t src_z_size, std::size_t dst_z_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, src_z_size * sizeof( value_type ), src_z_size, ny_input_local_ );
        params.dstPos = runtime_api_t::make_pos( dst_z_offset * sizeof( value_type ), 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_global_ * sizeof( value_type ), nz_global_, ny_input_local_ );
        params.extent = runtime_api_t::make_extent( src_z_size * sizeof( value_type ), ny_input_local_, nx_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_first_backward_zfast_stage1_to_default_z_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t src_y_size, std::size_t dst_z_offset,
        std::size_t dst_z_size, value_type *dst_ptr, typename runtime_api_t::memcpy_kind_t kind,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, src_y_offset, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.dstPos = runtime_api_t::make_pos( dst_z_offset * sizeof( value_type ), 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_global_ * sizeof( value_type ), nz_global_, ny_input_local_ );
        params.extent = runtime_api_t::make_extent( dst_z_size * sizeof( value_type ), src_y_size, nx_local_ );
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

    void pack_second_forward_zfast_chunk_async_(
        const value_type *src_ptr, int target_i, value_type *dst_ptr, typename runtime_api_t::memcpy_kind_t kind,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, output_dim_.start_y[target_i], 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPtr = runtime_api_t::make_pitched_ptr(
            dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, output_dim_.size_y[target_i]
        );
        params.extent =
            runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), output_dim_.size_y[target_i], nx_local_ );
        params.kind = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_second_forward_zfast_recv_to_output_async_(
        const value_type *src_ptr, std::size_t src_x_size, std::size_t dst_x_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.dstPos = runtime_api_t::make_pos( 0, 0, dst_x_offset );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), ny_output_local_, src_x_size );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_second_forward_zfast_stage1_to_output_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t src_y_size, std::size_t dst_x_offset,
        value_type *dst_ptr, typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, src_y_offset, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.dstPos = runtime_api_t::make_pos( 0, 0, dst_x_offset );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), src_y_size, nx_local_ );
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

    void pack_second_backward_zfast_chunk_async_(
        const value_type *src_ptr, std::size_t src_x_offset, std::size_t packed_x_size, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, 0, src_x_offset );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.dstPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), ny_output_local_, packed_x_size );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_second_backward_zfast_recv_to_stage1_async_(
        const value_type *src_ptr, std::size_t src_y_size, std::size_t dst_y_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.srcPos = runtime_api_t::make_pos( 0, 0, 0 );
        params.srcPtr =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, src_y_size );
        params.dstPos = runtime_api_t::make_pos( 0, dst_y_offset, 0 );
        params.dstPtr =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), src_y_size, nx_local_ );
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
        if ( mode_ == mpi_transpose_3d_mode::p2p_waitall && !reference_parity_forward_waitany_enabled_() )
            forward_first_waitall_( in, out );
        else
            forward_first_waitany_( in, out );
    }

    template <class Stage0Complex3>
    void pack_first_forward_chunk_( int p, const Stage0Complex3 &in )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *dst  = native_forward_first_device_send_buffer_ptr_( p );
        auto        kind = runtime_api_t::device_to_device_kind();
#else
        value_type *dst  = native_forward_first_host_send_buffer_ptr_( p );
        auto        kind = runtime_api_t::device_to_host_kind();
#endif
        if ( native_opt0_default_z_layout_ )
        {
            copy_first_forward_zfast_source_to_send_async_( in.raw_ptr(), p, dst, kind, streams_[p].stream() );
        }
        else
        {
            runtime_api_t::memcpy_async(
                dst, native_forward_first_send_source_ptr_( p, in ), bytes_from_elems_( first_forward_send_elems_( p ) ),
                kind, streams_[p].stream()
            );
        }
    }

    int first_forward_byte_send_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : row_comm_order_ )
            total_chunks += contiguous_forward_send_request_count_( first_forward_send_elems_( p ) );
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
        const value_type *send_ptr = native_forward_first_device_send_buffer_ptr_( p );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const std::size_t elems = first_forward_send_elems_( p );
            const std::size_t bytes = bytes_from_elems_( elems );
            if ( can_use_physical_forward_value_send_( elems ) )
            {
                post_contiguous_forward_value_isend_(
                    row_comm_info_, send_ptr, elems, p, myid_j_, byte_send_requests,
                    "first/physical_forward_send_post_value_count"
                );
                return;
            }
            if ( contiguous_forward_chunked_send_enabled_() )
            {
                post_contiguous_forward_chunked_byte_isend_(
                    row_comm_info_, send_ptr, bytes, p, myid_j_, byte_send_requests,
                    "first/contiguous_forward_send_post_chunked_byte"
                );
                return;
            }
            if ( can_use_contiguous_forward_value_send_( elems ) )
            {
                post_contiguous_forward_value_isend_(
                    row_comm_info_, send_ptr, elems, p, myid_j_, byte_send_requests,
                    "first/contiguous_forward_send_post_value_count"
                );
                return;
            }
            post_byte_isend_(
                row_comm_info_, send_ptr, bytes, p, myid_j_, byte_send_requests,
                persistent_byte_p2p_enabled_() ? &persistent_first_forward_send_ : nullptr, &persistent_byte_index,
                cached_large_byte_datatype_( first_forward_send_large_byte_types_, p, bytes )
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
            row_comm_info_, native_forward_first_host_send_buffer_ptr_( p ),
            first_forward_send_elems_( p ), p, myid_j_, row_send_requests_[p], persistent_first_forward_send_, p
        );
#endif
    }

    void forward_first_post_physical_stage_send_(
        int p, const value_type *send_ptr, std::vector<mpi_request_t> &byte_send_requests
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const std::size_t elems = first_forward_send_elems_( p );
        if ( !can_use_physical_forward_value_send_( elems ) )
        {
            throw std::logic_error( "owned reference physical forward first-stage send cannot use value count" );
        }
        post_contiguous_forward_value_isend_(
            row_comm_info_, send_ptr, elems, p, myid_j_, byte_send_requests,
            "first/physical_forward_stage_send_post_value_count"
        );
#else
        (void)p;
        (void)send_ptr;
        (void)byte_send_requests;
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
	            value_type *recv_ptr = native_forward_first_recv_target_ptr_( p, out );
	            if ( cuda_aware_byte_p2p_enabled_() )
	            {
                    if ( direct_reference_opt1_forward_byte_receive_enabled_() )
                    {
                        post_direct_forward_byte_irecv_(
                            row_comm_info_, recv_ptr, first_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                            byte_recv_peers, byte_recv_remaining
                        );
                    }
                    else
                    {
                            const std::size_t bytes = bytes_from_elems_( first_forward_recv_elems_( p ) );
		                    post_forward_byte_irecv_(
		                        row_comm_info_, recv_ptr, bytes, p, p,
		                        byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
			                        &persistent_byte_recv_index, "first/contiguous_forward_recv_post_chunked_byte",
                                    "first/physical_forward_recv_post_value_count",
	                                cached_large_byte_datatype_( first_forward_recv_large_byte_types_, p, bytes )
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
                row_comm_info_, native_forward_first_host_recv_buffer_ptr_( p ),
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
                total_chunks += contiguous_forward_send_request_count_( first_forward_send_elems_( p ) );
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
            const bool use_physical_forward_peer_exchange = physical_forward_peer_exchange_enabled_();
            if ( !use_physical_forward_peer_exchange && ready_stable_forward_byte_send_buffer_enabled_() )
            {
                for ( const int p : row_comm_order_ )
                {
	                value_type *recv_ptr = native_forward_first_recv_target_ptr_( p, out );
	                {
	                    auto phase = profile_scope_( "first/mpi_post_recv" );
                        if ( direct_reference_opt1_forward_byte_receive_enabled_() )
                        {
                            post_direct_forward_byte_irecv_(
                                row_comm_info_, recv_ptr, first_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                                byte_recv_peers, byte_recv_remaining
                            );
                        }
                        else
                        {
                                const std::size_t bytes = bytes_from_elems_( first_forward_recv_elems_( p ) );
		                        post_forward_byte_irecv_(
		                            row_comm_info_, recv_ptr, bytes, p, p,
		                            byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
			                            &persistent_byte_recv_index, "first/contiguous_forward_recv_post_chunked_byte",
                                        "first/physical_forward_recv_post_value_count",
	                                    cached_large_byte_datatype_( first_forward_recv_large_byte_types_, p, bytes )
			                        );
                        }
                    }
                }

                for ( const int p : row_comm_order_ )
                {
                    auto phase = profile_scope_( "first/stable_send_buffer_fill" );
                    pack_first_forward_chunk_( p, in );
                }

                std::vector<char> posted( row_size, 0 );
                int remaining = static_cast<int>( row_comm_order_.size() );
                while ( remaining > 0 )
                {
                    std::vector<int> ready_peers;
                    {
                        auto phase = profile_scope_( "first/stable_wait_send_buffer_ready" );
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
                return;
            }

            for ( const int p : row_comm_order_ )
            {
	                value_type *recv_ptr = native_forward_first_recv_target_ptr_( p, out );
	                {
	                    auto phase = profile_scope_( "first/mpi_post_recv" );
                        if ( direct_reference_opt1_forward_byte_receive_enabled_() )
                        {
                            post_direct_forward_byte_irecv_(
                                row_comm_info_, recv_ptr, first_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                                byte_recv_peers, byte_recv_remaining
                            );
                        }
                        else
                        {
                                const std::size_t bytes = bytes_from_elems_( first_forward_recv_elems_( p ) );
		                        post_forward_byte_irecv_(
		                            row_comm_info_, recv_ptr, bytes, p, p,
		                            byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
			                            &persistent_byte_recv_index, "first/contiguous_forward_recv_post_chunked_byte",
                                        "first/physical_forward_recv_post_value_count",
	                                    cached_large_byte_datatype_( first_forward_recv_large_byte_types_, p, bytes )
			                        );
                        }
                }
	                if ( ( use_stable_forward_byte_send_buffer_ || native_opt0_default_z_layout_ ) && !use_physical_forward_peer_exchange )
	                {
	                    {
	                        auto phase = profile_scope_( "first/stable_send_buffer_fill" );
                        pack_first_forward_chunk_( p, in );
                    }
                    {
                        auto phase = profile_scope_( "first/stable_send_buffer_sync" );
                        runtime_api_t::stream_synchronize( streams_[p].stream() );
                    }
                }
		                {
		                    auto phase = profile_scope_( "first/mpi_post_send" );
		                    if ( use_physical_forward_peer_exchange )
		                        forward_first_post_physical_stage_send_(
                                    p, native_forward_first_send_source_ptr_( p, in ), byte_send_requests
                                );
		                    else if ( use_stable_forward_byte_send_buffer_ || native_opt0_default_z_layout_ )
		                        forward_first_post_send_( p, byte_send_requests, persistent_byte_index );
		                    else
                        {
                            const std::size_t bytes = bytes_from_elems_( first_forward_send_elems_( p ) );
	                        post_byte_isend_(
	                            row_comm_info_, native_forward_first_send_source_ptr_( p, in ),
	                            bytes, p, myid_j_, byte_send_requests, persistent_byte_state, &persistent_byte_index,
                                cached_large_byte_datatype_( first_forward_send_large_byte_types_, p, bytes )
	                        );
                        }
	                }
            }
            return;
        }
#endif
        for ( const int p : row_comm_order_ )
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
	            value_type *recv_ptr = native_forward_first_recv_target_ptr_( p, out );
	            if ( cuda_aware_byte_p2p_enabled_() )
	            {
	                auto phase = profile_scope_( "first/mpi_post_recv" );
                    if ( direct_reference_opt1_forward_byte_receive_enabled_() )
                    {
                        post_direct_forward_byte_irecv_(
                            row_comm_info_, recv_ptr, first_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                            byte_recv_peers, byte_recv_remaining
                        );
                    }
                    else
                    {
                            const std::size_t bytes = bytes_from_elems_( first_forward_recv_elems_( p ) );
		                    post_forward_byte_irecv_(
		                        row_comm_info_, recv_ptr, bytes, p, p,
		                        byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
			                        &persistent_byte_recv_index, "first/contiguous_forward_recv_post_chunked_byte",
                                    "first/physical_forward_recv_post_value_count",
	                                cached_large_byte_datatype_( first_forward_recv_large_byte_types_, p, bytes )
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
                    row_comm_info_, native_forward_first_host_recv_buffer_ptr_( p ),
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
                if ( native_opt0_default_z_layout_ )
                {
                    pack_first_forward_chunk_( p, in );
                    runtime_api_t::stream_synchronize( streams_[p].stream() );
                    forward_first_post_send_( p, byte_send_requests, persistent_byte_index );
                }
                else
                {
                    const std::size_t bytes = bytes_from_elems_( first_forward_send_elems_( p ) );
                    post_byte_isend_(
                        row_comm_info_, native_forward_first_send_source_ptr_( p, in ),
                        bytes, p, myid_j_, byte_send_requests, persistent_byte_state, &persistent_byte_index,
                        cached_large_byte_datatype_( first_forward_send_large_byte_types_, p, bytes )
                    );
                }
            }
            else
            {
                auto phase = profile_scope_( "first/mpi_post_send" );
                post_value_isend_(
                    row_comm_info_, native_forward_first_send_source_ptr_( p, in ), first_forward_send_elems_( p ),
                    p, myid_j_, row_send_requests_[p], persistent_first_forward_send_, p
                );
            }
#else
            {
                auto phase = profile_scope_( "first/mpi_post_send" );
                post_value_isend_(
                    row_comm_info_, native_forward_first_host_send_buffer_ptr_( p ),
                    first_forward_send_elems_( p ), p, myid_j_, row_send_requests_[p], persistent_first_forward_send_, p
                );
            }
#endif
        }
    }

    template <class Stage0Complex3, class Stage1Complex3>
    void copy_first_forward_self_( const Stage0Complex3 &in, Stage1Complex3 &out )
    {
        if ( native_opt0_default_z_layout_ )
        {
            copy_first_forward_zfast_source_to_stage1_async_(
                in.raw_ptr(), transpose1_dim_.start_z[myid_j_], transpose1_dim_.size_z[myid_j_],
                half_input_dim_.start_y[myid_j_], out.raw_ptr(), runtime_api_t::device_to_device_kind(),
                streams_[myid_j_].stream()
            );
        }
        else
        {
            copy_first_forward_chunk_to_stage1_async_(
                in.raw_ptr() + first_forward_send_offset_( myid_j_ ), ny_input_local_, half_input_dim_.start_y[myid_j_],
                out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[myid_j_].stream()
            );
        }
    }

    template <class Stage1Complex3>
    void unpack_first_forward_recv_( int p, Stage1Complex3 &out )
    {
        if ( native_opt0_default_z_layout_ )
        {
            copy_first_forward_zfast_recv_to_stage1_async_(
                native_forward_first_device_recv_buffer_ptr_( p ), half_input_dim_.size_y[p], half_input_dim_.start_y[p],
                out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[p].stream()
            );
        }
        else
        {
            copy_first_forward_chunk_to_stage1_async_(
                native_forward_first_device_recv_buffer_ptr_( p ), half_input_dim_.size_y[p],
                half_input_dim_.start_y[p], out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[p].stream()
            );
        }
    }

    template <class Stage1Complex3>
    void unpack_first_forward_host_recv_( int p, Stage1Complex3 &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( native_opt0_default_z_layout_ )
        {
            copy_first_forward_zfast_recv_to_stage1_async_(
                native_forward_first_host_recv_buffer_ptr_( p ), half_input_dim_.size_y[p], half_input_dim_.start_y[p],
                out.raw_ptr(), runtime_api_t::host_to_device_kind(), streams_[p].stream()
            );
        }
        else
        {
            copy_first_forward_chunk_to_stage1_async_(
                native_forward_first_host_recv_buffer_ptr_( p ), half_input_dim_.size_y[p],
                half_input_dim_.start_y[p], out.raw_ptr(), runtime_api_t::host_to_device_kind(), streams_[p].stream()
            );
        }
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
	        if ( reference_byte_sync_enabled_() )
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
            wait_or_defer_send_requests_(
                deferred_send_stage::forward_first, deferred_send_communicator::row, row_comm_info_,
                byte_send_requests, persistent_first_forward_send_, row_send_requests_, row_size,
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
	        if ( reference_byte_sync_enabled_() )
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
            wait_or_defer_send_requests_(
                deferred_send_stage::forward_first, deferred_send_communicator::row, row_comm_info_,
                byte_send_requests, persistent_first_forward_send_, row_send_requests_, row_size,
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
        if ( mode_ == mpi_transpose_3d_mode::p2p_waitall && !reference_parity_forward_waitany_enabled_() )
            forward_second_xfast_waitall_( in, out );
        else
            forward_second_xfast_waitany_( in, out );
    }

    template <class XFastComplex3>
    void pack_second_forward_chunk_( int p, const XFastComplex3 &in )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *dst  = native_forward_second_device_send_buffer_ptr_( p );
        auto        kind = runtime_api_t::device_to_device_kind();
#else
        value_type *dst  = native_forward_second_host_send_buffer_ptr_( p );
        auto        kind = runtime_api_t::device_to_host_kind();
#endif
        if ( native_opt0_default_z_layout_ )
        {
            pack_second_forward_zfast_chunk_async_( in.raw_ptr(), p, dst, kind, streams_[p].stream() );
        }
        else
        {
            runtime_api_t::memcpy_async(
                dst, native_forward_second_send_source_ptr_( p, in ),
                bytes_from_elems_( second_forward_send_elems_( p ) ), kind, streams_[p].stream()
            );
        }
    }

    int second_forward_byte_send_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : line_comm_order_ )
            total_chunks += contiguous_forward_send_request_count_( second_forward_send_elems_( p ) );
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
        const value_type *send_ptr = native_forward_second_device_send_buffer_ptr_( p );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const std::size_t elems = second_forward_send_elems_( p );
            const std::size_t bytes = bytes_from_elems_( elems );
            if ( can_use_physical_forward_value_send_( elems ) )
            {
                post_contiguous_forward_value_isend_(
                    line_comm_info_, send_ptr, elems, p, myid_i_, byte_send_requests,
                    "second/physical_forward_send_post_value_count"
                );
                return;
            }
            if ( contiguous_forward_chunked_send_enabled_() )
            {
                post_contiguous_forward_chunked_byte_isend_(
                    line_comm_info_, send_ptr, bytes, p, myid_i_, byte_send_requests,
                    "second/contiguous_forward_send_post_chunked_byte"
                );
                return;
            }
            if ( can_use_contiguous_forward_value_send_( elems ) )
            {
                post_contiguous_forward_value_isend_(
                    line_comm_info_, send_ptr, elems, p, myid_i_, byte_send_requests,
                    "second/contiguous_forward_send_post_value_count"
                );
                return;
            }
            post_byte_isend_(
                line_comm_info_, send_ptr, bytes, p, myid_i_, byte_send_requests,
                persistent_byte_p2p_enabled_() ? &persistent_second_forward_send_ : nullptr, &persistent_byte_index,
                cached_large_byte_datatype_( second_forward_send_large_byte_types_, p, bytes )
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
            line_comm_info_, native_forward_second_host_send_buffer_ptr_( p ),
            second_forward_send_elems_( p ), p, myid_i_, line_send_requests_[p], persistent_second_forward_send_, p
        );
#endif
    }

    void forward_second_post_physical_stage_send_(
        int p, const value_type *send_ptr, std::vector<mpi_request_t> &byte_send_requests
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const std::size_t elems = second_forward_send_elems_( p );
        if ( !can_use_physical_forward_value_send_( elems ) )
        {
            throw std::logic_error( "owned reference physical forward second-stage send cannot use value count" );
        }
        post_contiguous_forward_value_isend_(
            line_comm_info_, send_ptr, elems, p, myid_i_, byte_send_requests,
            "second/physical_forward_stage_send_post_value_count"
        );
#else
        (void)p;
        (void)send_ptr;
        (void)byte_send_requests;
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
	            value_type *recv_ptr = native_forward_second_recv_target_ptr_( p, out );
	            if ( cuda_aware_byte_p2p_enabled_() )
	            {
                    if ( direct_reference_opt1_forward_byte_receive_enabled_() )
                    {
                        post_direct_forward_byte_irecv_(
                            line_comm_info_, recv_ptr, second_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                            byte_recv_peers, byte_recv_remaining
                        );
                    }
                    else
                    {
                        const std::size_t bytes = bytes_from_elems_( second_forward_recv_elems_( p ) );
	                    post_forward_byte_irecv_(
	                        line_comm_info_, recv_ptr, bytes, p, p,
	                        byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
	                        &persistent_byte_recv_index, "second/contiguous_forward_recv_post_chunked_byte",
                            "second/physical_forward_recv_post_value_count",
	                            cached_large_byte_datatype_( second_forward_recv_large_byte_types_, p, bytes )
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
                line_comm_info_, native_forward_second_host_recv_buffer_ptr_( p ),
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
                total_chunks += contiguous_forward_send_request_count_( second_forward_send_elems_( p ) );
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
            const bool use_physical_forward_peer_exchange = physical_forward_peer_exchange_enabled_();
            if ( !use_physical_forward_peer_exchange && ready_stable_forward_byte_send_buffer_enabled_() )
            {
                for ( const int p : line_comm_order_ )
                {
	                value_type *recv_ptr = native_forward_second_recv_target_ptr_( p, out );
	                {
	                    auto phase = profile_scope_( "second/mpi_post_recv" );
                        if ( direct_reference_opt1_forward_byte_receive_enabled_() )
                        {
                            post_direct_forward_byte_irecv_(
                                line_comm_info_, recv_ptr, second_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                                byte_recv_peers, byte_recv_remaining
                            );
                        }
                        else
                        {
                            const std::size_t bytes = bytes_from_elems_( second_forward_recv_elems_( p ) );
	                        post_forward_byte_irecv_(
	                            line_comm_info_, recv_ptr, bytes, p, p,
	                            byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
		                            &persistent_byte_recv_index, "second/contiguous_forward_recv_post_chunked_byte",
                                    "second/physical_forward_recv_post_value_count",
	                                cached_large_byte_datatype_( second_forward_recv_large_byte_types_, p, bytes )
		                        );
                        }
                    }
                }

                for ( const int p : line_comm_order_ )
                {
                    auto phase = profile_scope_( "second/stable_send_buffer_fill" );
                    pack_second_forward_chunk_( p, in );
                }

                std::vector<char> posted( line_size, 0 );
                int remaining = static_cast<int>( line_comm_order_.size() );
                while ( remaining > 0 )
                {
                    std::vector<int> ready_peers;
                    {
                        auto phase = profile_scope_( "second/stable_wait_send_buffer_ready" );
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
                return;
            }

            for ( const int p : line_comm_order_ )
            {
	                value_type *recv_ptr = native_forward_second_recv_target_ptr_( p, out );
	                {
	                    auto phase = profile_scope_( "second/mpi_post_recv" );
                        if ( direct_reference_opt1_forward_byte_receive_enabled_() )
                        {
                            post_direct_forward_byte_irecv_(
                                line_comm_info_, recv_ptr, second_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                                byte_recv_peers, byte_recv_remaining
                            );
                        }
                        else
                        {
                                const std::size_t bytes = bytes_from_elems_( second_forward_recv_elems_( p ) );
		                        post_forward_byte_irecv_(
		                            line_comm_info_, recv_ptr, bytes, p, p,
		                            byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
			                            &persistent_byte_recv_index, "second/contiguous_forward_recv_post_chunked_byte",
                                        "second/physical_forward_recv_post_value_count",
	                                    cached_large_byte_datatype_( second_forward_recv_large_byte_types_, p, bytes )
			                        );
                        }
                }
	                if ( ( use_stable_forward_byte_send_buffer_ || native_opt0_default_z_layout_ ) && !use_physical_forward_peer_exchange )
	                {
	                    {
	                        auto phase = profile_scope_( "second/stable_send_buffer_fill" );
                        pack_second_forward_chunk_( p, in );
                    }
                    {
                        auto phase = profile_scope_( "second/stable_send_buffer_sync" );
                        runtime_api_t::stream_synchronize( streams_[p].stream() );
                    }
                }
		                {
		                    auto phase = profile_scope_( "second/mpi_post_send" );
		                    if ( use_physical_forward_peer_exchange )
		                        forward_second_post_physical_stage_send_(
                                    p, native_forward_second_send_source_ptr_( p, in ), byte_send_requests
                                );
		                    else if ( use_stable_forward_byte_send_buffer_ || native_opt0_default_z_layout_ )
		                        forward_second_post_send_( p, byte_send_requests, persistent_byte_index );
		                    else
                        {
                            const std::size_t bytes = bytes_from_elems_( second_forward_send_elems_( p ) );
	                        post_byte_isend_(
	                            line_comm_info_, native_forward_second_send_source_ptr_( p, in ),
	                            bytes, p, myid_i_, byte_send_requests, persistent_byte_state, &persistent_byte_index,
                                cached_large_byte_datatype_( second_forward_send_large_byte_types_, p, bytes )
	                        );
                        }
	                }
            }
            return;
        }
#endif
	        for ( const int p : line_comm_order_ )
	        {
	#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
		            value_type *recv_ptr = native_forward_second_recv_target_ptr_( p, out );
		            if ( cuda_aware_byte_p2p_enabled_() )
		            {
		                auto phase = profile_scope_( "second/mpi_post_recv" );
                        if ( direct_reference_opt1_forward_byte_receive_enabled_() )
                        {
                            post_direct_forward_byte_irecv_(
                                line_comm_info_, recv_ptr, second_forward_direct_recvtypes_[p], p, p, byte_recv_requests,
                                byte_recv_peers, byte_recv_remaining
                            );
                        }
                        else
                        {
                                const std::size_t bytes = bytes_from_elems_( second_forward_recv_elems_( p ) );
			                    post_forward_byte_irecv_(
			                        line_comm_info_, recv_ptr, bytes, p, p,
			                        byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
				                        &persistent_byte_recv_index, "second/contiguous_forward_recv_post_chunked_byte",
                                        "second/physical_forward_recv_post_value_count",
	                                    cached_large_byte_datatype_( second_forward_recv_large_byte_types_, p, bytes )
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
	                    line_comm_info_, native_forward_second_host_recv_buffer_ptr_( p ),
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
                    if ( native_opt0_default_z_layout_ )
                    {
                        pack_second_forward_chunk_( p, in );
                        runtime_api_t::stream_synchronize( streams_[p].stream() );
                        forward_second_post_send_( p, byte_send_requests, persistent_byte_index );
                    }
                    else
                    {
                        const std::size_t bytes = bytes_from_elems_( second_forward_send_elems_( p ) );
	                    post_byte_isend_(
	                        line_comm_info_, native_forward_second_send_source_ptr_( p, in ),
	                        bytes, p, myid_i_, byte_send_requests, persistent_byte_state, &persistent_byte_index,
                            cached_large_byte_datatype_( second_forward_send_large_byte_types_, p, bytes )
	                    );
                    }
            }
	            else
	            {
	                auto phase = profile_scope_( "second/mpi_post_send" );
	                post_value_isend_(
	                    line_comm_info_, native_forward_second_send_source_ptr_( p, in ), second_forward_send_elems_( p ),
	                    p, myid_i_, line_send_requests_[p], persistent_second_forward_send_, p
	                );
	            }
	#else
	            {
	                auto phase = profile_scope_( "second/mpi_post_send" );
	                post_value_isend_(
	                    line_comm_info_, native_forward_second_host_send_buffer_ptr_( p ),
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
        if ( native_opt0_default_z_layout_ )
        {
            copy_second_forward_zfast_stage1_to_output_async_(
                in.raw_ptr(), output_dim_.start_y[myid_i_], ny_output_local_, transpose1_dim_.start_x[myid_i_],
                out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[myid_i_].stream()
            );
        }
        else
        {
            copy_second_forward_chunk_to_xstage_async_(
                in.raw_ptr() + second_forward_send_offset_( myid_i_ ), nx_local_, transpose1_dim_.start_x[myid_i_],
                out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[myid_i_].stream()
            );
        }
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
        if ( native_opt0_default_z_layout_ )
        {
            copy_second_forward_zfast_recv_to_output_async_(
                native_forward_second_device_recv_buffer_ptr_( p ), transpose1_dim_.size_x[p],
                transpose1_dim_.start_x[p], out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[p].stream()
            );
        }
        else
        {
            copy_second_forward_chunk_to_xstage_async_(
                native_forward_second_device_recv_buffer_ptr_( p ), transpose1_dim_.size_x[p],
                transpose1_dim_.start_x[p], out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[p].stream()
            );
        }
    }

    template <class XFFTComplex3>
    void unpack_second_forward_host_recv_( int p, XFFTComplex3 &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( native_opt0_default_z_layout_ )
        {
            copy_second_forward_zfast_recv_to_output_async_(
                native_forward_second_host_recv_buffer_ptr_( p ), transpose1_dim_.size_x[p],
                transpose1_dim_.start_x[p], out.raw_ptr(), runtime_api_t::host_to_device_kind(), streams_[p].stream()
            );
        }
        else
        {
            copy_second_forward_chunk_to_xstage_async_(
                native_forward_second_host_recv_buffer_ptr_( p ), transpose1_dim_.size_x[p],
                transpose1_dim_.start_x[p], out.raw_ptr(), runtime_api_t::host_to_device_kind(), streams_[p].stream()
            );
        }
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
	        if ( reference_byte_sync_enabled_() )
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
		        if ( !native_forward_second_receive_lands_in_stage_() )
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
            wait_or_defer_send_requests_(
                deferred_send_stage::forward_second, deferred_send_communicator::line, line_comm_info_,
                byte_send_requests, persistent_second_forward_send_, line_send_requests_, line_size,
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
	        if ( reference_byte_sync_enabled_() )
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
		                if ( byte_recv_remaining[p] == 0 && !native_forward_second_receive_lands_in_stage_() )
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
		                if ( !native_forward_second_receive_lands_in_stage_() )
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
            wait_or_defer_send_requests_(
                deferred_send_stage::forward_second, deferred_send_communicator::line, line_comm_info_,
                byte_send_requests, persistent_second_forward_send_, line_send_requests_, line_size,
                "owned second byte forward send requests", "owned second persistent forward send requests"
            );
        }
    }

    template <class XFastComplex3>
    void native_pack_second_forward_chunk_(
        int p, const native_schedule_slot &send_slot, const XFastComplex3 &in
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *dst  = native_forward_second_device_send_buffer_ptr_( send_slot );
        auto        kind = runtime_api_t::device_to_device_kind();
#else
        value_type *dst  = native_forward_second_host_send_buffer_ptr_( send_slot );
        auto        kind = runtime_api_t::device_to_host_kind();
#endif
        if ( native_opt0_default_z_layout_ )
        {
            pack_second_forward_zfast_chunk_async_( in.raw_ptr(), p, dst, kind, streams_[p].stream() );
        }
        else
        {
            runtime_api_t::memcpy_async(
                dst, native_forward_second_send_source_ptr_( send_slot, in ), bytes_from_elems_( send_slot.elems ), kind,
                streams_[p].stream()
            );
        }
    }

    void native_forward_second_post_buffer_send_(
        int p, const native_schedule_slot &send_slot, std::vector<mpi_request_t> &byte_send_requests,
        int &persistent_byte_index
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const value_type *send_ptr = native_forward_second_device_send_buffer_ptr_( send_slot );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const std::size_t elems = send_slot.elems;
            const std::size_t bytes = bytes_from_elems_( elems );
            if ( can_use_physical_forward_value_send_( elems ) )
            {
                post_contiguous_forward_value_isend_(
                    line_comm_info_, send_ptr, elems, send_slot.mpi_peer, send_slot.mpi_tag, byte_send_requests,
                    "second/physical_forward_send_post_value_count"
                );
                return;
            }
            if ( contiguous_forward_chunked_send_enabled_() )
            {
                post_contiguous_forward_chunked_byte_isend_(
                    line_comm_info_, send_ptr, bytes, send_slot.mpi_peer, send_slot.mpi_tag, byte_send_requests,
                    "second/contiguous_forward_send_post_chunked_byte"
                );
                return;
            }
            if ( can_use_contiguous_forward_value_send_( elems ) )
            {
                post_contiguous_forward_value_isend_(
                    line_comm_info_, send_ptr, elems, send_slot.mpi_peer, send_slot.mpi_tag, byte_send_requests,
                    "second/contiguous_forward_send_post_value_count"
                );
                return;
            }
            post_byte_isend_(
                line_comm_info_, send_ptr, bytes, send_slot.mpi_peer, send_slot.mpi_tag, byte_send_requests,
                persistent_byte_p2p_enabled_() ? &persistent_second_forward_send_ : nullptr, &persistent_byte_index,
                cached_large_byte_datatype_( second_forward_send_large_byte_types_, p, bytes )
            );
        }
        else
        {
            post_value_isend_(
                line_comm_info_, send_ptr, send_slot.elems, send_slot.mpi_peer, send_slot.mpi_tag,
                line_send_requests_[p], persistent_second_forward_send_, p
            );
        }
#else
        post_value_isend_(
            line_comm_info_, native_forward_second_host_send_buffer_ptr_( send_slot ), send_slot.elems,
            send_slot.mpi_peer, send_slot.mpi_tag, line_send_requests_[p], persistent_second_forward_send_, p
        );
#endif
    }

    void native_forward_second_post_physical_stage_send_(
        const native_schedule_slot &send_slot, const value_type *send_ptr,
        std::vector<mpi_request_t> &byte_send_requests
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( !can_use_physical_forward_value_send_( send_slot.elems ) )
        {
            throw std::logic_error( "native forward second-stage send cannot use value count" );
        }
        post_contiguous_forward_value_isend_(
            line_comm_info_, send_ptr, send_slot.elems, send_slot.mpi_peer, send_slot.mpi_tag, byte_send_requests,
            "second/physical_forward_stage_send_post_value_count"
        );
#else
        (void)send_slot;
        (void)send_ptr;
        (void)byte_send_requests;
#endif
    }

    template <class XFastComplex3>
    void native_forward_second_post_direct_send_(
        int p, const native_schedule_slot &send_slot, const XFastComplex3 &in,
        detail::persistent_send_requests<mpi_request_t> *persistent_byte_state, int &persistent_byte_index,
        std::vector<mpi_request_t> &byte_send_requests
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const std::size_t bytes = bytes_from_elems_( send_slot.elems );
        post_byte_isend_(
            line_comm_info_, native_forward_second_send_source_ptr_( send_slot, in ), bytes, send_slot.mpi_peer,
            send_slot.mpi_tag, byte_send_requests, persistent_byte_state, &persistent_byte_index,
            cached_large_byte_datatype_( second_forward_send_large_byte_types_, p, bytes )
        );
#else
        post_value_isend_(
            line_comm_info_, native_forward_second_host_send_buffer_ptr_( send_slot ), send_slot.elems,
            send_slot.mpi_peer, send_slot.mpi_tag, line_send_requests_[p], persistent_second_forward_send_, p
        );
        (void)in;
        (void)persistent_byte_state;
        (void)persistent_byte_index;
        (void)byte_send_requests;
#endif
    }

    template <class XFFTComplex3>
    void native_forward_second_post_recv_(
        int p, const native_schedule_slot &recv_slot, XFFTComplex3 &out,
        std::vector<mpi_request_t> &byte_recv_requests, std::vector<int> *byte_recv_peers,
        std::vector<int> *byte_recv_remaining,
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state, int &persistent_byte_recv_index
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *recv_ptr = native_forward_second_recv_target_ptr_( p, recv_slot, out );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            if ( direct_reference_opt1_forward_byte_receive_enabled_() )
            {
                post_direct_forward_byte_irecv_(
                    line_comm_info_, recv_ptr, second_forward_direct_recvtypes_[p], recv_slot.mpi_peer,
                    recv_slot.mpi_tag, byte_recv_requests, byte_recv_peers, byte_recv_remaining
                );
            }
            else
            {
                const std::size_t bytes = bytes_from_elems_( recv_slot.elems );
                post_forward_byte_irecv_(
                    line_comm_info_, recv_ptr, bytes, recv_slot.mpi_peer, recv_slot.mpi_tag, byte_recv_requests,
                    byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state, &persistent_byte_recv_index,
                    "second/contiguous_forward_recv_post_chunked_byte",
                    "second/physical_forward_recv_post_value_count",
                    cached_large_byte_datatype_( second_forward_recv_large_byte_types_, p, bytes )
                );
            }
        }
        else
        {
            if ( direct_forward_value_receive_enabled_() )
                post_typed_irecv_(
                    line_comm_info_, recv_ptr, 1, second_forward_direct_recvtypes_[p], recv_slot.mpi_peer,
                    recv_slot.mpi_tag, line_recv_requests_[p], persistent_second_forward_recv_, p
                );
            else
                post_value_irecv_(
                    line_comm_info_, recv_ptr, recv_slot.elems, recv_slot.mpi_peer, recv_slot.mpi_tag,
                    line_recv_requests_[p], persistent_second_forward_recv_, p
                );
        }
#else
        post_value_irecv_(
            line_comm_info_, native_forward_second_host_recv_buffer_ptr_( recv_slot ), recv_slot.elems,
            recv_slot.mpi_peer, recv_slot.mpi_tag, line_recv_requests_[p], persistent_second_forward_recv_, p
        );
        (void)out;
        (void)byte_recv_requests;
        (void)byte_recv_peers;
        (void)byte_recv_remaining;
        (void)persistent_byte_recv_state;
        (void)persistent_byte_recv_index;
#endif
    }

    template <class XFastComplex3, class XFFTComplex3>
    void native_forward_second_post_sends_recvs_(
        const XFastComplex3 &in, XFFTComplex3 &out, std::vector<mpi_request_t> &byte_recv_requests,
        std::vector<mpi_request_t> &byte_send_requests, std::vector<int> *byte_recv_peers = nullptr,
        std::vector<int> *byte_recv_remaining = nullptr
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( !peer_paired_large_count_byte_schedule_enabled_() )
        {
            forward_second_post_sends_recvs_(
                in, out, byte_recv_requests, byte_send_requests, byte_recv_peers, byte_recv_remaining
            );
            return;
        }

        auto native_phase = profile_scope_( "native/forward_second_peer_paired_loop" );
        const int line_size = line_comm_info_.num_procs;
        int persistent_byte_index = 0;
        detail::persistent_send_requests<mpi_request_t> *persistent_byte_state = nullptr;
        if ( persistent_byte_p2p_enabled_() )
        {
            detail::reset_persistent_send_requests( persistent_second_forward_send_, second_forward_byte_send_chunks_() );
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

        profile_chunk_count_( "second", second_forward_byte_send_chunks_(), second_forward_byte_recv_chunks_() );
        synchronize_before_direct_byte_sends_( "second/pre_send_stream_sync" );

        const bool use_physical_forward_peer_exchange = physical_forward_peer_exchange_enabled_();
        if ( !use_physical_forward_peer_exchange && ready_stable_forward_byte_send_buffer_enabled_() )
        {
            for ( const int p : line_comm_order_ )
            {
                const native_schedule_slot &recv_slot =
                    native_forward_second_recv_slot_( p, "native_second_forward_recv_post" );
                {
                    auto phase = profile_scope_( "second/mpi_post_recv" );
                    native_forward_second_post_recv_(
                        p, recv_slot, out, byte_recv_requests, byte_recv_peers, byte_recv_remaining,
                        persistent_byte_recv_state, persistent_byte_recv_index
                    );
                }
            }

            for ( const int p : line_comm_order_ )
            {
                const native_schedule_slot &send_slot =
                    native_forward_second_send_slot_( p, "native_second_forward_stable_fill" );
                auto phase = profile_scope_( "second/stable_send_buffer_fill" );
                native_pack_second_forward_chunk_( p, send_slot, in );
            }

            std::vector<char> posted( line_size, 0 );
            int remaining = static_cast<int>( line_comm_order_.size() );
            while ( remaining > 0 )
            {
                std::vector<int> ready_peers;
                {
                    auto phase = profile_scope_( "second/stable_wait_send_buffer_ready" );
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
                    const native_schedule_slot &send_slot =
                        native_forward_second_send_slot_( p, "native_second_forward_ready_send" );
                    auto phase = profile_scope_( "second/mpi_post_send" );
                    native_forward_second_post_buffer_send_( p, send_slot, byte_send_requests, persistent_byte_index );
                    posted[p] = 1;
                    --remaining;
                }
            }
            return;
        }

        for ( const int p : line_comm_order_ )
        {
            const native_schedule_slot &recv_slot =
                native_forward_second_recv_slot_( p, "native_second_forward_recv_post" );
            const native_schedule_slot &send_slot =
                native_forward_second_send_slot_( p, "native_second_forward_send_post" );
            {
                auto phase = profile_scope_( "second/mpi_post_recv" );
                native_forward_second_post_recv_(
                    p, recv_slot, out, byte_recv_requests, byte_recv_peers, byte_recv_remaining,
                    persistent_byte_recv_state, persistent_byte_recv_index
                );
            }
            if ( ( use_stable_forward_byte_send_buffer_ || native_opt0_default_z_layout_ ) && !use_physical_forward_peer_exchange )
            {
                {
                    auto phase = profile_scope_( "second/stable_send_buffer_fill" );
                    native_pack_second_forward_chunk_( p, send_slot, in );
                }
                {
                    auto phase = profile_scope_( "second/stable_send_buffer_sync" );
                    runtime_api_t::stream_synchronize( streams_[p].stream() );
                }
            }
            {
                auto phase = profile_scope_( "second/mpi_post_send" );
                if ( use_physical_forward_peer_exchange )
                    native_forward_second_post_physical_stage_send_(
                        send_slot, native_forward_second_send_source_ptr_( send_slot, in ), byte_send_requests
                    );
                else if ( use_stable_forward_byte_send_buffer_ || native_opt0_default_z_layout_ )
                    native_forward_second_post_buffer_send_( p, send_slot, byte_send_requests, persistent_byte_index );
                else
                    native_forward_second_post_direct_send_(
                        p, send_slot, in, persistent_byte_state, persistent_byte_index, byte_send_requests
                    );
            }
        }
#else
        forward_second_post_sends_recvs_(
            in, out, byte_recv_requests, byte_send_requests, byte_recv_peers, byte_recv_remaining
        );
#endif
    }

    template <class XFastComplex3, class XFFTComplex3>
    void native_copy_second_forward_self_( const XFastComplex3 &in, XFFTComplex3 &out )
    {
        const native_schedule_slot *send_slot = nullptr;
        if ( native_schedule_active_( second_forward_send_schedule_ ) && myid_i_ >= 0 &&
             myid_i_ < static_cast<int>( second_forward_send_schedule_.size() ) &&
             second_forward_send_schedule_[myid_i_].valid )
        {
            send_slot = &second_forward_send_schedule_[myid_i_];
        }
        if ( native_opt0_default_z_layout_ )
        {
            copy_second_forward_zfast_stage1_to_output_async_(
                in.raw_ptr(), output_dim_.start_y[myid_i_], ny_output_local_, transpose1_dim_.start_x[myid_i_],
                out.raw_ptr(),
                runtime_api_t::device_to_device_kind(), streams_[myid_i_].stream()
            );
        }
        else
        {
            const value_type *src_ptr =
                send_slot != nullptr ? native_forward_second_send_source_ptr_( *send_slot, in )
                                     : in.raw_ptr() + legacy_second_forward_send_offset_( myid_i_ );
            copy_second_forward_chunk_to_xstage_async_(
                src_ptr, nx_local_, transpose1_dim_.start_x[myid_i_], out.raw_ptr(),
                runtime_api_t::device_to_device_kind(), streams_[myid_i_].stream()
            );
        }
    }

    template <class XFFTComplex3>
    void native_unpack_second_forward_recv_( int p, XFFTComplex3 &out )
    {
        const native_schedule_slot &recv_slot =
            native_forward_second_recv_slot_( p, "native_second_forward_recv_unpack" );
        if ( native_opt0_default_z_layout_ )
        {
            copy_second_forward_zfast_recv_to_output_async_(
                native_forward_second_device_recv_buffer_ptr_( recv_slot ),
                native_forward_second_recv_x_size_( p, recv_slot, "native_second_forward_recv_unpack" ),
                transpose1_dim_.start_x[p], out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[p].stream()
            );
        }
        else
        {
            copy_second_forward_chunk_to_xstage_async_(
                native_forward_second_device_recv_buffer_ptr_( recv_slot ),
                native_forward_second_recv_x_size_( p, recv_slot, "native_second_forward_recv_unpack" ),
                transpose1_dim_.start_x[p], out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[p].stream()
            );
        }
    }

    template <class XFFTComplex3>
    void native_unpack_second_forward_host_recv_( int p, XFFTComplex3 &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const native_schedule_slot &recv_slot =
            native_forward_second_recv_slot_( p, "native_second_forward_host_recv_unpack" );
        if ( native_opt0_default_z_layout_ )
        {
            copy_second_forward_zfast_recv_to_output_async_(
                native_forward_second_host_recv_buffer_ptr_( recv_slot ),
                native_forward_second_recv_x_size_( p, recv_slot, "native_second_forward_host_recv_unpack" ),
                transpose1_dim_.start_x[p], out.raw_ptr(), runtime_api_t::host_to_device_kind(), streams_[p].stream()
            );
        }
        else
        {
            copy_second_forward_chunk_to_xstage_async_(
                native_forward_second_host_recv_buffer_ptr_( recv_slot ),
                native_forward_second_recv_x_size_( p, recv_slot, "native_second_forward_host_recv_unpack" ),
                transpose1_dim_.start_x[p], out.raw_ptr(), runtime_api_t::host_to_device_kind(), streams_[p].stream()
            );
        }
#else
        (void)p;
        (void)out;
#endif
    }

    template <class XFastComplex3, class XFFTComplex3>
    void native_forward_second_xfast_( const XFastComplex3 &in, XFFTComplex3 &out )
    {
        auto native_phase = profile_scope_( "native/forward_second_xfast" );
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
        if ( mode_ == mpi_transpose_3d_mode::p2p_waitall && !reference_parity_forward_waitany_enabled_() )
            native_forward_second_xfast_waitall_( in, out );
        else
            native_forward_second_xfast_waitany_( in, out );
    }

    template <class XFastComplex3, class XFFTComplex3>
    void native_forward_second_xfast_waitall_( const XFastComplex3 &in, XFFTComplex3 &out )
    {
        auto native_phase = profile_scope_( "native/forward_second_waitall" );
        const int line_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        if ( reference_byte_sync_enabled_() )
        {
            auto phase = profile_scope_( "second/post_recv_send_direct_byte" );
            native_forward_second_post_sends_recvs_( in, out, byte_recv_requests, byte_send_requests );
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
            native_copy_second_forward_self_( in, out );
        }
        {
            auto phase = profile_scope_( "second/wait_recv" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( line_comm_info_, byte_recv_requests, "owned native second byte forward recv requests" );
            else
#endif
                line_comm_info_.waitall( line_size, line_recv_requests_.data() );
        }
        if ( !native_forward_second_receive_lands_in_stage_() )
        {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            auto phase = profile_scope_( "second/stage_recv_to_device" );
#else
            auto phase = profile_scope_( "second/unpack_recv" );
#endif
            for ( const int p : line_comm_order_ )
            {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                native_unpack_second_forward_host_recv_( p, out );
#else
                native_unpack_second_forward_recv_( p, out );
#endif
            }
        }
        synchronize_streams_( line_size, "second/recv_copy_complete" );
        {
            auto phase = profile_scope_( "second/wait_send" );
            wait_or_defer_send_requests_(
                deferred_send_stage::forward_second, deferred_send_communicator::line, line_comm_info_,
                byte_send_requests, persistent_second_forward_send_, line_send_requests_, line_size,
                "owned native second byte forward send requests", "owned native second persistent forward send requests"
            );
        }
    }

    template <class XFastComplex3, class XFFTComplex3>
    void native_forward_second_xfast_waitany_( const XFastComplex3 &in, XFFTComplex3 &out )
    {
        auto native_phase = profile_scope_( "native/forward_second_waitany" );
        const int line_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( line_size, 0 );
        if ( reference_byte_sync_enabled_() )
        {
            auto phase = profile_scope_( "second/post_recv_send_direct_byte" );
            native_forward_second_post_sends_recvs_(
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
            native_copy_second_forward_self_( in, out );
        }

#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const int total =
                detail::mpi_int_cast( byte_recv_requests.size(), "owned native second byte forward waitany request count" );
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
                if ( byte_recv_remaining[p] == 0 && !native_forward_second_receive_lands_in_stage_() )
                {
                    auto phase = profile_scope_( "second/unpack_recv_chunk" );
                    native_unpack_second_forward_recv_( p, out );
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
                if ( !native_forward_second_receive_lands_in_stage_() )
                {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
                    auto phase = profile_scope_( "second/stage_recv_chunk_to_device" );
                    native_unpack_second_forward_host_recv_( p, out );
#else
                    auto phase = profile_scope_( "second/unpack_recv_chunk" );
                    native_unpack_second_forward_recv_( p, out );
#endif
                }
                ++completed;
            }
        }
        synchronize_streams_( line_size, "second/recv_copy_complete" );
        {
            auto phase = profile_scope_( "second/wait_send" );
            wait_or_defer_send_requests_(
                deferred_send_stage::forward_second, deferred_send_communicator::line, line_comm_info_,
                byte_send_requests, persistent_second_forward_send_, line_send_requests_, line_size,
                "owned native second byte forward send requests", "owned native second persistent forward send requests"
            );
        }
    }

    template <class XFFTComplex3>
    void native_pack_second_backward_chunk_(
        int p, const native_schedule_slot &send_slot, const XFFTComplex3 &in
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *dst  = native_backward_second_device_send_buffer_ptr_( send_slot );
        auto        kind = runtime_api_t::device_to_device_kind();
#else
        value_type *dst  = native_backward_second_host_send_buffer_ptr_( send_slot );
        auto        kind = runtime_api_t::device_to_host_kind();
#endif
        pack_second_backward_chunk_async_(
            in.raw_ptr(), transpose1_dim_.start_x[p],
            native_backward_second_send_x_size_( p, send_slot, "native_second_backward_pack" ), dst, kind,
            streams_[p].stream()
        );
    }

    template <class XFFTComplex3, class XFastComplex3>
    void native_copy_second_backward_self_( const XFFTComplex3 &in, XFastComplex3 &out )
    {
        const native_schedule_slot *recv_slot = nullptr;
        if ( native_schedule_active_( second_backward_recv_schedule_ ) && myid_i_ >= 0 &&
             myid_i_ < static_cast<int>( second_backward_recv_schedule_.size() ) &&
             second_backward_recv_schedule_[myid_i_].valid )
        {
            recv_slot = &second_backward_recv_schedule_[myid_i_];
        }
        value_type *dst_ptr =
            recv_slot != nullptr ? native_backward_second_recv_output_ptr_( *recv_slot, out )
                                 : out.raw_ptr() + legacy_second_backward_recv_offset_( myid_i_ );
        pack_second_backward_chunk_async_(
            in.raw_ptr(), transpose1_dim_.start_x[myid_i_], nx_local_, dst_ptr,
            runtime_api_t::device_to_device_kind(), streams_[myid_i_].stream()
        );
    }

    template <class XFastComplex3>
    void native_copy_second_backward_recv_to_out_( int p, XFastComplex3 &out )
    {
        const native_schedule_slot &recv_slot =
            native_backward_second_recv_slot_( p, "native_second_backward_recv_copy_to_out" );
        runtime_api_t::memcpy_async(
            native_backward_second_recv_output_ptr_( recv_slot, out ),
            native_backward_second_device_recv_buffer_ptr_( recv_slot ), bytes_from_elems_( recv_slot.elems ),
            runtime_api_t::device_to_device_kind(), streams_[p].stream()
        );
    }

    template <class XFastComplex3>
    void native_copy_second_backward_host_recv_to_out_( int p, XFastComplex3 &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const native_schedule_slot &recv_slot =
            native_backward_second_recv_slot_( p, "native_second_backward_host_recv_copy_to_out" );
        runtime_api_t::memcpy_async(
            native_backward_second_recv_output_ptr_( recv_slot, out ),
            native_backward_second_host_recv_buffer_ptr_( recv_slot ), bytes_from_elems_( recv_slot.elems ),
            runtime_api_t::host_to_device_kind(), streams_[p].stream()
        );
#else
        (void)p;
        (void)out;
#endif
    }

    template <class XFastComplex3>
    void native_backward_second_post_recv_(
        int p, const native_schedule_slot &recv_slot, XFastComplex3 &out,
        std::vector<mpi_request_t> &byte_recv_requests, std::vector<int> *byte_recv_peers,
        std::vector<int> *byte_recv_remaining,
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state, int &persistent_byte_recv_index
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *recv_ptr = native_backward_second_recv_target_ptr_( recv_slot, out );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const std::size_t bytes = bytes_from_elems_( recv_slot.elems );
            post_byte_irecv_(
                line_comm_info_, recv_ptr, bytes, recv_slot.mpi_peer, recv_slot.mpi_tag, byte_recv_requests,
                byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state, &persistent_byte_recv_index,
                cached_large_byte_datatype_( second_backward_recv_large_byte_types_, p, bytes )
            );
        }
        else
        {
            post_value_irecv_(
                line_comm_info_, recv_ptr, recv_slot.elems, recv_slot.mpi_peer, recv_slot.mpi_tag,
                line_recv_requests_[p], persistent_second_backward_recv_, p
            );
        }
#else
        post_value_irecv_(
            line_comm_info_, native_backward_second_host_recv_buffer_ptr_( recv_slot ), recv_slot.elems,
            recv_slot.mpi_peer, recv_slot.mpi_tag, line_recv_requests_[p], persistent_second_backward_recv_, p
        );
        (void)out;
        (void)byte_recv_requests;
        (void)byte_recv_peers;
        (void)byte_recv_remaining;
        (void)persistent_byte_recv_state;
        (void)persistent_byte_recv_index;
#endif
    }

    void native_backward_second_post_send_(
        int p, const native_schedule_slot &send_slot, std::vector<mpi_request_t> &byte_send_requests,
        int &persistent_byte_index
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        const value_type *send_ptr = native_backward_second_device_send_buffer_ptr_( send_slot );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const std::size_t bytes = bytes_from_elems_( send_slot.elems );
            post_byte_isend_(
                line_comm_info_, send_ptr, bytes, send_slot.mpi_peer, send_slot.mpi_tag, byte_send_requests,
                persistent_byte_p2p_enabled_() ? &persistent_second_backward_send_ : nullptr, &persistent_byte_index,
                cached_large_byte_datatype_( second_backward_send_large_byte_types_, p, bytes )
            );
        }
        else
        {
            post_value_isend_(
                line_comm_info_, send_ptr, send_slot.elems, send_slot.mpi_peer, send_slot.mpi_tag,
                line_send_requests_[p], persistent_second_backward_send_, p
            );
        }
#else
        post_value_isend_(
            line_comm_info_, native_backward_second_host_send_buffer_ptr_( send_slot ), send_slot.elems,
            send_slot.mpi_peer, send_slot.mpi_tag, line_send_requests_[p], persistent_second_backward_send_, p
        );
        (void)byte_send_requests;
        (void)persistent_byte_index;
#endif
    }

    template <class XFFTComplex3, class XFastComplex3>
    void native_backward_second_post_recv_pack_sync_send_pairs_(
        const XFFTComplex3 &in, XFastComplex3 &out, std::vector<mpi_request_t> &byte_recv_requests,
        std::vector<mpi_request_t> &byte_send_requests, std::vector<int> *byte_recv_peers = nullptr,
        std::vector<int> *byte_recv_remaining = nullptr
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( !native_backward_second_peer_loop_active_() )
        {
            backward_second_post_recv_pack_sync_send_pairs_(
                in, out, byte_recv_requests, byte_send_requests, byte_recv_peers, byte_recv_remaining
            );
            return;
        }

        auto native_phase = profile_scope_( "native/backward_second_peer_paired_loop" );
        prepare_second_backward_send_state_();
        profile_chunk_count_(
            "second_backward", second_backward_byte_send_chunks_(), second_backward_byte_recv_chunks_()
        );
        int persistent_byte_index      = 0;
        int persistent_byte_recv_index = 0;
        detail::persistent_recv_requests<mpi_request_t> *persistent_byte_recv_state = nullptr;
        for ( const int p : line_comm_order_ )
        {
            const native_schedule_slot &recv_slot =
                native_backward_second_recv_slot_( p, "native_second_backward_recv_post" );
            const native_schedule_slot &send_slot =
                native_backward_second_send_slot_( p, "native_second_backward_send_post" );
            {
                auto phase = profile_scope_( "second_backward/mpi_post_recv" );
                native_backward_second_post_recv_(
                    p, recv_slot, out, byte_recv_requests, byte_recv_peers, byte_recv_remaining,
                    persistent_byte_recv_state, persistent_byte_recv_index
                );
            }
            {
                auto phase = profile_scope_( "second_backward/pack_peer" );
                native_pack_second_backward_chunk_( p, send_slot, in );
            }
            {
                auto phase = profile_scope_( "second_backward/pre_send_stream_sync" );
                runtime_api_t::stream_synchronize( streams_[p].stream() );
            }
            {
                auto phase = profile_scope_( "second_backward/mpi_post_send" );
                native_backward_second_post_send_( p, send_slot, byte_send_requests, persistent_byte_index );
            }
        }
#else
        backward_second_post_recv_pack_sync_send_pairs_(
            in, out, byte_recv_requests, byte_send_requests, byte_recv_peers, byte_recv_remaining
        );
#endif
    }

    template <class XFFTComplex3, class XFastComplex3>
    void native_backward_second_xfast_( const XFFTComplex3 &in, XFastComplex3 &out )
    {
        auto native_phase = profile_scope_( "native/backward_second_xfast" );
        if ( !native_backward_second_peer_loop_active_() )
        {
            backward_second_xfast_( in, out );
            return;
        }
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
        if ( mode_ == mpi_transpose_3d_mode::p2p_waitall || reference_parity_backward_waitall_enabled_() )
            native_backward_second_xfast_waitall_( in, out );
        else
            native_backward_second_xfast_waitany_( in, out );
    }

    template <class XFFTComplex3, class XFastComplex3>
    void native_backward_second_xfast_waitall_( const XFFTComplex3 &in, XFastComplex3 &out )
    {
        auto native_phase = profile_scope_( "native/backward_second_waitall" );
        const int line_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        {
            auto phase = profile_scope_( "second_backward/recv_pack_sync_send_pairs" );
            native_backward_second_post_recv_pack_sync_send_pairs_( in, out, byte_recv_requests, byte_send_requests );
        }
        {
            auto phase = profile_scope_( "second_backward/self_copy" );
            native_copy_second_backward_self_( in, out );
        }
        {
            auto phase = profile_scope_( "second_backward/wait_recv" );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( line_comm_info_, byte_recv_requests, "owned native second byte backward recv requests" );
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
                native_copy_second_backward_host_recv_to_out_( p, out );
#else
                native_copy_second_backward_recv_to_out_( p, out );
#endif
        }
        synchronize_streams_( line_size, "second_backward/recv_copy_complete" );
        {
            auto phase = profile_scope_( "second_backward/wait_send" );
            wait_or_defer_send_requests_(
                deferred_send_stage::backward_second, deferred_send_communicator::line, line_comm_info_,
                byte_send_requests, persistent_second_backward_send_, line_send_requests_, line_size,
                "owned native second byte backward send requests", "owned native second persistent backward send requests"
            );
        }
    }

    template <class XFFTComplex3, class XFastComplex3>
    void native_backward_second_xfast_waitany_( const XFFTComplex3 &in, XFastComplex3 &out )
    {
        auto native_phase = profile_scope_( "native/backward_second_waitany" );
        const int line_size = line_comm_info_.num_procs;
        std::vector<mpi_request_t> byte_recv_requests;
        std::vector<mpi_request_t> byte_send_requests;
        std::vector<int>           byte_recv_peers;
        std::vector<int>           byte_recv_remaining( line_size, 0 );
        {
            auto phase = profile_scope_( "second_backward/recv_pack_sync_send_pairs" );
            native_backward_second_post_recv_pack_sync_send_pairs_(
                in, out, byte_recv_requests, byte_send_requests, &byte_recv_peers, &byte_recv_remaining
            );
        }
        {
            auto phase = profile_scope_( "second_backward/self_copy" );
            native_copy_second_backward_self_( in, out );
        }

#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const int total =
                detail::mpi_int_cast( byte_recv_requests.size(), "owned native second byte backward waitany request count" );
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
                    native_copy_second_backward_recv_to_out_( p, out );
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
                    native_copy_second_backward_host_recv_to_out_( p, out );
#else
                    auto phase = profile_scope_( "second_backward/unpack_recv_chunk" );
                    native_copy_second_backward_recv_to_out_( p, out );
#endif
                }
                ++completed;
            }
        }
        synchronize_streams_( line_size, "second_backward/recv_copy_complete" );
        {
            auto phase = profile_scope_( "second_backward/wait_send" );
            wait_or_defer_send_requests_(
                deferred_send_stage::backward_second, deferred_send_communicator::line, line_comm_info_,
                byte_send_requests, persistent_second_backward_send_, line_send_requests_, line_size,
                "owned native second byte backward send requests", "owned native second persistent backward send requests"
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
        if ( mode_ == mpi_transpose_3d_mode::p2p_waitall || reference_parity_backward_waitall_enabled_() )
            backward_second_xfast_waitall_( in, out );
        else
            backward_second_xfast_waitany_( in, out );
    }

    template <class XFFTComplex3>
    void pack_second_backward_chunk_( int p, const XFFTComplex3 &in )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *dst  = native_backward_second_device_send_buffer_ptr_( p );
        auto        kind = runtime_api_t::device_to_device_kind();
#else
        value_type *dst  = native_backward_second_host_send_buffer_ptr_( p );
        auto        kind = runtime_api_t::device_to_host_kind();
#endif
        if ( native_opt0_default_z_layout_ )
        {
            pack_second_backward_zfast_chunk_async_(
                in.raw_ptr(), transpose1_dim_.start_x[p], transpose1_dim_.size_x[p], dst, kind, streams_[p].stream()
            );
        }
        else
        {
            pack_second_backward_chunk_async_(
                in.raw_ptr(), transpose1_dim_.start_x[p], transpose1_dim_.size_x[p], dst, kind, streams_[p].stream()
            );
        }
    }

    template <class XFFTComplex3, class XFastComplex3>
    void copy_second_backward_self_( const XFFTComplex3 &in, XFastComplex3 &out )
    {
        if ( native_opt0_default_z_layout_ )
        {
            copy_second_backward_zfast_recv_to_stage1_async_(
                in.raw_ptr() + second_backward_send_offset_( myid_i_ ), ny_output_local_, output_dim_.start_y[myid_i_],
                out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[myid_i_].stream()
            );
        }
        else
        {
            pack_second_backward_chunk_async_(
                in.raw_ptr(), transpose1_dim_.start_x[myid_i_], nx_local_,
                out.raw_ptr() + second_backward_recv_offset_( myid_i_ ), runtime_api_t::device_to_device_kind(),
                streams_[myid_i_].stream()
            );
        }
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
        if ( native_opt0_default_z_layout_ )
        {
            copy_second_backward_zfast_recv_to_stage1_async_(
                native_backward_second_device_recv_buffer_ptr_( p ), output_dim_.size_y[p], output_dim_.start_y[p],
                out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[p].stream()
            );
        }
        else
        {
            runtime_api_t::memcpy_async(
                native_backward_second_recv_output_ptr_( p, out ),
                native_backward_second_device_recv_buffer_ptr_( p ),
                bytes_from_elems_( second_backward_recv_elems_( p ) ), runtime_api_t::device_to_device_kind(),
                streams_[p].stream()
            );
        }
    }

    template <class XFastComplex3>
    void copy_second_backward_host_recv_to_out_( int p, XFastComplex3 &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( native_opt0_default_z_layout_ )
        {
            copy_second_backward_zfast_recv_to_stage1_async_(
                native_backward_second_host_recv_buffer_ptr_( p ), output_dim_.size_y[p], output_dim_.start_y[p],
                out.raw_ptr(), runtime_api_t::host_to_device_kind(), streams_[p].stream()
            );
        }
        else
        {
            runtime_api_t::memcpy_async(
                native_backward_second_recv_output_ptr_( p, out ),
                native_backward_second_host_recv_buffer_ptr_( p ),
                bytes_from_elems_( second_backward_recv_elems_( p ) ), runtime_api_t::host_to_device_kind(),
                streams_[p].stream()
            );
        }
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
            value_type *recv_ptr = native_backward_second_recv_target_ptr_( p, out );
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                const std::size_t bytes = bytes_from_elems_( second_backward_recv_elems_( p ) );
                post_byte_irecv_(
                    line_comm_info_, recv_ptr, bytes, p, p,
                    byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
                    &persistent_byte_recv_index,
                    cached_large_byte_datatype_( second_backward_recv_large_byte_types_, p, bytes )
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
                line_comm_info_, native_backward_second_host_recv_buffer_ptr_( p ),
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
        const value_type *send_ptr = native_backward_second_device_send_buffer_ptr_( p );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const std::size_t bytes = bytes_from_elems_( second_backward_send_elems_( p ) );
            post_byte_isend_(
                line_comm_info_, send_ptr, bytes, p, myid_i_,
                byte_send_requests, persistent_byte_p2p_enabled_() ? &persistent_second_backward_send_ : nullptr,
                &persistent_byte_index,
                cached_large_byte_datatype_( second_backward_send_large_byte_types_, p, bytes )
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
            line_comm_info_, native_backward_second_host_send_buffer_ptr_( p ),
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
            value_type *recv_ptr = native_backward_second_recv_target_ptr_( p, out );
            {
                auto phase = profile_scope_( "second_backward/mpi_post_recv" );
                const std::size_t bytes = bytes_from_elems_( second_backward_recv_elems_( p ) );
                post_byte_irecv_(
                    line_comm_info_, recv_ptr, bytes, p, p, byte_recv_requests, byte_recv_peers,
                    byte_recv_remaining, persistent_byte_recv_state, &persistent_byte_recv_index,
                    cached_large_byte_datatype_( second_backward_recv_large_byte_types_, p, bytes )
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
	        if ( reference_byte_sync_enabled_() && !reference_parity_backward_ready_post_send_enabled_() )
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
            wait_or_defer_send_requests_(
                deferred_send_stage::backward_second, deferred_send_communicator::line, line_comm_info_,
                byte_send_requests, persistent_second_backward_send_, line_send_requests_, line_size,
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
	        if ( reference_byte_sync_enabled_() && !reference_parity_backward_ready_post_send_enabled_() )
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
            wait_or_defer_send_requests_(
                deferred_send_stage::backward_second, deferred_send_communicator::line, line_comm_info_,
                byte_send_requests, persistent_second_backward_send_, line_send_requests_, line_size,
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
        if ( mode_ == mpi_transpose_3d_mode::p2p_waitall || reference_parity_backward_waitall_enabled_() )
            backward_first_waitall_( in, out );
        else
            backward_first_waitany_( in, out );
    }

    template <class Stage1Complex3>
    void pack_first_backward_chunk_( int p, const Stage1Complex3 &in )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        value_type *dst  = native_backward_first_device_send_buffer_ptr_( p );
        auto        kind = runtime_api_t::device_to_device_kind();
#else
        value_type *dst  = native_backward_first_host_send_buffer_ptr_( p );
        auto        kind = runtime_api_t::device_to_host_kind();
#endif
        if ( native_opt0_default_z_layout_ )
        {
            pack_first_backward_zfast_chunk_async_(
                in.raw_ptr(), half_input_dim_.start_y[p], half_input_dim_.size_y[p], dst, kind, streams_[p].stream()
            );
        }
        else
        {
            pack_first_backward_chunk_async_(
                in.raw_ptr(), half_input_dim_.start_y[p], half_input_dim_.size_y[p], dst, kind, streams_[p].stream()
            );
        }
    }

    template <class Stage1Complex3, class Stage0Complex3>
    void copy_first_backward_self_( const Stage1Complex3 &in, Stage0Complex3 &out )
    {
        if ( native_opt0_default_z_layout_ )
        {
            copy_first_backward_zfast_stage1_to_default_z_async_(
                in.raw_ptr(), half_input_dim_.start_y[myid_j_], ny_input_local_, transpose1_dim_.start_z[myid_j_],
                transpose1_dim_.size_z[myid_j_], out.raw_ptr(), runtime_api_t::device_to_device_kind(),
                streams_[myid_j_].stream()
            );
        }
        else
        {
            pack_first_backward_chunk_async_(
                in.raw_ptr(), half_input_dim_.start_y[myid_j_], ny_input_local_,
                out.raw_ptr() + first_backward_recv_offset_( myid_j_ ), runtime_api_t::device_to_device_kind(),
                streams_[myid_j_].stream()
            );
        }
    }

    template <class Stage0Complex3>
    void copy_first_backward_recv_to_out_( int p, Stage0Complex3 &out )
    {
        if ( native_opt0_default_z_layout_ )
        {
            copy_first_backward_zfast_recv_to_default_z_async_(
                native_backward_first_device_recv_buffer_ptr_( p ), transpose1_dim_.size_z[p],
                transpose1_dim_.start_z[p], out.raw_ptr(), runtime_api_t::device_to_device_kind(), streams_[p].stream()
            );
        }
        else
        {
            runtime_api_t::memcpy_async(
                native_backward_first_recv_output_ptr_( p, out ),
                native_backward_first_device_recv_buffer_ptr_( p ),
                bytes_from_elems_( first_backward_recv_elems_( p ) ), runtime_api_t::device_to_device_kind(),
                streams_[p].stream()
            );
        }
    }

    template <class Stage0Complex3>
    void copy_first_backward_host_recv_to_out_( int p, Stage0Complex3 &out )
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( native_opt0_default_z_layout_ )
        {
            copy_first_backward_zfast_recv_to_default_z_async_(
                native_backward_first_host_recv_buffer_ptr_( p ), transpose1_dim_.size_z[p],
                transpose1_dim_.start_z[p], out.raw_ptr(), runtime_api_t::host_to_device_kind(), streams_[p].stream()
            );
        }
        else
        {
            runtime_api_t::memcpy_async(
                native_backward_first_recv_output_ptr_( p, out ),
                native_backward_first_host_recv_buffer_ptr_( p ),
                bytes_from_elems_( first_backward_recv_elems_( p ) ), runtime_api_t::host_to_device_kind(),
                streams_[p].stream()
            );
        }
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
            value_type *recv_ptr = native_backward_first_recv_target_ptr_( p, out );
            if ( cuda_aware_byte_p2p_enabled_() )
            {
                const std::size_t bytes = bytes_from_elems_( first_backward_recv_elems_( p ) );
                post_byte_irecv_(
                    row_comm_info_, recv_ptr, bytes, p, first_backward_recv_mpi_tag_( p ),
                    byte_recv_requests, byte_recv_peers, byte_recv_remaining, persistent_byte_recv_state,
                    &persistent_byte_recv_index,
                    cached_large_byte_datatype_( first_backward_recv_large_byte_types_, p, bytes )
                );
            }
            else
            {
                post_value_irecv_(
                    row_comm_info_, recv_ptr, first_backward_recv_elems_( p ), p,
                    first_backward_recv_mpi_tag_( p ), row_recv_requests_[p], persistent_first_backward_recv_, p
                );
            }
#else
            post_value_irecv_(
                row_comm_info_, native_backward_first_host_recv_buffer_ptr_( p ),
                first_backward_recv_elems_( p ), p, first_backward_recv_mpi_tag_( p ), row_recv_requests_[p],
                persistent_first_backward_recv_, p
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
        const value_type *send_ptr = native_backward_first_device_send_buffer_ptr_( p );
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            const std::size_t bytes = bytes_from_elems_( first_backward_send_elems_( p ) );
            post_byte_isend_(
                row_comm_info_, send_ptr, bytes, p, first_backward_send_mpi_tag_( p ),
                byte_send_requests, persistent_byte_p2p_enabled_() ? &persistent_first_backward_send_ : nullptr,
                &persistent_byte_index,
                cached_large_byte_datatype_( first_backward_send_large_byte_types_, p, bytes )
            );
        }
        else
        {
            post_value_isend_(
                row_comm_info_, send_ptr, first_backward_send_elems_( p ), p,
                first_backward_send_mpi_tag_( p ), row_send_requests_[p], persistent_first_backward_send_, p
            );
        }
#else
        post_value_isend_(
            row_comm_info_, native_backward_first_host_send_buffer_ptr_( p ),
            first_backward_send_elems_( p ), p, first_backward_send_mpi_tag_( p ), row_send_requests_[p],
            persistent_first_backward_send_, p
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
            value_type *recv_ptr = native_backward_first_recv_target_ptr_( p, out );
            {
                auto phase = profile_scope_( "first_backward/mpi_post_recv" );
                const std::size_t bytes = bytes_from_elems_( first_backward_recv_elems_( p ) );
                post_byte_irecv_(
                    row_comm_info_, recv_ptr, bytes, p, first_backward_recv_mpi_tag_( p ), byte_recv_requests, byte_recv_peers,
                    byte_recv_remaining, persistent_byte_recv_state, &persistent_byte_recv_index,
                    cached_large_byte_datatype_( first_backward_recv_large_byte_types_, p, bytes )
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
	        if ( reference_byte_sync_enabled_() && !reference_parity_backward_ready_post_send_enabled_() )
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
            wait_or_defer_send_requests_(
                deferred_send_stage::backward_first, deferred_send_communicator::row, row_comm_info_,
                byte_send_requests, persistent_first_backward_send_, row_send_requests_, row_size,
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
	        if ( reference_byte_sync_enabled_() && !reference_parity_backward_ready_post_send_enabled_() )
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
            wait_or_defer_send_requests_(
                deferred_send_stage::backward_first, deferred_send_communicator::row, row_comm_info_,
                byte_send_requests, persistent_first_backward_send_, row_send_requests_, row_size,
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
	    bool reference_parity_enabled_        = false;
    bool use_direct_forward_byte_receive_ = false;
    bool use_stable_forward_byte_send_buffer_ = false;
    bool use_ready_stable_forward_byte_send_buffer_ = false;
    bool use_contiguous_forward_byte_send_ = false;
    bool use_physical_forward_peer_exchange_ = false;
    ::fftm::fftm_3d_contiguous_forward_send_mode contiguous_forward_send_mode_ =
        ::fftm::fftm_3d_contiguous_forward_send_mode::single;
    std::size_t contiguous_forward_send_chunk_bytes_ = static_cast<std::size_t>( 1 ) << 30;
    ::fftm::fftm_3d_large_count_p2p_transport large_count_p2p_transport_ =
        ::fftm::fftm_3d_large_count_p2p_transport::hindexed;
    bool use_large_count_datatype_cache_ = false;
    bool use_native_backward_second_peer_loop_ = false;
    bool use_deferred_send_completion_ = false;
    deferred_send_state deferred_send_state_;
    bool native_opt0_default_z_layout_ = false;
    bool use_native_opt0_reference_y_buffer_topology_ = false;
    bool use_native_opt0_compact_y_workarea_ = false;
    bool use_native_opt0_tight_y_plan_sequence_ = false;
    bool use_native_opt0_shared_y_plan_handles_ = false;
    bool use_native_opt0_y_group_device_sync_ = false;
    bool use_native_opt0_y_no_sync_exec_ = false;
    bool use_native_opt0_raw_y_plan_array_executor_ = false;
    bool use_native_opt0_reference_y_plan_lifecycle_ = false;
    bool use_native_opt0_reference_y_plan_bundle_ = false;
    bool use_native_opt0_raw_y_plan_bundle_ = false;
    bool use_native_opt0_y_plan_bundle_stream_first_ = false;
    bool use_native_opt0_raw_y_plan_bundle_reference_streams_ = false;
    bool use_native_opt0_reference_local_plan_context_ = false;
    bool local_fft_diagnostics_enabled_ = false;
    bool native_stage_timers_enabled_ = false;
    bool native_stage_timing_active_ = false;
    int  native_stage_timing_iteration_ = -1;
    bool local_fft_plan_descriptors_dumped_ = false;
    bool native_opt0_forward_y_execution_diagnostics_dumped_ = false;
    bool native_opt0_backward_y_execution_diagnostics_dumped_ = false;
    bool native_opt0_forward_y_buffer_signature_dumped_ = false;
    bool native_opt0_backward_y_buffer_signature_dumped_ = false;
    bool native_opt0_forward_layout_offsets_dumped_ = false;
    bool native_opt0_backward_layout_offsets_dumped_ = false;
    std::size_t local_fft_call_index_ = 0;
    std::vector<::fftm::fftm_native_stage_timing> native_stage_timings_;
    std::size_t native_opt0_y_parallel_plan_count_ = 0;
    plan_sequence_id_t native_opt0_forward_y_plan_sequence_ = base_fft_t::invalid_plan_sequence_id();
    plan_sequence_id_t native_opt0_inverse_y_plan_sequence_ = base_fft_t::invalid_plan_sequence_id();
    c2c_plan_array_id_t native_opt0_y_plan_array_bundle_ = base_fft_t::invalid_c2c_plan_array_id();
    std::string local_fft_diagnostics_dir_ = ".";
    std::string local_fft_diagnostics_label_ = "fftm-native";
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
    std::vector<native_schedule_slot> first_forward_send_schedule_;
    std::vector<native_schedule_slot> first_forward_recv_schedule_;
    std::vector<native_schedule_slot> first_backward_send_schedule_;
    std::vector<native_schedule_slot> first_backward_recv_schedule_;
    std::vector<native_schedule_slot> second_forward_send_schedule_;
    std::vector<native_schedule_slot> second_forward_recv_schedule_;
    std::vector<native_schedule_slot> second_backward_send_schedule_;
    std::vector<native_schedule_slot> second_backward_recv_schedule_;
    large_byte_datatype_cache first_forward_send_large_byte_types_;
    large_byte_datatype_cache first_forward_recv_large_byte_types_;
    large_byte_datatype_cache second_forward_send_large_byte_types_;
    large_byte_datatype_cache second_forward_recv_large_byte_types_;
    large_byte_datatype_cache second_backward_send_large_byte_types_;
    large_byte_datatype_cache second_backward_recv_large_byte_types_;
    large_byte_datatype_cache first_backward_send_large_byte_types_;
    large_byte_datatype_cache first_backward_recv_large_byte_types_;
    std::vector<int> row_comm_order_;
    std::vector<int> line_comm_order_;
    std::vector<typename runtime_api_t::stream_wrap> streams_;
    mpi_dtype_t mpi_value_type_;
    plan_state_t native_plan_state_;
    bool         native_plan_state_bound_ = false;
};

} // namespace detail
} // namespace fftm

#endif
