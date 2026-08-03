#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_4D_SAME_XW_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_4D_SAME_XW_H__

#include "device_aware_mpi_config.h"
#include "mpi_transpose_4d_common.h"

namespace fftm
{
namespace detail
{

template <
    class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI>
class mpi_transpose_4d_same_xw
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
    using mpi_dtype_t       = scfd::communication::detail::mpi_data_type;
    using idx_t             = scfd::static_vec::vec<int, 4>;
    using range_t           = rect_4d_t<idx_t>;
    using for_each_t        = typename backend_t::template for_each_nd_type<4, int>;
    using runtime_api_t     = RuntimeAPI;
    using profiler_t        = ::fftm::fftm_profiler;
    using profiler_scope_t  = ::fftm::profile_scope<profiler_t>;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

private:
    template <class ArrayOut>
    void forward_alltoallv_slab_native_( ArrayOut &out );
    template <class ArrayOut>
    void forward_alltoallw_slab_native_( ArrayOut &out );
    template <class ArrayOut>
    void backward_alltoallv_slab_native_( ArrayOut &out );
    template <class ArrayOut>
    void backward_alltoallw_slab_native_( ArrayOut &out );

public:
    mpi_transpose_4d_same_xw( const MPIComm &mpi, const Log &log = Log() ) : mpi_( mpi ), log_( log )
    {
        static_assert(
            std::is_same<memory_t, typename runtime_api_t::memory_type>::value,
            "mpi_transpose_4d_same_xw requires the selected runtime backend memory "
            "type"
        );
        for_each_.block_size = 128;
    }

    ~mpi_transpose_4d_same_xw()
    {
        free_value_type_();
    }

    void set_profiler( profiler_t *profiler )
    {
        profiler_ = profiler;
    }

    void set_stage_timing_callback(
        void *context, mpi_transpose_4d_stage_timer::callback_t callback, const std::string &prefix
    )
    {
        stage_timer_.set( context, callback, prefix );
    }

    void set_slab_native_batched_peer_kernels_enabled( bool enabled )
    {
        slab_native_batched_peer_kernels_enabled_ = enabled;
    }

    void set_slab_native_tensor_coalesced_kernels_enabled( bool enabled )
    {
        slab_native_tensor_coalesced_kernels_enabled_ = enabled;
    }

    void set_slab_native_vector4_kernels_enabled( bool enabled )
    {
        slab_native_vector4_kernels_enabled_ = enabled;
    }

    void set_slab_native_tiled_kernels_enabled( bool enabled )
    {
        slab_native_tiled_kernels_enabled_ = enabled;
    }

    void set_native_xzwy_direct_layout_enabled( bool enabled )
    {
        native_xzwy_direct_layout_enabled_ = enabled;
    }

    void set_native_xzwy_chunked_transport( bool enabled, std::size_t chunk_bytes, std::size_t chunk_window )
    {
        if ( enabled && chunk_bytes == 0 )
        {
            throw std::logic_error( "mpi_transpose_4d_same_xw chunked transport requires a nonzero chunk size" );
        }
        if ( !enabled && chunk_window != 0 )
        {
            throw std::logic_error( "mpi_transpose_4d_same_xw chunk window requires chunked transport" );
        }
        if ( chunk_window > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
        {
            throw std::overflow_error( "mpi_transpose_4d_same_xw chunk window exceeds int range" );
        }
        native_xzwy_chunked_transport_enabled_ = enabled;
        native_xzwy_chunk_bytes_               = chunk_bytes;
        native_xzwy_chunk_window_              = static_cast<int>( chunk_window );
    }

    void set_native_xzwy_compact_staging_enabled( bool enabled )
    {
        native_xzwy_compact_staging_enabled_ = enabled;
    }

    void set_slab_native_wz_communication_layout_enabled( bool enabled )
    {
        slab_native_wz_communication_layout_enabled_ = enabled;
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
        if ( native_xzwy_workspace_alias_enabled_() )
        {
            return bytes_from_elems_( std::max( send_buffer_elems_, recv_buffer_elems_ ) );
        }
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

    void use_external_host_work_area()
    {
        use_external_host_work_area_ = true;
    }

    std::size_t get_host_work_size_bytes() const
    {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
                "mpi_transpose_4d_same_xw::set_external_host_work_area: external host work area was not enabled."
            );
        }
        external_host_work_area_ = external_host_work_area;
        bind_external_host_work_area_();
        update_memory_profile_();
    }

    void init( const partition_t &input_dim, const partition_t &output_dim, int myid_i, int myid_j, int myid_k )
    {
        auto scope  = profile_scope_( "mpi_transpose_4d_same_xw::init" );
        free_value_type_();
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

        init_value_type_();
        init_layouts_();
        ensure_native_xzwy_chunk_window_supported_();
        ensure_native_xzwy_compact_staging_supported_();
        ensure_slab_native_wz_communication_layout_supported_();
        ensure_native_xzwy_single_message_supported_();
        configure_native_xzwy_buffer_sizes_();
        host_send_buffer_elems_ = send_buffer_elems_;
        host_recv_buffer_elems_ = recv_buffer_elems_;
        if ( !use_external_work_area_ )
        {
            send_buffer_.init( send_buffer_elems_ );
            if ( native_xzwy_workspace_alias_enabled_() )
            {
                recv_buffer_.init_by_raw_data( send_buffer_.raw_ptr(), recv_buffer_elems_ );
            }
            else
            {
                recv_buffer_.init( recv_buffer_elems_ );
            }
        }
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        if ( !use_external_host_work_area_ )
        {
            host_send_buffer_.init( host_send_buffer_elems_ );
            host_recv_buffer_.init( host_recv_buffer_elems_ );
        }
#endif
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
        stage_timing_direction_ = "forward";
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack", [&]() { pack_forward_all_( in ); }
        );

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
        stage_timing_direction_ = "backward";
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack", [&]() { pack_backward_all_( in ); }
        );

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

    template <class ArrayIn, class ArrayOut>
    void transpose_xyzw_to_yzwx_slab_native( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_xw::transpose_xyzw_to_yzwx_slab_native/" ) +
            mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_forward_slab_native_shapes_( in, out );
        reset_requests_();
        stage_timing_direction_ = "forward";
        if ( mode == mpi_transpose_3d_mode::p2p_waitany )
        {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
            forward_p2p_waitany_peer_paired_(
                out, [&]( int p ) { pack_forward_slab_native_chunk_( in, p ); }, true
            );
#else
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "pack", [&]() { pack_forward_slab_native_all_( in ); }
            );
            forward_p2p_waitany_slab_native_( out );
#endif
            return;
        }
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack", [&]() { pack_forward_slab_native_all_( in ); }
        );

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            forward_p2p_waitall_slab_native_( out );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            forward_p2p_waitany_slab_native_( out );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            forward_alltoallv_slab_native_( out );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            forward_alltoallw_slab_native_( out );
            break;
        }
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_yzwx_to_xyzw_slab_native( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_xw::transpose_yzwx_to_xyzw_slab_native/" ) +
            mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_backward_slab_native_shapes_( in, out );
        reset_requests_();
        stage_timing_direction_ = "backward";
        if ( mode == mpi_transpose_3d_mode::p2p_waitany )
        {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
            backward_p2p_waitany_peer_paired_(
                out, [&]( int p ) { pack_backward_slab_native_chunk_( in, p ); }, true
            );
#else
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "pack", [&]() { pack_backward_slab_native_all_( in ); }
            );
            backward_p2p_waitany_slab_native_( out );
#endif
            return;
        }
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack", [&]() { pack_backward_slab_native_all_( in ); }
        );

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            backward_p2p_waitall_slab_native_( out );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            backward_p2p_waitany_slab_native_( out );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            backward_alltoallv_slab_native_( out );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            backward_alltoallw_slab_native_( out );
            break;
        }
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xyzw_to_xzwy_slab_native( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_xw::transpose_xyzw_to_xzwy_slab_native/" ) +
            mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_forward_slab_native_to_xzwy_shapes_( in, out );
        reset_requests_();
        stage_timing_direction_ = "forward";
        if ( mode == mpi_transpose_3d_mode::p2p_waitany )
        {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
            if ( native_xzwy_direct_layout_enabled_ )
            {
                forward_p2p_waitany_native_xzwy_direct_(
                    out,
                    [&]( int p ) { pack_forward_slab_native_xwzy_chunk_( in, p ); },
                    [&]() {
                        pack_forward_slab_native_xwzy_chunk_to_(
                            in, myid_j_, out.raw_ptr(), forward_recv_offsets_[myid_j_]
                        );
                    },
                    [&]( int peer, int chunk, value_type *slot ) {
                        pack_forward_slab_native_xwzy_compact_chunk_( in, peer, chunk, slot );
                    }
                );
                return;
            }
            forward_p2p_waitany_peer_paired_(
                out, [&]( int p ) { pack_forward_slab_native_chunk_( in, p ); }, false
            );
#else
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "pack", [&]() { pack_forward_slab_native_all_( in ); }
            );
            forward_p2p_waitany_( out );
#endif
            return;
        }
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack", [&]() { pack_forward_slab_native_all_( in ); }
        );

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
    void transpose_xzwy_to_xyzw_slab_native( const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_xw::transpose_xzwy_to_xyzw_slab_native/" ) +
            mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_backward_xzwy_to_slab_native_shapes_( in, out );
        reset_requests_();
        stage_timing_direction_ = "backward";
        if ( mode == mpi_transpose_3d_mode::p2p_waitany )
        {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
            if ( native_xzwy_direct_layout_enabled_ )
            {
                backward_p2p_waitany_native_xzwy_direct_(
                    in, out,
                    [&]( int p ) { unpack_backward_slab_native_xwzy_chunk_( p, out ); },
                    [&]() {
                        unpack_backward_slab_native_xwzy_chunk_from_(
                            in.raw_ptr(), backward_send_offsets_[myid_j_], myid_j_, out
                        );
                    },
                    [&]( const value_type *slot, int peer, int chunk ) {
                        unpack_backward_slab_native_xwzy_compact_chunk_( slot, peer, chunk, out );
                    }
                );
                return;
            }
            backward_p2p_waitany_peer_paired_(
                out, [&]( int p ) { pack_backward_chunk_( in, p ); }, true
            );
#else
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "pack", [&]() { pack_backward_all_( in ); }
            );
            backward_p2p_waitany_slab_native_( out );
#endif
            return;
        }
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack", [&]() { pack_backward_all_( in ); }
        );

        switch ( mode )
        {
        case mpi_transpose_3d_mode::p2p_waitall:
            backward_p2p_waitall_slab_native_( out );
            break;
        case mpi_transpose_3d_mode::p2p_waitany:
            backward_p2p_waitany_slab_native_( out );
            break;
        case mpi_transpose_3d_mode::alltoallv:
            backward_alltoallv_slab_native_( out );
            break;
        case mpi_transpose_3d_mode::alltoallw:
            backward_alltoallw_slab_native_( out );
            break;
        }
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xyzw_wz_communication_to_xzwy_slab_native(
        const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode
    )
    {
        auto scope = profile_scope_(
            std::string(
                "mpi_transpose_4d_same_xw::transpose_xyzw_wz_communication_to_xzwy_slab_native/"
            ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_forward_slab_native_to_xzwy_shapes_( in, out );
        if ( !slab_native_wz_communication_layout_enabled_ )
            throw std::logic_error( "FFTM 4D slab WZ communication layout was not enabled during initialization" );
        if ( mode != mpi_transpose_3d_mode::p2p_waitany )
            throw std::logic_error( "FFTM 4D slab WZ communication layout currently requires p2p-waitany" );
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        stage_timing_direction_ = "forward";
        forward_p2p_waitany_slab_wz_communication_( in, out );
#else
        (void)in;
        (void)out;
        throw std::logic_error( "FFTM 4D slab WZ communication layout requires device-aware MPI" );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xzwy_to_xyzw_wz_communication_slab_native(
        const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode
    )
    {
        auto scope = profile_scope_(
            std::string(
                "mpi_transpose_4d_same_xw::transpose_xzwy_to_xyzw_wz_communication_slab_native/"
            ) + mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_backward_xzwy_to_slab_native_shapes_( in, out );
        if ( !slab_native_wz_communication_layout_enabled_ )
            throw std::logic_error( "FFTM 4D slab WZ communication layout was not enabled during initialization" );
        if ( mode != mpi_transpose_3d_mode::p2p_waitany )
            throw std::logic_error( "FFTM 4D slab WZ communication layout currently requires p2p-waitany" );
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        stage_timing_direction_ = "backward";
        backward_p2p_waitany_slab_wz_communication_( in, out );
#else
        (void)in;
        (void)out;
        throw std::logic_error( "FFTM 4D slab WZ communication layout requires device-aware MPI" );
#endif
    }

    template <class ArrayIn, class ArrayOut, class LaunchPlane, class LaneReady, class SynchronizePlans>
    void transpose_xyzw_wz_communication_to_xzwy_slab_native_pipelined(
        ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode, std::size_t plan_lanes,
        LaunchPlane launch_plane, LaneReady lane_ready, SynchronizePlans synchronize_plans
    )
    {
        auto scope = profile_scope_(
            "mpi_transpose_4d_same_xw::transpose_xyzw_wz_communication_to_xzwy_slab_native_pipelined"
        );
        ensure_is_inited_();
        verify_forward_slab_native_to_xzwy_shapes_( in, out );
        if ( !slab_native_wz_communication_layout_enabled_ )
            throw std::logic_error( "FFTM 4D slab WZ communication pipeline requires the WZ communication layout" );
        if ( mode != mpi_transpose_3d_mode::p2p_waitany )
            throw std::logic_error( "FFTM 4D slab WZ communication pipeline requires p2p-waitany" );
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        stage_timing_direction_ = "forward";
        forward_p2p_waitany_slab_wz_communication_pipelined_(
            in, out, plan_lanes, launch_plane, lane_ready, synchronize_plans
        );
#else
        (void)in;
        (void)out;
        (void)plan_lanes;
        (void)launch_plane;
        (void)lane_ready;
        (void)synchronize_plans;
        throw std::logic_error( "FFTM 4D slab WZ communication pipeline requires device-aware MPI" );
#endif
    }

    template <class ArrayIn, class ArrayOut, class LaunchPlane, class LaneReady, class SynchronizePlans>
    void transpose_xzwy_to_xyzw_wz_communication_slab_native_pipelined(
        const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode, std::size_t plan_lanes,
        LaunchPlane launch_plane, LaneReady lane_ready, SynchronizePlans synchronize_plans
    )
    {
        auto scope = profile_scope_(
            "mpi_transpose_4d_same_xw::transpose_xzwy_to_xyzw_wz_communication_slab_native_pipelined"
        );
        ensure_is_inited_();
        verify_backward_xzwy_to_slab_native_shapes_( in, out );
        if ( !slab_native_wz_communication_layout_enabled_ )
            throw std::logic_error( "FFTM 4D slab WZ communication pipeline requires the WZ communication layout" );
        if ( mode != mpi_transpose_3d_mode::p2p_waitany )
            throw std::logic_error( "FFTM 4D slab WZ communication pipeline requires p2p-waitany" );
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        stage_timing_direction_ = "backward";
        backward_p2p_waitany_slab_wz_communication_pipelined_(
            in, out, plan_lanes, launch_plane, lane_ready, synchronize_plans
        );
#else
        (void)in;
        (void)out;
        (void)plan_lanes;
        (void)launch_plane;
        (void)lane_ready;
        (void)synchronize_plans;
        throw std::logic_error( "FFTM 4D slab WZ communication pipeline requires device-aware MPI" );
#endif
    }

    template <class ArrayIn, class ArrayOut>
    void transpose_xywz_to_xzwy_degenerate_pencil_native(
        const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode
    )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_xw::transpose_xywz_to_xzwy_degenerate_pencil_native/" ) +
            mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_forward_shapes_( in, out );
        reset_requests_();
        stage_timing_direction_ = "forward";
        if ( mode == mpi_transpose_3d_mode::p2p_waitany )
        {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
            if ( native_xzwy_direct_layout_enabled_ )
            {
                forward_p2p_waitany_native_xzwy_direct_(
                    out,
                    [&]( int p ) { pack_forward_pencil_xwzy_chunk_( in, p ); },
                    [&]() {
                        pack_forward_pencil_xwzy_chunk_to_(
                            in, myid_j_, out.raw_ptr(), forward_recv_offsets_[myid_j_]
                        );
                    },
                    [&]( int peer, int chunk, value_type *slot ) {
                        pack_forward_pencil_xwzy_compact_chunk_( in, peer, chunk, slot );
                    }
                );
                return;
            }
            forward_p2p_waitany_peer_paired_(
                out, [&]( int p ) { pack_forward_chunk_( in, p ); }, false
            );
#else
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "pack", [&]() { pack_forward_all_( in ); }
            );
            forward_p2p_waitany_( out );
#endif
            return;
        }
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack", [&]() { pack_forward_all_( in ); }
        );

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
    void transpose_xzwy_to_xywz_degenerate_pencil_native(
        const ArrayIn &in, ArrayOut &out, mpi_transpose_3d_mode mode
    )
    {
        auto scope = profile_scope_(
            std::string( "mpi_transpose_4d_same_xw::transpose_xzwy_to_xywz_degenerate_pencil_native/" ) +
            mpi_transpose_3d_mode_name( mode )
        );
        ensure_is_inited_();
        verify_backward_shapes_( in, out );
        reset_requests_();
        stage_timing_direction_ = "backward";
        if ( mode == mpi_transpose_3d_mode::p2p_waitany )
        {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
            if ( native_xzwy_direct_layout_enabled_ )
            {
                backward_p2p_waitany_native_xzwy_direct_(
                    in, out,
                    [&]( int p ) { unpack_backward_pencil_xwzy_chunk_( p, out ); },
                    [&]() {
                        unpack_backward_pencil_xwzy_chunk_from_(
                            in.raw_ptr(), backward_send_offsets_[myid_j_], myid_j_, out
                        );
                    },
                    [&]( const value_type *slot, int peer, int chunk ) {
                        unpack_backward_pencil_xwzy_compact_chunk_( slot, peer, chunk, out );
                    }
                );
                return;
            }
            backward_p2p_waitany_peer_paired_(
                out, [&]( int p ) { pack_backward_chunk_( in, p ); }, false
            );
#else
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "pack", [&]() { pack_backward_all_( in ); }
            );
            backward_p2p_waitany_( out );
#endif
            return;
        }
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "pack", [&]() { pack_backward_all_( in ); }
        );

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

    template <class Buffer, class ArrayIn>
    struct pack_forward_xyzw_slab_functor
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
                in( idx[0], idx[3], static_cast<int>( z_offset ) + idx[1], idx[2] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_forward_yzwx_slab_functor
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
            out( static_cast<int>( y_offset ) + idx[3], idx[1], idx[2], idx[0] ) =
                buffer( offset + packed_index_4d( idx, nx, nz, nw ) );
        }
    };

    template <class Buffer, class ArrayIn>
    struct pack_backward_yzwx_slab_functor
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
                in( static_cast<int>( y_offset ) + idx[3], idx[1], idx[2], idx[0] );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_backward_xyzw_slab_functor
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
            out( idx[0], idx[3], static_cast<int>( z_offset ) + idx[1], idx[2] ) =
                buffer( offset + packed_index_4d( idx, nx, nz, nw ) );
        }
    };

    // Native xzwy tensors use physical x,w,z,y order. These functors keep the
    // MPI payload in that same order so complete Y peer slices can be received
    // into, or sent from, the tensor storage directly.
    template <class ArrayIn>
    struct pack_forward_pencil_xwzy_ptr_functor
    {
        value_type *buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t z_offset;
        std::size_t nx;
        std::size_t nw;
        std::size_t nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer[offset + packed_index_4d( idx, nx, nw, nz )] =
                in( idx[0], idx[3], idx[1], static_cast<int>( z_offset ) + idx[2] );
        }
    };

    template <class ArrayIn>
    struct pack_forward_slab_xwzy_ptr_functor
    {
        value_type *buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t z_offset;
        std::size_t nx;
        std::size_t nw;
        std::size_t nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer[offset + packed_index_4d( idx, nx, nw, nz )] =
                in( idx[0], idx[3], static_cast<int>( z_offset ) + idx[2], idx[1] );
        }
    };

    template <class ArrayOut>
    struct unpack_backward_pencil_xwzy_ptr_functor
    {
        const value_type *buffer;
        ArrayOut          out;
        std::size_t       offset;
        std::size_t       z_offset;
        std::size_t       nx;
        std::size_t       nw;
        std::size_t       nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[0], idx[3], idx[1], static_cast<int>( z_offset ) + idx[2] ) =
                buffer[offset + packed_index_4d( idx, nx, nw, nz )];
        }
    };

    template <class ArrayOut>
    struct unpack_backward_slab_xwzy_ptr_functor
    {
        const value_type *buffer;
        ArrayOut          out;
        std::size_t       offset;
        std::size_t       z_offset;
        std::size_t       nx;
        std::size_t       nw;
        std::size_t       nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[0], idx[3], static_cast<int>( z_offset ) + idx[2], idx[1] ) =
                buffer[offset + packed_index_4d( idx, nx, nw, nz )];
        }
    };

    template <class ArrayIn, class ArrayOut>
    struct copy_forward_slab_wz_communication_self_functor
    {
        ArrayIn     in;
        ArrayOut    out;
        std::size_t source_z_offset;
        std::size_t destination_y_offset;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out(
                idx[0], idx[2], idx[1], static_cast<int>( destination_y_offset ) + idx[3]
            ) = in(
                idx[0], idx[3], static_cast<int>( source_z_offset ) + idx[2], idx[1]
            );
        }
    };

    template <class ArrayIn, class ArrayOut>
    struct copy_backward_slab_wz_communication_self_functor
    {
        ArrayIn     in;
        ArrayOut    out;
        std::size_t source_y_offset;
        std::size_t destination_z_offset;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out(
                idx[0], idx[3], static_cast<int>( destination_z_offset ) + idx[2], idx[1]
            ) = in(
                idx[0], idx[2], idx[1], static_cast<int>( source_y_offset ) + idx[3]
            );
        }
    };

    template <class ArrayIn>
    struct pack_forward_pencil_xwzy_y_chunk_ptr_functor
    {
        value_type *buffer;
        ArrayIn     in;
        std::size_t z_offset;
        std::size_t y_offset;
        std::size_t nx;
        std::size_t nw;
        std::size_t nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer[packed_index_4d( idx, nx, nw, nz )] =
                in( idx[0], static_cast<int>( y_offset ) + idx[3], idx[1],
                    static_cast<int>( z_offset ) + idx[2] );
        }
    };

    template <class ArrayIn>
    struct pack_forward_slab_xwzy_y_chunk_ptr_functor
    {
        value_type *buffer;
        ArrayIn     in;
        std::size_t z_offset;
        std::size_t y_offset;
        std::size_t nx;
        std::size_t nw;
        std::size_t nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            buffer[packed_index_4d( idx, nx, nw, nz )] =
                in( idx[0], static_cast<int>( y_offset ) + idx[3],
                    static_cast<int>( z_offset ) + idx[2], idx[1] );
        }
    };

    template <class ArrayOut>
    struct unpack_backward_pencil_xwzy_y_chunk_ptr_functor
    {
        const value_type *buffer;
        ArrayOut          out;
        std::size_t       z_offset;
        std::size_t       y_offset;
        std::size_t       nx;
        std::size_t       nw;
        std::size_t       nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[0], static_cast<int>( y_offset ) + idx[3], idx[1],
                 static_cast<int>( z_offset ) + idx[2] ) = buffer[packed_index_4d( idx, nx, nw, nz )];
        }
    };

    template <class ArrayOut>
    struct unpack_backward_slab_xwzy_y_chunk_ptr_functor
    {
        const value_type *buffer;
        ArrayOut          out;
        std::size_t       z_offset;
        std::size_t       y_offset;
        std::size_t       nx;
        std::size_t       nw;
        std::size_t       nz;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            out( idx[0], static_cast<int>( y_offset ) + idx[3],
                 static_cast<int>( z_offset ) + idx[2], idx[1] ) =
                buffer[packed_index_4d( idx, nx, nw, nz )];
        }
    };

    template <class Buffer, class ArrayIn>
    struct pack_forward_xyzw_slab_tensor_coalesced_functor
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
            const int w = idx[0];
            const int z = idx[1];
            const int y = idx[2];
            const int x = idx[3];
            buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) ) =
                in( x, y, static_cast<int>( z_offset ) + z, w );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_forward_yzwx_slab_tensor_coalesced_functor
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
            const int w = idx[0];
            const int z = idx[1];
            const int y = idx[2];
            const int x = idx[3];
            out( static_cast<int>( y_offset ) + y, z, w, x ) =
                buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) );
        }
    };

    template <class Buffer, class ArrayIn>
    struct pack_backward_yzwx_slab_tensor_coalesced_functor
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
            const int w = idx[0];
            const int z = idx[1];
            const int y = idx[2];
            const int x = idx[3];
            buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) ) =
                in( static_cast<int>( y_offset ) + y, z, w, x );
        }
    };

    template <class Buffer, class ArrayOut>
    struct unpack_backward_xyzw_slab_tensor_coalesced_functor
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
            const int w = idx[0];
            const int z = idx[1];
            const int y = idx[2];
            const int x = idx[3];
            out( x, y, static_cast<int>( z_offset ) + z, w ) =
                buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) );
        }
    };

    template <class Buffer, class ArrayIn, int VectorWidth>
    struct pack_forward_xyzw_slab_vector_x_functor
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
            const int x0 = idx[0] * VectorWidth;
            const int z  = idx[1];
            const int w  = idx[2];
            const int y  = idx[3];
#pragma unroll
            for ( int lane = 0; lane < VectorWidth; ++lane )
            {
                const int x = x0 + lane;
                if ( x < static_cast<int>( nx ) )
                {
                    buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) ) =
                        in( x, y, static_cast<int>( z_offset ) + z, w );
                }
            }
        }
    };

    template <class Buffer, class ArrayOut, int VectorWidth>
    struct unpack_forward_yzwx_slab_vector_x_functor
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
            const int x0 = idx[0] * VectorWidth;
            const int z  = idx[1];
            const int w  = idx[2];
            const int y  = idx[3];
#pragma unroll
            for ( int lane = 0; lane < VectorWidth; ++lane )
            {
                const int x = x0 + lane;
                if ( x < static_cast<int>( nx ) )
                {
                    out( static_cast<int>( y_offset ) + y, z, w, x ) =
                        buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) );
                }
            }
        }
    };

    template <class Buffer, class ArrayIn, int VectorWidth>
    struct pack_backward_yzwx_slab_vector_x_functor
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
            const int x0 = idx[0] * VectorWidth;
            const int z  = idx[1];
            const int w  = idx[2];
            const int y  = idx[3];
#pragma unroll
            for ( int lane = 0; lane < VectorWidth; ++lane )
            {
                const int x = x0 + lane;
                if ( x < static_cast<int>( nx ) )
                {
                    buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) ) =
                        in( static_cast<int>( y_offset ) + y, z, w, x );
                }
            }
        }
    };

    template <class Buffer, class ArrayOut, int VectorWidth>
    struct unpack_backward_xyzw_slab_vector_x_functor
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
            const int x0 = idx[0] * VectorWidth;
            const int z  = idx[1];
            const int w  = idx[2];
            const int y  = idx[3];
#pragma unroll
            for ( int lane = 0; lane < VectorWidth; ++lane )
            {
                const int x = x0 + lane;
                if ( x < static_cast<int>( nx ) )
                {
                    out( x, y, static_cast<int>( z_offset ) + z, w ) =
                        buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) );
                }
            }
        }
    };

    template <class Buffer, class ArrayIn, int TileX, int TileY>
    struct pack_forward_xyzw_slab_tile_xy_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t z_offset;
        std::size_t nx;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            const int x0 = idx[0] * TileX;
            const int z  = idx[1];
            const int w  = idx[2];
            const int y0 = idx[3] * TileY;
#pragma unroll
            for ( int yy = 0; yy < TileY; ++yy )
            {
                const int y = y0 + yy;
                if ( y >= static_cast<int>( ny ) )
                    continue;
#pragma unroll
                for ( int xx = 0; xx < TileX; ++xx )
                {
                    const int x = x0 + xx;
                    if ( x < static_cast<int>( nx ) )
                    {
                        buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) ) =
                            in( x, y, static_cast<int>( z_offset ) + z, w );
                    }
                }
            }
        }
    };

    template <class Buffer, class ArrayOut, int TileX, int TileY>
    struct unpack_forward_yzwx_slab_tile_xy_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t y_offset;
        std::size_t nx;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            const int x0 = idx[0] * TileX;
            const int z  = idx[1];
            const int w  = idx[2];
            const int y0 = idx[3] * TileY;
#pragma unroll
            for ( int yy = 0; yy < TileY; ++yy )
            {
                const int y = y0 + yy;
                if ( y >= static_cast<int>( ny ) )
                    continue;
#pragma unroll
                for ( int xx = 0; xx < TileX; ++xx )
                {
                    const int x = x0 + xx;
                    if ( x < static_cast<int>( nx ) )
                    {
                        out( static_cast<int>( y_offset ) + y, z, w, x ) =
                            buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) );
                    }
                }
            }
        }
    };

    template <class Buffer, class ArrayIn, int TileX, int TileY>
    struct pack_backward_yzwx_slab_tile_xy_functor
    {
        Buffer      buffer;
        ArrayIn     in;
        std::size_t offset;
        std::size_t y_offset;
        std::size_t nx;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            const int x0 = idx[0] * TileX;
            const int z  = idx[1];
            const int w  = idx[2];
            const int y0 = idx[3] * TileY;
#pragma unroll
            for ( int yy = 0; yy < TileY; ++yy )
            {
                const int y = y0 + yy;
                if ( y >= static_cast<int>( ny ) )
                    continue;
#pragma unroll
                for ( int xx = 0; xx < TileX; ++xx )
                {
                    const int x = x0 + xx;
                    if ( x < static_cast<int>( nx ) )
                    {
                        buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) ) =
                            in( static_cast<int>( y_offset ) + y, z, w, x );
                    }
                }
            }
        }
    };

    template <class Buffer, class ArrayOut, int TileX, int TileY>
    struct unpack_backward_xyzw_slab_tile_xy_functor
    {
        Buffer      buffer;
        ArrayOut    out;
        std::size_t offset;
        std::size_t z_offset;
        std::size_t nx;
        std::size_t ny;
        std::size_t nz;
        std::size_t nw;

        __DEVICE_TAG__ void operator()( const idx_t &idx ) const
        {
            const int x0 = idx[0] * TileX;
            const int z  = idx[1];
            const int w  = idx[2];
            const int y0 = idx[3] * TileY;
#pragma unroll
            for ( int yy = 0; yy < TileY; ++yy )
            {
                const int y = y0 + yy;
                if ( y >= static_cast<int>( ny ) )
                    continue;
#pragma unroll
                for ( int xx = 0; xx < TileX; ++xx )
                {
                    const int x = x0 + xx;
                    if ( x < static_cast<int>( nx ) )
                    {
                        out( x, y, static_cast<int>( z_offset ) + z, w ) =
                            buffer( offset + packed_index_4d<idx_t>( x, z, w, y, nx, nz, nw ) );
                    }
                }
            }
        }
    };

private:
    profiler_scope_t profile_scope_( const std::string &name )
    {
        return profiler_scope_t( profiler_, name );
    }

    void ensure_is_inited_() const
    {
        if ( slab_native_tensor_coalesced_kernels_enabled_ && slab_native_vector4_kernels_enabled_ )
        {
            throw std::logic_error(
                "mpi_transpose_4d_same_xw: tensor-coalesced and vector4 native XW kernels are mutually exclusive"
            );
        }
        const int native_kernel_variants = ( slab_native_tensor_coalesced_kernels_enabled_ ? 1 : 0 ) +
                                           ( slab_native_vector4_kernels_enabled_ ? 1 : 0 ) +
                                           ( slab_native_tiled_kernels_enabled_ ? 1 : 0 );
        if ( native_kernel_variants > 1 )
        {
            throw std::logic_error(
                "mpi_transpose_4d_same_xw: tensor-coalesced, vector4, and tiled native XW kernels are mutually exclusive"
            );
        }
        if ( !is_inited_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw::init must be "
                                    "called before transpose" );
        if ( use_external_work_area_ && external_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_4d_same_xw: external work area was not bound." );
        }
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        if ( use_external_host_work_area_ && get_host_work_size_bytes() != 0 && external_host_work_area_ == nullptr )
        {
            throw std::logic_error( "mpi_transpose_4d_same_xw: external host work area was not bound." );
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
            use_external_work_area_ || native_xzwy_workspace_alias_enabled_()
                ? 0
                                    : static_cast<memory_profiler_t::bytes_type>( recv_buffer_.size() ) *
                                          static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_send_buffer",
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
            0
#else
            use_external_host_work_area_ ? 0
                                         : static_cast<memory_profiler_t::bytes_type>( host_send_buffer_.size() ) *
                                               static_cast<memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_recv_buffer",
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
        char *recv_raw = native_xzwy_workspace_alias_enabled_()
                             ? raw
                             : raw + bytes_from_elems_( send_buffer_elems_ );
        recv_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( recv_raw ), recv_buffer_elems_ );
    }

    void bind_external_host_work_area_()
    {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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

    template <class ArrayIn, class ArrayOut>
    void verify_forward_slab_native_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nz_global_ ||
             static_cast<std::size_t>( in_size[3] ) != nw_local_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw slab-native forward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != ny_global_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != nx_local_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw slab-native forward output shape mismatch" );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_backward_slab_native_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != ny_global_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != nx_local_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw slab-native backward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nz_global_ ||
             static_cast<std::size_t>( out_size[3] ) != nw_local_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw slab-native backward output shape mismatch" );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_forward_slab_native_to_xzwy_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nz_global_ ||
             static_cast<std::size_t>( in_size[3] ) != nw_local_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw slab-native xzwy forward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( out_size[3] ) != ny_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw slab-native xzwy forward output shape mismatch" );
    }

    template <class ArrayIn, class ArrayOut>
    void verify_backward_xzwy_to_slab_native_shapes_( const ArrayIn &in, const ArrayOut &out ) const
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        if ( static_cast<std::size_t>( in_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( in_size[1] ) != nz_local_ ||
             static_cast<std::size_t>( in_size[2] ) != nw_local_ ||
             static_cast<std::size_t>( in_size[3] ) != ny_global_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw slab-native xzwy backward input shape mismatch" );
        if ( static_cast<std::size_t>( out_size[0] ) != nx_local_ ||
             static_cast<std::size_t>( out_size[1] ) != ny_local_ ||
             static_cast<std::size_t>( out_size[2] ) != nz_global_ ||
             static_cast<std::size_t>( out_size[3] ) != nw_local_ )
            throw std::logic_error( "mpi_transpose_4d_same_xw slab-native xzwy backward output shape mismatch" );
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

    void copy_send_buffer_to_host_()
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        runtime_api_t::memcpy(
            host_send_buffer_.raw_ptr(), send_buffer_.raw_ptr(), bytes_from_elems_( send_buffer_elems_ ),
            runtime_api_t::device_to_host_kind()
        );
#endif
    }

    void copy_recv_buffer_from_host_()
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr(), host_recv_buffer_.raw_ptr(), bytes_from_elems_( recv_buffer_elems_ ),
            runtime_api_t::host_to_device_kind()
        );
#endif
    }

    void copy_host_forward_chunk_to_device_( int source_j )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + forward_recv_offsets_[source_j],
            host_recv_buffer_.raw_ptr() + forward_recv_offsets_[source_j],
            bytes_from_elems_( forward_recv_chunk_elems_( source_j ) ), runtime_api_t::host_to_device_kind()
        );
#endif
    }

    void copy_host_backward_chunk_to_device_( int source_j )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        runtime_api_t::memcpy(
            recv_buffer_.raw_ptr() + backward_recv_offsets_[source_j],
            host_recv_buffer_.raw_ptr() + backward_recv_offsets_[source_j],
            bytes_from_elems_( backward_recv_chunk_elems_( source_j ) ), runtime_api_t::host_to_device_kind()
        );
#endif
    }

    void init_value_type_()
    {
        mpi_value_type_ = scfd::communication::detail::type_contiguous(
            detail::mpi_int_cast( sizeof( value_type ), "mpi_transpose_4d_same_xw value type extent" ),
            scfd::communication::detail::mpi_data_type_trait<char>::get()
        );
        scfd::communication::detail::type_commit( mpi_value_type_ );
        mpi_value_type_inited_ = true;
    }

    void free_value_type_()
    {
        if ( mpi_value_type_inited_ )
        {
            scfd::communication::detail::type_free( mpi_value_type_ );
            mpi_value_type_inited_ = false;
        }
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
        forward_sendtypes_w_.assign( comm_size, mpi_value_type_ );
        forward_recvcounts_w_.resize( comm_size );
        forward_rdispls_w_.resize( comm_size );
        forward_recvtypes_w_.assign( comm_size, mpi_value_type_ );

        backward_send_offsets_.resize( comm_size );
        backward_recv_offsets_.resize( comm_size );
        backward_sendcounts_.resize( comm_size );
        backward_sdispls_.resize( comm_size );
        backward_recvcounts_.resize( comm_size );
        backward_rdispls_.resize( comm_size );
        backward_sendcounts_w_.resize( comm_size );
        backward_sdispls_w_.resize( comm_size );
        backward_sendtypes_w_.assign( comm_size, mpi_value_type_ );
        backward_recvcounts_w_.resize( comm_size );
        backward_rdispls_w_.resize( comm_size );
        backward_recvtypes_w_.assign( comm_size, mpi_value_type_ );

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
            forward_sendcounts_[p] =
                detail::mpi_int_cast( forward_send_chunk_elems_( p ), "same_xw forward sendcount" );
            forward_sdispls_[p] = detail::mpi_int_cast( forward_send_offsets_[p], "same_xw forward sdispl" );
            forward_recvcounts_[p] =
                detail::mpi_int_cast( forward_recv_chunk_elems_( p ), "same_xw forward recvcount" );
            forward_rdispls_[p] = detail::mpi_int_cast( forward_recv_offsets_[p], "same_xw forward rdispl" );
            forward_sendcounts_w_[p] = forward_sendcounts_[p];
            forward_recvcounts_w_[p] = forward_recvcounts_[p];
            forward_sdispls_w_[p]    = 0;
            forward_rdispls_w_[p]    = 0;

            backward_sendcounts_[p] =
                detail::mpi_int_cast( backward_send_chunk_elems_( p ), "same_xw backward sendcount" );
            backward_sdispls_[p] = detail::mpi_int_cast( backward_send_offsets_[p], "same_xw backward sdispl" );
            backward_recvcounts_[p] =
                detail::mpi_int_cast( backward_recv_chunk_elems_( p ), "same_xw backward recvcount" );
            backward_rdispls_[p] = detail::mpi_int_cast( backward_recv_offsets_[p], "same_xw backward rdispl" );
            backward_sendcounts_w_[p] = backward_sendcounts_[p];
            backward_recvcounts_w_[p] = backward_recvcounts_[p];
            backward_sdispls_w_[p]    = 0;
            backward_rdispls_w_[p]    = 0;
        }
        forward_alltoallw_layout_ready_  = false;
        backward_alltoallw_layout_ready_ = false;
    }

    void ensure_forward_alltoallw_layout_()
    {
        if ( forward_alltoallw_layout_ready_ )
            return;
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            forward_sdispls_w_[p] =
                detail::mpi_int_cast( bytes_from_elems_( forward_send_offsets_[p] ), "same_xw forward sdispl_w" );
            forward_rdispls_w_[p] =
                detail::mpi_int_cast( bytes_from_elems_( forward_recv_offsets_[p] ), "same_xw forward rdispl_w" );
        }
        forward_alltoallw_layout_ready_ = true;
    }

    void ensure_backward_alltoallw_layout_()
    {
        if ( backward_alltoallw_layout_ready_ )
            return;
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            backward_sdispls_w_[p] =
                detail::mpi_int_cast( bytes_from_elems_( backward_send_offsets_[p] ), "same_xw backward sdispl_w" );
            backward_rdispls_w_[p] =
                detail::mpi_int_cast( bytes_from_elems_( backward_recv_offsets_[p] ), "same_xw backward rdispl_w" );
        }
        backward_alltoallw_layout_ready_ = true;
    }

    template <class ArrayIn>
    void launch_pack_forward_pencil_xwzy_chunk_to_(
        const ArrayIn &in, int p, value_type *buffer, std::size_t offset
    )
    {
        for_each_(
            pack_forward_pencil_xwzy_ptr_functor<ArrayIn>{
                buffer, in, offset, output_dim_.start_z[p], nx_local_, nw_local_, output_dim_.size_z[p] },
            make_range_4d<idx_t>( nx_local_, nw_local_, output_dim_.size_z[p], ny_local_ )
        );
    }

    template <class ArrayIn>
    void pack_forward_pencil_xwzy_chunk_to_(
        const ArrayIn &in, int p, value_type *buffer, std::size_t offset
    )
    {
        auto scope = profile_scope_( "pack_forward_pencil_xwzy_chunk" );
        launch_pack_forward_pencil_xwzy_chunk_to_( in, p, buffer, offset );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_forward_pencil_xwzy_chunk_( const ArrayIn &in, int p )
    {
        pack_forward_pencil_xwzy_chunk_to_( in, p, send_buffer_.raw_ptr(), forward_send_offsets_[p] );
    }

    template <class ArrayIn>
    void pack_forward_pencil_xwzy_compact_chunk_(
        const ArrayIn &in, int peer, int chunk, value_type *slot
    )
    {
        auto scope = profile_scope_( "pack_forward_pencil_xwzy_compact_chunk" );
        const native_xzwy_compact_chunk_span span = native_xzwy_compact_chunk_span_(
            output_dim_.size_z[peer], ny_local_, chunk
        );
        for_each_(
            pack_forward_pencil_xwzy_y_chunk_ptr_functor<ArrayIn>{
                slot, in, output_dim_.start_z[peer], span.y_begin, nx_local_, nw_local_,
                output_dim_.size_z[peer] },
            make_range_4d<idx_t>(
                nx_local_, nw_local_, output_dim_.size_z[peer], span.y_count
            )
        );
        for_each_.wait();
    }

    template <class ArrayIn>
    void launch_pack_forward_slab_native_xwzy_chunk_to_(
        const ArrayIn &in, int p, value_type *buffer, std::size_t offset
    )
    {
        for_each_(
            pack_forward_slab_xwzy_ptr_functor<ArrayIn>{
                buffer, in, offset, output_dim_.start_z[p], nx_local_, nw_local_, output_dim_.size_z[p] },
            make_range_4d<idx_t>( nx_local_, nw_local_, output_dim_.size_z[p], ny_local_ )
        );
    }

    template <class ArrayIn>
    void pack_forward_slab_native_xwzy_chunk_to_(
        const ArrayIn &in, int p, value_type *buffer, std::size_t offset
    )
    {
        auto scope = profile_scope_( "pack_forward_slab_native_xwzy_chunk" );
        launch_pack_forward_slab_native_xwzy_chunk_to_( in, p, buffer, offset );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_forward_slab_native_xwzy_chunk_( const ArrayIn &in, int p )
    {
        pack_forward_slab_native_xwzy_chunk_to_( in, p, send_buffer_.raw_ptr(), forward_send_offsets_[p] );
    }

    template <class ArrayIn>
    void pack_forward_slab_native_xwzy_compact_chunk_(
        const ArrayIn &in, int peer, int chunk, value_type *slot
    )
    {
        auto scope = profile_scope_( "pack_forward_slab_native_xwzy_compact_chunk" );
        const native_xzwy_compact_chunk_span span = native_xzwy_compact_chunk_span_(
            output_dim_.size_z[peer], ny_local_, chunk
        );
        for_each_(
            pack_forward_slab_xwzy_y_chunk_ptr_functor<ArrayIn>{
                slot, in, output_dim_.start_z[peer], span.y_begin, nx_local_, nw_local_,
                output_dim_.size_z[peer] },
            make_range_4d<idx_t>(
                nx_local_, nw_local_, output_dim_.size_z[peer], span.y_count
            )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void unpack_backward_pencil_xwzy_chunk_from_(
        const value_type *buffer, std::size_t offset, int source_j, ArrayOut &out
    )
    {
        auto scope = profile_scope_( "unpack_backward_pencil_xwzy_chunk" );
        for_each_(
            unpack_backward_pencil_xwzy_ptr_functor<ArrayOut>{
                buffer, out, offset, output_dim_.start_z[source_j], nx_local_, nw_local_,
                output_dim_.size_z[source_j] },
            make_range_4d<idx_t>( nx_local_, nw_local_, output_dim_.size_z[source_j], ny_local_ )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void unpack_backward_pencil_xwzy_chunk_( int source_j, ArrayOut &out )
    {
        unpack_backward_pencil_xwzy_chunk_from_(
            recv_buffer_.raw_ptr(), backward_recv_offsets_[source_j], source_j, out
        );
    }

    template <class ArrayOut>
    void unpack_backward_pencil_xwzy_compact_chunk_(
        const value_type *slot, int source_j, int chunk, ArrayOut &out
    )
    {
        auto scope = profile_scope_( "unpack_backward_pencil_xwzy_compact_chunk" );
        const native_xzwy_compact_chunk_span span = native_xzwy_compact_chunk_span_(
            output_dim_.size_z[source_j], ny_local_, chunk
        );
        for_each_(
            unpack_backward_pencil_xwzy_y_chunk_ptr_functor<ArrayOut>{
                slot, out, output_dim_.start_z[source_j], span.y_begin, nx_local_, nw_local_,
                output_dim_.size_z[source_j] },
            make_range_4d<idx_t>(
                nx_local_, nw_local_, output_dim_.size_z[source_j], span.y_count
            )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void unpack_backward_slab_native_xwzy_chunk_from_(
        const value_type *buffer, std::size_t offset, int source_j, ArrayOut &out
    )
    {
        auto scope = profile_scope_( "unpack_backward_slab_native_xwzy_chunk" );
        for_each_(
            unpack_backward_slab_xwzy_ptr_functor<ArrayOut>{
                buffer, out, offset, output_dim_.start_z[source_j], nx_local_, nw_local_,
                output_dim_.size_z[source_j] },
            make_range_4d<idx_t>( nx_local_, nw_local_, output_dim_.size_z[source_j], ny_local_ )
        );
        for_each_.wait();
    }

    template <class ArrayOut>
    void unpack_backward_slab_native_xwzy_chunk_( int source_j, ArrayOut &out )
    {
        unpack_backward_slab_native_xwzy_chunk_from_(
            recv_buffer_.raw_ptr(), backward_recv_offsets_[source_j], source_j, out
        );
    }

    template <class ArrayOut>
    void unpack_backward_slab_native_xwzy_compact_chunk_(
        const value_type *slot, int source_j, int chunk, ArrayOut &out
    )
    {
        auto scope = profile_scope_( "unpack_backward_slab_native_xwzy_compact_chunk" );
        const native_xzwy_compact_chunk_span span = native_xzwy_compact_chunk_span_(
            output_dim_.size_z[source_j], ny_local_, chunk
        );
        for_each_(
            unpack_backward_slab_xwzy_y_chunk_ptr_functor<ArrayOut>{
                slot, out, output_dim_.start_z[source_j], span.y_begin, nx_local_, nw_local_,
                output_dim_.size_z[source_j] },
            make_range_4d<idx_t>(
                nx_local_, nw_local_, output_dim_.size_z[source_j], span.y_count
            )
        );
        for_each_.wait();
    }

    template <class ArrayIn>
    void launch_pack_forward_chunk_( const ArrayIn &in, int p )
    {
        for_each_(
            pack_forward_functor<contiguous_buf_t, ArrayIn>{
                send_buffer_, in, forward_send_offsets_[p], output_dim_.start_z[p], nx_local_,
                output_dim_.size_z[p], nw_local_ },
            make_range_4d<idx_t>( nx_local_, output_dim_.size_z[p], nw_local_, ny_local_ )
        );
    }

    template <class ArrayIn>
    void pack_forward_chunk_( const ArrayIn &in, int p )
    {
        auto scope = profile_scope_( "pack_forward_chunk" );
        launch_pack_forward_chunk_( in, p );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_forward_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_forward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            pack_forward_chunk_( in, p );
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
    void launch_pack_backward_chunk_( const ArrayIn &in, int p )
    {
        for_each_(
            pack_backward_functor<contiguous_buf_t, ArrayIn>{
                send_buffer_, in, backward_send_offsets_[p], input_dim_.start_y[p], nx_local_, nz_local_,
                nw_local_ },
            make_range_4d<idx_t>( nx_local_, nz_local_, nw_local_, input_dim_.size_y[p] )
        );
    }

    template <class ArrayIn>
    void pack_backward_chunk_( const ArrayIn &in, int p )
    {
        auto scope = profile_scope_( "pack_backward_chunk" );
        launch_pack_backward_chunk_( in, p );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_backward_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_backward_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            pack_backward_chunk_( in, p );
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

    template <class ArrayIn>
    void launch_pack_forward_slab_native_chunk_( const ArrayIn &in, int p )
    {
        constexpr int vector_width = 4;
        constexpr int tile_x       = 4;
        constexpr int tile_y       = 2;
        if ( slab_native_vector4_kernels_enabled_ )
        {
            for_each_(
                pack_forward_xyzw_slab_vector_x_functor<contiguous_buf_t, ArrayIn, vector_width>{
                    send_buffer_, in, forward_send_offsets_[p], output_dim_.start_z[p], nx_local_,
                    output_dim_.size_z[p], nw_local_ },
                make_range_4d<idx_t>(
                    ( nx_local_ + static_cast<std::size_t>( vector_width ) - 1 ) /
                        static_cast<std::size_t>( vector_width ),
                    output_dim_.size_z[p], nw_local_, ny_local_
                )
            );
        }
        else if ( slab_native_tiled_kernels_enabled_ )
        {
            for_each_(
                pack_forward_xyzw_slab_tile_xy_functor<contiguous_buf_t, ArrayIn, tile_x, tile_y>{
                    send_buffer_, in, forward_send_offsets_[p], output_dim_.start_z[p], nx_local_, ny_local_,
                    output_dim_.size_z[p], nw_local_ },
                make_range_4d<idx_t>(
                    ( nx_local_ + static_cast<std::size_t>( tile_x ) - 1 ) /
                        static_cast<std::size_t>( tile_x ),
                    output_dim_.size_z[p], nw_local_,
                    ( ny_local_ + static_cast<std::size_t>( tile_y ) - 1 ) / static_cast<std::size_t>( tile_y )
                )
            );
        }
        else if ( slab_native_tensor_coalesced_kernels_enabled_ )
        {
            for_each_(
                pack_forward_xyzw_slab_tensor_coalesced_functor<contiguous_buf_t, ArrayIn>{
                    send_buffer_, in, forward_send_offsets_[p], output_dim_.start_z[p], nx_local_,
                    output_dim_.size_z[p], nw_local_ },
                make_range_4d<idx_t>( nw_local_, output_dim_.size_z[p], ny_local_, nx_local_ )
            );
        }
        else
        {
            for_each_(
                pack_forward_xyzw_slab_functor<contiguous_buf_t, ArrayIn>{
                    send_buffer_, in, forward_send_offsets_[p], output_dim_.start_z[p], nx_local_,
                    output_dim_.size_z[p], nw_local_ },
                make_range_4d<idx_t>( nx_local_, output_dim_.size_z[p], nw_local_, ny_local_ )
            );
        }
    }

    template <class ArrayIn>
    void pack_forward_slab_native_chunk_( const ArrayIn &in, int p )
    {
        auto scope = profile_scope_( "pack_forward_slab_native_chunk" );
        launch_pack_forward_slab_native_chunk_( in, p );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_forward_slab_native_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_forward_slab_native_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            launch_pack_forward_slab_native_chunk_( in, p );
            if ( !slab_native_batched_peer_kernels_enabled_ )
                for_each_.wait();
        }
        if ( slab_native_batched_peer_kernels_enabled_ )
            for_each_.wait();
    }

    template <class ArrayOut>
    void launch_unpack_forward_slab_native_chunk_( int source_j, ArrayOut &out )
    {
        constexpr int vector_width = 4;
        constexpr int tile_x       = 4;
        constexpr int tile_y       = 2;
        if ( slab_native_vector4_kernels_enabled_ )
        {
            for_each_(
                unpack_forward_yzwx_slab_vector_x_functor<contiguous_buf_t, ArrayOut, vector_width>{
                    recv_buffer_, out, forward_recv_offsets_[source_j], input_dim_.start_y[source_j], nx_local_,
                    nz_local_, nw_local_ },
                make_range_4d<idx_t>(
                    ( nx_local_ + static_cast<std::size_t>( vector_width ) - 1 ) /
                        static_cast<std::size_t>( vector_width ),
                    nz_local_, nw_local_, input_dim_.size_y[source_j]
                )
            );
        }
        else if ( slab_native_tiled_kernels_enabled_ )
        {
            for_each_(
                unpack_forward_yzwx_slab_tile_xy_functor<contiguous_buf_t, ArrayOut, tile_x, tile_y>{
                    recv_buffer_, out, forward_recv_offsets_[source_j], input_dim_.start_y[source_j], nx_local_,
                    input_dim_.size_y[source_j], nz_local_, nw_local_ },
                make_range_4d<idx_t>(
                    ( nx_local_ + static_cast<std::size_t>( tile_x ) - 1 ) /
                        static_cast<std::size_t>( tile_x ),
                    nz_local_, nw_local_,
                    ( input_dim_.size_y[source_j] + static_cast<std::size_t>( tile_y ) - 1 ) /
                        static_cast<std::size_t>( tile_y )
                )
            );
        }
        else if ( slab_native_tensor_coalesced_kernels_enabled_ )
        {
            for_each_(
                unpack_forward_yzwx_slab_tensor_coalesced_functor<contiguous_buf_t, ArrayOut>{
                    recv_buffer_, out, forward_recv_offsets_[source_j], input_dim_.start_y[source_j], nx_local_,
                    nz_local_, nw_local_ },
                make_range_4d<idx_t>( nw_local_, nz_local_, input_dim_.size_y[source_j], nx_local_ )
            );
        }
        else
        {
            for_each_(
                unpack_forward_yzwx_slab_functor<contiguous_buf_t, ArrayOut>{
                    recv_buffer_, out, forward_recv_offsets_[source_j], input_dim_.start_y[source_j], nx_local_,
                    nz_local_, nw_local_ },
                make_range_4d<idx_t>( nx_local_, nz_local_, nw_local_, input_dim_.size_y[source_j] )
            );
        }
    }

    template <class ArrayOut>
    void unpack_forward_slab_native_chunk_( int source_j, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_forward_slab_native_chunk" );
        launch_unpack_forward_slab_native_chunk_( source_j, out );
        for_each_.wait();
    }

    template <class ArrayOut>
    void unpack_forward_slab_native_all_( ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_forward_slab_native_all" );
        if ( !slab_native_batched_peer_kernels_enabled_ )
        {
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_forward_slab_native_chunk_( p, out );
            return;
        }
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            launch_unpack_forward_slab_native_chunk_( p, out );
        for_each_.wait();
    }

    template <class ArrayIn>
    void launch_pack_backward_slab_native_chunk_( const ArrayIn &in, int p )
    {
        constexpr int vector_width = 4;
        constexpr int tile_x       = 4;
        constexpr int tile_y       = 2;
        if ( slab_native_vector4_kernels_enabled_ )
        {
            for_each_(
                pack_backward_yzwx_slab_vector_x_functor<contiguous_buf_t, ArrayIn, vector_width>{
                    send_buffer_, in, backward_send_offsets_[p], input_dim_.start_y[p], nx_local_, nz_local_,
                    nw_local_ },
                make_range_4d<idx_t>(
                    ( nx_local_ + static_cast<std::size_t>( vector_width ) - 1 ) /
                        static_cast<std::size_t>( vector_width ),
                    nz_local_, nw_local_, input_dim_.size_y[p]
                )
            );
        }
        else if ( slab_native_tiled_kernels_enabled_ )
        {
            for_each_(
                pack_backward_yzwx_slab_tile_xy_functor<contiguous_buf_t, ArrayIn, tile_x, tile_y>{
                    send_buffer_, in, backward_send_offsets_[p], input_dim_.start_y[p], nx_local_,
                    input_dim_.size_y[p], nz_local_, nw_local_ },
                make_range_4d<idx_t>(
                    ( nx_local_ + static_cast<std::size_t>( tile_x ) - 1 ) /
                        static_cast<std::size_t>( tile_x ),
                    nz_local_, nw_local_,
                    ( input_dim_.size_y[p] + static_cast<std::size_t>( tile_y ) - 1 ) /
                        static_cast<std::size_t>( tile_y )
                )
            );
        }
        else if ( slab_native_tensor_coalesced_kernels_enabled_ )
        {
            for_each_(
                pack_backward_yzwx_slab_tensor_coalesced_functor<contiguous_buf_t, ArrayIn>{
                    send_buffer_, in, backward_send_offsets_[p], input_dim_.start_y[p], nx_local_, nz_local_,
                    nw_local_ },
                make_range_4d<idx_t>( nw_local_, nz_local_, input_dim_.size_y[p], nx_local_ )
            );
        }
        else
        {
            for_each_(
                pack_backward_yzwx_slab_functor<contiguous_buf_t, ArrayIn>{
                    send_buffer_, in, backward_send_offsets_[p], input_dim_.start_y[p], nx_local_, nz_local_,
                    nw_local_ },
                make_range_4d<idx_t>( nx_local_, nz_local_, nw_local_, input_dim_.size_y[p] )
            );
        }
    }

    template <class ArrayIn>
    void pack_backward_slab_native_chunk_( const ArrayIn &in, int p )
    {
        auto scope = profile_scope_( "pack_backward_slab_native_chunk" );
        launch_pack_backward_slab_native_chunk_( in, p );
        for_each_.wait();
    }

    template <class ArrayIn>
    void pack_backward_slab_native_all_( const ArrayIn &in )
    {
        auto scope = profile_scope_( "pack_backward_slab_native_all" );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            launch_pack_backward_slab_native_chunk_( in, p );
            if ( !slab_native_batched_peer_kernels_enabled_ )
                for_each_.wait();
        }
        if ( slab_native_batched_peer_kernels_enabled_ )
            for_each_.wait();
    }

    template <class ArrayOut>
    void launch_unpack_backward_slab_native_chunk_( int source_j, ArrayOut &out )
    {
        constexpr int vector_width = 4;
        constexpr int tile_x       = 4;
        constexpr int tile_y       = 2;
        if ( slab_native_vector4_kernels_enabled_ )
        {
            for_each_(
                unpack_backward_xyzw_slab_vector_x_functor<contiguous_buf_t, ArrayOut, vector_width>{
                    recv_buffer_, out, backward_recv_offsets_[source_j], output_dim_.start_z[source_j], nx_local_,
                    output_dim_.size_z[source_j], nw_local_ },
                make_range_4d<idx_t>(
                    ( nx_local_ + static_cast<std::size_t>( vector_width ) - 1 ) /
                        static_cast<std::size_t>( vector_width ),
                    output_dim_.size_z[source_j], nw_local_, ny_local_
                )
            );
        }
        else if ( slab_native_tiled_kernels_enabled_ )
        {
            for_each_(
                unpack_backward_xyzw_slab_tile_xy_functor<contiguous_buf_t, ArrayOut, tile_x, tile_y>{
                    recv_buffer_, out, backward_recv_offsets_[source_j], output_dim_.start_z[source_j], nx_local_,
                    ny_local_, output_dim_.size_z[source_j], nw_local_ },
                make_range_4d<idx_t>(
                    ( nx_local_ + static_cast<std::size_t>( tile_x ) - 1 ) /
                        static_cast<std::size_t>( tile_x ),
                    output_dim_.size_z[source_j], nw_local_,
                    ( ny_local_ + static_cast<std::size_t>( tile_y ) - 1 ) /
                        static_cast<std::size_t>( tile_y )
                )
            );
        }
        else if ( slab_native_tensor_coalesced_kernels_enabled_ )
        {
            for_each_(
                unpack_backward_xyzw_slab_tensor_coalesced_functor<contiguous_buf_t, ArrayOut>{
                    recv_buffer_, out, backward_recv_offsets_[source_j], output_dim_.start_z[source_j], nx_local_,
                    output_dim_.size_z[source_j], nw_local_ },
                make_range_4d<idx_t>( nw_local_, output_dim_.size_z[source_j], ny_local_, nx_local_ )
            );
        }
        else
        {
            for_each_(
                unpack_backward_xyzw_slab_functor<contiguous_buf_t, ArrayOut>{
                    recv_buffer_, out, backward_recv_offsets_[source_j], output_dim_.start_z[source_j], nx_local_,
                    output_dim_.size_z[source_j], nw_local_ },
                make_range_4d<idx_t>( nx_local_, output_dim_.size_z[source_j], nw_local_, ny_local_ )
            );
        }
    }

    template <class ArrayOut>
    void unpack_backward_slab_native_chunk_( int source_j, ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_backward_slab_native_chunk" );
        launch_unpack_backward_slab_native_chunk_( source_j, out );
        for_each_.wait();
    }

    template <class ArrayOut>
    void unpack_backward_slab_native_all_( ArrayOut &out )
    {
        auto scope = profile_scope_( "unpack_backward_slab_native_all" );
        if ( !slab_native_batched_peer_kernels_enabled_ )
        {
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_backward_slab_native_chunk_( p, out );
            return;
        }
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
            launch_unpack_backward_slab_native_chunk_( p, out );
        for_each_.wait();
    }

    struct scoped_bool_flag_
    {
        bool &slot;
        bool  previous;

        scoped_bool_flag_( bool &slot_, bool value ) : slot( slot_ ), previous( slot_ )
        {
            slot = value;
        }

        ~scoped_bool_flag_()
        {
            slot = previous;
        }
    };

    template <class ArrayOut>
    void unpack_forward_p2p_chunk_( int source_j, ArrayOut &out )
    {
        if ( p2p_slab_native_unpack_ )
            unpack_forward_slab_native_chunk_( source_j, out );
        else
            unpack_forward_chunk_( source_j, out );
    }

    template <class ArrayOut>
    void unpack_backward_p2p_chunk_( int source_j, ArrayOut &out )
    {
        if ( p2p_slab_native_unpack_ )
            unpack_backward_slab_native_chunk_( source_j, out );
        else
            unpack_backward_chunk_( source_j, out );
    }

    template <class ArrayOut>
    void forward_p2p_waitall_slab_native_( ArrayOut &out )
    {
        scoped_bool_flag_ use_slab_native_unpack( p2p_slab_native_unpack_, true );
        forward_p2p_waitall_( out );
    }

    template <class ArrayOut>
    void forward_p2p_waitany_slab_native_( ArrayOut &out )
    {
        scoped_bool_flag_ use_slab_native_unpack( p2p_slab_native_unpack_, true );
        forward_p2p_waitany_( out );
    }

    template <class ArrayOut>
    void backward_p2p_waitall_slab_native_( ArrayOut &out )
    {
        scoped_bool_flag_ use_slab_native_unpack( p2p_slab_native_unpack_, true );
        backward_p2p_waitall_( out );
    }

    template <class ArrayOut>
    void backward_p2p_waitany_slab_native_( ArrayOut &out )
    {
        scoped_bool_flag_ use_slab_native_unpack( p2p_slab_native_unpack_, true );
        backward_p2p_waitany_( out );
    }

    template <class Fn>
    double time_direct_host_phase_( Fn fn ) const
    {
        if ( !stage_timer_.enabled() )
        {
            fn();
            return 0.0;
        }
        scfd::utils::system_timer_event begin, end;
        begin.record();
        fn();
        end.record();
        return end.elapsed_time( begin );
    }

    void record_peer_phase_( const char *direction, const char *route, const char *phase, double ms ) const
    {
        if ( !stage_timer_.enabled() )
            return;
        const std::string stage = std::string( direction ) + "/" + route + "_" + phase;
        stage_timer_.record( stage.c_str(), ms );
    }

    void verify_native_xzwy_direct_offsets_() const
    {
        const std::size_t y_slice_elems = nx_local_ * nw_local_ * nz_local_;
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            const std::size_t expected = input_dim_.start_y[p] * y_slice_elems;
            if ( forward_recv_offsets_[p] != expected || backward_send_offsets_[p] != expected )
            {
                throw std::logic_error(
                    "mpi_transpose_4d_same_xw native xzwy direct peer offset does not match tensor layout"
                );
            }
        }
    }

    std::size_t effective_native_xzwy_chunk_bytes_() const
    {
        const std::size_t mpi_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
        std::size_t       bytes     = std::min( native_xzwy_chunk_bytes_, mpi_limit );
        bytes -= bytes % sizeof( value_type );
        if ( bytes == 0 )
        {
            throw std::logic_error( "mpi_transpose_4d_same_xw effective chunk size is zero" );
        }
        return bytes;
    }

    std::size_t native_xzwy_peer_bytes_( std::size_t elems ) const
    {
        if ( elems > std::numeric_limits<std::size_t>::max() / sizeof( value_type ) )
        {
            throw std::overflow_error( "mpi_transpose_4d_same_xw peer byte count overflow" );
        }
        return elems * sizeof( value_type );
    }

    void ensure_native_xzwy_chunk_window_supported_() const
    {
        if ( native_xzwy_chunk_window_ == 0 )
            return;
        if ( !native_xzwy_chunked_transport_enabled_ )
        {
            throw std::logic_error( "FFTM 4D native XW chunk window requires chunked transport" );
        }

        // MPI guarantees MPI_TAG_UB is at least 32767. Keep active-slot tags
        // within that portable range without exposing a raw MPI communicator.
        const int minimum_mpi_tag_ub = 32767;
        if ( line_comm_info_.num_procs > minimum_mpi_tag_ub / native_xzwy_chunk_window_ )
        {
            throw std::logic_error(
                "FFTM 4D native XW chunk window requires more distinct MPI tags than the portable range"
            );
        }
    }

    void ensure_native_xzwy_compact_staging_supported_() const
    {
        if ( !native_xzwy_compact_staging_enabled_ )
            return;
        if ( !native_xzwy_direct_layout_enabled_ ||
             !native_xzwy_chunked_transport_enabled_ ||
             native_xzwy_chunk_window_ <= 0 )
        {
            throw std::logic_error(
                "FFTM 4D native XW compact staging requires direct layout, chunked transport, and a bounded chunk window"
            );
        }
    }

    void ensure_slab_native_wz_communication_layout_supported_() const
    {
        if ( !slab_native_wz_communication_layout_enabled_ )
            return;
        if ( !native_xzwy_direct_layout_enabled_ ||
             !native_xzwy_chunked_transport_enabled_ ||
             native_xzwy_chunk_window_ <= 0 )
        {
            throw std::logic_error(
                "FFTM 4D slab WZ communication layout requires direct layout, chunked transport, and a bounded chunk window"
            );
        }
        const std::size_t max_plane_elems = nx_local_ * nw_local_ *
                                            std::max( nz_local_, *std::max_element(
                                                output_dim_.size_z.begin(), output_dim_.size_z.end()
                                            ) );
        (void)detail::mpi_int_cast(
            native_xzwy_peer_bytes_( max_plane_elems ), "same_xw WZ communication plane bytes"
        );
    }

    void configure_native_xzwy_buffer_sizes_()
    {
        native_xzwy_compact_peer_offsets_.assign(
            static_cast<std::size_t>( line_comm_info_.num_procs ), std::numeric_limits<std::size_t>::max()
        );
        // The WZ communication path sends and receives directly from the active
        // spectral tensors. Its plane scheduler never packs through the generic
        // native-XW staging buffers, so reserving one compact slot per peer only
        // adds O(comm_size) device memory without serving an execution path.
        if ( slab_native_wz_communication_layout_enabled_ )
        {
            send_buffer_elems_ = 0;
            recv_buffer_elems_ = 0;
            return;
        }
        if ( !native_xzwy_compact_staging_enabled_ )
        {
            send_buffer_elems_ = max_buffer_elems_;
            recv_buffer_elems_ = max_buffer_elems_;
            return;
        }

        const std::size_t chunk_elems = effective_native_xzwy_chunk_bytes_() / sizeof( value_type );
        std::size_t       slot_count  = 0;
        for ( int peer = 0; peer < line_comm_info_.num_procs; ++peer )
        {
            if ( peer == myid_j_ )
                continue;
            (void)native_xzwy_compact_forward_send_chunk_count_( peer );
            (void)native_xzwy_compact_forward_recv_chunk_count_( peer );
            (void)native_xzwy_compact_backward_send_chunk_count_( peer );
            (void)native_xzwy_compact_backward_recv_chunk_count_( peer );
            if ( slot_count > std::numeric_limits<std::size_t>::max() / chunk_elems )
                throw std::overflow_error( "FFTM 4D native XW compact staging offset overflow" );
            native_xzwy_compact_peer_offsets_[static_cast<std::size_t>( peer )] = slot_count * chunk_elems;
            if ( static_cast<std::size_t>( native_xzwy_chunk_window_ ) >
                 std::numeric_limits<std::size_t>::max() - slot_count )
            {
                throw std::overflow_error( "FFTM 4D native XW compact staging slot count overflow" );
            }
            slot_count += static_cast<std::size_t>( native_xzwy_chunk_window_ );
        }
        if ( slot_count > std::numeric_limits<std::size_t>::max() / chunk_elems )
            throw std::overflow_error( "FFTM 4D native XW compact staging size overflow" );

        send_buffer_elems_ = slot_count * chunk_elems;
        recv_buffer_elems_ = send_buffer_elems_;
    }

    std::size_t native_xzwy_compact_slot_offset_( int peer, int peer_slot ) const
    {
        if ( !native_xzwy_compact_staging_enabled_ || peer < 0 || peer >= line_comm_info_.num_procs ||
             peer == myid_j_ || peer_slot < 0 || peer_slot >= native_xzwy_chunk_window_ )
        {
            throw std::logic_error( "FFTM 4D native XW compact staging slot is invalid" );
        }
        const std::size_t base = native_xzwy_compact_peer_offsets_.at( static_cast<std::size_t>( peer ) );
        const std::size_t chunk_elems = effective_native_xzwy_chunk_bytes_() / sizeof( value_type );
        return base + static_cast<std::size_t>( peer_slot ) * chunk_elems;
    }

    void ensure_native_xzwy_single_message_supported_() const
    {
        if ( !native_xzwy_direct_layout_enabled_ || native_xzwy_chunked_transport_enabled_ )
            return;

        const std::size_t byte_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
        std::size_t       local_max_peer_bytes = 0;
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            const std::size_t max_peer_bytes = std::max(
                std::max(
                    native_xzwy_peer_bytes_( forward_send_chunk_elems_( p ) ),
                    native_xzwy_peer_bytes_( forward_recv_chunk_elems_( p ) )
                ),
                std::max(
                    native_xzwy_peer_bytes_( backward_send_chunk_elems_( p ) ),
                    native_xzwy_peer_bytes_( backward_recv_chunk_elems_( p ) )
                )
            );
            local_max_peer_bytes = std::max( local_max_peer_bytes, max_peer_bytes );
        }
        const std::size_t global_max_peer_bytes = line_comm_info_.all_reduce_max( local_max_peer_bytes );
        if ( global_max_peer_bytes > byte_limit )
        {
            throw std::logic_error(
                "FFTM 4D native XW direct peer message exceeds the safe single-message byte limit; "
                "enable the chunked native XW transport"
            );
        }
    }

    bool native_xzwy_workspace_alias_enabled_() const
    {
        return native_xzwy_direct_layout_enabled_;
    }

    int native_xzwy_chunk_count_( std::size_t elems ) const
    {
        const std::size_t bytes       = native_xzwy_peer_bytes_( elems );
        const std::size_t chunk_bytes = effective_native_xzwy_chunk_bytes_();
        return detail::mpi_int_cast(
            bytes / chunk_bytes + ( bytes % chunk_bytes != 0 ? 1 : 0 ),
            "same_xw native direct chunk request count"
        );
    }

    struct native_xzwy_compact_chunk_span
    {
        std::size_t begin_elems = 0;
        std::size_t elem_count  = 0;
        std::size_t y_begin     = 0;
        std::size_t y_count     = 0;
    };

    native_xzwy_compact_chunk_span native_xzwy_compact_chunk_span_(
        std::size_t nz, std::size_t y_count, int chunk
    ) const
    {
        if ( nz == 0 || y_count == 0 || chunk < 0 )
            throw std::logic_error( "FFTM 4D native XW compact chunk dimensions are invalid" );
        if ( nx_local_ > std::numeric_limits<std::size_t>::max() / nw_local_ ||
             nx_local_ * nw_local_ > std::numeric_limits<std::size_t>::max() / nz )
        {
            throw std::overflow_error( "FFTM 4D native XW compact chunk plane size overflow" );
        }
        const std::size_t plane_elems = nx_local_ * nw_local_ * nz;
        const std::size_t limit_elems = effective_native_xzwy_chunk_bytes_() / sizeof( value_type );
        const std::size_t y_per_chunk = limit_elems / plane_elems;
        if ( y_per_chunk == 0 )
        {
            throw std::logic_error(
                "FFTM 4D native XW compact staging requires a chunk size large enough for one contiguous XWZ plane"
            );
        }
        if ( static_cast<std::size_t>( chunk ) > std::numeric_limits<std::size_t>::max() / y_per_chunk )
            throw std::overflow_error( "FFTM 4D native XW compact chunk Y offset overflow" );
        const std::size_t y_begin = static_cast<std::size_t>( chunk ) * y_per_chunk;
        if ( y_begin >= y_count )
            throw std::logic_error( "FFTM 4D native XW compact chunk index exceeds the peer payload" );
        const std::size_t this_y = std::min( y_per_chunk, y_count - y_begin );
        return native_xzwy_compact_chunk_span{
            y_begin * plane_elems, this_y * plane_elems, y_begin, this_y
        };
    }

    int native_xzwy_compact_chunk_count_( std::size_t nz, std::size_t y_count ) const
    {
        if ( nz == 0 || y_count == 0 )
            return 0;
        if ( nx_local_ > std::numeric_limits<std::size_t>::max() / nw_local_ ||
             nx_local_ * nw_local_ > std::numeric_limits<std::size_t>::max() / nz )
        {
            throw std::overflow_error( "FFTM 4D native XW compact chunk plane size overflow" );
        }
        const std::size_t plane_elems = nx_local_ * nw_local_ * nz;
        const std::size_t limit_elems = effective_native_xzwy_chunk_bytes_() / sizeof( value_type );
        const std::size_t y_per_chunk = limit_elems / plane_elems;
        if ( y_per_chunk == 0 )
        {
            throw std::logic_error(
                "FFTM 4D native XW compact staging requires a chunk size large enough for one contiguous XWZ plane"
            );
        }
        return detail::mpi_int_cast(
            y_count / y_per_chunk + ( y_count % y_per_chunk != 0 ? 1 : 0 ),
            "same_xw native compact chunk count"
        );
    }

    int native_xzwy_compact_forward_send_chunk_count_( int peer ) const
    {
        return native_xzwy_compact_chunk_count_( output_dim_.size_z[peer], ny_local_ );
    }

    int native_xzwy_compact_forward_recv_chunk_count_( int peer ) const
    {
        return native_xzwy_compact_chunk_count_( nz_local_, input_dim_.size_y[peer] );
    }

    int native_xzwy_compact_backward_send_chunk_count_( int peer ) const
    {
        return native_xzwy_compact_chunk_count_( nz_local_, input_dim_.size_y[peer] );
    }

    int native_xzwy_compact_backward_recv_chunk_count_( int peer ) const
    {
        return native_xzwy_compact_chunk_count_( output_dim_.size_z[peer], ny_local_ );
    }

    struct native_xzwy_chunk_window_state
    {
        std::vector<mpi_request_t> requests;
        std::vector<int>           slot_peers;
        std::vector<int>           slot_peer_indices;
        std::vector<int>           slot_chunks;
        std::vector<int>           chunk_counts;
        std::vector<int>           next_chunks;
        int                        completed_chunks = 0;
        int                        total_chunks     = 0;
    };

    native_xzwy_chunk_window_state make_native_xzwy_chunk_window_state_(
        const std::vector<int> &counts, int self_peer
    ) const
    {
        std::vector<int> chunk_counts( static_cast<std::size_t>( line_comm_info_.num_procs ), 0 );
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            if ( p != self_peer )
                chunk_counts[static_cast<std::size_t>( p )] = native_xzwy_chunk_count_( counts[p] );
        }
        return make_native_xzwy_chunk_window_state_from_chunk_counts_( chunk_counts, self_peer );
    }

    native_xzwy_chunk_window_state make_native_xzwy_chunk_window_state_from_chunk_counts_(
        const std::vector<int> &chunk_counts, int self_peer
    ) const
    {
        native_xzwy_chunk_window_state state;
        const int                      comm_size = line_comm_info_.num_procs;
        if ( chunk_counts.size() != static_cast<std::size_t>( comm_size ) )
            throw std::logic_error( "FFTM 4D native XW chunk-count vector has the wrong size" );
        state.chunk_counts = chunk_counts;
        state.next_chunks.assign( comm_size, 0 );
        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == self_peer )
                continue;
            const int chunks = state.chunk_counts[static_cast<std::size_t>( p )];
            state.total_chunks += chunks;
            const int slots = std::min( native_xzwy_chunk_window_, chunks );
            for ( int slot = 0; slot < slots; ++slot )
            {
                state.requests.push_back( mpi_request_t() );
                state.slot_peers.push_back( p );
                state.slot_peer_indices.push_back( slot );
                state.slot_chunks.push_back( -1 );
            }
        }
        return state;
    }

    template <class PostChunk>
    void start_native_xzwy_chunk_window_peer_(
        native_xzwy_chunk_window_state &state, int peer, PostChunk post_chunk
    ) const
    {
        for ( std::size_t slot = 0; slot < state.requests.size(); ++slot )
        {
            if ( state.slot_peers[slot] != peer || state.slot_chunks[slot] >= 0 )
                continue;
            const int chunk = state.next_chunks[peer]++;
            if ( chunk >= state.chunk_counts[peer] )
                break;
            state.slot_chunks[slot] = chunk;
            post_chunk( peer, chunk, state.requests[slot] );
        }
    }

    template <class PostChunk>
    void start_native_xzwy_compact_chunk_window_peer_(
        native_xzwy_chunk_window_state &state, int peer, PostChunk post_chunk
    ) const
    {
        for ( std::size_t slot = 0; slot < state.requests.size(); ++slot )
        {
            if ( state.slot_peers[slot] != peer || state.slot_chunks[slot] >= 0 )
                continue;
            const int chunk = state.next_chunks[peer]++;
            if ( chunk >= state.chunk_counts[peer] )
                break;
            state.slot_chunks[slot] = chunk;
            post_chunk( peer, chunk, static_cast<int>( slot ), state.requests[slot] );
        }
    }

    bool complete_native_xzwy_chunk_window_slot_(
        native_xzwy_chunk_window_state &state, int slot, int &peer, int &completed_chunk, int &next_chunk
    ) const
    {
        if ( slot < 0 || static_cast<std::size_t>( slot ) >= state.requests.size() )
        {
            throw std::logic_error( "same_xw native XW chunk-window completion index is invalid" );
        }
        peer            = state.slot_peers[static_cast<std::size_t>( slot )];
        completed_chunk = state.slot_chunks[static_cast<std::size_t>( slot )];
        if ( peer < 0 || completed_chunk < 0 )
        {
            throw std::logic_error( "same_xw native XW chunk-window completed an inactive slot" );
        }

        ++state.completed_chunks;
        if ( state.next_chunks[peer] < state.chunk_counts[peer] )
        {
            next_chunk = state.next_chunks[peer]++;
            state.slot_chunks[static_cast<std::size_t>( slot )] = next_chunk;
            return true;
        }
        next_chunk = -1;
        state.slot_chunks[static_cast<std::size_t>( slot )] = -1;
        return false;
    }

    value_type *native_xzwy_compact_slot_ptr_(
        const native_xzwy_chunk_window_state &state, int slot
    )
    {
        if ( slot < 0 || static_cast<std::size_t>( slot ) >= state.slot_peers.size() )
            throw std::logic_error( "FFTM 4D native XW compact staging request slot is invalid" );
        const int peer = state.slot_peers[static_cast<std::size_t>( slot )];
        const int peer_slot = state.slot_peer_indices[static_cast<std::size_t>( slot )];
        return send_buffer_.raw_ptr() + native_xzwy_compact_slot_offset_( peer, peer_slot );
    }

    const value_type *native_xzwy_compact_slot_ptr_(
        const native_xzwy_chunk_window_state &state, int slot
    ) const
    {
        if ( slot < 0 || static_cast<std::size_t>( slot ) >= state.slot_peers.size() )
            throw std::logic_error( "FFTM 4D native XW compact staging request slot is invalid" );
        const int peer = state.slot_peers[static_cast<std::size_t>( slot )];
        const int peer_slot = state.slot_peer_indices[static_cast<std::size_t>( slot )];
        return send_buffer_.raw_ptr() + native_xzwy_compact_slot_offset_( peer, peer_slot );
    }

    int native_xzwy_chunk_window_tag_( int base_tag, int chunk ) const
    {
        if ( native_xzwy_chunk_window_ <= 0 || chunk < 0 )
            throw std::logic_error( "same_xw native XW chunk-window tag requires an active window and chunk" );
        return base_tag + line_comm_info_.num_procs * ( chunk % native_xzwy_chunk_window_ );
    }

    std::size_t native_xzwy_chunk_offset_bytes_( int chunk ) const
    {
        const std::size_t chunk_bytes = effective_native_xzwy_chunk_bytes_();
        if ( chunk < 0 || static_cast<std::size_t>( chunk ) > std::numeric_limits<std::size_t>::max() / chunk_bytes )
            throw std::overflow_error( "same_xw native XW chunk offset overflow" );
        return static_cast<std::size_t>( chunk ) * chunk_bytes;
    }

    void post_native_xzwy_windowed_irecv_chunk_(
        void *ptr, std::size_t elems, int source, int base_tag, int chunk, mpi_request_t &request
    ) const
    {
        char             *base        = static_cast<char *>( ptr );
        const mpi_dtype_t byte_type   = scfd::communication::detail::mpi_data_type_trait<char>::get();
        const std::size_t bytes       = native_xzwy_peer_bytes_( elems );
        const std::size_t offset      = native_xzwy_chunk_offset_bytes_( chunk );
        const std::size_t chunk_bytes = effective_native_xzwy_chunk_bytes_();
        if ( offset >= bytes )
            throw std::logic_error( "same_xw native XW receive chunk offset exceeds peer payload" );
        const std::size_t this_bytes = std::min( chunk_bytes, bytes - offset );
        line_comm_info_.irecv(
            base + offset, detail::mpi_int_cast( this_bytes, "same_xw native XW window recv bytes" ), byte_type,
            source, native_xzwy_chunk_window_tag_( base_tag, chunk ), request
        );
    }

    void post_native_xzwy_windowed_isend_chunk_(
        const void *ptr, std::size_t elems, int dest, int base_tag, int chunk, mpi_request_t &request
    ) const
    {
        const char       *base        = static_cast<const char *>( ptr );
        const mpi_dtype_t byte_type   = scfd::communication::detail::mpi_data_type_trait<char>::get();
        const std::size_t bytes       = native_xzwy_peer_bytes_( elems );
        const std::size_t offset      = native_xzwy_chunk_offset_bytes_( chunk );
        const std::size_t chunk_bytes = effective_native_xzwy_chunk_bytes_();
        if ( offset >= bytes )
            throw std::logic_error( "same_xw native XW send chunk offset exceeds peer payload" );
        const std::size_t this_bytes = std::min( chunk_bytes, bytes - offset );
        line_comm_info_.isend(
            base + offset, detail::mpi_int_cast( this_bytes, "same_xw native XW window send bytes" ), byte_type,
            dest, native_xzwy_chunk_window_tag_( base_tag, chunk ), request
        );
    }

    void post_native_xzwy_compact_irecv_(
        void *ptr, const native_xzwy_compact_chunk_span &span, int source, int base_tag, int chunk,
        mpi_request_t &request
    ) const
    {
        const mpi_dtype_t byte_type = scfd::communication::detail::mpi_data_type_trait<char>::get();
        line_comm_info_.irecv(
            ptr,
            detail::mpi_int_cast(
                native_xzwy_peer_bytes_( span.elem_count ), "same_xw native XW compact recv bytes"
            ),
            byte_type, source, native_xzwy_chunk_window_tag_( base_tag, chunk ), request
        );
    }

    void post_native_xzwy_compact_isend_(
        const void *ptr, const native_xzwy_compact_chunk_span &span, int dest, int base_tag, int chunk,
        mpi_request_t &request
    ) const
    {
        const mpi_dtype_t byte_type = scfd::communication::detail::mpi_data_type_trait<char>::get();
        line_comm_info_.isend(
            ptr,
            detail::mpi_int_cast(
                native_xzwy_peer_bytes_( span.elem_count ), "same_xw native XW compact send bytes"
            ),
            byte_type, dest, native_xzwy_chunk_window_tag_( base_tag, chunk ), request
        );
    }

    void reserve_native_xzwy_chunk_requests_(
        const std::vector<int> &counts, int self_peer, std::vector<mpi_request_t> &requests
    ) const
    {
        std::size_t request_count = 0;
        for ( int p = 0; p < line_comm_info_.num_procs; ++p )
        {
            if ( p != self_peer )
                request_count += static_cast<std::size_t>( native_xzwy_chunk_count_( counts[p] ) );
        }
        requests.reserve( request_count );
    }

    void post_native_xzwy_chunked_irecv_(
        void *ptr, std::size_t elems, int source, int tag, std::vector<mpi_request_t> &requests,
        std::vector<int> *request_peers = nullptr, std::vector<int> *remaining_by_peer = nullptr
    ) const
    {
        char             *base        = static_cast<char *>( ptr );
        const mpi_dtype_t byte_type   = scfd::communication::detail::mpi_data_type_trait<char>::get();
        const std::size_t bytes       = native_xzwy_peer_bytes_( elems );
        const std::size_t chunk_bytes = effective_native_xzwy_chunk_bytes_();
        int               chunks      = 0;
        for ( std::size_t offset = 0; offset < bytes; offset += chunk_bytes )
        {
            const std::size_t this_bytes = std::min( chunk_bytes, bytes - offset );
            requests.push_back( mpi_request_t() );
            line_comm_info_.irecv(
                base + offset, detail::mpi_int_cast( this_bytes, "same_xw native direct chunk recv bytes" ),
                byte_type, source, tag, requests.back()
            );
            if ( request_peers != nullptr )
                request_peers->push_back( source );
            ++chunks;
        }
        if ( remaining_by_peer != nullptr )
            ( *remaining_by_peer )[source] += chunks;
    }

    void post_native_xzwy_chunked_isend_(
        const void *ptr, std::size_t elems, int dest, int tag, std::vector<mpi_request_t> &requests
    ) const
    {
        const char       *base        = static_cast<const char *>( ptr );
        const mpi_dtype_t byte_type   = scfd::communication::detail::mpi_data_type_trait<char>::get();
        const std::size_t bytes       = native_xzwy_peer_bytes_( elems );
        const std::size_t chunk_bytes = effective_native_xzwy_chunk_bytes_();
        for ( std::size_t offset = 0; offset < bytes; offset += chunk_bytes )
        {
            const std::size_t this_bytes = std::min( chunk_bytes, bytes - offset );
            requests.push_back( mpi_request_t() );
            line_comm_info_.isend(
                base + offset, detail::mpi_int_cast( this_bytes, "same_xw native direct chunk send bytes" ),
                byte_type, dest, tag, requests.back()
            );
        }
    }

    void waitall_native_xzwy_chunks_( std::vector<mpi_request_t> &requests, const char *what ) const
    {
        if ( requests.empty() )
            return;
        line_comm_info_.waitall( detail::mpi_int_cast( requests.size(), what ), requests.data() );
    }

    void post_slab_wz_communication_plane_irecv_(
        void *ptr, std::size_t elems, int source, int base_tag, int plane, mpi_request_t &request
    ) const
    {
        const mpi_dtype_t byte_type = scfd::communication::detail::mpi_data_type_trait<char>::get();
        line_comm_info_.irecv(
            ptr, detail::mpi_int_cast( native_xzwy_peer_bytes_( elems ), "same_xw WZ plane recv bytes" ),
            byte_type, source, native_xzwy_chunk_window_tag_( base_tag, plane ), request
        );
    }

    void post_slab_wz_communication_plane_isend_(
        const void *ptr, std::size_t elems, int dest, int base_tag, int plane, mpi_request_t &request
    ) const
    {
        const mpi_dtype_t byte_type = scfd::communication::detail::mpi_data_type_trait<char>::get();
        line_comm_info_.isend(
            ptr, detail::mpi_int_cast( native_xzwy_peer_bytes_( elems ), "same_xw WZ plane send bytes" ),
            byte_type, dest, native_xzwy_chunk_window_tag_( base_tag, plane ), request
        );
    }

    template <class RecvPtr, class RecvElems, class RecvTag, class SendPtr, class SendElems, class SendTag,
              class SelfCopy>
    void run_slab_wz_communication_plane_exchange_(
        const char *direction, const std::vector<int> &recv_plane_counts,
        const std::vector<int> &send_plane_counts, RecvPtr recv_ptr, RecvElems recv_elems, RecvTag recv_tag,
        SendPtr send_ptr, SendElems send_elems, SendTag send_tag, SelfCopy self_copy
    )
    {
        auto scope = profile_scope_( std::string( direction ) + "_p2p_waitany_slab_wz_communication" );
        const int comm_size = line_comm_info_.num_procs;
        native_xzwy_chunk_window_state recv_state =
            make_native_xzwy_chunk_window_state_from_chunk_counts_( recv_plane_counts, myid_j_ );
        native_xzwy_chunk_window_state send_state =
            make_native_xzwy_chunk_window_state_from_chunk_counts_( send_plane_counts, myid_j_ );

        double post_recv_ms = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        auto post_recv_plane = [&]( int peer, int plane, mpi_request_t &request ) {
            post_slab_wz_communication_plane_irecv_(
                recv_ptr( peer, plane ), recv_elems( peer ), peer, recv_tag( peer ), plane, request
            );
        };
        auto post_send_plane = [&]( int peer, int plane, mpi_request_t &request ) {
            post_slab_wz_communication_plane_isend_(
                send_ptr( peer, plane ), send_elems( peer ), peer, send_tag( peer ), plane, request
            );
        };

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "wz_planes_post_recv" );
            for ( int peer = 0; peer < comm_size; ++peer )
            {
                if ( peer != myid_j_ )
                    start_native_xzwy_chunk_window_peer_( recv_state, peer, post_recv_plane );
            }
        } );

        self_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "wz_planes_self" );
            self_copy();
        } );

        post_send_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "wz_planes_post_send" );
            for ( int peer = 0; peer < comm_size; ++peer )
            {
                if ( peer != myid_j_ )
                    start_native_xzwy_chunk_window_peer_( send_state, peer, post_send_plane );
            }
        } );

        auto advance_recv = [&]( int slot ) {
            int peer = -1;
            int completed_plane = -1;
            int next_plane = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                recv_state, slot, peer, completed_plane, next_plane
            );
            if ( replenish )
            {
                post_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "wz_planes_repost_recv" );
                    post_recv_plane( peer, next_plane, recv_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };
        auto advance_send = [&]( int slot ) {
            int peer = -1;
            int completed_plane = -1;
            int next_plane = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                send_state, slot, peer, completed_plane, next_plane
            );
            if ( replenish )
            {
                post_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "wz_planes_repost_send" );
                    post_send_plane( peer, next_plane, send_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };
        auto drain_recv = [&]() {
            bool progressed = false;
            while ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int ready = 0;
                int slot  = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.testany(
                        detail::mpi_int_cast( recv_state.requests.size(), "same_xw WZ plane recv test count" ),
                        recv_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_recv( slot );
                progressed = true;
            }
            return progressed;
        };
        auto drain_send = [&]() {
            bool progressed = false;
            while ( send_state.completed_chunks < send_state.total_chunks )
            {
                int ready = 0;
                int slot  = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.testany(
                        detail::mpi_int_cast( send_state.requests.size(), "same_xw WZ plane send test count" ),
                        send_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_send( slot );
                progressed = true;
            }
            return progressed;
        };

        while ( recv_state.completed_chunks < recv_state.total_chunks ||
                send_state.completed_chunks < send_state.total_chunks )
        {
            const bool recv_progress = drain_recv();
            const bool send_progress = drain_send();
            if ( recv_progress || send_progress )
                continue;
            if ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int slot = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.waitany(
                        detail::mpi_int_cast( recv_state.requests.size(), "same_xw WZ plane recv wait count" ),
                        recv_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw WZ plane receive completion ended early" );
                advance_recv( slot );
            }
            else
            {
                int slot = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.waitany(
                        detail::mpi_int_cast( send_state.requests.size(), "same_xw WZ plane send wait count" ),
                        send_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw WZ plane send completion ended early" );
                advance_send( slot );
            }
        }

        record_peer_phase_( direction, "wz_planes", "post_recv", post_recv_ms );
        record_peer_phase_( direction, "wz_planes", "pack", 0.0 );
        record_peer_phase_( direction, "wz_planes", "post_send", post_send_ms );
        record_peer_phase_( direction, "wz_planes", "wait_recv", wait_recv_ms );
        record_peer_phase_( direction, "wz_planes", "unpack", 0.0 );
        record_peer_phase_( direction, "wz_planes", "self", self_ms );
        record_peer_phase_( direction, "wz_planes", "wait_send", wait_send_ms );
    }

    template <class ArrayIn, class ArrayOut, class LaunchPlane, class LaneReady, class SynchronizePlans>
    void forward_p2p_waitany_slab_wz_communication_pipelined_(
        ArrayIn &in, ArrayOut &out, std::size_t plan_lanes, LaunchPlane launch_plane,
        LaneReady lane_ready, SynchronizePlans synchronize_plans
    )
    {
        auto scope = profile_scope_( "forward_p2p_waitany_slab_wz_communication_pipelined" );
        const int comm_size   = line_comm_info_.num_procs;
        const int plane_count = detail::mpi_int_cast( ny_local_, "same_xw WZ pipeline plane count" );
        if ( plan_lanes == 0 || plan_lanes > static_cast<std::size_t>( plane_count ) )
            throw std::logic_error( "same_xw WZ forward pipeline has an invalid plan-lane count" );

        const std::size_t plane_elems = nx_local_ * nw_local_;
        std::vector<int> recv_planes( static_cast<std::size_t>( comm_size ), 0 );
        for ( int peer = 0; peer < comm_size; ++peer )
        {
            if ( peer != myid_j_ )
                recv_planes[static_cast<std::size_t>( peer )] = detail::mpi_int_cast(
                    input_dim_.size_y[peer], "same_xw WZ forward pipeline receive plane count"
                );
        }
        native_xzwy_chunk_window_state recv_state =
            make_native_xzwy_chunk_window_state_from_chunk_counts_( recv_planes, myid_j_ );

        double post_recv_ms = 0.0;
        double plan_launch_ms = 0.0;
        double plan_poll_ms = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        auto post_recv_plane = [&]( int peer, int plane, mpi_request_t &request ) {
            post_slab_wz_communication_plane_irecv_(
                out.raw_ptr() +
                    ( input_dim_.start_y[peer] + static_cast<std::size_t>( plane ) ) * nz_local_ * plane_elems,
                nz_local_ * plane_elems, peer, peer, plane, request
            );
        };
        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "wz_pipeline_post_recv" );
            for ( int peer = 0; peer < comm_size; ++peer )
            {
                if ( peer != myid_j_ )
                    start_native_xzwy_chunk_window_peer_( recv_state, peer, post_recv_plane );
            }
        } );

        enum lane_state_t
        {
            lane_idle,
            lane_fft_active,
            lane_fft_ready
        };
        std::vector<int> lane_state( plan_lanes, lane_idle );
        std::vector<int> plane_lane( static_cast<std::size_t>( plane_count ), -1 );
        std::vector<int> plane_send_remaining( static_cast<std::size_t>( plane_count ), 0 );
        const std::size_t plane_count_size = static_cast<std::size_t>( plane_count );
        const std::size_t comm_size_count  = static_cast<std::size_t>( comm_size );
        if ( comm_size > 0 && plane_count_size > std::numeric_limits<std::size_t>::max() / comm_size_count )
            throw std::overflow_error( "same_xw WZ pipeline plane-send request count overflow" );
        std::vector<mpi_request_t> send_requests( plane_count_size * comm_size_count );
        int next_plane_to_launch = 0;
        int next_plane_to_post   = 0;
        int completed_send_planes = 0;
        int active_send_requests  = 0;

        auto launch_next_plane = [&]( std::size_t lane ) {
            if ( next_plane_to_launch >= plane_count )
            {
                lane_state[lane] = lane_idle;
                return false;
            }
            const int plane = next_plane_to_launch++;
            plan_launch_ms += time_direct_host_phase_( [&]() { launch_plane( plane, lane ); } );
            lane_state[lane] = lane_fft_active;
            plane_lane[static_cast<std::size_t>( plane )] = static_cast<int>( lane );
            return true;
        };
        for ( std::size_t lane = 0; lane < plan_lanes; ++lane )
            launch_next_plane( lane );

        auto advance_recv = [&]( int slot ) {
            int peer = -1;
            int completed_plane = -1;
            int next_plane = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                recv_state, slot, peer, completed_plane, next_plane
            );
            if ( replenish )
            {
                post_recv_ms += time_direct_host_phase_( [&]() {
                    post_recv_plane( peer, next_plane, recv_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };

        while ( recv_state.completed_chunks < recv_state.total_chunks || completed_send_planes < plane_count )
        {
            bool progressed = false;
            for ( std::size_t lane = 0; lane < plan_lanes; ++lane )
            {
                if ( lane_state[lane] != lane_fft_active )
                    continue;
                bool ready = false;
                plan_poll_ms += time_direct_host_phase_( [&]() { ready = lane_ready( lane ); } );
                if ( ready )
                {
                    lane_state[lane] = lane_fft_ready;
                    progressed = true;
                }
            }

            while ( next_plane_to_post < plane_count )
            {
                const int lane_i = plane_lane[static_cast<std::size_t>( next_plane_to_post )];
                if ( lane_i < 0 || lane_state[static_cast<std::size_t>( lane_i )] != lane_fft_ready )
                    break;
                const std::size_t lane = static_cast<std::size_t>( lane_i );
                const int plane = next_plane_to_post++;
                int remaining = 0;
                post_send_ms += time_direct_host_phase_( [&]() {
                    for ( int peer = 0; peer < comm_size; ++peer )
                    {
                        if ( peer == myid_j_ )
                            continue;
                        const std::size_t slot = static_cast<std::size_t>( plane ) *
                                                    static_cast<std::size_t>( comm_size ) +
                                                 static_cast<std::size_t>( peer );
                        post_slab_wz_communication_plane_isend_(
                            in.raw_ptr() +
                                ( static_cast<std::size_t>( plane ) * nz_global_ + output_dim_.start_z[peer] ) *
                                    plane_elems,
                            output_dim_.size_z[peer] * plane_elems, peer, myid_j_, plane,
                            send_requests[slot]
                        );
                        ++remaining;
                        ++active_send_requests;
                    }
                } );
                plane_send_remaining[static_cast<std::size_t>( plane )] = remaining;
                lane_state[lane] = lane_idle;
                if ( remaining == 0 )
                    ++completed_send_planes;
                launch_next_plane( lane );
                progressed = true;
            }

            while ( recv_state.completed_chunks < recv_state.total_chunks && !recv_state.requests.empty() )
            {
                int ready = 0;
                int slot = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.testany(
                        detail::mpi_int_cast( recv_state.requests.size(), "same_xw WZ pipeline recv test count" ),
                        recv_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_recv( slot );
                progressed = true;
            }

            while ( active_send_requests > 0 )
            {
                int ready = 0;
                int slot = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.testany(
                        detail::mpi_int_cast( send_requests.size(), "same_xw WZ pipeline send test count" ),
                        send_requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                const std::size_t plane = static_cast<std::size_t>( slot ) /
                                          static_cast<std::size_t>( comm_size );
                int &remaining = plane_send_remaining.at( plane );
                if ( remaining <= 0 )
                    throw std::logic_error( "same_xw WZ pipeline completed an inactive plane send" );
                --active_send_requests;
                if ( --remaining == 0 )
                    ++completed_send_planes;
                progressed = true;
            }

            if ( progressed )
                continue;

            const bool fft_active = std::find( lane_state.begin(), lane_state.end(), lane_fft_active ) !=
                                    lane_state.end();
            if ( fft_active )
                continue;
            if ( next_plane_to_post < plane_count )
                throw std::logic_error( "same_xw WZ pipeline stalled before posting all plane sends" );
            if ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int slot = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.waitany(
                        detail::mpi_int_cast( recv_state.requests.size(), "same_xw WZ pipeline recv wait count" ),
                        recv_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw WZ pipeline receive completion ended early" );
                advance_recv( slot );
                continue;
            }
            if ( active_send_requests > 0 )
            {
                int slot = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.waitany(
                        detail::mpi_int_cast( send_requests.size(), "same_xw WZ pipeline send wait count" ),
                        send_requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw WZ pipeline send completion ended early" );
                const std::size_t plane = static_cast<std::size_t>( slot ) /
                                          static_cast<std::size_t>( comm_size );
                int &remaining = plane_send_remaining.at( plane );
                if ( remaining <= 0 )
                    throw std::logic_error( "same_xw WZ pipeline waited on an inactive plane send" );
                --active_send_requests;
                if ( --remaining == 0 )
                    ++completed_send_planes;
            }
        }

        if ( next_plane_to_launch != plane_count || next_plane_to_post != plane_count || active_send_requests != 0 )
            throw std::logic_error( "same_xw WZ pipeline finished with incomplete forward scheduler state" );

        synchronize_plans();
        self_ms += time_direct_host_phase_( [&]() {
            for_each_(
                copy_forward_slab_wz_communication_self_functor<ArrayIn, ArrayOut>{
                    in, out, output_dim_.start_z[myid_j_], input_dim_.start_y[myid_j_] },
                make_range_4d<idx_t>( nx_local_, nw_local_, nz_local_, ny_local_ )
            );
            for_each_.wait();
        } );

        record_peer_phase_( "forward", "wz_pipeline", "post_recv", post_recv_ms );
        record_peer_phase_( "forward", "wz_pipeline", "plan_launch", plan_launch_ms );
        record_peer_phase_( "forward", "wz_pipeline", "plan_poll", plan_poll_ms );
        record_peer_phase_( "forward", "wz_pipeline", "post_send", post_send_ms );
        record_peer_phase_( "forward", "wz_pipeline", "wait_recv", wait_recv_ms );
        record_peer_phase_( "forward", "wz_pipeline", "self", self_ms );
        record_peer_phase_( "forward", "wz_pipeline", "wait_send", wait_send_ms );
    }

    template <class ArrayIn, class ArrayOut, class LaunchPlane, class LaneReady, class SynchronizePlans>
    void backward_p2p_waitany_slab_wz_communication_pipelined_(
        const ArrayIn &in, ArrayOut &out, std::size_t plan_lanes, LaunchPlane launch_plane,
        LaneReady lane_ready, SynchronizePlans synchronize_plans
    )
    {
        auto scope = profile_scope_( "backward_p2p_waitany_slab_wz_communication_pipelined" );
        const int comm_size   = line_comm_info_.num_procs;
        const int plane_count = detail::mpi_int_cast( ny_local_, "same_xw inverse WZ pipeline plane count" );
        if ( plan_lanes == 0 || plan_lanes > static_cast<std::size_t>( plane_count ) )
            throw std::logic_error( "same_xw WZ backward pipeline has an invalid plan-lane count" );

        const std::size_t plane_elems = nx_local_ * nw_local_;
        std::vector<int> recv_planes( static_cast<std::size_t>( comm_size ), 0 );
        std::vector<int> send_planes( static_cast<std::size_t>( comm_size ), 0 );
        for ( int peer = 0; peer < comm_size; ++peer )
        {
            if ( peer == myid_j_ )
                continue;
            recv_planes[static_cast<std::size_t>( peer )] = plane_count;
            send_planes[static_cast<std::size_t>( peer )] = detail::mpi_int_cast(
                input_dim_.size_y[peer], "same_xw inverse WZ pipeline send plane count"
            );
        }
        native_xzwy_chunk_window_state recv_state =
            make_native_xzwy_chunk_window_state_from_chunk_counts_( recv_planes, myid_j_ );
        native_xzwy_chunk_window_state send_state =
            make_native_xzwy_chunk_window_state_from_chunk_counts_( send_planes, myid_j_ );

        double post_recv_ms = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double wait_send_ms = 0.0;
        double self_ms      = 0.0;
        double plan_launch_ms = 0.0;
        double plan_poll_ms = 0.0;

        auto post_recv_plane = [&]( int peer, int plane, mpi_request_t &request ) {
            post_slab_wz_communication_plane_irecv_(
                out.raw_ptr() +
                    ( static_cast<std::size_t>( plane ) * nz_global_ + output_dim_.start_z[peer] ) * plane_elems,
                output_dim_.size_z[peer] * plane_elems, peer, myid_j_, plane, request
            );
        };
        auto post_send_plane = [&]( int peer, int plane, mpi_request_t &request ) {
            post_slab_wz_communication_plane_isend_(
                in.raw_ptr() +
                    ( input_dim_.start_y[peer] + static_cast<std::size_t>( plane ) ) * nz_local_ * plane_elems,
                nz_local_ * plane_elems, peer, peer, plane, request
            );
        };
        post_recv_ms += time_direct_host_phase_( [&]() {
            for ( int peer = 0; peer < comm_size; ++peer )
            {
                if ( peer != myid_j_ )
                    start_native_xzwy_chunk_window_peer_( recv_state, peer, post_recv_plane );
            }
        } );
        post_send_ms += time_direct_host_phase_( [&]() {
            for ( int peer = 0; peer < comm_size; ++peer )
            {
                if ( peer != myid_j_ )
                    start_native_xzwy_chunk_window_peer_( send_state, peer, post_send_plane );
            }
        } );
        self_ms += time_direct_host_phase_( [&]() {
            for_each_(
                copy_backward_slab_wz_communication_self_functor<ArrayIn, ArrayOut>{
                    in, out, input_dim_.start_y[myid_j_], output_dim_.start_z[myid_j_] },
                make_range_4d<idx_t>( nx_local_, nw_local_, nz_local_, ny_local_ )
            );
            for_each_.wait();
        } );

        std::vector<int> remaining_parts( static_cast<std::size_t>( plane_count ), comm_size - 1 );
        std::vector<int> ready_planes;
        ready_planes.reserve( static_cast<std::size_t>( plane_count ) );
        if ( comm_size == 1 )
        {
            for ( int plane = 0; plane < plane_count; ++plane )
                ready_planes.push_back( plane );
        }
        std::size_t ready_head = 0;
        std::vector<int> lane_plane( plan_lanes, -1 );
        int completed_plan_count = 0;

        auto launch_ready_planes = [&]() {
            bool launched = false;
            for ( std::size_t lane = 0; lane < plan_lanes && ready_head < ready_planes.size(); ++lane )
            {
                if ( lane_plane[lane] >= 0 )
                    continue;
                const int plane = ready_planes[ready_head++];
                plan_launch_ms += time_direct_host_phase_( [&]() { launch_plane( plane, lane ); } );
                lane_plane[lane] = plane;
                launched = true;
            }
            return launched;
        };
        launch_ready_planes();

        auto complete_recv = [&]( int slot ) {
            int peer = -1;
            int completed_plane = -1;
            int next_plane = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                recv_state, slot, peer, completed_plane, next_plane
            );
            int &remaining = remaining_parts.at( static_cast<std::size_t>( completed_plane ) );
            if ( --remaining == 0 )
                ready_planes.push_back( completed_plane );
            if ( replenish )
            {
                post_recv_ms += time_direct_host_phase_( [&]() {
                    post_recv_plane( peer, next_plane, recv_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };
        auto complete_send = [&]( int slot ) {
            int peer = -1;
            int completed_plane = -1;
            int next_plane = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                send_state, slot, peer, completed_plane, next_plane
            );
            if ( replenish )
            {
                post_send_ms += time_direct_host_phase_( [&]() {
                    post_send_plane( peer, next_plane, send_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };

        while ( recv_state.completed_chunks < recv_state.total_chunks ||
                send_state.completed_chunks < send_state.total_chunks || completed_plan_count < plane_count )
        {
            bool progressed = launch_ready_planes();
            for ( std::size_t lane = 0; lane < plan_lanes; ++lane )
            {
                if ( lane_plane[lane] < 0 )
                    continue;
                bool ready = false;
                plan_poll_ms += time_direct_host_phase_( [&]() { ready = lane_ready( lane ); } );
                if ( ready )
                {
                    lane_plane[lane] = -1;
                    ++completed_plan_count;
                    progressed = true;
                }
            }
            progressed = launch_ready_planes() || progressed;

            while ( recv_state.completed_chunks < recv_state.total_chunks && !recv_state.requests.empty() )
            {
                int ready = 0;
                int slot = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.testany(
                        detail::mpi_int_cast( recv_state.requests.size(), "same_xw inverse WZ pipeline recv test count" ),
                        recv_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                complete_recv( slot );
                progressed = true;
            }
            while ( send_state.completed_chunks < send_state.total_chunks && !send_state.requests.empty() )
            {
                int ready = 0;
                int slot = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.testany(
                        detail::mpi_int_cast( send_state.requests.size(), "same_xw inverse WZ pipeline send test count" ),
                        send_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                complete_send( slot );
                progressed = true;
            }

            if ( progressed )
                continue;
            const bool fft_active = std::find_if(
                lane_plane.begin(), lane_plane.end(), []( int plane ) { return plane >= 0; }
            ) != lane_plane.end();
            if ( fft_active )
                continue;
            if ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int slot = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.waitany(
                        detail::mpi_int_cast( recv_state.requests.size(), "same_xw inverse WZ pipeline recv wait count" ),
                        recv_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw inverse WZ pipeline receive completion ended early" );
                complete_recv( slot );
                continue;
            }
            if ( send_state.completed_chunks < send_state.total_chunks )
            {
                int slot = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    slot = line_comm_info_.waitany(
                        detail::mpi_int_cast( send_state.requests.size(), "same_xw inverse WZ pipeline send wait count" ),
                        send_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw inverse WZ pipeline send completion ended early" );
                complete_send( slot );
            }
        }

        synchronize_plans();
        record_peer_phase_( "backward", "wz_pipeline", "post_recv", post_recv_ms );
        record_peer_phase_( "backward", "wz_pipeline", "post_send", post_send_ms );
        record_peer_phase_( "backward", "wz_pipeline", "wait_recv", wait_recv_ms );
        record_peer_phase_( "backward", "wz_pipeline", "wait_send", wait_send_ms );
        record_peer_phase_( "backward", "wz_pipeline", "self", self_ms );
        record_peer_phase_( "backward", "wz_pipeline", "plan_launch", plan_launch_ms );
        record_peer_phase_( "backward", "wz_pipeline", "plan_poll", plan_poll_ms );
    }

    template <class ArrayIn, class ArrayOut>
    void forward_p2p_waitany_slab_wz_communication_( const ArrayIn &in, ArrayOut &out )
    {
        const int         comm_size  = line_comm_info_.num_procs;
        const std::size_t plane_elems = nx_local_ * nw_local_;
        std::vector<int> recv_planes( static_cast<std::size_t>( comm_size ), 0 );
        std::vector<int> send_planes( static_cast<std::size_t>( comm_size ), 0 );
        for ( int peer = 0; peer < comm_size; ++peer )
        {
            if ( peer == myid_j_ )
                continue;
            recv_planes[static_cast<std::size_t>( peer )] = detail::mpi_int_cast(
                input_dim_.size_y[peer], "same_xw WZ forward receive plane count"
            );
            send_planes[static_cast<std::size_t>( peer )] =
                detail::mpi_int_cast( ny_local_, "same_xw WZ forward send plane count" );
        }
        run_slab_wz_communication_plane_exchange_(
            "forward", recv_planes, send_planes,
            [&]( int peer, int plane ) {
                return out.raw_ptr() +
                       ( input_dim_.start_y[peer] + static_cast<std::size_t>( plane ) ) * nz_local_ * plane_elems;
            },
            [&]( int ) { return nz_local_ * plane_elems; },
            [&]( int peer ) { return peer; },
            [&]( int peer, int plane ) {
                return in.raw_ptr() +
                       ( static_cast<std::size_t>( plane ) * nz_global_ + output_dim_.start_z[peer] ) * plane_elems;
            },
            [&]( int peer ) { return output_dim_.size_z[peer] * plane_elems; },
            [&]( int ) { return myid_j_; },
            [&]() {
                for_each_(
                    copy_forward_slab_wz_communication_self_functor<ArrayIn, ArrayOut>{
                        in, out, output_dim_.start_z[myid_j_], input_dim_.start_y[myid_j_] },
                    make_range_4d<idx_t>( nx_local_, nw_local_, nz_local_, ny_local_ )
                );
                for_each_.wait();
            }
        );
    }

    template <class ArrayIn, class ArrayOut>
    void backward_p2p_waitany_slab_wz_communication_( const ArrayIn &in, ArrayOut &out )
    {
        const int         comm_size  = line_comm_info_.num_procs;
        const std::size_t plane_elems = nx_local_ * nw_local_;
        std::vector<int> recv_planes( static_cast<std::size_t>( comm_size ), 0 );
        std::vector<int> send_planes( static_cast<std::size_t>( comm_size ), 0 );
        for ( int peer = 0; peer < comm_size; ++peer )
        {
            if ( peer == myid_j_ )
                continue;
            recv_planes[static_cast<std::size_t>( peer )] =
                detail::mpi_int_cast( ny_local_, "same_xw WZ backward receive plane count" );
            send_planes[static_cast<std::size_t>( peer )] = detail::mpi_int_cast(
                input_dim_.size_y[peer], "same_xw WZ backward send plane count"
            );
        }
        run_slab_wz_communication_plane_exchange_(
            "backward", recv_planes, send_planes,
            [&]( int peer, int plane ) {
                return out.raw_ptr() +
                       ( static_cast<std::size_t>( plane ) * nz_global_ + output_dim_.start_z[peer] ) * plane_elems;
            },
            [&]( int peer ) { return output_dim_.size_z[peer] * plane_elems; },
            [&]( int ) { return myid_j_; },
            [&]( int peer, int plane ) {
                return in.raw_ptr() +
                       ( input_dim_.start_y[peer] + static_cast<std::size_t>( plane ) ) * nz_local_ * plane_elems;
            },
            [&]( int ) { return nz_local_ * plane_elems; },
            [&]( int peer ) { return peer; },
            [&]() {
                for_each_(
                    copy_backward_slab_wz_communication_self_functor<ArrayIn, ArrayOut>{
                        in, out, input_dim_.start_y[myid_j_], output_dim_.start_z[myid_j_] },
                    make_range_4d<idx_t>( nx_local_, nw_local_, nz_local_, ny_local_ )
                );
                for_each_.wait();
            }
        );
    }

    template <class ArrayOut, class PackChunk, class PackSelf, class PackCompactChunk>
    void forward_p2p_waitany_native_xzwy_direct_(
        ArrayOut &out, PackChunk pack_chunk, PackSelf pack_self, PackCompactChunk pack_compact_chunk
    )
    {
        if ( native_xzwy_chunked_transport_enabled_ )
        {
            if ( native_xzwy_chunk_window_ == 0 )
                forward_p2p_waitany_native_xzwy_chunked_( out, pack_chunk, pack_self );
            else if ( native_xzwy_compact_staging_enabled_ )
                forward_p2p_waitany_native_xzwy_compact_windowed_( out, pack_compact_chunk, pack_self );
            else
                forward_p2p_waitany_native_xzwy_windowed_( out, pack_chunk, pack_self );
            return;
        }
        auto      scope     = profile_scope_( "forward_p2p_waitany_native_xzwy_direct" );
        const int comm_size = line_comm_info_.num_procs;
        verify_native_xzwy_direct_offsets_();

        double post_recv_ms = 0.0;
        double pack_ms      = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "direct_post_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    out.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p], mpi_value_type_, p, p,
                    recv_requests_[p]
                );
            }
        } );

        int completed_remote = 0;
        auto drain_ready     = [&]() {
            while ( completed_remote < comm_size - 1 )
            {
                int ready = 0;
                int p     = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "direct_test_recv_any" );
                    p          = line_comm_info_.testany( comm_size, recv_requests_.data(), &ready );
                } );
                if ( !ready || p == MPI_UNDEFINED )
                    break;
                ++completed_remote;
            }
        };

        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_j_ )
            {
                self_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "direct_self_pack" );
                    pack_self();
                } );
            }
            else
            {
                pack_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "direct_pack_send_peer" );
                    pack_chunk( p );
                } );
                post_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "direct_post_send_peer" );
                    line_comm_info_.isend(
                        send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p], mpi_value_type_,
                        p, myid_j_, send_requests_[p]
                    );
                } );
            }
            drain_ready();
        }

        while ( completed_remote < comm_size - 1 )
        {
            int p = MPI_UNDEFINED;
            wait_recv_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "direct_wait_recv_any" );
                p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
            } );
            if ( p == MPI_UNDEFINED )
                break;
            ++completed_remote;
        }

        wait_send_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "direct_wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        } );

        record_peer_phase_( "forward", "direct", "post_recv", post_recv_ms );
        record_peer_phase_( "forward", "direct", "pack", pack_ms );
        record_peer_phase_( "forward", "direct", "post_send", post_send_ms );
        record_peer_phase_( "forward", "direct", "wait_recv", wait_recv_ms );
        record_peer_phase_( "forward", "direct", "unpack", 0.0 );
        record_peer_phase_( "forward", "direct", "self", self_ms );
        record_peer_phase_( "forward", "direct", "wait_send", wait_send_ms );
    }

    template <class ArrayIn, class ArrayOut, class UnpackChunk, class UnpackSelf, class UnpackCompactChunk>
    void backward_p2p_waitany_native_xzwy_direct_(
        const ArrayIn &in, ArrayOut &out, UnpackChunk unpack_chunk, UnpackSelf unpack_self,
        UnpackCompactChunk unpack_compact_chunk
    )
    {
        if ( native_xzwy_chunked_transport_enabled_ )
        {
            if ( native_xzwy_chunk_window_ == 0 )
                backward_p2p_waitany_native_xzwy_chunked_( in, out, unpack_chunk, unpack_self );
            else if ( native_xzwy_compact_staging_enabled_ )
                backward_p2p_waitany_native_xzwy_compact_windowed_(
                    in, out, unpack_compact_chunk, unpack_self
                );
            else
                backward_p2p_waitany_native_xzwy_windowed_( in, out, unpack_chunk, unpack_self );
            return;
        }
        auto      scope     = profile_scope_( "backward_p2p_waitany_native_xzwy_direct" );
        const int comm_size = line_comm_info_.num_procs;
        verify_native_xzwy_direct_offsets_();

        double post_recv_ms = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double unpack_ms    = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "direct_post_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p], mpi_value_type_, p,
                    myid_j_, recv_requests_[p]
                );
            }
        } );

        int completed_remote = 0;
        auto drain_ready     = [&]() {
            while ( completed_remote < comm_size - 1 )
            {
                int ready = 0;
                int p     = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "direct_test_recv_any" );
                    p          = line_comm_info_.testany( comm_size, recv_requests_.data(), &ready );
                } );
                if ( !ready || p == MPI_UNDEFINED )
                    break;
                unpack_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "direct_unpack_recv_chunk" );
                    unpack_chunk( p );
                } );
                ++completed_remote;
            }
        };

        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_j_ )
            {
                self_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "direct_self_unpack" );
                    unpack_self();
                } );
            }
            else
            {
                post_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "direct_post_send_peer" );
                    line_comm_info_.isend(
                        in.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p], mpi_value_type_, p, p,
                        send_requests_[p]
                    );
                } );
            }
            drain_ready();
        }

        while ( completed_remote < comm_size - 1 )
        {
            int p = MPI_UNDEFINED;
            wait_recv_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "direct_wait_recv_any" );
                p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
            } );
            if ( p == MPI_UNDEFINED )
                break;
            unpack_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "direct_unpack_recv_chunk" );
                unpack_chunk( p );
            } );
            ++completed_remote;
        }

        wait_send_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "direct_wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        } );

        record_peer_phase_( "backward", "direct", "post_recv", post_recv_ms );
        record_peer_phase_( "backward", "direct", "pack", 0.0 );
        record_peer_phase_( "backward", "direct", "post_send", post_send_ms );
        record_peer_phase_( "backward", "direct", "wait_recv", wait_recv_ms );
        record_peer_phase_( "backward", "direct", "unpack", unpack_ms );
        record_peer_phase_( "backward", "direct", "self", self_ms );
        record_peer_phase_( "backward", "direct", "wait_send", wait_send_ms );
    }

    template <class ArrayOut, class PackChunk, class PackSelf>
    void forward_p2p_waitany_native_xzwy_chunked_( ArrayOut &out, PackChunk pack_chunk, PackSelf pack_self )
    {
        auto      scope     = profile_scope_( "forward_p2p_waitany_native_xzwy_chunked" );
        const int comm_size = line_comm_info_.num_procs;
        verify_native_xzwy_direct_offsets_();

        std::vector<mpi_request_t> recv_requests;
        std::vector<mpi_request_t> send_requests;
        reserve_native_xzwy_chunk_requests_( forward_recvcounts_, myid_j_, recv_requests );
        reserve_native_xzwy_chunk_requests_( forward_sendcounts_, myid_j_, send_requests );

        double post_recv_ms = 0.0;
        double pack_ms      = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "chunked_post_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                post_native_xzwy_chunked_irecv_(
                    out.raw_ptr() + forward_recv_offsets_[p], forward_recv_chunk_elems_( p ), p, p, recv_requests
                );
            }
        } );

        int completed_recv_chunks = 0;
        auto drain_ready           = [&]() {
            while ( completed_recv_chunks < static_cast<int>( recv_requests.size() ) )
            {
                int ready = 0;
                int index = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "chunked_test_recv_any" );
                    index      = line_comm_info_.testany(
                        detail::mpi_int_cast( recv_requests.size(), "same_xw chunked forward testany count" ),
                        recv_requests.data(), &ready
                    );
                } );
                if ( !ready || index == MPI_UNDEFINED )
                    break;
                ++completed_recv_chunks;
            }
        };

        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_j_ )
            {
                self_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "chunked_self_pack" );
                    pack_self();
                } );
            }
            else
            {
                pack_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "chunked_pack_send_peer" );
                    pack_chunk( p );
                } );
                post_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "chunked_post_send_peer" );
                    post_native_xzwy_chunked_isend_(
                        send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_send_chunk_elems_( p ), p,
                        myid_j_, send_requests
                    );
                } );
            }
            drain_ready();
        }

        while ( completed_recv_chunks < static_cast<int>( recv_requests.size() ) )
        {
            int index = MPI_UNDEFINED;
            wait_recv_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "chunked_wait_recv_any" );
                index      = line_comm_info_.waitany(
                    detail::mpi_int_cast( recv_requests.size(), "same_xw chunked forward waitany count" ),
                    recv_requests.data()
                );
            } );
            if ( index == MPI_UNDEFINED )
                throw std::logic_error( "same_xw chunked forward receive completion ended early" );
            ++completed_recv_chunks;
        }

        wait_send_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "chunked_wait_send" );
            waitall_native_xzwy_chunks_( send_requests, "same_xw chunked forward send waitall count" );
        } );

        record_peer_phase_( "forward", "chunked", "post_recv", post_recv_ms );
        record_peer_phase_( "forward", "chunked", "pack", pack_ms );
        record_peer_phase_( "forward", "chunked", "post_send", post_send_ms );
        record_peer_phase_( "forward", "chunked", "wait_recv", wait_recv_ms );
        record_peer_phase_( "forward", "chunked", "unpack", 0.0 );
        record_peer_phase_( "forward", "chunked", "self", self_ms );
        record_peer_phase_( "forward", "chunked", "wait_send", wait_send_ms );
    }

    template <class ArrayIn, class ArrayOut, class UnpackChunk, class UnpackSelf>
    void backward_p2p_waitany_native_xzwy_chunked_(
        const ArrayIn &in, ArrayOut &out, UnpackChunk unpack_chunk, UnpackSelf unpack_self
    )
    {
        auto      scope     = profile_scope_( "backward_p2p_waitany_native_xzwy_chunked" );
        const int comm_size = line_comm_info_.num_procs;
        verify_native_xzwy_direct_offsets_();

        std::vector<mpi_request_t> recv_requests;
        std::vector<mpi_request_t> send_requests;
        std::vector<int>           recv_request_peers;
        std::vector<int>           remaining_by_peer( comm_size, 0 );
        reserve_native_xzwy_chunk_requests_( backward_recvcounts_, myid_j_, recv_requests );
        reserve_native_xzwy_chunk_requests_( backward_sendcounts_, myid_j_, send_requests );
        recv_request_peers.reserve( recv_requests.capacity() );

        double post_recv_ms = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double unpack_ms    = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "chunked_post_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                post_native_xzwy_chunked_irecv_(
                    recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recv_chunk_elems_( p ), p,
                    myid_j_, recv_requests, &recv_request_peers, &remaining_by_peer
                );
            }
        } );

        int completed_recv_chunks = 0;
        auto complete_recv_chunk  = [&]( int index ) {
            if ( index < 0 || static_cast<std::size_t>( index ) >= recv_request_peers.size() )
                throw std::logic_error( "same_xw chunked backward receive index is invalid" );
            const int p = recv_request_peers[static_cast<std::size_t>( index )];
            --remaining_by_peer[p];
            if ( remaining_by_peer[p] == 0 )
            {
                unpack_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "chunked_unpack_recv_peer" );
                    unpack_chunk( p );
                } );
            }
            ++completed_recv_chunks;
        };
        auto drain_ready = [&]() {
            while ( completed_recv_chunks < static_cast<int>( recv_requests.size() ) )
            {
                int ready = 0;
                int index = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "chunked_test_recv_any" );
                    index      = line_comm_info_.testany(
                        detail::mpi_int_cast( recv_requests.size(), "same_xw chunked backward testany count" ),
                        recv_requests.data(), &ready
                    );
                } );
                if ( !ready || index == MPI_UNDEFINED )
                    break;
                complete_recv_chunk( index );
            }
        };

        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_j_ )
            {
                self_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "chunked_self_unpack" );
                    unpack_self();
                } );
            }
            else
            {
                post_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "chunked_post_send_peer" );
                    post_native_xzwy_chunked_isend_(
                        in.raw_ptr() + backward_send_offsets_[p], backward_send_chunk_elems_( p ), p, p,
                        send_requests
                    );
                } );
            }
            drain_ready();
        }

        while ( completed_recv_chunks < static_cast<int>( recv_requests.size() ) )
        {
            int index = MPI_UNDEFINED;
            wait_recv_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "chunked_wait_recv_any" );
                index      = line_comm_info_.waitany(
                    detail::mpi_int_cast( recv_requests.size(), "same_xw chunked backward waitany count" ),
                    recv_requests.data()
                );
            } );
            if ( index == MPI_UNDEFINED )
                throw std::logic_error( "same_xw chunked backward receive completion ended early" );
            complete_recv_chunk( index );
        }

        wait_send_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "chunked_wait_send" );
            waitall_native_xzwy_chunks_( send_requests, "same_xw chunked backward send waitall count" );
        } );

        record_peer_phase_( "backward", "chunked", "post_recv", post_recv_ms );
        record_peer_phase_( "backward", "chunked", "pack", 0.0 );
        record_peer_phase_( "backward", "chunked", "post_send", post_send_ms );
        record_peer_phase_( "backward", "chunked", "wait_recv", wait_recv_ms );
        record_peer_phase_( "backward", "chunked", "unpack", unpack_ms );
        record_peer_phase_( "backward", "chunked", "self", self_ms );
        record_peer_phase_( "backward", "chunked", "wait_send", wait_send_ms );
    }

    template <class ArrayOut, class PackCompactChunk, class PackSelf>
    void forward_p2p_waitany_native_xzwy_compact_windowed_(
        ArrayOut &out, PackCompactChunk pack_compact_chunk, PackSelf pack_self
    )
    {
        auto      scope     = profile_scope_( "forward_p2p_waitany_native_xzwy_compact_windowed" );
        const int comm_size = line_comm_info_.num_procs;
        verify_native_xzwy_direct_offsets_();

        std::vector<int> recv_chunk_counts( static_cast<std::size_t>( comm_size ), 0 );
        std::vector<int> send_chunk_counts( static_cast<std::size_t>( comm_size ), 0 );
        for ( int peer = 0; peer < comm_size; ++peer )
        {
            if ( peer == myid_j_ )
                continue;
            recv_chunk_counts[static_cast<std::size_t>( peer )] =
                native_xzwy_compact_forward_recv_chunk_count_( peer );
            send_chunk_counts[static_cast<std::size_t>( peer )] =
                native_xzwy_compact_forward_send_chunk_count_( peer );
        }
        native_xzwy_chunk_window_state recv_state =
            make_native_xzwy_chunk_window_state_from_chunk_counts_( recv_chunk_counts, myid_j_ );
        native_xzwy_chunk_window_state send_state =
            make_native_xzwy_chunk_window_state_from_chunk_counts_( send_chunk_counts, myid_j_ );

        double post_recv_ms = 0.0;
        double pack_ms      = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        auto post_recv_chunk = [&]( int peer, int chunk, int, mpi_request_t &request ) {
            const native_xzwy_compact_chunk_span span = native_xzwy_compact_chunk_span_(
                nz_local_, input_dim_.size_y[peer], chunk
            );
            post_native_xzwy_compact_irecv_(
                out.raw_ptr() + forward_recv_offsets_[peer] + span.begin_elems, span, peer, peer, chunk, request
            );
        };
        auto start_send_chunk = [&]( int peer, int chunk, int slot, mpi_request_t &request ) {
            value_type *slot_ptr = native_xzwy_compact_slot_ptr_( send_state, slot );
            pack_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "compact_window_pack_send_chunk" );
                pack_compact_chunk( peer, chunk, slot_ptr );
            } );
            const native_xzwy_compact_chunk_span span = native_xzwy_compact_chunk_span_(
                output_dim_.size_z[peer], ny_local_, chunk
            );
            post_send_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "compact_window_post_send_chunk" );
                post_native_xzwy_compact_isend_( slot_ptr, span, peer, myid_j_, chunk, request );
            } );
        };

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "compact_window_post_recv" );
            for ( int peer = 0; peer < comm_size; ++peer )
            {
                if ( peer != myid_j_ )
                    start_native_xzwy_compact_chunk_window_peer_( recv_state, peer, post_recv_chunk );
            }
        } );

        auto advance_recv = [&]( int slot ) {
            int peer = -1;
            int completed_chunk = -1;
            int next_chunk = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                recv_state, slot, peer, completed_chunk, next_chunk
            );
            if ( replenish )
            {
                post_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_repost_recv" );
                    post_recv_chunk( peer, next_chunk, slot, recv_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };
        auto advance_send = [&]( int slot ) {
            int peer = -1;
            int completed_chunk = -1;
            int next_chunk = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                send_state, slot, peer, completed_chunk, next_chunk
            );
            if ( replenish )
                start_send_chunk( peer, next_chunk, slot, send_state.requests[static_cast<std::size_t>( slot )] );
        };
        auto drain_recv = [&]() {
            bool progressed = false;
            while ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int ready = 0;
                int slot  = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_test_recv_any" );
                    slot = line_comm_info_.testany(
                        detail::mpi_int_cast(
                            recv_state.requests.size(), "same_xw compact forward recv test count"
                        ),
                        recv_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_recv( slot );
                progressed = true;
            }
            return progressed;
        };
        auto drain_send = [&]() {
            bool progressed = false;
            while ( send_state.completed_chunks < send_state.total_chunks )
            {
                int ready = 0;
                int slot  = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_test_send_any" );
                    slot = line_comm_info_.testany(
                        detail::mpi_int_cast(
                            send_state.requests.size(), "same_xw compact forward send test count"
                        ),
                        send_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_send( slot );
                progressed = true;
            }
            return progressed;
        };

        for ( int peer = 0; peer < comm_size; ++peer )
        {
            if ( peer == myid_j_ )
            {
                self_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_self_pack" );
                    pack_self();
                } );
            }
            else
            {
                start_native_xzwy_compact_chunk_window_peer_( send_state, peer, start_send_chunk );
            }
            drain_recv();
            drain_send();
        }

        while ( recv_state.completed_chunks < recv_state.total_chunks ||
                send_state.completed_chunks < send_state.total_chunks )
        {
            const bool recv_progress = drain_recv();
            const bool send_progress = drain_send();
            if ( recv_progress || send_progress )
                continue;
            if ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int slot = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_wait_recv_any" );
                    slot = line_comm_info_.waitany(
                        detail::mpi_int_cast(
                            recv_state.requests.size(), "same_xw compact forward recv wait count"
                        ),
                        recv_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw compact forward receive completion ended early" );
                advance_recv( slot );
            }
            else
            {
                int slot = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_wait_send_any" );
                    slot = line_comm_info_.waitany(
                        detail::mpi_int_cast(
                            send_state.requests.size(), "same_xw compact forward send wait count"
                        ),
                        send_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw compact forward send completion ended early" );
                advance_send( slot );
            }
        }

        record_peer_phase_( "forward", "compact_window", "post_recv", post_recv_ms );
        record_peer_phase_( "forward", "compact_window", "pack", pack_ms );
        record_peer_phase_( "forward", "compact_window", "post_send", post_send_ms );
        record_peer_phase_( "forward", "compact_window", "wait_recv", wait_recv_ms );
        record_peer_phase_( "forward", "compact_window", "unpack", 0.0 );
        record_peer_phase_( "forward", "compact_window", "self", self_ms );
        record_peer_phase_( "forward", "compact_window", "wait_send", wait_send_ms );
    }

    template <class ArrayIn, class ArrayOut, class UnpackCompactChunk, class UnpackSelf>
    void backward_p2p_waitany_native_xzwy_compact_windowed_(
        const ArrayIn &in, ArrayOut &out, UnpackCompactChunk unpack_compact_chunk, UnpackSelf unpack_self
    )
    {
        auto      scope     = profile_scope_( "backward_p2p_waitany_native_xzwy_compact_windowed" );
        const int comm_size = line_comm_info_.num_procs;
        verify_native_xzwy_direct_offsets_();

        std::vector<int> recv_chunk_counts( static_cast<std::size_t>( comm_size ), 0 );
        std::vector<int> send_chunk_counts( static_cast<std::size_t>( comm_size ), 0 );
        for ( int peer = 0; peer < comm_size; ++peer )
        {
            if ( peer == myid_j_ )
                continue;
            recv_chunk_counts[static_cast<std::size_t>( peer )] =
                native_xzwy_compact_backward_recv_chunk_count_( peer );
            send_chunk_counts[static_cast<std::size_t>( peer )] =
                native_xzwy_compact_backward_send_chunk_count_( peer );
        }
        native_xzwy_chunk_window_state recv_state =
            make_native_xzwy_chunk_window_state_from_chunk_counts_( recv_chunk_counts, myid_j_ );
        native_xzwy_chunk_window_state send_state =
            make_native_xzwy_chunk_window_state_from_chunk_counts_( send_chunk_counts, myid_j_ );

        double post_recv_ms = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double unpack_ms    = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        auto post_recv_chunk = [&]( int peer, int chunk, int slot, mpi_request_t &request ) {
            const native_xzwy_compact_chunk_span span = native_xzwy_compact_chunk_span_(
                output_dim_.size_z[peer], ny_local_, chunk
            );
            post_native_xzwy_compact_irecv_(
                native_xzwy_compact_slot_ptr_( recv_state, slot ), span, peer, myid_j_, chunk, request
            );
        };
        auto post_send_chunk = [&]( int peer, int chunk, int, mpi_request_t &request ) {
            const native_xzwy_compact_chunk_span span = native_xzwy_compact_chunk_span_(
                nz_local_, input_dim_.size_y[peer], chunk
            );
            post_send_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "compact_window_post_send_chunk" );
                post_native_xzwy_compact_isend_(
                    in.raw_ptr() + backward_send_offsets_[peer] + span.begin_elems,
                    span, peer, peer, chunk, request
                );
            } );
        };

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "compact_window_post_recv" );
            for ( int peer = 0; peer < comm_size; ++peer )
            {
                if ( peer != myid_j_ )
                    start_native_xzwy_compact_chunk_window_peer_( recv_state, peer, post_recv_chunk );
            }
        } );

        auto advance_recv = [&]( int slot ) {
            int peer = -1;
            int completed_chunk = -1;
            int next_chunk = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                recv_state, slot, peer, completed_chunk, next_chunk
            );
            unpack_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "compact_window_unpack_recv_chunk" );
                unpack_compact_chunk(
                    native_xzwy_compact_slot_ptr_( recv_state, slot ), peer, completed_chunk
                );
            } );
            if ( replenish )
            {
                post_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_repost_recv" );
                    post_recv_chunk( peer, next_chunk, slot, recv_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };
        auto advance_send = [&]( int slot ) {
            int peer = -1;
            int completed_chunk = -1;
            int next_chunk = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                send_state, slot, peer, completed_chunk, next_chunk
            );
            if ( replenish )
                post_send_chunk( peer, next_chunk, slot, send_state.requests[static_cast<std::size_t>( slot )] );
        };
        auto drain_recv = [&]() {
            bool progressed = false;
            while ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int ready = 0;
                int slot  = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_test_recv_any" );
                    slot = line_comm_info_.testany(
                        detail::mpi_int_cast(
                            recv_state.requests.size(), "same_xw compact backward recv test count"
                        ),
                        recv_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_recv( slot );
                progressed = true;
            }
            return progressed;
        };
        auto drain_send = [&]() {
            bool progressed = false;
            while ( send_state.completed_chunks < send_state.total_chunks )
            {
                int ready = 0;
                int slot  = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_test_send_any" );
                    slot = line_comm_info_.testany(
                        detail::mpi_int_cast(
                            send_state.requests.size(), "same_xw compact backward send test count"
                        ),
                        send_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_send( slot );
                progressed = true;
            }
            return progressed;
        };

        for ( int peer = 0; peer < comm_size; ++peer )
        {
            if ( peer == myid_j_ )
            {
                self_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_self_unpack" );
                    unpack_self();
                } );
            }
            else
            {
                start_native_xzwy_compact_chunk_window_peer_( send_state, peer, post_send_chunk );
            }
            drain_recv();
            drain_send();
        }

        while ( recv_state.completed_chunks < recv_state.total_chunks ||
                send_state.completed_chunks < send_state.total_chunks )
        {
            const bool recv_progress = drain_recv();
            const bool send_progress = drain_send();
            if ( recv_progress || send_progress )
                continue;
            if ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int slot = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_wait_recv_any" );
                    slot = line_comm_info_.waitany(
                        detail::mpi_int_cast(
                            recv_state.requests.size(), "same_xw compact backward recv wait count"
                        ),
                        recv_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw compact backward receive completion ended early" );
                advance_recv( slot );
            }
            else
            {
                int slot = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "compact_window_wait_send_any" );
                    slot = line_comm_info_.waitany(
                        detail::mpi_int_cast(
                            send_state.requests.size(), "same_xw compact backward send wait count"
                        ),
                        send_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw compact backward send completion ended early" );
                advance_send( slot );
            }
        }

        record_peer_phase_( "backward", "compact_window", "post_recv", post_recv_ms );
        record_peer_phase_( "backward", "compact_window", "pack", 0.0 );
        record_peer_phase_( "backward", "compact_window", "post_send", post_send_ms );
        record_peer_phase_( "backward", "compact_window", "wait_recv", wait_recv_ms );
        record_peer_phase_( "backward", "compact_window", "unpack", unpack_ms );
        record_peer_phase_( "backward", "compact_window", "self", self_ms );
        record_peer_phase_( "backward", "compact_window", "wait_send", wait_send_ms );
    }

    template <class ArrayOut, class PackChunk, class PackSelf>
    void forward_p2p_waitany_native_xzwy_windowed_( ArrayOut &out, PackChunk pack_chunk, PackSelf pack_self )
    {
        auto      scope     = profile_scope_( "forward_p2p_waitany_native_xzwy_windowed" );
        const int comm_size = line_comm_info_.num_procs;
        verify_native_xzwy_direct_offsets_();

        native_xzwy_chunk_window_state recv_state =
            make_native_xzwy_chunk_window_state_( forward_recvcounts_, myid_j_ );
        native_xzwy_chunk_window_state send_state =
            make_native_xzwy_chunk_window_state_( forward_sendcounts_, myid_j_ );

        double post_recv_ms = 0.0;
        double pack_ms      = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        auto post_recv_chunk = [&]( int peer, int chunk, mpi_request_t &request ) {
            post_native_xzwy_windowed_irecv_chunk_(
                out.raw_ptr() + forward_recv_offsets_[peer], forward_recv_chunk_elems_( peer ), peer, peer, chunk,
                request
            );
        };
        auto post_send_chunk = [&]( int peer, int chunk, mpi_request_t &request ) {
            post_native_xzwy_windowed_isend_chunk_(
                send_buffer_.raw_ptr() + forward_send_offsets_[peer], forward_send_chunk_elems_( peer ), peer,
                myid_j_, chunk, request
            );
        };

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "windowed_post_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p != myid_j_ )
                    start_native_xzwy_chunk_window_peer_( recv_state, p, post_recv_chunk );
            }
        } );

        auto advance_recv = [&]( int slot ) {
            int peer = -1;
            int completed_chunk = -1;
            int next_chunk = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                recv_state, slot, peer, completed_chunk, next_chunk
            );
            if ( replenish )
            {
                post_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_repost_recv" );
                    post_recv_chunk( peer, next_chunk, recv_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };
        auto advance_send = [&]( int slot ) {
            int peer = -1;
            int completed_chunk = -1;
            int next_chunk = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                send_state, slot, peer, completed_chunk, next_chunk
            );
            if ( replenish )
            {
                post_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_repost_send" );
                    post_send_chunk( peer, next_chunk, send_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };
        auto drain_recv = [&]() {
            bool progressed = false;
            while ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int ready = 0;
                int slot  = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_test_recv_any" );
                    slot       = line_comm_info_.testany(
                        detail::mpi_int_cast( recv_state.requests.size(), "same_xw windowed forward recv test count" ),
                        recv_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_recv( slot );
                progressed = true;
            }
            return progressed;
        };
        auto drain_send = [&]() {
            bool progressed = false;
            while ( send_state.completed_chunks < send_state.total_chunks )
            {
                int ready = 0;
                int slot  = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_test_send_any" );
                    slot       = line_comm_info_.testany(
                        detail::mpi_int_cast( send_state.requests.size(), "same_xw windowed forward send test count" ),
                        send_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_send( slot );
                progressed = true;
            }
            return progressed;
        };

        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_j_ )
            {
                self_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_self_pack" );
                    pack_self();
                } );
            }
            else
            {
                pack_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_pack_send_peer" );
                    pack_chunk( p );
                } );
                post_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_post_send_peer" );
                    start_native_xzwy_chunk_window_peer_( send_state, p, post_send_chunk );
                } );
            }
            drain_recv();
            drain_send();
        }

        while ( recv_state.completed_chunks < recv_state.total_chunks ||
                send_state.completed_chunks < send_state.total_chunks )
        {
            const bool progressed = drain_recv() || drain_send();
            if ( progressed )
                continue;
            if ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int slot = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_wait_recv_any" );
                    slot       = line_comm_info_.waitany(
                        detail::mpi_int_cast( recv_state.requests.size(), "same_xw windowed forward recv wait count" ),
                        recv_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw windowed forward receive completion ended early" );
                advance_recv( slot );
            }
            else
            {
                int slot = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_wait_send_any" );
                    slot       = line_comm_info_.waitany(
                        detail::mpi_int_cast( send_state.requests.size(), "same_xw windowed forward send wait count" ),
                        send_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw windowed forward send completion ended early" );
                advance_send( slot );
            }
        }

        record_peer_phase_( "forward", "chunked_window", "post_recv", post_recv_ms );
        record_peer_phase_( "forward", "chunked_window", "pack", pack_ms );
        record_peer_phase_( "forward", "chunked_window", "post_send", post_send_ms );
        record_peer_phase_( "forward", "chunked_window", "wait_recv", wait_recv_ms );
        record_peer_phase_( "forward", "chunked_window", "unpack", 0.0 );
        record_peer_phase_( "forward", "chunked_window", "self", self_ms );
        record_peer_phase_( "forward", "chunked_window", "wait_send", wait_send_ms );
    }

    template <class ArrayIn, class ArrayOut, class UnpackChunk, class UnpackSelf>
    void backward_p2p_waitany_native_xzwy_windowed_(
        const ArrayIn &in, ArrayOut &out, UnpackChunk unpack_chunk, UnpackSelf unpack_self
    )
    {
        auto      scope     = profile_scope_( "backward_p2p_waitany_native_xzwy_windowed" );
        const int comm_size = line_comm_info_.num_procs;
        verify_native_xzwy_direct_offsets_();

        native_xzwy_chunk_window_state recv_state =
            make_native_xzwy_chunk_window_state_( backward_recvcounts_, myid_j_ );
        native_xzwy_chunk_window_state send_state =
            make_native_xzwy_chunk_window_state_( backward_sendcounts_, myid_j_ );
        std::vector<int> remaining_by_peer = recv_state.chunk_counts;

        double post_recv_ms = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double unpack_ms    = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        auto post_recv_chunk = [&]( int peer, int chunk, mpi_request_t &request ) {
            post_native_xzwy_windowed_irecv_chunk_(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[peer], backward_recv_chunk_elems_( peer ), peer,
                myid_j_, chunk, request
            );
        };
        auto post_send_chunk = [&]( int peer, int chunk, mpi_request_t &request ) {
            post_native_xzwy_windowed_isend_chunk_(
                in.raw_ptr() + backward_send_offsets_[peer], backward_send_chunk_elems_( peer ), peer, peer, chunk,
                request
            );
        };

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "windowed_post_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p != myid_j_ )
                    start_native_xzwy_chunk_window_peer_( recv_state, p, post_recv_chunk );
            }
        } );

        auto advance_recv = [&]( int slot ) {
            int peer = -1;
            int completed_chunk = -1;
            int next_chunk = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                recv_state, slot, peer, completed_chunk, next_chunk
            );
            --remaining_by_peer[peer];
            if ( remaining_by_peer[peer] == 0 )
            {
                unpack_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_unpack_recv_peer" );
                    unpack_chunk( peer );
                } );
            }
            if ( replenish )
            {
                post_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_repost_recv" );
                    post_recv_chunk( peer, next_chunk, recv_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };
        auto advance_send = [&]( int slot ) {
            int peer = -1;
            int completed_chunk = -1;
            int next_chunk = -1;
            const bool replenish = complete_native_xzwy_chunk_window_slot_(
                send_state, slot, peer, completed_chunk, next_chunk
            );
            if ( replenish )
            {
                post_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_repost_send" );
                    post_send_chunk( peer, next_chunk, send_state.requests[static_cast<std::size_t>( slot )] );
                } );
            }
        };
        auto drain_recv = [&]() {
            bool progressed = false;
            while ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int ready = 0;
                int slot  = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_test_recv_any" );
                    slot       = line_comm_info_.testany(
                        detail::mpi_int_cast( recv_state.requests.size(), "same_xw windowed backward recv test count" ),
                        recv_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_recv( slot );
                progressed = true;
            }
            return progressed;
        };
        auto drain_send = [&]() {
            bool progressed = false;
            while ( send_state.completed_chunks < send_state.total_chunks )
            {
                int ready = 0;
                int slot  = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_test_send_any" );
                    slot       = line_comm_info_.testany(
                        detail::mpi_int_cast( send_state.requests.size(), "same_xw windowed backward send test count" ),
                        send_state.requests.data(), &ready
                    );
                } );
                if ( !ready || slot == MPI_UNDEFINED )
                    break;
                advance_send( slot );
                progressed = true;
            }
            return progressed;
        };

        for ( int p = 0; p < comm_size; ++p )
        {
            if ( p == myid_j_ )
            {
                self_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_self_unpack" );
                    unpack_self();
                } );
            }
            else
            {
                post_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_post_send_peer" );
                    start_native_xzwy_chunk_window_peer_( send_state, p, post_send_chunk );
                } );
            }
            drain_recv();
            drain_send();
        }

        while ( recv_state.completed_chunks < recv_state.total_chunks ||
                send_state.completed_chunks < send_state.total_chunks )
        {
            const bool progressed = drain_recv() || drain_send();
            if ( progressed )
                continue;
            if ( recv_state.completed_chunks < recv_state.total_chunks )
            {
                int slot = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_wait_recv_any" );
                    slot       = line_comm_info_.waitany(
                        detail::mpi_int_cast( recv_state.requests.size(), "same_xw windowed backward recv wait count" ),
                        recv_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw windowed backward receive completion ended early" );
                advance_recv( slot );
            }
            else
            {
                int slot = MPI_UNDEFINED;
                wait_send_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "windowed_wait_send_any" );
                    slot       = line_comm_info_.waitany(
                        detail::mpi_int_cast( send_state.requests.size(), "same_xw windowed backward send wait count" ),
                        send_state.requests.data()
                    );
                } );
                if ( slot == MPI_UNDEFINED )
                    throw std::logic_error( "same_xw windowed backward send completion ended early" );
                advance_send( slot );
            }
        }

        record_peer_phase_( "backward", "chunked_window", "post_recv", post_recv_ms );
        record_peer_phase_( "backward", "chunked_window", "pack", 0.0 );
        record_peer_phase_( "backward", "chunked_window", "post_send", post_send_ms );
        record_peer_phase_( "backward", "chunked_window", "wait_recv", wait_recv_ms );
        record_peer_phase_( "backward", "chunked_window", "unpack", unpack_ms );
        record_peer_phase_( "backward", "chunked_window", "self", self_ms );
        record_peer_phase_( "backward", "chunked_window", "wait_send", wait_send_ms );
    }

    template <class ArrayOut, class PackChunk>
    void forward_p2p_waitany_peer_paired_( ArrayOut &out, PackChunk pack_chunk, bool slab_native_unpack )
    {
        auto              scope = profile_scope_( "forward_p2p_waitany_peer_paired" );
        scoped_bool_flag_ use_slab_native_unpack( p2p_slab_native_unpack_, slab_native_unpack );
        const int         comm_size = line_comm_info_.num_procs;

        double post_recv_ms = 0.0;
        double pack_ms      = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double unpack_ms    = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "post_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p], mpi_value_type_, p,
                    p, recv_requests_[p]
                );
            }
        } );

        int completed_remote = 0;
        auto drain_ready     = [&]() {
            while ( completed_remote < comm_size - 1 )
            {
                int ready = 0;
                int p     = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "test_recv_any" );
                    p          = line_comm_info_.testany( comm_size, recv_requests_.data(), &ready );
                } );
                if ( !ready || p == MPI_UNDEFINED )
                    break;
                unpack_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_forward_p2p_chunk_( p, out );
                } );
                ++completed_remote;
            }
        };

        {
            auto phase = profile_scope_( "pack_send_peers" );
            for ( int p = 0; p < comm_size; ++p )
            {
                pack_ms += time_direct_host_phase_( [&]() { pack_chunk( p ); } );
                if ( p == myid_j_ )
                {
                    self_ms += time_direct_host_phase_( [&]() {
                        auto self_phase = profile_scope_( "self_copy" );
                        runtime_api_t::memcpy(
                            recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_j_],
                            send_buffer_.raw_ptr() + forward_send_offsets_[myid_j_],
                            bytes_from_elems_( forward_recv_chunk_elems_( myid_j_ ) ),
                            runtime_api_t::device_to_device_kind()
                        );
                    } );
                    self_ms += time_direct_host_phase_( [&]() {
                        auto self_phase = profile_scope_( "unpack_self" );
                        unpack_forward_p2p_chunk_( myid_j_, out );
                    } );
                }
                else
                {
                    post_send_ms += time_direct_host_phase_( [&]() {
                        auto send_phase = profile_scope_( "post_send_peer" );
                        line_comm_info_.isend(
                            send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                            mpi_value_type_, p, myid_j_, send_requests_[p]
                        );
                    } );
                }
                drain_ready();
            }
        }

        while ( completed_remote < comm_size - 1 )
        {
            int p = MPI_UNDEFINED;
            wait_recv_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "wait_recv_any" );
                p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
            } );
            if ( p == MPI_UNDEFINED )
                break;
            unpack_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "unpack_recv_chunk" );
                unpack_forward_p2p_chunk_( p, out );
            } );
            ++completed_remote;
        }

        wait_send_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        } );

        record_peer_phase_( "forward", "buffered", "post_recv", post_recv_ms );
        record_peer_phase_( "forward", "buffered", "pack", pack_ms );
        record_peer_phase_( "forward", "buffered", "post_send", post_send_ms );
        record_peer_phase_( "forward", "buffered", "wait_recv", wait_recv_ms );
        record_peer_phase_( "forward", "buffered", "unpack", unpack_ms );
        record_peer_phase_( "forward", "buffered", "self", self_ms );
        record_peer_phase_( "forward", "buffered", "wait_send", wait_send_ms );
    }

    template <class ArrayOut, class PackChunk>
    void backward_p2p_waitany_peer_paired_( ArrayOut &out, PackChunk pack_chunk, bool slab_native_unpack )
    {
        auto              scope = profile_scope_( "backward_p2p_waitany_peer_paired" );
        scoped_bool_flag_ use_slab_native_unpack( p2p_slab_native_unpack_, slab_native_unpack );
        const int         comm_size = line_comm_info_.num_procs;

        double post_recv_ms = 0.0;
        double pack_ms      = 0.0;
        double post_send_ms = 0.0;
        double wait_recv_ms = 0.0;
        double unpack_ms    = 0.0;
        double self_ms      = 0.0;
        double wait_send_ms = 0.0;

        post_recv_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "post_recv" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p], mpi_value_type_, p,
                    myid_j_, recv_requests_[p]
                );
            }
        } );

        int completed_remote = 0;
        auto drain_ready     = [&]() {
            while ( completed_remote < comm_size - 1 )
            {
                int ready = 0;
                int p     = MPI_UNDEFINED;
                wait_recv_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "test_recv_any" );
                    p          = line_comm_info_.testany( comm_size, recv_requests_.data(), &ready );
                } );
                if ( !ready || p == MPI_UNDEFINED )
                    break;
                unpack_ms += time_direct_host_phase_( [&]() {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_backward_p2p_chunk_( p, out );
                } );
                ++completed_remote;
            }
        };

        {
            auto phase = profile_scope_( "pack_send_peers" );
            for ( int p = 0; p < comm_size; ++p )
            {
                pack_ms += time_direct_host_phase_( [&]() { pack_chunk( p ); } );
                if ( p == myid_j_ )
                {
                    self_ms += time_direct_host_phase_( [&]() {
                        auto self_phase = profile_scope_( "self_copy" );
                        runtime_api_t::memcpy(
                            recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_j_],
                            send_buffer_.raw_ptr() + backward_send_offsets_[myid_j_],
                            bytes_from_elems_( backward_recv_chunk_elems_( myid_j_ ) ),
                            runtime_api_t::device_to_device_kind()
                        );
                    } );
                    self_ms += time_direct_host_phase_( [&]() {
                        auto self_phase = profile_scope_( "unpack_self" );
                        unpack_backward_p2p_chunk_( myid_j_, out );
                    } );
                }
                else
                {
                    post_send_ms += time_direct_host_phase_( [&]() {
                        auto send_phase = profile_scope_( "post_send_peer" );
                        line_comm_info_.isend(
                            send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                            mpi_value_type_, p, p, send_requests_[p]
                        );
                    } );
                }
                drain_ready();
            }
        }

        while ( completed_remote < comm_size - 1 )
        {
            int p = MPI_UNDEFINED;
            wait_recv_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "wait_recv_any" );
                p          = line_comm_info_.waitany( comm_size, recv_requests_.data() );
            } );
            if ( p == MPI_UNDEFINED )
                break;
            unpack_ms += time_direct_host_phase_( [&]() {
                auto phase = profile_scope_( "unpack_recv_chunk" );
                unpack_backward_p2p_chunk_( p, out );
            } );
            ++completed_remote;
        }

        wait_send_ms += time_direct_host_phase_( [&]() {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        } );

        record_peer_phase_( "backward", "buffered", "post_recv", post_recv_ms );
        record_peer_phase_( "backward", "buffered", "pack", pack_ms );
        record_peer_phase_( "backward", "buffered", "post_send", post_send_ms );
        record_peer_phase_( "backward", "buffered", "wait_recv", wait_recv_ms );
        record_peer_phase_( "backward", "buffered", "unpack", unpack_ms );
        record_peer_phase_( "backward", "buffered", "self", self_ms );
        record_peer_phase_( "backward", "buffered", "wait_send", wait_send_ms );
    }

    template <class ArrayOut>
    void forward_p2p_waitall_( ArrayOut &out )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
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
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                    mpi_value_type_, p, myid_j_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_j_],
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_j_],
                bytes_from_elems_( forward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
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
                if ( p != myid_j_ )
                {
                    copy_host_forward_chunk_to_device_( p );
                }
            }
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                unpack_forward_p2p_chunk_( p, out );
        }
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
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                    mpi_value_type_, p, myid_j_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_j_],
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_j_],
                bytes_from_elems_( forward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                unpack_forward_p2p_chunk_( p, out );
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
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
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
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                    mpi_value_type_, p, myid_j_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_j_],
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_j_],
                bytes_from_elems_( forward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_forward_p2p_chunk_( myid_j_, out );
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
                    copy_host_forward_chunk_to_device_( p );
                }
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_forward_p2p_chunk_( p, out );
                }
                completed++;
            }
        }

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
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + forward_recv_offsets_[p], forward_recvcounts_[p],
                    mpi_value_type_, p, p, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + forward_send_offsets_[p], forward_sendcounts_[p],
                    mpi_value_type_, p, myid_j_, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + forward_recv_offsets_[myid_j_],
                send_buffer_.raw_ptr() + forward_send_offsets_[myid_j_],
                bytes_from_elems_( forward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_forward_p2p_chunk_( myid_j_, out );
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
                    unpack_forward_p2p_chunk_( p, out );
                }
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
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
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
            copy_recv_buffer_from_host_();
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_forward_chunk_( p, out );
        }
#else
        auto scope = profile_scope_( "forward_alltoallv" );
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "mpi_alltoallv", [&]() {
                    line_comm_info_.alltoallv(
                        static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                        forward_sdispls_.data(), mpi_value_type_, static_cast<void *>( recv_buffer_.raw_ptr() ),
                        forward_recvcounts_.data(), forward_rdispls_.data(), mpi_value_type_
                    );
                }
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "unpack", [&]() {
                    for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                        unpack_forward_chunk_( p, out );
                }
            );
        }
#endif
    }

    template <class ArrayOut>
    void forward_alltoallw_( ArrayOut &out )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto scope = profile_scope_( "forward_alltoallw" );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            ensure_forward_alltoallw_layout_();
        }
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), forward_sendcounts_w_.data(),
                forward_sdispls_w_.data(), forward_sendtypes_w_.data(),
                static_cast<void *>( host_recv_buffer_.raw_ptr() ), forward_recvcounts_w_.data(),
                forward_rdispls_w_.data(), forward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            copy_recv_buffer_from_host_();
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_forward_chunk_( p, out );
        }
#else
        auto scope = profile_scope_( "forward_alltoallw" );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            ensure_forward_alltoallw_layout_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "mpi_alltoallw", [&]() {
                    line_comm_info_.alltoallw(
                        static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_w_.data(),
                        forward_sdispls_w_.data(), forward_sendtypes_w_.data(),
                        static_cast<void *>( recv_buffer_.raw_ptr() ), forward_recvcounts_w_.data(),
                        forward_rdispls_w_.data(), forward_recvtypes_w_.data()
                    );
                }
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "unpack", [&]() {
                    for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                        unpack_forward_chunk_( p, out );
                }
            );
        }
#endif
    }

    template <class ArrayOut>
    void backward_p2p_waitall_( ArrayOut &out )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto      scope     = profile_scope_( "backward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                    mpi_value_type_, p, myid_j_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_j_],
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_j_],
                bytes_from_elems_( backward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
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
                if ( p != myid_j_ )
                {
                    copy_host_backward_chunk_to_device_( p );
                }
            }
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                unpack_backward_p2p_chunk_( p, out );
        }
        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#else
        auto      scope     = profile_scope_( "backward_p2p_waitall" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                    mpi_value_type_, p, myid_j_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_j_],
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_j_],
                bytes_from_elems_( backward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
            );
        }

        {
            auto phase = profile_scope_( "wait_recv" );
            line_comm_info_.waitall( comm_size, recv_requests_.data() );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < comm_size; ++p )
                unpack_backward_p2p_chunk_( p, out );
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
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto      scope     = profile_scope_( "backward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    host_recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                    mpi_value_type_, p, myid_j_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    host_send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_j_],
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_j_],
                bytes_from_elems_( backward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_backward_p2p_chunk_( myid_j_, out );
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
                    copy_host_backward_chunk_to_device_( p );
                }
                {
                    auto phase = profile_scope_( "unpack_recv_chunk" );
                    unpack_backward_p2p_chunk_( p, out );
                }
                completed++;
            }
        }

        {
            auto phase = profile_scope_( "wait_send" );
            line_comm_info_.waitall( comm_size, send_requests_.data() );
        }
#else
        auto      scope     = profile_scope_( "backward_p2p_waitany" );
        const int comm_size = line_comm_info_.num_procs;
        {
            auto phase = profile_scope_( "post_recv_send" );
            for ( int p = 0; p < comm_size; ++p )
            {
                if ( p == myid_j_ )
                    continue;
                line_comm_info_.irecv(
                    recv_buffer_.raw_ptr() + backward_recv_offsets_[p], backward_recvcounts_[p],
                    mpi_value_type_, p, myid_j_, recv_requests_[p]
                );
                line_comm_info_.isend(
                    send_buffer_.raw_ptr() + backward_send_offsets_[p], backward_sendcounts_[p],
                    mpi_value_type_, p, p, send_requests_[p]
                );
            }
        }

        {
            auto phase = profile_scope_( "self_copy" );
            runtime_api_t::memcpy(
                recv_buffer_.raw_ptr() + backward_recv_offsets_[myid_j_],
                send_buffer_.raw_ptr() + backward_send_offsets_[myid_j_],
                bytes_from_elems_( backward_recv_chunk_elems_( myid_j_ ) ), runtime_api_t::device_to_device_kind()
            );
        }
        {
            auto phase = profile_scope_( "unpack_self" );
            unpack_backward_p2p_chunk_( myid_j_, out );
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
                    unpack_backward_p2p_chunk_( p, out );
                }
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
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto scope = profile_scope_( "backward_alltoallv" );
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
            copy_recv_buffer_from_host_();
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_backward_chunk_( p, out );
        }
#else
        auto scope = profile_scope_( "backward_alltoallv" );
        {
            auto phase = profile_scope_( "mpi_alltoallv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "mpi_alltoallv", [&]() {
                    line_comm_info_.alltoallv(
                        static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                        backward_sdispls_.data(), mpi_value_type_, static_cast<void *>( recv_buffer_.raw_ptr() ),
                        backward_recvcounts_.data(), backward_rdispls_.data(), mpi_value_type_
                    );
                }
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "unpack", [&]() {
                    for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                        unpack_backward_chunk_( p, out );
                }
            );
        }
#endif
    }

    template <class ArrayOut>
    void backward_alltoallw_( ArrayOut &out )
    {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        auto scope = profile_scope_( "backward_alltoallw" );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            ensure_backward_alltoallw_layout_();
        }
        {
            auto phase = profile_scope_( "stage_send_to_host" );
            copy_send_buffer_to_host_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            line_comm_info_.alltoallw(
                static_cast<const void *>( host_send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
                backward_sdispls_w_.data(), backward_sendtypes_w_.data(),
                static_cast<void *>( host_recv_buffer_.raw_ptr() ), backward_recvcounts_w_.data(),
                backward_rdispls_w_.data(), backward_recvtypes_w_.data()
            );
        }
        {
            auto phase = profile_scope_( "stage_recv_to_device" );
            copy_recv_buffer_from_host_();
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                unpack_backward_chunk_( p, out );
        }
#else
        auto scope = profile_scope_( "backward_alltoallw" );
        {
            auto phase = profile_scope_( "prepare_alltoallw_layout" );
            ensure_backward_alltoallw_layout_();
        }
        {
            auto phase = profile_scope_( "mpi_alltoallw" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "mpi_alltoallw", [&]() {
                    line_comm_info_.alltoallw(
                        static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
                        backward_sdispls_w_.data(), backward_sendtypes_w_.data(),
                        static_cast<void *>( recv_buffer_.raw_ptr() ), backward_recvcounts_w_.data(),
                        backward_rdispls_w_.data(), backward_recvtypes_w_.data()
                    );
                }
            );
        }
        {
            auto phase = profile_scope_( "unpack_recv" );
            time_mpi_transpose_4d_substage<runtime_api_t>(
                stage_timer_, stage_timing_direction_, "unpack", [&]() {
                    for ( int p = 0; p < line_comm_info_.num_procs; ++p )
                        unpack_backward_chunk_( p, out );
                }
            );
        }
#endif
    }

private:
    MPIComm            mpi_;
    Log                log_;
    profiler_t        *profiler_        = nullptr;
    memory_profiler_t *memory_profiler_ = nullptr;
    std::string        memory_profile_prefix_;
    mpi_transpose_4d_stage_timer stage_timer_;
    const char                  *stage_timing_direction_ = "unknown";

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
    bool                               slab_native_batched_peer_kernels_enabled_ = false;
    bool                               slab_native_tensor_coalesced_kernels_enabled_ = false;
    bool                               slab_native_vector4_kernels_enabled_ = false;
    bool                               slab_native_tiled_kernels_enabled_ = false;
    bool                               native_xzwy_direct_layout_enabled_ = false;
    bool                               native_xzwy_chunked_transport_enabled_ = false;
    std::size_t                        native_xzwy_chunk_bytes_ = static_cast<std::size_t>( 1 ) << 30;
    int                                native_xzwy_chunk_window_ = 0;
    bool                               native_xzwy_compact_staging_enabled_ = false;
    bool                               slab_native_wz_communication_layout_enabled_ = false;
    bool                               p2p_slab_native_unpack_ = false;
    contiguous_buf_t                   send_buffer_;
    contiguous_buf_t                   recv_buffer_;
    std::size_t                        send_buffer_elems_      = 0;
    std::size_t                        recv_buffer_elems_      = 0;
    host_buf_t                         host_send_buffer_;
    host_buf_t                         host_recv_buffer_;
    std::size_t                        host_send_buffer_elems_ = 0;
    std::size_t                        host_recv_buffer_elems_ = 0;
    bool                               use_external_work_area_ = false;
    void                              *external_work_area_     = nullptr;
    bool                               use_external_host_work_area_ = false;
    void                              *external_host_work_area_     = nullptr;
    for_each_t                         for_each_;
    std::vector<mpi_request_t>         send_requests_;
    std::vector<mpi_request_t>         recv_requests_;
    mpi_dtype_t                        mpi_value_type_        = mpi_dtype_t();
    bool                               mpi_value_type_inited_ = false;

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
    std::vector<std::size_t> native_xzwy_compact_peer_offsets_;
    bool                     forward_alltoallw_layout_ready_  = false;
    bool                     backward_alltoallw_layout_ready_ = false;

    std::size_t max_buffer_elems_ = 0;
};

template <class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI>
template <class ArrayOut>
void mpi_transpose_4d_same_xw<ValueType, Backend, MPIComm, Log, RuntimeAPI>::forward_alltoallv_slab_native_(
    ArrayOut &out
)
{
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
    auto scope = profile_scope_( "forward_alltoallv_slab_native" );
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
        copy_recv_buffer_from_host_();
    }
    {
        auto phase = profile_scope_( "unpack_recv" );
        unpack_forward_slab_native_all_( out );
    }
#else
    auto scope = profile_scope_( "forward_alltoallv_slab_native" );
    {
        auto phase = profile_scope_( "mpi_alltoallv" );
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "mpi_alltoallv", [&]() {
                line_comm_info_.alltoallv(
                    static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_.data(),
                    forward_sdispls_.data(), mpi_value_type_, static_cast<void *>( recv_buffer_.raw_ptr() ),
                    forward_recvcounts_.data(), forward_rdispls_.data(), mpi_value_type_
                );
            }
        );
    }
    {
        auto phase = profile_scope_( "unpack_recv" );
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "unpack", [&]() {
                unpack_forward_slab_native_all_( out );
            }
        );
    }
#endif
}

template <class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI>
template <class ArrayOut>
void mpi_transpose_4d_same_xw<ValueType, Backend, MPIComm, Log, RuntimeAPI>::forward_alltoallw_slab_native_(
    ArrayOut &out
)
{
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
    auto scope = profile_scope_( "forward_alltoallw_slab_native" );
    {
        auto phase = profile_scope_( "prepare_alltoallw_layout" );
        ensure_forward_alltoallw_layout_();
    }
    {
        auto phase = profile_scope_( "stage_send_to_host" );
        copy_send_buffer_to_host_();
    }
    {
        auto phase = profile_scope_( "mpi_alltoallw" );
        line_comm_info_.alltoallw(
            static_cast<const void *>( host_send_buffer_.raw_ptr() ), forward_sendcounts_w_.data(),
            forward_sdispls_w_.data(), forward_sendtypes_w_.data(),
            static_cast<void *>( host_recv_buffer_.raw_ptr() ), forward_recvcounts_w_.data(),
            forward_rdispls_w_.data(), forward_recvtypes_w_.data()
        );
    }
    {
        auto phase = profile_scope_( "stage_recv_to_device" );
        copy_recv_buffer_from_host_();
    }
    {
        auto phase = profile_scope_( "unpack_recv" );
        unpack_forward_slab_native_all_( out );
    }
#else
    auto scope = profile_scope_( "forward_alltoallw_slab_native" );
    {
        auto phase = profile_scope_( "prepare_alltoallw_layout" );
        ensure_forward_alltoallw_layout_();
    }
    {
        auto phase = profile_scope_( "mpi_alltoallw" );
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "mpi_alltoallw", [&]() {
                line_comm_info_.alltoallw(
                    static_cast<const void *>( send_buffer_.raw_ptr() ), forward_sendcounts_w_.data(),
                    forward_sdispls_w_.data(), forward_sendtypes_w_.data(),
                    static_cast<void *>( recv_buffer_.raw_ptr() ), forward_recvcounts_w_.data(),
                    forward_rdispls_w_.data(), forward_recvtypes_w_.data()
                );
            }
        );
    }
    {
        auto phase = profile_scope_( "unpack_recv" );
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "unpack", [&]() {
                unpack_forward_slab_native_all_( out );
            }
        );
    }
#endif
}

template <class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI>
template <class ArrayOut>
void mpi_transpose_4d_same_xw<ValueType, Backend, MPIComm, Log, RuntimeAPI>::backward_alltoallv_slab_native_(
    ArrayOut &out
)
{
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
    auto scope = profile_scope_( "backward_alltoallv_slab_native" );
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
        copy_recv_buffer_from_host_();
    }
    {
        auto phase = profile_scope_( "unpack_recv" );
        unpack_backward_slab_native_all_( out );
    }
#else
    auto scope = profile_scope_( "backward_alltoallv_slab_native" );
    {
        auto phase = profile_scope_( "mpi_alltoallv" );
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "mpi_alltoallv", [&]() {
                line_comm_info_.alltoallv(
                    static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_.data(),
                    backward_sdispls_.data(), mpi_value_type_, static_cast<void *>( recv_buffer_.raw_ptr() ),
                    backward_recvcounts_.data(), backward_rdispls_.data(), mpi_value_type_
                );
            }
        );
    }
    {
        auto phase = profile_scope_( "unpack_recv" );
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "unpack", [&]() {
                unpack_backward_slab_native_all_( out );
            }
        );
    }
#endif
}

template <class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI>
template <class ArrayOut>
void mpi_transpose_4d_same_xw<ValueType, Backend, MPIComm, Log, RuntimeAPI>::backward_alltoallw_slab_native_(
    ArrayOut &out
)
{
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
    auto scope = profile_scope_( "backward_alltoallw_slab_native" );
    {
        auto phase = profile_scope_( "prepare_alltoallw_layout" );
        ensure_backward_alltoallw_layout_();
    }
    {
        auto phase = profile_scope_( "stage_send_to_host" );
        copy_send_buffer_to_host_();
    }
    {
        auto phase = profile_scope_( "mpi_alltoallw" );
        line_comm_info_.alltoallw(
            static_cast<const void *>( host_send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
            backward_sdispls_w_.data(), backward_sendtypes_w_.data(),
            static_cast<void *>( host_recv_buffer_.raw_ptr() ), backward_recvcounts_w_.data(),
            backward_rdispls_w_.data(), backward_recvtypes_w_.data()
        );
    }
    {
        auto phase = profile_scope_( "stage_recv_to_device" );
        copy_recv_buffer_from_host_();
    }
    {
        auto phase = profile_scope_( "unpack_recv" );
        unpack_backward_slab_native_all_( out );
    }
#else
    auto scope = profile_scope_( "backward_alltoallw_slab_native" );
    {
        auto phase = profile_scope_( "prepare_alltoallw_layout" );
        ensure_backward_alltoallw_layout_();
    }
    {
        auto phase = profile_scope_( "mpi_alltoallw" );
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "mpi_alltoallw", [&]() {
                line_comm_info_.alltoallw(
                    static_cast<const void *>( send_buffer_.raw_ptr() ), backward_sendcounts_w_.data(),
                    backward_sdispls_w_.data(), backward_sendtypes_w_.data(),
                    static_cast<void *>( recv_buffer_.raw_ptr() ), backward_recvcounts_w_.data(),
                    backward_rdispls_w_.data(), backward_recvtypes_w_.data()
                );
            }
        );
    }
    {
        auto phase = profile_scope_( "unpack_recv" );
        time_mpi_transpose_4d_substage<runtime_api_t>(
            stage_timer_, stage_timing_direction_, "unpack", [&]() {
                unpack_backward_slab_native_all_( out );
            }
        );
    }
#endif
}


} // namespace detail
} // namespace fftm

#endif // __FFTM_DETAIL_MPI_TRANSPOSE_4D_SAME_XW_H__
