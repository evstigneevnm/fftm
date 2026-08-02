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

#include "mpi_transpose_3d_pencil_pencil_reference_owned_diagnostics_api.inc"

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
