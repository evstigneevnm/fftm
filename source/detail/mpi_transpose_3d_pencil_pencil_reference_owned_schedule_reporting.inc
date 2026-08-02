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
        if ( !print_schedule_ )
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
