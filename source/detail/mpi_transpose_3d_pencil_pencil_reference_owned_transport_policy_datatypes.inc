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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        return use_persistent_p2p_ && cuda_aware_byte_p2p_enabled_();
#else
        return false;
#endif
    }

    bool direct_backward_receive_enabled_() const
    {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        return direct_p2p_cuda_aware_ && use_direct_backward_receive_;
#else
        return false;
#endif
    }

	    bool direct_forward_value_receive_enabled_() const
	    {
	#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
	        return direct_p2p_cuda_aware_ && use_direct_backward_receive_ && !cuda_aware_byte_p2p_enabled_();
	#else
	        return false;
	#endif
	    }

    bool direct_reference_opt1_forward_byte_receive_enabled_() const
    {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
            throw std::logic_error( "owned deferred send completion requires device-aware MPI" );
#else
            if ( !reference_parity_enabled_ || mode_ != mpi_transpose_3d_mode::p2p_waitany ||
                 !use_p2p_byte_transfer_ || use_persistent_p2p_ )
            {
                throw std::logic_error(
                    "owned deferred send completion requires reference-parity, p2p-waitany, device-aware byte "
                    "transfers, and nonpersistent requests"
                );
            }
#endif
        }
        if ( !reference_parity_enabled_ )
            return;
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        return use_contiguous_forward_byte_send_ && use_stable_forward_byte_send_buffer_ &&
               cuda_aware_byte_p2p_enabled_();
#else
        return false;
#endif
    }

    bool physical_forward_peer_exchange_enabled_() const
    {
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
        /*
         * Experimental reference-like forward path:
         * - send directly from peer-contiguous forward stage blocks;
         * - use value-count MPI requests from/to contiguous buffers;
         * - keep final stage-layout reshaping as explicit runtime_api_t copies.
         *
         * This deliberately avoids direct strided device-aware receives, which
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
