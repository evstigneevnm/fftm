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
