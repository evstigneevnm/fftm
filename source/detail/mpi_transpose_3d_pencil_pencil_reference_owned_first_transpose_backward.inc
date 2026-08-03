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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
            if ( cuda_aware_byte_p2p_enabled_() )
                waitall_vector_( row_comm_info_, byte_recv_requests, "owned first byte backward recv requests" );
            else
#endif
                row_comm_info_.waitall( row_size, row_recv_requests_.data() );
        }
        if ( !backward_receive_lands_in_output_() )
        {
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
            auto phase = profile_scope_( "first_backward/stage_recv_to_device" );
#else
            auto phase = profile_scope_( "first_backward/unpack_recv" );
#endif
            for ( const int p : row_comm_order_ )
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
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

#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
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
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
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
