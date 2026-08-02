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
