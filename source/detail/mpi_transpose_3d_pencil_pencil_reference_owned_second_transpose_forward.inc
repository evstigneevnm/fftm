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
