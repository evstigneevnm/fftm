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
