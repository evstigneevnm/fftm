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
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
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
