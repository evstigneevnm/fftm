#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_PLAN_SCHEDULE_INC__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_PLAN_SCHEDULE_INC__

    static bool partitions_equal_( const partition_t &lhs, const partition_t &rhs )
    {
        return lhs.size_x == rhs.size_x && lhs.size_y == rhs.size_y && lhs.size_z == rhs.size_z &&
               lhs.size_w == rhs.size_w && lhs.start_x == rhs.start_x && lhs.start_y == rhs.start_y &&
               lhs.start_z == rhs.start_z && lhs.start_w == rhs.start_w;
    }

    static void require_partitions_equal_(
        const partition_t &lhs, const partition_t &rhs, const std::string &context
    )
    {
        if ( !partitions_equal_( lhs, rhs ) )
            throw std::logic_error( "owned reference native pencil plan-state partition mismatch: " + context );
    }

    static void validate_peer_order_( const std::vector<int> &order, int num_procs, const std::string &context )
    {
        if ( static_cast<int>( order.size() ) != std::max( 0, num_procs - 1 ) )
        {
            throw std::logic_error(
                "owned reference native pencil plan-state peer count mismatch for " + context
            );
        }
        std::vector<int> seen( num_procs, 0 );
        for ( const int peer : order )
        {
            if ( peer < 0 || peer >= num_procs )
            {
                throw std::logic_error(
                    "owned reference native pencil plan-state peer out of range for " + context
                );
            }
            if ( seen[peer] != 0 )
            {
                throw std::logic_error(
                    "owned reference native pencil plan-state duplicate peer for " + context
                );
            }
            seen[peer] = 1;
        }
    }

    void validate_native_plan_state_inputs_(
        const plan_state_t &plan_state, const partition_t &expected_half_input_dim,
        const partition_t &expected_transpose1_dim, const partition_t &expected_output_dim, int expected_myid_i,
        int expected_myid_j
    ) const
    {
        if ( !plan_state.is_inited() )
            throw std::logic_error( "owned reference native pencil plan-state is not initialized" );
        if ( plan_state.rank_i() != expected_myid_i || plan_state.rank_j() != expected_myid_j )
            throw std::logic_error( "owned reference native pencil plan-state rank mismatch" );
        require_partitions_equal_(
            plan_state.half_input_partition(), expected_half_input_dim, "half-input partition"
        );
        require_partitions_equal_( plan_state.stage1_partition(), expected_transpose1_dim, "stage1 partition" );
        require_partitions_equal_( plan_state.output_partition(), expected_output_dim, "output partition" );
    }

    void validate_native_plan_state_after_init_() const
    {
        if ( !native_plan_state_bound_ )
            return;
        if ( native_plan_state_.rank_i() != myid_i_ || native_plan_state_.rank_j() != myid_j_ )
            throw std::logic_error( "owned reference native pencil plan-state active rank mismatch" );
        require_partitions_equal_( native_plan_state_.half_input_partition(), half_input_dim_, "active half-input" );
        require_partitions_equal_( native_plan_state_.stage1_partition(), transpose1_dim_, "active stage1" );
        require_partitions_equal_( native_plan_state_.output_partition(), output_dim_, "active output" );
        validate_peer_order_( row_comm_order_, row_comm_info_.num_procs, "row communicator" );
        validate_peer_order_( line_comm_order_, line_comm_info_.num_procs, "line communicator" );
    }

    void clear_native_plan_schedule_slots_()
    {
        first_forward_send_schedule_.clear();
        first_forward_recv_schedule_.clear();
        first_backward_send_schedule_.clear();
        first_backward_recv_schedule_.clear();
        second_forward_send_schedule_.clear();
        second_forward_recv_schedule_.clear();
        second_backward_send_schedule_.clear();
        second_backward_recv_schedule_.clear();
    }

    static std::string native_schedule_context_(
        const char *stage, int transpose, int direction, int operation, int peer
    )
    {
        std::ostringstream ss;
        ss << stage << " transpose=" << transpose << " direction=" << direction
           << " operation=" << operation << " peer=" << peer;
        return ss.str();
    }

    static std::size_t checked_nonnegative_size_( long long value, const std::string &context )
    {
        if ( value < 0 )
            throw std::logic_error( "owned reference native pencil schedule has negative value for " + context );
        return static_cast<std::size_t>( value );
    }

    std::size_t schedule_bytes_to_elems_( long long bytes, const std::string &context ) const
    {
        const std::size_t value_bytes = sizeof( value_type );
        const std::size_t byte_count  = checked_nonnegative_size_( bytes, context + " bytes" );
        if ( byte_count % value_bytes != 0 )
            throw std::logic_error( "owned reference native pencil schedule byte count is not value-aligned for " + context );
        return byte_count / value_bytes;
    }

    void set_native_schedule_slot_(
        std::vector<native_schedule_slot> &slots, const ::fftm::detail::pencil_pencil_schedule_record &record,
        int expected_transpose, int expected_direction, int expected_operation, const char *stage
    ) const
    {
        const int peer = static_cast<int>( checked_nonnegative_size_( record.peer, std::string( stage ) + " peer" ) );
        const std::string context =
            native_schedule_context_( stage, expected_transpose, expected_direction, expected_operation, peer );

        if ( record.transpose != expected_transpose || record.direction != expected_direction ||
             record.operation != expected_operation )
        {
            throw std::logic_error( "owned reference native pencil schedule routed an unexpected record for " + context );
        }
        if ( peer < 0 || peer >= static_cast<int>( slots.size() ) )
            throw std::logic_error( "owned reference native pencil schedule peer is out of range for " + context );
        if ( slots[peer].valid )
            throw std::logic_error( "owned reference native pencil schedule duplicate peer record for " + context );

        native_schedule_slot slot;
        slot.offset_elems =
            checked_nonnegative_size_( record.offset_elems, context + " offset_elems" );
        slot.elems    = schedule_bytes_to_elems_( record.bytes, context );
        slot.mpi_peer = static_cast<int>( checked_nonnegative_size_( record.mpi_peer, context + " mpi_peer" ) );
        slot.mpi_tag  = static_cast<int>( checked_nonnegative_size_( record.mpi_tag, context + " mpi_tag" ) );
        slot.valid    = true;
        if ( slot.mpi_peer < 0 || slot.mpi_peer >= static_cast<int>( slots.size() ) )
            throw std::logic_error( "owned reference native pencil schedule MPI peer is out of range for " + context );
        slots[peer] = slot;
    }

    void validate_native_schedule_slot_(
        const std::vector<native_schedule_slot> &slots, int peer, std::size_t expected_offset,
        std::size_t expected_elems, const char *stage
    ) const
    {
        if ( peer < 0 || peer >= static_cast<int>( slots.size() ) || !slots[peer].valid )
        {
            throw std::logic_error(
                "owned reference native pencil schedule missing peer metadata for " + std::string( stage )
            );
        }
        const native_schedule_slot &slot = slots[peer];
        if ( slot.offset_elems != expected_offset || slot.elems != expected_elems )
        {
            std::ostringstream ss;
            ss << "owned reference native pencil schedule count/offset mismatch for " << stage << " peer=" << peer
               << " expected_offset=" << expected_offset << " schedule_offset=" << slot.offset_elems
               << " expected_elems=" << expected_elems << " schedule_elems=" << slot.elems;
            throw std::logic_error( ss.str() );
        }
        if ( slot.mpi_peer != peer )
        {
            throw std::logic_error(
                "owned reference native pencil schedule MPI peer mismatch for " + std::string( stage )
            );
        }
    }

    void validate_native_plan_first_schedule_slots_() const
    {
        for ( const int peer : row_comm_order_ )
        {
            validate_native_schedule_slot_(
                first_forward_recv_schedule_, peer, legacy_first_forward_recv_offset_( peer ),
                legacy_first_forward_recv_elems_( peer ), "first_forward_recv"
            );
            validate_native_schedule_slot_(
                first_forward_send_schedule_, peer, legacy_first_forward_send_offset_( peer ),
                legacy_first_forward_send_elems_( peer ), "first_forward_send"
            );
            validate_native_schedule_slot_(
                first_backward_recv_schedule_, peer, legacy_first_backward_recv_offset_( peer ),
                legacy_first_backward_recv_elems_( peer ), "first_backward_recv"
            );
            validate_native_schedule_slot_(
                first_backward_send_schedule_, peer, legacy_first_backward_send_offset_( peer ),
                legacy_first_backward_send_elems_( peer ), "first_backward_send"
            );
        }
    }

    void validate_native_plan_second_schedule_slots_() const
    {
        for ( const int peer : line_comm_order_ )
        {
            validate_native_schedule_slot_(
                second_forward_recv_schedule_, peer, legacy_second_forward_recv_offset_( peer ),
                legacy_second_forward_recv_elems_( peer ), "second_forward_recv"
            );
            validate_native_schedule_slot_(
                second_forward_send_schedule_, peer, legacy_second_forward_send_offset_( peer ),
                legacy_second_forward_send_elems_( peer ), "second_forward_send"
            );
            validate_native_schedule_slot_(
                second_backward_recv_schedule_, peer, legacy_second_backward_recv_offset_( peer ),
                legacy_second_backward_recv_elems_( peer ), "second_backward_recv"
            );
            validate_native_schedule_slot_(
                second_backward_send_schedule_, peer, legacy_second_backward_send_offset_( peer ),
                legacy_second_backward_send_elems_( peer ), "second_backward_send"
            );
        }
    }

    void build_native_plan_schedule_slots_()
    {
        clear_native_plan_schedule_slots_();
        if ( !native_plan_state_bound_ )
            return;

        first_forward_send_schedule_.assign( row_comm_info_.num_procs, native_schedule_slot() );
        first_forward_recv_schedule_.assign( row_comm_info_.num_procs, native_schedule_slot() );
        first_backward_send_schedule_.assign( row_comm_info_.num_procs, native_schedule_slot() );
        first_backward_recv_schedule_.assign( row_comm_info_.num_procs, native_schedule_slot() );
        second_forward_send_schedule_.assign( line_comm_info_.num_procs, native_schedule_slot() );
        second_forward_recv_schedule_.assign( line_comm_info_.num_procs, native_schedule_slot() );
        second_backward_send_schedule_.assign( line_comm_info_.num_procs, native_schedule_slot() );
        second_backward_recv_schedule_.assign( line_comm_info_.num_procs, native_schedule_slot() );

        const std::vector<::fftm::detail::pencil_pencil_schedule_record> records =
            native_plan_state_.local_schedule_records();
        for ( const auto &record : records )
        {
            if ( record.pidx_i != myid_i_ || record.pidx_j != myid_j_ )
                throw std::logic_error( "owned reference native pencil schedule rank-index mismatch" );

            if ( record.transpose == 1 && record.direction == 0 && record.operation == 0 )
            {
                set_native_schedule_slot_( first_forward_recv_schedule_, record, 1, 0, 0, "first_forward_recv" );
            }
            else if ( record.transpose == 1 && record.direction == 0 && record.operation == 1 )
            {
                set_native_schedule_slot_( first_forward_send_schedule_, record, 1, 0, 1, "first_forward_send" );
            }
            else if ( record.transpose == 1 && record.direction == 1 && record.operation == 0 )
            {
                set_native_schedule_slot_( first_backward_recv_schedule_, record, 1, 1, 0, "first_backward_recv" );
            }
            else if ( record.transpose == 1 && record.direction == 1 && record.operation == 1 )
            {
                set_native_schedule_slot_( first_backward_send_schedule_, record, 1, 1, 1, "first_backward_send" );
            }
            else if ( record.transpose == 2 && record.direction == 0 && record.operation == 0 )
            {
                set_native_schedule_slot_( second_forward_recv_schedule_, record, 2, 0, 0, "second_forward_recv" );
            }
            else if ( record.transpose == 2 && record.direction == 0 && record.operation == 1 )
            {
                set_native_schedule_slot_( second_forward_send_schedule_, record, 2, 0, 1, "second_forward_send" );
            }
            else if ( record.transpose == 2 && record.direction == 1 && record.operation == 0 )
            {
                set_native_schedule_slot_( second_backward_recv_schedule_, record, 2, 1, 0, "second_backward_recv" );
            }
            else if ( record.transpose == 2 && record.direction == 1 && record.operation == 1 )
            {
                set_native_schedule_slot_( second_backward_send_schedule_, record, 2, 1, 1, "second_backward_send" );
            }
            else
            {
                throw std::logic_error( "owned reference native pencil schedule unsupported transpose record" );
            }
        }
        validate_native_plan_first_schedule_slots_();
        validate_native_plan_second_schedule_slots_();
    }

    bool native_first_schedule_active_() const
    {
        return native_plan_state_bound_ && !first_forward_send_schedule_.empty();
    }

    bool native_second_schedule_active_() const
    {
        return native_plan_state_bound_ && !second_forward_send_schedule_.empty();
    }

    bool native_forward_first_execution_active_() const
    {
        return native_first_schedule_active_();
    }

    bool native_forward_second_execution_active_() const
    {
        return native_second_schedule_active_();
    }

    bool native_forward_second_facade_active_() const
    {
        return native_forward_second_execution_active_();
    }

    bool native_forward_second_peer_loop_active_() const
    {
        return native_forward_second_facade_active_() && peer_paired_large_count_byte_schedule_enabled_();
    }

    bool native_backward_second_execution_active_() const
    {
        return native_second_schedule_active_();
    }

    bool native_backward_second_facade_active_() const
    {
        return native_backward_second_execution_active_();
    }

    bool native_backward_second_peer_loop_active_() const
    {
        return use_native_backward_second_peer_loop_ && native_backward_second_facade_active_() &&
               peer_paired_large_count_byte_schedule_enabled_();
    }

    bool native_backward_first_execution_active_() const
    {
        return native_first_schedule_active_();
    }

    bool native_schedule_active_( const std::vector<native_schedule_slot> &slots ) const
    {
        return native_plan_state_bound_ && !slots.empty();
    }

    const native_schedule_slot *native_schedule_slot_for_execution_(
        const std::vector<native_schedule_slot> &slots, int peer, const char *stage
    ) const
    {
        if ( !native_schedule_active_( slots ) )
            return nullptr;
        if ( peer < 0 || peer >= static_cast<int>( slots.size() ) || !slots[peer].valid )
        {
            throw std::logic_error(
                "owned reference native pencil execution missing peer metadata for " + std::string( stage )
            );
        }
        return &slots[peer];
    }

    void validate_native_execution_slot_(
        const std::vector<native_schedule_slot> &slots, int peer, int expected_tag, const char *stage
    ) const
    {
        const native_schedule_slot *slot = native_schedule_slot_for_execution_( slots, peer, stage );
        if ( slot == nullptr )
            return;
        if ( slot->mpi_peer != peer || slot->mpi_tag != expected_tag )
        {
            std::ostringstream ss;
            ss << "owned reference native pencil execution peer/tag mismatch for " << stage << " peer=" << peer
               << " schedule_peer=" << slot->mpi_peer << " expected_tag=" << expected_tag
               << " schedule_tag=" << slot->mpi_tag;
            throw std::logic_error( ss.str() );
        }
    }

    std::size_t native_or_legacy_schedule_elems_(
        const std::vector<native_schedule_slot> &slots, int peer, std::size_t legacy_elems
    ) const
    {
        if ( !native_schedule_active_( slots ) || peer < 0 || peer >= static_cast<int>( slots.size() ) ||
             !slots[peer].valid )
        {
            return legacy_elems;
        }
        return slots[peer].elems;
    }

    std::size_t native_or_legacy_schedule_offset_(
        const std::vector<native_schedule_slot> &slots, int peer, std::size_t legacy_offset
    ) const
    {
        if ( !native_schedule_active_( slots ) || peer < 0 || peer >= static_cast<int>( slots.size() ) ||
             !slots[peer].valid )
        {
            return legacy_offset;
        }
        return slots[peer].offset_elems;
    }

#endif
