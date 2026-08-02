    std::size_t legacy_first_forward_send_elems_( int target_j ) const
    {
        return nx_local_ * ny_input_local_ * transpose1_dim_.size_z[target_j];
    }

    std::size_t legacy_first_forward_send_offset_( int target_j ) const
    {
        return transpose1_dim_.start_z[target_j] * nx_local_ * ny_input_local_;
    }

    std::size_t legacy_first_forward_recv_elems_( int source_j ) const
    {
        return nx_local_ * half_input_dim_.size_y[source_j] * nz_local_;
    }

    std::size_t legacy_first_forward_recv_offset_( int source_j ) const
    {
        return nx_local_ * nz_local_ * half_input_dim_.start_y[source_j];
    }

    std::size_t legacy_first_backward_send_elems_( int target_j ) const
    {
        return nx_local_ * half_input_dim_.size_y[target_j] * nz_local_;
    }

    std::size_t legacy_first_backward_send_offset_( int target_j ) const
    {
        return legacy_first_forward_recv_offset_( target_j );
    }

    std::size_t legacy_first_backward_recv_elems_( int source_j ) const
    {
        return nx_local_ * ny_input_local_ * transpose1_dim_.size_z[source_j];
    }

    std::size_t legacy_first_backward_recv_offset_( int source_j ) const
    {
        return legacy_first_forward_send_offset_( source_j );
    }

    std::size_t first_forward_send_elems_( int target_j ) const
    {
        return native_or_legacy_schedule_elems_(
            first_forward_send_schedule_, target_j, legacy_first_forward_send_elems_( target_j )
        );
    }

    std::size_t first_forward_send_offset_( int target_j ) const
    {
        return native_or_legacy_schedule_offset_(
            first_forward_send_schedule_, target_j, legacy_first_forward_send_offset_( target_j )
        );
    }

    std::size_t first_forward_recv_elems_( int source_j ) const
    {
        return native_or_legacy_schedule_elems_(
            first_forward_recv_schedule_, source_j, legacy_first_forward_recv_elems_( source_j )
        );
    }

    std::size_t first_forward_recv_offset_( int source_j ) const
    {
        return native_or_legacy_schedule_offset_(
            first_forward_recv_schedule_, source_j, legacy_first_forward_recv_offset_( source_j )
        );
    }

    template <class Stage0Complex3>
    const value_type *native_forward_first_send_source_ptr_( int target_j, const Stage0Complex3 &in ) const
    {
        validate_native_execution_slot_(
            first_forward_send_schedule_, target_j, myid_j_, "first_forward_send_source"
        );
        return in.raw_ptr() + first_forward_send_offset_( target_j );
    }

    value_type *native_forward_first_device_send_buffer_ptr_( int target_j )
    {
        validate_native_execution_slot_(
            first_forward_send_schedule_, target_j, myid_j_, "first_forward_device_send_buffer"
        );
        return send_buffer_.raw_ptr() + first_forward_send_offset_( target_j );
    }

    value_type *native_forward_first_host_send_buffer_ptr_( int target_j )
    {
        validate_native_execution_slot_(
            first_forward_send_schedule_, target_j, myid_j_, "first_forward_host_send_buffer"
        );
        return host_send_buffer_.raw_ptr() + first_forward_send_offset_( target_j );
    }

    value_type *native_forward_first_device_recv_buffer_ptr_( int source_j )
    {
        validate_native_execution_slot_(
            first_forward_recv_schedule_, source_j, source_j, "first_forward_device_recv_buffer"
        );
        return recv_buffer_.raw_ptr() + first_forward_recv_offset_( source_j );
    }

    value_type *native_forward_first_host_recv_buffer_ptr_( int source_j )
    {
        validate_native_execution_slot_(
            first_forward_recv_schedule_, source_j, source_j, "first_forward_host_recv_buffer"
        );
        return host_recv_buffer_.raw_ptr() + first_forward_recv_offset_( source_j );
    }

    template <class Stage1Complex3>
    value_type *native_forward_first_recv_target_ptr_( int source_j, Stage1Complex3 &out )
    {
        validate_native_execution_slot_(
            first_forward_recv_schedule_, source_j, source_j, "first_forward_recv_target"
        );
        if ( forward_receive_lands_in_stage_() )
            return out.raw_ptr() + half_input_dim_.start_y[source_j];
        return native_forward_first_device_recv_buffer_ptr_( source_j );
    }

    std::size_t first_backward_send_elems_( int target_j ) const
    {
        return native_or_legacy_schedule_elems_(
            first_backward_send_schedule_, target_j, legacy_first_backward_send_elems_( target_j )
        );
    }

    std::size_t first_backward_send_offset_( int target_j ) const
    {
        return native_or_legacy_schedule_offset_(
            first_backward_send_schedule_, target_j, legacy_first_backward_send_offset_( target_j )
        );
    }

    std::size_t first_backward_recv_elems_( int source_j ) const
    {
        return native_or_legacy_schedule_elems_(
            first_backward_recv_schedule_, source_j, legacy_first_backward_recv_elems_( source_j )
        );
    }

    std::size_t first_backward_recv_offset_( int source_j ) const
    {
        return native_or_legacy_schedule_offset_(
            first_backward_recv_schedule_, source_j, legacy_first_backward_recv_offset_( source_j )
        );
    }

    std::size_t first_backward_packed_recv_offset_( int source_j ) const
    {
        return first_backward_recv_offset_( source_j );
    }

    int first_backward_send_mpi_tag_( int target_j ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( first_backward_send_schedule_, target_j, "first_backward_send_tag" );
        if ( slot != nullptr )
            return slot->mpi_tag;
        return target_j;
    }

    int first_backward_recv_mpi_tag_( int source_j ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( first_backward_recv_schedule_, source_j, "first_backward_recv_tag" );
        if ( slot != nullptr )
            return slot->mpi_tag;
        return myid_j_;
    }

    value_type *native_backward_first_device_send_buffer_ptr_( int target_j )
    {
        validate_native_execution_slot_(
            first_backward_send_schedule_, target_j, first_backward_send_mpi_tag_( target_j ),
            "first_backward_device_send_buffer"
        );
        return send_buffer_.raw_ptr() + first_backward_send_offset_( target_j );
    }

    value_type *native_backward_first_host_send_buffer_ptr_( int target_j )
    {
        validate_native_execution_slot_(
            first_backward_send_schedule_, target_j, first_backward_send_mpi_tag_( target_j ),
            "first_backward_host_send_buffer"
        );
        return host_send_buffer_.raw_ptr() + first_backward_send_offset_( target_j );
    }

    template <class Stage0Complex3>
    value_type *native_backward_first_recv_output_ptr_( int source_j, Stage0Complex3 &out )
    {
        validate_native_execution_slot_(
            first_backward_recv_schedule_, source_j, first_backward_recv_mpi_tag_( source_j ),
            "first_backward_recv_output"
        );
        return out.raw_ptr() + first_backward_recv_offset_( source_j );
    }

    value_type *native_backward_first_device_recv_buffer_ptr_( int source_j )
    {
        validate_native_execution_slot_(
            first_backward_recv_schedule_, source_j, first_backward_recv_mpi_tag_( source_j ),
            "first_backward_device_recv_buffer"
        );
        return recv_buffer_.raw_ptr() + first_backward_packed_recv_offset_( source_j );
    }

    value_type *native_backward_first_host_recv_buffer_ptr_( int source_j )
    {
        validate_native_execution_slot_(
            first_backward_recv_schedule_, source_j, first_backward_recv_mpi_tag_( source_j ),
            "first_backward_host_recv_buffer"
        );
        return host_recv_buffer_.raw_ptr() + first_backward_packed_recv_offset_( source_j );
    }

    template <class Stage0Complex3>
    value_type *native_backward_first_recv_target_ptr_( int source_j, Stage0Complex3 &out )
    {
        validate_native_execution_slot_(
            first_backward_recv_schedule_, source_j, first_backward_recv_mpi_tag_( source_j ),
            "first_backward_recv_target"
        );
        if ( backward_receive_lands_in_output_() )
            return native_backward_first_recv_output_ptr_( source_j, out );
        return native_backward_first_device_recv_buffer_ptr_( source_j );
    }
