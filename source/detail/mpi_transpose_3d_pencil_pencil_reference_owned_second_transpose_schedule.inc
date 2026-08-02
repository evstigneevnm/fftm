    std::size_t legacy_second_forward_send_elems_( int target_i ) const
    {
        return nx_local_ * output_dim_.size_y[target_i] * nz_local_;
    }

    std::size_t legacy_second_forward_send_offset_( int target_i ) const
    {
        return nx_local_ * nz_local_ * output_dim_.start_y[target_i];
    }

    std::size_t legacy_second_forward_recv_elems_( int source_i ) const
    {
        return transpose1_dim_.size_x[source_i] * ny_output_local_ * nz_local_;
    }

    std::size_t legacy_second_forward_recv_offset_( int source_i ) const
    {
        std::size_t offset = 0;
        for ( int p = 0; p < source_i; ++p )
        {
            offset += legacy_second_forward_recv_elems_( p );
        }
        return offset;
    }

    std::size_t legacy_second_backward_send_elems_( int target_i ) const
    {
        return transpose1_dim_.size_x[target_i] * ny_output_local_ * nz_local_;
    }

    std::size_t legacy_second_backward_send_offset_( int target_i ) const
    {
        return legacy_second_forward_recv_offset_( target_i );
    }

    std::size_t legacy_second_backward_recv_elems_( int source_i ) const
    {
        return nx_local_ * output_dim_.size_y[source_i] * nz_local_;
    }

    std::size_t legacy_second_backward_recv_offset_( int source_i ) const
    {
        return legacy_second_forward_send_offset_( source_i );
    }

    std::size_t second_forward_send_elems_( int target_i ) const
    {
        return native_or_legacy_schedule_elems_(
            second_forward_send_schedule_, target_i, legacy_second_forward_send_elems_( target_i )
        );
    }

    std::size_t second_forward_send_offset_( int target_i ) const
    {
        return native_or_legacy_schedule_offset_(
            second_forward_send_schedule_, target_i, legacy_second_forward_send_offset_( target_i )
        );
    }

    std::size_t second_forward_recv_elems_( int source_i ) const
    {
        return native_or_legacy_schedule_elems_(
            second_forward_recv_schedule_, source_i, legacy_second_forward_recv_elems_( source_i )
        );
    }

    std::size_t second_forward_recv_offset_( int source_i ) const
    {
        return native_or_legacy_schedule_offset_(
            second_forward_recv_schedule_, source_i, legacy_second_forward_recv_offset_( source_i )
        );
    }

    template <class XFastComplex3>
    const value_type *native_forward_second_send_source_ptr_( int target_i, const XFastComplex3 &in ) const
    {
        validate_native_execution_slot_(
            second_forward_send_schedule_, target_i, myid_i_, "second_forward_send_source"
        );
        return in.raw_ptr() + second_forward_send_offset_( target_i );
    }

    value_type *native_forward_second_device_send_buffer_ptr_( int target_i )
    {
        validate_native_execution_slot_(
            second_forward_send_schedule_, target_i, myid_i_, "second_forward_device_send_buffer"
        );
        return send_buffer_.raw_ptr() + second_forward_send_offset_( target_i );
    }

    value_type *native_forward_second_host_send_buffer_ptr_( int target_i )
    {
        validate_native_execution_slot_(
            second_forward_send_schedule_, target_i, myid_i_, "second_forward_host_send_buffer"
        );
        return host_send_buffer_.raw_ptr() + second_forward_send_offset_( target_i );
    }

    value_type *native_forward_second_device_recv_buffer_ptr_( int source_i )
    {
        validate_native_execution_slot_(
            second_forward_recv_schedule_, source_i, source_i, "second_forward_device_recv_buffer"
        );
        return recv_buffer_.raw_ptr() + second_forward_recv_offset_( source_i );
    }

    value_type *native_forward_second_host_recv_buffer_ptr_( int source_i )
    {
        validate_native_execution_slot_(
            second_forward_recv_schedule_, source_i, source_i, "second_forward_host_recv_buffer"
        );
        return host_recv_buffer_.raw_ptr() + second_forward_recv_offset_( source_i );
    }

    template <class XFFTComplex3>
    value_type *native_forward_second_recv_target_ptr_( int source_i, XFFTComplex3 &out )
    {
        validate_native_execution_slot_(
            second_forward_recv_schedule_, source_i, source_i, "second_forward_recv_target"
        );
        if ( native_forward_second_receive_lands_in_stage_() )
            return out.raw_ptr() + transpose1_dim_.start_x[source_i];
        return native_forward_second_device_recv_buffer_ptr_( source_i );
    }

    const native_schedule_slot &native_forward_second_send_slot_( int target_i, const char *stage ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( second_forward_send_schedule_, target_i, stage );
        if ( slot == nullptr )
            throw std::logic_error( "owned reference native forward-second send slot is not active" );
        return *slot;
    }

    const native_schedule_slot &native_forward_second_recv_slot_( int source_i, const char *stage ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( second_forward_recv_schedule_, source_i, stage );
        if ( slot == nullptr )
            throw std::logic_error( "owned reference native forward-second recv slot is not active" );
        return *slot;
    }

    template <class XFastComplex3>
    const value_type *native_forward_second_send_source_ptr_(
        const native_schedule_slot &slot, const XFastComplex3 &in
    ) const
    {
        return in.raw_ptr() + slot.offset_elems;
    }

    value_type *native_forward_second_device_send_buffer_ptr_( const native_schedule_slot &slot )
    {
        return send_buffer_.raw_ptr() + slot.offset_elems;
    }

    value_type *native_forward_second_host_send_buffer_ptr_( const native_schedule_slot &slot )
    {
        return host_send_buffer_.raw_ptr() + slot.offset_elems;
    }

    value_type *native_forward_second_device_recv_buffer_ptr_( const native_schedule_slot &slot )
    {
        return recv_buffer_.raw_ptr() + slot.offset_elems;
    }

    value_type *native_forward_second_host_recv_buffer_ptr_( const native_schedule_slot &slot )
    {
        return host_recv_buffer_.raw_ptr() + slot.offset_elems;
    }

    template <class XFFTComplex3>
    value_type *native_forward_second_recv_target_ptr_(
        int source_i, const native_schedule_slot &slot, XFFTComplex3 &out
    )
    {
        if ( native_forward_second_receive_lands_in_stage_() )
            return out.raw_ptr() + transpose1_dim_.start_x[source_i];
        return native_forward_second_device_recv_buffer_ptr_( slot );
    }

    std::size_t native_forward_second_recv_x_size_(
        int source_i, const native_schedule_slot &slot, const char *stage
    ) const
    {
        const std::size_t yz = ny_output_local_ * nz_local_;
        if ( yz == 0 || slot.elems % yz != 0 )
        {
            throw std::logic_error(
                "owned reference native forward-second schedule has invalid recv extent for " + std::string( stage )
            );
        }
        const std::size_t x_size = slot.elems / yz;
        if ( x_size != transpose1_dim_.size_x[source_i] )
        {
            std::ostringstream ss;
            ss << "owned reference native forward-second recv extent mismatch for " << stage
               << " peer=" << source_i << " schedule_x=" << x_size
               << " partition_x=" << transpose1_dim_.size_x[source_i];
            throw std::logic_error( ss.str() );
        }
        return x_size;
    }

    std::size_t second_backward_send_elems_( int target_i ) const
    {
        return native_or_legacy_schedule_elems_(
            second_backward_send_schedule_, target_i, legacy_second_backward_send_elems_( target_i )
        );
    }

    std::size_t second_backward_send_offset_( int target_i ) const
    {
        return native_or_legacy_schedule_offset_(
            second_backward_send_schedule_, target_i, legacy_second_backward_send_offset_( target_i )
        );
    }

    std::size_t second_backward_recv_elems_( int source_i ) const
    {
        return native_or_legacy_schedule_elems_(
            second_backward_recv_schedule_, source_i, legacy_second_backward_recv_elems_( source_i )
        );
    }

    std::size_t second_backward_recv_offset_( int source_i ) const
    {
        return native_or_legacy_schedule_offset_(
            second_backward_recv_schedule_, source_i, legacy_second_backward_recv_offset_( source_i )
        );
    }

    std::size_t second_backward_packed_recv_offset_( int source_i ) const
    {
        return second_backward_recv_offset_( source_i );
    }

    value_type *native_backward_second_device_send_buffer_ptr_( int target_i )
    {
        validate_native_execution_slot_(
            second_backward_send_schedule_, target_i, myid_i_, "second_backward_device_send_buffer"
        );
        return send_buffer_.raw_ptr() + second_backward_send_offset_( target_i );
    }

    value_type *native_backward_second_host_send_buffer_ptr_( int target_i )
    {
        validate_native_execution_slot_(
            second_backward_send_schedule_, target_i, myid_i_, "second_backward_host_send_buffer"
        );
        return host_send_buffer_.raw_ptr() + second_backward_send_offset_( target_i );
    }

    template <class XFastComplex3>
    value_type *native_backward_second_recv_output_ptr_( int source_i, XFastComplex3 &out )
    {
        validate_native_execution_slot_(
            second_backward_recv_schedule_, source_i, source_i, "second_backward_recv_output"
        );
        return out.raw_ptr() + second_backward_recv_offset_( source_i );
    }

    value_type *native_backward_second_device_recv_buffer_ptr_( int source_i )
    {
        validate_native_execution_slot_(
            second_backward_recv_schedule_, source_i, source_i, "second_backward_device_recv_buffer"
        );
        return recv_buffer_.raw_ptr() + second_backward_packed_recv_offset_( source_i );
    }

    value_type *native_backward_second_host_recv_buffer_ptr_( int source_i )
    {
        validate_native_execution_slot_(
            second_backward_recv_schedule_, source_i, source_i, "second_backward_host_recv_buffer"
        );
        return host_recv_buffer_.raw_ptr() + second_backward_packed_recv_offset_( source_i );
    }

    template <class XFastComplex3>
    value_type *native_backward_second_recv_target_ptr_( int source_i, XFastComplex3 &out )
    {
        validate_native_execution_slot_(
            second_backward_recv_schedule_, source_i, source_i, "second_backward_recv_target"
        );
        if ( backward_receive_lands_in_output_() )
            return native_backward_second_recv_output_ptr_( source_i, out );
        return native_backward_second_device_recv_buffer_ptr_( source_i );
    }

    const native_schedule_slot &native_backward_second_send_slot_( int target_i, const char *stage ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( second_backward_send_schedule_, target_i, stage );
        if ( slot == nullptr )
            throw std::logic_error( "owned reference native backward-second send slot is not active" );
        return *slot;
    }

    const native_schedule_slot &native_backward_second_recv_slot_( int source_i, const char *stage ) const
    {
        const native_schedule_slot *slot =
            native_schedule_slot_for_execution_( second_backward_recv_schedule_, source_i, stage );
        if ( slot == nullptr )
            throw std::logic_error( "owned reference native backward-second recv slot is not active" );
        return *slot;
    }

    value_type *native_backward_second_device_send_buffer_ptr_( const native_schedule_slot &slot )
    {
        return send_buffer_.raw_ptr() + slot.offset_elems;
    }

    value_type *native_backward_second_host_send_buffer_ptr_( const native_schedule_slot &slot )
    {
        return host_send_buffer_.raw_ptr() + slot.offset_elems;
    }

    template <class XFastComplex3>
    value_type *native_backward_second_recv_output_ptr_( const native_schedule_slot &slot, XFastComplex3 &out )
    {
        return out.raw_ptr() + slot.offset_elems;
    }

    value_type *native_backward_second_device_recv_buffer_ptr_( const native_schedule_slot &slot )
    {
        return recv_buffer_.raw_ptr() + slot.offset_elems;
    }

    value_type *native_backward_second_host_recv_buffer_ptr_( const native_schedule_slot &slot )
    {
        return host_recv_buffer_.raw_ptr() + slot.offset_elems;
    }

    template <class XFastComplex3>
    value_type *native_backward_second_recv_target_ptr_( const native_schedule_slot &slot, XFastComplex3 &out )
    {
        if ( backward_receive_lands_in_output_() )
            return native_backward_second_recv_output_ptr_( slot, out );
        return native_backward_second_device_recv_buffer_ptr_( slot );
    }

    std::size_t native_backward_second_send_x_size_(
        int target_i, const native_schedule_slot &slot, const char *stage
    ) const
    {
        const std::size_t yz = ny_output_local_ * nz_local_;
        if ( yz == 0 || slot.elems % yz != 0 )
        {
            throw std::logic_error(
                "owned reference native backward-second schedule has invalid send extent for " + std::string( stage )
            );
        }
        const std::size_t x_size = slot.elems / yz;
        if ( x_size != transpose1_dim_.size_x[target_i] )
        {
            std::ostringstream ss;
            ss << "owned reference native backward-second send extent mismatch for " << stage
               << " peer=" << target_i << " schedule_x=" << x_size
               << " partition_x=" << transpose1_dim_.size_x[target_i];
            throw std::logic_error( ss.str() );
        }
        return x_size;
    }
