    std::size_t bytes_from_elems_( std::size_t elems ) const
    {
        return elems * sizeof( value_type );
    }

    static std::size_t checked_mul_bytes_( std::size_t lhs, std::size_t rhs, const char *what )
    {
        if ( rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs )
        {
            std::ostringstream ss;
            ss << what << " overflows size_t";
            throw std::overflow_error( ss.str() );
        }
        return lhs * rhs;
    }

    static std::size_t checked_add_bytes_( std::size_t lhs, std::size_t rhs, const char *what )
    {
        if ( lhs > std::numeric_limits<std::size_t>::max() - rhs )
        {
            std::ostringstream ss;
            ss << what << " overflows size_t";
            throw std::overflow_error( ss.str() );
        }
        return lhs + rhs;
    }

    std::size_t native_opt0_reference_workspace_slot_offset_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( native_opt0_reference_y_buffer_topology_enabled_() && use_native_opt0_compact_y_workarea_ )
            return 1;
        return 3;
#else
        return 1;
#endif
    }

    std::size_t native_opt0_reference_domain_size_bytes_() const
    {
        return bytes_from_elems_( max_redistribution_buffer_elems_() );
    }

    bool native_opt0_reference_y_buffer_topology_enabled_() const
    {
        return native_opt0_default_z_layout_ && use_native_opt0_reference_y_buffer_topology_;
    }

    value_type *native_opt0_reference_slot_ptr_( std::size_t slot ) const
    {
        if ( external_work_area_ == nullptr )
            throw std::logic_error( "native opt0 reference Y-buffer topology requires an external work area" );
        char *raw = static_cast<char *>( external_work_area_ );
        return reinterpret_cast<value_type *>( raw + slot * native_opt0_reference_domain_size_bytes_() );
    }

    value_type *native_opt0_reference_backward_y_output_ptr_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( use_native_opt0_compact_y_workarea_ )
            return native_opt0_reference_slot_ptr_( 3 );
        return native_opt0_reference_slot_ptr_( 2 );
#else
        return native_opt0_reference_slot_ptr_( 0 );
#endif
    }

    static std::size_t align_up_bytes_( std::size_t value, std::size_t alignment )
    {
        return ( value + alignment - 1 ) / alignment * alignment;
    }

    void bind_external_work_area_()
    {
        if ( external_work_area_ == nullptr )
        {
            return;
        }
        char *raw = static_cast<char *>( external_work_area_ );
        if ( native_opt0_reference_y_buffer_topology_enabled_() )
        {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            const std::size_t domain_bytes = native_opt0_reference_domain_size_bytes_();
            send_buffer_.init_by_raw_data(
                reinterpret_cast<value_type *>( raw + 2 * domain_bytes ), send_buffer_elems_
            );
            recv_buffer_.init_by_raw_data(
                reinterpret_cast<value_type *>( raw + domain_bytes ), recv_buffer_elems_
            );
#else
            send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), send_buffer_elems_ );
            recv_buffer_.init_by_raw_data(
                reinterpret_cast<value_type *>( raw + bytes_from_elems_( send_buffer_elems_ ) ), recv_buffer_elems_
            );
#endif
            return;
        }
        send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), send_buffer_elems_ );
        recv_buffer_.init_by_raw_data(
            reinterpret_cast<value_type *>( raw + bytes_from_elems_( send_buffer_elems_ ) ), recv_buffer_elems_
        );
    }

    void bind_external_host_work_area_()
    {
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( external_host_work_area_ == nullptr )
        {
            return;
        }
        char *raw = static_cast<char *>( external_host_work_area_ );
        host_send_buffer_.init_by_raw_data( reinterpret_cast<value_type *>( raw ), host_send_buffer_elems_ );
        host_recv_buffer_.init_by_raw_data(
            reinterpret_cast<value_type *>( raw + bytes_from_elems_( host_send_buffer_elems_ ) ),
            host_recv_buffer_elems_
        );
#endif
    }

    void update_memory_profile_()
    {
        if ( memory_profiler_ == nullptr || memory_profile_prefix_.empty() )
        {
            return;
        }
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/send_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<typename memory_profiler_t::bytes_type>( send_buffer_.size() ) *
                                          static_cast<typename memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/recv_buffer",
            use_external_work_area_ ? 0
                                    : static_cast<typename memory_profiler_t::bytes_type>( recv_buffer_.size() ) *
                                          static_cast<typename memory_profiler_t::bytes_type>( sizeof( value_type ) )
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_send_buffer",
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            0
#else
            use_external_host_work_area_
                ? 0
                : static_cast<typename memory_profiler_t::bytes_type>( host_send_buffer_.size() ) *
                      static_cast<typename memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
        );
        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/host_recv_buffer",
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            0
#else
            use_external_host_work_area_
                ? 0
                : static_cast<typename memory_profiler_t::bytes_type>( host_recv_buffer_.size() ) *
                      static_cast<typename memory_profiler_t::bytes_type>( sizeof( value_type ) )
#endif
        );
    }
