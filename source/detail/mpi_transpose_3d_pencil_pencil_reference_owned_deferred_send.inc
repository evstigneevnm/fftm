    static const char *deferred_send_stage_name_( deferred_send_stage stage )
    {
        switch ( stage )
        {
        case deferred_send_stage::forward_first:
            return "forward_first";
        case deferred_send_stage::forward_second:
            return "forward_second";
        case deferred_send_stage::backward_second:
            return "backward_second";
        case deferred_send_stage::backward_first:
            return "backward_first";
        case deferred_send_stage::none:
            return "none";
        }
        return "unknown";
    }

    bool deferred_send_completion_active_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return use_deferred_send_completion_ && reference_parity_enabled_ &&
               mode_ == mpi_transpose_3d_mode::p2p_waitany && cuda_aware_byte_p2p_enabled_() &&
               !use_persistent_p2p_;
#else
        return false;
#endif
    }

    scfd::communication::mpi_comm_info &deferred_send_comm_info_()
    {
        return deferred_send_state_.communicator == deferred_send_communicator::row ? row_comm_info_
                                                                                    : line_comm_info_;
    }

    std::string deferred_send_profile_name_( deferred_send_stage stage, const char *action ) const
    {
        return std::string( "deferred_send/" ) + deferred_send_stage_name_( stage ) + "/" + action;
    }

    template <class Fn>
    void time_deferred_send_action_( deferred_send_stage stage, const char *action, Fn fn )
    {
        const std::string label = deferred_send_profile_name_( stage, action );
        auto              scope = profile_scope_( label );
        if ( !native_stage_timing_active_ )
        {
            fn();
            return;
        }

        scfd::utils::system_timer_event begin;
        scfd::utils::system_timer_event end;
        begin.record();
        fn();
        end.record();
        record_native_stage_timing_( label.c_str(), end.elapsed_time( begin ) );
    }

    void ensure_no_deferred_send_( const char *where ) const
    {
        if ( !deferred_send_state_.active() )
            return;
        throw std::logic_error(
            std::string( "owned reference deferred send still active at " ) + where + ": " +
            deferred_send_stage_name_( deferred_send_state_.stage )
        );
    }

    void defer_byte_send_requests_(
        deferred_send_stage stage, deferred_send_communicator communicator,
        std::vector<mpi_request_t> &byte_send_requests
    )
    {
        ensure_no_deferred_send_( "new send deferral" );
        time_deferred_send_action_( stage, "send_deferred", [&]() {
            deferred_send_state_.stage        = stage;
            deferred_send_state_.communicator = communicator;
            deferred_send_state_.requests.swap( byte_send_requests );
            deferred_send_state_.remaining =
                detail::mpi_int_cast( deferred_send_state_.requests.size(), "owned deferred send request count" );
            if ( deferred_send_state_.remaining == 0 )
                deferred_send_state_.reset();
        } );
    }

    void probe_deferred_send_after_fft_( deferred_send_stage expected_stage )
    {
        if ( !deferred_send_state_.active() )
            return;
        if ( deferred_send_state_.stage != expected_stage )
        {
            throw std::logic_error(
                std::string( "owned reference deferred send stage mismatch after FFT: expected " ) +
                deferred_send_stage_name_( expected_stage ) + ", active " +
                deferred_send_stage_name_( deferred_send_state_.stage )
            );
        }

        time_deferred_send_action_( expected_stage, "send_ready_after_fft", [&]() {
            auto &comm_info = deferred_send_comm_info_();
            while ( deferred_send_state_.remaining > 0 )
            {
                int flag  = 0;
                const int index = comm_info.testany(
                    detail::mpi_int_cast(
                        deferred_send_state_.requests.size(), "owned deferred send readiness request count"
                    ),
                    deferred_send_state_.requests.data(), &flag
                );
                if ( !flag )
                    break;
                if ( index == MPI_UNDEFINED )
                {
                    deferred_send_state_.remaining = 0;
                    break;
                }
                --deferred_send_state_.remaining;
            }
        } );
    }

    void drain_deferred_send_( deferred_send_stage expected_stage )
    {
        if ( !deferred_send_state_.active() )
            return;
        if ( deferred_send_state_.stage != expected_stage )
        {
            throw std::logic_error(
                std::string( "owned reference deferred send drain stage mismatch: expected " ) +
                deferred_send_stage_name_( expected_stage ) + ", active " +
                deferred_send_stage_name_( deferred_send_state_.stage )
            );
        }

        time_deferred_send_action_( expected_stage, "send_drain", [&]() {
            if ( deferred_send_state_.remaining > 0 )
            {
                auto &comm_info = deferred_send_comm_info_();
                comm_info.waitall(
                    detail::mpi_int_cast(
                        deferred_send_state_.requests.size(), "owned deferred send drain request count"
                    ),
                    deferred_send_state_.requests.data()
                );
            }
        } );
        deferred_send_state_.reset();
    }

    void finish_deferred_send_after_fft_( deferred_send_stage expected_stage )
    {
        probe_deferred_send_after_fft_( expected_stage );
        drain_deferred_send_( expected_stage );
    }

    template <class Fn>
    void execute_with_deferred_send_overlap_( deferred_send_stage stage, Fn &&fn )
    {
        try
        {
            std::forward<Fn>( fn )();
        }
        catch ( ... )
        {
            drain_deferred_send_noexcept_();
            throw;
        }
        finish_deferred_send_after_fft_( stage );
    }

    void drain_deferred_send_noexcept_() noexcept
    {
        if ( !deferred_send_state_.active() )
            return;
        try
        {
            auto &comm_info = deferred_send_comm_info_();
            if ( !deferred_send_state_.requests.empty() )
            {
                comm_info.waitall(
                    detail::mpi_int_cast(
                        deferred_send_state_.requests.size(), "owned deferred send destructor request count"
                    ),
                    deferred_send_state_.requests.data()
                );
            }
        }
        catch ( ... )
        {
        }
        deferred_send_state_.reset();
    }

    template <class CommInfo>
    void wait_send_requests_(
        CommInfo &comm_info, std::vector<mpi_request_t> &byte_send_requests,
        detail::persistent_send_requests<mpi_request_t> &persistent_send_state,
        std::vector<mpi_request_t> &value_send_requests, int value_request_count, const char *byte_what,
        const char *persistent_what
    )
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( cuda_aware_byte_p2p_enabled_() )
        {
            if ( persistent_byte_p2p_enabled_() )
            {
                waitall_vector_( comm_info, byte_send_requests, byte_what );
                if ( !persistent_send_state.requests.empty() )
                {
                    comm_info.waitall(
                        detail::mpi_int_cast( persistent_send_state.requests.size(), persistent_what ),
                        persistent_send_state.requests.data()
                    );
                }
            }
            else
            {
                waitall_vector_( comm_info, byte_send_requests, byte_what );
            }
            return;
        }
#endif
        if ( persistent_value_p2p_enabled_() )
        {
            comm_info.waitall( value_request_count, persistent_send_state.requests.data() );
        }
        else
        {
            comm_info.waitall( value_request_count, value_send_requests.data() );
        }
    }

    template <class CommInfo>
    void wait_or_defer_send_requests_(
        deferred_send_stage stage, deferred_send_communicator communicator, CommInfo &comm_info,
        std::vector<mpi_request_t> &byte_send_requests,
        detail::persistent_send_requests<mpi_request_t> &persistent_send_state,
        std::vector<mpi_request_t> &value_send_requests, int value_request_count, const char *byte_what,
        const char *persistent_what
    )
    {
        if ( deferred_send_completion_active_() )
        {
            defer_byte_send_requests_( stage, communicator, byte_send_requests );
            return;
        }
        wait_send_requests_(
            comm_info, byte_send_requests, persistent_send_state, value_send_requests, value_request_count, byte_what,
            persistent_what
        );
    }

    void null_completed_request_( std::vector<mpi_request_t> &requests, int index )
    {
        if ( index >= 0 && index < static_cast<int>( requests.size() ) )
            requests[index] = mpi_request_t();
    }
