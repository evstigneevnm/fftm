		    template <class ArrayIn, class ArrayOut>
		    void exec_local_fft_(
		        const char *profile_name, const char *plan_name, const ArrayIn &in, ArrayOut &out
		    )
		    {
		        auto phase = optional_profiler_.scoped_tic( profile_name );
            time_native_stage_( profile_name, [&]() {
                if ( local_fft_diagnostics_enabled_ )
                {
                    using timer_event_t = typename backend_t::timer_event_type;
                    timer_event_t begin;
                    timer_event_t end;
                    begin.record();
                    SCFD_SAFE_CALL( base_fft_.template exec<ArrayIn, ArrayOut>( plan_name, in, out ) );
                    end.record();
                    const double elapsed_ms = end.elapsed_time( begin );
                    append_local_fft_timing_row_( profile_name, plan_name, elapsed_ms );
                }
                else
                {
		            SCFD_SAFE_CALL( base_fft_.template exec<ArrayIn, ArrayOut>( plan_name, in, out ) );
                }
            } );
		        synchronize_after_local_fft_if_needed_();
		    }

    template <class ArrayIn, class ArrayOut>
    void exec_local_fft_many_offsets_(
        const char *profile_name, const char *plan_label, const std::vector<std::string> &plan_names,
        const std::vector<std::size_t> &offsets, const ArrayIn &in, ArrayOut &out,
        plan_sequence_id_t plan_sequence = base_fft_t::invalid_plan_sequence_id(),
        bool use_explicit_exec_direction = false, direction exec_direction = direction::C2CF
    )
    {
        if ( plan_names.size() != offsets.size() )
            throw std::logic_error( "native opt0 z-fast local FFT plan/offset count mismatch" );
        if ( native_opt0_default_z_layout_ && plan_names.size() > streams_.size() )
            throw std::logic_error( "native opt0 z-fast local FFT stream count mismatch" );
        const bool use_cached_sequence = plan_sequence != base_fft_t::invalid_plan_sequence_id();

        auto phase = optional_profiler_.scoped_tic( profile_name );
        const bool dump_y_signature =
            should_dump_native_opt0_y_buffer_signature_( profile_name, plan_label );
        append_native_opt0_y_execution_diagnostics_(
            profile_name, plan_label, plan_names, offsets, in.raw_ptr(), out.raw_ptr(), plan_sequence,
            use_cached_sequence, use_explicit_exec_direction, exec_direction
        );
        if ( dump_y_signature )
        {
            runtime_api_t::device_synchronize();
            append_native_opt0_y_buffer_signature_(
                profile_name, plan_label, "pre", "input", plan_names, offsets, in.raw_ptr()
            );
            append_native_opt0_y_buffer_signature_(
                profile_name, plan_label, "pre", "output", plan_names, offsets, out.raw_ptr()
            );
        }
        time_native_stage_( profile_name, [&]() {
            if ( local_fft_diagnostics_enabled_ )
            {
                using timer_event_t = typename backend_t::timer_event_type;
                timer_event_t begin;
                timer_event_t end;
                begin.record();
                if ( use_cached_sequence )
                {
                    if ( use_native_opt0_y_no_sync_exec_ )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_direction_no_sync(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_no_sync(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                    else if ( use_native_opt0_raw_y_plan_array_executor_ )
                    {
                        SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_opaque_direction(
                            plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                        ) );
                    }
                    else if ( use_native_opt0_tight_y_plan_sequence_ )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_tight_direction(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_tight(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                    else
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_direction(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                }
                else
                {
                    for ( std::size_t i = 0; i < plan_names.size(); ++i )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_raw_direction(
                                plan_names[i], exec_direction, static_cast<void *>( in.raw_ptr() + offsets[i] ),
                                static_cast<void *>( out.raw_ptr() + offsets[i] )
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_raw(
                                plan_names[i], static_cast<void *>( in.raw_ptr() + offsets[i] ),
                                static_cast<void *>( out.raw_ptr() + offsets[i] )
                            ) );
                        }
                    }
                }
                synchronize_native_opt0_y_plan_streams_( plan_names.size() );
                end.record();
                const double elapsed_ms = end.elapsed_time( begin );
                append_local_fft_timing_row_( profile_name, plan_label, elapsed_ms );
            }
            else
            {
                if ( use_cached_sequence )
                {
                    if ( use_native_opt0_y_no_sync_exec_ )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_direction_no_sync(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_no_sync(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                    else if ( use_native_opt0_raw_y_plan_array_executor_ )
                    {
                        SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_opaque_direction(
                            plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                        ) );
                    }
                    else if ( use_native_opt0_tight_y_plan_sequence_ )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_tight_direction(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_tight(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                    else
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets_direction(
                                plan_sequence, offsets, exec_direction, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_plan_sequence_offsets(
                                plan_sequence, offsets, in.raw_ptr(), out.raw_ptr()
                            ) );
                        }
                    }
                }
                else
                {
                    for ( std::size_t i = 0; i < plan_names.size(); ++i )
                    {
                        if ( use_explicit_exec_direction )
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_raw_direction(
                                plan_names[i], exec_direction, static_cast<void *>( in.raw_ptr() + offsets[i] ),
                                static_cast<void *>( out.raw_ptr() + offsets[i] )
                            ) );
                        }
                        else
                        {
                            SCFD_SAFE_CALL( base_fft_.exec_raw(
                                plan_names[i], static_cast<void *>( in.raw_ptr() + offsets[i] ),
                                static_cast<void *>( out.raw_ptr() + offsets[i] )
                            ) );
                        }
                    }
                }
                synchronize_native_opt0_y_plan_streams_( plan_names.size() );
            }
        } );
        if ( dump_y_signature )
        {
            runtime_api_t::device_synchronize();
            append_native_opt0_y_buffer_signature_(
                profile_name, plan_label, "post", "input", plan_names, offsets, in.raw_ptr()
            );
            append_native_opt0_y_buffer_signature_(
                profile_name, plan_label, "post", "output", plan_names, offsets, out.raw_ptr()
            );
        }
        if ( !native_opt0_y_group_device_sync_enabled_() )
        {
            synchronize_after_local_fft_if_needed_();
        }
    }

    bool native_opt0_y_plan_array_bundle_active_() const
    {
        return ( use_native_opt0_reference_y_plan_bundle_ || use_native_opt0_raw_y_plan_bundle_ ||
                 use_native_opt0_reference_local_plan_context_ ) &&
               native_opt0_y_plan_array_bundle_ != base_fft_t::invalid_c2c_plan_array_id();
    }

    template <class ArrayIn, class ArrayOut>
    void exec_local_fft_plan_array_bundle_(
        const char *profile_name, const char *plan_label, const ArrayIn &in, ArrayOut &out,
        direction exec_direction
    )
    {
        if ( !native_opt0_y_plan_array_bundle_active_() )
        {
            throw std::logic_error( "native opt0 Y plan-array bundle execution requested without an active bundle" );
        }

        auto phase = optional_profiler_.scoped_tic( profile_name );
        time_native_stage_( profile_name, [&]() {
            if ( local_fft_diagnostics_enabled_ )
            {
                using timer_event_t = typename backend_t::timer_event_type;
                timer_event_t begin;
                timer_event_t end;
                begin.record();
                if ( use_native_opt0_y_no_sync_exec_ )
                {
                    SCFD_SAFE_CALL( base_fft_.exec_c2c_plan_array_direction_no_sync(
                        native_opt0_y_plan_array_bundle_, exec_direction, in.raw_ptr(), out.raw_ptr()
                    ) );
                }
                else
                {
                    SCFD_SAFE_CALL( base_fft_.exec_c2c_plan_array_direction(
                        native_opt0_y_plan_array_bundle_, exec_direction, in.raw_ptr(), out.raw_ptr()
                    ) );
                }
                synchronize_native_opt0_y_plan_array_bundle_();
                end.record();
                const double elapsed_ms = end.elapsed_time( begin );
                append_local_fft_timing_row_( profile_name, plan_label, elapsed_ms );
            }
            else
            {
                if ( use_native_opt0_y_no_sync_exec_ )
                {
                    SCFD_SAFE_CALL( base_fft_.exec_c2c_plan_array_direction_no_sync(
                        native_opt0_y_plan_array_bundle_, exec_direction, in.raw_ptr(), out.raw_ptr()
                    ) );
                }
                else
                {
                    SCFD_SAFE_CALL( base_fft_.exec_c2c_plan_array_direction(
                        native_opt0_y_plan_array_bundle_, exec_direction, in.raw_ptr(), out.raw_ptr()
                    ) );
                }
                synchronize_native_opt0_y_plan_array_bundle_();
            }
        } );
        if ( !native_opt0_y_group_device_sync_enabled_() )
        {
            synchronize_after_local_fft_if_needed_();
        }
    }
