#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_STAGE_DIAGNOSTICS_INC__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_STAGE_DIAGNOSTICS_INC__

        void record_native_stage_timing_( const char *stage, double ms )
        {
            if ( !native_stage_timers_enabled_ || !native_stage_timing_active_ )
                return;
            ::fftm::fftm_native_stage_timing row;
            row.stage = stage;
            row.ms    = ms;
            native_stage_timings_.push_back( row );
        }

        template <class Fn>
        void time_native_stage_( const char *stage, Fn fn )
        {
            if ( !native_stage_timers_enabled_ || !native_stage_timing_active_ )
            {
                fn();
                return;
            }

            runtime_api_t::device_synchronize();
            scfd::utils::system_timer_event begin;
            scfd::utils::system_timer_event end;
            begin.record();
            fn();
            runtime_api_t::device_synchronize();
            end.record();
            record_native_stage_timing_( stage, end.elapsed_time( begin ) );
        }

    void debug_native_opt0_topology_marker_( const char *stage )
    {
        if ( !print_schedule_ || !native_opt0_reference_y_buffer_topology_enabled_() ||
             !( local_fft_diagnostics_enabled_ || native_stage_timers_enabled_ ) )
            return;
        std::ostringstream ss;
        ss << "native_opt0_reference_y_buffer_topology_marker rank_i=" << myid_i_ << " rank_j=" << myid_j_
           << " stage=" << stage;
        log_.info( ss.str() );
    }

    void append_local_fft_meta_row_( const std::string &key, long long index, long long value )
    {
        const std::string header = "source,run_label,rank,key,index,value,directory";
        std::ostringstream row;
        row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << key << ',' << index
            << ',' << value << ',' << local_fft_diagnostics_dir_;
        append_local_fft_diagnostics_row_( "local_fft_meta", header, row.str() );
    }

    void append_native_opt0_y_workarea_meta_(
        std::size_t plan_count, std::size_t base_offset_bytes, std::size_t work_stride_bytes
    )
    {
        if ( !local_fft_diagnostics_enabled_ )
            return;
        append_local_fft_meta_row_(
            "native_opt0_workspace_slot_offset", -1,
            static_cast<long long>( native_opt0_reference_workspace_slot_offset_() )
        );
        append_local_fft_meta_row_(
            "native_opt0_domainsize_bytes", -1,
            static_cast<long long>( native_opt0_reference_domain_size_bytes_() )
        );
        append_local_fft_meta_row_(
            "native_opt0_y_workarea_stride_bytes", -1, static_cast<long long>( work_stride_bytes )
        );
        append_local_fft_meta_row_(
            "native_opt0_y_workarea_extent_bytes", -1,
            static_cast<long long>( native_opt0_y_reference_workspace_size_bytes( plan_count, work_stride_bytes ) )
        );
        for ( std::size_t i = 0; i < plan_count; ++i )
        {
            const std::size_t offset = checked_add_bytes_(
                base_offset_bytes, checked_mul_bytes_( i, work_stride_bytes, "native opt0 y diagnostic offset" ),
                "native opt0 y diagnostic offset"
            );
            append_local_fft_meta_row_(
                "native_opt0_y_workarea_offset_bytes", static_cast<long long>( i ), static_cast<long long>( offset )
            );
        }
    }

#endif
