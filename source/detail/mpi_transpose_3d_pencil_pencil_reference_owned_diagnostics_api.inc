#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_DIAGNOSTICS_API_INC__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_DIAGNOSTICS_API_INC__

    void set_local_fft_diagnostics( bool enabled, const std::string &directory, const std::string &label )
    {
        local_fft_diagnostics_enabled_ = enabled;
        local_fft_diagnostics_dir_     = directory.empty() ? "." : directory;
        local_fft_diagnostics_label_   = label.empty() ? "fftm-native" : label;
    }

    void set_native_stage_timers_enabled( bool enabled )
    {
        native_stage_timers_enabled_ = enabled;
    }

    void begin_native_stage_timing_iteration( int iteration )
    {
        native_stage_timing_iteration_ = iteration;
        native_stage_timings_.clear();
        native_stage_timing_active_ = native_stage_timers_enabled_;
    }

    void end_native_stage_timing_iteration()
    {
        native_stage_timing_active_ = false;
    }

    const std::vector<::fftm::fftm_native_stage_timing> &native_stage_timings() const
    {
        return native_stage_timings_;
    }

    void dump_local_fft_plan_descriptors()
    {
        if ( !local_fft_diagnostics_enabled_ || local_fft_plan_descriptors_dumped_ )
        {
            return;
        }

        const std::string header =
            "source,run_label,rank,plan_name,direction,cufft_type,rank_fft,default_layout,n0,n1,n2,inembed0,inembed1,inembed2,"
            "istride,idist,onembed0,onembed1,onembed2,ostride,odist,batch,work_size_bytes,work_area_token,"
            "stream_token,directory";
        for ( const auto &plan : base_fft_.plan_descriptors() )
        {
            const auto &desc = plan.second;
            std::ostringstream row;
            row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << plan.first << ','
                << local_fft_direction_name_( desc.dir ) << ',' << desc.cufft_type << ',' << desc.rank << ','
                << ( desc.default_layout ? 1 : 0 );
            for ( int i = 0; i < 3; ++i )
            {
                row << ',' << desc.n[static_cast<std::size_t>( i )];
            }
            for ( int i = 0; i < 3; ++i )
            {
                row << ',' << desc.inembed[static_cast<std::size_t>( i )];
            }
            row << ',' << desc.istride << ',' << desc.idist;
            for ( int i = 0; i < 3; ++i )
            {
                row << ',' << desc.onembed[static_cast<std::size_t>( i )];
            }
            row << ',' << desc.ostride << ',' << desc.odist << ',' << desc.batch << ',' << desc.work_size << ','
                << static_cast<unsigned long long>( desc.work_area_token ) << ','
                << static_cast<unsigned long long>( desc.stream_token ) << ',' << local_fft_diagnostics_dir_;
            append_local_fft_diagnostics_row_( "local_fft_plans", header, row.str() );
        }

        local_fft_plan_descriptors_dumped_ = true;
    }

#endif
