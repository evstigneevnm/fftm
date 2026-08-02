    bool native_opt0_y_group_device_sync_enabled_() const
    {
        return native_opt0_default_z_layout_ && use_native_opt0_y_group_device_sync_;
    }

    bool native_forward_second_receive_lands_in_stage_() const
    {
        return forward_receive_lands_in_stage_();
    }

    void synchronize_native_opt0_y_plan_streams_( std::size_t count )
    {
        if ( !native_opt0_default_z_layout_ )
            return;
        if ( count > streams_.size() )
            throw std::logic_error( "native opt0 y-plan stream synchronization count mismatch" );
        if ( native_opt0_y_group_device_sync_enabled_() )
        {
            auto scope = profile_scope_( "native_opt0_y_fft_device_sync" );
            runtime_api_t::device_synchronize();
            return;
        }
        auto scope = profile_scope_( "native_opt0_y_fft_stream_sync" );
        for ( std::size_t i = 0; i < count; ++i )
        {
            runtime_api_t::stream_synchronize( streams_[i].stream() );
        }
    }

    void synchronize_native_opt0_y_plan_array_bundle_()
    {
        if ( !native_opt0_default_z_layout_ )
            return;
        if ( native_opt0_y_group_device_sync_enabled_() )
        {
            auto scope = profile_scope_( "native_opt0_y_fft_device_sync" );
            runtime_api_t::device_synchronize();
            return;
        }
        auto scope = profile_scope_( "native_opt0_y_fft_stream_sync" );
        base_fft_.synchronize_c2c_plan_array_streams( native_opt0_y_plan_array_bundle_ );
    }
