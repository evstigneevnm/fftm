    template <class ArrayIn, class ArrayOut>
    void exec_local_fft( const char *profile_name, const char *plan_name, const ArrayIn &in, ArrayOut &out )
    {
        exec_local_fft_( profile_name, plan_name, in, out );
    }

    template <class ArrayIn, class ArrayOut>
    void exec_backward_z_with_deferred_send( const ArrayIn &in, ArrayOut &out )
    {
        execute_with_deferred_send_overlap_( deferred_send_stage::backward_first, [&]() {
            exec_local_fft_( "reference_owned/backward_z_fft", "inverse_z", in, out );
        } );
    }

    template <class ArrayIn, class ArrayOut>
    void exec_local_fft_many_offsets(
        const char *profile_name, const char *plan_label, const std::vector<std::string> &plan_names,
        const std::vector<std::size_t> &offsets, const ArrayIn &in, ArrayOut &out
    )
    {
        exec_local_fft_many_offsets_( profile_name, plan_label, plan_names, offsets, in, out );
    }

    template <class FFTWrap>
    void bind_native_opt0_y_parallel_fft_plans(
        FFTWrap &base_fft, const std::vector<std::string> &forward_plan_names,
        const std::vector<std::string> &inverse_plan_names, std::size_t minimum_work_stride_bytes = 0
    )
    {
        if ( !native_opt0_default_z_layout_ )
            return;
        if ( !is_inited_ )
            throw std::logic_error( "native opt0 y-plan binding requires initialized pencil executor" );
        if ( !use_external_work_area_ || external_work_area_ == nullptr )
            throw std::logic_error( "native opt0 y-plan binding requires an external shared work area" );
        if ( forward_plan_names.size() != inverse_plan_names.size() )
            throw std::logic_error( "native opt0 y-plan binding forward/inverse count mismatch" );
        if ( forward_plan_names.size() > streams_.size() )
            throw std::logic_error( "native opt0 y-plan binding stream count mismatch" );
        if ( use_native_opt0_shared_y_plan_handles_ )
        {
            for ( std::size_t i = 0; i < forward_plan_names.size(); ++i )
            {
                if ( forward_plan_names[i] != inverse_plan_names[i] )
                {
                    throw std::logic_error(
                        "native opt0 shared Y plan handles require forward/inverse plan-name identity"
                    );
                }
            }
        }

        std::size_t work_stride = minimum_work_stride_bytes;
        for ( std::size_t i = 0; i < forward_plan_names.size(); ++i )
        {
            const auto forward_desc = base_fft.plan_descriptor( forward_plan_names[i] );
            const auto inverse_desc = base_fft.plan_descriptor( inverse_plan_names[i] );
            work_stride = std::max( work_stride, std::max( forward_desc.work_size, inverse_desc.work_size ) );
        }

        char             *raw         = static_cast<char *>( external_work_area_ );
        const std::size_t base_offset = native_opt0_y_reference_work_base_offset_bytes();
        for ( std::size_t i = 0; i < forward_plan_names.size(); ++i )
        {
            const std::size_t work_offset = checked_add_bytes_(
                base_offset, checked_mul_bytes_( i, work_stride, "native opt0 y-plan workarea offset" ),
                "native opt0 y-plan workarea offset"
            );
            void *work_area = static_cast<void *>( raw + work_offset );
            if ( use_native_opt0_reference_y_plan_lifecycle_ )
            {
                base_fft.recreate_plan_with_stream_and_work_area(
                    forward_plan_names[i], streams_[i].stream(), work_area
                );
                if ( inverse_plan_names[i] != forward_plan_names[i] )
                {
                    base_fft.recreate_plan_with_stream_and_work_area(
                        inverse_plan_names[i], streams_[i].stream(), work_area
                    );
                }
            }
            else
            {
                base_fft.set_work_area( forward_plan_names[i], work_area );
                base_fft.set_work_area( inverse_plan_names[i], work_area );
                base_fft.set_stream( forward_plan_names[i], streams_[i].stream() );
                base_fft.set_stream( inverse_plan_names[i], streams_[i].stream() );
            }
        }

        native_opt0_forward_y_plan_sequence_ = base_fft.make_plan_sequence( forward_plan_names );
        native_opt0_inverse_y_plan_sequence_ = base_fft.make_plan_sequence( inverse_plan_names );
        native_opt0_y_parallel_plan_count_ = forward_plan_names.size();

        append_native_opt0_y_workarea_meta_( forward_plan_names.size(), base_offset, work_stride );

        if ( !print_schedule_ )
            return;

        const std::size_t total_span =
            native_opt0_y_reference_workspace_size_bytes( forward_plan_names.size(), work_stride );
        std::ostringstream ss;
        ss << "native_opt0_y_plan_sequence active=1 plans=" << native_opt0_y_parallel_plan_count_
           << " forward_sequence=" << native_opt0_forward_y_plan_sequence_
           << " inverse_sequence=" << native_opt0_inverse_y_plan_sequence_
           << " executor=" << ( use_native_opt0_raw_y_plan_array_executor_ ? "opaque-raw" :
                                 use_native_opt0_tight_y_plan_sequence_ ? "tight" : "cached-virtual" )
           << " exec_check=" << ( use_native_opt0_y_no_sync_exec_ ? "no-sync" : "sync" )
           << " handle_mode=" << ( use_native_opt0_shared_y_plan_handles_ ? "shared-bidir" : "separate" )
           << " lifecycle=" << ( use_native_opt0_reference_y_plan_lifecycle_ ? "reference" : "standard" )
           << " workarea_base_offset_bytes=" << base_offset
           << " workarea_stride_bytes=" << work_stride
           << " workarea_total_span_bytes=" << total_span;
        log_.info( ss.str() );
    }

    template <class FFTWrap>
    void bind_native_opt0_y_plan_array_bundle(
        FFTWrap &base_fft, typename FFTWrap::c2c_plan_array_id_t bundle_id,
        std::size_t minimum_work_stride_bytes = 0
    )
    {
        if ( !native_opt0_default_z_layout_ )
            return;
        if ( !is_inited_ )
            throw std::logic_error( "native opt0 Y plan-array binding requires initialized pencil executor" );
        if ( !use_external_work_area_ || external_work_area_ == nullptr )
            throw std::logic_error( "native opt0 Y plan-array binding requires an external shared work area" );
        if ( bundle_id == FFTWrap::invalid_c2c_plan_array_id() )
            throw std::logic_error( "native opt0 Y plan-array binding received an invalid bundle id" );

        const std::size_t plan_count  = base_fft.c2c_plan_array_size( bundle_id );
        const std::size_t work_stride =
            std::max( minimum_work_stride_bytes, base_fft.c2c_plan_array_work_size( bundle_id ) );
        char             *raw         = static_cast<char *>( external_work_area_ );
        const std::size_t base_offset = native_opt0_y_reference_work_base_offset_bytes();
        if ( use_native_opt0_raw_y_plan_bundle_ )
        {
            if ( plan_count > streams_.size() )
                throw std::logic_error( "native opt0 raw Y plan-array binding stream count mismatch" );
            std::vector<typename FFTWrap::runtime_api::stream_t> stream_handles;
            stream_handles.reserve( plan_count );
            for ( std::size_t i = 0; i < plan_count; ++i )
                stream_handles.push_back( streams_[i].stream() );
            if ( use_native_opt0_y_plan_bundle_stream_first_ && !use_native_opt0_raw_y_plan_bundle_reference_streams_ )
                base_fft.bind_c2c_plan_array_streams( bundle_id, stream_handles );
            base_fft.bind_c2c_plan_array_work_areas( bundle_id, static_cast<void *>( raw + base_offset ), work_stride );
            if ( !use_native_opt0_y_plan_bundle_stream_first_ && !use_native_opt0_raw_y_plan_bundle_reference_streams_ )
                base_fft.bind_c2c_plan_array_streams( bundle_id, stream_handles );
        }
        else
        {
            base_fft.bind_c2c_plan_array_work_areas( bundle_id, static_cast<void *>( raw + base_offset ), work_stride );
        }

        native_opt0_forward_y_plan_sequence_ = base_fft_t::invalid_plan_sequence_id();
        native_opt0_inverse_y_plan_sequence_ = base_fft_t::invalid_plan_sequence_id();
        native_opt0_y_plan_array_bundle_     = bundle_id;
        native_opt0_y_parallel_plan_count_   = plan_count;

        append_native_opt0_y_workarea_meta_( plan_count, base_offset, work_stride );

        if ( !print_schedule_ )
            return;

        const std::size_t total_span = native_opt0_y_reference_workspace_size_bytes( plan_count, work_stride );
        std::ostringstream ss;
        ss << "native_opt0_y_plan_bundle active=1 plans=" << native_opt0_y_parallel_plan_count_
           << " bundle_id=" << native_opt0_y_plan_array_bundle_
           << " executor=" << ( use_native_opt0_reference_local_plan_context_ ? "reference-local-context" :
                                use_native_opt0_raw_y_plan_bundle_ ? "raw-reference-bundle" : "reference-bundle" )
           << " exec_check=" << ( use_native_opt0_y_no_sync_exec_ ? "no-sync" : "sync" )
           << " handle_mode=shared-bidir"
           << " lifecycle=" << ( use_native_opt0_reference_local_plan_context_ ? "reference-local-context" :
                                 use_native_opt0_raw_y_plan_bundle_ ? "raw-reference-owned" : "bundle-owned" )
           << " stream_topology=" << ( use_native_opt0_raw_y_plan_bundle_ || use_native_opt0_reference_local_plan_context_ ?
                                       "pencil-external" : "bundle-owned" )
           << " stream_bind_order=" << ( use_native_opt0_y_plan_bundle_stream_first_ ? "stream-before-work" : "work-before-stream" )
           << " raw_reference_streams=" << ( use_native_opt0_raw_y_plan_bundle_reference_streams_ ? 1 : 0 )
           << " reference_local_context=" << ( use_native_opt0_reference_local_plan_context_ ? 1 : 0 )
           << " workarea_base_offset_bytes=" << base_offset
           << " workarea_stride_bytes=" << work_stride
           << " workarea_total_span_bytes=" << total_span;
        log_.info( ss.str() );
    }

    template <class FFTWrap>
    void bind_native_opt0_y_diagnostic_plan_array_bundle(
        FFTWrap &base_fft, typename FFTWrap::c2c_plan_array_id_t bundle_id,
        std::size_t minimum_work_stride_bytes = 0
    )
    {
        if ( !native_opt0_default_z_layout_ )
            return;
        if ( !is_inited_ )
            throw std::logic_error( "native opt0 Y diagnostic plan-array binding requires initialized pencil executor" );
        if ( !use_external_work_area_ || external_work_area_ == nullptr )
            throw std::logic_error( "native opt0 Y diagnostic plan-array binding requires an external shared work area" );
        if ( bundle_id == FFTWrap::invalid_c2c_plan_array_id() )
            throw std::logic_error( "native opt0 Y diagnostic plan-array binding received an invalid bundle id" );

        const std::size_t plan_count  = base_fft.c2c_plan_array_size( bundle_id );
        const std::size_t work_stride =
            std::max( minimum_work_stride_bytes, base_fft.c2c_plan_array_work_size( bundle_id ) );
        if ( plan_count > streams_.size() )
            throw std::logic_error( "native opt0 Y diagnostic plan-array stream count mismatch" );

        char             *raw         = static_cast<char *>( external_work_area_ );
        const std::size_t base_offset = native_opt0_y_reference_work_base_offset_bytes();
        std::vector<typename FFTWrap::runtime_api::stream_t> stream_handles;
        stream_handles.reserve( plan_count );
        for ( std::size_t i = 0; i < plan_count; ++i )
            stream_handles.push_back( streams_[i].stream() );
        if ( use_native_opt0_y_plan_bundle_stream_first_ )
            base_fft.bind_c2c_plan_array_streams( bundle_id, stream_handles );
        base_fft.bind_c2c_plan_array_work_areas( bundle_id, static_cast<void *>( raw + base_offset ), work_stride );
        if ( !use_native_opt0_y_plan_bundle_stream_first_ )
            base_fft.bind_c2c_plan_array_streams( bundle_id, stream_handles );
    }

    template <class FFTWrap, class ArrayIn, class ArrayOut>
    void run_native_opt0_y_same_buffer_microbench(
        FFTWrap &base_fft, const std::vector<std::string> &forward_plan_names,
        const std::vector<std::string> &inverse_plan_names, const std::vector<std::size_t> &offsets,
        const ArrayIn &forward_in, ArrayOut &forward_out,
        typename FFTWrap::c2c_plan_array_id_t diagnostic_raw_bundle,
        value_type *production_backward_y_output_ptr, std::size_t iterations, std::size_t warmup
    );

    std::size_t native_opt0_y_reference_work_base_offset_bytes() const
    {
        if ( !native_opt0_default_z_layout_ )
            return 0;
        if ( !is_inited_ )
            throw std::logic_error( "native opt0 y-plan workarea query requires initialized pencil executor" );
        if ( !native_opt0_reference_y_buffer_topology_enabled_() )
            return 0;
        return checked_mul_bytes_(
            native_opt0_reference_workspace_slot_offset_(), native_opt0_reference_domain_size_bytes_(),
            "native opt0 y-plan workarea base offset"
        );
    }

    std::size_t native_opt0_y_reference_workspace_size_bytes(
        std::size_t plan_count, std::size_t plan_work_stride_bytes
    ) const
    {
        if ( !native_opt0_default_z_layout_ || plan_count == 0 )
            return 0;
        return checked_add_bytes_(
            native_opt0_y_reference_work_base_offset_bytes(),
            checked_mul_bytes_( plan_count, plan_work_stride_bytes, "native opt0 y-plan workarea span" ),
            "native opt0 y-plan workspace size"
        );
    }
