#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_Y_DIAGNOSTICS_H__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_Y_DIAGNOSTICS_H__

template <
    class BaseFFT, class ValueType, class Backend, class MPIComm, class Log, class RuntimeAPI,
    class OptionalProfiler>
template <class FFTWrap, class ArrayIn, class ArrayOut>
void mpi_transpose_3d_pencil_pencil_reference_owned<BaseFFT, ValueType, Backend, MPIComm, Log, RuntimeAPI, OptionalProfiler>::run_native_opt0_y_same_buffer_microbench(
        FFTWrap &base_fft, const std::vector<std::string> &forward_plan_names,
        const std::vector<std::string> &inverse_plan_names, const std::vector<std::size_t> &offsets,
        const ArrayIn &forward_in, ArrayOut &forward_out,
        typename FFTWrap::c2c_plan_array_id_t diagnostic_raw_bundle,
        value_type *production_backward_y_output_ptr, std::size_t iterations, std::size_t warmup
    )
    {
        if ( !native_opt0_default_z_layout_ )
            throw std::logic_error( "native opt0 Y microbenchmark requires native opt0 default-Z layout" );
        if ( !is_inited_ )
            throw std::logic_error( "native opt0 Y microbenchmark requires initialized pencil executor" );
        const bool have_active_bundle = native_opt0_y_plan_array_bundle_active_();
        if ( offsets.empty() )
            throw std::logic_error( "native opt0 Y microbenchmark received empty Y plan offsets" );
        if ( forward_plan_names.empty() || inverse_plan_names.empty() )
        {
            if ( !have_active_bundle && diagnostic_raw_bundle == FFTWrap::invalid_c2c_plan_array_id() )
            {
                throw std::logic_error( "native opt0 Y microbenchmark received no Y plan handles" );
            }
        }
        else if ( forward_plan_names.size() != offsets.size() || inverse_plan_names.size() != offsets.size() )
        {
            throw std::logic_error( "native opt0 Y microbenchmark received inconsistent Y plan metadata" );
        }
        if ( iterations == 0 )
            iterations = 1;

        const bool have_forward_sequence =
            native_opt0_forward_y_plan_sequence_ != base_fft_t::invalid_plan_sequence_id();
        const bool have_inverse_sequence =
            native_opt0_inverse_y_plan_sequence_ != base_fft_t::invalid_plan_sequence_id();
        if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
        {
            bind_native_opt0_y_diagnostic_plan_array_bundle( base_fft, diagnostic_raw_bundle );
        }

        auto first_desc = typename FFTWrap::plan_descriptor_t{};
        if ( !forward_plan_names.empty() )
            first_desc = base_fft.plan_descriptor( forward_plan_names.front() );
        else if ( have_active_bundle )
            first_desc = base_fft.c2c_plan_array_descriptor( native_opt0_y_plan_array_bundle_ );
        else
            first_desc = base_fft.c2c_plan_array_descriptor( diagnostic_raw_bundle );
        std::size_t descriptor_work_stride = 0;
        if ( !forward_plan_names.empty() )
        {
            for ( std::size_t i = 0; i < offsets.size(); ++i )
            {
                descriptor_work_stride = std::max(
                    descriptor_work_stride,
                    std::max(
                        base_fft.plan_descriptor( forward_plan_names[i] ).work_size,
                        base_fft.plan_descriptor( inverse_plan_names[i] ).work_size
                    )
                );
            }
        }
        else if ( have_active_bundle )
        {
            descriptor_work_stride = base_fft.c2c_plan_array_work_size( native_opt0_y_plan_array_bundle_ );
        }
        else
        {
            descriptor_work_stride = base_fft.c2c_plan_array_work_size( diagnostic_raw_bundle );
        }

        const bool have_reference_workarea =
            native_opt0_reference_y_buffer_topology_enabled_() && external_work_area_ != nullptr;
        value_type *reference_slot0_ptr       = nullptr;
        value_type *reference_slot2_ptr       = nullptr;
        value_type *production_bwd_y_ptr  = production_backward_y_output_ptr;
        value_type *first_y_work_ptr      = nullptr;
        value_type *last_y_work_ptr       = nullptr;
        if ( have_reference_workarea )
        {
            char             *raw              = static_cast<char *>( external_work_area_ );
            const std::size_t y_work_base      = native_opt0_y_reference_work_base_offset_bytes();
            const std::size_t y_work_span      = checked_mul_bytes_(
                offsets.size(), descriptor_work_stride, "native opt0 y microbench work span"
            );
            const std::size_t y_work_end       = checked_add_bytes_(
                y_work_base, y_work_span, "native opt0 y microbench work end"
            );
            const std::size_t after_ywork      = align_up_bytes_( y_work_end, 256 );
            reference_slot0_ptr                    = native_opt0_reference_slot_ptr_( 0 );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            reference_slot2_ptr = native_opt0_reference_slot_ptr_( 2 );
#endif
            if ( production_bwd_y_ptr == nullptr )
            {
                production_bwd_y_ptr = reinterpret_cast<value_type *>( raw + after_ywork );
            }
            first_y_work_ptr      = reinterpret_cast<value_type *>( raw + y_work_base );
            last_y_work_ptr       = offsets.size() > 1
                                  ? reinterpret_cast<value_type *>(
                                        raw + y_work_base + ( offsets.size() - 1 ) * descriptor_work_stride
                                    )
                                  : first_y_work_ptr;
        }

        const auto pointer_token = []( const void *ptr ) -> unsigned long long {
            return static_cast<unsigned long long>( reinterpret_cast<std::uintptr_t>( ptr ) );
        };

	        const auto append_row = [&]( const char *direction_name, const char *variant, double total_ms,
	                                     double event_total_ms, const char *event_timer,
	                                     typename FFTWrap::plan_sequence_id_t sequence_id,
	                                     typename FFTWrap::c2c_plan_array_id_t bundle_id, const void *in_ptr,
	                                     const void *out_ptr ) {
	            const double avg_ms = total_ms / static_cast<double>( iterations );
	            const bool   have_event = event_total_ms >= 0.0;
	            const double event_avg_ms = have_event ? event_total_ms / static_cast<double>( iterations ) : -1.0;
	            const double host_overhead_total_ms = have_event ? total_ms - event_total_ms : 0.0;
	            const double host_overhead_avg_ms =
	                have_event ? host_overhead_total_ms / static_cast<double>( iterations ) : 0.0;
	            const std::string header =
	                "source,run_label,rank,direction,variant,iterations,warmup,total_ms,avg_ms,"
	                "event_total_ms,event_avg_ms,host_overhead_total_ms,host_overhead_avg_ms,event_timer,plan_count,"
	                "sequence_id,bundle_id,n,inembed,istride,idist,onembed,ostride,odist,batch,work_stride_bytes,"
	                "in_token,out_token,work_base_token,first_work_token,last_work_token,reference_slot0_token,"
	                "reference_slot2_token,production_bwd_y_token,first_offset_elems,last_offset_elems,use_tight,"
                "use_shared,use_device_sync,"
	                "use_opaque,use_ref_lifecycle,use_ref_bundle,use_raw_bundle,use_stream_first,"
	                "use_reference_streams,use_local_context,directory";
	            std::ostringstream row;
	            row << "fftm-native-y-micro," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ','
	                << direction_name << ',' << variant << ',' << iterations << ',' << warmup << ',' << total_ms << ','
	                << avg_ms << ',';
	            if ( have_event )
	            {
	                row << event_total_ms << ',' << event_avg_ms << ',' << host_overhead_total_ms << ','
	                    << host_overhead_avg_ms << ',' << event_timer << ',';
	            }
	            else
	            {
	                row << ",,,,none,";
	            }
	            row << offsets.size() << ','
	                << sequence_id << ',' << bundle_id << ',' << first_desc.n[0] << ',' << first_desc.inembed[0] << ','
	                << first_desc.istride << ',' << first_desc.idist << ',' << first_desc.onembed[0] << ','
                << first_desc.ostride << ',' << first_desc.odist << ',' << first_desc.batch << ','
                << descriptor_work_stride << ',' << pointer_token( in_ptr )
                << ',' << pointer_token( out_ptr )
                << ',' << pointer_token( external_work_area_ )
                << ',' << pointer_token( first_y_work_ptr )
                << ',' << pointer_token( last_y_work_ptr )
                << ',' << pointer_token( reference_slot0_ptr )
                << ',' << pointer_token( reference_slot2_ptr )
                << ',' << pointer_token( production_bwd_y_ptr )
                << ',' << offsets.front() << ',' << offsets.back() << ','
                << ( use_native_opt0_tight_y_plan_sequence_ ? 1 : 0 ) << ','
                << ( use_native_opt0_shared_y_plan_handles_ ? 1 : 0 ) << ','
                << ( use_native_opt0_y_group_device_sync_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_raw_y_plan_array_executor_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_reference_y_plan_lifecycle_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_reference_y_plan_bundle_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_raw_y_plan_bundle_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_y_plan_bundle_stream_first_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_raw_y_plan_bundle_reference_streams_ ? 1 : 0 ) << ','
	                << ( use_native_opt0_reference_local_plan_context_ ? 1 : 0 ) << ','
	                << local_fft_diagnostics_dir_;
            append_local_fft_diagnostics_row_( "native_y_microbench", header, row.str() );
        };

	        struct native_y_event_pair
	        {
	            typename runtime_api_t::event_t start = nullptr;
	            typename runtime_api_t::event_t stop  = nullptr;

	            native_y_event_pair()
	            {
	                start = runtime_api_t::event_create();
	                stop  = runtime_api_t::event_create();
	            }

	            ~native_y_event_pair()
	            {
	                runtime_api_t::event_destroy( stop );
	                runtime_api_t::event_destroy( start );
	            }

	            void record_start()
	            {
	                runtime_api_t::event_record( start, runtime_api_t::default_stream() );
	            }

	            void record_stop()
	            {
	                runtime_api_t::event_record( stop, runtime_api_t::default_stream() );
	            }

	            void synchronize_stop()
	            {
	                runtime_api_t::event_synchronize( stop );
	            }

	            double elapsed_ms() const
	            {
	                return runtime_api_t::event_elapsed_time_ms( start, stop );
	            }
	        };

	        const auto run_timed = [&]( const char *direction_name, const char *variant,
	                                    typename FFTWrap::plan_sequence_id_t sequence_id,
	                                    typename FFTWrap::c2c_plan_array_id_t bundle_id, direction exec_direction,
	                                    value_type *in_ptr, value_type *out_ptr, const auto &launch ) {
	            runtime_api_t::device_synchronize();
	            for ( std::size_t i = 0; i < warmup; ++i )
	                launch( exec_direction, in_ptr, out_ptr );
	            runtime_api_t::device_synchronize();

	            native_y_event_pair event_pair;
	            scfd::utils::system_timer_event begin;
	            scfd::utils::system_timer_event end;
	            begin.record();
	            event_pair.record_start();
	            for ( std::size_t i = 0; i < iterations; ++i )
	                launch( exec_direction, in_ptr, out_ptr );
	            event_pair.record_stop();
	            runtime_api_t::device_synchronize();
	            event_pair.synchronize_stop();
	            end.record();
	            append_row(
	                direction_name, variant, end.elapsed_time( begin ), event_pair.elapsed_ms(), "default_stream",
	                sequence_id, bundle_id, in_ptr, out_ptr
	            );
	        };

	        const auto sync_native_y_streams_explicit = [&]( std::size_t count ) {
	            if ( count > streams_.size() )
	                throw std::logic_error( "native opt0 y microbench explicit stream sync count mismatch" );
	            for ( std::size_t i = 0; i < count; ++i )
	                runtime_api_t::stream_synchronize( streams_[i].stream() );
	        };

	        const auto sync_device_explicit = [&]() {
	            runtime_api_t::device_synchronize();
	        };

	        const auto sync_sequence_streams_explicit = [&]() {
	            sync_native_y_streams_explicit( offsets.size() );
	        };

	        const auto run_timed_batched = [&]( const char *direction_name, const char *variant,
	                                            typename FFTWrap::plan_sequence_id_t sequence_id,
	                                            typename FFTWrap::c2c_plan_array_id_t bundle_id,
	                                            direction exec_direction, value_type *in_ptr, value_type *out_ptr,
	                                            const auto &launch_no_sync, const auto &final_sync ) {
	            runtime_api_t::device_synchronize();
	            for ( std::size_t i = 0; i < warmup; ++i )
	                launch_no_sync( exec_direction, in_ptr, out_ptr );
	            final_sync();
	            runtime_api_t::device_synchronize();

	            native_y_event_pair event_pair;
	            scfd::utils::system_timer_event begin;
	            scfd::utils::system_timer_event end;
	            begin.record();
	            event_pair.record_start();
	            for ( std::size_t i = 0; i < iterations; ++i )
	                launch_no_sync( exec_direction, in_ptr, out_ptr );
	            event_pair.record_stop();
	            final_sync();
	            event_pair.synchronize_stop();
	            end.record();
	            append_row(
	                direction_name, variant, end.elapsed_time( begin ), event_pair.elapsed_ms(), "default_stream",
	                sequence_id, bundle_id, in_ptr, out_ptr
	            );
	        };

	        const auto run_timed_repeated = [&]( const char *direction_name, const char *variant,
	                                             typename FFTWrap::plan_sequence_id_t sequence_id,
	                                             typename FFTWrap::c2c_plan_array_id_t bundle_id,
	                                             direction exec_direction, value_type *in_ptr, value_type *out_ptr,
	                                             const auto &repeat_launch_no_sync, const auto &final_sync ) {
	            runtime_api_t::device_synchronize();
	            if ( warmup > 0 )
	                repeat_launch_no_sync( exec_direction, in_ptr, out_ptr, warmup );
	            final_sync();
	            runtime_api_t::device_synchronize();

	            native_y_event_pair event_pair;
	            scfd::utils::system_timer_event begin;
	            scfd::utils::system_timer_event end;
	            begin.record();
	            event_pair.record_start();
	            repeat_launch_no_sync( exec_direction, in_ptr, out_ptr, iterations );
	            event_pair.record_stop();
	            final_sync();
	            event_pair.synchronize_stop();
	            end.record();
	            append_row(
	                direction_name, variant, end.elapsed_time( begin ), event_pair.elapsed_ms(), "default_stream",
	                sequence_id, bundle_id, in_ptr, out_ptr
	            );
	        };

	        const auto launch_current_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                             direction exec_direction, value_type *in_ptr,
	                                                             value_type *out_ptr ) {
	            if ( use_native_opt0_raw_y_plan_array_executor_ )
	            {
	                base_fft.exec_plan_sequence_offsets_opaque_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
            }
            else if ( use_native_opt0_tight_y_plan_sequence_ )
            {
                base_fft.exec_plan_sequence_offsets_tight_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
            }
            else
	            {
	                base_fft.exec_plan_sequence_offsets_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
	            }
	        };

	        const auto repeat_current_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                             direction exec_direction, value_type *in_ptr,
	                                                             value_type *out_ptr, std::size_t repeats ) {
	            if ( use_native_opt0_raw_y_plan_array_executor_ )
	            {
	                base_fft.exec_plan_sequence_offsets_opaque_direction_repeated(
	                    sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	                );
	            }
	            else if ( use_native_opt0_tight_y_plan_sequence_ )
	            {
	                base_fft.exec_plan_sequence_offsets_tight_direction_repeated(
	                    sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	                );
	            }
	            else
	            {
	                base_fft.exec_plan_sequence_offsets_direction_repeated(
	                    sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	                );
	            }
	        };

	        const auto launch_current_sequence = [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	            const auto sequence_id = exec_direction == direction::C2CF ? native_opt0_forward_y_plan_sequence_
	                                                                       : native_opt0_inverse_y_plan_sequence_;
	            launch_current_sequence_id_no_sync( sequence_id, exec_direction, in_ptr, out_ptr );
	            synchronize_native_opt0_y_plan_streams_( offsets.size() );
	        };

	        const auto launch_virtual_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                             direction exec_direction, value_type *in_ptr,
	                                                             value_type *out_ptr ) {
	            base_fft.exec_plan_sequence_offsets_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
	        };

	        const auto repeat_virtual_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                             direction exec_direction, value_type *in_ptr,
	                                                             value_type *out_ptr, std::size_t repeats ) {
	            base_fft.exec_plan_sequence_offsets_direction_repeated(
	                sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	            );
	        };

	        const auto launch_virtual_sequence = [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	            const auto sequence_id = exec_direction == direction::C2CF ? native_opt0_forward_y_plan_sequence_
	                                                                       : native_opt0_inverse_y_plan_sequence_;
	            launch_virtual_sequence_id_no_sync( sequence_id, exec_direction, in_ptr, out_ptr );
	            synchronize_native_opt0_y_plan_streams_( offsets.size() );
	        };

	        const auto launch_tight_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                           direction exec_direction, value_type *in_ptr,
	                                                           value_type *out_ptr ) {
	            base_fft.exec_plan_sequence_offsets_tight_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
	        };

	        const auto repeat_tight_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                           direction exec_direction, value_type *in_ptr,
	                                                           value_type *out_ptr, std::size_t repeats ) {
	            base_fft.exec_plan_sequence_offsets_tight_direction_repeated(
	                sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	            );
	        };

	        const auto launch_tight_sequence = [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	            const auto sequence_id = exec_direction == direction::C2CF ? native_opt0_forward_y_plan_sequence_
	                                                                       : native_opt0_inverse_y_plan_sequence_;
	            launch_tight_sequence_id_no_sync( sequence_id, exec_direction, in_ptr, out_ptr );
	            synchronize_native_opt0_y_plan_streams_( offsets.size() );
	        };

	        const auto launch_opaque_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                            direction exec_direction, value_type *in_ptr,
	                                                            value_type *out_ptr ) {
	            base_fft.exec_plan_sequence_offsets_opaque_direction( sequence_id, offsets, exec_direction, in_ptr, out_ptr );
	        };

	        const auto repeat_opaque_sequence_id_no_sync = [&]( typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                            direction exec_direction, value_type *in_ptr,
	                                                            value_type *out_ptr, std::size_t repeats ) {
	            base_fft.exec_plan_sequence_offsets_opaque_direction_repeated(
	                sequence_id, offsets, exec_direction, in_ptr, out_ptr, repeats
	            );
	        };

	        const auto launch_opaque_sequence = [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	            const auto sequence_id = exec_direction == direction::C2CF ? native_opt0_forward_y_plan_sequence_
	                                                                       : native_opt0_inverse_y_plan_sequence_;
	            launch_opaque_sequence_id_no_sync( sequence_id, exec_direction, in_ptr, out_ptr );
	            synchronize_native_opt0_y_plan_streams_( offsets.size() );
	        };

	        const auto launch_bundle_no_sync = [&]( typename FFTWrap::c2c_plan_array_id_t bundle_id,
	                                                direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	            base_fft.exec_c2c_plan_array_direction( bundle_id, exec_direction, in_ptr, out_ptr );
	        };

	        const auto repeat_bundle_no_sync = [&]( typename FFTWrap::c2c_plan_array_id_t bundle_id,
	                                                direction exec_direction, value_type *in_ptr, value_type *out_ptr,
	                                                std::size_t repeats ) {
	            base_fft.exec_c2c_plan_array_direction_repeated( bundle_id, exec_direction, in_ptr, out_ptr, repeats );
	        };

	        const auto launch_bundle = [&]( typename FFTWrap::c2c_plan_array_id_t bundle_id, direction exec_direction,
	                                        value_type *in_ptr, value_type *out_ptr ) {
	            launch_bundle_no_sync( bundle_id, exec_direction, in_ptr, out_ptr );
	            base_fft.synchronize_c2c_plan_array_streams( bundle_id );
	        };

	        value_type *forward_in_ptr  = const_cast<value_type *>( forward_in.raw_ptr() );
	        value_type *forward_out_ptr = forward_out.raw_ptr();

	        const auto prefixed_variant = []( const std::string &prefix, const char *suffix ) {
	            return prefix + suffix;
	        };

	        const auto run_sequence_launch_sync_matrix = [&]( const char *direction_name, const std::string &prefix,
	                                                          typename FFTWrap::plan_sequence_id_t sequence_id,
	                                                          direction exec_direction, value_type *in_ptr,
	                                                          value_type *out_ptr ) {
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-current-final-device" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_current_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-current-final-streams" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_current_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-virtual-final-device" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_virtual_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-virtual-final-streams" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_virtual_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-tight-final-device" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_tight_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-tight-final-streams" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_tight_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-opaque-final-device" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_opaque_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-opaque-final-streams" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_opaque_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-current-final-device" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_current_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-current-final-streams" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_current_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-virtual-final-device" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_virtual_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-virtual-final-streams" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_virtual_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-tight-final-device" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_tight_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-tight-final-streams" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_tight_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-opaque-final-device" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_opaque_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-opaque-final-streams" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), sequence_id, FFTWrap::invalid_c2c_plan_array_id(),
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_opaque_sequence_id_no_sync( sequence_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_sequence_streams_explicit
	                );
	            }
	        };

	        const auto run_bundle_launch_sync_matrix = [&]( const char *direction_name, const std::string &prefix,
	                                                        typename FFTWrap::c2c_plan_array_id_t bundle_id,
	                                                        direction exec_direction, value_type *in_ptr,
	                                                        value_type *out_ptr ) {
	            const auto sync_bundle_streams_explicit = [&]() {
	                base_fft.synchronize_c2c_plan_array_streams( bundle_id );
	            };
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-bundle-final-device" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), FFTWrap::invalid_plan_sequence_id(), bundle_id,
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_bundle_no_sync( bundle_id, dir, in_raw, out_raw );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "batch-bundle-final-streams" );
	                run_timed_batched(
	                    direction_name, variant.c_str(), FFTWrap::invalid_plan_sequence_id(), bundle_id,
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw ) {
	                        launch_bundle_no_sync( bundle_id, dir, in_raw, out_raw );
	                    },
	                    sync_bundle_streams_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-bundle-final-device" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), FFTWrap::invalid_plan_sequence_id(), bundle_id,
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_bundle_no_sync( bundle_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_device_explicit
	                );
	            }
	            {
	                const std::string variant = prefixed_variant( prefix, "wrap-repeat-bundle-final-streams" );
	                run_timed_repeated(
	                    direction_name, variant.c_str(), FFTWrap::invalid_plan_sequence_id(), bundle_id,
	                    exec_direction, in_ptr, out_ptr,
	                    [&]( direction dir, value_type *in_raw, value_type *out_raw, std::size_t repeats ) {
	                        repeat_bundle_no_sync( bundle_id, dir, in_raw, out_raw, repeats );
	                    },
	                    sync_bundle_streams_explicit
	                );
	            }
	        };

	        if ( have_forward_sequence )
	        {
            run_timed(
                "forward", "seq-current", native_opt0_forward_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, forward_in_ptr, forward_out_ptr,
                launch_current_sequence
            );
            run_timed(
                "forward", "seq-virtual", native_opt0_forward_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, forward_in_ptr, forward_out_ptr,
                launch_virtual_sequence
            );
            run_timed(
                "forward", "seq-tight", native_opt0_forward_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, forward_in_ptr, forward_out_ptr,
                launch_tight_sequence
            );
	            run_timed(
	                "forward", "seq-opaque", native_opt0_forward_y_plan_sequence_,
	                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, forward_in_ptr, forward_out_ptr,
	                launch_opaque_sequence
	            );
	            run_sequence_launch_sync_matrix(
	                "forward", "", native_opt0_forward_y_plan_sequence_, direction::C2CF, forward_in_ptr,
	                forward_out_ptr
	            );
	        }
        if ( native_opt0_y_plan_array_bundle_active_() )
        {
            run_timed(
                "forward", "active-bundle", FFTWrap::invalid_plan_sequence_id(), native_opt0_y_plan_array_bundle_,
                direction::C2CF, forward_in_ptr, forward_out_ptr,
	                [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                    launch_bundle( native_opt0_y_plan_array_bundle_, exec_direction, in_ptr, out_ptr );
	                }
	            );
	            run_bundle_launch_sync_matrix(
	                "forward", "active-", native_opt0_y_plan_array_bundle_, direction::C2CF, forward_in_ptr,
	                forward_out_ptr
	            );
	        }
        if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
        {
            run_timed(
                "forward", "diag-raw-bundle", FFTWrap::invalid_plan_sequence_id(), diagnostic_raw_bundle,
                direction::C2CF, forward_in_ptr, forward_out_ptr,
	                [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                    launch_bundle( diagnostic_raw_bundle, exec_direction, in_ptr, out_ptr );
	                }
	            );
	            run_bundle_launch_sync_matrix(
	                "forward", "diag-raw-", diagnostic_raw_bundle, direction::C2CF, forward_in_ptr, forward_out_ptr
	            );
	        }

        if ( have_reference_workarea )
        {
            value_type *reference_backward_output_ptr = native_opt0_reference_backward_y_output_ptr_();
            if ( have_forward_sequence )
            {
                run_timed(
                    "forward", "reference-workarea-seq-current", native_opt0_forward_y_plan_sequence_,
                    FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, reference_slot0_ptr, forward_out_ptr,
                    launch_current_sequence
                );
	                run_timed(
	                    "forward", "reference-workarea-seq-opaque", native_opt0_forward_y_plan_sequence_,
	                    FFTWrap::invalid_c2c_plan_array_id(), direction::C2CF, reference_slot0_ptr, forward_out_ptr,
	                    launch_opaque_sequence
	                );
	                run_sequence_launch_sync_matrix(
	                    "forward", "reference-workarea-", native_opt0_forward_y_plan_sequence_, direction::C2CF,
	                    reference_slot0_ptr, forward_out_ptr
	                );
	            }
            if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
            {
                run_timed(
                    "forward", "reference-workarea-diag-raw-bundle", FFTWrap::invalid_plan_sequence_id(),
                    diagnostic_raw_bundle, direction::C2CF, reference_slot0_ptr, forward_out_ptr,
	                    [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                        launch_bundle( diagnostic_raw_bundle, exec_direction, in_ptr, out_ptr );
	                    }
	                );
	                run_bundle_launch_sync_matrix(
	                    "forward", "reference-workarea-diag-raw-", diagnostic_raw_bundle, direction::C2CF,
	                    reference_slot0_ptr, forward_out_ptr
	                );
	            }

            if ( have_inverse_sequence )
            {
                run_timed(
                    "backward", "reference-workarea-seq-current", native_opt0_inverse_y_plan_sequence_,
                    FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr,
                    reference_backward_output_ptr, launch_current_sequence
                );
	                run_timed(
	                    "backward", "reference-workarea-seq-opaque", native_opt0_inverse_y_plan_sequence_,
	                    FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr,
	                    reference_backward_output_ptr, launch_opaque_sequence
	                );
	                run_sequence_launch_sync_matrix(
	                    "backward", "reference-workarea-", native_opt0_inverse_y_plan_sequence_, direction::C2CB,
	                    forward_out_ptr, reference_backward_output_ptr
	                );
	            }
            if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
            {
                run_timed(
                    "backward", "reference-workarea-diag-raw-bundle", FFTWrap::invalid_plan_sequence_id(),
                    diagnostic_raw_bundle, direction::C2CB, forward_out_ptr, reference_backward_output_ptr,
	                    [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                        launch_bundle( diagnostic_raw_bundle, exec_direction, in_ptr, out_ptr );
	                    }
	                );
	                run_bundle_launch_sync_matrix(
	                    "backward", "reference-workarea-diag-raw-", diagnostic_raw_bundle, direction::C2CB,
	                    forward_out_ptr, reference_backward_output_ptr
	                );
	            }

            if ( production_bwd_y_ptr != nullptr )
            {
                if ( have_inverse_sequence )
                {
                    run_timed(
                        "backward", "after-ywork-seq-current", native_opt0_inverse_y_plan_sequence_,
                        FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr,
                        production_bwd_y_ptr, launch_current_sequence
                    );
	                    run_timed(
	                        "backward", "after-ywork-seq-opaque", native_opt0_inverse_y_plan_sequence_,
	                        FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr,
	                        production_bwd_y_ptr, launch_opaque_sequence
	                    );
	                    run_sequence_launch_sync_matrix(
	                        "backward", "after-ywork-", native_opt0_inverse_y_plan_sequence_, direction::C2CB,
	                        forward_out_ptr, production_bwd_y_ptr
	                    );
	                }
                if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
                {
                    run_timed(
                        "backward", "after-ywork-diag-raw-bundle", FFTWrap::invalid_plan_sequence_id(),
                        diagnostic_raw_bundle, direction::C2CB, forward_out_ptr, production_bwd_y_ptr,
	                        [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                            launch_bundle( diagnostic_raw_bundle, exec_direction, in_ptr, out_ptr );
	                        }
	                    );
	                    run_bundle_launch_sync_matrix(
	                        "backward", "after-ywork-diag-raw-", diagnostic_raw_bundle, direction::C2CB,
	                        forward_out_ptr, production_bwd_y_ptr
	                    );
	                }
	            }
	        }

        if ( have_inverse_sequence )
        {
            run_timed(
                "backward", "seq-current", native_opt0_inverse_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr, forward_in_ptr,
                launch_current_sequence
            );
            run_timed(
                "backward", "seq-virtual", native_opt0_inverse_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr, forward_in_ptr,
                launch_virtual_sequence
            );
            run_timed(
                "backward", "seq-tight", native_opt0_inverse_y_plan_sequence_,
                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr, forward_in_ptr,
                launch_tight_sequence
            );
	            run_timed(
	                "backward", "seq-opaque", native_opt0_inverse_y_plan_sequence_,
	                FFTWrap::invalid_c2c_plan_array_id(), direction::C2CB, forward_out_ptr, forward_in_ptr,
	                launch_opaque_sequence
	            );
	            run_sequence_launch_sync_matrix(
	                "backward", "", native_opt0_inverse_y_plan_sequence_, direction::C2CB, forward_out_ptr,
	                forward_in_ptr
	            );
	        }
        if ( native_opt0_y_plan_array_bundle_active_() )
        {
            run_timed(
                "backward", "active-bundle", FFTWrap::invalid_plan_sequence_id(), native_opt0_y_plan_array_bundle_,
                direction::C2CB, forward_out_ptr, forward_in_ptr,
	                [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                    launch_bundle( native_opt0_y_plan_array_bundle_, exec_direction, in_ptr, out_ptr );
	                }
	            );
	            run_bundle_launch_sync_matrix(
	                "backward", "active-", native_opt0_y_plan_array_bundle_, direction::C2CB, forward_out_ptr,
	                forward_in_ptr
	            );
	        }
        if ( diagnostic_raw_bundle != FFTWrap::invalid_c2c_plan_array_id() )
        {
            run_timed(
                "backward", "diag-raw-bundle", FFTWrap::invalid_plan_sequence_id(), diagnostic_raw_bundle,
                direction::C2CB, forward_out_ptr, forward_in_ptr,
	                [&]( direction exec_direction, value_type *in_ptr, value_type *out_ptr ) {
	                    launch_bundle( diagnostic_raw_bundle, exec_direction, in_ptr, out_ptr );
	                }
	            );
	            run_bundle_launch_sync_matrix(
	                "backward", "diag-raw-", diagnostic_raw_bundle, direction::C2CB, forward_out_ptr, forward_in_ptr
	            );
	        }
    }

#endif
