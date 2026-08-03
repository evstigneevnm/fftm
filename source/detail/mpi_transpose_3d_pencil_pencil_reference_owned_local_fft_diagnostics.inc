#ifndef __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_LOCAL_FFT_DIAGNOSTICS_INC__
#define __FFTM_DETAIL_MPI_TRANSPOSE_3D_PENCIL_PENCIL_REFERENCE_OWNED_LOCAL_FFT_DIAGNOSTICS_INC__

    std::string local_fft_diagnostics_path_( const char *kind ) const
    {
        std::ostringstream path;
        path << local_fft_diagnostics_dir_;
        if ( !local_fft_diagnostics_dir_.empty() &&
             local_fft_diagnostics_dir_[local_fft_diagnostics_dir_.size() - 1] != '/' )
        {
            path << '/';
        }
        path << kind << "_r" << mpi_.myid << ".csv";
        return path.str();
    }

    static bool local_fft_file_exists_( const std::string &path )
    {
        std::ifstream in( path.c_str() );
        return static_cast<bool>( in );
    }

    void append_local_fft_diagnostics_row_(
        const char *kind, const std::string &header, const std::string &row
    ) const
    {
        const std::string path         = local_fft_diagnostics_path_( kind );
        const bool        write_header = !local_fft_file_exists_( path );
        std::ofstream     out( path.c_str(), std::ios::out | std::ios::app );
        if ( !out )
        {
            throw std::runtime_error( "failed to open local FFT diagnostics CSV '" + path + "'" );
        }
        if ( write_header )
        {
            out << header << '\n';
        }
        out << row << '\n';
    }

	    void append_local_fft_timing_row_( const char *profile_name, const char *plan_name, double elapsed_ms )
	    {
	        const std::string header = "source,run_label,rank,call_index,profile_name,plan_name,event_ms,directory";
	        std::ostringstream row;
	        row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << local_fft_call_index_++
	            << ',' << profile_name << ',' << plan_name << ',' << elapsed_ms << ',' << local_fft_diagnostics_dir_;
	        append_local_fft_diagnostics_row_( "local_fft_times", header, row.str() );
	    }

	    struct opt0_layout_coord_t
	    {
	        const char  *sample;
	        std::size_t c0;
	        std::size_t c1;
	        std::size_t c2;
	    };

	    static void add_opt0_layout_coord_(
	        std::vector<opt0_layout_coord_t> &coords, const char *sample, std::size_t c0, std::size_t c1,
	        std::size_t c2, std::size_t d0, std::size_t d1, std::size_t d2
	    )
	    {
	        if ( d0 == 0 || d1 == 0 || d2 == 0 || c0 >= d0 || c1 >= d1 || c2 >= d2 )
	            return;
	        for ( const auto &coord : coords )
	        {
	            if ( coord.c0 == c0 && coord.c1 == c1 && coord.c2 == c2 )
	                return;
	        }
	        coords.push_back( opt0_layout_coord_t{ sample, c0, c1, c2 } );
	    }

	    static std::vector<opt0_layout_coord_t> opt0_layout_coords_(
	        std::size_t d0, std::size_t d1, std::size_t d2
	    )
	    {
	        std::vector<opt0_layout_coord_t> coords;
	        add_opt0_layout_coord_( coords, "origin", 0, 0, 0, d0, d1, d2 );
	        add_opt0_layout_coord_( coords, "z1", 0, 0, std::min<std::size_t>( 1, d2 - 1 ), d0, d1, d2 );
	        add_opt0_layout_coord_( coords, "y1", 0, std::min<std::size_t>( 1, d1 - 1 ), 0, d0, d1, d2 );
	        add_opt0_layout_coord_( coords, "x1", std::min<std::size_t>( 1, d0 - 1 ), 0, 0, d0, d1, d2 );
	        add_opt0_layout_coord_( coords, "middle", d0 / 2, d1 / 2, d2 / 2, d0, d1, d2 );
	        add_opt0_layout_coord_( coords, "last", d0 - 1, d1 - 1, d2 - 1, d0, d1, d2 );
	        return coords;
	    }

	    void append_native_opt0_layout_offset_row_(
	        const char *direction, const char *role, const char *buffer_name, const char *layout_name,
	        const char *sample, std::size_t c0, std::size_t c1, std::size_t c2, std::size_t d0,
	        std::size_t d1, std::size_t d2, const void *base_ptr, std::size_t offset_elems,
	        const char *notes
	    ) const
	    {
	        const std::string header =
	            "source,run_label,rank,direction,role,buffer_name,layout_name,sample,coord0,coord1,coord2,"
	            "dim0,dim1,dim2,base_token,offset_elems,token,notes,directory";
	        const std::uintptr_t base_token = reinterpret_cast<std::uintptr_t>( base_ptr );
	        const std::uintptr_t offset_bytes =
	            checked_mul_bytes_( offset_elems, sizeof( value_type ), "native opt0 layout offset bytes" );
	        std::ostringstream row;
	        row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << direction << ','
	            << role << ',' << buffer_name << ',' << layout_name << ',' << sample << ',' << c0 << ',' << c1
	            << ',' << c2 << ',' << d0 << ',' << d1 << ',' << d2 << ','
	            << static_cast<unsigned long long>( base_token ) << ',' << offset_elems << ','
	            << static_cast<unsigned long long>( base_token + offset_bytes ) << ',' << notes << ','
	            << local_fft_diagnostics_dir_;
	        append_local_fft_diagnostics_row_( "opt0_layout_offsets", header, row.str() );
	    }

	    template <class Array>
	    void append_native_opt0_array_layout_offsets_(
	        const char *direction, const char *role, const char *buffer_name, const char *layout_name,
	        const Array &array, std::size_t d0, std::size_t d1, std::size_t d2, const char *notes
	    ) const
	    {
	        for ( const auto &coord : opt0_layout_coords_( d0, d1, d2 ) )
	        {
	            const std::size_t offset = static_cast<std::size_t>( array.calc_lin_index(
	                static_cast<int>( coord.c0 ), static_cast<int>( coord.c1 ), static_cast<int>( coord.c2 )
	            ) );
	            append_native_opt0_layout_offset_row_(
	                direction, role, buffer_name, layout_name, coord.sample, coord.c0, coord.c1, coord.c2, d0,
	                d1, d2, array.raw_ptr(), offset, notes
	            );
	        }
	    }

	    template <class Ptr>
	    void append_native_opt0_y_plan_layout_offsets_(
	        const char *direction, const char *role, const char *buffer_name, Ptr base_ptr,
	        const std::vector<std::size_t> &offsets
	    ) const
	    {
	        if ( offsets.empty() )
	            return;
	        const std::size_t x_size = transpose1_dim_.size_x[myid_i_];
	        const std::size_t y_size = ny_global_;
	        const std::size_t z_size = transpose1_dim_.size_z[myid_j_];
	        if ( x_size == 0 || y_size == 0 || z_size == 0 )
	            return;

	        const std::size_t plan_count     = offsets.size();
	        const std::size_t batches_offset = x_size <= z_size ? z_size * y_size : 1;
	        const std::size_t istride        = z_size;
	        const std::size_t idist          = x_size <= z_size ? 1 : z_size * y_size;
	        const std::size_t batch          = std::max( x_size, z_size );

	        for ( const auto &coord : opt0_layout_coords_( plan_count, batch, y_size ) )
	        {
	            const std::size_t plan_index = coord.c0;
	            const std::size_t batch_index = coord.c1;
	            const std::size_t k_index = coord.c2;
	            const std::size_t batch_offset = checked_mul_bytes_(
	                batch_index, idist, "native opt0 layout Y batch offset"
	            );
	            const std::size_t k_offset = checked_mul_bytes_(
	                k_index, istride, "native opt0 layout Y k offset"
	            );
	            const std::size_t relative_offset = checked_add_bytes_(
	                batch_offset, k_offset, "native opt0 layout Y relative offset"
	            );
	            const std::size_t offset = checked_add_bytes_(
	                offsets[plan_index], relative_offset, "native opt0 layout Y absolute offset"
	            );
	            append_native_opt0_layout_offset_row_(
	                direction, role, buffer_name, "y_plan_offsets", coord.sample, plan_index, batch_index,
	                k_index, plan_count, batch, y_size, base_ptr, offset,
	                "coord0=plan_index;coord1=batch_index;coord2=y_index"
	            );
	        }

	        append_native_opt0_layout_offset_row_(
	            direction, role, buffer_name, "y_plan_metadata", "batches_offset", 0, 0, 0, plan_count, batch,
	            y_size, base_ptr, batches_offset, "offset_elems=batches_offset;istride=z_size;idist=see-meta"
	        );
	        append_native_opt0_layout_offset_row_(
	            direction, role, buffer_name, "y_plan_metadata", "istride", 0, 0, 0, plan_count, batch,
	            y_size, base_ptr, istride, "offset_elems=istride"
	        );
	        append_native_opt0_layout_offset_row_(
	            direction, role, buffer_name, "y_plan_metadata", "idist", 0, 0, 0, plan_count, batch,
	            y_size, base_ptr, idist, "offset_elems=idist"
	        );
	    }

	    template <class Stage0ZFast3, class Stage1ZFast3, class Stage1YFast3, class XStage3, class ComplexArray3>
	    void append_native_opt0_forward_layout_offsets_(
	        const Stage0ZFast3 &stage0, const Stage1ZFast3 &stage1, const Stage1YFast3 &stage1_yfft,
	        const XStage3 &x_stage, const ComplexArray3 &out, const std::vector<std::size_t> &y_plan_offsets
	    )
	    {
	        if ( !local_fft_diagnostics_enabled_ || !native_opt0_reference_y_buffer_topology_enabled_() ||
	             native_opt0_forward_layout_offsets_dumped_ )
	            return;
	        native_opt0_forward_layout_offsets_dumped_ = true;

	        append_native_opt0_array_layout_offsets_(
	            "forward", "r2c_output_complex", "stage0_default_z", "zfast_210", stage0,
	            half_input_dim_.size_x[myid_i_], half_input_dim_.size_y[myid_j_], nz_global_, "native R2C scratch"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "forward", "first_transpose_output_temp", "stage1_zfast", "zfast_210", stage1,
	            half_input_dim_.size_x[myid_i_], ny_global_, transpose1_dim_.size_z[myid_j_], "reference mem_d[0] alias"
	        );
	        append_native_opt0_y_plan_layout_offsets_(
	            "forward", "y_input_temp", "stage1_zfast", stage1.raw_ptr(), y_plan_offsets
	        );
	        append_native_opt0_y_plan_layout_offsets_(
	            "forward", "y_output_complex", "stage1_yfft_zfast", stage1_yfft.raw_ptr(), y_plan_offsets
	        );
	        append_native_opt0_array_layout_offsets_(
	            "forward", "second_transpose_input_complex", "stage1_yfft_zfast", "zfast_210", stage1_yfft,
	            half_input_dim_.size_x[myid_i_], ny_global_, transpose1_dim_.size_z[myid_j_],
	            "public output storage interpreted as reference complex before second transpose"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "forward", "second_transpose_output_temp", "x_fft_zfast", "zfast_210", x_stage,
	            output_dim_.size_x[0], output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_],
	            "reference mem_d[0] alias before X FFT"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "forward", "final_x_output_complex", "public_complex_out", "zfast_210", out,
	            output_dim_.size_x[0], output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_],
	            "FFTM public spectral output layout"
	        );
	    }

	    template <class ComplexArray3, class XStage3, class Stage1YFast3, class Stage1ZFast3, class Stage0ZFast3>
	    void append_native_opt0_backward_layout_offsets_(
	        const ComplexArray3 &in, const XStage3 &x_stage, const Stage1YFast3 &stage1_yfft,
	        const Stage1ZFast3 &stage1, const Stage0ZFast3 &stage0,
	        const std::vector<std::size_t> &y_plan_offsets
	    )
	    {
	        if ( !local_fft_diagnostics_enabled_ || !native_opt0_reference_y_buffer_topology_enabled_() ||
	             native_opt0_backward_layout_offsets_dumped_ )
	            return;
	        native_opt0_backward_layout_offsets_dumped_ = true;

	        append_native_opt0_array_layout_offsets_(
	            "backward", "public_complex_input", "public_complex_in", "zfast_210", in,
	            output_dim_.size_x[0], output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_],
	            "FFTM public spectral input layout"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "backward", "inverse_x_output_temp", "x_fft_zfast", "zfast_210", x_stage,
	            output_dim_.size_x[0], output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_],
	            "reference mem_d[0] alias after inverse X"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "backward", "backward_second_output_y_input", "stage1_yfft_zfast", "zfast_210", stage1_yfft,
	            half_input_dim_.size_x[myid_i_], ny_global_, transpose1_dim_.size_z[myid_j_],
	            "public input storage interpreted as reference complex before inverse Y"
	        );
	        append_native_opt0_y_plan_layout_offsets_(
	            "backward", "y_input_complex", "stage1_yfft_zfast", stage1_yfft.raw_ptr(), y_plan_offsets
	        );
	        append_native_opt0_y_plan_layout_offsets_(
	            "backward", "y_output_temp", "stage1_zfast", stage1.raw_ptr(), y_plan_offsets
	        );
	        append_native_opt0_array_layout_offsets_(
	            "backward", "backward_y_output_first_transpose_input", "stage1_zfast", "zfast_210", stage1,
	            half_input_dim_.size_x[myid_i_], ny_global_, transpose1_dim_.size_z[myid_j_],
	            "reference send-safe temp slot before backward first transpose"
	        );
	        append_native_opt0_array_layout_offsets_(
	            "backward", "final_z_input_complex", "stage0_default_z", "zfast_210", stage0,
	            half_input_dim_.size_x[myid_i_], half_input_dim_.size_y[myid_j_], nz_global_,
	            "native C2R input scratch"
	        );
	    }

	    bool should_dump_native_opt0_y_execution_diagnostics_( const char *profile_name, const char *plan_label )
	    {
        if ( !local_fft_diagnostics_enabled_ || !native_opt0_default_z_layout_ )
            return false;
        const std::string profile = profile_name == nullptr ? "" : profile_name;
        const std::string label   = plan_label == nullptr ? "" : plan_label;
        const bool        forward = profile.find( "forward_y" ) != std::string::npos ||
                             label.find( "forward_y" ) != std::string::npos;
        const bool backward = profile.find( "backward_y" ) != std::string::npos ||
                              profile.find( "inverse_y" ) != std::string::npos ||
                              label.find( "inverse_y" ) != std::string::npos;
        if ( forward )
        {
            if ( native_opt0_forward_y_execution_diagnostics_dumped_ )
                return false;
            native_opt0_forward_y_execution_diagnostics_dumped_ = true;
            return true;
        }
        if ( backward )
        {
            if ( native_opt0_backward_y_execution_diagnostics_dumped_ )
                return false;
            native_opt0_backward_y_execution_diagnostics_dumped_ = true;
            return true;
        }
        return false;
    }

    template <class InPtr, class OutPtr>
    void append_native_opt0_y_execution_diagnostics_(
        const char *profile_name, const char *plan_label, const std::vector<std::string> &plan_names,
        const std::vector<std::size_t> &offsets, InPtr in, OutPtr out, plan_sequence_id_t plan_sequence,
        bool use_cached_sequence, bool use_explicit_exec_direction, direction exec_direction
    )
    {
        if ( plan_names.empty() ||
             !should_dump_native_opt0_y_execution_diagnostics_( profile_name, plan_label ) )
        {
            return;
        }
        const std::string header =
            "source,run_label,rank,profile_name,plan_label,sequence_id,sequence_kind,executor,sync_mode,exec_check,"
            "plan_index,plan_name,offset_elems,input_base_token,output_base_token,input_token,output_token,"
            "work_area_token,work_area_offset_bytes,stream_token,handle_mode,plan_direction,exec_direction,"
            "backend_fft_type,rank_fft,n0,inembed0,"
            "istride,idist,onembed0,ostride,odist,batch,work_size_bytes,directory";

        const std::uintptr_t input_base  = reinterpret_cast<std::uintptr_t>( in );
        const std::uintptr_t output_base = reinterpret_cast<std::uintptr_t>( out );
        const std::uintptr_t work_base   = reinterpret_cast<std::uintptr_t>( external_work_area_ );
        const char          *executor    = use_cached_sequence
                                               ? ( use_native_opt0_raw_y_plan_array_executor_ ? "opaque-raw" :
                                                   use_native_opt0_tight_y_plan_sequence_ ? "tight" : "cached-virtual" )
                                               : "raw-loop";
        const char *sync_mode      = native_opt0_y_group_device_sync_enabled_() ? "device" : "per-stream";
        const char *exec_check     = use_native_opt0_y_no_sync_exec_ ? "no-sync" : "sync";
        const char *handle_mode    = use_native_opt0_shared_y_plan_handles_ ? "shared-bidir" : "separate";
        const char *exec_dir_label = use_explicit_exec_direction ? local_fft_direction_name_( exec_direction ) : "plan";

        for ( std::size_t i = 0; i < plan_names.size(); ++i )
        {
            const auto desc = base_fft_.plan_descriptor( plan_names[i] );
            const std::uintptr_t offset_bytes =
                checked_mul_bytes_( offsets[i], sizeof( value_type ), "native opt0 y diagnostic data offset" );
            long long work_area_offset = -1;
            if ( work_base != 0 && desc.work_area_token >= work_base )
            {
                const std::uintptr_t diff = desc.work_area_token - work_base;
                if ( diff <= static_cast<std::uintptr_t>( std::numeric_limits<long long>::max() ) )
                    work_area_offset = static_cast<long long>( diff );
            }

            std::ostringstream row;
            row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << profile_name << ','
                << plan_label << ',';
            if ( plan_sequence == base_fft_t::invalid_plan_sequence_id() )
                row << -1 << ",invalid,";
            else
                row << static_cast<unsigned long long>( plan_sequence ) << ','
                    << base_fft_.plan_sequence_kind_name( plan_sequence ) << ',';
            row << executor << ',' << sync_mode << ',' << exec_check << ',' << i << ',' << plan_names[i] << ',' << offsets[i] << ','
                << static_cast<unsigned long long>( input_base ) << ','
                << static_cast<unsigned long long>( output_base ) << ','
                << static_cast<unsigned long long>( input_base + offset_bytes ) << ','
                << static_cast<unsigned long long>( output_base + offset_bytes ) << ','
                << static_cast<unsigned long long>( desc.work_area_token ) << ',' << work_area_offset << ','
                << static_cast<unsigned long long>( desc.stream_token ) << ',' << handle_mode << ','
                << local_fft_direction_name_( desc.dir ) << ',' << exec_dir_label
                << ',' << desc.backend_type << ',' << desc.rank << ',' << desc.n[0] << ',' << desc.inembed[0] << ','
                << desc.istride << ',' << desc.idist << ',' << desc.onembed[0] << ',' << desc.ostride << ','
                << desc.odist << ',' << desc.batch << ',' << desc.work_size << ',' << local_fft_diagnostics_dir_;
            append_local_fft_diagnostics_row_( "native_opt0_y_execution", header, row.str() );
        }
    }

    bool should_dump_native_opt0_y_buffer_signature_( const char *profile_name, const char *plan_label )
    {
        if ( !local_fft_diagnostics_enabled_ || !native_opt0_default_z_layout_ )
            return false;
        const std::string profile = profile_name == nullptr ? "" : profile_name;
        const std::string label   = plan_label == nullptr ? "" : plan_label;
        const bool        forward = profile.find( "forward_y" ) != std::string::npos ||
                             label.find( "forward_y" ) != std::string::npos;
        const bool backward = profile.find( "backward_y" ) != std::string::npos ||
                              profile.find( "inverse_y" ) != std::string::npos ||
                              label.find( "inverse_y" ) != std::string::npos;
        if ( forward )
        {
            if ( native_opt0_forward_y_buffer_signature_dumped_ )
                return false;
            native_opt0_forward_y_buffer_signature_dumped_ = true;
            return true;
        }
        if ( backward )
        {
            if ( native_opt0_backward_y_buffer_signature_dumped_ )
                return false;
            native_opt0_backward_y_buffer_signature_dumped_ = true;
            return true;
        }
        return false;
    }

    static void add_signature_index_( std::vector<std::size_t> &indices, std::size_t count, std::size_t index )
    {
        if ( count == 0 || index >= count )
            return;
        indices.push_back( index );
    }

    static std::vector<std::size_t> signature_indices_( std::size_t count )
    {
        std::vector<std::size_t> indices;
        if ( count == 0 )
            return indices;
        add_signature_index_( indices, count, 0 );
        add_signature_index_( indices, count, 1 );
        add_signature_index_( indices, count, 2 );
        add_signature_index_( indices, count, 3 );
        add_signature_index_( indices, count, count / 4 );
        add_signature_index_( indices, count, count / 2 );
        add_signature_index_( indices, count, ( 3 * count ) / 4 );
        if ( count >= 4 )
        {
            add_signature_index_( indices, count, count - 4 );
            add_signature_index_( indices, count, count - 3 );
            add_signature_index_( indices, count, count - 2 );
        }
        add_signature_index_( indices, count, count - 1 );
        std::sort( indices.begin(), indices.end() );
        indices.erase( std::unique( indices.begin(), indices.end() ), indices.end() );
        return indices;
    }

    static std::vector<std::size_t> signature_indices_from_signed_( long long int count )
    {
        if ( count <= 0 )
            return std::vector<std::size_t>();
        return signature_indices_( static_cast<std::size_t>( count ) );
    }

    static double complex_real_( const value_type &value )
    {
        return static_cast<double>( value.x );
    }

    static double complex_imag_( const value_type &value )
    {
        return static_cast<double>( value.y );
    }

    template <class Ptr>
    void append_native_opt0_y_buffer_signature_(
        const char *profile_name, const char *plan_label, const char *phase, const char *buffer_role,
        const std::vector<std::string> &plan_names, const std::vector<std::size_t> &offsets, Ptr buffer_base
    )
    {
        if ( !local_fft_diagnostics_enabled_ || plan_names.empty() || plan_names.size() != offsets.size() )
            return;

        const std::string header =
            "source,run_label,rank,profile_name,plan_label,phase,buffer_role,plan_index,plan_name,"
            "base_token,plan_token,offset_elems,n,istride,idist,batch,sample_count,sum_re,sum_im,"
            "sumsq_abs,l2_abs,min_abs,max_abs,first_re,first_im,mid_re,mid_im,last_re,last_im,directory";

        const std::vector<std::size_t> plan_indices = signature_indices_( plan_names.size() );
        const std::uintptr_t          base_token   = reinterpret_cast<std::uintptr_t>( buffer_base );

        for ( const std::size_t plan_index : plan_indices )
        {
            const auto desc = base_fft_.plan_descriptor( plan_names[plan_index] );
            const std::vector<std::size_t> batch_indices = signature_indices_from_signed_( desc.batch );
            const std::vector<std::size_t> k_indices     = signature_indices_from_signed_( desc.n[0] );
            if ( batch_indices.empty() || k_indices.empty() || desc.istride <= 0 || desc.idist <= 0 )
                continue;

            double      sum_re      = 0.0;
            double      sum_im      = 0.0;
            double      sumsq_abs   = 0.0;
            double      min_abs     = std::numeric_limits<double>::infinity();
            double      max_abs     = 0.0;
            double      first_re    = 0.0;
            double      first_im    = 0.0;
            double      mid_re      = 0.0;
            double      mid_im      = 0.0;
            double      last_re     = 0.0;
            double      last_im     = 0.0;
            std::size_t sample_count = 0;

            const std::size_t mid_batch = batch_indices[batch_indices.size() / 2];
            const std::size_t mid_k     = k_indices[k_indices.size() / 2];
            for ( const std::size_t batch_index : batch_indices )
            {
                for ( const std::size_t k_index : k_indices )
                {
                    const std::size_t batch_offset = checked_mul_bytes_(
                        batch_index, static_cast<std::size_t>( desc.idist ),
                        "native opt0 y signature batch offset"
                    );
                    const std::size_t k_offset = checked_mul_bytes_(
                        k_index, static_cast<std::size_t>( desc.istride ),
                        "native opt0 y signature element offset"
                    );
                    const std::size_t relative_offset = checked_add_bytes_(
                        batch_offset, k_offset, "native opt0 y signature relative offset"
                    );
                    const std::size_t absolute_offset = checked_add_bytes_(
                        offsets[plan_index], relative_offset, "native opt0 y signature absolute offset"
                    );

                    value_type sample;
                    runtime_api_t::memcpy(
                        &sample, buffer_base + absolute_offset, sizeof( value_type ),
                        runtime_api_t::device_to_host_kind()
                    );
                    const double re  = complex_real_( sample );
                    const double im  = complex_imag_( sample );
                    const double mag = std::sqrt( re * re + im * im );
                    sum_re += re;
                    sum_im += im;
                    sumsq_abs += re * re + im * im;
                    min_abs = std::min( min_abs, mag );
                    max_abs = std::max( max_abs, mag );
                    if ( sample_count == 0 )
                    {
                        first_re = re;
                        first_im = im;
                    }
                    if ( batch_index == mid_batch && k_index == mid_k )
                    {
                        mid_re = re;
                        mid_im = im;
                    }
                    last_re = re;
                    last_im = im;
                    ++sample_count;
                }
            }
            if ( sample_count == 0 )
            {
                min_abs = 0.0;
            }

            const std::uintptr_t offset_bytes = checked_mul_bytes_(
                offsets[plan_index], sizeof( value_type ), "native opt0 y signature data offset"
            );
            std::ostringstream row;
            row << "fftm-native," << local_fft_diagnostics_label_ << ',' << mpi_.myid << ',' << profile_name << ','
                << plan_label << ',' << phase << ',' << buffer_role << ',' << plan_index << ',' << plan_names[plan_index]
                << ',' << static_cast<unsigned long long>( base_token ) << ','
                << static_cast<unsigned long long>( base_token + offset_bytes ) << ',' << offsets[plan_index] << ','
                << desc.n[0] << ',' << desc.istride << ',' << desc.idist << ',' << desc.batch << ','
                << sample_count << ',' << sum_re << ',' << sum_im << ',' << sumsq_abs << ','
                << std::sqrt( sumsq_abs ) << ',' << min_abs << ',' << max_abs << ',' << first_re << ',' << first_im
                << ',' << mid_re << ',' << mid_im << ',' << last_re << ',' << last_im << ','
                << local_fft_diagnostics_dir_;
            append_local_fft_diagnostics_row_( "native_opt0_y_signature", header, row.str() );
        }
    }

    static const char *local_fft_direction_name_( direction dir )
    {
        switch ( dir )
        {
        case direction::R2C:
            return "R2C";
        case direction::C2R:
            return "C2R";
        case direction::C2CF:
            return "C2CF";
        case direction::C2CB:
            return "C2CB";
        }
        return "unknown";
    }

#endif
