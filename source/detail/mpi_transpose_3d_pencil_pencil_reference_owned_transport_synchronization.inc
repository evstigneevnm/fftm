    void profile_chunk_count_( const char *label, int send_requests, int recv_requests )
    {
        {
            std::ostringstream ss;
            ss << label << "/chunk_count/send=" << send_requests << "/recv=" << recv_requests;
            auto phase = profile_scope_( ss.str() );
        }
        {
            std::ostringstream ss;
            ss << label << "/byte_requests/send=" << send_requests << "/recv=" << recv_requests;
            auto phase = profile_scope_( ss.str() );
        }
    }

	    void synchronize_before_direct_byte_sends_( const char *label )
	    {
	#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
	        if ( cuda_aware_byte_p2p_enabled_() && !reference_parity_byte_path_enabled_() )
	        {
	            auto phase = profile_scope_( label );
	            runtime_api_t::device_synchronize();
	        }
#else
        (void)label;
	#endif
	    }

	    void synchronize_after_local_fft_if_needed_()
	    {
	#ifdef FFTM_ENABLE_DEVICE_AWARE_MPI
	        if ( reference_parity_byte_path_enabled_() )
	            runtime_api_t::device_synchronize();
	#endif
	    }
