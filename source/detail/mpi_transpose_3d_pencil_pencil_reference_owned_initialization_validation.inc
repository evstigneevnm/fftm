    void validate_partitions_() const
    {
        if ( half_input_dim_.size_z.size() != 1 )
        {
            throw std::logic_error( "owned reference pencil first input must keep Z undistributed" );
        }
        if ( transpose1_dim_.size_y.size() != 1 )
        {
            throw std::logic_error( "owned reference pencil middle layout must keep Y undistributed" );
        }
        if ( output_dim_.size_x.size() != 1 )
        {
            throw std::logic_error( "owned reference pencil output layout must keep X undistributed" );
        }
        if ( half_input_dim_.size_y.size() != transpose1_dim_.size_z.size() )
        {
            throw std::logic_error( "owned reference pencil first redistribution partition mismatch" );
        }
        if ( transpose1_dim_.size_x.size() != output_dim_.size_y.size() )
        {
            throw std::logic_error( "owned reference pencil second redistribution partition mismatch" );
        }
        if ( half_input_dim_.size_x != transpose1_dim_.size_x )
        {
            throw std::logic_error( "owned reference pencil first redistribution must preserve X ownership" );
        }
        if ( transpose1_dim_.size_z != output_dim_.size_z )
        {
            throw std::logic_error( "owned reference pencil second redistribution must preserve Z ownership" );
        }
    }

    void ensure_inited_( mpi_transpose_3d_mode mode ) const
    {
        if ( !is_inited_ )
        {
            throw std::logic_error( "owned reference pencil pipeline is not initialized" );
        }
        if ( mode != mode_ )
        {
            throw std::logic_error( "owned reference pencil pipeline mode mismatch" );
        }
        if ( use_external_work_area_ && external_work_area_ == nullptr )
        {
            throw std::logic_error( "owned reference pencil pipeline external work area is not bound" );
        }
#ifndef FFTM_ENABLE_DEVICE_AWARE_MPI
        if ( use_external_host_work_area_ && get_host_work_size_bytes() != 0 && external_host_work_area_ == nullptr )
        {
            throw std::logic_error( "owned reference pencil pipeline external host work area is not bound" );
        }
#endif
        (void)use_p2p_send_thread_;
    }
