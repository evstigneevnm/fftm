    int first_forward_byte_recv_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : row_comm_order_ )
            total_chunks += contiguous_forward_byte_request_count_(
                bytes_from_elems_( first_forward_recv_elems_( p ) )
            );
        return total_chunks;
    }

    int second_forward_byte_recv_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : line_comm_order_ )
            total_chunks += contiguous_forward_byte_request_count_(
                bytes_from_elems_( second_forward_recv_elems_( p ) )
            );
        return total_chunks;
    }

    int second_backward_byte_recv_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : line_comm_order_ )
            total_chunks += mpi_byte_request_count_( bytes_from_elems_( second_backward_recv_elems_( p ) ) );
        return total_chunks;
    }

    int first_backward_byte_recv_chunks_() const
    {
        int total_chunks = 0;
        for ( const int p : row_comm_order_ )
            total_chunks += mpi_byte_request_count_( bytes_from_elems_( first_backward_recv_elems_( p ) ) );
        return total_chunks;
    }

    void synchronize_streams_( int nstreams, const char *label )
    {
        auto scope = profile_scope_( label );
        for ( int i = 0; i < nstreams; ++i )
        {
            runtime_api_t::stream_synchronize( streams_[i].stream() );
        }
    }

    void reset_row_requests_()
    {
        ensure_no_deferred_send_( "row request reset" );
        std::fill( row_send_requests_.begin(), row_send_requests_.end(), mpi_request_t() );
        std::fill( row_recv_requests_.begin(), row_recv_requests_.end(), mpi_request_t() );
    }

    void reset_line_requests_()
    {
        ensure_no_deferred_send_( "line request reset" );
        std::fill( line_send_requests_.begin(), line_send_requests_.end(), mpi_request_t() );
        std::fill( line_recv_requests_.begin(), line_recv_requests_.end(), mpi_request_t() );
    }

    void build_peer_orders_()
    {
        if ( native_plan_state_bound_ )
        {
            row_comm_order_  = native_plan_state_.first_transpose_order();
            line_comm_order_ = native_plan_state_.second_transpose_order();
            return;
        }

        row_comm_order_.clear();
        row_comm_order_.reserve( std::max( 0, row_comm_info_.num_procs - 1 ) );
        for ( int step = 1; step < row_comm_info_.num_procs; ++step )
        {
            row_comm_order_.push_back( ( myid_j_ + step ) % row_comm_info_.num_procs );
        }
        if ( pencil_layout_selector_ == 2 && !reference_parity_enabled_ )
            std::reverse( row_comm_order_.begin(), row_comm_order_.end() );

        line_comm_order_.clear();
        line_comm_order_.reserve( std::max( 0, line_comm_info_.num_procs - 1 ) );
        for ( int step = 1; step < line_comm_info_.num_procs; ++step )
        {
            line_comm_order_.push_back( ( myid_i_ + step ) % line_comm_info_.num_procs );
        }
        if ( pencil_layout_selector_ == 2 && !reference_parity_enabled_ )
            std::reverse( line_comm_order_.begin(), line_comm_order_.end() );
    }
