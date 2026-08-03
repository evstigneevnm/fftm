    void copy_second_forward_chunk_to_xstage_async_(
        const value_type *src_ptr, std::size_t src_x_size, std::size_t dst_x_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, src_x_size * sizeof( value_type ), src_x_size, nz_local_ );
        params.destination_position = runtime_api_t::make_pos( dst_x_offset * sizeof( value_type ), 0, 0 );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, nx_global_ * sizeof( value_type ), nx_global_, nz_local_ );
        params.extent = runtime_api_t::make_extent( src_x_size * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void pack_second_forward_zfast_chunk_async_(
        const value_type *src_ptr, int target_i, value_type *dst_ptr, typename runtime_api_t::memcpy_kind_t kind,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( 0, output_dim_.start_y[target_i], 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.destination_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.destination = runtime_api_t::make_pitched_ptr(
            dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, output_dim_.size_y[target_i]
        );
        params.extent =
            runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), output_dim_.size_y[target_i], nx_local_ );
        params.kind = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_second_forward_zfast_recv_to_output_async_(
        const value_type *src_ptr, std::size_t src_x_size, std::size_t dst_x_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.destination_position = runtime_api_t::make_pos( 0, 0, dst_x_offset );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), ny_output_local_, src_x_size );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_second_forward_zfast_stage1_to_output_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t src_y_size, std::size_t dst_x_offset,
        value_type *dst_ptr, typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( 0, src_y_offset, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.destination_position = runtime_api_t::make_pos( 0, 0, dst_x_offset );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), src_y_size, nx_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void pack_second_backward_chunk_async_(
        const value_type *src_ptr, std::size_t src_x_offset, std::size_t packed_x_size, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( src_x_offset * sizeof( value_type ), 0, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, nx_global_ * sizeof( value_type ), nx_global_, nz_local_ );
        params.destination_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, packed_x_size * sizeof( value_type ), packed_x_size, nz_local_ );
        params.extent = runtime_api_t::make_extent( packed_x_size * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void pack_second_backward_zfast_chunk_async_(
        const value_type *src_ptr, std::size_t src_x_offset, std::size_t packed_x_size, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( 0, 0, src_x_offset );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.destination_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_output_local_ );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), ny_output_local_, packed_x_size );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_second_backward_zfast_recv_to_stage1_async_(
        const value_type *src_ptr, std::size_t src_y_size, std::size_t dst_y_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, src_y_size );
        params.destination_position = runtime_api_t::make_pos( 0, dst_y_offset, 0 );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), src_y_size, nx_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }
