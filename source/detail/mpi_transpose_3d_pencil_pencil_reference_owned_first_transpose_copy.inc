    void copy_first_forward_chunk_to_stage1_async_(
        const value_type *src_ptr, std::size_t src_y_size, std::size_t dst_y_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, src_y_size * sizeof( value_type ), src_y_size, nx_local_ );
        params.destination_position = runtime_api_t::make_pos( dst_y_offset * sizeof( value_type ), 0, 0 );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, ny_global_ * sizeof( value_type ), ny_global_, nx_local_ );
        params.extent = runtime_api_t::make_extent( src_y_size * sizeof( value_type ), nx_local_, nz_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_first_forward_zfast_source_to_send_async_(
        const value_type *src_ptr, int target_j, value_type *dst_ptr, typename runtime_api_t::memcpy_kind_t kind,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( transpose1_dim_.start_z[target_j] * sizeof( value_type ), 0, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_global_ * sizeof( value_type ), nz_global_, ny_input_local_ );
        params.destination_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.destination = runtime_api_t::make_pitched_ptr(
            dst_ptr, transpose1_dim_.size_z[target_j] * sizeof( value_type ), transpose1_dim_.size_z[target_j],
            ny_input_local_
        );
        params.extent =
            runtime_api_t::make_extent( transpose1_dim_.size_z[target_j] * sizeof( value_type ), ny_input_local_, nx_local_ );
        params.kind = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_first_forward_zfast_source_to_stage1_async_(
        const value_type *src_ptr, std::size_t src_z_offset, std::size_t src_z_size, std::size_t dst_y_offset,
        value_type *dst_ptr, typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( src_z_offset * sizeof( value_type ), 0, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_global_ * sizeof( value_type ), nz_global_, ny_input_local_ );
        params.destination_position = runtime_api_t::make_pos( 0, dst_y_offset, 0 );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.extent = runtime_api_t::make_extent( src_z_size * sizeof( value_type ), ny_input_local_, nx_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_first_forward_zfast_recv_to_stage1_async_(
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

    void pack_first_backward_chunk_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t packed_y_size, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( src_y_offset * sizeof( value_type ), 0, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, ny_global_ * sizeof( value_type ), ny_global_, nx_local_ );
        params.destination_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, packed_y_size * sizeof( value_type ), packed_y_size, nx_local_ );
        params.extent = runtime_api_t::make_extent( packed_y_size * sizeof( value_type ), nx_local_, nz_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void pack_first_backward_zfast_chunk_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t packed_y_size, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( 0, src_y_offset, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.destination_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_local_ * sizeof( value_type ), nz_local_, packed_y_size );
        params.extent = runtime_api_t::make_extent( nz_local_ * sizeof( value_type ), packed_y_size, nx_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_first_backward_zfast_recv_to_default_z_async_(
        const value_type *src_ptr, std::size_t src_z_size, std::size_t dst_z_offset, value_type *dst_ptr,
        typename runtime_api_t::memcpy_kind_t kind, typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( 0, 0, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, src_z_size * sizeof( value_type ), src_z_size, ny_input_local_ );
        params.destination_position = runtime_api_t::make_pos( dst_z_offset * sizeof( value_type ), 0, 0 );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_global_ * sizeof( value_type ), nz_global_, ny_input_local_ );
        params.extent = runtime_api_t::make_extent( src_z_size * sizeof( value_type ), ny_input_local_, nx_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }

    void copy_first_backward_zfast_stage1_to_default_z_async_(
        const value_type *src_ptr, std::size_t src_y_offset, std::size_t src_y_size, std::size_t dst_z_offset,
        std::size_t dst_z_size, value_type *dst_ptr, typename runtime_api_t::memcpy_kind_t kind,
        typename runtime_api_t::stream_t stream
    ) const
    {
        typename runtime_api_t::memcpy_3d_params_t params = {};
        params.source_position = runtime_api_t::make_pos( 0, src_y_offset, 0 );
        params.source =
            runtime_api_t::make_pitched_ptr( src_ptr, nz_local_ * sizeof( value_type ), nz_local_, ny_global_ );
        params.destination_position = runtime_api_t::make_pos( dst_z_offset * sizeof( value_type ), 0, 0 );
        params.destination =
            runtime_api_t::make_pitched_ptr( dst_ptr, nz_global_ * sizeof( value_type ), nz_global_, ny_input_local_ );
        params.extent = runtime_api_t::make_extent( dst_z_size * sizeof( value_type ), src_y_size, nx_local_ );
        params.kind   = kind;
        runtime_api_t::memcpy_3d_async( &params, stream );
    }
