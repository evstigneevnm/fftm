    std::size_t max_first_stage_send_elems_() const
    {
        return nx_local_ * ny_input_local_ * nz_global_;
    }

    std::size_t max_first_stage_recv_elems_() const
    {
        return nx_local_ * ny_global_ * nz_local_;
    }

    std::size_t max_second_stage_send_elems_() const
    {
        return nx_local_ * ny_global_ * nz_local_;
    }

    std::size_t max_second_stage_recv_elems_() const
    {
        return nx_global_ * ny_output_local_ * nz_local_;
    }

    std::size_t max_redistribution_buffer_elems_() const
    {
        return std::max(
            std::max( max_first_stage_send_elems_(), max_first_stage_recv_elems_() ),
            std::max( max_second_stage_send_elems_(), max_second_stage_recv_elems_() )
        );
    }
