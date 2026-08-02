    int   pencil_layout_selector_   = 0;
    void *external_work_area_       = nullptr;
    void *external_host_work_area_  = nullptr;

    mpi_transpose_3d_mode mode_ = mpi_transpose_3d_mode::p2p_waitany;
    int myid_i_ = 0;
    int myid_j_ = 0;

    std::size_t nx_local_        = 0;
    std::size_t ny_input_local_  = 0;
    std::size_t ny_output_local_ = 0;
    std::size_t ny_global_       = 0;
    std::size_t nz_global_       = 0;
    std::size_t nz_local_        = 0;
    std::size_t nx_global_       = 0;

    partition_t half_input_dim_;
    partition_t transpose1_dim_;
    partition_t output_dim_;

    std::unique_ptr<mpi_comm_t> row_comm_;
    std::unique_ptr<mpi_comm_t> line_comm_;
    scfd::communication::mpi_comm_info row_comm_info_;
    scfd::communication::mpi_comm_info line_comm_info_;

    contiguous_buf_t send_buffer_;
    contiguous_buf_t recv_buffer_;
    std::size_t send_buffer_elems_ = 0;
    std::size_t recv_buffer_elems_ = 0;
    host_buf_t host_send_buffer_;
    host_buf_t host_recv_buffer_;
    std::size_t host_send_buffer_elems_ = 0;
    std::size_t host_recv_buffer_elems_ = 0;
