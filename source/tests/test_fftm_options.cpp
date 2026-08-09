#include <cassert>
#include <stdexcept>

#include <fftm_options.hpp>

int main()
{
    {
        const fftm::fftm_init_options options;
        assert( options.reporting.profiling_key.empty() );
        assert( options.reporting.memory_profiling_key.empty() );
        assert( !options.reporting.verbose );
        assert( !options.reporting.print_profile_summary_on_destroy );
        assert( !options.diagnostics.enable_native_stage_timers );
        assert( !options.diagnostics.allow_native_opt0_diagnostic_variants );
        assert( !options.execution.use_4d_pencil_node_aligned_wz_pipeline );
        assert( !options.diagnostics.use_4d_pencil_p3_degenerate_wz_pipeline );
        assert( !options.diagnostics.use_4d_slab_native_wz_send_overlap );
        assert( !options.diagnostics.use_4d_slab_native_wz_send_overlap_nonblocking_stream );
        assert( options.execution.slab_native_wz_backward_plane_credit_window == 0 );
        assert( !options.execution.use_4d_slab_native_wz_backward_cyclic_peer_order );
    }

    {
        const auto reporting = fftm::profiling_reporting_options();
        assert( reporting.profiling_key == "fftm_prof" );
        assert( reporting.memory_profiling_key == "fftm_mem" );
        assert( reporting.print_profile_summary_on_destroy );
        assert( reporting.print_profile_totals_on_destroy );
        assert( reporting.print_memory_profile_on_destroy );
        assert( reporting.print_memory_totals_on_destroy );
    }

    {
        const auto options = fftm::production_options_3d(
            fftm::transform_strategy_3d::pencil_pencil, 8, true
        );
        assert( options.pencil_layout_3d == fftm::fftm_3d_pencil_layout::opt0 );
        assert( options.pencil_pipeline_3d == fftm::fftm_3d_pencil_pipeline::reference_parity );
        assert( options.execution.use_p2p_byte_transfer );
        assert( options.execution.use_native_opt0_default_z_layout );
        assert( options.execution.use_native_opt0_reference_y_buffer_topology );
        assert( options.execution.use_native_opt0_tight_y_plan_sequence );
        assert( options.execution.use_native_opt0_shared_y_plan_handles );
        assert( options.execution.use_native_opt0_y_group_device_sync );
        assert( options.execution.use_native_opt0_y_no_sync_exec );
        assert( options.execution.use_native_opt0_raw_y_plan_array_executor );
        assert( !options.diagnostics.use_fft_exec_no_sync );
    }

    {
        const auto options = fftm::production_options_3d(
            fftm::transform_strategy_3d::pencil_pencil, 6, true
        );
        assert( options.pencil_layout_3d == fftm::fftm_3d_pencil_layout::opt1 );
        assert( !options.execution.use_native_opt0_default_z_layout );
    }

    {
        const auto options = fftm::production_options_3d(
            fftm::transform_strategy_3d::pencil_pencil, 64, true
        );
        assert( options.pencil_layout_3d == fftm::fftm_3d_pencil_layout::opt0 );
        assert( options.pencil_pipeline_3d == fftm::fftm_3d_pencil_pipeline::reference_parity );
        assert( options.execution.use_native_opt0_default_z_layout );
        assert( options.execution.use_native_opt0_reference_y_buffer_topology );
        assert( options.execution.use_native_opt0_tight_y_plan_sequence );
        assert( options.execution.use_native_opt0_shared_y_plan_handles );
        assert( options.execution.use_native_opt0_y_group_device_sync );
        assert( options.execution.use_native_opt0_y_no_sync_exec );
        assert( options.execution.use_native_opt0_raw_y_plan_array_executor );
    }

    {
        const auto options = fftm::production_options_3d(
            fftm::transform_strategy_3d::pencil_pencil, 64, false
        );
        assert( options.pencil_layout_3d == fftm::fftm_3d_pencil_layout::opt1 );
        assert( options.pencil_pipeline_3d == fftm::fftm_3d_pencil_pipeline::reference );
        assert( !options.execution.use_native_opt0_default_z_layout );
    }

    {
        const auto options = fftm::production_options_3d(
            fftm::transform_strategy_3d::pencil_pencil, 8, false
        );
        assert( options.pencil_layout_3d == fftm::fftm_3d_pencil_layout::opt1 );
        assert( options.pencil_pipeline_3d == fftm::fftm_3d_pencil_pipeline::reference );
        assert( !options.execution.direct_p2p_cuda_aware );
        assert( !options.execution.use_p2p_byte_transfer );
    }

    {
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::slab_slab,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true
        );
        assert( options.execution.use_4d_slab_native_xw_transpose );
        assert( options.execution.use_4d_native_xw_direct_layout );
        assert( options.execution.use_4d_native_xw_chunked_transport );
        assert( options.execution.native_xw_chunk_window == 1 );
        assert( options.execution.use_4d_native_xw_compact_staging );
        assert( options.execution.use_4d_slab_native_work_area_alias );
        assert( options.execution.use_4d_slab_native_wz_communication_layout );
        assert( options.execution.slab_native_wz_plan_concurrency == 4 );
        assert( options.execution.use_4d_slab_native_wz_ready_pipeline );
        assert( !options.diagnostics.use_4d_slab_native_wz_send_overlap );
        assert( !options.diagnostics.use_4d_slab_native_wz_send_overlap_nonblocking_stream );
        assert( options.execution.slab_native_wz_backward_plane_credit_window == 0 );
        assert( !options.execution.use_4d_slab_native_wz_backward_cyclic_peer_order );
    }

    {
        auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::slab_slab,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true
        );
        options.execution.slab_native_wz_backward_plane_credit_window = 8;
        assert( options.execution.slab_native_wz_plan_concurrency == 4 );
        assert( options.execution.slab_native_wz_backward_plane_credit_window >
                options.execution.slab_native_wz_plan_concurrency );
    }

    {
        const fftm::fftm_4d_production_topology topology( 16, 8, 1, 16, 1 );
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::slab_slab,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true,
            topology
        );
        assert( options.execution.slab_native_wz_backward_plane_credit_window == 8 );
        assert( !options.execution.use_4d_slab_native_wz_backward_cyclic_peer_order );
    }

    {
        const fftm::fftm_4d_production_topology topology( 24, 8, 1, 24, 1 );
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::slab_slab,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true,
            topology
        );
        assert( options.execution.slab_native_wz_backward_plane_credit_window == 6 );
        assert( options.execution.use_4d_slab_native_wz_backward_cyclic_peer_order );
    }

    {
        const fftm::fftm_4d_production_topology topology( 32, 8, 1, 32, 1 );
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::slab_slab,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true,
            topology
        );
        assert( options.execution.slab_native_wz_backward_plane_credit_window == 4 );
        assert( options.execution.use_4d_slab_native_wz_backward_cyclic_peer_order );
    }

    {
        const fftm::fftm_4d_production_topology topology(
            24, 8, 1, 24, 1, fftm::fftm_4d_pencil_pipeline::auto_select,
            fftm::fftm_4d_slab_backward_credit_policy::disabled
        );
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::slab_slab,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true,
            topology
        );
        assert( options.execution.slab_native_wz_backward_plane_credit_window == 0 );
        assert( !options.execution.use_4d_slab_native_wz_backward_cyclic_peer_order );
    }

    {
        const fftm::fftm_4d_production_topology topology( 24, 4, 1, 24, 1 );
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::slab_slab,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true,
            topology
        );
        assert( options.execution.slab_native_wz_backward_plane_credit_window == 0 );
    }

    {
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::slab_slab,
            fftm::fftm_4d_spectral_layout::public_yzwx,
            true
        );
        assert( options.execution.use_4d_slab_native_xw_transpose );
        assert( !options.execution.use_4d_native_xw_direct_layout );
        assert( !options.execution.use_4d_slab_native_wz_communication_layout );
    }

    {
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::pencil_pencil,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true
        );
        assert( options.execution.use_4d_pencil_same_zw_native_layout );
        assert( options.execution.use_4d_pencil_degenerate_local_transposes );
        assert( options.execution.use_4d_pencil_degenerate_same_xw_native );
        assert( options.execution.use_4d_pencil_degenerate_wz_sliced_z_fft );
        assert( !options.diagnostics.use_4d_pencil_degenerate_xw_slab_path );
        assert( !options.diagnostics.use_4d_pencil_p3_degenerate_wz_pipeline );
    }

    {
        const fftm::fftm_4d_production_topology topology( 16, 8, 2, 8, 1 );
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::pencil_pencil,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true,
            topology
        );
        assert( fftm::select_production_pencil_pipeline_4d(
                    topology, fftm::fftm_4d_spectral_layout::native_xzwy, true
                ) == fftm::fftm_4d_pencil_pipeline::node_aligned_wz );
        assert( options.execution.use_4d_pencil_node_aligned_wz_pipeline );
        assert( options.execution.use_4d_native_xw_direct_layout );
        assert( options.execution.use_4d_native_xw_chunked_transport );
        assert( options.execution.native_xw_chunk_window == 1 );
        assert( options.execution.use_4d_native_xw_compact_staging );
        assert( options.execution.slab_native_wz_plan_concurrency == 4 );
        assert( options.execution.use_4d_slab_native_wz_ready_pipeline );
        assert( !options.execution.use_4d_pencil_degenerate_local_transposes );
        assert( !options.diagnostics.use_4d_pencil_p3_degenerate_wz_pipeline );
    }

    {
        const fftm::fftm_4d_production_topology topology( 32, 8, 4, 8, 1 );
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::pencil_pencil,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true,
            topology
        );
        assert( options.execution.use_4d_pencil_node_aligned_wz_pipeline );
    }

    {
        const fftm::fftm_4d_production_topology topology(
            16, 8, 2, 8, 1, fftm::fftm_4d_pencil_pipeline::standard
        );
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::pencil_pencil,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true,
            topology
        );
        assert( !options.execution.use_4d_pencil_node_aligned_wz_pipeline );
        assert( !options.execution.use_4d_native_xw_direct_layout );
        assert( !options.execution.use_4d_pencil_degenerate_local_transposes );
    }

    {
        const fftm::fftm_4d_production_topology topology( 16, 8, 2, 8, 1 );
        const auto options = fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::pencil_pencil,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            false,
            topology
        );
        assert( !options.execution.use_4d_pencil_node_aligned_wz_pipeline );
    }

    bool rejected = false;
    try
    {
        (void)fftm::production_options_3d( fftm::transform_strategy_3d::pencil_pencil, 0 );
    }
    catch ( const std::logic_error & )
    {
        rejected = true;
    }
    assert( rejected );

    rejected = false;
    try
    {
        const fftm::fftm_4d_production_topology topology(
            16, 8, 4, 4, 1, fftm::fftm_4d_pencil_pipeline::node_aligned_wz
        );
        (void)fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::pencil_pencil,
            fftm::fftm_4d_spectral_layout::native_xzwy,
            true,
            topology
        );
    }
    catch ( const std::logic_error & )
    {
        rejected = true;
    }
    assert( rejected );
    return 0;
}
