#ifndef __FFTM_FFTM_OPTIONS_HPP__
#define __FFTM_FFTM_OPTIONS_HPP__

#include <cstddef>
#include <stdexcept>
#include <string>

namespace fftm
{

enum class mpi_transpose_3d_mode
{
    p2p_waitall,
    p2p_waitany,
    alltoallv,
    alltoallw
};

inline const char *mpi_transpose_3d_mode_name( mpi_transpose_3d_mode mode )
{
    switch ( mode )
    {
    case mpi_transpose_3d_mode::p2p_waitall:
        return "p2p-waitall";
    case mpi_transpose_3d_mode::p2p_waitany:
        return "p2p-waitany";
    case mpi_transpose_3d_mode::alltoallv:
        return "alltoallv";
    case mpi_transpose_3d_mode::alltoallw:
        return "alltoallw";
    }
    return "unknown";
}

enum class transform_strategy_3d
{
    slab_pencil,
    pencil_slab,
    pencil_pencil
};

enum class transform_strategy_4d_mpi
{
    pencil_pencil,
    slab_slab
};

enum class fftm_4d_spectral_layout
{
    public_yzwx,
    native_xzwy
};

inline const char *fftm_4d_spectral_layout_name( fftm_4d_spectral_layout layout )
{
    switch ( layout )
    {
    case fftm_4d_spectral_layout::public_yzwx:
        return "public-yzwx";
    case fftm_4d_spectral_layout::native_xzwy:
        return "native-xzwy";
    }
    return "unknown";
}

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv, bool UseOptimized = true>
struct strategy_3d_slab_pencil
{
};

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv, bool UseOptimized = true>
struct strategy_3d_pencil_slab
{
};

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv, bool UseOptimized = true>
struct strategy_3d_pencil_pencil
{
};

enum class fftm_3d_pencil_layout
{
    auto_select,
    opt0,
    opt1,
    legacy
};

enum class fftm_3d_pencil_pipeline
{
    staged,
    fused,
    reference,
    reference_parity,
    egger = reference,
    egger_parity = reference_parity
};

enum class fftm_3d_large_count_p2p_transport
{
    hindexed,
    mpi_count,
    element_count,
    chunked
};

enum class fftm_3d_contiguous_forward_send_mode
{
    single,
    chunked
};

inline const char *fftm_3d_large_count_p2p_transport_name( fftm_3d_large_count_p2p_transport transport )
{
    switch ( transport )
    {
    case fftm_3d_large_count_p2p_transport::hindexed:
        return "hindexed";
    case fftm_3d_large_count_p2p_transport::mpi_count:
        return "mpi-count";
    case fftm_3d_large_count_p2p_transport::element_count:
        return "element-count";
    case fftm_3d_large_count_p2p_transport::chunked:
        return "chunked";
    }
    return "unknown";
}

inline const char *fftm_3d_contiguous_forward_send_mode_name( fftm_3d_contiguous_forward_send_mode mode )
{
    switch ( mode )
    {
    case fftm_3d_contiguous_forward_send_mode::single:
        return "single";
    case fftm_3d_contiguous_forward_send_mode::chunked:
        return "chunked";
    }
    return "unknown";
}

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv>
struct strategy_4d_pencil_pencil_mpi
{
};

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv>
struct strategy_4d_slab_slab_mpi
{
};

struct fftm_reporting_options
{
    std::string profiling_key;
    std::string memory_profiling_key;
    bool        verbose                          = false;
    bool        print_profile_summary_on_destroy = false;
    bool        print_profile_totals_on_destroy  = false;
    bool        print_memory_profile_on_destroy  = false;
    bool        print_memory_totals_on_destroy   = false;
};

inline fftm_reporting_options profiling_reporting_options()
{
    fftm_reporting_options options;
    options.profiling_key                    = "fftm_prof";
    options.memory_profiling_key             = "fftm_mem";
    options.print_profile_summary_on_destroy = true;
    options.print_profile_totals_on_destroy  = true;
    options.print_memory_profile_on_destroy  = true;
    options.print_memory_totals_on_destroy   = true;
    return options;
}

namespace detail
{

// Internal execution policy populated by production presets or the autotuner.
// It remains visible for benchmark construction, but is not the normal user API.
struct fftm_execution_options
{
    bool        use_direct_backward_receive = false;
    // Legacy benchmark/cache spelling. This capability means device-aware MPI
    // for either the CUDA or HIP backend.
    bool        direct_p2p_cuda_aware       = true;
    bool        use_p2p_byte_transfer       = false;
    bool        use_persistent_p2p          = false;
    fftm_3d_large_count_p2p_transport large_count_p2p_transport =
        fftm_3d_large_count_p2p_transport::hindexed;

    bool        use_native_opt0_default_z_layout               = false;
    bool        use_native_opt0_reference_y_buffer_topology     = false;
    bool        use_native_opt0_compact_y_workarea              = false;
    bool        use_native_opt0_tight_y_plan_sequence           = false;
    bool        use_native_opt0_shared_y_plan_handles           = false;
    bool        use_native_opt0_y_group_device_sync             = false;
    bool        use_native_opt0_y_no_sync_exec                  = true;
    bool        use_native_opt0_raw_y_plan_array_executor       = false;
    bool        use_native_opt0_memory_feasibility_guard        = true;
    std::size_t native_opt0_memory_feasibility_reserve_bytes =
        static_cast<std::size_t>( 512 ) * static_cast<std::size_t>( 1024 ) * static_cast<std::size_t>( 1024 );

    bool        use_4d_slab_native_xw_transpose = true;
    bool        use_4d_native_xw_direct_layout = false;
    bool        use_4d_native_xw_chunked_transport = false;
    std::size_t native_xw_chunk_bytes =
        static_cast<std::size_t>( 512 ) * static_cast<std::size_t>( 1024 ) * static_cast<std::size_t>( 1024 );
    std::size_t native_xw_chunk_window = 0;
    bool        use_4d_native_xw_compact_staging = false;
    bool        use_4d_slab_native_work_area_alias = false;
    bool        use_4d_slab_native_wz_communication_layout = false;
    std::size_t slab_native_wz_plan_concurrency = 1;
    bool        use_4d_slab_native_wz_ready_pipeline = false;
    bool        use_4d_pencil_same_zw_native_layout = true;
    bool        use_4d_pencil_degenerate_local_transposes = false;
    bool        use_4d_pencil_degenerate_same_xw_native = false;
    bool        use_4d_pencil_degenerate_wz_sliced_z_fft = true;
};

// Unsafe, failed, or measurement-only variants live here so they cannot be
// mistaken for ordinary production controls.
struct fftm_diagnostic_options
{
    bool        use_p2p_send_thread = false;
    bool        use_ready_p2p_send = false;
    bool        print_pencil_schedule = false;
    bool        use_direct_forward_byte_receive = false;
    bool        use_stable_forward_byte_send_buffer = false;
    bool        use_ready_stable_forward_byte_send_buffer = false;
    bool        use_contiguous_forward_byte_send = false;
    bool        use_physical_forward_peer_exchange = false;
    fftm_3d_contiguous_forward_send_mode contiguous_forward_send_mode =
        fftm_3d_contiguous_forward_send_mode::single;
    std::size_t contiguous_forward_send_chunk_bytes = static_cast<std::size_t>( 1 ) << 30;
    bool        use_large_count_datatype_cache = false;
    bool        use_fft_exec_no_sync = false;
    bool        use_native_backward_second_peer_loop = false;
    bool        use_3d_deferred_send_completion = false;

    bool        use_native_opt0_reference_y_plan_lifecycle = false;
    bool        use_native_opt0_reference_y_plan_bundle = false;
    bool        use_native_opt0_raw_y_plan_bundle = false;
    bool        use_native_opt0_y_plan_bundle_stream_first = false;
    bool        use_native_opt0_raw_y_plan_bundle_reference_streams = false;
    bool        use_native_opt0_reference_local_plan_context = false;
    bool        allow_native_opt0_diagnostic_variants = false;

    bool        enable_local_fft_diagnostics = false;
    std::string local_fft_diagnostics_directory;
    std::string local_fft_diagnostics_label;
    bool        enable_native_stage_timers = false;

    bool        use_4d_slab_native_xw_batched_peer_kernels = false;
    bool        use_4d_slab_native_xw_tensor_coalesced_kernels = false;
    bool        use_4d_slab_native_xw_vector4_kernels = false;
    bool        use_4d_slab_native_xw_tiled_kernels = false;
    bool        use_4d_slab_native_xw_layout_stage = false;
    bool        use_4d_pencil_same_zw_peer_paired = false;
    bool        use_4d_pencil_degenerate_xw_slab_path = false;

    // Deprecated benchmark compatibility alias. Prefer fftm_init_options::spectral_layout_4d.
    bool        use_4d_slab_native_xw_native_spectral_layout = false;
};

} // namespace detail

struct fftm_init_options
{
    bool                    use_optimized      = true;
    fftm_3d_pencil_layout   pencil_layout_3d  = fftm_3d_pencil_layout::auto_select;
    fftm_3d_pencil_pipeline pencil_pipeline_3d = fftm_3d_pencil_pipeline::staged;
    fftm_4d_spectral_layout spectral_layout_4d = fftm_4d_spectral_layout::public_yzwx;

    fftm_reporting_options          reporting;
    detail::fftm_execution_options  execution;
    detail::fftm_diagnostic_options diagnostics;
};

inline fftm_init_options production_options_3d(
    transform_strategy_3d strategy, int num_procs, bool device_aware_mpi = true
)
{
    if ( num_procs <= 0 )
        throw std::logic_error( "fftm::production_options_3d requires a positive process count" );

    fftm_init_options options;
    options.execution.direct_p2p_cuda_aware = device_aware_mpi;
    if ( strategy != transform_strategy_3d::pencil_pencil || num_procs == 1 )
        return options;

    options.pencil_pipeline_3d =
        device_aware_mpi ? fftm_3d_pencil_pipeline::reference_parity : fftm_3d_pencil_pipeline::reference;
    options.execution.use_p2p_byte_transfer = device_aware_mpi;
    options.execution.large_count_p2p_transport = fftm_3d_large_count_p2p_transport::hindexed;

    const bool use_opt0 = device_aware_mpi && ( num_procs == 7 || num_procs == 8 );
    options.pencil_layout_3d = use_opt0 ? fftm_3d_pencil_layout::opt0 : fftm_3d_pencil_layout::opt1;
    if ( use_opt0 )
    {
        options.execution.use_native_opt0_default_z_layout = true;
        options.execution.use_native_opt0_reference_y_buffer_topology = true;
        options.execution.use_native_opt0_tight_y_plan_sequence = true;
        options.execution.use_native_opt0_shared_y_plan_handles = true;
        options.execution.use_native_opt0_y_group_device_sync = true;
        options.execution.use_native_opt0_y_no_sync_exec = true;
        options.execution.use_native_opt0_raw_y_plan_array_executor = true;
    }
    return options;
}

inline fftm_init_options production_options_4d(
    transform_strategy_4d_mpi strategy,
    fftm_4d_spectral_layout spectral_layout = fftm_4d_spectral_layout::public_yzwx,
    bool device_aware_mpi = true
)
{
    fftm_init_options options;
    options.spectral_layout_4d = spectral_layout;
    options.execution.direct_p2p_cuda_aware = device_aware_mpi;

    if ( strategy == transform_strategy_4d_mpi::slab_slab )
    {
        options.execution.use_4d_slab_native_xw_transpose = true;
        if ( spectral_layout == fftm_4d_spectral_layout::native_xzwy && device_aware_mpi )
        {
            options.execution.use_4d_native_xw_direct_layout = true;
            options.execution.use_4d_native_xw_chunked_transport = true;
            options.execution.native_xw_chunk_window = 1;
            options.execution.use_4d_native_xw_compact_staging = true;
            options.execution.use_4d_slab_native_work_area_alias = true;
            options.execution.use_4d_slab_native_wz_communication_layout = true;
            options.execution.slab_native_wz_plan_concurrency = 4;
            options.execution.use_4d_slab_native_wz_ready_pipeline = true;
        }
    }
    else if ( spectral_layout == fftm_4d_spectral_layout::native_xzwy )
    {
        options.execution.use_4d_pencil_same_zw_native_layout = true;
        options.execution.use_4d_pencil_degenerate_local_transposes = true;
        options.execution.use_4d_pencil_degenerate_same_xw_native = true;
        options.execution.use_4d_pencil_degenerate_wz_sliced_z_fft = true;
    }
    return options;
}

} // namespace fftm

#endif
