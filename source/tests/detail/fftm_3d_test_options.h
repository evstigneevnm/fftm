#ifndef __FFTM_TESTS_DETAIL_FFTM_3D_TEST_OPTIONS_H__
#define __FFTM_TESTS_DETAIL_FFTM_3D_TEST_OPTIONS_H__

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include <fftm.hpp>
#include <fftm_autotune.hpp>

namespace fftm
{
namespace test
{
namespace detail
{

enum class fftm_3d_strategy_kind
{
    slab_pencil,
    pencil_slab,
    pencil_pencil
};

enum class fftm_3d_pencil_layout_kind
{
    auto_select,
    opt0,
    opt1,
    legacy
};

enum class fftm_3d_pencil_pipeline_kind
{
    staged,
    fused,
    reference,
    reference_parity
};

enum class fftm_3d_backend_kind
{
    native,
    fftm3d_scfd_fft_facade
};

struct fftm_3d_test_options
{
    fftm_3d_strategy_kind         strategy = fftm_3d_strategy_kind::slab_pencil;
    bool                          run_all  = false;
    ::fftm::mpi_transpose_3d_mode mode     = ::fftm::mpi_transpose_3d_mode::alltoallv;
    std::size_t                   nx       = 16;
    std::size_t                   ny       = 16;
    std::size_t                   nz       = 16;
    std::size_t                   p1       = 0;
    std::size_t                   p2       = 0;
    double                        threshold = 1.0e-11;
    int                           times    = 1;
    int                           warmup   = 0;
    bool                          use_direct_backward_receive = false;
    bool                          direct_p2p_cuda_aware       = true;
    bool                          use_p2p_send_thread         = false;
	    bool                          use_p2p_byte_transfer       = false;
	    bool                          use_persistent_p2p          = false;
	    bool                          use_ready_p2p_send          = false;
    bool                          print_pencil_schedule       = false;
    bool                          use_direct_forward_byte_receive = false;
    bool                          use_stable_forward_byte_send_buffer = false;
    bool                          use_ready_stable_forward_byte_send_buffer = false;
    bool                          use_contiguous_forward_byte_send = false;
    bool                          use_physical_forward_peer_exchange = false;
    bool                          use_autotune_config = false;
    bool                          disable_autotune_config = false;
    std::string                   autotune_config;
    fftm_3d_backend_kind          backend = fftm_3d_backend_kind::native;
    bool                          backend_explicit = false;
    ::fftm::fftm_3d_contiguous_forward_send_mode contiguous_forward_send_mode =
        ::fftm::fftm_3d_contiguous_forward_send_mode::single;
    std::size_t                   contiguous_forward_send_chunk_bytes = static_cast<std::size_t>( 1 ) << 30;
    ::fftm::fftm_3d_large_count_p2p_transport large_count_p2p_transport =
        ::fftm::fftm_3d_large_count_p2p_transport::hindexed;
    bool                          use_large_count_datatype_cache = false;
    bool                          use_fft_exec_no_sync = false;
    bool                          use_native_backward_second_peer_loop = false;
    bool                          use_3d_deferred_send_completion = false;
    bool                          use_native_opt0_default_z_layout = false;
    bool                          use_native_opt0_reference_y_buffer_topology = false;
    bool                          use_native_opt0_compact_y_workarea = false;
    bool                          use_native_opt0_tight_y_plan_sequence = false;
    bool                          use_native_opt0_shared_y_plan_handles = false;
    bool                          use_native_opt0_y_group_device_sync = false;
    bool                          use_native_opt0_y_no_sync_exec = true;
    bool                          use_native_opt0_raw_y_plan_array_executor = false;
    bool                          use_native_opt0_reference_y_plan_lifecycle = false;
    bool                          use_native_opt0_reference_y_plan_bundle = false;
    bool                          use_native_opt0_raw_y_plan_bundle = false;
    bool                          use_native_opt0_y_plan_bundle_stream_first = false;
    bool                          use_native_opt0_raw_y_plan_bundle_reference_streams = false;
    bool                          use_native_opt0_reference_local_plan_context = false;
    bool                          allow_native_opt0_diagnostic_variants = false;
    bool                          use_native_opt0_memory_feasibility_guard = true;
    std::size_t                   native_opt0_memory_feasibility_reserve_bytes =
        static_cast<std::size_t>( 512 ) * static_cast<std::size_t>( 1024 ) * static_cast<std::size_t>( 1024 );
    bool                          enable_local_fft_diagnostics = false;
    bool                          enable_native_stage_timers = false;
    bool                          enable_gpu_telemetry = false;
    std::string                   local_fft_diagnostics_directory;
	    fftm_3d_pencil_layout_kind    pencil_layout               = fftm_3d_pencil_layout_kind::auto_select;
    fftm_3d_pencil_pipeline_kind  pencil_pipeline             = fftm_3d_pencil_pipeline_kind::staged;
};

inline const char *pencil_layout_name( fftm_3d_pencil_layout_kind layout )
{
    switch ( layout )
    {
    case fftm_3d_pencil_layout_kind::auto_select:
        return "auto";
    case fftm_3d_pencil_layout_kind::opt0:
        return "opt0";
    case fftm_3d_pencil_layout_kind::opt1:
        return "opt1";
    case fftm_3d_pencil_layout_kind::legacy:
        return "legacy";
    }
    return "unknown";
}

inline const char *fftm_3d_backend_name( fftm_3d_backend_kind backend )
{
    switch ( backend )
    {
    case fftm_3d_backend_kind::native:
        return "native";
    case fftm_3d_backend_kind::fftm3d_scfd_fft_facade:
        return "fftm3d-scfd-fft-facade";
    }
    return "unknown";
}

inline fftm_3d_backend_kind parse_fftm_3d_backend( const std::string &value )
{
    if ( value == "native" || value == "fftm" || value == "default" || value.empty() )
        return fftm_3d_backend_kind::native;
    if ( value == "fftm3d-scfd-fft-facade" || value == "fftm3d_scfd_fft_facade" ||
         value == "scfd-fft-facade" )
        return fftm_3d_backend_kind::fftm3d_scfd_fft_facade;
    throw std::logic_error( "Unknown FFTM 3D backend '" + value + "'" );
}

inline std::string first_config_csv_value( const std::string &value )
{
    const std::size_t comma = value.find( ',' );
    return ::fftm::autotune::trim( comma == std::string::npos ? value : value.substr( 0, comma ) );
}

inline ::fftm::fftm_3d_pencil_layout to_fftm_pencil_layout( fftm_3d_pencil_layout_kind layout )
{
    switch ( layout )
    {
    case fftm_3d_pencil_layout_kind::auto_select:
        return ::fftm::fftm_3d_pencil_layout::auto_select;
    case fftm_3d_pencil_layout_kind::opt0:
        return ::fftm::fftm_3d_pencil_layout::opt0;
    case fftm_3d_pencil_layout_kind::opt1:
        return ::fftm::fftm_3d_pencil_layout::opt1;
    case fftm_3d_pencil_layout_kind::legacy:
        return ::fftm::fftm_3d_pencil_layout::legacy;
    }
    return ::fftm::fftm_3d_pencil_layout::auto_select;
}

inline fftm_3d_pencil_layout_kind from_fftm_pencil_layout( ::fftm::fftm_3d_pencil_layout layout )
{
    switch ( layout )
    {
    case ::fftm::fftm_3d_pencil_layout::auto_select:
        return fftm_3d_pencil_layout_kind::auto_select;
    case ::fftm::fftm_3d_pencil_layout::opt0:
        return fftm_3d_pencil_layout_kind::opt0;
    case ::fftm::fftm_3d_pencil_layout::opt1:
        return fftm_3d_pencil_layout_kind::opt1;
    case ::fftm::fftm_3d_pencil_layout::legacy:
        return fftm_3d_pencil_layout_kind::legacy;
    }
    return fftm_3d_pencil_layout_kind::auto_select;
}

inline const char *pencil_pipeline_name( fftm_3d_pencil_pipeline_kind pipeline )
{
    switch ( pipeline )
    {
    case fftm_3d_pencil_pipeline_kind::staged:
        return "staged";
    case fftm_3d_pencil_pipeline_kind::fused:
        return "fused";
    case fftm_3d_pencil_pipeline_kind::reference:
        return "reference";
    case fftm_3d_pencil_pipeline_kind::reference_parity:
        return "reference-parity";
    }
    return "unknown";
}

inline ::fftm::fftm_3d_pencil_pipeline to_fftm_pencil_pipeline( fftm_3d_pencil_pipeline_kind pipeline )
{
    switch ( pipeline )
    {
    case fftm_3d_pencil_pipeline_kind::staged:
        return ::fftm::fftm_3d_pencil_pipeline::staged;
    case fftm_3d_pencil_pipeline_kind::fused:
        return ::fftm::fftm_3d_pencil_pipeline::fused;
    case fftm_3d_pencil_pipeline_kind::reference:
        return ::fftm::fftm_3d_pencil_pipeline::reference;
    case fftm_3d_pencil_pipeline_kind::reference_parity:
        return ::fftm::fftm_3d_pencil_pipeline::reference_parity;
    }
    return ::fftm::fftm_3d_pencil_pipeline::staged;
}

inline fftm_3d_pencil_pipeline_kind from_fftm_pencil_pipeline( ::fftm::fftm_3d_pencil_pipeline pipeline )
{
    switch ( pipeline )
    {
    case ::fftm::fftm_3d_pencil_pipeline::staged:
        return fftm_3d_pencil_pipeline_kind::staged;
    case ::fftm::fftm_3d_pencil_pipeline::fused:
        return fftm_3d_pencil_pipeline_kind::fused;
    case ::fftm::fftm_3d_pencil_pipeline::reference:
        return fftm_3d_pencil_pipeline_kind::reference;
    case ::fftm::fftm_3d_pencil_pipeline::reference_parity:
        return fftm_3d_pencil_pipeline_kind::reference_parity;
    }
    return fftm_3d_pencil_pipeline_kind::staged;
}

inline ::fftm::fftm_3d_large_count_p2p_transport parse_large_count_p2p_transport( const std::string &value )
{
    if ( value == "hindexed" )
        return ::fftm::fftm_3d_large_count_p2p_transport::hindexed;
    if ( value == "mpi-count" || value == "mpi_count" )
        return ::fftm::fftm_3d_large_count_p2p_transport::mpi_count;
    if ( value == "element-count" || value == "element_count" || value == "complex-count" ||
         value == "complex_count" )
        return ::fftm::fftm_3d_large_count_p2p_transport::element_count;
    if ( value == "chunked" )
        return ::fftm::fftm_3d_large_count_p2p_transport::chunked;
    throw std::logic_error( "Unknown large-count P2P transport '" + value + "'" );
}

inline ::fftm::fftm_3d_contiguous_forward_send_mode
parse_contiguous_forward_send_mode( const std::string &value )
{
    if ( value == "single" )
        return ::fftm::fftm_3d_contiguous_forward_send_mode::single;
    if ( value == "chunked" )
        return ::fftm::fftm_3d_contiguous_forward_send_mode::chunked;
    throw std::logic_error( "Unknown contiguous forward send mode '" + value + "'" );
}

inline std::pair<std::size_t, std::size_t> choose_pencil_grid_3d( std::size_t num_procs )
{
    std::size_t p1 = 1;
    for ( std::size_t d = 1; d * d <= num_procs; ++d )
    {
        if ( num_procs % d == 0 )
            p1 = d;
    }
    return std::make_pair( p1, num_procs / p1 );
}

inline std::pair<std::size_t, std::size_t>
choose_grid_3d( const fftm_3d_test_options &options, fftm_3d_strategy_kind strategy, int num_procs )
{
    if ( options.p1 != 0 && options.p2 != 0 )
        return std::make_pair( options.p1, options.p2 );

    const auto pencil_grid = choose_pencil_grid_3d( static_cast<std::size_t>( num_procs ) );
    if ( strategy == fftm_3d_strategy_kind::slab_pencil )
        return std::make_pair( static_cast<std::size_t>( num_procs ), 1u );
    if ( strategy == fftm_3d_strategy_kind::pencil_slab )
        return std::make_pair( 1u, static_cast<std::size_t>( num_procs ) );
    return pencil_grid;
}

inline std::string
usage_fftm_3d_test( const std::string &binary_name, bool allow_strategy_all, bool allow_times, bool allow_threshold = false )
{
    std::string usage = "USAGE: " + binary_name + " [--strategy slab-pencil|pencil-slab|pencil-pencil";
    if ( allow_strategy_all )
        usage += "|all";
    usage += "] [--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw] [--grid P1 P2]";
    if ( allow_threshold )
        usage += " [--threshold eps]";
    if ( allow_times )
        usage += " [--times repeats] [--warmup repeats]";
    usage += " [--use-direct-backward-receive|--no-direct-backward-receive]";
    usage += " [--direct-p2p-cuda-aware|--no-direct-p2p-cuda-aware]";
    usage += " [--use-p2p-send-thread|--no-p2p-send-thread]";
    usage += " [--use-p2p-byte-transfer|--no-p2p-byte-transfer]";
	    usage += " [--use-persistent-p2p|--no-persistent-p2p]";
	    usage += " [--use-ready-p2p-send|--no-ready-p2p-send]";
	    usage += " [--print-pencil-schedule|--no-print-pencil-schedule]";
    usage += " [--use-direct-forward-byte-receive|--no-direct-forward-byte-receive]";
    usage += " [--use-stable-forward-byte-send-buffer|--no-stable-forward-byte-send-buffer]";
    usage += " [--use-ready-stable-forward-byte-send-buffer|--no-ready-stable-forward-byte-send-buffer]";
    usage += " [--use-contiguous-forward-byte-send|--no-contiguous-forward-byte-send]";
    usage += " [--use-physical-forward-peer-exchange|--no-physical-forward-peer-exchange]";
    usage += " [--use-autotune-config|--no-autotune-config|--autotune-config path]";
    usage += " [--fftm-3d-backend native|fftm3d-scfd-fft-facade]";
    usage += " [--contiguous-forward-send-mode single|chunked]";
    usage += " [--contiguous-forward-send-chunk-mib MiB]";
    usage += " [--large-count-p2p-transport hindexed|mpi-count|element-count|chunked]";
    usage += " [--use-large-count-datatype-cache|--no-large-count-datatype-cache]";
    usage += " [--use-fft-exec-no-sync|--no-fft-exec-no-sync]";
    usage += " [--use-native-backward-second-peer-loop|--no-native-backward-second-peer-loop]";
    usage += " [--use-3d-deferred-send-completion|--no-3d-deferred-send-completion]";
    usage += " [--use-native-opt0-compact-y-workarea|--no-native-opt0-compact-y-workarea]";
    usage += " [--use-native-opt0-tight-y-plan-sequence|--no-native-opt0-tight-y-plan-sequence]";
    usage += " [--use-native-opt0-shared-y-plan-handles|--no-native-opt0-shared-y-plan-handles]";
    usage += " [--use-native-opt0-y-group-device-sync|--no-native-opt0-y-group-device-sync]";
    usage += " [--use-native-opt0-y-no-sync-exec|--no-native-opt0-y-no-sync-exec]";
    usage += " [--use-native-opt0-raw-y-plan-array-executor|--no-native-opt0-raw-y-plan-array-executor]";
    usage += " [--use-native-opt0-reference-y-plan-lifecycle|--no-native-opt0-reference-y-plan-lifecycle]";
    usage += " [--use-native-opt0-reference-y-plan-bundle|--no-native-opt0-reference-y-plan-bundle]";
    usage += " [--use-native-opt0-raw-y-plan-bundle|--no-native-opt0-raw-y-plan-bundle]";
    usage += " [--use-native-opt0-y-plan-bundle-stream-first|--no-native-opt0-y-plan-bundle-stream-first]";
    usage += " [--use-native-opt0-reference-y-buffer-topology|--no-native-opt0-reference-y-buffer-topology]";
    usage += " [--use-native-opt0-raw-y-plan-bundle-reference-streams|--no-native-opt0-raw-y-plan-bundle-reference-streams]";
    usage += " [--use-native-opt0-reference-local-plan-context|--no-native-opt0-reference-local-plan-context]";
    usage += " [--allow-native-opt0-diagnostic-variants|--no-native-opt0-diagnostic-variants]";
    usage += " [--use-native-opt0-memory-feasibility-guard|--no-native-opt0-memory-feasibility-guard]";
    usage += " [--native-opt0-memory-feasibility-reserve-mib MiB]";
    usage += " [--enable-local-fft-diagnostics|--disable-local-fft-diagnostics]";
    usage += " [--enable-native-stage-timers|--disable-native-stage-timers]";
    usage += " [--local-fft-diagnostics-directory path]";
    usage += " [--pencil-layout auto|opt0|opt1|legacy]";
    usage += " [--pencil-pipeline staged|fused|reference|reference-parity|native-parity|egger-parity]";
    usage += " [Nx Ny Nz]";
    return usage;
}

inline ::fftm::fftm_init_options make_fftm_init_options( const fftm_3d_test_options &options );

inline void copy_fftm_init_options_to_fftm_3d_test_options(
    fftm_3d_test_options &options, const ::fftm::fftm_init_options &init_options
)
{
    options.use_direct_backward_receive = init_options.execution.use_direct_backward_receive;
    options.direct_p2p_cuda_aware       = init_options.execution.direct_p2p_cuda_aware;
    options.use_p2p_send_thread         = init_options.diagnostics.use_p2p_send_thread;
    options.use_p2p_byte_transfer       = init_options.execution.use_p2p_byte_transfer;
    options.use_persistent_p2p          = init_options.execution.use_persistent_p2p;
    options.use_ready_p2p_send          = init_options.diagnostics.use_ready_p2p_send;
    options.print_pencil_schedule       = init_options.diagnostics.print_pencil_schedule;
    options.use_direct_forward_byte_receive = init_options.diagnostics.use_direct_forward_byte_receive;
    options.use_stable_forward_byte_send_buffer = init_options.diagnostics.use_stable_forward_byte_send_buffer;
    options.use_ready_stable_forward_byte_send_buffer =
        init_options.diagnostics.use_ready_stable_forward_byte_send_buffer;
    options.use_contiguous_forward_byte_send = init_options.diagnostics.use_contiguous_forward_byte_send;
    options.use_physical_forward_peer_exchange = init_options.diagnostics.use_physical_forward_peer_exchange;
    options.contiguous_forward_send_mode = init_options.diagnostics.contiguous_forward_send_mode;
    options.contiguous_forward_send_chunk_bytes = init_options.diagnostics.contiguous_forward_send_chunk_bytes;
    options.large_count_p2p_transport = init_options.execution.large_count_p2p_transport;
    options.use_large_count_datatype_cache = init_options.diagnostics.use_large_count_datatype_cache;
    options.use_fft_exec_no_sync = init_options.diagnostics.use_fft_exec_no_sync;
    options.use_native_backward_second_peer_loop = init_options.diagnostics.use_native_backward_second_peer_loop;
    options.use_3d_deferred_send_completion = init_options.diagnostics.use_3d_deferred_send_completion;
    options.use_native_opt0_default_z_layout = init_options.execution.use_native_opt0_default_z_layout;
    options.use_native_opt0_reference_y_buffer_topology = init_options.execution.use_native_opt0_reference_y_buffer_topology;
    options.use_native_opt0_compact_y_workarea = init_options.execution.use_native_opt0_compact_y_workarea;
    options.use_native_opt0_tight_y_plan_sequence = init_options.execution.use_native_opt0_tight_y_plan_sequence;
    options.use_native_opt0_shared_y_plan_handles = init_options.execution.use_native_opt0_shared_y_plan_handles;
    options.use_native_opt0_y_group_device_sync = init_options.execution.use_native_opt0_y_group_device_sync;
    options.use_native_opt0_y_no_sync_exec = init_options.execution.use_native_opt0_y_no_sync_exec;
    options.use_native_opt0_raw_y_plan_array_executor = init_options.execution.use_native_opt0_raw_y_plan_array_executor;
    options.use_native_opt0_reference_y_plan_lifecycle =
        init_options.diagnostics.use_native_opt0_reference_y_plan_lifecycle;
    options.use_native_opt0_reference_y_plan_bundle = init_options.diagnostics.use_native_opt0_reference_y_plan_bundle;
    options.use_native_opt0_raw_y_plan_bundle = init_options.diagnostics.use_native_opt0_raw_y_plan_bundle;
    options.use_native_opt0_y_plan_bundle_stream_first = init_options.diagnostics.use_native_opt0_y_plan_bundle_stream_first;
    options.use_native_opt0_raw_y_plan_bundle_reference_streams =
        init_options.diagnostics.use_native_opt0_raw_y_plan_bundle_reference_streams;
    options.use_native_opt0_reference_local_plan_context = init_options.diagnostics.use_native_opt0_reference_local_plan_context;
    options.allow_native_opt0_diagnostic_variants = init_options.diagnostics.allow_native_opt0_diagnostic_variants;
    options.use_native_opt0_memory_feasibility_guard =
        init_options.execution.use_native_opt0_memory_feasibility_guard;
    options.native_opt0_memory_feasibility_reserve_bytes =
        init_options.execution.native_opt0_memory_feasibility_reserve_bytes;
    options.enable_local_fft_diagnostics = init_options.diagnostics.enable_local_fft_diagnostics;
    options.enable_native_stage_timers = init_options.diagnostics.enable_native_stage_timers;
    options.local_fft_diagnostics_directory = init_options.diagnostics.local_fft_diagnostics_directory;
    options.pencil_layout = from_fftm_pencil_layout( init_options.pencil_layout_3d );
    options.pencil_pipeline = from_fftm_pencil_pipeline( init_options.pencil_pipeline_3d );
}

inline bool apply_fftm_3d_autotune_config(
    fftm_3d_test_options &options, int num_procs, bool allow_grid_update
)
{
    if ( options.disable_autotune_config )
        return false;

    const bool has_env_request = ::fftm::autotune::getenv_or_null( "FFTM_AUTOTUNE" ) != nullptr ||
                                 ::fftm::autotune::getenv_or_null( "FFTM_AUTOTUNE_CONFIG" ) != nullptr;
    if ( !options.use_autotune_config && options.autotune_config.empty() && !has_env_request )
        return false;

    ::fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );

    ::fftm::processor_grid grid;
    if ( options.p1 != 0 && options.p2 != 0 )
        grid.init( options.p1, options.p2 );

    ::fftm::fftm_init_options init_options = make_fftm_init_options( options );
    bool                      applied      = false;
    auto apply_backend_config = [&]( const ::fftm::autotune::config_map &config ) {
        if ( options.backend_explicit )
            return;
        std::string backend = ::fftm::autotune::value_or_empty( config, "FFTM_AUTOTUNE_BACKEND_3D" );
        if ( backend.empty() )
            backend = ::fftm::autotune::value_or_empty( config, "FFTM_3D_BACKEND" );
        if ( backend.empty() )
            backend = ::fftm::autotune::value_or_empty( config, "FFTM_3D_BACKENDS" );
        if ( backend.empty() || ::fftm::autotune::configured_sentinel( backend ) )
            return;
        options.backend = parse_fftm_3d_backend( first_config_csv_value( backend ) );
        applied = true;
    };

    if ( !options.autotune_config.empty() )
    {
        ::fftm::autotune::config_map config = ::fftm::autotune::load_key_value_file( options.autotune_config );
        applied = ::fftm::autotune::apply_config_3d( config, num_procs, sizes, grid, init_options );
        apply_backend_config( config );
    }
    else if ( options.use_autotune_config )
    {
        ::fftm::autotune::config_map config;
        ::fftm::autotune::overlay_environment( config );
        applied = ::fftm::autotune::apply_config_3d( config, num_procs, sizes, grid, init_options );
        apply_backend_config( config );
    }
    else
    {
        ::fftm::autotune::config_map config = ::fftm::autotune::load_config_from_environment();
        applied = ::fftm::autotune::apply_config_3d( config, num_procs, sizes, grid, init_options );
        apply_backend_config( config );
    }

    if ( !applied )
        return false;

    copy_fftm_init_options_to_fftm_3d_test_options( options, init_options );
    if ( allow_grid_update && grid.p1 != 0 && grid.p2 != 0 )
    {
        options.p1 = grid.p1;
        options.p2 = grid.p2;
    }
    return true;
}

inline fftm_3d_test_options parse_fftm_3d_test_options(
    int argc, char *argv[], int num_procs, const std::string &binary_name, bool allow_strategy_all, bool allow_times,
    const fftm_3d_test_options &defaults = fftm_3d_test_options(), bool allow_threshold = false
)
{
    fftm_3d_test_options options = defaults;
    int                  argi    = 1;

    while ( argi < argc )
    {
        const std::string arg = argv[argi];
        if ( arg == "--strategy" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --strategy" );
            const std::string value = argv[argi + 1];
            if ( value == "slab-pencil" )
                options.strategy = fftm_3d_strategy_kind::slab_pencil;
            else if ( value == "pencil-slab" )
                options.strategy = fftm_3d_strategy_kind::pencil_slab;
            else if ( value == "pencil-pencil" )
                options.strategy = fftm_3d_strategy_kind::pencil_pencil;
            else if ( allow_strategy_all && value == "all" )
                options.run_all = true;
            else
                throw std::logic_error( "Unknown strategy '" + value + "'" );
            argi += 2;
        }
        else if ( arg == "--mode" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --mode" );
            const std::string value = argv[argi + 1];
            if ( value == "p2p-waitall" )
                options.mode = ::fftm::mpi_transpose_3d_mode::p2p_waitall;
            else if ( value == "p2p-waitany" )
                options.mode = ::fftm::mpi_transpose_3d_mode::p2p_waitany;
            else if ( value == "alltoallv" )
                options.mode = ::fftm::mpi_transpose_3d_mode::alltoallv;
            else if ( value == "alltoallw" )
                options.mode = ::fftm::mpi_transpose_3d_mode::alltoallw;
            else
                throw std::logic_error( "Unknown mode '" + value + "'" );
            argi += 2;
        }
        else if ( arg == "--grid" )
        {
            if ( argi + 2 >= argc )
                throw std::logic_error( "Missing values for --grid P1 P2" );
            options.p1 = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            options.p2 = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
            argi += 3;
        }
        else if ( arg == "--times" )
        {
            if ( !allow_times )
                throw std::logic_error( "Unknown option '--times'" );
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --times" );
            options.times = std::atoi( argv[argi + 1] );
            if ( options.times < 1 )
                throw std::logic_error( "--times must be at least 1" );
            argi += 2;
        }
        else if ( arg == "--warmup" )
        {
            if ( !allow_times )
                throw std::logic_error( "Unknown option '--warmup'" );
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --warmup" );
            options.warmup = std::atoi( argv[argi + 1] );
            if ( options.warmup < 0 )
                throw std::logic_error( "--warmup must be non-negative" );
            argi += 2;
        }
        else if ( arg == "--threshold" )
        {
            if ( !allow_threshold )
                throw std::logic_error( "Unknown option '--threshold'" );
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --threshold" );
            options.threshold = std::atof( argv[argi + 1] );
            argi += 2;
        }
        else if ( arg == "--use-direct-backward-receive" )
        {
            options.use_direct_backward_receive = true;
            argi += 1;
        }
        else if ( arg == "--no-direct-backward-receive" )
        {
            options.use_direct_backward_receive = false;
            argi += 1;
        }
        else if ( arg == "--direct-p2p-cuda-aware" || arg == "--direct-p2p-CUDA-aware" )
        {
            options.direct_p2p_cuda_aware = true;
            argi += 1;
        }
        else if ( arg == "--no-direct-p2p-cuda-aware" || arg == "--no-direct-p2p-CUDA-aware" )
        {
            options.direct_p2p_cuda_aware = false;
            argi += 1;
        }
        else if ( arg == "--use-p2p-send-thread" )
        {
            options.use_p2p_send_thread = true;
            argi += 1;
        }
        else if ( arg == "--no-p2p-send-thread" )
        {
            options.use_p2p_send_thread = false;
            argi += 1;
        }
        else if ( arg == "--use-p2p-byte-transfer" )
        {
            options.use_p2p_byte_transfer = true;
            argi += 1;
        }
        else if ( arg == "--no-p2p-byte-transfer" )
        {
            options.use_p2p_byte_transfer = false;
            argi += 1;
        }
        else if ( arg == "--use-persistent-p2p" )
        {
            options.use_persistent_p2p = true;
            argi += 1;
        }
        else if ( arg == "--no-persistent-p2p" )
        {
            options.use_persistent_p2p = false;
            argi += 1;
        }
        else if ( arg == "--use-ready-p2p-send" )
        {
            options.use_ready_p2p_send = true;
            argi += 1;
        }
	        else if ( arg == "--no-ready-p2p-send" )
	        {
	            options.use_ready_p2p_send = false;
	            argi += 1;
	        }
	        else if ( arg == "--print-pencil-schedule" )
	        {
	            options.print_pencil_schedule = true;
	            argi += 1;
	        }
	        else if ( arg == "--no-print-pencil-schedule" )
	        {
	            options.print_pencil_schedule = false;
	            argi += 1;
	        }
        else if ( arg == "--use-direct-forward-byte-receive" )
        {
            options.use_direct_forward_byte_receive = true;
            argi += 1;
        }
        else if ( arg == "--no-direct-forward-byte-receive" )
        {
            options.use_direct_forward_byte_receive = false;
            argi += 1;
        }
        else if ( arg == "--use-stable-forward-byte-send-buffer" )
        {
            options.use_stable_forward_byte_send_buffer = true;
            argi += 1;
        }
        else if ( arg == "--no-stable-forward-byte-send-buffer" )
        {
            options.use_stable_forward_byte_send_buffer = false;
            argi += 1;
        }
        else if ( arg == "--use-ready-stable-forward-byte-send-buffer" )
        {
            options.use_ready_stable_forward_byte_send_buffer = true;
            argi += 1;
        }
        else if ( arg == "--no-ready-stable-forward-byte-send-buffer" )
        {
            options.use_ready_stable_forward_byte_send_buffer = false;
            argi += 1;
        }
        else if ( arg == "--use-contiguous-forward-byte-send" )
        {
            options.use_contiguous_forward_byte_send = true;
            argi += 1;
        }
        else if ( arg == "--no-contiguous-forward-byte-send" )
        {
            options.use_contiguous_forward_byte_send = false;
            argi += 1;
        }
        else if ( arg == "--use-physical-forward-peer-exchange" )
        {
            options.use_physical_forward_peer_exchange = true;
            argi += 1;
        }
        else if ( arg == "--no-physical-forward-peer-exchange" )
        {
            options.use_physical_forward_peer_exchange = false;
            argi += 1;
        }
        else if ( arg == "--use-autotune-config" )
        {
            options.use_autotune_config    = true;
            options.disable_autotune_config = false;
            argi += 1;
        }
        else if ( arg == "--no-autotune-config" )
        {
            options.use_autotune_config    = false;
            options.disable_autotune_config = true;
            options.autotune_config.clear();
            argi += 1;
        }
        else if ( arg == "--autotune-config" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --autotune-config" );
            options.autotune_config         = argv[argi + 1];
            options.use_autotune_config    = true;
            options.disable_autotune_config = false;
            argi += 2;
        }
        else if ( arg == "--fftm-3d-backend" || arg == "--backend-3d" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for " + arg );
            options.backend = parse_fftm_3d_backend( argv[argi + 1] );
            options.backend_explicit = true;
            argi += 2;
        }
        else if ( arg == "--contiguous-forward-send-mode" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --contiguous-forward-send-mode" );
            options.contiguous_forward_send_mode = parse_contiguous_forward_send_mode( argv[argi + 1] );
            argi += 2;
        }
        else if ( arg == "--contiguous-forward-send-chunk-mib" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --contiguous-forward-send-chunk-mib" );
            const std::size_t mib = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            if ( mib == 0 )
                throw std::logic_error( "--contiguous-forward-send-chunk-mib must be positive" );
            options.contiguous_forward_send_chunk_bytes = mib * static_cast<std::size_t>( 1024 ) *
                                                          static_cast<std::size_t>( 1024 );
            argi += 2;
        }
        else if ( arg == "--large-count-p2p-transport" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --large-count-p2p-transport" );
            options.large_count_p2p_transport = parse_large_count_p2p_transport( argv[argi + 1] );
            argi += 2;
        }
        else if ( arg == "--use-large-count-datatype-cache" )
        {
            options.use_large_count_datatype_cache = true;
            argi += 1;
        }
        else if ( arg == "--no-large-count-datatype-cache" )
        {
            options.use_large_count_datatype_cache = false;
            argi += 1;
        }
        else if ( arg == "--use-fft-exec-no-sync" )
        {
            options.use_fft_exec_no_sync = true;
            argi += 1;
        }
        else if ( arg == "--no-fft-exec-no-sync" )
        {
            options.use_fft_exec_no_sync = false;
            argi += 1;
        }
        else if ( arg == "--use-native-backward-second-peer-loop" )
        {
            options.use_native_backward_second_peer_loop = true;
            argi += 1;
        }
        else if ( arg == "--no-native-backward-second-peer-loop" )
        {
            options.use_native_backward_second_peer_loop = false;
            argi += 1;
        }
        else if ( arg == "--use-3d-deferred-send-completion" )
        {
            options.use_3d_deferred_send_completion = true;
            argi += 1;
        }
        else if ( arg == "--no-3d-deferred-send-completion" )
        {
            options.use_3d_deferred_send_completion = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-default-z-layout" )
        {
            options.use_native_opt0_default_z_layout = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-default-z-layout" )
        {
            options.use_native_opt0_default_z_layout = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-egger-y-buffer-topology" )
        {
            options.use_native_opt0_reference_y_buffer_topology = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-egger-y-buffer-topology" )
        {
            options.use_native_opt0_reference_y_buffer_topology = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-reference-y-buffer-topology" )
        {
            options.use_native_opt0_reference_y_buffer_topology = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-reference-y-buffer-topology" )
        {
            options.use_native_opt0_reference_y_buffer_topology = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-compact-y-workarea" )
        {
            options.use_native_opt0_compact_y_workarea = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-compact-y-workarea" )
        {
            options.use_native_opt0_compact_y_workarea = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-tight-y-plan-sequence" )
        {
            options.use_native_opt0_tight_y_plan_sequence = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-tight-y-plan-sequence" )
        {
            options.use_native_opt0_tight_y_plan_sequence = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-shared-y-plan-handles" )
        {
            options.use_native_opt0_shared_y_plan_handles = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-shared-y-plan-handles" )
        {
            options.use_native_opt0_shared_y_plan_handles = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-y-group-device-sync" )
        {
            options.use_native_opt0_y_group_device_sync = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-y-group-device-sync" )
        {
            options.use_native_opt0_y_group_device_sync = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-y-no-sync-exec" )
        {
            options.use_native_opt0_y_no_sync_exec = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-y-no-sync-exec" )
        {
            options.use_native_opt0_y_no_sync_exec = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-raw-y-plan-array-executor" )
        {
            options.use_native_opt0_raw_y_plan_array_executor = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-raw-y-plan-array-executor" )
        {
            options.use_native_opt0_raw_y_plan_array_executor = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-reference-y-plan-lifecycle" )
        {
            options.use_native_opt0_reference_y_plan_lifecycle = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-reference-y-plan-lifecycle" )
        {
            options.use_native_opt0_reference_y_plan_lifecycle = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-reference-y-plan-bundle" )
        {
            options.use_native_opt0_reference_y_plan_bundle = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-reference-y-plan-bundle" )
        {
            options.use_native_opt0_reference_y_plan_bundle = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-raw-y-plan-bundle" )
        {
            options.use_native_opt0_raw_y_plan_bundle = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-raw-y-plan-bundle" )
        {
            options.use_native_opt0_raw_y_plan_bundle = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-y-plan-bundle-stream-first" )
        {
            options.use_native_opt0_y_plan_bundle_stream_first = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-y-plan-bundle-stream-first" )
        {
            options.use_native_opt0_y_plan_bundle_stream_first = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-raw-y-plan-bundle-egger-streams" )
        {
            options.use_native_opt0_raw_y_plan_bundle_reference_streams = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-raw-y-plan-bundle-egger-streams" )
        {
            options.use_native_opt0_raw_y_plan_bundle_reference_streams = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-raw-y-plan-bundle-reference-streams" )
        {
            options.use_native_opt0_raw_y_plan_bundle_reference_streams = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-raw-y-plan-bundle-reference-streams" )
        {
            options.use_native_opt0_raw_y_plan_bundle_reference_streams = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-egger-local-plan-context" )
        {
            options.use_native_opt0_reference_local_plan_context = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-egger-local-plan-context" )
        {
            options.use_native_opt0_reference_local_plan_context = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-reference-local-plan-context" )
        {
            options.use_native_opt0_reference_local_plan_context = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-reference-local-plan-context" )
        {
            options.use_native_opt0_reference_local_plan_context = false;
            argi += 1;
        }
        else if ( arg == "--allow-native-opt0-diagnostic-variants" )
        {
            options.allow_native_opt0_diagnostic_variants = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-diagnostic-variants" )
        {
            options.allow_native_opt0_diagnostic_variants = false;
            argi += 1;
        }
        else if ( arg == "--use-native-opt0-memory-feasibility-guard" )
        {
            options.use_native_opt0_memory_feasibility_guard = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-memory-feasibility-guard" )
        {
            options.use_native_opt0_memory_feasibility_guard = false;
            argi += 1;
        }
        else if ( arg == "--native-opt0-memory-feasibility-reserve-mib" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --native-opt0-memory-feasibility-reserve-mib" );
            const std::size_t mib = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            options.native_opt0_memory_feasibility_reserve_bytes =
                mib * static_cast<std::size_t>( 1024 ) * static_cast<std::size_t>( 1024 );
            argi += 2;
        }
        else if ( arg == "--enable-local-fft-diagnostics" )
        {
            options.enable_local_fft_diagnostics = true;
            argi += 1;
        }
        else if ( arg == "--disable-local-fft-diagnostics" )
        {
            options.enable_local_fft_diagnostics = false;
            argi += 1;
        }
        else if ( arg == "--enable-native-stage-timers" )
        {
            options.enable_native_stage_timers = true;
            argi += 1;
        }
        else if ( arg == "--disable-native-stage-timers" )
        {
            options.enable_native_stage_timers = false;
            argi += 1;
        }
        else if ( arg == "--enable-gpu-telemetry" )
        {
            options.enable_gpu_telemetry = true;
            argi += 1;
        }
        else if ( arg == "--disable-gpu-telemetry" )
        {
            options.enable_gpu_telemetry = false;
            argi += 1;
        }
        else if ( arg == "--local-fft-diagnostics-directory" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --local-fft-diagnostics-directory" );
            options.local_fft_diagnostics_directory = argv[argi + 1];
            argi += 2;
        }
	        else if ( arg == "--pencil-layout" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --pencil-layout" );
            const std::string value = argv[argi + 1];
            if ( value == "auto" )
                options.pencil_layout = fftm_3d_pencil_layout_kind::auto_select;
            else if ( value == "opt0" )
                options.pencil_layout = fftm_3d_pencil_layout_kind::opt0;
            else if ( value == "opt1" )
                options.pencil_layout = fftm_3d_pencil_layout_kind::opt1;
            else if ( value == "legacy" )
                options.pencil_layout = fftm_3d_pencil_layout_kind::legacy;
            else
                throw std::logic_error( "Unknown pencil layout '" + value + "'" );
            argi += 2;
        }
        else if ( arg == "--pencil-pipeline" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --pencil-pipeline" );
            const std::string value = argv[argi + 1];
            if ( value == "staged" )
                options.pencil_pipeline = fftm_3d_pencil_pipeline_kind::staged;
            else if ( value == "fused" )
                options.pencil_pipeline = fftm_3d_pencil_pipeline_kind::fused;
            else if ( value == "reference" || value == "native-compatible" || value == "reference-compatible" ||
                      value == "compatible" || value == "egger" )
                options.pencil_pipeline = fftm_3d_pencil_pipeline_kind::reference;
            else if ( value == "reference-parity" || value == "reference_parity" || value == "native-parity" ||
                      value == "native_parity" ||
                      value == "compatible-parity" || value == "compatible_parity" || value == "egger-parity" ||
                      value == "egger_parity" )
                options.pencil_pipeline = fftm_3d_pencil_pipeline_kind::reference_parity;
            else
                throw std::logic_error( "Unknown pencil pipeline '" + value + "'" );
            argi += 2;
        }
        else
        {
            break;
        }
    }

    if ( argc - argi == 3 )
    {
        options.nx = static_cast<std::size_t>( std::strtoull( argv[argi], NULL, 10 ) );
        options.ny = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
        options.nz = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
    }
    else if ( argc != argi )
    {
        throw std::logic_error( usage_fftm_3d_test( binary_name, allow_strategy_all, allow_times, allow_threshold ) );
    }

    const bool grid_was_explicit = options.p1 != 0 && options.p2 != 0;
    apply_fftm_3d_autotune_config( options, num_procs, !grid_was_explicit );

    if ( options.p1 == 0 || options.p2 == 0 )
    {
        const auto grid = choose_grid_3d( options, options.strategy, num_procs );
        options.p1      = grid.first;
        options.p2      = grid.second;
    }

    return options;
}

inline ::fftm::fftm_init_options make_fftm_init_options( const fftm_3d_test_options &options )
{
    ::fftm::fftm_init_options init_options;
    init_options.reporting = ::fftm::profiling_reporting_options();
    init_options.execution.use_direct_backward_receive = options.use_direct_backward_receive;
    init_options.execution.direct_p2p_cuda_aware       = options.direct_p2p_cuda_aware;
    init_options.diagnostics.use_p2p_send_thread         = options.use_p2p_send_thread;
	    init_options.execution.use_p2p_byte_transfer       = options.use_p2p_byte_transfer;
	    init_options.execution.use_persistent_p2p          = options.use_persistent_p2p;
	    init_options.diagnostics.use_ready_p2p_send          = options.use_ready_p2p_send;
    init_options.diagnostics.print_pencil_schedule       = options.print_pencil_schedule;
    init_options.diagnostics.use_direct_forward_byte_receive = options.use_direct_forward_byte_receive;
    init_options.diagnostics.use_stable_forward_byte_send_buffer = options.use_stable_forward_byte_send_buffer;
    init_options.diagnostics.use_ready_stable_forward_byte_send_buffer = options.use_ready_stable_forward_byte_send_buffer;
    init_options.diagnostics.use_contiguous_forward_byte_send = options.use_contiguous_forward_byte_send;
    init_options.diagnostics.use_physical_forward_peer_exchange = options.use_physical_forward_peer_exchange;
    init_options.diagnostics.contiguous_forward_send_mode = options.contiguous_forward_send_mode;
    init_options.diagnostics.contiguous_forward_send_chunk_bytes = options.contiguous_forward_send_chunk_bytes;
    init_options.execution.large_count_p2p_transport  = options.large_count_p2p_transport;
    init_options.diagnostics.use_large_count_datatype_cache = options.use_large_count_datatype_cache;
    init_options.diagnostics.use_fft_exec_no_sync = options.use_fft_exec_no_sync;
    init_options.diagnostics.use_native_backward_second_peer_loop = options.use_native_backward_second_peer_loop;
    init_options.diagnostics.use_3d_deferred_send_completion = options.use_3d_deferred_send_completion;
    init_options.execution.use_native_opt0_default_z_layout = options.use_native_opt0_default_z_layout;
    init_options.execution.use_native_opt0_reference_y_buffer_topology = options.use_native_opt0_reference_y_buffer_topology;
    init_options.execution.use_native_opt0_compact_y_workarea = options.use_native_opt0_compact_y_workarea;
    init_options.execution.use_native_opt0_tight_y_plan_sequence = options.use_native_opt0_tight_y_plan_sequence;
    init_options.execution.use_native_opt0_shared_y_plan_handles = options.use_native_opt0_shared_y_plan_handles;
    init_options.execution.use_native_opt0_y_group_device_sync = options.use_native_opt0_y_group_device_sync;
    init_options.execution.use_native_opt0_y_no_sync_exec = options.use_native_opt0_y_no_sync_exec;
    init_options.execution.use_native_opt0_raw_y_plan_array_executor = options.use_native_opt0_raw_y_plan_array_executor;
    init_options.diagnostics.use_native_opt0_reference_y_plan_lifecycle =
        options.use_native_opt0_reference_y_plan_lifecycle;
    init_options.diagnostics.use_native_opt0_reference_y_plan_bundle = options.use_native_opt0_reference_y_plan_bundle;
    init_options.diagnostics.use_native_opt0_raw_y_plan_bundle = options.use_native_opt0_raw_y_plan_bundle;
    init_options.diagnostics.use_native_opt0_y_plan_bundle_stream_first = options.use_native_opt0_y_plan_bundle_stream_first;
    init_options.diagnostics.use_native_opt0_raw_y_plan_bundle_reference_streams =
        options.use_native_opt0_raw_y_plan_bundle_reference_streams;
    init_options.diagnostics.use_native_opt0_reference_local_plan_context = options.use_native_opt0_reference_local_plan_context;
    init_options.diagnostics.allow_native_opt0_diagnostic_variants = options.allow_native_opt0_diagnostic_variants;
    init_options.execution.use_native_opt0_memory_feasibility_guard =
        options.use_native_opt0_memory_feasibility_guard;
    init_options.execution.native_opt0_memory_feasibility_reserve_bytes =
        options.native_opt0_memory_feasibility_reserve_bytes;
    init_options.diagnostics.enable_local_fft_diagnostics = options.enable_local_fft_diagnostics;
    init_options.diagnostics.enable_native_stage_timers = options.enable_native_stage_timers;
    init_options.diagnostics.local_fft_diagnostics_directory =
        options.local_fft_diagnostics_directory.empty() ? "." : options.local_fft_diagnostics_directory;
    init_options.diagnostics.local_fft_diagnostics_label =
        "fftm-native_g" + std::to_string( static_cast<unsigned long long>( options.p1 * options.p2 ) ) + "_p" +
        std::to_string( static_cast<unsigned long long>( options.p1 ) ) + "x" +
        std::to_string( static_cast<unsigned long long>( options.p2 ) ) + "_" + pencil_layout_name( options.pencil_layout ) +
        "_" + std::to_string( static_cast<unsigned long long>( options.nx ) ) + "x" +
        std::to_string( static_cast<unsigned long long>( options.ny ) ) + "x" +
        std::to_string( static_cast<unsigned long long>( options.nz ) );
	    init_options.pencil_layout_3d            = to_fftm_pencil_layout( options.pencil_layout );
    init_options.pencil_pipeline_3d          = to_fftm_pencil_pipeline( options.pencil_pipeline );
    return init_options;
}

} // namespace detail
} // namespace test
} // namespace fftm

#endif
