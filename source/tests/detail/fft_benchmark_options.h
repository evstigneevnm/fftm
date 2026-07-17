#ifndef __FFTM_TESTS_DETAIL_FFT_BENCHMARK_OPTIONS_H__
#define __FFTM_TESTS_DETAIL_FFT_BENCHMARK_OPTIONS_H__

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>

#include "fft_benchmark_common.h"
#include "fftm_3d_test_options.h"
#include "fftm_4d_test_options.h"

namespace fftm
{
namespace test
{
namespace detail
{

enum class ffts_4d_strategy_kind
{
    pencil_direct,
    pencil_memcpy,
    slab_direct,
    slab_memcpy
};

template <class T>
struct ffts_3d_benchmark_options
{
    std::size_t nx        = 128;
    std::size_t ny        = 128;
    std::size_t nz        = 128;
    int         times     = 1;
    int         warmup    = 0;
    T           epsilon   = default_benchmark_epsilon<T>();
    std::string directory = "./resutls";
};

template <class T>
struct ffts_4d_benchmark_options
{
    ffts_4d_strategy_kind strategy  = ffts_4d_strategy_kind::pencil_direct;
    bool                  run_all   = false;
    std::size_t           nx        = 16;
    std::size_t           ny        = 16;
    std::size_t           nz        = 16;
    std::size_t           nw        = 16;
    int                   times     = 1;
    int                   warmup    = 0;
    T                     epsilon   = default_benchmark_epsilon<T>();
    std::string           directory = "./resutls";
};

template <class T>
struct fftm_3d_benchmark_options
{
    fftm_3d_strategy_kind         strategy  = fftm_3d_strategy_kind::slab_pencil;
    bool                          run_all   = false;
    ::fftm::mpi_transpose_3d_mode mode      = ::fftm::mpi_transpose_3d_mode::alltoallv;
    std::size_t                   nx        = 128;
    std::size_t                   ny        = 128;
    std::size_t                   nz        = 128;
    std::size_t                   p1        = 0;
    std::size_t                   p2        = 0;
    int                           times     = 1;
    int                           warmup    = 0;
    T                             epsilon   = default_benchmark_epsilon<T>();
    std::string                   directory = "./resutls";
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
    bool                          write_native_pencil_schedule = false;
    bool                          check_native_pencil_schedule = false;
    bool                          native_pencil_schedule_check_only = false;
    bool                          skip_native_pencil_rank_device_check = false;
    std::string                   native_pencil_reference_dir;
    fftm_3d_backend_kind          backend = fftm_3d_backend_kind::native;
    bool                          backend_explicit = false;
    bool                          enable_fftm3d_backend_stage_timers = false;
    ::fftm::fftm_3d_contiguous_forward_send_mode contiguous_forward_send_mode =
        ::fftm::fftm_3d_contiguous_forward_send_mode::single;
    std::size_t                   contiguous_forward_send_chunk_bytes = static_cast<std::size_t>( 1 ) << 30;
    int                           contiguous_forward_send_registration_warmups = 0;
    ::fftm::fftm_3d_large_count_p2p_transport large_count_p2p_transport =
        ::fftm::fftm_3d_large_count_p2p_transport::hindexed;
    bool                          use_large_count_datatype_cache = false;
    bool                          use_fft_exec_no_sync = false;
    bool                          use_native_backward_second_peer_loop = false;
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
    bool                          native_opt0_y_microbench = false;
    bool                          native_opt0_y_cross_microbench = false;
    std::string                   native_opt0_y_cross_factory_mode = "single";
    int                           native_opt0_y_microbench_iterations = 20;
    int                           native_opt0_y_microbench_warmup = 3;
    bool                          enable_local_fft_diagnostics = false;
    bool                          enable_native_stage_timers = false;
    bool                          enable_gpu_telemetry = false;
    std::string                   local_fft_diagnostics_directory;
	    fftm_3d_pencil_layout_kind    pencil_layout               = fftm_3d_pencil_layout_kind::auto_select;
    fftm_3d_pencil_pipeline_kind  pencil_pipeline             = fftm_3d_pencil_pipeline_kind::staged;
};

template <class T>
struct fftm_4d_benchmark_options
{
    fftm_4d_strategy_kind         strategy  = fftm_4d_strategy_kind::pencil_pencil;
    bool                          run_all   = false;
    ::fftm::mpi_transpose_3d_mode mode      = ::fftm::mpi_transpose_3d_mode::alltoallv;
    std::size_t                   nx        = 12;
    std::size_t                   ny        = 10;
    std::size_t                   nz        = 8;
    std::size_t                   nw        = 10;
    std::size_t                   p1        = 0;
    std::size_t                   p2        = 0;
    std::size_t                   p3        = 0;
    int                           times     = 1;
    int                           warmup    = 0;
    T                             epsilon   = default_benchmark_epsilon<T>();
    std::string                   directory = "./resutls";
    bool                          use_direct_backward_receive = false;
    bool                          direct_p2p_cuda_aware       = true;
    bool                          use_p2p_byte_transfer       = false;
    bool                          use_fft_exec_no_sync        = false;
    bool                          enable_native_stage_timers  = false;
    bool                          use_4d_slab_native_xw_transpose = true;
    bool                          use_4d_slab_native_xw_batched_peer_kernels = false;
    bool                          use_4d_slab_native_xw_tensor_coalesced_kernels = false;
    bool                          use_4d_slab_native_xw_vector4_kernels = false;
    bool                          use_4d_slab_native_xw_tiled_kernels = false;
    bool                          use_4d_slab_native_xw_layout_stage = false;
    bool                          use_4d_slab_native_xw_native_spectral_layout = false;
};

inline std::string usage_ffts_3d_benchmark( const std::string &binary_name )
{
    return "USAGE: " + binary_name + " [--times repeats] [--warmup repeats] [--epsilon eps] [--directory path] [Nx Ny Nz]";
}

inline std::string usage_ffts_4d_benchmark( const std::string &binary_name )
{
    return "USAGE: " + binary_name +
           " [--strategy pencil-direct|pencil-memcpy|slab-direct|slab-memcpy|all]"
           " [--times repeats] [--warmup repeats] [--epsilon eps] [--directory path] [Nx Ny Nz Nw]";
}

inline std::string usage_fftm_3d_benchmark( const std::string &binary_name )
{
    return "USAGE: " + binary_name +
           " [--strategy slab-pencil|pencil-slab|pencil-pencil|all]"
           " [--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw]"
           " [--grid P1 P2] [--times repeats] [--warmup repeats] [--epsilon eps] [--directory path]"
           " [--use-direct-backward-receive|--no-direct-backward-receive]"
           " [--direct-p2p-cuda-aware|--no-direct-p2p-cuda-aware]"
           " [--use-p2p-send-thread|--no-p2p-send-thread]"
	           " [--use-p2p-byte-transfer|--no-p2p-byte-transfer]"
	           " [--use-persistent-p2p|--no-persistent-p2p]"
	           " [--use-ready-p2p-send|--no-ready-p2p-send]"
	           " [--print-pencil-schedule|--no-print-pencil-schedule]"
           " [--use-direct-forward-byte-receive|--no-direct-forward-byte-receive]"
           " [--use-stable-forward-byte-send-buffer|--no-stable-forward-byte-send-buffer]"
           " [--use-ready-stable-forward-byte-send-buffer|--no-ready-stable-forward-byte-send-buffer]"
           " [--use-contiguous-forward-byte-send|--no-contiguous-forward-byte-send]"
           " [--use-physical-forward-peer-exchange|--no-physical-forward-peer-exchange]"
           " [--use-autotune-config|--no-autotune-config|--autotune-config path]"
           " [--write-native-pencil-schedule]"
           " [--check-native-pencil-reference-dir path]"
           " [--native-pencil-schedule-check-only]"
           " [--skip-native-pencil-rank-device-check]"
           " [--fftm-3d-backend native|fftm3d-scfd-fft-facade]"
           " [--enable-fftm3d-backend-stage-timers|--disable-fftm3d-backend-stage-timers]"
           " [--contiguous-forward-send-mode single|chunked]"
           " [--contiguous-forward-send-chunk-mib MiB]"
           " [--contiguous-forward-send-registration-warmups repeats]"
           " [--large-count-p2p-transport hindexed|mpi-count|element-count|chunked]"
           " [--use-large-count-datatype-cache|--no-large-count-datatype-cache]"
           " [--use-fft-exec-no-sync|--no-fft-exec-no-sync]"
           " [--use-native-backward-second-peer-loop|--no-native-backward-second-peer-loop]"
           " [--use-native-opt0-compact-y-workarea|--no-native-opt0-compact-y-workarea]"
           " [--use-native-opt0-tight-y-plan-sequence|--no-native-opt0-tight-y-plan-sequence]"
           " [--use-native-opt0-shared-y-plan-handles|--no-native-opt0-shared-y-plan-handles]"
           " [--use-native-opt0-y-group-device-sync|--no-native-opt0-y-group-device-sync]"
           " [--use-native-opt0-y-no-sync-exec|--no-native-opt0-y-no-sync-exec]"
           " [--use-native-opt0-raw-y-plan-array-executor|--no-native-opt0-raw-y-plan-array-executor]"
           " [--use-native-opt0-reference-y-plan-lifecycle|--no-native-opt0-reference-y-plan-lifecycle]"
           " [--use-native-opt0-reference-y-plan-bundle|--no-native-opt0-reference-y-plan-bundle]"
           " [--use-native-opt0-raw-y-plan-bundle|--no-native-opt0-raw-y-plan-bundle]"
           " [--use-native-opt0-y-plan-bundle-stream-first|--no-native-opt0-y-plan-bundle-stream-first]"
	           " [--use-native-opt0-reference-y-buffer-topology|--no-native-opt0-reference-y-buffer-topology]"
	           " [--use-native-opt0-raw-y-plan-bundle-reference-streams|--no-native-opt0-raw-y-plan-bundle-reference-streams]"
	           " [--use-native-opt0-reference-local-plan-context|--no-native-opt0-reference-local-plan-context]"
           " [--allow-native-opt0-diagnostic-variants|--no-native-opt0-diagnostic-variants]"
           " [--use-native-opt0-memory-feasibility-guard|--no-native-opt0-memory-feasibility-guard]"
           " [--native-opt0-memory-feasibility-reserve-mib MiB]"
           " [--native-opt0-y-microbench]"
           " [--native-opt0-y-cross-factory-mode single|ref-only|native-current-only|native-current-nosync-only|native-exact-only|native-owned-only|native-direct-only|native-direct-nosync-only|native-minimal-only|native-minimal-nosync-only|ref-then-native-current|native-current-then-ref|ref-then-native-current-nosync|native-current-nosync-then-ref|ref-then-native-exact|native-exact-then-ref|ref-then-native-owned|native-owned-then-ref|ref-then-native-direct|native-direct-then-ref|ref-then-native-direct-nosync|native-direct-nosync-then-ref|ref-then-native-minimal|native-minimal-then-ref|ref-then-native-minimal-nosync|native-minimal-nosync-then-ref]"
           " [--native-opt0-y-microbench-iterations repeats]"
           " [--native-opt0-y-microbench-warmup repeats]"
           " [--enable-local-fft-diagnostics|--disable-local-fft-diagnostics]"
           " [--enable-native-stage-timers|--disable-native-stage-timers]"
           " [--enable-gpu-telemetry|--disable-gpu-telemetry]"
           " [--local-fft-diagnostics-directory path]"
	           " [--pencil-layout auto|opt0|opt1|legacy]"
	           " [--pencil-pipeline staged|fused|reference|reference-parity|native-parity|egger-parity] [Nx Ny Nz]";
}

template <class T>
inline fftm_3d_test_options make_fftm_3d_test_options_base( const fftm_3d_benchmark_options<T> &options )
{
    fftm_3d_test_options base_options;
    base_options.strategy = options.strategy;
    base_options.run_all  = options.run_all;
    base_options.mode     = options.mode;
    base_options.nx       = options.nx;
    base_options.ny       = options.ny;
    base_options.nz       = options.nz;
    base_options.p1       = options.p1;
    base_options.p2       = options.p2;
    base_options.times    = options.times;
    base_options.warmup   = options.warmup;
    base_options.use_direct_backward_receive = options.use_direct_backward_receive;
    base_options.direct_p2p_cuda_aware       = options.direct_p2p_cuda_aware;
    base_options.use_p2p_send_thread         = options.use_p2p_send_thread;
    base_options.use_p2p_byte_transfer       = options.use_p2p_byte_transfer;
    base_options.use_persistent_p2p          = options.use_persistent_p2p;
    base_options.use_ready_p2p_send          = options.use_ready_p2p_send;
    base_options.print_pencil_schedule       = options.print_pencil_schedule;
    base_options.use_direct_forward_byte_receive = options.use_direct_forward_byte_receive;
    base_options.use_stable_forward_byte_send_buffer = options.use_stable_forward_byte_send_buffer;
    base_options.use_ready_stable_forward_byte_send_buffer =
        options.use_ready_stable_forward_byte_send_buffer;
    base_options.use_contiguous_forward_byte_send = options.use_contiguous_forward_byte_send;
    base_options.use_physical_forward_peer_exchange = options.use_physical_forward_peer_exchange;
    base_options.use_autotune_config = options.use_autotune_config;
    base_options.disable_autotune_config = options.disable_autotune_config;
    base_options.autotune_config = options.autotune_config;
    base_options.backend = options.backend;
    base_options.backend_explicit = options.backend_explicit;
    base_options.contiguous_forward_send_mode = options.contiguous_forward_send_mode;
    base_options.contiguous_forward_send_chunk_bytes = options.contiguous_forward_send_chunk_bytes;
    base_options.large_count_p2p_transport = options.large_count_p2p_transport;
    base_options.use_large_count_datatype_cache = options.use_large_count_datatype_cache;
    base_options.use_fft_exec_no_sync = options.use_fft_exec_no_sync;
    base_options.use_native_backward_second_peer_loop = options.use_native_backward_second_peer_loop;
    base_options.use_native_opt0_default_z_layout = options.use_native_opt0_default_z_layout;
    base_options.use_native_opt0_reference_y_buffer_topology = options.use_native_opt0_reference_y_buffer_topology;
    base_options.use_native_opt0_compact_y_workarea = options.use_native_opt0_compact_y_workarea;
    base_options.use_native_opt0_tight_y_plan_sequence = options.use_native_opt0_tight_y_plan_sequence;
    base_options.use_native_opt0_shared_y_plan_handles = options.use_native_opt0_shared_y_plan_handles;
    base_options.use_native_opt0_y_group_device_sync = options.use_native_opt0_y_group_device_sync;
    base_options.use_native_opt0_y_no_sync_exec = options.use_native_opt0_y_no_sync_exec;
    base_options.use_native_opt0_raw_y_plan_array_executor = options.use_native_opt0_raw_y_plan_array_executor;
    base_options.use_native_opt0_reference_y_plan_lifecycle = options.use_native_opt0_reference_y_plan_lifecycle;
    base_options.use_native_opt0_reference_y_plan_bundle = options.use_native_opt0_reference_y_plan_bundle;
    base_options.use_native_opt0_raw_y_plan_bundle = options.use_native_opt0_raw_y_plan_bundle;
    base_options.use_native_opt0_y_plan_bundle_stream_first = options.use_native_opt0_y_plan_bundle_stream_first;
    base_options.use_native_opt0_raw_y_plan_bundle_reference_streams =
        options.use_native_opt0_raw_y_plan_bundle_reference_streams;
    base_options.use_native_opt0_reference_local_plan_context = options.use_native_opt0_reference_local_plan_context;
    base_options.allow_native_opt0_diagnostic_variants = options.allow_native_opt0_diagnostic_variants;
    base_options.use_native_opt0_memory_feasibility_guard =
        options.use_native_opt0_memory_feasibility_guard;
    base_options.native_opt0_memory_feasibility_reserve_bytes =
        options.native_opt0_memory_feasibility_reserve_bytes;
    base_options.enable_local_fft_diagnostics = options.enable_local_fft_diagnostics;
    base_options.enable_native_stage_timers = options.enable_native_stage_timers;
    base_options.enable_gpu_telemetry = options.enable_gpu_telemetry;
    base_options.local_fft_diagnostics_directory = options.local_fft_diagnostics_directory;
    base_options.pencil_layout = options.pencil_layout;
    base_options.pencil_pipeline = options.pencil_pipeline;
    return base_options;
}

template <class T>
inline void copy_fftm_3d_test_options_base(
    fftm_3d_benchmark_options<T> &options, const fftm_3d_test_options &base_options
)
{
    options.strategy = base_options.strategy;
    options.run_all  = base_options.run_all;
    options.mode     = base_options.mode;
    options.nx       = base_options.nx;
    options.ny       = base_options.ny;
    options.nz       = base_options.nz;
    options.p1       = base_options.p1;
    options.p2       = base_options.p2;
    options.times    = base_options.times;
    options.warmup   = base_options.warmup;
    options.use_direct_backward_receive = base_options.use_direct_backward_receive;
    options.direct_p2p_cuda_aware       = base_options.direct_p2p_cuda_aware;
    options.use_p2p_send_thread         = base_options.use_p2p_send_thread;
    options.use_p2p_byte_transfer       = base_options.use_p2p_byte_transfer;
    options.use_persistent_p2p          = base_options.use_persistent_p2p;
    options.use_ready_p2p_send          = base_options.use_ready_p2p_send;
    options.print_pencil_schedule       = base_options.print_pencil_schedule;
    options.use_direct_forward_byte_receive = base_options.use_direct_forward_byte_receive;
    options.use_stable_forward_byte_send_buffer = base_options.use_stable_forward_byte_send_buffer;
    options.use_ready_stable_forward_byte_send_buffer =
        base_options.use_ready_stable_forward_byte_send_buffer;
    options.use_contiguous_forward_byte_send = base_options.use_contiguous_forward_byte_send;
    options.use_physical_forward_peer_exchange = base_options.use_physical_forward_peer_exchange;
    options.use_autotune_config = base_options.use_autotune_config;
    options.disable_autotune_config = base_options.disable_autotune_config;
    options.autotune_config = base_options.autotune_config;
    options.backend = base_options.backend;
    options.backend_explicit = base_options.backend_explicit;
    options.contiguous_forward_send_mode = base_options.contiguous_forward_send_mode;
    options.contiguous_forward_send_chunk_bytes = base_options.contiguous_forward_send_chunk_bytes;
    options.large_count_p2p_transport = base_options.large_count_p2p_transport;
    options.use_large_count_datatype_cache = base_options.use_large_count_datatype_cache;
    options.use_fft_exec_no_sync = base_options.use_fft_exec_no_sync;
    options.use_native_backward_second_peer_loop = base_options.use_native_backward_second_peer_loop;
    options.use_native_opt0_default_z_layout = base_options.use_native_opt0_default_z_layout;
    options.use_native_opt0_reference_y_buffer_topology = base_options.use_native_opt0_reference_y_buffer_topology;
    options.use_native_opt0_compact_y_workarea = base_options.use_native_opt0_compact_y_workarea;
    options.use_native_opt0_tight_y_plan_sequence = base_options.use_native_opt0_tight_y_plan_sequence;
    options.use_native_opt0_shared_y_plan_handles = base_options.use_native_opt0_shared_y_plan_handles;
    options.use_native_opt0_y_group_device_sync = base_options.use_native_opt0_y_group_device_sync;
    options.use_native_opt0_y_no_sync_exec = base_options.use_native_opt0_y_no_sync_exec;
    options.use_native_opt0_raw_y_plan_array_executor = base_options.use_native_opt0_raw_y_plan_array_executor;
    options.use_native_opt0_reference_y_plan_lifecycle = base_options.use_native_opt0_reference_y_plan_lifecycle;
    options.use_native_opt0_reference_y_plan_bundle = base_options.use_native_opt0_reference_y_plan_bundle;
    options.use_native_opt0_raw_y_plan_bundle = base_options.use_native_opt0_raw_y_plan_bundle;
    options.use_native_opt0_y_plan_bundle_stream_first = base_options.use_native_opt0_y_plan_bundle_stream_first;
    options.use_native_opt0_raw_y_plan_bundle_reference_streams =
        base_options.use_native_opt0_raw_y_plan_bundle_reference_streams;
    options.use_native_opt0_reference_local_plan_context = base_options.use_native_opt0_reference_local_plan_context;
    options.allow_native_opt0_diagnostic_variants = base_options.allow_native_opt0_diagnostic_variants;
    options.use_native_opt0_memory_feasibility_guard =
        base_options.use_native_opt0_memory_feasibility_guard;
    options.native_opt0_memory_feasibility_reserve_bytes =
        base_options.native_opt0_memory_feasibility_reserve_bytes;
    options.enable_local_fft_diagnostics = base_options.enable_local_fft_diagnostics;
    options.enable_native_stage_timers = base_options.enable_native_stage_timers;
    options.enable_gpu_telemetry = base_options.enable_gpu_telemetry;
    options.local_fft_diagnostics_directory = base_options.local_fft_diagnostics_directory;
    options.pencil_layout = base_options.pencil_layout;
    options.pencil_pipeline = base_options.pencil_pipeline;
}

inline std::string usage_fftm_4d_benchmark( const std::string &binary_name )
{
    return "USAGE: " + binary_name +
           " [--strategy pencil-pencil|slab-slab|all]"
           " [--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw]"
           " [--grid P1 P2 P3] [--times repeats] [--warmup repeats] [--epsilon eps] [--directory path]"
           " [--use-direct-backward-receive|--no-direct-backward-receive]"
           " [--direct-p2p-cuda-aware|--no-direct-p2p-cuda-aware]"
           " [--use-p2p-byte-transfer|--no-p2p-byte-transfer]"
           " [--use-fft-exec-no-sync|--no-fft-exec-no-sync]"
           " [--enable-native-stage-timers|--disable-native-stage-timers]"
           " [--use-4d-slab-native-xw-transpose|--no-4d-slab-native-xw-transpose]"
           " [--use-4d-slab-native-xw-batched-peer-kernels|--no-4d-slab-native-xw-batched-peer-kernels]"
           " [--use-4d-slab-native-xw-tensor-coalesced-kernels|--no-4d-slab-native-xw-tensor-coalesced-kernels]"
           " [--use-4d-slab-native-xw-vector4-kernels|--no-4d-slab-native-xw-vector4-kernels]"
           " [--use-4d-slab-native-xw-tiled-kernels|--no-4d-slab-native-xw-tiled-kernels]"
           " [--use-4d-slab-native-xw-layout-stage|--no-4d-slab-native-xw-layout-stage]"
           " [--use-4d-slab-native-xw-native-spectral-layout|--no-4d-slab-native-xw-native-spectral-layout]"
           " [Nx Ny Nz Nw]";
}

template <class T>
inline ffts_3d_benchmark_options<T>
parse_ffts_3d_benchmark_options( int argc, char *argv[], const std::string &binary_name )
{
    ffts_3d_benchmark_options<T> options;
    int                          argi = 1;

    while ( argi < argc )
    {
        const std::string arg = argv[argi];
        if ( arg == "--times" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --times" );
            options.times = std::atoi( argv[argi + 1] );
            if ( options.times < 1 )
                throw std::logic_error( "--times must be at least 1" );
            argi += 2;
        }
        else if ( arg == "--warmup" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --warmup" );
            options.warmup = std::atoi( argv[argi + 1] );
            if ( options.warmup < 0 )
                throw std::logic_error( "--warmup must be non-negative" );
            argi += 2;
        }
        else if ( arg == "--epsilon" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --epsilon" );
            options.epsilon = static_cast<T>( std::atof( argv[argi + 1] ) );
            argi += 2;
        }
        else if ( arg == "--directory" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --directory" );
            options.directory = argv[argi + 1];
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
        throw std::logic_error( usage_ffts_3d_benchmark( binary_name ) );
    }

    return options;
}

template <class T>
inline ffts_4d_benchmark_options<T>
parse_ffts_4d_benchmark_options( int argc, char *argv[], const std::string &binary_name )
{
    ffts_4d_benchmark_options<T> options;
    int                          argi = 1;

    while ( argi < argc )
    {
        const std::string arg = argv[argi];
        if ( arg == "--strategy" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --strategy" );
            const std::string value = argv[argi + 1];
            if ( value == "pencil-direct" )
                options.strategy = ffts_4d_strategy_kind::pencil_direct;
            else if ( value == "pencil-memcpy" )
                options.strategy = ffts_4d_strategy_kind::pencil_memcpy;
            else if ( value == "slab-direct" )
                options.strategy = ffts_4d_strategy_kind::slab_direct;
            else if ( value == "slab-memcpy" )
                options.strategy = ffts_4d_strategy_kind::slab_memcpy;
            else if ( value == "all" )
                options.run_all = true;
            else
                throw std::logic_error( "Unknown strategy '" + value + "'" );
            argi += 2;
        }
        else if ( arg == "--times" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --times" );
            options.times = std::atoi( argv[argi + 1] );
            if ( options.times < 1 )
                throw std::logic_error( "--times must be at least 1" );
            argi += 2;
        }
        else if ( arg == "--warmup" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --warmup" );
            options.warmup = std::atoi( argv[argi + 1] );
            if ( options.warmup < 0 )
                throw std::logic_error( "--warmup must be non-negative" );
            argi += 2;
        }
        else if ( arg == "--epsilon" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --epsilon" );
            options.epsilon = static_cast<T>( std::atof( argv[argi + 1] ) );
            argi += 2;
        }
        else if ( arg == "--directory" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --directory" );
            options.directory = argv[argi + 1];
            argi += 2;
        }
        else
        {
            break;
        }
    }

    if ( argc - argi == 4 )
    {
        options.nx = static_cast<std::size_t>( std::strtoull( argv[argi], NULL, 10 ) );
        options.ny = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
        options.nz = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
        options.nw = static_cast<std::size_t>( std::strtoull( argv[argi + 3], NULL, 10 ) );
    }
    else if ( argc != argi )
    {
        throw std::logic_error( usage_ffts_4d_benchmark( binary_name ) );
    }

    if ( options.nw % 2 != 0 )
        throw std::logic_error( "Nw must be even." );

    return options;
}

template <class T>
inline fftm_3d_benchmark_options<T>
parse_fftm_3d_benchmark_options( int argc, char *argv[], int num_procs, const std::string &binary_name )
{
    fftm_3d_benchmark_options<T> options;
    int                          argi = 1;

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
            else if ( value == "all" )
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
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --times" );
            options.times = std::atoi( argv[argi + 1] );
            if ( options.times < 1 )
                throw std::logic_error( "--times must be at least 1" );
            argi += 2;
        }
        else if ( arg == "--warmup" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --warmup" );
            options.warmup = std::atoi( argv[argi + 1] );
            if ( options.warmup < 0 )
                throw std::logic_error( "--warmup must be non-negative" );
            argi += 2;
        }
        else if ( arg == "--epsilon" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --epsilon" );
            options.epsilon = static_cast<T>( std::atof( argv[argi + 1] ) );
            argi += 2;
        }
        else if ( arg == "--directory" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --directory" );
            options.directory = argv[argi + 1];
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
        else if ( arg == "--write-native-pencil-schedule" )
        {
            options.write_native_pencil_schedule = true;
            argi += 1;
        }
        else if ( arg == "--check-native-pencil-reference-dir" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --check-native-pencil-reference-dir" );
            options.native_pencil_reference_dir = argv[argi + 1];
            options.check_native_pencil_schedule = true;
            options.write_native_pencil_schedule = true;
            argi += 2;
        }
        else if ( arg == "--native-pencil-schedule-check-only" )
        {
            options.native_pencil_schedule_check_only = true;
            options.write_native_pencil_schedule = true;
            argi += 1;
        }
        else if ( arg == "--skip-native-pencil-rank-device-check" )
        {
            options.skip_native_pencil_rank_device_check = true;
            argi += 1;
        }
        else if ( arg == "--fftm-3d-backend" || arg == "--backend-3d" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for " + arg );
            options.backend = parse_fftm_3d_backend( argv[argi + 1] );
            options.backend_explicit = true;
            argi += 2;
        }
        else if ( arg == "--enable-fftm3d-backend-stage-timers" )
        {
            options.enable_fftm3d_backend_stage_timers = true;
            argi += 1;
        }
        else if ( arg == "--disable-fftm3d-backend-stage-timers" )
        {
            options.enable_fftm3d_backend_stage_timers = false;
            argi += 1;
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
        else if ( arg == "--contiguous-forward-send-registration-warmups" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --contiguous-forward-send-registration-warmups" );
            options.contiguous_forward_send_registration_warmups = std::atoi( argv[argi + 1] );
            if ( options.contiguous_forward_send_registration_warmups < 0 )
                throw std::logic_error( "--contiguous-forward-send-registration-warmups must be non-negative" );
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
        else if ( arg == "--native-opt0-y-microbench" )
        {
            options.native_opt0_y_microbench = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-y-microbench" )
        {
            options.native_opt0_y_microbench = false;
            argi += 1;
        }
        else if ( arg == "--native-opt0-y-cross-microbench" )
        {
            options.native_opt0_y_cross_microbench = true;
            argi += 1;
        }
        else if ( arg == "--no-native-opt0-y-cross-microbench" )
        {
            options.native_opt0_y_cross_microbench = false;
            argi += 1;
        }
        else if ( arg == "--native-opt0-y-cross-factory-mode" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --native-opt0-y-cross-factory-mode" );
            options.native_opt0_y_cross_factory_mode = argv[argi + 1];
            if ( options.native_opt0_y_cross_factory_mode != "single" &&
                 options.native_opt0_y_cross_factory_mode != "ref-only" &&
                 options.native_opt0_y_cross_factory_mode != "native-current-only" &&
                 options.native_opt0_y_cross_factory_mode != "native-current-nosync-only" &&
                 options.native_opt0_y_cross_factory_mode != "native-exact-only" &&
                 options.native_opt0_y_cross_factory_mode != "native-owned-only" &&
                 options.native_opt0_y_cross_factory_mode != "native-direct-only" &&
                 options.native_opt0_y_cross_factory_mode != "native-direct-nosync-only" &&
                 options.native_opt0_y_cross_factory_mode != "native-minimal-only" &&
                 options.native_opt0_y_cross_factory_mode != "native-minimal-nosync-only" &&
                 options.native_opt0_y_cross_factory_mode != "ref-then-native-current" &&
                 options.native_opt0_y_cross_factory_mode != "native-current-then-ref" &&
                 options.native_opt0_y_cross_factory_mode != "ref-then-native-current-nosync" &&
                 options.native_opt0_y_cross_factory_mode != "native-current-nosync-then-ref" &&
                 options.native_opt0_y_cross_factory_mode != "ref-then-native-exact" &&
                 options.native_opt0_y_cross_factory_mode != "native-exact-then-ref" &&
                 options.native_opt0_y_cross_factory_mode != "ref-then-native-owned" &&
                 options.native_opt0_y_cross_factory_mode != "native-owned-then-ref" &&
                 options.native_opt0_y_cross_factory_mode != "ref-then-native-direct" &&
                 options.native_opt0_y_cross_factory_mode != "native-direct-then-ref" &&
                 options.native_opt0_y_cross_factory_mode != "ref-then-native-direct-nosync" &&
                 options.native_opt0_y_cross_factory_mode != "native-direct-nosync-then-ref" &&
                 options.native_opt0_y_cross_factory_mode != "ref-then-native-minimal" &&
                 options.native_opt0_y_cross_factory_mode != "native-minimal-then-ref" &&
                 options.native_opt0_y_cross_factory_mode != "ref-then-native-minimal-nosync" &&
                 options.native_opt0_y_cross_factory_mode != "native-minimal-nosync-then-ref" )
            {
                throw std::logic_error(
                    "--native-opt0-y-cross-factory-mode must be single, ref-only, native-current-only, "
                    "native-current-nosync-only, native-exact-only, native-owned-only, native-direct-only, "
                    "native-direct-nosync-only, native-minimal-only, native-minimal-nosync-only, "
                    "ref-then-native-current, native-current-then-ref, "
                    "ref-then-native-current-nosync, native-current-nosync-then-ref, "
                    "ref-then-native-exact, native-exact-then-ref, ref-then-native-owned, or "
                    "native-owned-then-ref, ref-then-native-direct, native-direct-then-ref, "
                    "ref-then-native-direct-nosync, native-direct-nosync-then-ref, "
                    "ref-then-native-minimal, native-minimal-then-ref, "
                    "ref-then-native-minimal-nosync, or native-minimal-nosync-then-ref"
                );
            }
            argi += 2;
        }
        else if ( arg == "--native-opt0-y-microbench-iterations" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --native-opt0-y-microbench-iterations" );
            options.native_opt0_y_microbench_iterations = std::atoi( argv[argi + 1] );
            if ( options.native_opt0_y_microbench_iterations < 1 )
                throw std::logic_error( "--native-opt0-y-microbench-iterations must be at least 1" );
            argi += 2;
        }
        else if ( arg == "--native-opt0-y-microbench-warmup" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --native-opt0-y-microbench-warmup" );
            options.native_opt0_y_microbench_warmup = std::atoi( argv[argi + 1] );
            if ( options.native_opt0_y_microbench_warmup < 0 )
                throw std::logic_error( "--native-opt0-y-microbench-warmup must be non-negative" );
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
        throw std::logic_error( usage_fftm_3d_benchmark( binary_name ) );
    }

    const bool grid_was_explicit = options.p1 != 0 && options.p2 != 0;
    {
        fftm_3d_test_options base_options = make_fftm_3d_test_options_base( options );
        if ( apply_fftm_3d_autotune_config( base_options, num_procs, !grid_was_explicit ) )
            copy_fftm_3d_test_options_base( options, base_options );
    }

    if ( options.p1 == 0 || options.p2 == 0 )
    {
        fftm_3d_test_options base_options = make_fftm_3d_test_options_base( options );
        const auto grid       = choose_grid_3d( base_options, options.strategy, num_procs );
        options.p1            = grid.first;
        options.p2            = grid.second;
    }

    return options;
}

template <class T>
inline fftm_4d_benchmark_options<T>
parse_fftm_4d_benchmark_options( int argc, char *argv[], int num_procs, const std::string &binary_name )
{
    (void)num_procs;
    fftm_4d_benchmark_options<T> options;
    int                          argi = 1;

    while ( argi < argc )
    {
        const std::string arg = argv[argi];
        if ( arg == "--strategy" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --strategy" );
            const std::string value = argv[argi + 1];
            if ( value == "pencil-pencil" )
                options.strategy = fftm_4d_strategy_kind::pencil_pencil;
            else if ( value == "slab-slab" )
                options.strategy = fftm_4d_strategy_kind::slab_slab;
            else if ( value == "all" )
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
            if ( argi + 3 >= argc )
                throw std::logic_error( "Missing values for --grid P1 P2 P3" );
            options.p1 = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            options.p2 = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
            options.p3 = static_cast<std::size_t>( std::strtoull( argv[argi + 3], NULL, 10 ) );
            argi += 4;
        }
        else if ( arg == "--times" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --times" );
            options.times = std::atoi( argv[argi + 1] );
            if ( options.times < 1 )
                throw std::logic_error( "--times must be at least 1" );
            argi += 2;
        }
        else if ( arg == "--warmup" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --warmup" );
            options.warmup = std::atoi( argv[argi + 1] );
            if ( options.warmup < 0 )
                throw std::logic_error( "--warmup must be non-negative" );
            argi += 2;
        }
        else if ( arg == "--epsilon" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --epsilon" );
            options.epsilon = static_cast<T>( std::atof( argv[argi + 1] ) );
            argi += 2;
        }
        else if ( arg == "--directory" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --directory" );
            options.directory = argv[argi + 1];
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
        else if ( arg == "--use-4d-slab-native-xw-transpose" )
        {
            options.use_4d_slab_native_xw_transpose = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-slab-native-xw-transpose" )
        {
            options.use_4d_slab_native_xw_transpose = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-slab-native-xw-batched-peer-kernels" )
        {
            options.use_4d_slab_native_xw_batched_peer_kernels = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-slab-native-xw-batched-peer-kernels" )
        {
            options.use_4d_slab_native_xw_batched_peer_kernels = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-slab-native-xw-tensor-coalesced-kernels" )
        {
            options.use_4d_slab_native_xw_tensor_coalesced_kernels = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-slab-native-xw-tensor-coalesced-kernels" )
        {
            options.use_4d_slab_native_xw_tensor_coalesced_kernels = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-slab-native-xw-vector4-kernels" )
        {
            options.use_4d_slab_native_xw_vector4_kernels = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-slab-native-xw-vector4-kernels" )
        {
            options.use_4d_slab_native_xw_vector4_kernels = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-slab-native-xw-tiled-kernels" )
        {
            options.use_4d_slab_native_xw_tiled_kernels = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-slab-native-xw-tiled-kernels" )
        {
            options.use_4d_slab_native_xw_tiled_kernels = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-slab-native-xw-layout-stage" )
        {
            options.use_4d_slab_native_xw_layout_stage = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-slab-native-xw-layout-stage" )
        {
            options.use_4d_slab_native_xw_layout_stage = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-slab-native-xw-native-spectral-layout" )
        {
            options.use_4d_slab_native_xw_native_spectral_layout = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-slab-native-xw-native-spectral-layout" )
        {
            options.use_4d_slab_native_xw_native_spectral_layout = false;
            argi += 1;
        }
        else
        {
            break;
        }
    }

    if ( argc - argi == 4 )
    {
        options.nx = static_cast<std::size_t>( std::strtoull( argv[argi], NULL, 10 ) );
        options.ny = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
        options.nz = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
        options.nw = static_cast<std::size_t>( std::strtoull( argv[argi + 3], NULL, 10 ) );
    }
    else if ( argc != argi )
    {
        throw std::logic_error( usage_fftm_4d_benchmark( binary_name ) );
    }

    if ( options.nw % 2 != 0 )
        throw std::logic_error( "Nw must be even." );

    if ( options.p1 == 0 || options.p2 == 0 || options.p3 == 0 )
    {
        fftm_4d_test_options base_options;
        base_options.strategy = options.strategy;
        base_options.run_all  = options.run_all;
        base_options.mode     = options.mode;
        base_options.nx       = options.nx;
        base_options.ny       = options.ny;
        base_options.nz       = options.nz;
        base_options.nw       = options.nw;
        base_options.p1       = options.p1;
        base_options.p2       = options.p2;
        base_options.p3       = options.p3;
        base_options.times    = options.times;
        base_options.use_direct_backward_receive = options.use_direct_backward_receive;
        base_options.direct_p2p_cuda_aware       = options.direct_p2p_cuda_aware;
        base_options.use_p2p_byte_transfer       = options.use_p2p_byte_transfer;
        base_options.use_fft_exec_no_sync        = options.use_fft_exec_no_sync;
        base_options.enable_native_stage_timers  = options.enable_native_stage_timers;
        const auto grid       = choose_grid_4d( base_options, options.strategy, num_procs );
        options.p1            = std::get<0>( grid );
        options.p2            = std::get<1>( grid );
        options.p3            = std::get<2>( grid );
    }

    return options;
}

template <class T>
inline ::fftm::fftm_init_options make_fftm_init_options( const fftm_3d_benchmark_options<T> &options )
{
    ::fftm::fftm_init_options init_options;
    init_options.use_direct_backward_receive = options.use_direct_backward_receive;
    init_options.direct_p2p_cuda_aware       = options.direct_p2p_cuda_aware;
    init_options.use_p2p_send_thread         = options.use_p2p_send_thread;
	    init_options.use_p2p_byte_transfer       = options.use_p2p_byte_transfer;
	    init_options.use_persistent_p2p          = options.use_persistent_p2p;
	    init_options.use_ready_p2p_send          = options.use_ready_p2p_send;
    init_options.print_pencil_schedule       = options.print_pencil_schedule;
    init_options.use_direct_forward_byte_receive = options.use_direct_forward_byte_receive;
    init_options.use_stable_forward_byte_send_buffer = options.use_stable_forward_byte_send_buffer;
    init_options.use_ready_stable_forward_byte_send_buffer = options.use_ready_stable_forward_byte_send_buffer;
    init_options.use_contiguous_forward_byte_send = options.use_contiguous_forward_byte_send;
    init_options.use_physical_forward_peer_exchange = options.use_physical_forward_peer_exchange;
    init_options.contiguous_forward_send_mode = options.contiguous_forward_send_mode;
    init_options.contiguous_forward_send_chunk_bytes = options.contiguous_forward_send_chunk_bytes;
    init_options.large_count_p2p_transport  = options.large_count_p2p_transport;
    init_options.use_large_count_datatype_cache = options.use_large_count_datatype_cache;
    init_options.use_fft_exec_no_sync = options.use_fft_exec_no_sync;
    init_options.use_native_backward_second_peer_loop = options.use_native_backward_second_peer_loop;
    init_options.use_native_opt0_default_z_layout = options.use_native_opt0_default_z_layout;
    init_options.use_native_opt0_reference_y_buffer_topology = options.use_native_opt0_reference_y_buffer_topology;
    init_options.use_native_opt0_compact_y_workarea = options.use_native_opt0_compact_y_workarea;
    init_options.use_native_opt0_tight_y_plan_sequence = options.use_native_opt0_tight_y_plan_sequence;
    init_options.use_native_opt0_shared_y_plan_handles = options.use_native_opt0_shared_y_plan_handles;
    init_options.use_native_opt0_y_group_device_sync = options.use_native_opt0_y_group_device_sync;
    init_options.use_native_opt0_y_no_sync_exec = options.use_native_opt0_y_no_sync_exec;
    init_options.use_native_opt0_raw_y_plan_array_executor = options.use_native_opt0_raw_y_plan_array_executor;
    init_options.use_native_opt0_reference_y_plan_lifecycle = options.use_native_opt0_reference_y_plan_lifecycle;
    init_options.use_native_opt0_reference_y_plan_bundle = options.use_native_opt0_reference_y_plan_bundle;
    init_options.use_native_opt0_raw_y_plan_bundle = options.use_native_opt0_raw_y_plan_bundle;
    init_options.use_native_opt0_y_plan_bundle_stream_first = options.use_native_opt0_y_plan_bundle_stream_first;
    init_options.use_native_opt0_raw_y_plan_bundle_reference_streams =
        options.use_native_opt0_raw_y_plan_bundle_reference_streams;
    init_options.use_native_opt0_reference_local_plan_context = options.use_native_opt0_reference_local_plan_context;
    init_options.allow_native_opt0_diagnostic_variants = options.allow_native_opt0_diagnostic_variants;
    init_options.use_native_opt0_memory_feasibility_guard =
        options.use_native_opt0_memory_feasibility_guard;
    init_options.native_opt0_memory_feasibility_reserve_bytes =
        options.native_opt0_memory_feasibility_reserve_bytes;
    init_options.enable_local_fft_diagnostics =
        options.enable_local_fft_diagnostics || options.native_opt0_y_microbench;
    init_options.enable_native_stage_timers = options.enable_native_stage_timers;
    init_options.local_fft_diagnostics_directory =
        options.local_fft_diagnostics_directory.empty() ? options.directory : options.local_fft_diagnostics_directory;
    init_options.local_fft_diagnostics_label =
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

template <class T>
inline ::fftm::fftm_init_options make_fftm_init_options( const fftm_4d_benchmark_options<T> &options )
{
    ::fftm::fftm_init_options init_options;
    init_options.use_direct_backward_receive = options.use_direct_backward_receive;
    init_options.direct_p2p_cuda_aware       = options.direct_p2p_cuda_aware;
    init_options.use_p2p_byte_transfer       = options.use_p2p_byte_transfer;
    init_options.use_fft_exec_no_sync        = options.use_fft_exec_no_sync;
    init_options.enable_native_stage_timers  = options.enable_native_stage_timers;
    init_options.use_4d_slab_native_xw_transpose = options.use_4d_slab_native_xw_transpose;
    init_options.use_4d_slab_native_xw_batched_peer_kernels =
        options.use_4d_slab_native_xw_batched_peer_kernels;
    init_options.use_4d_slab_native_xw_tensor_coalesced_kernels =
        options.use_4d_slab_native_xw_tensor_coalesced_kernels;
    init_options.use_4d_slab_native_xw_vector4_kernels = options.use_4d_slab_native_xw_vector4_kernels;
    init_options.use_4d_slab_native_xw_tiled_kernels = options.use_4d_slab_native_xw_tiled_kernels;
    init_options.use_4d_slab_native_xw_layout_stage = options.use_4d_slab_native_xw_layout_stage;
    init_options.spectral_layout_4d =
        options.use_4d_slab_native_xw_native_spectral_layout ? ::fftm::fftm_4d_spectral_layout::native_xzwy
                                                             : ::fftm::fftm_4d_spectral_layout::public_yzwx;
    init_options.use_4d_slab_native_xw_native_spectral_layout =
        options.use_4d_slab_native_xw_native_spectral_layout;
    return init_options;
}

inline const char *ffts_4d_strategy_name( ffts_4d_strategy_kind strategy )
{
    switch ( strategy )
    {
    case ffts_4d_strategy_kind::pencil_direct:
        return "pencil-direct";
    case ffts_4d_strategy_kind::pencil_memcpy:
        return "pencil-memcpy";
    case ffts_4d_strategy_kind::slab_direct:
        return "slab-direct";
    case ffts_4d_strategy_kind::slab_memcpy:
        return "slab-memcpy";
    }
    return "unknown";
}

} // namespace detail
} // namespace test
} // namespace fftm

#endif
