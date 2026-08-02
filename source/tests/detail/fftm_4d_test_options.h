#ifndef __FFTM_TESTS_DETAIL_FFTM_4D_TEST_OPTIONS_H__
#define __FFTM_TESTS_DETAIL_FFTM_4D_TEST_OPTIONS_H__

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include <fftm.hpp>

namespace fftm
{
namespace test
{
namespace detail
{

enum class fftm_4d_strategy_kind
{
    pencil_pencil,
    slab_slab
};

struct fftm_4d_test_options
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
    double                        threshold = 1.0e-11;
    double                        l2_threshold = -1.0;
    double                        h1_threshold = -1.0;
    int                           times     = 1;
    int                           warmup    = 0;
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
    bool                          use_4d_native_xw_direct_layout = false;
    bool                          use_4d_native_xw_chunked_transport = false;
    std::size_t                   native_xw_chunk_mib = 512;
    std::size_t                   native_xw_chunk_window = 0;
    bool                          use_4d_native_xw_compact_staging = false;
    bool                          use_4d_slab_native_work_area_alias = false;
    bool                          use_4d_slab_native_wz_communication_layout = false;
    std::size_t                   slab_native_wz_plan_concurrency = 1;
    bool                          use_4d_slab_native_wz_ready_pipeline = false;
    bool                          use_4d_pencil_same_zw_peer_paired = false;
    bool                          use_4d_pencil_same_zw_native_layout = true;
    bool                          use_4d_pencil_degenerate_xw_slab_path = false;
    bool                          use_4d_pencil_degenerate_local_transposes = false;
    bool                          use_4d_pencil_degenerate_same_xw_native = false;
    bool                          use_4d_pencil_degenerate_wz_sliced_z_fft = true;
    bool                          use_4d_slab_native_xw_native_spectral_layout = false;
};

inline std::size_t parse_native_xw_chunk_window( const char *value )
{
    const std::string text = value == nullptr ? std::string() : std::string( value );
    if ( text == "all" || text == "full" || text == "0" )
        return 0;
    char               *end    = nullptr;
    const unsigned long long parsed = std::strtoull( text.c_str(), &end, 10 );
    if ( text.empty() || end == text.c_str() || *end != '\0' || parsed == 0 ||
         parsed > static_cast<unsigned long long>( std::numeric_limits<std::size_t>::max() ) )
    {
        throw std::logic_error( "--4d-native-xw-chunk-window must be 'all' or a positive integer" );
    }
    return static_cast<std::size_t>( parsed );
}

inline std::tuple<std::size_t, std::size_t, std::size_t> choose_balanced_grid_4d( std::size_t num_procs )
{
    std::size_t best_p1   = 1;
    std::size_t best_p2   = 1;
    std::size_t best_p3   = num_procs;
    std::size_t best_span = best_p3 - best_p1;

    for ( std::size_t p1 = 1; p1 <= num_procs; ++p1 )
    {
        if ( num_procs % p1 != 0 )
            continue;

        const std::size_t rem1 = num_procs / p1;
        for ( std::size_t p2 = 1; p2 <= rem1; ++p2 )
        {
            if ( rem1 % p2 != 0 )
                continue;

            const std::size_t p3      = rem1 / p2;
            const std::size_t max_dim = std::max( p1, std::max( p2, p3 ) );
            const std::size_t min_dim = std::min( p1, std::min( p2, p3 ) );
            const std::size_t span    = max_dim - min_dim;

            if ( span < best_span )
            {
                best_span = span;
                best_p1   = p1;
                best_p2   = p2;
                best_p3   = p3;
            }
        }
    }

    return std::make_tuple( best_p1, best_p2, best_p3 );
}

inline std::tuple<std::size_t, std::size_t, std::size_t>
choose_grid_4d( const fftm_4d_test_options &options, fftm_4d_strategy_kind strategy, int num_procs )
{
    if ( options.p1 != 0 && options.p2 != 0 && options.p3 != 0 )
        return std::make_tuple( options.p1, options.p2, options.p3 );

    if ( strategy == fftm_4d_strategy_kind::slab_slab )
        return std::make_tuple( 1u, static_cast<std::size_t>( num_procs ), 1u );

    return choose_balanced_grid_4d( static_cast<std::size_t>( num_procs ) );
}

inline std::string
usage_fftm_4d_test( const std::string &binary_name, bool allow_strategy_all, bool allow_threshold, bool allow_times )
{
    std::string usage = "USAGE: " + binary_name + " [--strategy pencil-pencil|slab-slab";
    if ( allow_strategy_all )
        usage += "|all";
    usage += "] [--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw] [--grid P1 P2 P3]";
    if ( allow_threshold )
        usage += " [--threshold eps] [--l2-threshold eps] [--h1-threshold eps]";
    if ( allow_times )
        usage += " [--times repeats] [--warmup repeats]";
    usage += " [--use-direct-backward-receive|--no-direct-backward-receive]";
    usage += " [--direct-p2p-cuda-aware|--no-direct-p2p-cuda-aware]";
    usage += " [--use-p2p-byte-transfer|--no-p2p-byte-transfer]";
    usage += " [--use-fft-exec-no-sync|--no-fft-exec-no-sync]";
    usage += " [--enable-native-stage-timers|--disable-native-stage-timers]";
    usage += " [--use-4d-slab-native-xw-transpose|--no-4d-slab-native-xw-transpose]";
    usage += " [--use-4d-slab-native-xw-batched-peer-kernels|--no-4d-slab-native-xw-batched-peer-kernels]";
    usage += " [--use-4d-slab-native-xw-tensor-coalesced-kernels|--no-4d-slab-native-xw-tensor-coalesced-kernels]";
    usage += " [--use-4d-slab-native-xw-vector4-kernels|--no-4d-slab-native-xw-vector4-kernels]";
    usage += " [--use-4d-slab-native-xw-tiled-kernels|--no-4d-slab-native-xw-tiled-kernels]";
    usage += " [--use-4d-slab-native-xw-layout-stage|--no-4d-slab-native-xw-layout-stage]";
    usage += " [--use-4d-native-xw-direct-layout|--no-4d-native-xw-direct-layout]";
    usage += " [--use-4d-native-xw-chunked-transport|--no-4d-native-xw-chunked-transport]";
    usage += " [--4d-native-xw-chunk-mib MiB]";
    usage += " [--4d-native-xw-chunk-window all|N]";
    usage += " [--use-4d-native-xw-compact-staging|--no-4d-native-xw-compact-staging]";
    usage += " [--use-4d-slab-native-work-area-alias|--no-4d-slab-native-work-area-alias]";
    usage += " [--use-4d-slab-native-wz-communication-layout|--no-4d-slab-native-wz-communication-layout]";
    usage += " [--use-4d-pencil-same-zw-peer-paired|--no-4d-pencil-same-zw-peer-paired]";
    usage += " [--use-4d-pencil-same-zw-native-layout|--no-4d-pencil-same-zw-native-layout]";
    usage += " [--use-4d-pencil-degenerate-xw-slab-path|--no-4d-pencil-degenerate-xw-slab-path]";
    usage += " [--use-4d-pencil-degenerate-local-transposes|--no-4d-pencil-degenerate-local-transposes]";
    usage += " [--use-4d-pencil-degenerate-same-xw-native|--no-4d-pencil-degenerate-same-xw-native]";
    usage += " [--use-4d-pencil-degenerate-wz-sliced-z-fft|--no-4d-pencil-degenerate-wz-sliced-z-fft]";
    usage +=
        " [--use-4d-slab-native-xw-native-spectral-layout|--no-4d-slab-native-xw-native-spectral-layout]";
    usage += " [Nx Ny Nz Nw]";
    return usage;
}

inline fftm_4d_test_options parse_fftm_4d_test_options(
    int argc, char *argv[], const std::string &binary_name, bool allow_strategy_all, bool allow_threshold,
    bool allow_times, const fftm_4d_test_options &defaults = fftm_4d_test_options()
)
{
    fftm_4d_test_options options = defaults;
    int                  argi    = 1;

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
            if ( argi + 3 >= argc )
                throw std::logic_error( "Missing values for --grid P1 P2 P3" );

            options.p1 = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            options.p2 = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
            options.p3 = static_cast<std::size_t>( std::strtoull( argv[argi + 3], NULL, 10 ) );
            argi += 4;
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
        else if ( arg == "--l2-threshold" )
        {
            if ( !allow_threshold )
                throw std::logic_error( "Unknown option '--l2-threshold'" );
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --l2-threshold" );

            options.l2_threshold = std::atof( argv[argi + 1] );
            argi += 2;
        }
        else if ( arg == "--h1-threshold" )
        {
            if ( !allow_threshold )
                throw std::logic_error( "Unknown option '--h1-threshold'" );
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --h1-threshold" );

            options.h1_threshold = std::atof( argv[argi + 1] );
            argi += 2;
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
        else if ( arg == "--use-4d-native-xw-direct-layout" )
        {
            options.use_4d_native_xw_direct_layout = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-native-xw-direct-layout" )
        {
            options.use_4d_native_xw_direct_layout = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-native-xw-chunked-transport" )
        {
            options.use_4d_native_xw_chunked_transport = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-native-xw-chunked-transport" )
        {
            options.use_4d_native_xw_chunked_transport = false;
            argi += 1;
        }
        else if ( arg == "--4d-native-xw-chunk-mib" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --4d-native-xw-chunk-mib" );
            options.native_xw_chunk_mib = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            if ( options.native_xw_chunk_mib == 0 )
                throw std::logic_error( "--4d-native-xw-chunk-mib must be positive" );
            argi += 2;
        }
        else if ( arg == "--4d-native-xw-chunk-window" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --4d-native-xw-chunk-window" );
            options.native_xw_chunk_window = parse_native_xw_chunk_window( argv[argi + 1] );
            argi += 2;
        }
        else if ( arg == "--use-4d-native-xw-compact-staging" )
        {
            options.use_4d_native_xw_compact_staging = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-native-xw-compact-staging" )
        {
            options.use_4d_native_xw_compact_staging = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-slab-native-work-area-alias" )
        {
            options.use_4d_slab_native_work_area_alias = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-slab-native-work-area-alias" )
        {
            options.use_4d_slab_native_work_area_alias = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-slab-native-wz-communication-layout" )
        {
            options.use_4d_slab_native_wz_communication_layout = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-slab-native-wz-communication-layout" )
        {
            options.use_4d_slab_native_wz_communication_layout = false;
            argi += 1;
        }
        else if ( arg == "--4d-slab-native-wz-plan-concurrency" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --4d-slab-native-wz-plan-concurrency" );
            options.slab_native_wz_plan_concurrency =
                static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            if ( options.slab_native_wz_plan_concurrency == 0 )
                throw std::logic_error( "--4d-slab-native-wz-plan-concurrency must be positive" );
            argi += 2;
        }
        else if ( arg == "--use-4d-slab-native-wz-ready-pipeline" )
        {
            options.use_4d_slab_native_wz_ready_pipeline = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-slab-native-wz-ready-pipeline" )
        {
            options.use_4d_slab_native_wz_ready_pipeline = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-pencil-same-zw-peer-paired" )
        {
            options.use_4d_pencil_same_zw_peer_paired = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-pencil-same-zw-peer-paired" )
        {
            options.use_4d_pencil_same_zw_peer_paired = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-pencil-same-zw-native-layout" )
        {
            options.use_4d_pencil_same_zw_native_layout = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-pencil-same-zw-native-layout" )
        {
            options.use_4d_pencil_same_zw_native_layout = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-pencil-degenerate-xw-slab-path" )
        {
            options.use_4d_pencil_degenerate_xw_slab_path = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-pencil-degenerate-xw-slab-path" )
        {
            options.use_4d_pencil_degenerate_xw_slab_path = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-pencil-degenerate-local-transposes" )
        {
            options.use_4d_pencil_degenerate_local_transposes = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-pencil-degenerate-local-transposes" )
        {
            options.use_4d_pencil_degenerate_local_transposes = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-pencil-degenerate-same-xw-native" )
        {
            options.use_4d_pencil_degenerate_same_xw_native = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-pencil-degenerate-same-xw-native" )
        {
            options.use_4d_pencil_degenerate_same_xw_native = false;
            argi += 1;
        }
        else if ( arg == "--use-4d-pencil-degenerate-wz-sliced-z-fft" )
        {
            options.use_4d_pencil_degenerate_wz_sliced_z_fft = true;
            argi += 1;
        }
        else if ( arg == "--no-4d-pencil-degenerate-wz-sliced-z-fft" )
        {
            options.use_4d_pencil_degenerate_wz_sliced_z_fft = false;
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
        throw std::logic_error( usage_fftm_4d_test( binary_name, allow_strategy_all, allow_threshold, allow_times ) );
    }

    if ( options.nw % 2 != 0 )
        throw std::logic_error( "Nw must be even." );

    return options;
}

inline ::fftm::fftm_init_options make_fftm_init_options( const fftm_4d_test_options &options )
{
    ::fftm::fftm_init_options init_options;
    init_options.reporting = ::fftm::profiling_reporting_options();
    init_options.execution.use_direct_backward_receive = options.use_direct_backward_receive;
    init_options.execution.direct_p2p_cuda_aware       = options.direct_p2p_cuda_aware;
    init_options.execution.use_p2p_byte_transfer       = options.use_p2p_byte_transfer;
    init_options.diagnostics.use_fft_exec_no_sync        = options.use_fft_exec_no_sync;
    init_options.diagnostics.enable_native_stage_timers  = options.enable_native_stage_timers;
    init_options.execution.use_4d_slab_native_xw_transpose = options.use_4d_slab_native_xw_transpose;
    init_options.diagnostics.use_4d_slab_native_xw_batched_peer_kernels =
        options.use_4d_slab_native_xw_batched_peer_kernels;
    init_options.diagnostics.use_4d_slab_native_xw_tensor_coalesced_kernels =
        options.use_4d_slab_native_xw_tensor_coalesced_kernels;
    init_options.diagnostics.use_4d_slab_native_xw_vector4_kernels = options.use_4d_slab_native_xw_vector4_kernels;
    init_options.diagnostics.use_4d_slab_native_xw_tiled_kernels = options.use_4d_slab_native_xw_tiled_kernels;
    init_options.diagnostics.use_4d_slab_native_xw_layout_stage = options.use_4d_slab_native_xw_layout_stage;
    init_options.execution.use_4d_native_xw_direct_layout = options.use_4d_native_xw_direct_layout;
    init_options.execution.use_4d_native_xw_chunked_transport = options.use_4d_native_xw_chunked_transport;
    init_options.execution.native_xw_chunk_bytes =
        options.native_xw_chunk_mib * static_cast<std::size_t>( 1024 ) * static_cast<std::size_t>( 1024 );
    init_options.execution.native_xw_chunk_window = options.native_xw_chunk_window;
    init_options.execution.use_4d_native_xw_compact_staging = options.use_4d_native_xw_compact_staging;
    init_options.execution.use_4d_slab_native_work_area_alias = options.use_4d_slab_native_work_area_alias;
    init_options.execution.use_4d_slab_native_wz_communication_layout =
        options.use_4d_slab_native_wz_communication_layout;
    init_options.execution.slab_native_wz_plan_concurrency = options.slab_native_wz_plan_concurrency;
    init_options.execution.use_4d_slab_native_wz_ready_pipeline = options.use_4d_slab_native_wz_ready_pipeline;
    init_options.diagnostics.use_4d_pencil_same_zw_peer_paired = options.use_4d_pencil_same_zw_peer_paired;
    init_options.execution.use_4d_pencil_same_zw_native_layout = options.use_4d_pencil_same_zw_native_layout;
    init_options.diagnostics.use_4d_pencil_degenerate_xw_slab_path = options.use_4d_pencil_degenerate_xw_slab_path;
    init_options.execution.use_4d_pencil_degenerate_local_transposes =
        options.use_4d_pencil_degenerate_local_transposes;
    init_options.execution.use_4d_pencil_degenerate_same_xw_native =
        options.use_4d_pencil_degenerate_same_xw_native;
    init_options.execution.use_4d_pencil_degenerate_wz_sliced_z_fft =
        options.use_4d_pencil_degenerate_wz_sliced_z_fft;
    init_options.spectral_layout_4d =
        options.use_4d_slab_native_xw_native_spectral_layout ? ::fftm::fftm_4d_spectral_layout::native_xzwy
                                                             : ::fftm::fftm_4d_spectral_layout::public_yzwx;
    init_options.diagnostics.use_4d_slab_native_xw_native_spectral_layout =
        options.use_4d_slab_native_xw_native_spectral_layout;
    return init_options;
}

} // namespace detail
} // namespace test
} // namespace fftm

#endif
