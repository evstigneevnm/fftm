#ifndef __FFTM_TESTS_DETAIL_FFTM_4D_TEST_OPTIONS_H__
#define __FFTM_TESTS_DETAIL_FFTM_4D_TEST_OPTIONS_H__

#include <algorithm>
#include <cstdlib>
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
    int                           times     = 1;
    int                           warmup    = 0;
    bool                          use_direct_backward_receive = false;
    bool                          direct_p2p_cuda_aware       = true;
    bool                          use_p2p_byte_transfer       = false;
    bool                          use_fft_exec_no_sync        = false;
};

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
        usage += " [--threshold eps]";
    if ( allow_times )
        usage += " [--times repeats] [--warmup repeats]";
    usage += " [--use-direct-backward-receive|--no-direct-backward-receive]";
    usage += " [--direct-p2p-cuda-aware|--no-direct-p2p-cuda-aware]";
    usage += " [--use-p2p-byte-transfer|--no-p2p-byte-transfer]";
    usage += " [--use-fft-exec-no-sync|--no-fft-exec-no-sync]";
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
    init_options.use_direct_backward_receive = options.use_direct_backward_receive;
    init_options.direct_p2p_cuda_aware       = options.direct_p2p_cuda_aware;
    init_options.use_p2p_byte_transfer       = options.use_p2p_byte_transfer;
    init_options.use_fft_exec_no_sync        = options.use_fft_exec_no_sync;
    return init_options;
}

} // namespace detail
} // namespace test
} // namespace fftm

#endif
