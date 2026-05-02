#ifndef __FFTM_TESTS_DETAIL_FFTM_3D_TEST_OPTIONS_H__
#define __FFTM_TESTS_DETAIL_FFTM_3D_TEST_OPTIONS_H__

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include <fftm.hpp>

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
};

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
    usage += " [Nx Ny Nz]";
    return usage;
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
    init_options.use_direct_backward_receive = options.use_direct_backward_receive;
    init_options.direct_p2p_cuda_aware       = options.direct_p2p_cuda_aware;
    init_options.use_p2p_send_thread         = options.use_p2p_send_thread;
    init_options.use_p2p_byte_transfer       = options.use_p2p_byte_transfer;
    return init_options;
}

} // namespace detail
} // namespace test
} // namespace fftm

#endif
