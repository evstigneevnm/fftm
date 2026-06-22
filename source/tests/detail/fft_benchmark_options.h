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
    ::fftm::fftm_3d_large_count_p2p_transport large_count_p2p_transport =
        ::fftm::fftm_3d_large_count_p2p_transport::hindexed;
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
           " [--large-count-p2p-transport hindexed|mpi-count|element-count|chunked]"
	           " [--pencil-layout auto|opt0|opt1|legacy]"
           " [--pencil-pipeline staged|fused|egger|egger-parity] [Nx Ny Nz]";
}

inline std::string usage_fftm_4d_benchmark( const std::string &binary_name )
{
    return "USAGE: " + binary_name +
           " [--strategy pencil-pencil|slab-slab|all]"
           " [--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw]"
           " [--grid P1 P2 P3] [--times repeats] [--warmup repeats] [--epsilon eps] [--directory path]"
           " [--use-direct-backward-receive|--no-direct-backward-receive]"
           " [--direct-p2p-cuda-aware|--no-direct-p2p-cuda-aware]"
           " [--use-p2p-byte-transfer|--no-p2p-byte-transfer] [Nx Ny Nz Nw]";
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
        else if ( arg == "--large-count-p2p-transport" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --large-count-p2p-transport" );
            options.large_count_p2p_transport = parse_large_count_p2p_transport( argv[argi + 1] );
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
            else if ( value == "egger" )
                options.pencil_pipeline = fftm_3d_pencil_pipeline_kind::egger;
            else if ( value == "egger-parity" || value == "egger_parity" )
                options.pencil_pipeline = fftm_3d_pencil_pipeline_kind::egger_parity;
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

    if ( options.p1 == 0 || options.p2 == 0 )
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
        base_options.use_direct_backward_receive = options.use_direct_backward_receive;
        base_options.direct_p2p_cuda_aware       = options.direct_p2p_cuda_aware;
        base_options.use_p2p_send_thread         = options.use_p2p_send_thread;
	        base_options.use_p2p_byte_transfer       = options.use_p2p_byte_transfer;
	        base_options.use_persistent_p2p          = options.use_persistent_p2p;
	        base_options.use_ready_p2p_send          = options.use_ready_p2p_send;
	        base_options.print_pencil_schedule       = options.print_pencil_schedule;
        base_options.use_direct_forward_byte_receive = options.use_direct_forward_byte_receive;
        base_options.large_count_p2p_transport  = options.large_count_p2p_transport;
	        base_options.pencil_layout               = options.pencil_layout;
        base_options.pencil_pipeline             = options.pencil_pipeline;
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
    init_options.large_count_p2p_transport  = options.large_count_p2p_transport;
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
