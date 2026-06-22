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
    egger,
    egger_parity
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
    ::fftm::fftm_3d_large_count_p2p_transport large_count_p2p_transport =
        ::fftm::fftm_3d_large_count_p2p_transport::hindexed;
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

inline const char *pencil_pipeline_name( fftm_3d_pencil_pipeline_kind pipeline )
{
    switch ( pipeline )
    {
    case fftm_3d_pencil_pipeline_kind::staged:
        return "staged";
    case fftm_3d_pencil_pipeline_kind::fused:
        return "fused";
    case fftm_3d_pencil_pipeline_kind::egger:
        return "egger";
    case fftm_3d_pencil_pipeline_kind::egger_parity:
        return "egger-parity";
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
    case fftm_3d_pencil_pipeline_kind::egger:
        return ::fftm::fftm_3d_pencil_pipeline::egger;
    case fftm_3d_pencil_pipeline_kind::egger_parity:
        return ::fftm::fftm_3d_pencil_pipeline::egger_parity;
    }
    return ::fftm::fftm_3d_pencil_pipeline::staged;
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
    usage += " [--large-count-p2p-transport hindexed|mpi-count|element-count|chunked]";
	    usage += " [--pencil-layout auto|opt0|opt1|legacy]";
    usage += " [--pencil-pipeline staged|fused|egger|egger-parity]";
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
	    init_options.use_persistent_p2p          = options.use_persistent_p2p;
	    init_options.use_ready_p2p_send          = options.use_ready_p2p_send;
	    init_options.print_pencil_schedule       = options.print_pencil_schedule;
    init_options.use_direct_forward_byte_receive = options.use_direct_forward_byte_receive;
    init_options.large_count_p2p_transport  = options.large_count_p2p_transport;
	    init_options.pencil_layout_3d            = to_fftm_pencil_layout( options.pencil_layout );
    init_options.pencil_pipeline_3d          = to_fftm_pencil_pipeline( options.pencil_pipeline );
    return init_options;
}

} // namespace detail
} // namespace test
} // namespace fftm

#endif
