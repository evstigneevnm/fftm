#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>

#include <fftm_autotune_measure_4d.hpp>
#include <fftm_backend.hpp>

namespace
{

using real_t = double;
using fft_backend_t = fftm::device_backend::fft<real_t>;
using runtime_api_t = typename fft_backend_t::runtime_api;
using backend_t = fftm::device_backend::scfd_backend;
using comm_t = scfd::communication::mpi_comm_info;
using log_t = scfd::utils::log_mpi;

struct runtime_options
{
    std::size_t size = 32;
    int warmup = 0;
    int iterations = 1;
    std::string cache_file = "/tmp/fftm_autotune_4d_runtime.env";
    std::string expected_source;
    bool device_aware_mpi = fftm::device_backend::device_aware_mpi_enabled();
    bool expect_mismatch = false;
    std::vector<fftm::fftm_4d_spectral_layout> accepted_spectral_layouts{
        fftm::fftm_4d_spectral_layout::public_yzwx,
        fftm::fftm_4d_spectral_layout::native_xzwy
    };
};

bool parse_bool( const std::string &value, const char *name )
{
    if ( fftm::autotune::truthy( value ) )
        return true;
    if ( fftm::autotune::falsey( value ) )
        return false;
    throw std::logic_error( std::string( name ) + " must be a boolean value" );
}

runtime_options parse_options( int argc, char **argv )
{
    runtime_options result;
    for ( int index = 1; index < argc; ++index )
    {
        const std::string option = argv[index];
        const auto value = [&]( const char *name ) -> std::string {
            if ( index + 1 >= argc )
                throw std::logic_error( std::string( "Missing value for " ) + name );
            return argv[++index];
        };
        if ( option == "--size" )
            result.size = static_cast<std::size_t>( std::strtoull( value( "--size" ).c_str(), nullptr, 10 ) );
        else if ( option == "--cache" )
            result.cache_file = value( "--cache" );
        else if ( option == "--warmup" )
            result.warmup = std::atoi( value( "--warmup" ).c_str() );
        else if ( option == "--times" )
            result.iterations = std::atoi( value( "--times" ).c_str() );
        else if ( option == "--expect-source" )
            result.expected_source = value( "--expect-source" );
        else if ( option == "--device-aware-mpi" )
            result.device_aware_mpi = parse_bool( value( "--device-aware-mpi" ), "--device-aware-mpi" );
        else if ( option == "--accepted-layouts" )
        {
            result.accepted_spectral_layouts.clear();
            for ( const auto &token : fftm::autotune::split( value( "--accepted-layouts" ), ',' ) )
            {
                if ( !token.empty() )
                {
                    result.accepted_spectral_layouts.push_back(
                        fftm::autotune::parse_spectral_layout_4d( token )
                    );
                }
            }
        }
        else if ( option == "--expect-mismatch" )
            result.expect_mismatch = true;
        else if ( option == "--help" )
        {
            std::cout
                << "USAGE: test_fftm_autotune_4d_runtime.bin [options]\n"
                << "  --size N\n"
                << "  --cache FILE\n"
                << "  --warmup N\n"
                << "  --times N\n"
                << "  --expect-source measured|cache\n"
                << "  --device-aware-mpi 0|1\n"
                << "  --accepted-layouts public-yzwx,native-xzwy\n"
                << "  --expect-mismatch\n";
            std::exit( 0 );
        }
        else
            throw std::logic_error( "Unknown option '" + option + "'" );
    }
    if ( result.size < 8 )
        throw std::logic_error( "--size must be at least 8" );
    if ( result.warmup < 0 || result.iterations <= 0 )
        throw std::logic_error( "--warmup must be nonnegative and --times must be positive" );
    if ( result.cache_file.empty() )
        throw std::logic_error( "--cache must not be empty" );
    if ( result.accepted_spectral_layouts.empty() )
        throw std::logic_error( "--accepted-layouts must contain at least one layout" );
    return result;
}

bool wrap_mpi_processes_over_gpus_enabled()
{
    const char *value = std::getenv( "FFTM_WRAP_PROCS_GPUS" );
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int run( const runtime_options &options, const comm_t &comm, log_t &log )
{
    fftm::global_sizes sizes;
    sizes.init( options.size, options.size, options.size, options.size );

    fftm::autotune::autotune_options_4d autotune;
    autotune.cache_file = options.cache_file;
    autotune.create_if_missing = true;
    autotune.mismatch_policy = fftm::autotune::cache_mismatch_policy::error;
    autotune.validate_hardware = true;
    autotune.strict_device_identity = true;
    autotune.requested_spectral_layout = fftm::fftm_4d_spectral_layout::native_xzwy;
    autotune.accepted_spectral_layouts = options.accepted_spectral_layouts;
    autotune.device_aware_mpi = options.device_aware_mpi;

    fftm::autotune::measured_4d_options measured;
    measured.warmup = options.warmup;
    measured.iterations = options.iterations;

    fftm::autotune::fftm_4d_candidate_evaluator<
        fft_backend_t, comm_t, backend_t, log_t
    > evaluator( comm, sizes, log );

    try
    {
        const auto selected = fftm::autotune::load_or_measure_4d_config<runtime_api_t>(
            comm, sizes, autotune, measured, evaluator
        );
        if ( options.expect_mismatch )
            throw std::logic_error( "Expected the 4D cache validation to reject this run" );
        if ( !options.expected_source.empty() && selected.source != options.expected_source )
        {
            throw std::logic_error(
                "Expected source='" + options.expected_source + "' but selected source='" +
                selected.source + "'"
            );
        }

        fftm::autotune::measured_4d_options validation = measured;
        validation.warmup = 0;
        validation.iterations = 1;
        const auto check = evaluator( selected, validation );
        if ( !check.valid )
            throw std::logic_error( "Selected 4D configuration failed runtime validation" );

        if ( comm.myid == 0 )
        {
            std::cout
                << "FFTM_4D_AUTOTUNE_RESULT"
                << " source=" << selected.source
                << " strategy=" << selected.strategy_4d
                << " mode=" << selected.mode
                << " grid=" << fftm::autotune::grid_to_string_4d( selected.grid )
                << " layout=" << fftm::autotune::value_or_empty(
                       selected.config, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D"
                   )
                << " selected_check_ms=" << check.median_wall_ms
                << " cache=" << options.cache_file
                << '\n';
        }
        return 0;
    }
    catch ( const std::exception &error )
    {
        if ( options.expect_mismatch )
        {
            if ( comm.myid == 0 )
                std::cout << "FFTM_4D_AUTOTUNE_MISMATCH_REJECTED error=" << error.what() << '\n';
            return 0;
        }
        throw;
    }
}

} // namespace

int main( int argc, char **argv )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    const comm_t comm = mpi.comm_world();
    log_t log;
    try
    {
        fftm::device_backend::init_mpi(
            log, comm, 0, wrap_mpi_processes_over_gpus_enabled()
        );
        return run( parse_options( argc, argv ), comm, log );
    }
    catch ( const std::exception &error )
    {
        log.error( scfd::utils::nested_exception_to_multistring( error ) );
        return 1;
    }
}
