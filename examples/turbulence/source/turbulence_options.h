#ifndef FFTM_EXAMPLES_TURBULENCE_OPTIONS_H
#define FFTM_EXAMPLES_TURBULENCE_OPTIONS_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fftm
{
namespace examples
{
namespace turbulence
{

struct turbulence_options
{
    std::size_t nx = 32;
    std::size_t ny = 32;
    std::size_t nz = 32;

    double reynolds     = 1600.0;
    double dt           = 1.0e-3;
    double dt_min       = 0.0;
    double dt_max       = 0.0;
    double dt_growth    = 1.1;
    double target_cfl   = 0.0;
    double final_time   = 1.0e-2;
    std::size_t steps   = 0;
    double cfl_fail     = 1.0;

    std::size_t diagnostics_every = 1;
    std::size_t snapshot_every    = 0;
    std::size_t viz_every         = 0;
    double      snapshot_period   = 0.0;
    double      viz_period        = 0.0;
    std::size_t snapshot_size     = 0;
    std::size_t viz_snapshot_size = 0;

    std::string cache_file = "fftm_taylor_green_autotune.env";
    std::string output_dir = "taylor_green_output";
    bool        write_initial_snapshot = false;
    bool        verify_initial_energy  = true;
};

inline std::size_t parse_size_value( const char *name, const std::string &value )
{
    char              *end = nullptr;
    const unsigned long long parsed = std::strtoull( value.c_str(), &end, 10 );
    if ( end == value.c_str() || *end != '\0' || parsed > std::numeric_limits<std::size_t>::max() )
        throw std::logic_error( std::string( "Invalid value for " ) + name + ": '" + value + "'" );
    return static_cast<std::size_t>( parsed );
}

inline double parse_real_value( const char *name, const std::string &value )
{
    char        *end = nullptr;
    const double parsed = std::strtod( value.c_str(), &end );
    if ( end == value.c_str() || *end != '\0' )
        throw std::logic_error( std::string( "Invalid value for " ) + name + ": '" + value + "'" );
    return parsed;
}

inline bool parse_bool_value( const char *name, const std::string &value )
{
    if ( value == "1" || value == "true" || value == "on" || value == "yes" )
        return true;
    if ( value == "0" || value == "false" || value == "off" || value == "no" )
        return false;
    throw std::logic_error( std::string( "Invalid boolean value for " ) + name + ": '" + value + "'" );
}

inline std::string require_option_value( int argc, char **argv, int &index )
{
    if ( index + 1 >= argc )
        throw std::logic_error( std::string( "Missing value after " ) + argv[index] );
    return argv[++index];
}

inline turbulence_options parse_options( int argc, char **argv )
{
    turbulence_options options;

    for ( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if ( arg == "--size" )
        {
            const std::size_t n = parse_size_value( "--size", require_option_value( argc, argv, i ) );
            options.nx = options.ny = options.nz = n;
        }
        else if ( arg == "--nx" )
            options.nx = parse_size_value( "--nx", require_option_value( argc, argv, i ) );
        else if ( arg == "--ny" )
            options.ny = parse_size_value( "--ny", require_option_value( argc, argv, i ) );
        else if ( arg == "--nz" )
            options.nz = parse_size_value( "--nz", require_option_value( argc, argv, i ) );
        else if ( arg == "--reynolds" )
            options.reynolds = parse_real_value( "--reynolds", require_option_value( argc, argv, i ) );
        else if ( arg == "--dt" )
            options.dt = parse_real_value( "--dt", require_option_value( argc, argv, i ) );
        else if ( arg == "--dt-min" )
            options.dt_min = parse_real_value( "--dt-min", require_option_value( argc, argv, i ) );
        else if ( arg == "--dt-max" )
            options.dt_max = parse_real_value( "--dt-max", require_option_value( argc, argv, i ) );
        else if ( arg == "--dt-growth" )
            options.dt_growth = parse_real_value( "--dt-growth", require_option_value( argc, argv, i ) );
        else if ( arg == "--target-cfl" || arg == "--cfl-max" )
            options.target_cfl = parse_real_value( "--target-cfl", require_option_value( argc, argv, i ) );
        else if ( arg == "--final-time" )
            options.final_time = parse_real_value( "--final-time", require_option_value( argc, argv, i ) );
        else if ( arg == "--steps" )
            options.steps = parse_size_value( "--steps", require_option_value( argc, argv, i ) );
        else if ( arg == "--cfl-fail" )
            options.cfl_fail = parse_real_value( "--cfl-fail", require_option_value( argc, argv, i ) );
        else if ( arg == "--diagnostics-every" )
        {
            options.diagnostics_every =
                parse_size_value( "--diagnostics-every", require_option_value( argc, argv, i ) );
        }
        else if ( arg == "--snapshot-every" )
            options.snapshot_every = parse_size_value( "--snapshot-every", require_option_value( argc, argv, i ) );
        else if ( arg == "--viz-every" )
            options.viz_every = parse_size_value( "--viz-every", require_option_value( argc, argv, i ) );
        else if ( arg == "--snapshot-period" )
        {
            options.snapshot_period =
                parse_real_value( "--snapshot-period", require_option_value( argc, argv, i ) );
        }
        else if ( arg == "--viz-period" )
            options.viz_period = parse_real_value( "--viz-period", require_option_value( argc, argv, i ) );
        else if ( arg == "--snapshot-size" )
            options.snapshot_size = parse_size_value( "--snapshot-size", require_option_value( argc, argv, i ) );
        else if ( arg == "--viz-snapshot-size" )
        {
            options.viz_snapshot_size =
                parse_size_value( "--viz-snapshot-size", require_option_value( argc, argv, i ) );
        }
        else if ( arg == "--cache" )
            options.cache_file = require_option_value( argc, argv, i );
        else if ( arg == "--output" )
            options.output_dir = require_option_value( argc, argv, i );
        else if ( arg == "--write-initial-snapshot" )
        {
            options.write_initial_snapshot =
                parse_bool_value( "--write-initial-snapshot", require_option_value( argc, argv, i ) );
        }
        else if ( arg == "--verify-initial-energy" )
        {
            options.verify_initial_energy =
                parse_bool_value( "--verify-initial-energy", require_option_value( argc, argv, i ) );
        }
        else if ( arg == "--help" || arg == "-h" )
        {
            throw std::logic_error(
                "Usage: taylor_green_3d_autotuned.bin [options]\n"
                "  --size N | --nx NX --ny NY --nz NZ\n"
                "  --reynolds RE --dt DT --final-time T [--steps N]\n"
                "  --target-cfl C [--dt-min DT --dt-max DT --dt-growth R]\n"
                "  --cache FILE --output DIR\n"
                "  --diagnostics-every N --snapshot-every N --viz-every N\n"
                "  --snapshot-period T --viz-period T\n"
                "  --snapshot-size N [--viz-snapshot-size N] --cfl-fail C\n"
                "  --write-initial-snapshot 0|1 --verify-initial-energy 0|1"
            );
        }
        else
            throw std::logic_error( "Unknown option: '" + arg + "'" );
    }

    if ( options.nx == 0 || options.ny == 0 || options.nz == 0 )
        throw std::logic_error( "All grid dimensions must be positive" );
    if ( options.nx % 2 != 0 || options.ny % 2 != 0 || options.nz % 2 != 0 )
        throw std::logic_error( "Taylor-Green pseudospectral dimensions must be even" );
    if ( !std::isfinite( options.reynolds ) || options.reynolds <= 0.0 )
        throw std::logic_error( "Reynolds number must be positive" );
    if ( !std::isfinite( options.dt ) || options.dt <= 0.0 )
        throw std::logic_error( "Time step must be positive" );
    if ( !std::isfinite( options.dt_min ) || options.dt_min < 0.0 )
        throw std::logic_error( "Minimum time step must be finite and nonnegative" );
    if ( !std::isfinite( options.dt_max ) || options.dt_max < 0.0 )
        throw std::logic_error( "Maximum time step must be finite and nonnegative" );
    if ( options.dt_max > 0.0 && options.dt_min > options.dt_max )
        throw std::logic_error( "--dt-min cannot exceed --dt-max" );
    if ( !std::isfinite( options.dt_growth ) || options.dt_growth < 1.0 )
        throw std::logic_error( "Time-step growth factor must be finite and at least one" );
    if ( !std::isfinite( options.target_cfl ) || options.target_cfl < 0.0 )
        throw std::logic_error( "Target CFL must be finite and nonnegative" );
    if ( !std::isfinite( options.final_time ) || options.final_time < 0.0 )
        throw std::logic_error( "Final time must be nonnegative" );
    if ( options.steps == 0 && options.final_time == 0.0 )
        throw std::logic_error( "Either --steps or a positive --final-time is required" );
    if ( options.target_cfl > 0.0 && options.steps != 0 )
        throw std::logic_error( "Adaptive CFL stepping requires --steps 0 and a positive --final-time" );
    if ( !std::isfinite( options.cfl_fail ) || options.cfl_fail <= 0.0 )
        throw std::logic_error( "CFL failure threshold must be positive" );
    if ( options.target_cfl > options.cfl_fail )
        throw std::logic_error( "--target-cfl cannot exceed --cfl-fail" );
    if ( !std::isfinite( options.snapshot_period ) || options.snapshot_period < 0.0 )
        throw std::logic_error( "Snapshot period must be finite and nonnegative" );
    if ( !std::isfinite( options.viz_period ) || options.viz_period < 0.0 )
        throw std::logic_error( "Visualization period must be finite and nonnegative" );
    if ( options.snapshot_every != 0 && options.snapshot_period > 0.0 )
        throw std::logic_error( "Use either --snapshot-every or --snapshot-period, not both" );
    if ( options.viz_every != 0 && options.viz_period > 0.0 )
        throw std::logic_error( "Use either --viz-every or --viz-period, not both" );
    if ( options.snapshot_size != 0 )
    {
        if ( options.snapshot_size > options.nx || options.snapshot_size > options.ny ||
             options.snapshot_size > options.nz )
        {
            throw std::logic_error( "Snapshot size cannot exceed a simulation dimension" );
        }
        if ( options.nx % options.snapshot_size != 0 || options.ny % options.snapshot_size != 0 ||
             options.nz % options.snapshot_size != 0 )
        {
            throw std::logic_error(
                "Snapshot size must divide every simulation dimension exactly; use 512 for a 2048^3 run"
            );
        }
    }
    if ( options.viz_snapshot_size == 0 )
        options.viz_snapshot_size = options.snapshot_size;
    if ( options.viz_snapshot_size != 0 )
    {
        if ( options.viz_snapshot_size > options.nx || options.viz_snapshot_size > options.ny ||
             options.viz_snapshot_size > options.nz )
        {
            throw std::logic_error( "Visualization snapshot size cannot exceed a simulation dimension" );
        }
        if ( options.nx % options.viz_snapshot_size != 0 || options.ny % options.viz_snapshot_size != 0 ||
             options.nz % options.viz_snapshot_size != 0 )
        {
            throw std::logic_error( "Visualization snapshot size must divide every simulation dimension exactly" );
        }
    }
    if ( ( options.snapshot_every != 0 || options.viz_every != 0 || options.snapshot_period > 0.0 ||
           options.viz_period > 0.0 ) &&
         options.snapshot_size == 0 )
    {
        throw std::logic_error( "--snapshot-size is required when snapshot output is enabled" );
    }
    if ( options.cache_file.empty() )
        throw std::logic_error( "Autotune cache filename must not be empty" );
    if ( options.output_dir.empty() )
        throw std::logic_error( "Output directory must not be empty" );

    return options;
}

inline std::size_t resolved_step_count( const turbulence_options &options )
{
    if ( options.steps != 0 )
        return options.steps;
    if ( options.target_cfl > 0.0 || options.snapshot_period > 0.0 || options.viz_period > 0.0 )
        return 0;
    return static_cast<std::size_t>( std::ceil( options.final_time / options.dt ) );
}

} // namespace turbulence
} // namespace examples
} // namespace fftm

#endif
