#ifndef FFTM_EXAMPLES_TURBULENCE_SPACETIME_OPTIONS_H
#define FFTM_EXAMPLES_TURBULENCE_SPACETIME_OPTIONS_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace fftm
{
namespace examples
{
namespace turbulence
{

struct spacetime_options
{
    std::size_t spatial_size   = 512;
    std::size_t frames         = 100;
    std::size_t frame_offset   = 0;
    std::size_t mode_min       = 1;
    std::size_t mode_max       = 4;
    std::size_t spatial_cutoff = 0;
    std::size_t write_every    = 10;
    std::string field          = "omega_z";
    std::string input_dir      = "taylor_green_output";
    std::string output_dir     = "taylor_green_4d_filtered";
    bool        subtract_mean  = true;
    bool        hann_window    = true;
};

inline std::size_t parse_spacetime_size( const char *name, const std::string &value )
{
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull( value.c_str(), &end, 10 );
    if ( end == value.c_str() || *end != '\0' || parsed > std::numeric_limits<std::size_t>::max() )
        throw std::logic_error( std::string( "Invalid value for " ) + name + ": '" + value + "'" );
    return static_cast<std::size_t>( parsed );
}

inline bool parse_spacetime_bool( const char *name, const std::string &value )
{
    if ( value == "1" || value == "true" || value == "on" || value == "yes" )
        return true;
    if ( value == "0" || value == "false" || value == "off" || value == "no" )
        return false;
    throw std::logic_error( std::string( "Invalid boolean value for " ) + name + ": '" + value + "'" );
}

inline std::string require_spacetime_value( int argc, char **argv, int &index )
{
    if ( index + 1 >= argc )
        throw std::logic_error( std::string( "Missing value after " ) + argv[index] );
    return argv[++index];
}

inline spacetime_options parse_spacetime_options( int argc, char **argv )
{
    spacetime_options options;
    for ( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if ( arg == "--size" )
            options.spatial_size = parse_spacetime_size( "--size", require_spacetime_value( argc, argv, i ) );
        else if ( arg == "--frames" )
            options.frames = parse_spacetime_size( "--frames", require_spacetime_value( argc, argv, i ) );
        else if ( arg == "--frame-offset" )
        {
            options.frame_offset =
                parse_spacetime_size( "--frame-offset", require_spacetime_value( argc, argv, i ) );
        }
        else if ( arg == "--mode-min" )
            options.mode_min = parse_spacetime_size( "--mode-min", require_spacetime_value( argc, argv, i ) );
        else if ( arg == "--mode-max" )
            options.mode_max = parse_spacetime_size( "--mode-max", require_spacetime_value( argc, argv, i ) );
        else if ( arg == "--spatial-cutoff" )
        {
            options.spatial_cutoff =
                parse_spacetime_size( "--spatial-cutoff", require_spacetime_value( argc, argv, i ) );
        }
        else if ( arg == "--write-every" )
            options.write_every = parse_spacetime_size( "--write-every", require_spacetime_value( argc, argv, i ) );
        else if ( arg == "--field" )
            options.field = require_spacetime_value( argc, argv, i );
        else if ( arg == "--input" )
            options.input_dir = require_spacetime_value( argc, argv, i );
        else if ( arg == "--output" )
            options.output_dir = require_spacetime_value( argc, argv, i );
        else if ( arg == "--subtract-mean" )
        {
            options.subtract_mean =
                parse_spacetime_bool( "--subtract-mean", require_spacetime_value( argc, argv, i ) );
        }
        else if ( arg == "--hann-window" )
        {
            options.hann_window =
                parse_spacetime_bool( "--hann-window", require_spacetime_value( argc, argv, i ) );
        }
        else if ( arg == "--help" || arg == "-h" )
        {
            throw std::logic_error(
                "Usage: taylor_green_spacetime_4d.bin [options]\n"
                "  --input DIR --output DIR --field omega_z\n"
                "  --size N --frames N [--frame-offset N]\n"
                "  --mode-min M --mode-max M [--spatial-cutoff K]\n"
                "  --write-every N --subtract-mean 0|1 --hann-window 0|1"
            );
        }
        else
            throw std::logic_error( "Unknown option: '" + arg + "'" );
    }

    if ( options.spatial_size == 0 )
        throw std::logic_error( "Spatial size must be positive" );
    if ( options.frames < 4 || options.frames % 2 != 0 )
        throw std::logic_error( "The number of 4D frames must be even and at least four" );
    if ( options.mode_min > options.mode_max || options.mode_max > options.frames / 2 )
        throw std::logic_error( "Temporal mode range must satisfy 0 <= min <= max <= frames/2" );
    if ( options.spatial_cutoff > options.spatial_size / 2 )
        throw std::logic_error( "Spatial cutoff cannot exceed size/2" );
    if ( options.write_every == 0 )
        throw std::logic_error( "write-every must be positive" );
    if ( options.field.empty() || options.input_dir.empty() || options.output_dir.empty() )
        throw std::logic_error( "Input, output, and field names must not be empty" );
    return options;
}

} // namespace turbulence
} // namespace examples
} // namespace fftm

#endif
