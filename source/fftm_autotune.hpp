#ifndef __FFTM_AUTOTUNE_HPP__
#define __FFTM_AUTOTUNE_HPP__

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fftm.hpp"

namespace fftm
{
namespace autotune
{

using config_map = std::map<std::string, std::string>;

inline std::string trim( const std::string &value )
{
    const std::string whitespace = " \t\r\n";
    const std::size_t first      = value.find_first_not_of( whitespace );
    if ( first == std::string::npos )
        return "";
    const std::size_t last = value.find_last_not_of( whitespace );
    return value.substr( first, last - first + 1 );
}

inline std::vector<std::string> split( const std::string &value, char delimiter )
{
    std::vector<std::string> result;
    std::stringstream        ss( value );
    std::string              item;
    while ( std::getline( ss, item, delimiter ) )
        result.push_back( trim( item ) );
    return result;
}

inline bool truthy( const std::string &value )
{
    const std::string v = trim( value );
    return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "YES" || v == "on" || v == "ON";
}

inline bool falsey( const std::string &value )
{
    const std::string v = trim( value );
    return v == "0" || v == "false" || v == "FALSE" || v == "no" || v == "NO" || v == "off" || v == "OFF";
}

inline bool configured_sentinel( const std::string &value )
{
    return trim( value ) == "configured";
}

inline const char *getenv_or_null( const char *name )
{
    const char *value = std::getenv( name );
    return value != nullptr && *value != '\0' ? value : nullptr;
}

inline config_map load_key_value_file( const std::string &path )
{
    std::ifstream input( path.c_str() );
    if ( !input )
        throw std::logic_error( "FFTM autotune config cannot be opened: " + path );

    config_map   result;
    std::string  line;
    std::size_t  line_no = 0;
    while ( std::getline( input, line ) )
    {
        ++line_no;
        line = trim( line );
        if ( line.empty() || line[0] == '#' )
            continue;
        const std::size_t eq = line.find( '=' );
        if ( eq == std::string::npos )
            throw std::logic_error(
                "FFTM autotune config " + path + ":" + std::to_string( line_no ) + " is not KEY=VALUE"
            );
        result[trim( line.substr( 0, eq ) )] = trim( line.substr( eq + 1 ) );
    }
    if ( result.empty() )
        throw std::logic_error( "FFTM autotune config is empty: " + path );
    return result;
}

inline void overlay_environment( config_map &config )
{
    static const char *keys[] = {
        "FFTM_AUTOTUNE_GRID_3D",
        "FFTM_AUTOTUNE_NUM_GPUS",
        "FFTM_AUTOTUNE_SIZE_3D",
        "FFTM_AUTOTUNE_STRATEGY_3D",
        "FFTM_AUTOTUNE_MODE",
        "FFTM_AUTOTUNE_BACKEND_3D",
        "FFTM_3D_BACKEND",
        "FFTM_3D_BACKENDS",
        "FFTM_AUTOTUNE_PENCIL_LAYOUT",
        "FFTM_AUTOTUNE_PENCIL_PIPELINE",
        "FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS",
        "FFTM_PENCIL_LAYOUTS",
        "FFTM_PENCIL_PIPELINES",
        "FFTM_LARGE_COUNT_P2P_TRANSPORTS",
        "FFTM_USE_DIRECT_BACKWARD_RECEIVE",
        "FFTM_DIRECT_P2P_CUDA_AWARE",
        "FFTM_USE_P2P_SEND_THREAD",
        "FFTM_USE_P2P_BYTE_TRANSFER",
        "FFTM_USE_PERSISTENT_P2P",
        "FFTM_USE_READY_P2P_SEND",
        "FFTM_USE_DIRECT_FORWARD_BYTE_RECEIVE",
        "FFTM_USE_STABLE_FORWARD_BYTE_SEND_BUFFER",
        "FFTM_USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER",
        "FFTM_USE_CONTIGUOUS_FORWARD_BYTE_SEND",
        "FFTM_USE_PHYSICAL_FORWARD_PEER_EXCHANGE",
        "FFTM_CONTIGUOUS_FORWARD_SEND_MODE",
        "FFTM_CONTIGUOUS_FORWARD_SEND_MODES",
        "FFTM_CONTIGUOUS_FORWARD_SEND_CHUNK_MIB",
        "FFTM_USE_LARGE_COUNT_DATATYPE_CACHE",
        "FFTM_USE_FFT_EXEC_NO_SYNC",
        "FFTM_ALLOW_FFT_EXEC_NO_SYNC",
        "FFTM_USE_NATIVE_BACKWARD_SECOND_PEER_LOOP",
        "FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT",
        "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY",
        "FFTM_USE_NATIVE_OPT0_EGGER_Y_BUFFER_TOPOLOGY",
        "FFTM_USE_NATIVE_OPT0_COMPACT_Y_WORKAREA",
        "FFTM_USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE",
        "FFTM_USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES",
        "FFTM_USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC",
        "FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC",
        "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR",
        "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_LIFECYCLE",
        "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_BUNDLE",
        "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE",
        "FFTM_USE_NATIVE_OPT0_Y_PLAN_BUNDLE_STREAM_FIRST",
        "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_REFERENCE_STREAMS",
        "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_EGGER_STREAMS",
        "FFTM_USE_NATIVE_OPT0_REFERENCE_LOCAL_PLAN_CONTEXT",
        "FFTM_USE_NATIVE_OPT0_EGGER_LOCAL_PLAN_CONTEXT",
        "FFTM_USE_NATIVE_OPT0_MEMORY_FEASIBILITY_GUARD",
        "FFTM_NATIVE_OPT0_MEMORY_FEASIBILITY_RESERVE_MIB",
        "FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS",
        "FFTM_AUTOTUNE_DIAGNOSTIC_ONLY",
        "FFTM3D_GRIDS",
        "FFTM3D_OPTS",
    };
    for ( const char *key : keys )
    {
        if ( const char *value = getenv_or_null( key ) )
        {
            if ( !configured_sentinel( value ) )
                config[key] = value;
        }
    }
}

inline std::string value_or_empty( const config_map &config, const std::string &key )
{
    const auto it = config.find( key );
    return it == config.end() ? std::string() : it->second;
}

inline bool bool_value( const config_map &config, const std::string &key, bool fallback )
{
    const std::string value = value_or_empty( config, key );
    if ( value.empty() )
        return fallback;
    if ( truthy( value ) )
        return true;
    if ( falsey( value ) )
        return false;
    throw std::logic_error( "FFTM autotune boolean key " + key + " has invalid value '" + value + "'" );
}

inline std::size_t size_value( const config_map &config, const std::string &key, std::size_t fallback )
{
    const std::string value = value_or_empty( config, key );
    if ( value.empty() )
        return fallback;
    return static_cast<std::size_t>( std::strtoull( value.c_str(), nullptr, 10 ) );
}

inline bool parse_grid_token( const std::string &token, std::size_t &p1, std::size_t &p2 )
{
    const std::size_t x = token.find( 'x' );
    if ( x == std::string::npos )
        return false;
    p1 = static_cast<std::size_t>( std::strtoull( token.substr( 0, x ).c_str(), nullptr, 10 ) );
    p2 = static_cast<std::size_t>( std::strtoull( token.substr( x + 1 ).c_str(), nullptr, 10 ) );
    return p1 != 0 && p2 != 0;
}

inline bool parse_fftm3d_grid_list( const std::string &value, int num_procs, std::size_t &p1, std::size_t &p2 )
{
    for ( const std::string &item : split( value, ',' ) )
    {
        if ( item.empty() )
            continue;
        const std::size_t colon = item.find( ':' );
        if ( colon != std::string::npos )
        {
            const int count = std::atoi( item.substr( 0, colon ).c_str() );
            if ( count != num_procs )
                continue;
            return parse_grid_token( item.substr( colon + 1 ), p1, p2 );
        }
        if ( parse_grid_token( item, p1, p2 ) )
            return true;
    }
    return false;
}

inline std::string first_csv_value( const std::string &value )
{
    const std::vector<std::string> values = split( value, ',' );
    return values.empty() ? std::string() : values.front();
}

inline std::pair<std::size_t, std::size_t> default_pencil_grid_3d( int num_procs )
{
    std::size_t p1 = 1;
    for ( std::size_t d = 1; d * d <= static_cast<std::size_t>( num_procs ); ++d )
    {
        if ( static_cast<std::size_t>( num_procs ) % d == 0 )
            p1 = d;
    }
    return std::make_pair( p1, static_cast<std::size_t>( num_procs ) / p1 );
}

inline bool select_grid( const config_map &config, int num_procs, processor_grid &grid )
{
    std::size_t p1 = 0;
    std::size_t p2 = 0;

    const std::string direct_grid = value_or_empty( config, "FFTM_AUTOTUNE_GRID_3D" );
    if ( !direct_grid.empty() )
    {
        if ( !parse_grid_token( direct_grid, p1, p2 ) )
            throw std::logic_error( "Invalid FFTM_AUTOTUNE_GRID_3D='" + direct_grid + "'" );
        grid.init( p1, p2 );
        return true;
    }

    const std::string fftm3d_grids = value_or_empty( config, "FFTM3D_GRIDS" );
    if ( !fftm3d_grids.empty() && parse_fftm3d_grid_list( fftm3d_grids, num_procs, p1, p2 ) )
    {
        grid.init( p1, p2 );
        return true;
    }

    const std::string orientation = value_or_empty( config, "FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS" );
    if ( !orientation.empty() && orientation != "both" && orientation != "all" )
    {
        const auto def = default_pencil_grid_3d( num_procs );
        if ( orientation == "default" )
        {
            grid.init( def.first, def.second );
            return true;
        }
        if ( orientation == "reversed" )
        {
            grid.init( def.second, def.first );
            return true;
        }
        if ( orientation == "production" && num_procs == 8 )
        {
            const std::string layout =
                !value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_LAYOUT" ).empty() ?
                    value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_LAYOUT" ) :
                    value_or_empty( config, "FFTM_PENCIL_LAYOUTS" );
            if ( layout == "opt1" )
                grid.init( def.first, def.second );
            else
                grid.init( def.second, def.first );
            return true;
        }
    }
    return false;
}

inline fftm_3d_pencil_layout parse_pencil_layout( const std::string &value )
{
    if ( value == "auto" || value == "auto_select" || value.empty() )
        return fftm_3d_pencil_layout::auto_select;
    if ( value == "opt0" )
        return fftm_3d_pencil_layout::opt0;
    if ( value == "opt1" )
        return fftm_3d_pencil_layout::opt1;
    if ( value == "legacy" )
        return fftm_3d_pencil_layout::legacy;
    throw std::logic_error( "Invalid FFTM pencil layout '" + value + "'" );
}

inline fftm_3d_pencil_pipeline parse_pencil_pipeline( const std::string &value )
{
    if ( value == "staged" || value.empty() )
        return fftm_3d_pencil_pipeline::staged;
    if ( value == "fused" )
        return fftm_3d_pencil_pipeline::fused;
    if ( value == "reference" || value == "reference-compatible" || value == "native-compatible" ||
         value == "compatible" || value == "egger" )
        return fftm_3d_pencil_pipeline::reference;
    if ( value == "reference-parity" || value == "reference_parity" || value == "native-parity" ||
         value == "native_parity" || value == "compatible-parity" || value == "compatible_parity" ||
         value == "egger-parity" || value == "egger_parity" )
        return fftm_3d_pencil_pipeline::reference_parity;
    throw std::logic_error( "Invalid FFTM pencil pipeline '" + value + "'" );
}

inline fftm_3d_large_count_p2p_transport parse_large_count_transport( const std::string &value )
{
    if ( value == "hindexed" || value.empty() )
        return fftm_3d_large_count_p2p_transport::hindexed;
    if ( value == "mpi-count" || value == "mpi_count" )
        return fftm_3d_large_count_p2p_transport::mpi_count;
    if ( value == "element-count" || value == "element_count" )
        return fftm_3d_large_count_p2p_transport::element_count;
    if ( value == "chunked" )
        return fftm_3d_large_count_p2p_transport::chunked;
    throw std::logic_error( "Invalid FFTM large-count transport '" + value + "'" );
}

inline fftm_3d_contiguous_forward_send_mode parse_contiguous_forward_send_mode( const std::string &value )
{
    if ( value == "single" || value.empty() )
        return fftm_3d_contiguous_forward_send_mode::single;
    if ( value == "chunked" )
        return fftm_3d_contiguous_forward_send_mode::chunked;
    throw std::logic_error( "Invalid FFTM contiguous forward send mode '" + value + "'" );
}

inline void validate_match( const config_map &config, int num_procs, const global_sizes &sizes )
{
    const std::string expected_gpus = value_or_empty( config, "FFTM_AUTOTUNE_NUM_GPUS" );
    if ( !expected_gpus.empty() && std::atoi( expected_gpus.c_str() ) != num_procs )
    {
        throw std::logic_error(
            "FFTM autotune config GPU count " + expected_gpus + " does not match MPI size " +
            std::to_string( num_procs )
        );
    }

    const std::string expected_size = value_or_empty( config, "FFTM_AUTOTUNE_SIZE_3D" );
    if ( !expected_size.empty() )
    {
        std::size_t nx = 0, ny = 0, nz = 0;
        const std::vector<std::string> dims = split( expected_size, 'x' );
        if ( dims.size() != 3 )
            throw std::logic_error( "Invalid FFTM_AUTOTUNE_SIZE_3D='" + expected_size + "'" );
        nx = static_cast<std::size_t>( std::strtoull( dims[0].c_str(), nullptr, 10 ) );
        ny = static_cast<std::size_t>( std::strtoull( dims[1].c_str(), nullptr, 10 ) );
        nz = static_cast<std::size_t>( std::strtoull( dims[2].c_str(), nullptr, 10 ) );
        if ( nx != sizes.Nx || ny != sizes.Ny || nz != sizes.Nz )
        {
            throw std::logic_error(
                "FFTM autotune config size " + expected_size + " does not match requested size " +
                std::to_string( sizes.Nx ) + "x" + std::to_string( sizes.Ny ) + "x" +
                std::to_string( sizes.Nz )
            );
        }
    }
}

inline bool apply_config_3d(
    const config_map &config, int num_procs, const global_sizes &sizes, processor_grid &grid,
    fftm_init_options &options
)
{
    if ( config.empty() )
        return false;
    validate_match( config, num_procs, sizes );

    bool applied = select_grid( config, num_procs, grid );

    const std::string layout =
        !value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_LAYOUT" ).empty() ?
            value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_LAYOUT" ) :
            value_or_empty( config, "FFTM_PENCIL_LAYOUTS" );
    if ( !layout.empty() && !configured_sentinel( layout ) )
    {
        options.pencil_layout_3d = parse_pencil_layout( first_csv_value( layout ) );
        applied                  = true;
    }

    const std::string pipeline =
        !value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_PIPELINE" ).empty() ?
            value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_PIPELINE" ) :
            value_or_empty( config, "FFTM_PENCIL_PIPELINES" );
    if ( !pipeline.empty() && !configured_sentinel( pipeline ) )
    {
        options.pencil_pipeline_3d = parse_pencil_pipeline( first_csv_value( pipeline ) );
        applied                    = true;
    }

    const std::string large_count = value_or_empty( config, "FFTM_LARGE_COUNT_P2P_TRANSPORTS" );
    if ( !large_count.empty() && !configured_sentinel( large_count ) )
    {
        options.large_count_p2p_transport = parse_large_count_transport( split( large_count, ',' ).front() );
        applied                           = true;
    }

    const bool allow_fft_exec_no_sync = bool_value( config, "FFTM_ALLOW_FFT_EXEC_NO_SYNC", false );
    options.allow_native_opt0_diagnostic_variants = bool_value(
        config, "FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS", options.allow_native_opt0_diagnostic_variants
    );
    if ( allow_fft_exec_no_sync )
        options.allow_native_opt0_diagnostic_variants = true;
    if ( bool_value( config, "FFTM_AUTOTUNE_DIAGNOSTIC_ONLY", false ) &&
         !options.allow_native_opt0_diagnostic_variants )
    {
        throw std::logic_error(
            "FFTM autotune config is marked FFTM_AUTOTUNE_DIAGNOSTIC_ONLY=1. "
            "Set FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS=1 only for diagnostic runs."
        );
    }

    const std::string contiguous_mode =
        !value_or_empty( config, "FFTM_CONTIGUOUS_FORWARD_SEND_MODE" ).empty() ?
            value_or_empty( config, "FFTM_CONTIGUOUS_FORWARD_SEND_MODE" ) :
            value_or_empty( config, "FFTM_CONTIGUOUS_FORWARD_SEND_MODES" );
    if ( !contiguous_mode.empty() && !configured_sentinel( contiguous_mode ) )
    {
        options.contiguous_forward_send_mode = parse_contiguous_forward_send_mode( first_csv_value( contiguous_mode ) );
        applied                              = true;
    }

    options.use_direct_backward_receive = bool_value(
        config, "FFTM_USE_DIRECT_BACKWARD_RECEIVE", options.use_direct_backward_receive
    );
    options.direct_p2p_cuda_aware = bool_value( config, "FFTM_DIRECT_P2P_CUDA_AWARE", options.direct_p2p_cuda_aware );
    options.use_p2p_send_thread = bool_value( config, "FFTM_USE_P2P_SEND_THREAD", options.use_p2p_send_thread );
    options.use_p2p_byte_transfer = bool_value( config, "FFTM_USE_P2P_BYTE_TRANSFER", options.use_p2p_byte_transfer );
    options.use_persistent_p2p = bool_value( config, "FFTM_USE_PERSISTENT_P2P", options.use_persistent_p2p );
    options.use_ready_p2p_send = bool_value( config, "FFTM_USE_READY_P2P_SEND", options.use_ready_p2p_send );
    options.use_direct_forward_byte_receive = bool_value(
        config, "FFTM_USE_DIRECT_FORWARD_BYTE_RECEIVE", options.use_direct_forward_byte_receive
    );
    options.use_stable_forward_byte_send_buffer = bool_value(
        config, "FFTM_USE_STABLE_FORWARD_BYTE_SEND_BUFFER", options.use_stable_forward_byte_send_buffer
    );
    options.use_ready_stable_forward_byte_send_buffer = bool_value(
        config, "FFTM_USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER",
        options.use_ready_stable_forward_byte_send_buffer
    );
    options.use_contiguous_forward_byte_send = bool_value(
        config, "FFTM_USE_CONTIGUOUS_FORWARD_BYTE_SEND", options.use_contiguous_forward_byte_send
    );
    options.use_physical_forward_peer_exchange = bool_value(
        config, "FFTM_USE_PHYSICAL_FORWARD_PEER_EXCHANGE", options.use_physical_forward_peer_exchange
    );
    options.use_large_count_datatype_cache = bool_value(
        config, "FFTM_USE_LARGE_COUNT_DATATYPE_CACHE", options.use_large_count_datatype_cache
    );
    options.use_fft_exec_no_sync = bool_value( config, "FFTM_USE_FFT_EXEC_NO_SYNC", options.use_fft_exec_no_sync );
    if ( options.use_fft_exec_no_sync && !allow_fft_exec_no_sync )
    {
        throw std::logic_error(
            "FFTM_USE_FFT_EXEC_NO_SYNC=1 is diagnostic-only. It removes generic cuFFT execution synchronization "
            "and can race later MPI/copy stages. Use the native opt0 Y no-sync option instead, or set "
            "FFTM_ALLOW_FFT_EXEC_NO_SYNC=1 only for explicit diagnostics."
        );
    }
    options.use_native_backward_second_peer_loop = bool_value(
        config, "FFTM_USE_NATIVE_BACKWARD_SECOND_PEER_LOOP", options.use_native_backward_second_peer_loop
    );
    options.use_native_opt0_default_z_layout = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT", options.use_native_opt0_default_z_layout
    );
    options.use_native_opt0_reference_y_buffer_topology = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY",
        options.use_native_opt0_reference_y_buffer_topology
    );
    options.use_native_opt0_reference_y_buffer_topology = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_EGGER_Y_BUFFER_TOPOLOGY",
        options.use_native_opt0_reference_y_buffer_topology
    );
    options.use_native_opt0_compact_y_workarea = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_COMPACT_Y_WORKAREA",
        options.use_native_opt0_compact_y_workarea
    );
    options.use_native_opt0_tight_y_plan_sequence = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE", options.use_native_opt0_tight_y_plan_sequence
    );
    options.use_native_opt0_shared_y_plan_handles = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES", options.use_native_opt0_shared_y_plan_handles
    );
    options.use_native_opt0_y_group_device_sync = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC", options.use_native_opt0_y_group_device_sync
    );
    options.use_native_opt0_y_no_sync_exec = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC", options.use_native_opt0_y_no_sync_exec
    );
    options.use_native_opt0_raw_y_plan_array_executor = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR",
        options.use_native_opt0_raw_y_plan_array_executor
    );
    options.use_native_opt0_reference_y_plan_lifecycle = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_LIFECYCLE",
        options.use_native_opt0_reference_y_plan_lifecycle
    );
    options.use_native_opt0_reference_y_plan_bundle = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_BUNDLE", options.use_native_opt0_reference_y_plan_bundle
    );
    options.use_native_opt0_raw_y_plan_bundle = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE", options.use_native_opt0_raw_y_plan_bundle
    );
    options.use_native_opt0_y_plan_bundle_stream_first = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_Y_PLAN_BUNDLE_STREAM_FIRST",
        options.use_native_opt0_y_plan_bundle_stream_first
    );
    options.use_native_opt0_raw_y_plan_bundle_reference_streams = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_REFERENCE_STREAMS",
        options.use_native_opt0_raw_y_plan_bundle_reference_streams
    );
    options.use_native_opt0_raw_y_plan_bundle_reference_streams = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_EGGER_STREAMS",
        options.use_native_opt0_raw_y_plan_bundle_reference_streams
    );
    options.use_native_opt0_reference_local_plan_context = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_REFERENCE_LOCAL_PLAN_CONTEXT",
        options.use_native_opt0_reference_local_plan_context
    );
    options.use_native_opt0_reference_local_plan_context = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_EGGER_LOCAL_PLAN_CONTEXT",
        options.use_native_opt0_reference_local_plan_context
    );
    options.use_native_opt0_memory_feasibility_guard = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_MEMORY_FEASIBILITY_GUARD",
        options.use_native_opt0_memory_feasibility_guard
    );
    options.native_opt0_memory_feasibility_reserve_bytes =
        size_value(
            config, "FFTM_NATIVE_OPT0_MEMORY_FEASIBILITY_RESERVE_MIB",
            options.native_opt0_memory_feasibility_reserve_bytes / ( 1024u * 1024u )
        ) *
        static_cast<std::size_t>( 1024u * 1024u );
    options.contiguous_forward_send_chunk_bytes =
        size_value( config, "FFTM_CONTIGUOUS_FORWARD_SEND_CHUNK_MIB", options.contiguous_forward_send_chunk_bytes /
                                                                              ( 1024u * 1024u ) ) *
        static_cast<std::size_t>( 1024u * 1024u );

    static const char *option_keys[] = {
        "FFTM_LARGE_COUNT_P2P_TRANSPORTS",
        "FFTM_CONTIGUOUS_FORWARD_SEND_MODE",
        "FFTM_CONTIGUOUS_FORWARD_SEND_MODES",
        "FFTM_USE_DIRECT_BACKWARD_RECEIVE",
        "FFTM_DIRECT_P2P_CUDA_AWARE",
        "FFTM_USE_P2P_SEND_THREAD",
        "FFTM_USE_P2P_BYTE_TRANSFER",
        "FFTM_USE_PERSISTENT_P2P",
        "FFTM_USE_READY_P2P_SEND",
        "FFTM_USE_DIRECT_FORWARD_BYTE_RECEIVE",
        "FFTM_USE_STABLE_FORWARD_BYTE_SEND_BUFFER",
        "FFTM_USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER",
        "FFTM_USE_CONTIGUOUS_FORWARD_BYTE_SEND",
        "FFTM_USE_PHYSICAL_FORWARD_PEER_EXCHANGE",
        "FFTM_USE_LARGE_COUNT_DATATYPE_CACHE",
        "FFTM_USE_FFT_EXEC_NO_SYNC",
        "FFTM_ALLOW_FFT_EXEC_NO_SYNC",
        "FFTM_CONTIGUOUS_FORWARD_SEND_CHUNK_MIB",
        "FFTM_USE_NATIVE_BACKWARD_SECOND_PEER_LOOP",
        "FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT",
        "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY",
        "FFTM_USE_NATIVE_OPT0_EGGER_Y_BUFFER_TOPOLOGY",
        "FFTM_USE_NATIVE_OPT0_COMPACT_Y_WORKAREA",
        "FFTM_USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE",
        "FFTM_USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES",
        "FFTM_USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC",
        "FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC",
        "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR",
        "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_LIFECYCLE",
        "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_BUNDLE",
        "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE",
        "FFTM_USE_NATIVE_OPT0_Y_PLAN_BUNDLE_STREAM_FIRST",
        "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_REFERENCE_STREAMS",
        "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_EGGER_STREAMS",
        "FFTM_USE_NATIVE_OPT0_REFERENCE_LOCAL_PLAN_CONTEXT",
        "FFTM_USE_NATIVE_OPT0_EGGER_LOCAL_PLAN_CONTEXT",
        "FFTM_USE_NATIVE_OPT0_MEMORY_FEASIBILITY_GUARD",
        "FFTM_NATIVE_OPT0_MEMORY_FEASIBILITY_RESERVE_MIB",
        "FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS",
        "FFTM_AUTOTUNE_DIAGNOSTIC_ONLY",
    };
    for ( const char *key : option_keys )
    {
        if ( config.find( key ) != config.end() )
        {
            applied = true;
            break;
        }
    }

    return applied;
}

inline config_map load_config_from_environment()
{
    config_map config;
    if ( const char *path = getenv_or_null( "FFTM_AUTOTUNE_CONFIG" ) )
        config = load_key_value_file( path );
    overlay_environment( config );
    return config;
}

inline bool apply_environment_3d(
    int num_procs, const global_sizes &sizes, processor_grid &grid, fftm_init_options &options
)
{
    const bool explicit_enable = getenv_or_null( "FFTM_AUTOTUNE" ) && truthy( getenv_or_null( "FFTM_AUTOTUNE" ) );
    const bool explicit_disable = getenv_or_null( "FFTM_AUTOTUNE" ) && falsey( getenv_or_null( "FFTM_AUTOTUNE" ) );
    if ( explicit_disable )
        return false;
    if ( !explicit_enable && getenv_or_null( "FFTM_AUTOTUNE_CONFIG" ) == nullptr )
        return false;
    const config_map config = load_config_from_environment();
    return apply_config_3d( config, num_procs, sizes, grid, options );
}

} // namespace autotune
} // namespace fftm

#endif
