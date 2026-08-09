#ifndef __FFTM_AUTOTUNE_HPP__
#define __FFTM_AUTOTUNE_HPP__

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fftm.hpp"
#include "detail/runtime_hardware_identity.h"
#include "external_wrap/mpi_runtime_identity.h"

namespace fftm
{
namespace autotune
{

using config_map = std::map<std::string, std::string>;

enum class cache_mismatch_policy
{
    error,
    overwrite
};

struct autotune_constraints_3d
{
    std::string strategy_3d;
    std::string mode;
    std::string backend_3d;
    std::string grid_3d;
    std::string pencil_layout;
};

struct autotune_options
{
    std::string           cache_file;
    bool                  create_if_missing = true;
    cache_mismatch_policy mismatch_policy   = cache_mismatch_policy::error;
    bool                  validate_hardware = true;
    bool                  strict_device_identity = false;
    bool                  allow_legacy_hardware_signature = false;
    autotune_constraints_3d constraints_3d;
};

struct selected_3d_config
{
    config_map        config;
    processor_grid    grid;
    fftm_init_options init_options;
    std::string       strategy_3d;
    std::string       mode;
    std::string       backend_3d;
    std::string       source;
    // Runtime-only workspace and spectrum pool retained after an in-process
    // measured sweep. They are deliberately absent from the serialized cache.
    std::shared_ptr<detail::reusable_workspace_resource> runtime_workspace;
};

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

inline bool file_exists( const std::string &path )
{
    std::ifstream input( path.c_str() );
    return static_cast<bool>( input );
}

inline std::string sizes_to_string( const global_sizes &sizes )
{
    return std::to_string( sizes.Nx ) + "x" + std::to_string( sizes.Ny ) + "x" + std::to_string( sizes.Nz );
}

inline void save_key_value_file( const std::string &path, const config_map &config )
{
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream output( tmp_path.c_str() );
        if ( !output )
            throw std::logic_error( "FFTM autotune config cannot be written: " + tmp_path );
        output << "# FFTM C++ autotune cache\n";
        for ( const auto &entry : config )
            output << entry.first << "=" << entry.second << "\n";
    }
    if ( std::rename( tmp_path.c_str(), path.c_str() ) != 0 )
    {
        std::remove( tmp_path.c_str() );
        throw std::logic_error( "FFTM autotune config cannot be moved into place: " + path );
    }
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

inline bool constraint_matches( const std::string &required, const std::string &actual )
{
    return required.empty() || required == actual;
}

inline void validate_config_constraints_3d(
    const config_map &config, const autotune_constraints_3d &constraints,
    const std::string &cache_kind
)
{
    if ( !constraint_matches( constraints.strategy_3d, value_or_empty( config, "FFTM_AUTOTUNE_STRATEGY_3D" ) ) ||
         !constraint_matches( constraints.mode, value_or_empty( config, "FFTM_AUTOTUNE_MODE" ) ) ||
         !constraint_matches( constraints.backend_3d, value_or_empty( config, "FFTM_AUTOTUNE_BACKEND_3D" ) ) ||
         !constraint_matches( constraints.grid_3d, value_or_empty( config, "FFTM_AUTOTUNE_GRID_3D" ) ) ||
         !constraint_matches(
             constraints.pencil_layout, value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_LAYOUT" )
         ) )
    {
        throw std::logic_error(
            "FFTM " + cache_kind + " cache does not satisfy the requested 3D constraints"
        );
    }
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

inline const char *current_3d_policy_version()
{
    return "4";
}

inline const char *current_3d_policy_source()
{
    return "cpp-policy-cache-v4";
}

inline void validate_policy_cache_version( const config_map &config )
{
    const std::string prefix = "cpp-policy-cache-";
    const std::string source = value_or_empty( config, "FFTM_AUTOTUNE_SOURCE" );
    if ( source.compare( 0, prefix.size(), prefix ) != 0 )
        return;

    const std::string version = value_or_empty( config, "FFTM_AUTOTUNE_POLICY_VERSION" );
    if ( version != current_3d_policy_version() )
    {
        throw std::logic_error(
            "FFTM policy cache version " + ( version.empty() ? std::string( "missing" ) : version ) +
            " does not match current version " + current_3d_policy_version()
        );
    }
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

    validate_policy_cache_version( config );
}

#include "detail/fftm_autotune_hardware.inc"

inline std::size_t homogeneous_ranks_per_node( const hardware_inventory &inventory )
{
    if ( inventory.ranks_per_node.empty() || inventory.ranks_per_node.front() <= 0 )
        return 0;
    const int expected = inventory.ranks_per_node.front();
    for ( int value : inventory.ranks_per_node )
    {
        if ( value != expected )
            return 0;
    }
    return static_cast<std::size_t>( expected );
}

inline void set_common_native_pencil_options( config_map &config )
{
    config["FFTM_AUTOTUNE_BACKEND_3D"]     = "native";
    config["FFTM_AUTOTUNE_MODE"]           = "p2p-waitany";
    config["FFTM_AUTOTUNE_PENCIL_PIPELINE"] = "reference-parity";
    config["FFTM_LARGE_COUNT_P2P_TRANSPORTS"] = "hindexed";
    config["FFTM_DIRECT_P2P_CUDA_AWARE"] = "1";
    config["FFTM_USE_DIRECT_BACKWARD_RECEIVE"] = "0";
    config["FFTM_USE_DIRECT_FORWARD_BYTE_RECEIVE"] = "0";
    config["FFTM_USE_P2P_SEND_THREAD"] = "0";
    config["FFTM_USE_P2P_BYTE_TRANSFER"] = "1";
    config["FFTM_USE_PERSISTENT_P2P"] = "0";
    config["FFTM_USE_READY_P2P_SEND"] = "0";
    config["FFTM_USE_STABLE_FORWARD_BYTE_SEND_BUFFER"] = "0";
    config["FFTM_USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER"] = "0";
    config["FFTM_USE_CONTIGUOUS_FORWARD_BYTE_SEND"] = "0";
    config["FFTM_USE_PHYSICAL_FORWARD_PEER_EXCHANGE"] = "0";
    config["FFTM_USE_FFT_EXEC_NO_SYNC"] = "0";
    config["FFTM_USE_NATIVE_BACKWARD_SECOND_PEER_LOOP"] = "0";
    config["FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT"] = "0";
    config["FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY"] = "0";
    config["FFTM_USE_NATIVE_OPT0_COMPACT_Y_WORKAREA"] = "0";
    config["FFTM_USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE"] = "0";
    config["FFTM_USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES"] = "0";
    config["FFTM_USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC"] = "0";
    config["FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC"] = "0";
    config["FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR"] = "0";
    config["FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_LIFECYCLE"] = "0";
    config["FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_BUNDLE"] = "0";
    config["FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE"] = "0";
    config["FFTM_USE_NATIVE_OPT0_Y_PLAN_BUNDLE_STREAM_FIRST"] = "0";
    config["FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_REFERENCE_STREAMS"] = "0";
    config["FFTM_USE_NATIVE_OPT0_REFERENCE_LOCAL_PLAN_CONTEXT"] = "0";
    config["FFTM_USE_NATIVE_OPT0_MEMORY_FEASIBILITY_GUARD"] = "1";
    config["FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS"] = "0";
    config["FFTM_AUTOTUNE_DIAGNOSTIC_ONLY"] = "0";
}

inline void set_native_opt0_hot_y_options( config_map &config )
{
    config["FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT"] = "1";
    config["FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY"] = "1";
    config["FFTM_USE_NATIVE_OPT0_COMPACT_Y_WORKAREA"] = "0";
    config["FFTM_USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE"] = "1";
    config["FFTM_USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES"] = "1";
    config["FFTM_USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC"] = "1";
    config["FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC"] = "1";
    config["FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR"] = "1";
    config["FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_LIFECYCLE"] = "0";
    config["FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_BUNDLE"] = "0";
    config["FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE"] = "0";
    config["FFTM_USE_NATIVE_OPT0_Y_PLAN_BUNDLE_STREAM_FIRST"] = "0";
    config["FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_REFERENCE_STREAMS"] = "0";
    config["FFTM_USE_NATIVE_OPT0_REFERENCE_LOCAL_PLAN_CONTEXT"] = "0";
}

inline bool is_power_of_two( std::size_t value )
{
    return value != 0 && ( value & ( value - 1 ) ) == 0;
}

inline bool is_power_of_two_cube( const global_sizes &sizes )
{
    return sizes.Nx == sizes.Ny && sizes.Ny == sizes.Nz && is_power_of_two( sizes.Nx );
}

inline void set_staged_3d_policy(
    config_map &config, const std::string &strategy, std::size_t p1, std::size_t p2
)
{
    set_common_native_pencil_options( config );
    config["FFTM_AUTOTUNE_STRATEGY_3D"] = strategy;
    config["FFTM_AUTOTUNE_GRID_3D"] = std::to_string( p1 ) + "x" + std::to_string( p2 );
    config["FFTM_AUTOTUNE_PENCIL_LAYOUT"] = "auto";
    config["FFTM_AUTOTUNE_PENCIL_PIPELINE"] = "staged";
}

inline void set_default_pencil_pencil_3d_policy(
    config_map &config, int num_procs, std::size_t ranks_per_node = 0
)
{
    set_common_native_pencil_options( config );
    config["FFTM_AUTOTUNE_STRATEGY_3D"] = "pencil-pencil";

    if ( num_procs == 8 )
    {
        config["FFTM_AUTOTUNE_GRID_3D"] = "4x2";
        config["FFTM_AUTOTUNE_PENCIL_LAYOUT"] = "opt0";
        set_native_opt0_hot_y_options( config );
    }
    else if ( num_procs == 7 )
    {
        config["FFTM_AUTOTUNE_GRID_3D"] = "7x1";
        config["FFTM_AUTOTUNE_PENCIL_LAYOUT"] = "opt0";
        set_native_opt0_hot_y_options( config );
    }
    else if ( num_procs == 6 )
    {
        config["FFTM_AUTOTUNE_GRID_3D"] = "2x3";
        config["FFTM_AUTOTUNE_PENCIL_LAYOUT"] = "opt1";
    }
    else if ( num_procs == 5 )
    {
        config["FFTM_AUTOTUNE_GRID_3D"] = "5x1";
        config["FFTM_AUTOTUNE_PENCIL_LAYOUT"] = "opt1";
    }
    else if ( select_production_pencil_layout_3d( num_procs, true ) == fftm_3d_pencil_layout::opt0 )
    {
        auto grid = default_pencil_grid_3d( num_procs );
        if ( ranks_per_node > 1 && static_cast<std::size_t>( num_procs ) % ranks_per_node == 0 )
        {
            grid = std::make_pair(
                static_cast<std::size_t>( num_procs ) / ranks_per_node, ranks_per_node
            );
        }
        config["FFTM_AUTOTUNE_GRID_3D"] =
            std::to_string( grid.first ) + "x" + std::to_string( grid.second );
        config["FFTM_AUTOTUNE_PENCIL_LAYOUT"] = "opt0";
        set_native_opt0_hot_y_options( config );
    }
    else
    {
        const auto grid = default_pencil_grid_3d( num_procs );
        config["FFTM_AUTOTUNE_GRID_3D"] = std::to_string( grid.first ) + "x" + std::to_string( grid.second );
        config["FFTM_AUTOTUNE_PENCIL_LAYOUT"] = "opt1";
    }
}

inline config_map make_default_3d_policy_config(
    int num_procs, const global_sizes &sizes, std::size_t ranks_per_node = 0
)
{
    if ( num_procs <= 0 )
        throw std::logic_error( "FFTM 3D policy requires a positive MPI size" );

    config_map config;
    config["FFTM_AUTOTUNE_SCHEMA"] = "2";
    config["FFTM_AUTOTUNE_POLICY_VERSION"] = current_3d_policy_version();
    config["FFTM_AUTOTUNE_LIBRARY"] = "fftm";
    config["FFTM_AUTOTUNE_DIM"] = "3";
    config["FFTM_AUTOTUNE_NUM_GPUS"] = std::to_string( num_procs );
    config["FFTM_AUTOTUNE_SIZE_3D"] = sizes_to_string( sizes );
    config["FFTM_AUTOTUNE_SOURCE"] = current_3d_policy_source();

    if ( num_procs <= 1 )
    {
        set_staged_3d_policy( config, "slab-pencil", 1, 1 );
        return config;
    }

    // Release data favors slab-pencil for power-of-two cubes and for the
    // validated 8-32 rank multinode range. Fitted non-power-of-two cases on a
    // single node favor pencil-slab. At 64 ranks and above, the native opt0
    // pencil path uses a node-aligned grid when uniform node occupancy is
    // available from the hardware inventory. Measured tuning remains
    // authoritative whenever it is enabled.
    if ( num_procs <= 32 && ( is_power_of_two_cube( sizes ) || num_procs > 8 ) )
    {
        set_staged_3d_policy( config, "slab-pencil", static_cast<std::size_t>( num_procs ), 1 );
        if ( num_procs == 32 )
            config["FFTM_AUTOTUNE_MODE"] = "alltoallv";
    }
    else if ( num_procs <= 8 )
        set_staged_3d_policy( config, "pencil-slab", 1, static_cast<std::size_t>( num_procs ) );
    else
        set_default_pencil_pencil_3d_policy( config, num_procs, ranks_per_node );
    return config;
}

inline config_map make_default_3d_config(
    int num_procs, const global_sizes &sizes, const hardware_inventory &inventory
)
{
    config_map config = make_default_3d_policy_config(
        num_procs, sizes, homogeneous_ranks_per_node( inventory )
    );
    const auto signature = make_hardware_signature_record(
        inventory, hardware_transport_policy( config, num_procs )
    );
    set_hardware_signature_metadata( config, inventory, signature );
    return config;
}

template <class RuntimeApi>
inline config_map make_default_3d_config( int num_procs, const global_sizes &sizes )
{
    return make_default_3d_config(
        num_procs, sizes, query_local_hardware_inventory<RuntimeApi>( num_procs )
    );
}

inline bool apply_config_3d(
    const config_map &config, int num_procs, const global_sizes &sizes, processor_grid &grid,
    fftm_init_options &options
);

template <class RuntimeApi>
inline selected_3d_config load_or_create_3d_config(
    int num_procs, const global_sizes &sizes, const autotune_options &autotune
)
{
    if ( autotune.cache_file.empty() )
        throw std::logic_error( "fftm::autotune_options::cache_file must not be empty" );

    const hardware_inventory inventory = query_local_hardware_inventory<RuntimeApi>( num_procs );
    config_map config;
    std::string source;
    if ( file_exists( autotune.cache_file ) )
    {
        config = load_key_value_file( autotune.cache_file );
        try
        {
            validate_match( config, num_procs, sizes );
            if ( autotune.validate_hardware )
                validate_hardware_match( config, inventory, autotune );
        }
        catch ( const std::exception & )
        {
            if ( autotune.mismatch_policy != cache_mismatch_policy::overwrite )
                throw;
            config = make_default_3d_config( num_procs, sizes, inventory );
            save_key_value_file( autotune.cache_file, config );
            source = "created";
        }
        if ( source.empty() )
            source = "cache";
    }
    else
    {
        if ( !autotune.create_if_missing )
            throw std::logic_error( "FFTM autotune cache file does not exist: " + autotune.cache_file );
        config = make_default_3d_config( num_procs, sizes, inventory );
        save_key_value_file( autotune.cache_file, config );
        source = "created";
    }

    validate_config_constraints_3d( config, autotune.constraints_3d, "policy" );

    processor_grid    grid;
    fftm_init_options options;
    if ( !apply_config_3d( config, num_procs, sizes, grid, options ) )
        throw std::logic_error( "FFTM autotune config did not select a usable 3D plan" );

    selected_3d_config selected;
    selected.config       = config;
    selected.grid         = grid;
    selected.init_options = options;
    selected.strategy_3d  = value_or_empty( config, "FFTM_AUTOTUNE_STRATEGY_3D" );
    selected.mode         = value_or_empty( config, "FFTM_AUTOTUNE_MODE" );
    selected.backend_3d   = value_or_empty( config, "FFTM_AUTOTUNE_BACKEND_3D" );
    selected.source       = source;
    return selected;
}

template <class RuntimeApi, class MPIComm>
inline selected_3d_config load_or_create_3d_config(
    const MPIComm &comm, const global_sizes &sizes, const autotune_options &autotune
)
{
    if ( autotune.cache_file.empty() )
        throw std::logic_error( "fftm::autotune_options::cache_file must not be empty" );

    // Every rank participates before rank 0 performs cache I/O. The inventory
    // gathers are collective for an SCFD communicator.
    const hardware_inventory inventory = query_distributed_hardware_inventory<RuntimeApi>( comm );

    int cache_state        = 0; // 0: existing cache, 1: created/replaced by rank 0, 2: missing and creation disabled.
    int root_prepare_error = 0;
    if ( comm.myid == 0 )
    {
        try
        {
            const bool exists_on_root = file_exists( autotune.cache_file );
            if ( exists_on_root )
            {
                bool cache_valid = true;
                try
                {
                    const config_map config = load_key_value_file( autotune.cache_file );
                    validate_match( config, comm.num_procs, sizes );
                    if ( autotune.validate_hardware )
                        validate_hardware_match( config, inventory, autotune );
                }
                catch ( const std::exception & )
                {
                    cache_valid = false;
                }
                if ( !cache_valid && autotune.mismatch_policy == cache_mismatch_policy::overwrite )
                {
                    save_key_value_file(
                        autotune.cache_file, make_default_3d_config( comm.num_procs, sizes, inventory )
                    );
                    cache_state = 1;
                }
            }
            else if ( autotune.create_if_missing )
            {
                save_key_value_file(
                    autotune.cache_file, make_default_3d_config( comm.num_procs, sizes, inventory )
                );
                cache_state = 1;
            }
            else
            {
                cache_state = 2;
            }
        }
        catch ( const std::exception & )
        {
            root_prepare_error = 1;
        }
    }

    comm.bcast( &root_prepare_error, 1, 0 );
    comm.bcast( &cache_state, 1, 0 );
    comm.barrier();
    if ( root_prepare_error )
        throw std::logic_error( "Rank 0 failed to prepare FFTM autotune cache: " + autotune.cache_file );
    if ( cache_state == 2 )
        throw std::logic_error( "FFTM autotune cache file does not exist: " + autotune.cache_file );

    config_map config = load_key_value_file( autotune.cache_file );
    validate_match( config, comm.num_procs, sizes );
    if ( autotune.validate_hardware )
        validate_hardware_match( config, inventory, autotune );
    validate_config_constraints_3d( config, autotune.constraints_3d, "policy" );

    processor_grid    grid;
    fftm_init_options options;
    if ( !apply_config_3d( config, comm.num_procs, sizes, grid, options ) )
        throw std::logic_error( "FFTM autotune config did not select a usable 3D plan" );

    selected_3d_config selected;
    selected.config       = config;
    selected.grid         = grid;
    selected.init_options = options;
    selected.strategy_3d  = value_or_empty( config, "FFTM_AUTOTUNE_STRATEGY_3D" );
    selected.mode         = value_or_empty( config, "FFTM_AUTOTUNE_MODE" );
    selected.backend_3d   = value_or_empty( config, "FFTM_AUTOTUNE_BACKEND_3D" );
    selected.source       = cache_state == 0 ? "cache" : "created";
    return selected;
}

template <class FFTMPlan>
inline void init_autotuned_3d_plan(
    FFTMPlan &plan, const selected_3d_config &selected, const global_sizes &sizes
)
{
    if ( selected.runtime_workspace )
        plan.set_reusable_workspace_resource( selected.runtime_workspace );
    plan.template init<3>( selected.grid, sizes, selected.init_options );
}

template <class Sizes>
inline std::size_t checked_local_3d_bytes(
    const Sizes &sizes, std::size_t element_size, const char *description
)
{
    std::size_t result = element_size;
    for ( int dimension = 0; dimension < 3; ++dimension )
    {
        const std::size_t extent = dimension == 0 ? static_cast<std::size_t>( std::get<0>( sizes ) )
                                 : dimension == 1 ? static_cast<std::size_t>( std::get<1>( sizes ) )
                                                  : static_cast<std::size_t>( std::get<2>( sizes ) );
        if ( extent != 0 && result > std::numeric_limits<std::size_t>::max() / extent )
            throw std::overflow_error( std::string( "FFTM autotune " ) + description + " size overflows size_t" );
        result *= extent;
    }
    return result;
}

template <class RealArray, class SpectrumArray, class InputSizes, class SpectrumSizes>
inline void init_autotuned_3d_data_arrays(
    RealArray &input, SpectrumArray &spectrum, const selected_3d_config &selected,
    const InputSizes &input_sizes, const SpectrumSizes &spectrum_sizes
)
{
    if ( !selected.runtime_workspace )
    {
        input.init( std::get<0>( input_sizes ), std::get<1>( input_sizes ), std::get<2>( input_sizes ) );
        spectrum.init(
            std::get<0>( spectrum_sizes ), std::get<1>( spectrum_sizes ), std::get<2>( spectrum_sizes )
        );
        return;
    }

    const std::size_t spectrum_bytes = checked_local_3d_bytes(
        spectrum_sizes, sizeof( typename SpectrumArray::value_type ), "spectrum buffer"
    );
    // The real input is not exposed to device-aware MPI in the validated 3D
    // candidates and releases cleanly. Keep its normal SCFD ownership and only
    // retain the communication-visible spectrum allocation between candidates.
    input.init( std::get<0>( input_sizes ), std::get<1>( input_sizes ), std::get<2>( input_sizes ) );
    selected.runtime_workspace->require_spectrum_size_bytes( spectrum_bytes );
    selected.runtime_workspace->activate_spectrum();

    spectrum.init_by_raw_data(
        static_cast<typename SpectrumArray::pointer_type>( selected.runtime_workspace->spectrum_ptr() ),
        std::get<0>( spectrum_sizes ), std::get<1>( spectrum_sizes ), std::get<2>( spectrum_sizes )
    );
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
        options.execution.large_count_p2p_transport = parse_large_count_transport( split( large_count, ',' ).front() );
        applied                           = true;
    }

    const bool allow_fft_exec_no_sync = bool_value( config, "FFTM_ALLOW_FFT_EXEC_NO_SYNC", false );
    options.diagnostics.allow_native_opt0_diagnostic_variants = bool_value(
        config, "FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS", options.diagnostics.allow_native_opt0_diagnostic_variants
    );
    if ( allow_fft_exec_no_sync )
        options.diagnostics.allow_native_opt0_diagnostic_variants = true;
    if ( bool_value( config, "FFTM_AUTOTUNE_DIAGNOSTIC_ONLY", false ) &&
         !options.diagnostics.allow_native_opt0_diagnostic_variants )
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
        options.diagnostics.contiguous_forward_send_mode = parse_contiguous_forward_send_mode( first_csv_value( contiguous_mode ) );
        applied                              = true;
    }

    options.execution.use_direct_backward_receive = bool_value(
        config, "FFTM_USE_DIRECT_BACKWARD_RECEIVE", options.execution.use_direct_backward_receive
    );
    // Keep the historical key as a cache/config compatibility alias. It means
    // direct device-aware MPI for whichever backend was selected at compile time.
    options.execution.direct_p2p_cuda_aware = bool_value(
        config, "FFTM_DIRECT_P2P_CUDA_AWARE", options.execution.direct_p2p_cuda_aware
    );
    options.diagnostics.use_p2p_send_thread = bool_value( config, "FFTM_USE_P2P_SEND_THREAD", options.diagnostics.use_p2p_send_thread );
    options.execution.use_p2p_byte_transfer = bool_value( config, "FFTM_USE_P2P_BYTE_TRANSFER", options.execution.use_p2p_byte_transfer );
    options.execution.use_persistent_p2p = bool_value( config, "FFTM_USE_PERSISTENT_P2P", options.execution.use_persistent_p2p );
    options.diagnostics.use_ready_p2p_send = bool_value( config, "FFTM_USE_READY_P2P_SEND", options.diagnostics.use_ready_p2p_send );
    options.diagnostics.use_direct_forward_byte_receive = bool_value(
        config, "FFTM_USE_DIRECT_FORWARD_BYTE_RECEIVE", options.diagnostics.use_direct_forward_byte_receive
    );
    options.diagnostics.use_stable_forward_byte_send_buffer = bool_value(
        config, "FFTM_USE_STABLE_FORWARD_BYTE_SEND_BUFFER", options.diagnostics.use_stable_forward_byte_send_buffer
    );
    options.diagnostics.use_ready_stable_forward_byte_send_buffer = bool_value(
        config, "FFTM_USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER",
        options.diagnostics.use_ready_stable_forward_byte_send_buffer
    );
    options.diagnostics.use_contiguous_forward_byte_send = bool_value(
        config, "FFTM_USE_CONTIGUOUS_FORWARD_BYTE_SEND", options.diagnostics.use_contiguous_forward_byte_send
    );
    options.diagnostics.use_physical_forward_peer_exchange = bool_value(
        config, "FFTM_USE_PHYSICAL_FORWARD_PEER_EXCHANGE", options.diagnostics.use_physical_forward_peer_exchange
    );
    options.diagnostics.use_large_count_datatype_cache = bool_value(
        config, "FFTM_USE_LARGE_COUNT_DATATYPE_CACHE", options.diagnostics.use_large_count_datatype_cache
    );
    options.diagnostics.use_fft_exec_no_sync = bool_value( config, "FFTM_USE_FFT_EXEC_NO_SYNC", options.diagnostics.use_fft_exec_no_sync );
    if ( options.diagnostics.use_fft_exec_no_sync && !allow_fft_exec_no_sync )
    {
        throw std::logic_error(
            "FFTM_USE_FFT_EXEC_NO_SYNC=1 is diagnostic-only. It removes generic cuFFT execution synchronization "
            "and can race later MPI/copy stages. Use the native opt0 Y no-sync option instead, or set "
            "FFTM_ALLOW_FFT_EXEC_NO_SYNC=1 only for explicit diagnostics."
        );
    }
    options.diagnostics.use_native_backward_second_peer_loop = bool_value(
        config, "FFTM_USE_NATIVE_BACKWARD_SECOND_PEER_LOOP", options.diagnostics.use_native_backward_second_peer_loop
    );
    options.execution.use_native_opt0_default_z_layout = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT", options.execution.use_native_opt0_default_z_layout
    );
    options.execution.use_native_opt0_reference_y_buffer_topology = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY",
        options.execution.use_native_opt0_reference_y_buffer_topology
    );
    options.execution.use_native_opt0_reference_y_buffer_topology = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_EGGER_Y_BUFFER_TOPOLOGY",
        options.execution.use_native_opt0_reference_y_buffer_topology
    );
    options.execution.use_native_opt0_compact_y_workarea = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_COMPACT_Y_WORKAREA",
        options.execution.use_native_opt0_compact_y_workarea
    );
    options.execution.use_native_opt0_tight_y_plan_sequence = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE", options.execution.use_native_opt0_tight_y_plan_sequence
    );
    options.execution.use_native_opt0_shared_y_plan_handles = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES", options.execution.use_native_opt0_shared_y_plan_handles
    );
    options.execution.use_native_opt0_y_group_device_sync = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC", options.execution.use_native_opt0_y_group_device_sync
    );
    options.execution.use_native_opt0_y_no_sync_exec = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC", options.execution.use_native_opt0_y_no_sync_exec
    );
    options.execution.use_native_opt0_raw_y_plan_array_executor = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR",
        options.execution.use_native_opt0_raw_y_plan_array_executor
    );
    options.diagnostics.use_native_opt0_reference_y_plan_lifecycle = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_LIFECYCLE",
        options.diagnostics.use_native_opt0_reference_y_plan_lifecycle
    );
    options.diagnostics.use_native_opt0_reference_y_plan_bundle = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_BUNDLE", options.diagnostics.use_native_opt0_reference_y_plan_bundle
    );
    options.diagnostics.use_native_opt0_raw_y_plan_bundle = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE", options.diagnostics.use_native_opt0_raw_y_plan_bundle
    );
    options.diagnostics.use_native_opt0_y_plan_bundle_stream_first = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_Y_PLAN_BUNDLE_STREAM_FIRST",
        options.diagnostics.use_native_opt0_y_plan_bundle_stream_first
    );
    options.diagnostics.use_native_opt0_raw_y_plan_bundle_reference_streams = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_REFERENCE_STREAMS",
        options.diagnostics.use_native_opt0_raw_y_plan_bundle_reference_streams
    );
    options.diagnostics.use_native_opt0_raw_y_plan_bundle_reference_streams = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_EGGER_STREAMS",
        options.diagnostics.use_native_opt0_raw_y_plan_bundle_reference_streams
    );
    options.diagnostics.use_native_opt0_reference_local_plan_context = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_REFERENCE_LOCAL_PLAN_CONTEXT",
        options.diagnostics.use_native_opt0_reference_local_plan_context
    );
    options.diagnostics.use_native_opt0_reference_local_plan_context = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_EGGER_LOCAL_PLAN_CONTEXT",
        options.diagnostics.use_native_opt0_reference_local_plan_context
    );
    options.execution.use_native_opt0_memory_feasibility_guard = bool_value(
        config, "FFTM_USE_NATIVE_OPT0_MEMORY_FEASIBILITY_GUARD",
        options.execution.use_native_opt0_memory_feasibility_guard
    );
    options.execution.native_opt0_memory_feasibility_reserve_bytes =
        size_value(
            config, "FFTM_NATIVE_OPT0_MEMORY_FEASIBILITY_RESERVE_MIB",
            options.execution.native_opt0_memory_feasibility_reserve_bytes / ( 1024u * 1024u )
        ) *
        static_cast<std::size_t>( 1024u * 1024u );
    options.diagnostics.contiguous_forward_send_chunk_bytes =
        size_value( config, "FFTM_CONTIGUOUS_FORWARD_SEND_CHUNK_MIB", options.diagnostics.contiguous_forward_send_chunk_bytes /
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
