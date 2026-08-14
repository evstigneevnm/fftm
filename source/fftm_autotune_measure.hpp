#ifndef __FFTM_AUTOTUNE_MEASURE_HPP__
#define __FFTM_AUTOTUNE_MEASURE_HPP__

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fftm_autotune.hpp"

namespace fftm
{
namespace autotune
{

struct measured_3d_options
{
    int         warmup = 2;
    int         iterations = 5;
    bool        include_slab_pencil = true;
    bool        include_pencil_slab = true;
    bool        include_pencil_pencil = true;
    bool        include_p2p_waitany = true;
    bool        include_alltoallv = true;
    bool        include_opt0 = true;
    bool        include_opt1 = true;
    bool        include_high_memory_opt0 = false;
    bool        production_candidates_only = true;
    bool        replace_policy_cache = true;
    bool        continue_on_candidate_error = true;
    bool        log_candidate_memory = false;
    bool        verify_candidate_memory_recovery = true;
    std::size_t candidate_memory_recovery_tolerance_bytes =
        static_cast<std::size_t>( 1024 ) * static_cast<std::size_t>( 1024 ) * static_cast<std::size_t>( 1024 );
    std::size_t max_pencil_grids = 4;
};

struct candidate_measurement
{
    bool        valid = false;
    double      median_wall_ms = std::numeric_limits<double>::infinity();
    std::string error;
};

struct measured_3d_candidate
{
    config_map             config;
    candidate_measurement  measurement;
};

inline std::string measured_double_string( double value )
{
    std::ostringstream out;
    out << std::setprecision( 17 ) << value;
    return out.str();
}

inline std::string candidate_id_3d( const config_map &config )
{
    return normalize_hardware_token(
        value_or_empty( config, "FFTM_AUTOTUNE_STRATEGY_3D" ) + "_" +
        value_or_empty( config, "FFTM_AUTOTUNE_MODE" ) + "_" +
        value_or_empty( config, "FFTM_AUTOTUNE_GRID_3D" ) + "_" +
        value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_LAYOUT" )
    );
}

inline void append_unique_grid(
    std::vector<std::pair<std::size_t, std::size_t>> &grids, std::size_t p1, std::size_t p2,
    std::size_t max_grids
)
{
    if ( p1 == 0 || p2 == 0 || p1 * p2 == 0 || grids.size() >= max_grids )
        return;
    const auto value = std::make_pair( p1, p2 );
    if ( std::find( grids.begin(), grids.end(), value ) == grids.end() )
        grids.push_back( value );
}

inline std::vector<std::pair<std::size_t, std::size_t>> measured_pencil_grids_3d(
    int num_procs, std::size_t max_grids
)
{
    std::vector<std::pair<std::size_t, std::size_t>> result;
    if ( num_procs <= 0 || max_grids == 0 )
        return result;
    const auto balanced = default_pencil_grid_3d( num_procs );
    append_unique_grid( result, balanced.first, balanced.second, max_grids );
    append_unique_grid( result, balanced.second, balanced.first, max_grids );
    append_unique_grid( result, static_cast<std::size_t>( num_procs ), 1, max_grids );
    append_unique_grid( result, 1, static_cast<std::size_t>( num_procs ), max_grids );
    return result;
}

inline config_map make_measured_3d_candidate(
    int num_procs, const global_sizes &sizes, const std::string &strategy, const std::string &mode,
    std::size_t p1, std::size_t p2, const std::string &layout
)
{
    config_map config = make_default_3d_policy_config( num_procs, sizes );
    set_common_native_pencil_options( config );
    config["FFTM_AUTOTUNE_SOURCE"] = "cpp-measured-candidate";
    config["FFTM_AUTOTUNE_STRATEGY_3D"] = strategy;
    config["FFTM_AUTOTUNE_MODE"] = mode;
    config["FFTM_AUTOTUNE_GRID_3D"] = std::to_string( p1 ) + "x" + std::to_string( p2 );
    config["FFTM_AUTOTUNE_PENCIL_LAYOUT"] = layout;

    if ( strategy == "pencil-pencil" )
    {
        config["FFTM_AUTOTUNE_PENCIL_PIPELINE"] = "reference-parity";
        if ( layout == "opt0" )
            set_native_opt0_hot_y_options( config );
    }
    else
    {
        config["FFTM_AUTOTUNE_PENCIL_PIPELINE"] = "staged";
    }
    config["FFTM_AUTOTUNE_CANDIDATE_ID"] = candidate_id_3d( config );
    return config;
}

inline void append_unique_measured_3d_candidate(
    std::vector<config_map> &candidates, config_map candidate
)
{
    candidate["FFTM_AUTOTUNE_CANDIDATE_ID"] = candidate_id_3d( candidate );
    const std::string id = candidate["FFTM_AUTOTUNE_CANDIDATE_ID"];
    for ( const auto &existing : candidates )
    {
        if ( value_or_empty( existing, "FFTM_AUTOTUNE_CANDIDATE_ID" ) == id )
            return;
    }
    candidates.push_back( std::move( candidate ) );
}

inline bool measured_strategy_enabled_3d(
    const std::string &strategy, const measured_3d_options &options
)
{
    if ( strategy == "slab-pencil" )
        return options.include_slab_pencil;
    if ( strategy == "pencil-slab" )
        return options.include_pencil_slab;
    if ( strategy == "pencil-pencil" )
        return options.include_pencil_pencil;
    return false;
}

inline int measured_workspace_priority_3d( const config_map &candidate )
{
    const std::string strategy = value_or_empty( candidate, "FFTM_AUTOTUNE_STRATEGY_3D" );
    const std::string layout = value_or_empty( candidate, "FFTM_AUTOTUNE_PENCIL_LAYOUT" );
    if ( strategy == "pencil-pencil" && layout == "opt0" )
        return 4;
    if ( strategy == "pencil-pencil" )
        return 3;
    if ( strategy == "pencil-slab" )
        return 2;
    return 1;
}

inline std::vector<config_map> make_measured_3d_candidates(
    int num_procs, const global_sizes &sizes, const measured_3d_options &options,
    std::size_t ranks_per_node = 0
)
{
    if ( num_procs <= 0 )
        throw std::logic_error( "FFTM measured autotune requires a positive MPI size" );
    if ( options.warmup < 0 || options.iterations <= 0 )
        throw std::logic_error( "FFTM measured autotune requires warmup >= 0 and iterations > 0" );

    std::vector<std::string> modes;
    if ( options.include_p2p_waitany )
        modes.push_back( "p2p-waitany" );
    if ( options.include_alltoallv )
        modes.push_back( "alltoallv" );
    if ( modes.empty() )
        throw std::logic_error( "FFTM measured autotune has no enabled communication mode" );

    std::vector<config_map> result;
    for ( const auto &mode : modes )
    {
        if ( options.production_candidates_only )
        {
            if ( options.include_slab_pencil )
            {
                append_unique_measured_3d_candidate( result, make_measured_3d_candidate(
                    num_procs, sizes, "slab-pencil", mode,
                    static_cast<std::size_t>( num_procs ), 1, "auto"
                ) );
            }
            if ( num_procs > 1 && options.include_pencil_slab )
            {
                append_unique_measured_3d_candidate( result, make_measured_3d_candidate(
                    num_procs, sizes, "pencil-slab", mode, 1,
                    static_cast<std::size_t>( num_procs ), "auto"
                ) );
            }
            if ( num_procs > 1 && options.include_pencil_pencil )
            {
                config_map pencil = make_default_3d_policy_config(
                    num_procs, sizes, ranks_per_node
                );
                set_default_pencil_pencil_3d_policy( pencil, num_procs, ranks_per_node );
                pencil["FFTM_AUTOTUNE_SOURCE"] = "cpp-measured-candidate";
                pencil["FFTM_AUTOTUNE_MODE"] = mode;
                pencil["FFTM_AUTOTUNE_CANDIDATE_ID"] = candidate_id_3d( pencil );
                append_unique_measured_3d_candidate( result, std::move( pencil ) );
            }
            continue;
        }

        if ( options.include_slab_pencil )
        {
            result.push_back( make_measured_3d_candidate(
                num_procs, sizes, "slab-pencil", mode, static_cast<std::size_t>( num_procs ), 1, "auto"
            ) );
        }
        if ( num_procs > 1 && options.include_pencil_slab )
        {
            result.push_back( make_measured_3d_candidate(
                num_procs, sizes, "pencil-slab", mode, 1, static_cast<std::size_t>( num_procs ), "auto"
            ) );
        }
        if ( num_procs > 1 && options.include_pencil_pencil )
        {
            const auto grids = measured_pencil_grids_3d( num_procs, options.max_pencil_grids );
            for ( const auto &grid : grids )
            {
                // The optimized opt1 p2-degenerate shortcut still consumes the
                // legacy (x,z,y) public shape. It is not a valid candidate for
                // the optimized (x,y,z) API contract.
                if ( options.include_opt1 && ( grid.second != 1 || grid.first == 1 ) )
                {
                    result.push_back( make_measured_3d_candidate(
                        num_procs, sizes, "pencil-pencil", mode, grid.first, grid.second, "opt1"
                    ) );
                }
                const bool high_memory_count = num_procs >= 2 && num_procs <= 6;
                if ( options.include_opt0 && ( !high_memory_count || options.include_high_memory_opt0 ) )
                {
                    result.push_back( make_measured_3d_candidate(
                        num_procs, sizes, "pencil-pencil", mode, grid.first, grid.second, "opt0"
                    ) );
                }
            }
        }
    }
    std::stable_sort( result.begin(), result.end(), []( const config_map &lhs, const config_map &rhs ) {
        return measured_workspace_priority_3d( lhs ) > measured_workspace_priority_3d( rhs );
    } );
    if ( result.empty() )
        throw std::logic_error( "FFTM measured autotune produced no candidates" );
    return result;
}

inline selected_3d_config selected_3d_from_config(
    const config_map &config, int num_procs, const global_sizes &sizes, const std::string &source
)
{
    selected_3d_config selected;
    selected.config = config;
    if ( !apply_config_3d( config, num_procs, sizes, selected.grid, selected.init_options ) )
        throw std::logic_error( "FFTM measured autotune candidate did not select a usable 3D plan" );
    selected.strategy_3d = value_or_empty( config, "FFTM_AUTOTUNE_STRATEGY_3D" );
    selected.mode = value_or_empty( config, "FFTM_AUTOTUNE_MODE" );
    selected.backend_3d = value_or_empty( config, "FFTM_AUTOTUNE_BACKEND_3D" );
    selected.source = source;
    return selected;
}

inline bool measured_cache_3d( const config_map &config )
{
    return value_or_empty( config, "FFTM_AUTOTUNE_SOURCE" ) == "cpp-measured-v2" &&
           value_or_empty( config, "FFTM_AUTOTUNE_CANDIDATE_POLICY_VERSION" ) == "2" &&
           !value_or_empty( config, "FFTM_AUTOTUNE_WINNER_MEDIAN_MS" ).empty();
}

inline std::string measured_candidate_prefix( std::size_t index )
{
    std::ostringstream prefix;
    prefix << "FFTM_AUTOTUNE_CANDIDATE_" << std::setfill( '0' ) << std::setw( 3 ) << index << '_';
    return prefix.str();
}

inline bool measured_candidate_matches(
    const config_map &cache, std::size_t index, const autotune_constraints_3d &constraints
)
{
    const std::string prefix = measured_candidate_prefix( index );
    return truthy( value_or_empty( cache, prefix + "VALID" ) ) &&
           constraint_matches( constraints.strategy_3d, value_or_empty( cache, prefix + "STRATEGY_3D" ) ) &&
           constraint_matches( constraints.mode, value_or_empty( cache, prefix + "MODE" ) ) &&
           constraint_matches( constraints.backend_3d, value_or_empty( cache, prefix + "BACKEND_3D" ) ) &&
           constraint_matches( constraints.grid_3d, value_or_empty( cache, prefix + "GRID_3D" ) ) &&
           constraint_matches(
               constraints.pencil_layout, value_or_empty( cache, prefix + "PENCIL_LAYOUT" )
           );
}

inline config_map select_measured_candidate_config(
    const config_map &cache, const autotune_constraints_3d &constraints
)
{
    if ( !measured_cache_3d( cache ) )
    {
        validate_config_constraints_3d( cache, constraints, "policy" );
        return cache;
    }

    const std::size_t count = static_cast<std::size_t>( std::strtoull(
        value_or_empty( cache, "FFTM_AUTOTUNE_CANDIDATE_COUNT" ).c_str(), nullptr, 10
    ) );
    std::size_t best_index = count;
    double      best_ms = std::numeric_limits<double>::infinity();
    for ( std::size_t index = 0; index < count; ++index )
    {
        if ( !measured_candidate_matches( cache, index, constraints ) )
            continue;
        const std::string prefix = measured_candidate_prefix( index );
        const double wall_ms = std::strtod(
            value_or_empty( cache, prefix + "MEDIAN_MS" ).c_str(), nullptr
        );
        if ( std::isfinite( wall_ms ) && wall_ms > 0.0 && wall_ms < best_ms )
        {
            best_ms = wall_ms;
            best_index = index;
        }
    }
    if ( best_index == count )
        throw std::logic_error( "No measured FFTM 3D candidate satisfies the requested constraints" );

    const std::string prefix = measured_candidate_prefix( best_index );
    config_map selected = cache;
    set_common_native_pencil_options( selected );
    selected["FFTM_AUTOTUNE_STRATEGY_3D"] = value_or_empty( cache, prefix + "STRATEGY_3D" );
    selected["FFTM_AUTOTUNE_MODE"] = value_or_empty( cache, prefix + "MODE" );
    selected["FFTM_AUTOTUNE_BACKEND_3D"] = value_or_empty( cache, prefix + "BACKEND_3D" );
    selected["FFTM_AUTOTUNE_GRID_3D"] = value_or_empty( cache, prefix + "GRID_3D" );
    selected["FFTM_AUTOTUNE_PENCIL_LAYOUT"] = value_or_empty( cache, prefix + "PENCIL_LAYOUT" );
    selected["FFTM_AUTOTUNE_PENCIL_PIPELINE"] = value_or_empty( cache, prefix + "PENCIL_PIPELINE" );
    if ( selected["FFTM_AUTOTUNE_STRATEGY_3D"] == "pencil-pencil" &&
         selected["FFTM_AUTOTUNE_PENCIL_LAYOUT"] == "opt0" )
    {
        set_native_opt0_hot_y_options( selected );
    }
    selected["FFTM_AUTOTUNE_ACTIVE_CANDIDATE_INDEX"] = std::to_string( best_index );
    selected["FFTM_AUTOTUNE_ACTIVE_MEDIAN_MS"] = measured_double_string( best_ms );
    return selected;
}

inline void store_measured_candidates(
    config_map &winner, const std::vector<measured_3d_candidate> &candidates,
    std::size_t winner_index, const measured_3d_options &options
)
{
    winner["FFTM_AUTOTUNE_SOURCE"] = "cpp-measured-v2";
    winner["FFTM_AUTOTUNE_CANDIDATE_POLICY_VERSION"] = "2";
    winner["FFTM_AUTOTUNE_MEASURED"] = "1";
    winner["FFTM_AUTOTUNE_MEASURE_WARMUP"] = std::to_string( options.warmup );
    winner["FFTM_AUTOTUNE_MEASURE_ITERATIONS"] = std::to_string( options.iterations );
    winner["FFTM_AUTOTUNE_PRODUCTION_CANDIDATES_ONLY"] =
        options.production_candidates_only ? "1" : "0";
    winner["FFTM_AUTOTUNE_CANDIDATE_COUNT"] = std::to_string( candidates.size() );
    winner["FFTM_AUTOTUNE_WINNER_INDEX"] = std::to_string( winner_index );
    winner["FFTM_AUTOTUNE_WINNER_MEDIAN_MS"] =
        measured_double_string( candidates[winner_index].measurement.median_wall_ms );

    for ( std::size_t i = 0; i < candidates.size(); ++i )
    {
        const std::string prefix = measured_candidate_prefix( i );
        const auto &candidate = candidates[i];
        winner[prefix + "ID"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_CANDIDATE_ID" );
        winner[prefix + "VALID"] = candidate.measurement.valid ? "1" : "0";
        winner[prefix + "MEDIAN_MS"] = candidate.measurement.valid ?
            measured_double_string( candidate.measurement.median_wall_ms ) : "nan";
        winner[prefix + "STRATEGY_3D"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_STRATEGY_3D" );
        winner[prefix + "MODE"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_MODE" );
        winner[prefix + "BACKEND_3D"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_BACKEND_3D" );
        winner[prefix + "GRID_3D"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_GRID_3D" );
        winner[prefix + "PENCIL_LAYOUT"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_PENCIL_LAYOUT" );
        winner[prefix + "PENCIL_PIPELINE"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_PENCIL_PIPELINE" );
        if ( !candidate.measurement.error.empty() )
            winner[prefix + "ERROR"] = normalize_hardware_token( candidate.measurement.error );
    }
}

template <class Evaluator>
inline auto retained_workspace_from_evaluator( Evaluator &evaluator, int )
    -> decltype( evaluator.retained_workspace() )
{
    return evaluator.retained_workspace();
}

template <class Evaluator>
inline std::shared_ptr<detail::reusable_workspace_resource>
retained_workspace_from_evaluator( Evaluator &, long )
{
    return std::shared_ptr<detail::reusable_workspace_resource>();
}

template <class Evaluator>
inline auto prepare_evaluator_candidates(
    Evaluator &evaluator, const std::vector<config_map> &candidates, int
) -> decltype( evaluator.prepare_candidates( candidates ), void() )
{
    evaluator.prepare_candidates( candidates );
}

template <class Evaluator>
inline void prepare_evaluator_candidates(
    Evaluator &, const std::vector<config_map> &, long
)
{
}

template <class RuntimeApi, class MPIComm, class Evaluator>
inline selected_3d_config load_or_measure_3d_config(
    const MPIComm &comm, const global_sizes &sizes, const autotune_options &autotune,
    const measured_3d_options &measurement_options, Evaluator &evaluator
)
{
    if ( autotune.cache_file.empty() )
        throw std::logic_error( "fftm::autotune_options::cache_file must not be empty" );

    const hardware_inventory inventory = query_distributed_hardware_inventory<RuntimeApi>( comm );
    int state = 0; // 0: use cache, 1: measure, 2: missing, 3: mismatched, 4: root I/O failure.
    if ( comm.myid == 0 )
    {
        try
        {
            if ( !file_exists( autotune.cache_file ) )
            {
                state = autotune.create_if_missing ? 1 : 2;
            }
            else
            {
                try
                {
                    const auto cached = load_key_value_file( autotune.cache_file );
                    validate_match( cached, comm.num_procs, sizes );
                    if ( autotune.validate_hardware )
                        validate_hardware_match( cached, inventory, autotune );
                    if ( measurement_options.replace_policy_cache && !measured_cache_3d( cached ) )
                        state = 1;
                }
                catch ( const std::exception & )
                {
                    state = autotune.mismatch_policy == cache_mismatch_policy::overwrite ? 1 : 3;
                }
            }
        }
        catch ( const std::exception & )
        {
            state = 4;
        }
    }
    comm.bcast( &state, 1, 0 );

    if ( state == 2 )
        throw std::logic_error( "FFTM autotune cache file does not exist: " + autotune.cache_file );
    if ( state == 4 )
        throw std::logic_error( "Rank 0 failed to inspect FFTM autotune cache: " + autotune.cache_file );
    if ( state == 3 )
    {
        const auto cached = load_key_value_file( autotune.cache_file );
        validate_match( cached, comm.num_procs, sizes );
        if ( autotune.validate_hardware )
            validate_hardware_match( cached, inventory, autotune );
        throw std::logic_error( "FFTM autotune cache is not a valid measured cache" );
    }
    if ( state == 0 )
    {
        const auto cached = load_key_value_file( autotune.cache_file );
        return selected_3d_from_config(
            select_measured_candidate_config( cached, autotune.constraints_3d ),
            comm.num_procs, sizes, "cache"
        );
    }

    const auto candidate_configs = make_measured_3d_candidates(
        comm.num_procs, sizes, measurement_options, homogeneous_ranks_per_node( inventory )
    );
    prepare_evaluator_candidates( evaluator, candidate_configs, 0 );
    std::vector<measured_3d_candidate> candidates;
    candidates.reserve( candidate_configs.size() );
    for ( const auto &candidate_config : candidate_configs )
    {
        measured_3d_candidate candidate;
        candidate.config = candidate_config;
        const auto selected = selected_3d_from_config(
            candidate.config, comm.num_procs, sizes, "candidate"
        );
        if ( measurement_options.continue_on_candidate_error )
        {
            try
            {
                candidate.measurement = evaluator( selected, measurement_options );
            }
            catch ( const std::exception &error )
            {
                candidate.measurement.valid = false;
                candidate.measurement.error = error.what();
            }
        }
        else
        {
            try
            {
                candidate.measurement = evaluator( selected, measurement_options );
            }
            catch ( const std::exception &error )
            {
                throw std::logic_error(
                    "FFTM measured autotune candidate '" +
                    value_or_empty( candidate.config, "FFTM_AUTOTUNE_CANDIDATE_ID" ) +
                    "' failed: " + error.what()
                );
            }
        }
        if ( candidate.measurement.valid &&
             ( !std::isfinite( candidate.measurement.median_wall_ms ) ||
               candidate.measurement.median_wall_ms <= 0.0 ) )
        {
            candidate.measurement.valid = false;
            candidate.measurement.error = "non-positive or non-finite timing";
        }
        candidates.push_back( candidate );
    }

    int winner_index = -1;
    if ( comm.myid == 0 )
    {
        double best = std::numeric_limits<double>::infinity();
        for ( std::size_t i = 0; i < candidates.size(); ++i )
        {
            if ( candidates[i].measurement.valid && candidates[i].measurement.median_wall_ms < best )
            {
                best = candidates[i].measurement.median_wall_ms;
                winner_index = static_cast<int>( i );
            }
        }
    }
    comm.bcast( &winner_index, 1, 0 );
    if ( winner_index < 0 )
        throw std::logic_error( "FFTM measured autotune found no valid 3D candidate" );

    int save_error = 0;
    if ( comm.myid == 0 )
    {
        try
        {
            config_map winner = candidates[static_cast<std::size_t>( winner_index )].config;
            winner.erase( "FFTM_AUTOTUNE_CANDIDATE_ID" );
            const auto signature = make_hardware_signature_record(
                inventory, hardware_transport_policy( winner, comm.num_procs )
            );
            set_hardware_signature_metadata( winner, inventory, signature );
            store_measured_candidates(
                winner, candidates, static_cast<std::size_t>( winner_index ), measurement_options
            );
            save_key_value_file( autotune.cache_file, winner );
        }
        catch ( const std::exception & )
        {
            save_error = 1;
        }
    }
    comm.bcast( &save_error, 1, 0 );
    comm.barrier();
    if ( save_error )
        throw std::logic_error( "Rank 0 failed to save measured FFTM cache: " + autotune.cache_file );

    const auto winner = load_key_value_file( autotune.cache_file );
    validate_match( winner, comm.num_procs, sizes );
    if ( autotune.validate_hardware )
        validate_hardware_match( winner, inventory, autotune );
    selected_3d_config selected = selected_3d_from_config(
        select_measured_candidate_config( winner, autotune.constraints_3d ),
        comm.num_procs, sizes, "measured"
    );
    selected.runtime_workspace = retained_workspace_from_evaluator( evaluator, 0 );
    return selected;
}

template <class BaseFFT, class MPIComm, class Backend, class Log>
class fftm_3d_candidate_evaluator
{
public:
    using runtime_api_t = typename BaseFFT::runtime_api;
    using memory_t = typename runtime_api_t::memory_type;

    fftm_3d_candidate_evaluator( const MPIComm &comm, const global_sizes &sizes, const Log &log = Log() )
        : comm_( comm ), sizes_( sizes ), log_( log ),
          reusable_workspace_( detail::make_reusable_workspace_resource<memory_t>() )
    {
    }

    std::shared_ptr<detail::reusable_workspace_resource> retained_workspace() const
    {
        return reusable_workspace_;
    }

    void prepare_candidates( const std::vector<config_map> &candidate_configs ) const
    {
        std::size_t max_spectrum_bytes = 0;
        for ( const auto &candidate : candidate_configs )
        {
            processor_grid grid;
            fftm_init_options options;
            if ( !apply_config_3d( candidate, comm_.num_procs, sizes_, grid, options ) )
                throw std::logic_error( "FFTM measured candidate cannot be prepared" );

            const std::size_t p1 = grid.p1;
            const std::size_t p2 = grid.p2;
            if ( p1 == 0 || p2 == 0 )
                throw std::logic_error( "FFTM measured candidate has an empty process-grid dimension" );

            const auto spectrum_shape = std::make_tuple(
                sizes_.Nx, ceil_div_( sizes_.Ny, p1 ), ceil_div_( sizes_.Nz / 2 + 1, p2 )
            );
            max_spectrum_bytes = std::max(
                max_spectrum_bytes,
                checked_local_3d_bytes(
                    spectrum_shape, sizeof( typename BaseFFT::complex ), "spectrum pool"
                )
            );
        }

        reusable_workspace_->acquire( detail::workspace_memory_type_token<memory_t>() );
        try
        {
            reusable_workspace_->require_spectrum_size_bytes( max_spectrum_bytes );
            reusable_workspace_->release_lease();
        }
        catch ( ... )
        {
            reusable_workspace_->release_lease();
            throw;
        }
    }

    candidate_measurement operator()(
        const selected_3d_config &selected, const measured_3d_options &options
    ) const
    {
        if ( selected.mode == "p2p-waitany" )
            return evaluate_mode_<mpi_transpose_3d_mode::p2p_waitany>( selected, options );
        if ( selected.mode == "alltoallv" )
            return evaluate_mode_<mpi_transpose_3d_mode::alltoallv>( selected, options );
        throw std::logic_error( "Unsupported measured 3D mode '" + selected.mode + "'" );
    }

private:
    static std::size_t ceil_div_( std::size_t value, std::size_t divisor )
    {
        return value / divisor + ( value % divisor == 0 ? 0 : 1 );
    }

    template <mpi_transpose_3d_mode Mode>
    candidate_measurement evaluate_mode_(
        const selected_3d_config &selected, const measured_3d_options &options
    ) const
    {
        if ( selected.strategy_3d == "slab-pencil" )
            return evaluate_strategy_<strategy_3d_slab_pencil<Mode>>( selected, options );
        if ( selected.strategy_3d == "pencil-slab" )
            return evaluate_strategy_<strategy_3d_pencil_slab<Mode>>( selected, options );
        if ( selected.strategy_3d == "pencil-pencil" )
        {
            if ( selected.init_options.pencil_layout_3d == fftm_3d_pencil_layout::legacy )
                return evaluate_strategy_<strategy_3d_pencil_pencil<Mode, false>>( selected, options );
            return evaluate_strategy_<strategy_3d_pencil_pencil<Mode, true>>( selected, options );
        }
        throw std::logic_error( "Unsupported measured 3D strategy '" + selected.strategy_3d + "'" );
    }

    template <class Strategy>
    candidate_measurement evaluate_strategy_(
        const selected_3d_config &selected, const measured_3d_options &options
    ) const
    {
        collective_memory_boundary_();
        const auto memory_before = runtime_api_t::get_device_memory_info();
        const std::size_t retained_pool_capacity_before =
            reusable_workspace_->retained_device_capacity_bytes();
        log_candidate_memory_( selected, options, "begin" );
        try
        {
            const auto result = evaluate_strategy_body_<Strategy>( selected, options );
            // Device-aware MPI may retain peer device-IPC mappings until every
            // rank has released or quiesced the corresponding allocation and
            // progressed the communicator. Do not sample while peers can
            // still hold a candidate pool registration open.
            collective_memory_boundary_();
            log_candidate_result_( selected, options, result );
            log_candidate_memory_( selected, options, "released" );
            verify_candidate_memory_recovery_(
                selected, options, memory_before, retained_pool_capacity_before
            );
            return result;
        }
        catch ( const std::exception & )
        {
            // The body has already unwound, so this reports memory after the
            // plan lease is released. Reusable buffers intentionally remain.
            try
            {
                log_candidate_memory_( selected, options, "failed-released" );
            }
            catch ( const std::exception & )
            {
                // Never hide the candidate failure with diagnostic telemetry.
            }
            throw;
        }
    }

    template <class Strategy>
    candidate_measurement evaluate_strategy_body_(
        const selected_3d_config &selected, const measured_3d_options &options
    ) const
    {
        using plan_t = fftm<BaseFFT, MPIComm, Backend, Strategy, Log>;
        using real_t = typename plan_t::real;
        using real_array_t = typename plan_t::template real_array_t<3>;
        using complex_array_t = typename plan_t::template complex_array_t<3>;

        selected_3d_config selected_with_runtime = selected;
        selected_with_runtime.runtime_workspace = reusable_workspace_;
        plan_t plan( comm_, log_ );
        init_autotuned_3d_plan( plan, selected_with_runtime, sizes_ );
        const auto input_sizes = plan.get_local_input_sizes();
        const auto output_sizes = plan.get_local_output_sizes();

        real_array_t input;
        complex_array_t spectrum;
        init_autotuned_3d_data_arrays(
            input, spectrum, selected_with_runtime, input_sizes, output_sizes
        );
        log_candidate_memory_( selected, options, "allocated" );

        const std::size_t local_count = static_cast<std::size_t>( input.size() );
        if ( local_count == 0 )
            throw std::logic_error( "FFTM measured candidate has an empty local input partition" );
        std::vector<std::size_t> validation_offsets{ 0, local_count / 2, local_count - 1 };
        std::sort( validation_offsets.begin(), validation_offsets.end() );
        validation_offsets.erase(
            std::unique( validation_offsets.begin(), validation_offsets.end() ), validation_offsets.end()
        );
        std::vector<real_t> validation_values( validation_offsets.size() );
        for ( std::size_t index = 0; index < validation_offsets.size(); ++index )
        {
            validation_values[index] = static_cast<real_t>(
                ( static_cast<double>( comm_.myid + 1 ) * static_cast<double>( index + 1 ) ) / 7.0
            );
        }

        const auto reset_input = [&]() {
            runtime_api_t::memset_zero( input.raw_ptr(), input.size() * sizeof( real_t ) );
            for ( std::size_t index = 0; index < validation_offsets.size(); ++index )
            {
                runtime_api_t::memcpy(
                    input.raw_ptr() + validation_offsets[index], &validation_values[index], sizeof( real_t ),
                    runtime_api_t::host_to_device_kind()
                );
            }
            runtime_api_t::device_synchronize();
        };

        for ( int iteration = 0; iteration < options.warmup; ++iteration )
        {
            reset_input();
            comm_.barrier();
            plan.forward( input, spectrum );
            plan.backward( spectrum, input );
            runtime_api_t::device_synchronize();
        }

        std::vector<double> samples;
        samples.reserve( static_cast<std::size_t>( options.iterations ) );
        for ( int iteration = 0; iteration < options.iterations; ++iteration )
        {
            reset_input();
            comm_.barrier();
            const double start = comm_.wtime();
            plan.forward( input, spectrum );
            plan.backward( spectrum, input );
            runtime_api_t::device_synchronize();
            const double local_ms = ( comm_.wtime() - start ) * 1000.0;
            samples.push_back( comm_.all_reduce_max( local_ms ) );
        }

        const double normalization = static_cast<double>( sizes_.Nx ) * static_cast<double>( sizes_.Ny ) *
                                     static_cast<double>( sizes_.Nz );
        double local_max_relative_error = 0.0;
        for ( std::size_t index = 0; index < validation_offsets.size(); ++index )
        {
            real_t sample = real_t( 0 );
            runtime_api_t::memcpy(
                &sample, input.raw_ptr() + validation_offsets[index], sizeof( real_t ),
                runtime_api_t::device_to_host_kind()
            );
            const double expected = static_cast<double>( validation_values[index] ) * normalization;
            const double error = std::abs( static_cast<double>( sample ) - expected ) /
                                 std::max( std::abs( expected ), 1.0 );
            local_max_relative_error = std::max( local_max_relative_error, error );
        }
        const double max_relative_error = comm_.all_reduce_max( local_max_relative_error );
        const double tolerance = 1000.0 * static_cast<double>( std::numeric_limits<real_t>::epsilon() ) *
                                 std::max( 1.0, std::log2( normalization ) );
        if ( !std::isfinite( max_relative_error ) || max_relative_error > tolerance )
        {
            throw std::logic_error(
                "FFTM measured candidate failed its deterministic round-trip check: relative error=" +
                measured_double_string( max_relative_error ) +
                ", tolerance=" + measured_double_string( tolerance )
            );
        }

        std::sort( samples.begin(), samples.end() );
        const std::size_t middle = samples.size() / 2;
        const double median = samples.size() % 2 == 0 ?
            0.5 * ( samples[middle - 1] + samples[middle] ) : samples[middle];
        candidate_measurement result;
        result.valid = true;
        result.median_wall_ms = median;
        plan.release_resources();
        collective_memory_boundary_();
        log_candidate_memory_( selected, options, "plan-released" );
        return result;
    }

    void collective_memory_boundary_() const
    {
        // The first barrier ensures all peer-visible buffers have been freed.
        // The second gives device-aware MPI another collective progress point
        // after local device teardown before free device memory is queried.
        comm_.barrier();
        runtime_api_t::device_synchronize();
        comm_.barrier();
    }

    void log_candidate_memory_(
        const selected_3d_config &selected, const measured_3d_options &options,
        const char *phase
    ) const
    {
        if ( !options.log_candidate_memory || comm_.myid != 0 )
            return;
        const auto memory = runtime_api_t::get_device_memory_info();
        std::ostringstream out;
        out << "FFTM_CPP_AUTOTUNE_MEMORY candidate="
            << value_or_empty( selected.config, "FFTM_AUTOTUNE_CANDIDATE_ID" )
            << " phase=" << phase;
        if ( memory.free_bytes_known )
            out << " free_mib=" << memory.free_bytes / ( 1024 * 1024 );
        else
            out << " free_mib=unknown";
        if ( memory.total_bytes_known )
            out << " total_mib=" << memory.total_bytes / ( 1024 * 1024 );
        else
            out << " total_mib=unknown";
        out << " workspace_pool_mib="
            << reusable_workspace_->device_capacity_bytes() / ( 1024 * 1024 )
            << " input_pool_mib="
            << reusable_workspace_->input_capacity_bytes() / ( 1024 * 1024 )
            << " spectrum_pool_mib="
            << reusable_workspace_->spectrum_capacity_bytes() / ( 1024 * 1024 )
            << " retained_pool_mib="
            << reusable_workspace_->retained_device_capacity_bytes() / ( 1024 * 1024 )
            << " workspace_pool_leased=" << ( reusable_workspace_->is_leased() ? 1 : 0 );
        log_.info( out.str() );
    }

    void log_candidate_result_(
        const selected_3d_config &selected, const measured_3d_options &options,
        const candidate_measurement &result
    ) const
    {
        if ( !options.log_candidate_memory || comm_.myid != 0 )
            return;
        std::ostringstream out;
        out << "FFTM_CPP_AUTOTUNE_RESULT candidate="
            << value_or_empty( selected.config, "FFTM_AUTOTUNE_CANDIDATE_ID" )
            << " median_ms=" << std::setprecision( 17 ) << result.median_wall_ms;
        log_.info( out.str() );
    }

    template <class DeviceMemoryInfo>
    void verify_candidate_memory_recovery_(
        const selected_3d_config &selected, const measured_3d_options &options,
        const DeviceMemoryInfo &memory_before, std::size_t retained_pool_capacity_before
    ) const
    {
        if ( !options.verify_candidate_memory_recovery || !memory_before.free_bytes_known )
            return;

        const auto memory_after = runtime_api_t::get_device_memory_info();
        if ( !memory_after.free_bytes_known )
            return;

        const std::size_t retained_bytes = memory_before.free_bytes > memory_after.free_bytes
                                               ? memory_before.free_bytes - memory_after.free_bytes
                                               : 0;
        const std::size_t retained_pool_capacity_after =
            reusable_workspace_->retained_device_capacity_bytes();
        const std::size_t retained_pool_growth =
            retained_pool_capacity_after > retained_pool_capacity_before
                ? retained_pool_capacity_after - retained_pool_capacity_before
                : 0;
        const std::size_t unexplained_retained_bytes = retained_bytes > retained_pool_growth
                                                           ? retained_bytes - retained_pool_growth
                                                           : 0;
        const double max_retained_bytes =
            comm_.all_reduce_max( static_cast<double>( retained_bytes ) );
        const double max_retained_pool_growth =
            comm_.all_reduce_max( static_cast<double>( retained_pool_growth ) );
        const double max_unexplained_retained_bytes =
            comm_.all_reduce_max( static_cast<double>( unexplained_retained_bytes ) );

        if ( options.log_candidate_memory && comm_.myid == 0 )
        {
            std::ostringstream recovery;
            recovery << "FFTM_CPP_AUTOTUNE_MEMORY_RECOVERY candidate="
                     << value_or_empty( selected.config, "FFTM_AUTOTUNE_CANDIDATE_ID" )
                     << " retained_mib=" << std::setprecision( 6 )
                     << max_retained_bytes / ( 1024.0 * 1024.0 )
                     << " retained_pool_growth_mib="
                     << max_retained_pool_growth / ( 1024.0 * 1024.0 )
                     << " unexplained_retained_mib="
                     << max_unexplained_retained_bytes / ( 1024.0 * 1024.0 );
            log_.info( recovery.str() );
        }

        if ( max_unexplained_retained_bytes <=
             static_cast<double>( options.candidate_memory_recovery_tolerance_bytes ) )
            return;

        std::ostringstream out;
        out << "FFTM measured autotune candidate '"
            << value_or_empty( selected.config, "FFTM_AUTOTUNE_CANDIDATE_ID" )
            << "' retained " << std::setprecision( 6 )
            << max_unexplained_retained_bytes / ( 1024.0 * 1024.0 )
               << " MiB beyond the reusable workspace and spectrum pool after plan destruction; allowed "
            << static_cast<double>( options.candidate_memory_recovery_tolerance_bytes ) /
                   ( 1024.0 * 1024.0 )
            << " MiB";
        throw std::runtime_error( out.str() );
    }

    const MPIComm &comm_;
    global_sizes   sizes_;
    mutable Log    log_;
    std::shared_ptr<detail::reusable_workspace_resource> reusable_workspace_;
};

} // namespace autotune
} // namespace fftm

#endif
