#ifndef __FFTM_AUTOTUNE_MEASURE_4D_HPP__
#define __FFTM_AUTOTUNE_MEASURE_4D_HPP__

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fftm_autotune_4d.hpp"
#include "fftm_autotune_measure.hpp"

namespace fftm
{
namespace autotune
{

struct measured_4d_options
{
    int         warmup = 1;
    int         iterations = 3;
    bool        include_slab_slab = true;
    bool        include_pencil_pencil = true;
    bool        include_p2p_waitany = true;
    bool        include_alltoallv = true;
    bool        include_slab_credit_fallback = true;
    bool        replace_policy_cache = true;
    bool        continue_on_candidate_error = true;
    bool        log_candidate_memory = false;
    bool        verify_candidate_memory_recovery = true;
    std::size_t candidate_memory_recovery_tolerance_bytes =
        static_cast<std::size_t>( 1024 ) * static_cast<std::size_t>( 1024 ) *
        static_cast<std::size_t>( 1024 );
};

struct measured_4d_candidate
{
    config_map            config;
    candidate_measurement measurement;
};

inline std::string candidate_id_4d( const config_map &config )
{
    return normalize_hardware_token(
        value_or_empty( config, "FFTM_AUTOTUNE_STRATEGY_4D" ) + "_" +
        value_or_empty( config, "FFTM_AUTOTUNE_MODE" ) + "_" +
        value_or_empty( config, "FFTM_AUTOTUNE_GRID_4D" ) + "_" +
        value_or_empty( config, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D" ) + "_" +
        value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_PIPELINE_4D" ) + "_" +
        value_or_empty( config, "FFTM_AUTOTUNE_4D_SLAB_BACKWARD_CREDIT_POLICY" )
    );
}

inline config_map make_measured_4d_candidate(
    int num_procs, const global_sizes &sizes, std::size_t ranks_per_node,
    const autotune_options_4d &autotune, const std::string &strategy,
    const std::string &mode, std::size_t p1, std::size_t p2, std::size_t p3,
    fftm_4d_spectral_layout spectral_layout,
    const std::string &credit_policy = "auto"
)
{
    config_map config = make_default_4d_policy_config(
        num_procs, sizes, ranks_per_node, autotune
    );
    config["FFTM_AUTOTUNE_SOURCE"] = "cpp-measured-4d-candidate";
    config["FFTM_AUTOTUNE_STRATEGY_4D"] = strategy;
    config["FFTM_AUTOTUNE_MODE"] = mode;
    config["FFTM_AUTOTUNE_GRID_4D"] = std::to_string( p1 ) + "x" +
                                        std::to_string( p2 ) + "x" +
                                        std::to_string( p3 );
    config["FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D"] =
        fftm_4d_spectral_layout_name( spectral_layout );
    config["FFTM_AUTOTUNE_ACCEPTED_SPECTRAL_LAYOUTS_4D"] =
        accepted_spectral_layouts_string_4d( autotune );
    config["FFTM_AUTOTUNE_4D_SLAB_BACKWARD_CREDIT_POLICY"] = credit_policy;
    if ( strategy == "pencil-pencil" )
    {
        const fftm_4d_production_topology topology(
            static_cast<std::size_t>( num_procs ),
            ranks_per_node == 0 ? static_cast<std::size_t>( num_procs ) : ranks_per_node,
            p1, p2, p3
        );
        config["FFTM_AUTOTUNE_PENCIL_PIPELINE_4D"] = fftm_4d_pencil_pipeline_name(
            select_production_pencil_pipeline_4d(
                topology, spectral_layout, autotune.device_aware_mpi
            )
        );
    }
    config["FFTM_AUTOTUNE_CANDIDATE_ID"] = candidate_id_4d( config );
    return config;
}

inline void append_unique_measured_4d_candidate(
    std::vector<config_map> &candidates, config_map candidate
)
{
    candidate["FFTM_AUTOTUNE_CANDIDATE_ID"] = candidate_id_4d( candidate );
    const std::string id = candidate["FFTM_AUTOTUNE_CANDIDATE_ID"];
    for ( const auto &existing : candidates )
    {
        if ( value_or_empty( existing, "FFTM_AUTOTUNE_CANDIDATE_ID" ) == id )
            return;
    }
    candidates.push_back( std::move( candidate ) );
}

inline void append_supported_measured_4d_candidate(
    std::vector<config_map> &candidates, config_map candidate, int num_procs,
    const global_sizes &sizes, std::size_t ranks_per_node
)
{
    const selected_4d_config selected = selected_4d_from_config(
        candidate, num_procs, sizes, ranks_per_node, "candidate-preflight"
    );
    if ( selected.init_options.execution.use_4d_native_xw_direct_layout &&
         selected.mode != "p2p-waitany" )
    {
        return;
    }
    append_unique_measured_4d_candidate( candidates, std::move( candidate ) );
}

inline bool slab_credit_policy_is_active_4d( int num_procs )
{
    return num_procs == 16 || num_procs == 24 || num_procs == 32;
}

inline std::tuple<std::size_t, std::size_t, std::size_t> production_pencil_grid_4d(
    int num_procs, std::size_t ranks_per_node
)
{
    const std::size_t count = static_cast<std::size_t>( num_procs );
    if ( ranks_per_node != 0 && ranks_per_node < count && count % ranks_per_node == 0 )
        return std::make_tuple( count / ranks_per_node, ranks_per_node, 1 );
    return std::make_tuple( 1, count, 1 );
}

inline int measured_workspace_priority_4d( const config_map &candidate )
{
    return value_or_empty( candidate, "FFTM_AUTOTUNE_STRATEGY_4D" ) == "pencil-pencil" ? 2 : 1;
}

inline std::vector<config_map> make_measured_4d_candidates(
    int num_procs, const global_sizes &sizes, std::size_t ranks_per_node,
    const autotune_options_4d &autotune, const measured_4d_options &options
)
{
    if ( num_procs <= 0 || sizes.Nw == 0 )
        throw std::logic_error( "FFTM measured 4D autotune requires four dimensions and a positive MPI size" );
    if ( options.warmup < 0 || options.iterations <= 0 )
        throw std::logic_error( "FFTM measured 4D autotune requires warmup >= 0 and iterations > 0" );

    std::vector<std::string> modes;
    if ( options.include_p2p_waitany )
        modes.push_back( "p2p-waitany" );
    if ( options.include_alltoallv && num_procs > 1 )
        modes.push_back( "alltoallv" );
    if ( modes.empty() )
        throw std::logic_error( "FFTM measured 4D autotune has no enabled communication mode" );

    std::vector<config_map> result;
    const auto spectral_layouts = effective_accepted_spectral_layouts_4d( autotune );
    const auto pencil_grid = production_pencil_grid_4d( num_procs, ranks_per_node );
    for ( const auto spectral_layout : spectral_layouts )
    {
        for ( const auto &mode : modes )
        {
            if ( options.include_slab_slab )
            {
                append_supported_measured_4d_candidate(
                    result,
                    make_measured_4d_candidate(
                        num_procs, sizes, ranks_per_node, autotune, "slab-slab", mode,
                        1, static_cast<std::size_t>( num_procs ), 1, spectral_layout
                    ),
                    num_procs, sizes, ranks_per_node
                );
                if ( options.include_slab_credit_fallback && mode == "p2p-waitany" &&
                     slab_credit_policy_is_active_4d( num_procs ) &&
                     spectral_layout == fftm_4d_spectral_layout::native_xzwy )
                {
                    append_supported_measured_4d_candidate(
                        result,
                        make_measured_4d_candidate(
                            num_procs, sizes, ranks_per_node, autotune, "slab-slab", mode,
                            1, static_cast<std::size_t>( num_procs ), 1, spectral_layout,
                            "disabled"
                        ),
                        num_procs, sizes, ranks_per_node
                    );
                }
            }
            if ( options.include_pencil_pencil && num_procs > 1 )
            {
                append_supported_measured_4d_candidate(
                    result,
                    make_measured_4d_candidate(
                        num_procs, sizes, ranks_per_node, autotune, "pencil-pencil", mode,
                        std::get<0>( pencil_grid ), std::get<1>( pencil_grid ),
                        std::get<2>( pencil_grid ), spectral_layout
                    ),
                    num_procs, sizes, ranks_per_node
                );
            }
        }
    }
    std::stable_sort( result.begin(), result.end(), []( const config_map &lhs, const config_map &rhs ) {
        return measured_workspace_priority_4d( lhs ) > measured_workspace_priority_4d( rhs );
    } );
    if ( result.empty() )
        throw std::logic_error( "FFTM measured 4D autotune produced no candidates" );
    return result;
}

inline bool measured_cache_4d( const config_map &config )
{
    return value_or_empty( config, "FFTM_AUTOTUNE_SOURCE" ) == "cpp-measured-4d-v2" &&
           value_or_empty( config, "FFTM_AUTOTUNE_4D_CANDIDATE_POLICY_VERSION" ) == "2" &&
           !value_or_empty( config, "FFTM_AUTOTUNE_WINNER_MEDIAN_MS" ).empty();
}

inline std::string measured_candidate_prefix_4d( std::size_t index )
{
    std::ostringstream prefix;
    prefix << "FFTM_AUTOTUNE_4D_CANDIDATE_" << std::setfill( '0' ) << std::setw( 3 )
           << index << '_';
    return prefix.str();
}

inline bool measured_candidate_matches_4d(
    const config_map &cache, std::size_t index, const autotune_constraints_4d &constraints,
    const std::vector<fftm_4d_spectral_layout> &accepted_spectral_layouts = {}
)
{
    const std::string prefix = measured_candidate_prefix_4d( index );
    const std::string layout_name = value_or_empty( cache, prefix + "SPECTRAL_LAYOUT_4D" );
    const bool layout_accepted = accepted_spectral_layouts.empty() || accepts_spectral_layout_4d(
        accepted_spectral_layouts, parse_spectral_layout_4d( layout_name )
    );
    return layout_accepted && truthy( value_or_empty( cache, prefix + "VALID" ) ) &&
           constraint_matches( constraints.strategy_4d, value_or_empty( cache, prefix + "STRATEGY_4D" ) ) &&
           constraint_matches( constraints.mode, value_or_empty( cache, prefix + "MODE" ) ) &&
           constraint_matches( constraints.backend_4d, value_or_empty( cache, prefix + "BACKEND_4D" ) ) &&
           constraint_matches( constraints.grid_4d, value_or_empty( cache, prefix + "GRID_4D" ) ) &&
           constraint_matches(
               constraints.spectral_layout_4d, value_or_empty( cache, prefix + "SPECTRAL_LAYOUT_4D" )
           ) &&
           constraint_matches(
               constraints.pencil_pipeline_4d, value_or_empty( cache, prefix + "PENCIL_PIPELINE_4D" )
           );
}

inline config_map select_measured_candidate_config_4d(
    const config_map &cache, const autotune_constraints_4d &constraints,
    const std::vector<fftm_4d_spectral_layout> &accepted_spectral_layouts = {}
)
{
    if ( !measured_cache_4d( cache ) )
    {
        validate_config_constraints_4d( cache, constraints, "policy" );
        return cache;
    }

    const std::size_t count = static_cast<std::size_t>( std::strtoull(
        value_or_empty( cache, "FFTM_AUTOTUNE_CANDIDATE_COUNT" ).c_str(), nullptr, 10
    ) );
    std::size_t best_index = count;
    double best_ms = std::numeric_limits<double>::infinity();
    for ( std::size_t index = 0; index < count; ++index )
    {
        if ( !measured_candidate_matches_4d(
                 cache, index, constraints, accepted_spectral_layouts
             ) )
            continue;
        const std::string prefix = measured_candidate_prefix_4d( index );
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
        throw std::logic_error( "No measured FFTM 4D candidate satisfies the requested constraints" );

    const std::string prefix = measured_candidate_prefix_4d( best_index );
    config_map selected = cache;
    selected["FFTM_AUTOTUNE_STRATEGY_4D"] = value_or_empty( cache, prefix + "STRATEGY_4D" );
    selected["FFTM_AUTOTUNE_MODE"] = value_or_empty( cache, prefix + "MODE" );
    selected["FFTM_AUTOTUNE_BACKEND_4D"] = value_or_empty( cache, prefix + "BACKEND_4D" );
    selected["FFTM_AUTOTUNE_GRID_4D"] = value_or_empty( cache, prefix + "GRID_4D" );
    selected["FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D"] =
        value_or_empty( cache, prefix + "SPECTRAL_LAYOUT_4D" );
    selected["FFTM_AUTOTUNE_PENCIL_PIPELINE_4D"] =
        value_or_empty( cache, prefix + "PENCIL_PIPELINE_4D" );
    selected["FFTM_AUTOTUNE_4D_SLAB_BACKWARD_CREDIT_POLICY"] =
        value_or_empty( cache, prefix + "SLAB_BACKWARD_CREDIT_POLICY" );
    selected["FFTM_AUTOTUNE_ACTIVE_CANDIDATE_INDEX"] = std::to_string( best_index );
    selected["FFTM_AUTOTUNE_ACTIVE_MEDIAN_MS"] = measured_double_string( best_ms );
    return selected;
}

inline void store_measured_candidates_4d(
    config_map &winner, const std::vector<measured_4d_candidate> &candidates,
    std::size_t winner_index, const measured_4d_options &options
)
{
    winner["FFTM_AUTOTUNE_SOURCE"] = "cpp-measured-4d-v2";
    winner["FFTM_AUTOTUNE_4D_CANDIDATE_POLICY_VERSION"] = "2";
    winner["FFTM_AUTOTUNE_MEASURED"] = "1";
    winner["FFTM_AUTOTUNE_MEASURE_WARMUP"] = std::to_string( options.warmup );
    winner["FFTM_AUTOTUNE_MEASURE_ITERATIONS"] = std::to_string( options.iterations );
    winner["FFTM_AUTOTUNE_CANDIDATE_COUNT"] = std::to_string( candidates.size() );
    winner["FFTM_AUTOTUNE_WINNER_INDEX"] = std::to_string( winner_index );
    winner["FFTM_AUTOTUNE_WINNER_MEDIAN_MS"] =
        measured_double_string( candidates[winner_index].measurement.median_wall_ms );

    for ( std::size_t index = 0; index < candidates.size(); ++index )
    {
        const std::string prefix = measured_candidate_prefix_4d( index );
        const auto &candidate = candidates[index];
        winner[prefix + "ID"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_CANDIDATE_ID" );
        winner[prefix + "VALID"] = candidate.measurement.valid ? "1" : "0";
        winner[prefix + "MEDIAN_MS"] = candidate.measurement.valid ?
            measured_double_string( candidate.measurement.median_wall_ms ) : "nan";
        winner[prefix + "STRATEGY_4D"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_STRATEGY_4D" );
        winner[prefix + "MODE"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_MODE" );
        winner[prefix + "BACKEND_4D"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_BACKEND_4D" );
        winner[prefix + "GRID_4D"] = value_or_empty( candidate.config, "FFTM_AUTOTUNE_GRID_4D" );
        winner[prefix + "SPECTRAL_LAYOUT_4D"] =
            value_or_empty( candidate.config, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D" );
        winner[prefix + "PENCIL_PIPELINE_4D"] =
            value_or_empty( candidate.config, "FFTM_AUTOTUNE_PENCIL_PIPELINE_4D" );
        winner[prefix + "SLAB_BACKWARD_CREDIT_POLICY"] =
            value_or_empty( candidate.config, "FFTM_AUTOTUNE_4D_SLAB_BACKWARD_CREDIT_POLICY" );
        if ( !candidate.measurement.error.empty() )
            winner[prefix + "ERROR"] = normalize_hardware_token( candidate.measurement.error );
    }
}

template <class RuntimeApi, class MPIComm, class Evaluator>
inline selected_4d_config load_or_measure_4d_config(
    const MPIComm &comm, const global_sizes &sizes, const autotune_options_4d &autotune,
    const measured_4d_options &measurement_options, Evaluator &evaluator
)
{
    if ( autotune.cache_file.empty() )
        throw std::logic_error( "fftm::autotune_options_4d::cache_file must not be empty" );

    const hardware_inventory inventory = query_distributed_hardware_inventory<RuntimeApi>( comm );
    const std::size_t ranks_per_node = homogeneous_ranks_per_node( inventory );
    int state = 0; // 0: cache, 1: measure, 2: missing, 3: mismatch/I/O.
    if ( comm.myid == 0 )
    {
        try
        {
            if ( !file_exists( autotune.cache_file ) )
                state = autotune.create_if_missing ? 1 : 2;
            else
            {
                try
                {
                    const config_map cached = load_key_value_file( autotune.cache_file );
                    validate_match_4d( cached, comm.num_procs, sizes );
                    validate_device_aware_mpi_request(
                        cached, autotune.device_aware_mpi, "measured 4D"
                    );
                    if ( measured_cache_4d( cached ) )
                        validate_spectral_layout_request_4d( cached, autotune, true );
                    if ( autotune.validate_hardware )
                        validate_hardware_match( cached, inventory, autotune );
                    if ( measurement_options.replace_policy_cache && !measured_cache_4d( cached ) )
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
            state = 3;
        }
    }
    comm.bcast( &state, 1, 0 );
    if ( state == 2 )
        throw std::logic_error( "FFTM 4D autotune cache file does not exist: " + autotune.cache_file );
    if ( state == 3 )
        throw std::logic_error( "FFTM 4D autotune cache is invalid or mismatched: " + autotune.cache_file );
    if ( state == 0 )
    {
        const config_map cached = load_key_value_file( autotune.cache_file );
        const auto accepted = effective_accepted_spectral_layouts_4d( autotune );
        validate_device_aware_mpi_request(
            cached, autotune.device_aware_mpi, "measured 4D"
        );
        validate_spectral_layout_request_4d( cached, autotune, measured_cache_4d( cached ) );
        return selected_4d_from_config(
            select_measured_candidate_config_4d(
                cached, autotune.constraints_4d, accepted
            ),
            comm.num_procs, sizes, ranks_per_node, "cache"
        );
    }

    const auto candidate_configs = make_measured_4d_candidates(
        comm.num_procs, sizes, ranks_per_node, autotune, measurement_options
    );
    prepare_evaluator_candidates( evaluator, candidate_configs, 0 );
    std::vector<measured_4d_candidate> candidates;
    candidates.reserve( candidate_configs.size() );
    for ( const auto &candidate_config : candidate_configs )
    {
        measured_4d_candidate candidate;
        candidate.config = candidate_config;
        const selected_4d_config selected = selected_4d_from_config(
            candidate.config, comm.num_procs, sizes, ranks_per_node, "candidate"
        );
        try
        {
            candidate.measurement = evaluator( selected, measurement_options );
        }
        catch ( const std::exception &error )
        {
            if ( !measurement_options.continue_on_candidate_error )
            {
                throw std::logic_error(
                    "FFTM measured 4D candidate '" +
                    value_or_empty( candidate.config, "FFTM_AUTOTUNE_CANDIDATE_ID" ) +
                    "' failed: " + error.what()
                );
            }
            candidate.measurement.valid = false;
            candidate.measurement.error = error.what();
        }
        const int globally_valid = comm.all_reduce_min( candidate.measurement.valid ? 1 : 0 );
        candidate.measurement.valid = globally_valid != 0;
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
        for ( std::size_t index = 0; index < candidates.size(); ++index )
        {
            if ( candidates[index].measurement.valid &&
                 candidates[index].measurement.median_wall_ms < best )
            {
                best = candidates[index].measurement.median_wall_ms;
                winner_index = static_cast<int>( index );
            }
        }
    }
    comm.bcast( &winner_index, 1, 0 );
    if ( winner_index < 0 )
        throw std::logic_error( "FFTM measured 4D autotune found no valid candidate" );

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
            store_measured_candidates_4d(
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
        throw std::logic_error( "Rank 0 failed to save measured FFTM 4D cache: " + autotune.cache_file );

    const config_map winner = load_key_value_file( autotune.cache_file );
    validate_match_4d( winner, comm.num_procs, sizes );
    validate_device_aware_mpi_request(
        winner, autotune.device_aware_mpi, "measured 4D"
    );
    validate_spectral_layout_request_4d( winner, autotune, true );
    if ( autotune.validate_hardware )
        validate_hardware_match( winner, inventory, autotune );
    selected_4d_config selected = selected_4d_from_config(
        select_measured_candidate_config_4d(
            winner, autotune.constraints_4d,
            effective_accepted_spectral_layouts_4d( autotune )
        ),
        comm.num_procs, sizes, ranks_per_node, "measured"
    );
    selected.runtime_workspace = retained_workspace_from_evaluator( evaluator, 0 );
    return selected;
}

template <class BaseFFT, class MPIComm, class Backend, class Log>
class fftm_4d_candidate_evaluator
{
public:
    using runtime_api_t = typename BaseFFT::runtime_api;
    using memory_t = typename runtime_api_t::memory_type;

    fftm_4d_candidate_evaluator(
        const MPIComm &comm, const global_sizes &sizes, const Log &log = Log()
    )
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
        std::size_t max_input_bytes = 0;
        std::size_t max_spectrum_bytes = 0;
        for ( const auto &candidate : candidate_configs )
        {
            const processor_grid grid = parse_selected_grid_4d( candidate, comm_.num_procs );
            fft_partitioning<MPIComm> partitioning( comm_ );
            partitioning.init( grid, sizes_ );
            int i = 0, j = 0, k = 0;
            std::tie( i, j, k ) = partitioning.get_my_grid();
            partition input, t1, t2, output;
            std::tie( input, t1, t2, output ) = partitioning.get_partitioning_4D();
            const auto input_shape = std::make_tuple(
                input.size_x[static_cast<std::size_t>( i )],
                input.size_y[static_cast<std::size_t>( j )],
                input.size_z[static_cast<std::size_t>( k )], sizes_.Nw
            );
            const auto layout = parse_spectral_layout_4d(
                value_or_empty( candidate, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D" )
            );
            const bool pencil = value_or_empty(
                candidate, "FFTM_AUTOTUNE_STRATEGY_4D"
            ) == "pencil-pencil";
            const auto spectrum_shape = layout == fftm_4d_spectral_layout::public_yzwx
                ? std::make_tuple(
                      output.size_y[static_cast<std::size_t>( i )],
                      output.size_z[static_cast<std::size_t>( j )],
                      output.size_w[static_cast<std::size_t>( k )], output.size_x[0]
                  )
                : pencil
                      ? std::make_tuple(
                            output.size_x[0], output.size_z[static_cast<std::size_t>( j )],
                            output.size_w[static_cast<std::size_t>( k )],
                            output.size_y[static_cast<std::size_t>( i )]
                        )
                      : std::make_tuple(
                            t2.size_x[static_cast<std::size_t>( i )],
                            t2.size_z[static_cast<std::size_t>( j )],
                            t2.size_w[static_cast<std::size_t>( k )], t2.size_y[0]
                        );
            max_input_bytes = std::max(
                max_input_bytes,
                checked_local_4d_bytes( input_shape, sizeof( typename BaseFFT::real ), "input pool" )
            );
            max_spectrum_bytes = std::max(
                max_spectrum_bytes,
                checked_local_4d_bytes(
                    spectrum_shape, sizeof( typename BaseFFT::complex ), "spectrum pool"
                )
            );
        }
        reusable_workspace_->acquire( detail::workspace_memory_type_token<memory_t>() );
        try
        {
            reusable_workspace_->require_input_size_bytes( max_input_bytes );
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
        const selected_4d_config &selected, const measured_4d_options &options
    ) const
    {
        if ( selected.mode == "p2p-waitany" )
            return evaluate_mode_<mpi_transpose_3d_mode::p2p_waitany>( selected, options );
        if ( selected.mode == "alltoallv" )
            return evaluate_mode_<mpi_transpose_3d_mode::alltoallv>( selected, options );
        throw std::logic_error( "Unsupported measured 4D mode '" + selected.mode + "'" );
    }

private:
    template <mpi_transpose_3d_mode Mode>
    candidate_measurement evaluate_mode_(
        const selected_4d_config &selected, const measured_4d_options &options
    ) const
    {
        if ( selected.strategy_4d == "slab-slab" )
            return evaluate_strategy_<strategy_4d_slab_slab_mpi<Mode>>( selected, options );
        if ( selected.strategy_4d == "pencil-pencil" )
            return evaluate_strategy_<strategy_4d_pencil_pencil_mpi<Mode>>( selected, options );
        throw std::logic_error( "Unsupported measured 4D strategy '" + selected.strategy_4d + "'" );
    }

    template <class Strategy4D>
    candidate_measurement evaluate_strategy_(
        const selected_4d_config &selected, const measured_4d_options &options
    ) const
    {
        collective_memory_boundary_();
        const auto memory_before = runtime_api_t::get_device_memory_info();
        const std::size_t retained_before = reusable_workspace_->retained_device_capacity_bytes();
        log_candidate_memory_( selected, options, "begin" );
        const candidate_measurement result = evaluate_strategy_body_<Strategy4D>( selected, options );
        collective_memory_boundary_();
        log_candidate_memory_( selected, options, "released" );
        verify_candidate_memory_recovery_( selected, options, memory_before, retained_before );
        return result;
    }

    template <class Strategy4D>
    candidate_measurement evaluate_strategy_body_(
        const selected_4d_config &selected, const measured_4d_options &options
    ) const
    {
        using plan_t = fftm<
            BaseFFT, MPIComm, Backend,
            strategy_3d_slab_pencil<mpi_transpose_3d_mode::alltoallv>, Log, Strategy4D>;
        using real_t = typename plan_t::real;
        using real_array_t = typename plan_t::template real_array_t<4>;
        using complex_array_t = typename plan_t::template complex_array_t<4>;

        selected_4d_config selected_with_runtime = selected;
        selected_with_runtime.runtime_workspace = reusable_workspace_;
        plan_t plan( comm_, log_ );
        init_autotuned_4d_plan( plan, selected_with_runtime, sizes_ );
        const auto input_sizes = plan.get_local_input_sizes_4d();
        const auto spectrum_sizes = plan.get_local_spectral_sizes_4d();
        real_array_t input;
        complex_array_t spectrum;
        init_autotuned_4d_data_arrays(
            input, spectrum, selected_with_runtime, input_sizes, spectrum_sizes
        );
        log_candidate_memory_( selected, options, "allocated" );

        const std::size_t local_count = static_cast<std::size_t>( input.size() );
        if ( local_count == 0 )
            throw std::logic_error( "FFTM measured 4D candidate has an empty local input partition" );
        std::vector<std::size_t> offsets{ 0, local_count / 2, local_count - 1 };
        std::sort( offsets.begin(), offsets.end() );
        offsets.erase( std::unique( offsets.begin(), offsets.end() ), offsets.end() );
        std::vector<real_t> values( offsets.size() );
        for ( std::size_t index = 0; index < offsets.size(); ++index )
        {
            values[index] = static_cast<real_t>(
                ( static_cast<double>( comm_.myid + 1 ) * static_cast<double>( index + 1 ) ) / 11.0
            );
        }

        const auto reset_input = [&]() {
            runtime_api_t::memset_zero( input.raw_ptr(), input.size() * sizeof( real_t ) );
            for ( std::size_t index = 0; index < offsets.size(); ++index )
            {
                runtime_api_t::memcpy(
                    input.raw_ptr() + offsets[index], &values[index], sizeof( real_t ),
                    runtime_api_t::host_to_device_kind()
                );
            }
            runtime_api_t::device_synchronize();
        };
        const auto execute_pair = [&]() {
            if ( plan.uses_native_spectral_layout_4d() )
            {
                auto native = plan.make_native_spectral_view_4d( spectrum );
                plan.forward_native_spectral_4d( input, native );
                plan.backward_native_spectral_4d( native, input );
            }
            else
            {
                plan.forward( input, spectrum );
                plan.backward( spectrum, input );
            }
        };

        for ( int iteration = 0; iteration < options.warmup; ++iteration )
        {
            reset_input();
            comm_.barrier();
            execute_pair();
            runtime_api_t::device_synchronize();
        }

        std::vector<double> samples;
        samples.reserve( static_cast<std::size_t>( options.iterations ) );
        for ( int iteration = 0; iteration < options.iterations; ++iteration )
        {
            reset_input();
            comm_.barrier();
            const double start = comm_.wtime();
            execute_pair();
            runtime_api_t::device_synchronize();
            samples.push_back( comm_.all_reduce_max( ( comm_.wtime() - start ) * 1000.0 ) );
        }

        const double normalization = static_cast<double>( sizes_.Nx ) *
                                     static_cast<double>( sizes_.Ny ) *
                                     static_cast<double>( sizes_.Nz ) *
                                     static_cast<double>( sizes_.Nw );
        double local_error = 0.0;
        for ( std::size_t index = 0; index < offsets.size(); ++index )
        {
            real_t sample = real_t( 0 );
            runtime_api_t::memcpy(
                &sample, input.raw_ptr() + offsets[index], sizeof( real_t ),
                runtime_api_t::device_to_host_kind()
            );
            const double expected = static_cast<double>( values[index] ) * normalization;
            local_error = std::max(
                local_error, std::abs( static_cast<double>( sample ) - expected ) /
                                 std::max( std::abs( expected ), 1.0 )
            );
        }
        const double max_error = comm_.all_reduce_max( local_error );
        const double tolerance = 2000.0 * static_cast<double>( std::numeric_limits<real_t>::epsilon() ) *
                                 std::max( 1.0, std::log2( normalization ) );
        if ( !std::isfinite( max_error ) || max_error > tolerance )
        {
            throw std::logic_error(
                "FFTM measured 4D candidate failed its round-trip check: relative error=" +
                measured_double_string( max_error ) + ", tolerance=" + measured_double_string( tolerance )
            );
        }

        std::sort( samples.begin(), samples.end() );
        const std::size_t middle = samples.size() / 2;
        candidate_measurement result;
        result.valid = true;
        result.median_wall_ms = samples.size() % 2 == 0 ?
            0.5 * ( samples[middle - 1] + samples[middle] ) : samples[middle];
        plan.release_resources();
        collective_memory_boundary_();
        return result;
    }

    void collective_memory_boundary_() const
    {
        comm_.barrier();
        runtime_api_t::device_synchronize();
        comm_.barrier();
    }

    void log_candidate_memory_(
        const selected_4d_config &selected, const measured_4d_options &options,
        const char *phase
    ) const
    {
        if ( !options.log_candidate_memory || comm_.myid != 0 )
            return;
        const auto memory = runtime_api_t::get_device_memory_info();
        std::ostringstream out;
        out << "FFTM_CPP_AUTOTUNE_4D_MEMORY candidate="
            << value_or_empty( selected.config, "FFTM_AUTOTUNE_CANDIDATE_ID" )
            << " phase=" << phase << " free_mib=";
        if ( memory.free_bytes_known )
            out << memory.free_bytes / ( 1024 * 1024 );
        else
            out << "unknown";
        out << " retained_pool_mib="
            << reusable_workspace_->retained_device_capacity_bytes() / ( 1024 * 1024 );
        log_.info( out.str() );
    }

    template <class DeviceMemoryInfo>
    void verify_candidate_memory_recovery_(
        const selected_4d_config &selected, const measured_4d_options &options,
        const DeviceMemoryInfo &memory_before, std::size_t retained_before
    ) const
    {
        if ( !options.verify_candidate_memory_recovery || !memory_before.free_bytes_known )
            return;
        const auto memory_after = runtime_api_t::get_device_memory_info();
        if ( !memory_after.free_bytes_known )
            return;
        const std::size_t retained = memory_before.free_bytes > memory_after.free_bytes ?
            memory_before.free_bytes - memory_after.free_bytes : 0;
        const std::size_t retained_after = reusable_workspace_->retained_device_capacity_bytes();
        const std::size_t pool_growth = retained_after > retained_before ?
            retained_after - retained_before : 0;
        const std::size_t unexplained = retained > pool_growth ? retained - pool_growth : 0;
        const double max_unexplained = comm_.all_reduce_max( static_cast<double>( unexplained ) );
        if ( max_unexplained >
             static_cast<double>( options.candidate_memory_recovery_tolerance_bytes ) )
        {
            throw std::runtime_error(
                "FFTM measured 4D candidate '" +
                value_or_empty( selected.config, "FFTM_AUTOTUNE_CANDIDATE_ID" ) +
                "' retained unexplained device memory after release"
            );
        }
    }

    const MPIComm &comm_;
    global_sizes sizes_;
    mutable Log log_;
    std::shared_ptr<detail::reusable_workspace_resource> reusable_workspace_;
};

} // namespace autotune
} // namespace fftm

#endif
