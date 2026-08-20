#ifndef __FFTM_AUTOTUNE_4D_HPP__
#define __FFTM_AUTOTUNE_4D_HPP__

#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include "fftm_autotune.hpp"

namespace fftm
{
namespace autotune
{

struct autotune_constraints_4d
{
    std::string strategy_4d;
    std::string mode;
    std::string backend_4d;
    std::string grid_4d;
    std::string spectral_layout_4d;
    std::string pencil_pipeline_4d;
};

struct autotune_options_4d : autotune_options
{
    autotune_constraints_4d constraints_4d;
    fftm_4d_spectral_layout requested_spectral_layout = fftm_4d_spectral_layout::public_yzwx;
    bool                    device_aware_mpi = true;
};

struct selected_4d_config
{
    config_map        config;
    processor_grid    grid;
    fftm_init_options init_options;
    std::string       strategy_4d;
    std::string       mode;
    std::string       backend_4d;
    std::string       source;
    std::shared_ptr<detail::reusable_workspace_resource> runtime_workspace;
};

inline const char *current_4d_policy_version()
{
    return "1";
}

inline const char *current_4d_policy_source()
{
    return "cpp-4d-policy-cache-v1";
}

inline std::string sizes_to_string_4d( const global_sizes &sizes )
{
    return std::to_string( sizes.Nx ) + "x" + std::to_string( sizes.Ny ) + "x" +
           std::to_string( sizes.Nz ) + "x" + std::to_string( sizes.Nw );
}

inline std::string grid_to_string_4d( const processor_grid &grid )
{
    return std::to_string( grid.p1 ) + "x" + std::to_string( grid.p2 ) + "x" +
           std::to_string( grid.p3 );
}

inline bool parse_grid_token_4d(
    const std::string &token, std::size_t &p1, std::size_t &p2, std::size_t &p3
)
{
    const std::vector<std::string> dims = split( token, 'x' );
    if ( dims.size() != 3 )
        return false;
    p1 = static_cast<std::size_t>( std::strtoull( dims[0].c_str(), nullptr, 10 ) );
    p2 = static_cast<std::size_t>( std::strtoull( dims[1].c_str(), nullptr, 10 ) );
    p3 = static_cast<std::size_t>( std::strtoull( dims[2].c_str(), nullptr, 10 ) );
    return p1 != 0 && p2 != 0 && p3 != 0;
}

inline transform_strategy_4d_mpi parse_strategy_4d( const std::string &value )
{
    if ( value == "slab-slab" )
        return transform_strategy_4d_mpi::slab_slab;
    if ( value == "pencil-pencil" )
        return transform_strategy_4d_mpi::pencil_pencil;
    throw std::logic_error( "Invalid FFTM 4D strategy '" + value + "'" );
}

inline fftm_4d_spectral_layout parse_spectral_layout_4d( const std::string &value )
{
    if ( value == "public-yzwx" )
        return fftm_4d_spectral_layout::public_yzwx;
    if ( value == "native-xzwy" )
        return fftm_4d_spectral_layout::native_xzwy;
    throw std::logic_error( "Invalid FFTM 4D spectral layout '" + value + "'" );
}

inline fftm_4d_pencil_pipeline parse_pencil_pipeline_4d( const std::string &value )
{
    if ( value.empty() || value == "auto" )
        return fftm_4d_pencil_pipeline::auto_select;
    if ( value == "standard" )
        return fftm_4d_pencil_pipeline::standard;
    if ( value == "node-aligned-wz" )
        return fftm_4d_pencil_pipeline::node_aligned_wz;
    throw std::logic_error( "Invalid FFTM 4D pencil pipeline '" + value + "'" );
}

inline void validate_config_constraints_4d(
    const config_map &config, const autotune_constraints_4d &constraints,
    const std::string &cache_kind
)
{
    if ( !constraint_matches( constraints.strategy_4d, value_or_empty( config, "FFTM_AUTOTUNE_STRATEGY_4D" ) ) ||
         !constraint_matches( constraints.mode, value_or_empty( config, "FFTM_AUTOTUNE_MODE" ) ) ||
         !constraint_matches( constraints.backend_4d, value_or_empty( config, "FFTM_AUTOTUNE_BACKEND_4D" ) ) ||
         !constraint_matches( constraints.grid_4d, value_or_empty( config, "FFTM_AUTOTUNE_GRID_4D" ) ) ||
         !constraint_matches(
             constraints.spectral_layout_4d,
             value_or_empty( config, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D" )
         ) ||
         !constraint_matches(
             constraints.pencil_pipeline_4d,
             value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_PIPELINE_4D" )
         ) )
    {
        throw std::logic_error(
            "FFTM " + cache_kind + " cache does not satisfy the requested 4D constraints"
        );
    }
}

inline void validate_match_4d(
    const config_map &config, int num_procs, const global_sizes &sizes
)
{
    if ( sizes.Nw == 0 )
        throw std::logic_error( "FFTM 4D autotune requires a nonzero fourth dimension" );

    const std::string dimension = value_or_empty( config, "FFTM_AUTOTUNE_DIM" );
    if ( dimension != "4" )
        throw std::logic_error( "FFTM autotune cache is not a 4D cache" );

    const std::string expected_gpus = value_or_empty( config, "FFTM_AUTOTUNE_NUM_GPUS" );
    if ( expected_gpus.empty() || std::atoi( expected_gpus.c_str() ) != num_procs )
    {
        throw std::logic_error(
            "FFTM 4D autotune cache GPU count '" + expected_gpus + "' does not match MPI size " +
            std::to_string( num_procs )
        );
    }

    const std::string expected_size = value_or_empty( config, "FFTM_AUTOTUNE_SIZE_4D" );
    const std::vector<std::string> dims = split( expected_size, 'x' );
    if ( dims.size() != 4 )
        throw std::logic_error( "Invalid FFTM_AUTOTUNE_SIZE_4D='" + expected_size + "'" );
    const std::size_t nx = static_cast<std::size_t>( std::strtoull( dims[0].c_str(), nullptr, 10 ) );
    const std::size_t ny = static_cast<std::size_t>( std::strtoull( dims[1].c_str(), nullptr, 10 ) );
    const std::size_t nz = static_cast<std::size_t>( std::strtoull( dims[2].c_str(), nullptr, 10 ) );
    const std::size_t nw = static_cast<std::size_t>( std::strtoull( dims[3].c_str(), nullptr, 10 ) );
    if ( nx != sizes.Nx || ny != sizes.Ny || nz != sizes.Nz || nw != sizes.Nw )
    {
        throw std::logic_error(
            "FFTM 4D autotune cache size " + expected_size + " does not match requested size " +
            sizes_to_string_4d( sizes )
        );
    }

    const std::string source = value_or_empty( config, "FFTM_AUTOTUNE_SOURCE" );
    if ( source.compare( 0, 20, "cpp-4d-policy-cache-" ) == 0 )
    {
        const std::string version = value_or_empty( config, "FFTM_AUTOTUNE_4D_POLICY_VERSION" );
        if ( version != current_4d_policy_version() )
        {
            throw std::logic_error(
                "FFTM 4D policy cache version " +
                ( version.empty() ? std::string( "missing" ) : version ) +
                " does not match current version " + current_4d_policy_version()
            );
        }
    }
}

inline processor_grid parse_selected_grid_4d( const config_map &config, int num_procs )
{
    std::size_t p1 = 0, p2 = 0, p3 = 0;
    const std::string value = value_or_empty( config, "FFTM_AUTOTUNE_GRID_4D" );
    if ( !parse_grid_token_4d( value, p1, p2, p3 ) )
        throw std::logic_error( "Invalid FFTM_AUTOTUNE_GRID_4D='" + value + "'" );
    if ( p1 > static_cast<std::size_t>( num_procs ) / p2 ||
         p1 * p2 > static_cast<std::size_t>( num_procs ) / p3 ||
         p1 * p2 * p3 != static_cast<std::size_t>( num_procs ) )
    {
        throw std::logic_error( "FFTM_AUTOTUNE_GRID_4D does not match the MPI size" );
    }
    processor_grid grid;
    grid.init( p1, p2, p3 );
    return grid;
}

inline bool apply_config_4d(
    const config_map &config, int num_procs, const global_sizes &sizes,
    std::size_t ranks_per_node, processor_grid &grid, fftm_init_options &options
)
{
    if ( config.empty() )
        return false;
    validate_match_4d( config, num_procs, sizes );

    grid = parse_selected_grid_4d( config, num_procs );
    const transform_strategy_4d_mpi strategy =
        parse_strategy_4d( value_or_empty( config, "FFTM_AUTOTUNE_STRATEGY_4D" ) );
    const fftm_4d_spectral_layout layout =
        parse_spectral_layout_4d( value_or_empty( config, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D" ) );
    const bool device_aware = bool_value( config, "FFTM_DIRECT_P2P_CUDA_AWARE", true );
    const fftm_4d_pencil_pipeline pipeline = parse_pencil_pipeline_4d(
        value_or_empty( config, "FFTM_AUTOTUNE_PENCIL_PIPELINE_4D" )
    );
    const std::string credit_policy =
        value_or_empty( config, "FFTM_AUTOTUNE_4D_SLAB_BACKWARD_CREDIT_POLICY" );
    fftm_4d_slab_backward_credit_policy credit = fftm_4d_slab_backward_credit_policy::auto_select;
    if ( credit_policy == "disabled" )
        credit = fftm_4d_slab_backward_credit_policy::disabled;
    else if ( !credit_policy.empty() && credit_policy != "auto" )
        throw std::logic_error( "Invalid FFTM 4D slab backward-credit policy '" + credit_policy + "'" );

    const fftm_4d_production_topology topology(
        static_cast<std::size_t>( num_procs ), ranks_per_node == 0 ? static_cast<std::size_t>( num_procs ) : ranks_per_node,
        grid.p1, grid.p2, grid.p3, pipeline, credit
    );
    options = production_options_4d( strategy, layout, device_aware, topology );
    return true;
}

inline config_map make_default_4d_policy_config(
    int num_procs, const global_sizes &sizes, std::size_t ranks_per_node,
    const autotune_options_4d &autotune
)
{
    if ( num_procs <= 0 || sizes.Nw == 0 )
        throw std::logic_error( "FFTM 4D policy requires a positive MPI size and four dimensions" );

    config_map config;
    config["FFTM_AUTOTUNE_SCHEMA"] = "2";
    config["FFTM_AUTOTUNE_4D_POLICY_VERSION"] = current_4d_policy_version();
    config["FFTM_AUTOTUNE_LIBRARY"] = "fftm";
    config["FFTM_AUTOTUNE_DIM"] = "4";
    config["FFTM_AUTOTUNE_NUM_GPUS"] = std::to_string( num_procs );
    config["FFTM_AUTOTUNE_SIZE_4D"] = sizes_to_string_4d( sizes );
    config["FFTM_AUTOTUNE_SOURCE"] = current_4d_policy_source();
    config["FFTM_AUTOTUNE_BACKEND_4D"] = "native";
    config["FFTM_AUTOTUNE_STRATEGY_4D"] = "slab-slab";
    config["FFTM_AUTOTUNE_MODE"] = "p2p-waitany";
    config["FFTM_AUTOTUNE_GRID_4D"] = "1x" + std::to_string( num_procs ) + "x1";
    config["FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D"] =
        fftm_4d_spectral_layout_name( autotune.requested_spectral_layout );
    config["FFTM_AUTOTUNE_PENCIL_PIPELINE_4D"] = "auto";
    config["FFTM_AUTOTUNE_4D_SLAB_BACKWARD_CREDIT_POLICY"] = "auto";
    config["FFTM_DIRECT_P2P_CUDA_AWARE"] = autotune.device_aware_mpi ? "1" : "0";
    config["FFTM_USE_P2P_BYTE_TRANSFER"] = autotune.device_aware_mpi ? "1" : "0";
    (void)ranks_per_node;
    return config;
}

inline config_map make_default_4d_config(
    int num_procs, const global_sizes &sizes, const hardware_inventory &inventory,
    const autotune_options_4d &autotune
)
{
    config_map config = make_default_4d_policy_config(
        num_procs, sizes, homogeneous_ranks_per_node( inventory ), autotune
    );
    const auto signature = make_hardware_signature_record(
        inventory, hardware_transport_policy( config, num_procs )
    );
    set_hardware_signature_metadata( config, inventory, signature );
    return config;
}

inline selected_4d_config selected_4d_from_config(
    const config_map &config, int num_procs, const global_sizes &sizes,
    std::size_t ranks_per_node, const std::string &source
)
{
    selected_4d_config selected;
    selected.config = config;
    if ( !apply_config_4d(
             config, num_procs, sizes, ranks_per_node, selected.grid, selected.init_options
         ) )
    {
        throw std::logic_error( "FFTM 4D autotune config did not select a usable plan" );
    }
    selected.strategy_4d = value_or_empty( config, "FFTM_AUTOTUNE_STRATEGY_4D" );
    selected.mode = value_or_empty( config, "FFTM_AUTOTUNE_MODE" );
    selected.backend_4d = value_or_empty( config, "FFTM_AUTOTUNE_BACKEND_4D" );
    selected.source = source;
    return selected;
}

template <class RuntimeApi, class MPIComm>
inline selected_4d_config load_or_create_4d_config(
    const MPIComm &comm, const global_sizes &sizes, const autotune_options_4d &autotune
)
{
    if ( autotune.cache_file.empty() )
        throw std::logic_error( "fftm::autotune_options_4d::cache_file must not be empty" );

    const hardware_inventory inventory = query_distributed_hardware_inventory<RuntimeApi>( comm );
    int state = 0; // 0: cache, 1: created/replaced, 2: missing, 3: invalid.
    if ( comm.myid == 0 )
    {
        try
        {
            if ( !file_exists( autotune.cache_file ) )
            {
                if ( !autotune.create_if_missing )
                    state = 2;
                else
                {
                    save_key_value_file(
                        autotune.cache_file,
                        make_default_4d_config( comm.num_procs, sizes, inventory, autotune )
                    );
                    state = 1;
                }
            }
            else
            {
                try
                {
                    const config_map cached = load_key_value_file( autotune.cache_file );
                    validate_match_4d( cached, comm.num_procs, sizes );
                    if ( autotune.validate_hardware )
                        validate_hardware_match( cached, inventory, autotune );
                }
                catch ( const std::exception & )
                {
                    if ( autotune.mismatch_policy != cache_mismatch_policy::overwrite )
                        state = 3;
                    else
                    {
                        save_key_value_file(
                            autotune.cache_file,
                            make_default_4d_config( comm.num_procs, sizes, inventory, autotune )
                        );
                        state = 1;
                    }
                }
            }
        }
        catch ( const std::exception & )
        {
            state = 3;
        }
    }
    comm.bcast( &state, 1, 0 );
    comm.barrier();
    if ( state == 2 )
        throw std::logic_error( "FFTM 4D autotune cache file does not exist: " + autotune.cache_file );
    if ( state == 3 )
        throw std::logic_error( "FFTM 4D autotune cache is invalid or mismatched: " + autotune.cache_file );

    const config_map config = load_key_value_file( autotune.cache_file );
    validate_match_4d( config, comm.num_procs, sizes );
    if ( autotune.validate_hardware )
        validate_hardware_match( config, inventory, autotune );
    validate_config_constraints_4d( config, autotune.constraints_4d, "policy" );
    return selected_4d_from_config(
        config, comm.num_procs, sizes, homogeneous_ranks_per_node( inventory ),
        state == 0 ? "cache" : "created"
    );
}

template <class FFTMPlan>
inline void init_autotuned_4d_plan(
    FFTMPlan &plan, const selected_4d_config &selected, const global_sizes &sizes
)
{
    if ( selected.runtime_workspace )
        plan.set_reusable_workspace_resource( selected.runtime_workspace );
    plan.template init<4>( selected.grid, sizes, selected.init_options );
}

template <class Sizes>
inline std::size_t checked_local_4d_bytes(
    const Sizes &sizes, std::size_t element_size, const char *description
)
{
    std::size_t result = element_size;
    const std::size_t extents[4] = {
        static_cast<std::size_t>( std::get<0>( sizes ) ),
        static_cast<std::size_t>( std::get<1>( sizes ) ),
        static_cast<std::size_t>( std::get<2>( sizes ) ),
        static_cast<std::size_t>( std::get<3>( sizes ) )
    };
    for ( int dimension = 0; dimension < 4; ++dimension )
    {
        if ( extents[dimension] != 0 &&
             result > std::numeric_limits<std::size_t>::max() / extents[dimension] )
        {
            throw std::overflow_error(
                std::string( "FFTM 4D autotune " ) + description + " size overflows size_t"
            );
        }
        result *= extents[dimension];
    }
    return result;
}

template <class RealArray, class SpectrumArray, class InputSizes, class SpectrumSizes>
inline void init_autotuned_4d_data_arrays(
    RealArray &input, SpectrumArray &spectrum, const selected_4d_config &selected,
    const InputSizes &input_sizes, const SpectrumSizes &spectrum_sizes
)
{
    if ( !selected.runtime_workspace )
    {
        input.init(
            std::get<0>( input_sizes ), std::get<1>( input_sizes ),
            std::get<2>( input_sizes ), std::get<3>( input_sizes )
        );
        spectrum.init(
            std::get<0>( spectrum_sizes ), std::get<1>( spectrum_sizes ),
            std::get<2>( spectrum_sizes ), std::get<3>( spectrum_sizes )
        );
        return;
    }

    const std::size_t input_bytes = checked_local_4d_bytes(
        input_sizes, sizeof( typename RealArray::value_type ), "input buffer"
    );
    const std::size_t spectrum_bytes = checked_local_4d_bytes(
        spectrum_sizes, sizeof( typename SpectrumArray::value_type ), "spectrum buffer"
    );
    selected.runtime_workspace->require_input_size_bytes( input_bytes );
    selected.runtime_workspace->require_spectrum_size_bytes( spectrum_bytes );
    selected.runtime_workspace->activate_input();
    selected.runtime_workspace->activate_spectrum();
    input.init_by_raw_data(
        static_cast<typename RealArray::pointer_type>( selected.runtime_workspace->input_ptr() ),
        std::get<0>( input_sizes ), std::get<1>( input_sizes ),
        std::get<2>( input_sizes ), std::get<3>( input_sizes )
    );
    spectrum.init_by_raw_data(
        static_cast<typename SpectrumArray::pointer_type>( selected.runtime_workspace->spectrum_ptr() ),
        std::get<0>( spectrum_sizes ), std::get<1>( spectrum_sizes ),
        std::get<2>( spectrum_sizes ), std::get<3>( spectrum_sizes )
    );
}

} // namespace autotune
} // namespace fftm

#endif
