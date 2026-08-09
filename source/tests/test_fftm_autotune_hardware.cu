#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <fftm_autotune_measure.hpp>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/communication/mpi_wrap.h>

namespace
{

struct fake_runtime_api
{
    static fftm::detail::runtime_hardware_identity get_hardware_identity()
    {
        fftm::detail::runtime_hardware_identity result;
        result.backend            = "test";
        result.device_name        = "Synthetic Accelerator";
        result.device_uuid        = "synthetic-uuid";
        result.pci_bus_id         = "0000:01:00.0";
        result.architecture       = "test-1";
        result.runtime_version    = 12060;
        result.driver_version     = 12080;
        result.total_memory_bytes = 80ULL * 1024ULL * 1024ULL * 1024ULL;
        result.total_memory_known = true;
        return result;
    }
};

struct fake_candidate_evaluator
{
    int calls = 0;

    fftm::autotune::candidate_measurement operator()(
        const fftm::autotune::selected_3d_config &selected,
        const fftm::autotune::measured_3d_options &
    )
    {
        ++calls;
        double wall_ms = 30.0;
        if ( selected.strategy_3d == "slab-pencil" )
            wall_ms = 10.0;
        else if ( selected.strategy_3d == "pencil-slab" )
            wall_ms = 20.0;
        if ( selected.mode == "alltoallv" )
            wall_ms += 5.0;

        fftm::autotune::candidate_measurement result;
        result.valid = true;
        result.median_wall_ms = wall_ms;
        return result;
    }
};

fftm::autotune::hardware_inventory make_inventory()
{
    fftm::autotune::hardware_inventory result;
    result.mpi_ranks = 8;
    result.mpi_library_version = "Open MPI synthetic";
    result.rank_node_names.assign( 8, "node-a" );
    result.node_names = { "node-a" };
    result.ranks_per_node = { 8 };
    result.distributed_details_available = true;
    for ( int rank = 0; rank < result.mpi_ranks; ++rank )
    {
        result.device_classes.push_back(
            "backend=cuda:model=NVIDIA A100-SXM4-80GB:arch=8.0:memory=85899345920:runtime=12060:driver=12080"
        );
        result.device_backends.push_back( "cuda" );
        result.device_models.push_back( "NVIDIA A100-SXM4-80GB" );
        result.device_architectures.push_back( "8.0" );
        result.device_uuids.push_back( "uuid-" + std::to_string( rank ) );
        result.device_pci_bus_ids.push_back( "pci-" + std::to_string( rank ) );
        result.runtime_versions.push_back( "12060" );
        result.driver_versions.push_back( "12080" );
        result.device_total_bytes.push_back( "85899345920" );
        result.transport_environments.push_back( "affinity=hca:ucx_tls=default:mpi_pml=ucx" );
    }
    return result;
}

bool rejects_hardware(
    const fftm::autotune::config_map &config,
    const fftm::autotune::hardware_inventory &inventory,
    const fftm::autotune::autotune_options &options = fftm::autotune::autotune_options{}
)
{
    try
    {
        fftm::autotune::validate_hardware_match( config, inventory, options );
    }
    catch ( const std::logic_error & )
    {
        return true;
    }
    return false;
}

void test_signature_semantics()
{
    fftm::global_sizes sizes;
    sizes.init( 2048, 2048, 2048 );

    const auto inventory = make_inventory();
    const auto config = fftm::autotune::make_default_3d_config( 8, sizes, inventory );
    assert( fftm::autotune::value_or_empty( config, "FFTM_AUTOTUNE_SCHEMA" ) == "2" );
    assert( fftm::autotune::value_or_empty( config, "FFTM_AUTOTUNE_HARDWARE_SIGNATURE_VERSION" ) == "2" );
    assert( fftm::autotune::value_or_empty( config, "FFTM_AUTOTUNE_GPU_MODELS" ) == "NVIDIA_A100-SXM4-80GB" );
    assert( fftm::autotune::value_or_empty( config, "FFTM_AUTOTUNE_NODE_COUNT" ) == "1" );
    assert( fftm::autotune::value_or_empty( config, "FFTM_AUTOTUNE_RANKS_PER_NODE" ) == "8" );
    assert( fftm::autotune::value_or_empty( config, "FFTM_AUTOTUNE_DEVICES_PER_NODE" ) == "8" );
    assert( fftm::autotune::value_or_empty( config, "FFTM_AUTOTUNE_POLICY_VERSION" ) == "4" );
    assert( fftm::autotune::value_or_empty( config, "FFTM_AUTOTUNE_STRATEGY_3D" ) == "slab-pencil" );
    assert( fftm::autotune::value_or_empty( config, "FFTM_AUTOTUNE_GRID_3D" ) == "8x1" );
    assert( !rejects_hardware( config, inventory ) );

    fftm::global_sizes fitted_sizes;
    fitted_sizes.init( 1920, 1920, 1920 );
    const auto fitted_config = fftm::autotune::make_default_3d_policy_config( 6, fitted_sizes );
    assert( fftm::autotune::value_or_empty(
        fitted_config, "FFTM_AUTOTUNE_STRATEGY_3D"
    ) == "pencil-slab" );
    assert( fftm::autotune::value_or_empty( fitted_config, "FFTM_AUTOTUNE_GRID_3D" ) == "1x6" );

    const auto multinode_config = fftm::autotune::make_default_3d_policy_config( 16, sizes );
    assert( fftm::autotune::value_or_empty(
        multinode_config, "FFTM_AUTOTUNE_STRATEGY_3D"
    ) == "slab-pencil" );
    assert( fftm::autotune::value_or_empty( multinode_config, "FFTM_AUTOTUNE_GRID_3D" ) == "16x1" );
    assert( fftm::autotune::value_or_empty( multinode_config, "FFTM_AUTOTUNE_MODE" ) == "p2p-waitany" );

    const auto multinode32_config = fftm::autotune::make_default_3d_policy_config( 32, sizes );
    assert( fftm::autotune::value_or_empty(
        multinode32_config, "FFTM_AUTOTUNE_STRATEGY_3D"
    ) == "slab-pencil" );
    assert( fftm::autotune::value_or_empty( multinode32_config, "FFTM_AUTOTUNE_GRID_3D" ) == "32x1" );
    assert( fftm::autotune::value_or_empty( multinode32_config, "FFTM_AUTOTUNE_MODE" ) == "alltoallv" );

    const auto large_multinode_config = fftm::autotune::make_default_3d_policy_config( 64, sizes );
    assert( fftm::autotune::value_or_empty(
        large_multinode_config, "FFTM_AUTOTUNE_STRATEGY_3D"
    ) == "pencil-pencil" );
    assert( fftm::autotune::value_or_empty(
        large_multinode_config, "FFTM_AUTOTUNE_GRID_3D"
    ) == "8x8" );
    assert( fftm::autotune::value_or_empty(
        large_multinode_config, "FFTM_AUTOTUNE_PENCIL_LAYOUT"
    ) == "opt0" );
    assert( fftm::autotune::value_or_empty(
        large_multinode_config, "FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC"
    ) == "1" );

    const auto multinode96_config = fftm::autotune::make_default_3d_policy_config( 96, sizes, 8 );
    assert( fftm::autotune::value_or_empty( multinode96_config, "FFTM_AUTOTUNE_GRID_3D" ) == "12x8" );
    assert( fftm::autotune::value_or_empty(
        multinode96_config, "FFTM_AUTOTUNE_PENCIL_LAYOUT"
    ) == "opt0" );

    const auto multinode120_config = fftm::autotune::make_default_3d_policy_config( 120, sizes, 8 );
    assert( fftm::autotune::value_or_empty( multinode120_config, "FFTM_AUTOTUNE_GRID_3D" ) == "15x8" );
    assert( fftm::autotune::value_or_empty(
        multinode120_config, "FFTM_AUTOTUNE_PENCIL_LAYOUT"
    ) == "opt0" );

    auto stale_policy = large_multinode_config;
    stale_policy["FFTM_AUTOTUNE_SOURCE"] = "cpp-policy-cache-v3";
    stale_policy["FFTM_AUTOTUNE_POLICY_VERSION"] = "3";
    bool stale_policy_rejected = false;
    try
    {
        fftm::autotune::validate_match( stale_policy, 64, sizes );
    }
    catch ( const std::logic_error & )
    {
        stale_policy_rejected = true;
    }
    assert( stale_policy_rejected );

    auto replacement_devices = inventory;
    replacement_devices.device_uuids[0] = "replacement-uuid";
    replacement_devices.device_pci_bus_ids[0] = "replacement-pci";
    replacement_devices.node_names[0] = "node-b";
    assert( !rejects_hardware( config, replacement_devices ) );

    fftm::autotune::autotune_options strict;
    strict.strict_device_identity = true;
    assert( rejects_hardware( config, replacement_devices, strict ) );
    assert( !rejects_hardware( config, inventory, strict ) );

    auto different_model = inventory;
    different_model.device_classes[0] =
        "backend=cuda:model=NVIDIA H100:arch=9.0:memory=85899345920:runtime=12060:driver=12080";
    assert( rejects_hardware( config, different_model ) );

    auto different_mpi = inventory;
    different_mpi.mpi_library_version = "Different MPI";
    assert( rejects_hardware( config, different_mpi ) );

    auto different_topology = inventory;
    different_topology.node_names = { "node-a", "node-b" };
    different_topology.ranks_per_node = { 4, 4 };
    different_topology.rank_node_names = {
        "node-a", "node-a", "node-a", "node-a", "node-b", "node-b", "node-b", "node-b"
    };
    assert( rejects_hardware( config, different_topology ) );

    auto wrapped_devices = inventory;
    wrapped_devices.device_uuids.assign( 8, "one-shared-device" );
    assert( rejects_hardware( config, wrapped_devices ) );

    auto different_transport = config;
    different_transport["FFTM_DIRECT_P2P_CUDA_AWARE"] = "0";
    assert( rejects_hardware( different_transport, inventory ) );

    auto legacy = config;
    legacy["FFTM_AUTOTUNE_HARDWARE_SIGNATURE"] = "mpi=8;device_total=85899345920";
    assert( rejects_hardware( legacy, inventory ) );
    fftm::autotune::autotune_options allow_legacy;
    allow_legacy.allow_legacy_hardware_signature = true;
    assert( !rejects_hardware( legacy, inventory, allow_legacy ) );
}

void test_measured_candidate_policy()
{
    fftm::global_sizes sizes;
    sizes.init( 2048, 2048, 2048 );

    fftm::autotune::measured_3d_options options;
    assert( options.include_alltoallv );
    assert( options.verify_candidate_memory_recovery );
    assert( options.candidate_memory_recovery_tolerance_bytes == 1024ULL * 1024ULL * 1024ULL );
    const auto candidates6 = fftm::autotune::make_measured_3d_candidates( 6, sizes, options );
    const auto candidates7 = fftm::autotune::make_measured_3d_candidates( 7, sizes, options );
    const auto candidates8 = fftm::autotune::make_measured_3d_candidates( 8, sizes, options );
    const auto candidates64 = fftm::autotune::make_measured_3d_candidates( 64, sizes, options, 8 );
    assert( candidates6.size() == 6 );
    assert( candidates7.size() == 6 );
    assert( candidates8.size() == 6 );
    assert( candidates64.size() == 6 );
    for ( const auto *candidates : { &candidates6, &candidates7, &candidates8 } )
    {
        for ( const char *mode : { "p2p-waitany", "alltoallv" } )
        {
            for ( const char *strategy : { "slab-pencil", "pencil-slab", "pencil-pencil" } )
            {
                bool found = false;
                for ( const auto &candidate : *candidates )
                {
                    found = found ||
                        ( fftm::autotune::value_or_empty( candidate, "FFTM_AUTOTUNE_MODE" ) == mode &&
                          fftm::autotune::value_or_empty( candidate, "FFTM_AUTOTUNE_STRATEGY_3D" ) == strategy );
                }
                assert( found );
            }
        }
    }

    const auto find_pencil = []( const std::vector<fftm::autotune::config_map> &candidates,
                                 const char *mode ) -> const fftm::autotune::config_map & {
        for ( const auto &candidate : candidates )
        {
            if ( fftm::autotune::value_or_empty( candidate, "FFTM_AUTOTUNE_STRATEGY_3D" ) == "pencil-pencil" &&
                 fftm::autotune::value_or_empty( candidate, "FFTM_AUTOTUNE_MODE" ) == mode )
                return candidate;
        }
        throw std::logic_error( "missing pencil-pencil test candidate" );
    };
    assert( fftm::autotune::value_or_empty( find_pencil( candidates6, "p2p-waitany" ), "FFTM_AUTOTUNE_GRID_3D" ) == "2x3" );
    assert( fftm::autotune::value_or_empty( find_pencil( candidates6, "p2p-waitany" ), "FFTM_AUTOTUNE_PENCIL_LAYOUT" ) == "opt1" );
    assert( fftm::autotune::value_or_empty( find_pencil( candidates7, "p2p-waitany" ), "FFTM_AUTOTUNE_GRID_3D" ) == "7x1" );
    assert( fftm::autotune::value_or_empty( find_pencil( candidates7, "p2p-waitany" ), "FFTM_AUTOTUNE_PENCIL_LAYOUT" ) == "opt0" );
    assert( fftm::autotune::value_or_empty( find_pencil( candidates8, "p2p-waitany" ), "FFTM_AUTOTUNE_GRID_3D" ) == "4x2" );
    assert( fftm::autotune::value_or_empty( find_pencil( candidates8, "p2p-waitany" ), "FFTM_AUTOTUNE_PENCIL_LAYOUT" ) == "opt0" );
    assert( fftm::autotune::value_or_empty( find_pencil( candidates64, "p2p-waitany" ), "FFTM_AUTOTUNE_GRID_3D" ) == "8x8" );
    assert( fftm::autotune::value_or_empty( find_pencil( candidates64, "p2p-waitany" ), "FFTM_AUTOTUNE_PENCIL_LAYOUT" ) == "opt0" );
    assert( fftm::autotune::value_or_empty(
        find_pencil( candidates64, "p2p-waitany" ), "FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC"
    ) == "1" );

    options.production_candidates_only = false;
    options.max_pencil_grids = 1;
    const auto broad = fftm::autotune::make_measured_3d_candidates( 8, sizes, options );
    bool found_pencil_slab = false;
    for ( const auto &candidate : broad )
    {
        found_pencil_slab = found_pencil_slab ||
            fftm::autotune::value_or_empty( candidate, "FFTM_AUTOTUNE_STRATEGY_3D" ) == "pencil-slab";
    }
    assert( found_pencil_slab );
}

void test_collective_cache(
    const scfd::communication::mpi_comm_info &comm, const std::string &cache_file
)
{
    if ( comm.myid == 0 )
        std::remove( cache_file.c_str() );
    comm.barrier();

    fftm::global_sizes sizes;
    sizes.init( 32, 32, 32 );

    fftm::autotune::autotune_options options;
    options.cache_file = cache_file;
    options.strict_device_identity = true;

    const auto created = fftm::autotune::load_or_create_3d_config<fake_runtime_api>( comm, sizes, options );
    assert( created.source == "created" );
    assert( fftm::autotune::value_or_empty( created.config, "FFTM_AUTOTUNE_HARDWARE_SIGNATURE_VERSION" ) == "2" );
    assert( fftm::autotune::value_or_empty( created.config, "FFTM_AUTOTUNE_MPI_LIBRARY" ) != "unavailable" );
    assert( fftm::autotune::value_or_empty( created.config, "FFTM_AUTOTUNE_NODE_COUNT" ) != "unknown" );

    const auto cached = fftm::autotune::load_or_create_3d_config<fake_runtime_api>( comm, sizes, options );
    assert( cached.source == "cache" );

    options.constraints_3d.strategy_3d =
        cached.strategy_3d == "pencil-pencil" ? "slab-pencil" : "pencil-pencil";
    bool constraint_rejected = false;
    try
    {
        (void)fftm::autotune::load_or_create_3d_config<fake_runtime_api>( comm, sizes, options );
    }
    catch ( const std::logic_error & )
    {
        constraint_rejected = true;
    }
    assert( constraint_rejected );
    options.constraints_3d = fftm::autotune::autotune_constraints_3d{};

    if ( comm.myid == 0 )
    {
        auto corrupt = fftm::autotune::load_key_value_file( cache_file );
        corrupt["FFTM_AUTOTUNE_HARDWARE_SIGNATURE"] = "v2:0000000000000000";
        fftm::autotune::save_key_value_file( cache_file, corrupt );
    }
    comm.barrier();

    bool rejected = false;
    try
    {
        (void)fftm::autotune::load_or_create_3d_config<fake_runtime_api>( comm, sizes, options );
    }
    catch ( const std::logic_error & )
    {
        rejected = true;
    }
    assert( rejected );

    options.mismatch_policy = fftm::autotune::cache_mismatch_policy::overwrite;
    const auto replaced = fftm::autotune::load_or_create_3d_config<fake_runtime_api>( comm, sizes, options );
    assert( replaced.source == "created" );

    comm.barrier();
    if ( comm.myid == 0 )
        std::remove( cache_file.c_str() );
}

void test_measured_cache(
    const scfd::communication::mpi_comm_info &comm, const std::string &cache_file
)
{
    if ( comm.myid == 0 )
        std::remove( cache_file.c_str() );
    comm.barrier();

    fftm::global_sizes sizes;
    sizes.init( 64, 64, 64 );

    fftm::autotune::autotune_options options;
    options.cache_file = cache_file;
    options.strict_device_identity = true;

    fftm::autotune::measured_3d_options measured;
    measured.warmup = 1;
    measured.iterations = 3;
    measured.max_pencil_grids = 2;

    fake_candidate_evaluator evaluator;
    const auto selected = fftm::autotune::load_or_measure_3d_config<fake_runtime_api>(
        comm, sizes, options, measured, evaluator
    );
    assert( selected.source == "measured" );
    assert( selected.strategy_3d == "slab-pencil" );
    assert( selected.mode == "p2p-waitany" );
    assert( evaluator.calls > 1 );
    assert( fftm::autotune::value_or_empty( selected.config, "FFTM_AUTOTUNE_SOURCE" ) == "cpp-measured-v2" );
    assert( fftm::autotune::value_or_empty(
        selected.config, "FFTM_AUTOTUNE_CANDIDATE_POLICY_VERSION"
    ) == "2" );
    assert( std::atoi( fftm::autotune::value_or_empty(
        selected.config, "FFTM_AUTOTUNE_CANDIDATE_COUNT"
    ).c_str() ) == evaluator.calls );

    const int calls_after_measurement = evaluator.calls;
    const auto cached = fftm::autotune::load_or_measure_3d_config<fake_runtime_api>(
        comm, sizes, options, measured, evaluator
    );
    assert( cached.source == "cache" );
    assert( evaluator.calls == calls_after_measurement );

    options.constraints_3d.mode = "alltoallv";
    const auto constrained = fftm::autotune::load_or_measure_3d_config<fake_runtime_api>(
        comm, sizes, options, measured, evaluator
    );
    assert( constrained.source == "cache" );
    assert( constrained.mode == "alltoallv" );
    assert( evaluator.calls == calls_after_measurement );
    options.constraints_3d = fftm::autotune::autotune_constraints_3d{};

    const auto inventory = fftm::autotune::query_distributed_hardware_inventory<fake_runtime_api>( comm );
    if ( comm.myid == 0 )
    {
        const auto policy = fftm::autotune::make_default_3d_config( comm.num_procs, sizes, inventory );
        fftm::autotune::save_key_value_file( cache_file, policy );
    }
    comm.barrier();

    fake_candidate_evaluator policy_evaluator;
    const auto upgraded = fftm::autotune::load_or_measure_3d_config<fake_runtime_api>(
        comm, sizes, options, measured, policy_evaluator
    );
    assert( upgraded.source == "measured" );
    assert( policy_evaluator.calls > 1 );

    comm.barrier();
    if ( comm.myid == 0 )
        std::remove( cache_file.c_str() );
}

} // namespace

int main( int argc, char *argv[] )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    const auto comm = mpi.comm_world();
    const std::string cache_file =
        argc > 1 ? argv[1] : "/tmp/fftm_autotune_hardware_test.env";

    test_signature_semantics();
    test_measured_candidate_policy();
    test_collective_cache( comm, cache_file );
    test_measured_cache( comm, cache_file + ".measured" );
    return 0;
}
