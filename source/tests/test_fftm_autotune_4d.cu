#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <fftm_autotune_measure_4d.hpp>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/communication/mpi_wrap.h>

namespace
{

struct fake_runtime_api
{
    static fftm::detail::runtime_hardware_identity get_hardware_identity()
    {
        fftm::detail::runtime_hardware_identity result;
        result.backend = "test";
        result.device_name = "Synthetic Accelerator";
        result.device_uuid = "synthetic-uuid";
        result.pci_bus_id = "0000:01:00.0";
        result.architecture = "test-1";
        result.runtime_version = 12060;
        result.driver_version = 12080;
        result.total_memory_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
        result.total_memory_known = true;
        return result;
    }
};

struct fake_candidate_evaluator
{
    int calls = 0;
    std::string fail_mode;

    fftm::autotune::candidate_measurement operator()(
        const fftm::autotune::selected_4d_config &selected,
        const fftm::autotune::measured_4d_options &
    )
    {
        ++calls;
        if ( !fail_mode.empty() && selected.mode == fail_mode )
            throw std::runtime_error( "synthetic 4D candidate failure" );

        double wall_ms = selected.strategy_4d == "slab-slab" ? 10.0 : 20.0;
        if ( selected.mode == "alltoallv" )
            wall_ms += 5.0;
        if ( fftm::autotune::value_or_empty(
                 selected.config, "FFTM_AUTOTUNE_4D_SLAB_BACKWARD_CREDIT_POLICY"
             ) == "disabled" )
        {
            wall_ms += 1.0;
        }
        if ( fftm::autotune::value_or_empty(
                 selected.config, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D"
             ) == "public-yzwx" )
        {
            wall_ms += 2.0;
        }

        fftm::autotune::candidate_measurement result;
        result.valid = true;
        result.median_wall_ms = wall_ms;
        return result;
    }
};

void test_policy_and_candidate_generation()
{
    fftm::global_sizes sizes;
    sizes.init( 320, 320, 320, 320 );

    fftm::autotune::autotune_options_4d autotune;
    autotune.requested_spectral_layout = fftm::fftm_4d_spectral_layout::native_xzwy;

    const auto policy = fftm::autotune::make_default_4d_policy_config(
        16, sizes, 8, autotune
    );
    assert( fftm::autotune::value_or_empty( policy, "FFTM_AUTOTUNE_DIM" ) == "4" );
    assert( fftm::autotune::value_or_empty(
        policy, "FFTM_AUTOTUNE_STRATEGY_4D"
    ) == "slab-slab" );
    assert( fftm::autotune::value_or_empty(
        policy, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D"
    ) == "native-xzwy" );
    assert( fftm::autotune::value_or_empty(
        policy, "FFTM_AUTOTUNE_GRID_4D"
    ) == "1x16x1" );

    fftm::autotune::measured_4d_options measured;
    const auto candidates = fftm::autotune::make_measured_4d_candidates(
        16, sizes, 8, autotune, measured
    );
    assert( candidates.size() == 3 );

    bool found_node_aligned_pencil = false;
    bool found_credit_fallback = false;
    for ( const auto &candidate : candidates )
    {
        const std::string strategy = fftm::autotune::value_or_empty(
            candidate, "FFTM_AUTOTUNE_STRATEGY_4D"
        );
        found_node_aligned_pencil = found_node_aligned_pencil ||
            ( strategy == "pencil-pencil" &&
              fftm::autotune::value_or_empty(
                  candidate, "FFTM_AUTOTUNE_GRID_4D"
              ) == "2x8x1" &&
              fftm::autotune::value_or_empty(
                  candidate, "FFTM_AUTOTUNE_PENCIL_PIPELINE_4D"
              ) == "node-aligned-wz" );
        found_credit_fallback = found_credit_fallback ||
            ( strategy == "slab-slab" &&
              fftm::autotune::value_or_empty(
                  candidate, "FFTM_AUTOTUNE_MODE"
              ) == "p2p-waitany" &&
              fftm::autotune::value_or_empty(
                  candidate, "FFTM_AUTOTUNE_4D_SLAB_BACKWARD_CREDIT_POLICY"
              ) == "disabled" );
    }
    assert( found_node_aligned_pencil );
    assert( found_credit_fallback );
    for ( const auto &candidate : candidates )
    {
        assert( fftm::autotune::value_or_empty(
            candidate, "FFTM_AUTOTUNE_MODE"
        ) == "p2p-waitany" );
    }

    fftm::autotune::autotune_options_4d two_layouts = autotune;
    two_layouts.accepted_spectral_layouts = {
        fftm::fftm_4d_spectral_layout::public_yzwx,
        fftm::fftm_4d_spectral_layout::native_xzwy
    };
    const auto two_layout_candidates = fftm::autotune::make_measured_4d_candidates(
        16, sizes, 8, two_layouts, measured
    );
    assert( two_layout_candidates.size() == 7 );
    bool found_public = false;
    bool found_native = false;
    for ( const auto &candidate : two_layout_candidates )
    {
        const std::string layout = fftm::autotune::value_or_empty(
            candidate, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D"
        );
        found_public = found_public || layout == "public-yzwx";
        found_native = found_native || layout == "native-xzwy";
        assert( fftm::autotune::value_or_empty(
            candidate, "FFTM_AUTOTUNE_ACCEPTED_SPECTRAL_LAYOUTS_4D"
        ) == "public-yzwx,native-xzwy" );
    }
    assert( found_public && found_native );
}

void test_measured_candidate_filtering()
{
    fftm::global_sizes sizes;
    sizes.init( 320, 320, 320, 320 );
    fftm::autotune::autotune_options_4d autotune;
    autotune.requested_spectral_layout = fftm::fftm_4d_spectral_layout::native_xzwy;
    autotune.accepted_spectral_layouts = {
        fftm::fftm_4d_spectral_layout::public_yzwx,
        fftm::fftm_4d_spectral_layout::native_xzwy
    };
    fftm::autotune::measured_4d_options measured;
    const auto configs = fftm::autotune::make_measured_4d_candidates(
        2, sizes, 2, autotune, measured
    );

    std::vector<fftm::autotune::measured_4d_candidate> candidates;
    for ( const auto &config : configs )
    {
        fftm::autotune::measured_4d_candidate candidate;
        candidate.config = config;
        candidate.measurement.valid = true;
        candidate.measurement.median_wall_ms =
            fftm::autotune::value_or_empty(
                config, "FFTM_AUTOTUNE_STRATEGY_4D"
            ) == "slab-slab" ? 10.0 : 20.0;
        if ( fftm::autotune::value_or_empty(
                 config, "FFTM_AUTOTUNE_MODE"
             ) == "alltoallv" )
        {
            candidate.measurement.median_wall_ms += 5.0;
        }
        candidates.push_back( candidate );
    }

    fftm::autotune::config_map cache = candidates[0].config;
    fftm::autotune::store_measured_candidates_4d(
        cache, candidates, 0, measured
    );
    fftm::autotune::autotune_constraints_4d constraints;
    constraints.strategy_4d = "pencil-pencil";
    constraints.mode = "alltoallv";
    const auto selected = fftm::autotune::select_measured_candidate_config_4d(
        cache, constraints
    );
    assert( fftm::autotune::value_or_empty(
        selected, "FFTM_AUTOTUNE_STRATEGY_4D"
    ) == "pencil-pencil" );
    assert( fftm::autotune::value_or_empty(
        selected, "FFTM_AUTOTUNE_MODE"
    ) == "alltoallv" );
}

void test_collective_measured_cache(
    const scfd::communication::mpi_comm_info &comm, const std::string &cache_file
)
{
    if ( comm.myid == 0 )
        std::remove( cache_file.c_str() );
    comm.barrier();

    fftm::global_sizes sizes;
    sizes.init( 64, 64, 64, 64 );
    fftm::autotune::autotune_options_4d autotune;
    autotune.cache_file = cache_file;
    autotune.strict_device_identity = false;
    autotune.requested_spectral_layout = fftm::fftm_4d_spectral_layout::native_xzwy;
    autotune.accepted_spectral_layouts = {
        fftm::fftm_4d_spectral_layout::public_yzwx,
        fftm::fftm_4d_spectral_layout::native_xzwy
    };

    fftm::autotune::measured_4d_options measured;
    measured.warmup = 0;
    measured.iterations = 1;

    fake_candidate_evaluator evaluator;
    const auto selected = fftm::autotune::load_or_measure_4d_config<fake_runtime_api>(
        comm, sizes, autotune, measured, evaluator
    );
    assert( selected.source == "measured" );
    assert( selected.strategy_4d == "slab-slab" );
    assert( selected.mode == "p2p-waitany" );
    assert( evaluator.calls > 0 );
    const int calls_after_measurement = evaluator.calls;

    const auto cached = fftm::autotune::load_or_measure_4d_config<fake_runtime_api>(
        comm, sizes, autotune, measured, evaluator
    );
    assert( cached.source == "cache" );
    assert( evaluator.calls == calls_after_measurement );
    assert( fftm::autotune::value_or_empty(
        cached.config, "FFTM_AUTOTUNE_SOURCE"
    ) == "cpp-measured-4d-v2" );
    assert( fftm::autotune::value_or_empty(
        cached.config, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D"
    ) == "native-xzwy" );

    fftm::autotune::autotune_options_4d restricted = autotune;
    restricted.accepted_spectral_layouts = {
        fftm::fftm_4d_spectral_layout::native_xzwy
    };
    bool layout_domain_rejected = false;
    try
    {
        (void)fftm::autotune::load_or_measure_4d_config<fake_runtime_api>(
            comm, sizes, restricted, measured, evaluator
        );
    }
    catch ( const std::logic_error & )
    {
        layout_domain_rejected = true;
    }
    assert( layout_domain_rejected );

    fftm::global_sizes mismatch;
    mismatch.init( 65, 64, 64, 64 );
    bool mismatch_rejected = false;
    try
    {
        (void)fftm::autotune::load_or_measure_4d_config<fake_runtime_api>(
            comm, mismatch, autotune, measured, evaluator
        );
    }
    catch ( const std::logic_error & )
    {
        mismatch_rejected = true;
    }
    assert( mismatch_rejected );

    comm.barrier();
    if ( comm.myid == 0 )
        std::remove( cache_file.c_str() );
}

void test_collective_policy_cache(
    const scfd::communication::mpi_comm_info &comm, const std::string &cache_file
)
{
    if ( comm.myid == 0 )
        std::remove( cache_file.c_str() );
    comm.barrier();

    fftm::global_sizes sizes;
    sizes.init( 32, 32, 32, 32 );
    fftm::autotune::autotune_options_4d autotune;
    autotune.cache_file = cache_file;
    autotune.requested_spectral_layout = fftm::fftm_4d_spectral_layout::native_xzwy;

    const auto created = fftm::autotune::load_or_create_4d_config<fake_runtime_api>(
        comm, sizes, autotune
    );
    assert( created.source == "created" );
    assert( fftm::autotune::value_or_empty(
        created.config, "FFTM_AUTOTUNE_HARDWARE_SIGNATURE_VERSION"
    ) == "2" );

    const auto cached = fftm::autotune::load_or_create_4d_config<fake_runtime_api>(
        comm, sizes, autotune
    );
    assert( cached.source == "cache" );

    autotune.constraints_4d.strategy_4d = "pencil-pencil";
    bool constraint_rejected = false;
    try
    {
        (void)fftm::autotune::load_or_create_4d_config<fake_runtime_api>(
            comm, sizes, autotune
        );
    }
    catch ( const std::logic_error & )
    {
        constraint_rejected = true;
    }
    assert( constraint_rejected );

    comm.barrier();
    if ( comm.myid == 0 )
        std::remove( cache_file.c_str() );
}

void test_failed_measured_candidate(
    const scfd::communication::mpi_comm_info &comm, const std::string &cache_file
)
{
    if ( comm.myid == 0 )
        std::remove( cache_file.c_str() );
    comm.barrier();

    fftm::global_sizes sizes;
    sizes.init( 64, 64, 64, 64 );
    fftm::autotune::autotune_options_4d autotune;
    autotune.cache_file = cache_file;
    autotune.requested_spectral_layout = fftm::fftm_4d_spectral_layout::native_xzwy;

    fftm::autotune::measured_4d_options measured;
    measured.warmup = 0;
    measured.iterations = 1;

    fake_candidate_evaluator evaluator;
    evaluator.fail_mode = "alltoallv";
    const auto selected = fftm::autotune::load_or_measure_4d_config<fake_runtime_api>(
        comm, sizes, autotune, measured, evaluator
    );
    assert( selected.source == "measured" );
    assert( selected.mode == "p2p-waitany" );

    if ( comm.myid == 0 )
    {
        const auto cache = fftm::autotune::load_key_value_file( cache_file );
        const std::size_t count = static_cast<std::size_t>( std::strtoull(
            fftm::autotune::value_or_empty(
                cache, "FFTM_AUTOTUNE_CANDIDATE_COUNT"
            ).c_str(), nullptr, 10
        ) );
        std::size_t invalid = 0;
        for ( std::size_t index = 0; index < count; ++index )
        {
            const std::string prefix = fftm::autotune::measured_candidate_prefix_4d( index );
            if ( !fftm::autotune::truthy(
                     fftm::autotune::value_or_empty( cache, prefix + "VALID" )
                 ) )
            {
                ++invalid;
                assert( !fftm::autotune::value_or_empty(
                    cache, prefix + "ERROR"
                ).empty() );
            }
        }
        assert( invalid == ( comm.num_procs > 1 ? 1U : 0U ) );
    }

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
        argc > 1 ? argv[1] : "/tmp/fftm_autotune_4d_test.env";

    test_policy_and_candidate_generation();
    test_measured_candidate_filtering();
    test_collective_policy_cache( comm, cache_file + ".policy" );
    test_collective_measured_cache( comm, cache_file );
    test_failed_measured_candidate( comm, cache_file + ".failure" );
    return 0;
}
