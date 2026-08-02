#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include <scfd/backend/cuda.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>

#include <external_wrap/cufft_wrap_many.h>
#include <fftm.hpp>
#include <fftm_autotune.hpp>

#include "taylor_green_solver.h"
#include "turbulence_options.h"

namespace
{

#if defined( FFTM_TURBULENCE_SINGLE_PRECISION )
using real_t = float;
#else
using real_t = double;
#endif

using fft_backend_t = fftm::wrap::cufft_wrap_many<real_t>;
using runtime_api_t = typename fft_backend_t::runtime_api;
using backend_t     = scfd::backend::cuda;

bool wrap_mpi_processes_over_gpus_enabled()
{
    const char *value = std::getenv( "FFTM_WRAP_PROCS_GPUS" );
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

fftm::global_sizes make_global_sizes( const fftm::examples::turbulence::turbulence_options &options )
{
    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );
    return sizes;
}

template <class Strategy>
int run_taylor_green(
    const fftm::examples::turbulence::turbulence_options &app_options,
    const fftm::autotune::selected_3d_config &selected, const scfd::communication::mpi_comm_info &comm_info,
    scfd::utils::log_mpi &log
)
{
    using fftm_t =
        fftm::fftm<fft_backend_t, scfd::communication::mpi_comm_info, backend_t, Strategy, scfd::utils::log_mpi>;

    fftm_t distributed_fft( comm_info, log );
    fftm::fftm_init_options init_options = selected.init_options;

    fftm::autotune::selected_3d_config selected_for_plan = selected;
    selected_for_plan.init_options = init_options;
    fftm::autotune::init_autotuned_3d_plan(
        distributed_fft, selected_for_plan, make_global_sizes( app_options )
    );

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( selected.grid, make_global_sizes( app_options ) );
    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();
    (void)myid_k;

    const auto &input_part  = distributed_fft.input_partition();
    const auto &output_part = distributed_fft.output_partition();
    const int input_start_x = static_cast<int>( input_part.start_x[myid_i] );
    const int input_start_y = static_cast<int>( input_part.start_y[myid_j] );
    const int spectral_start_y = static_cast<int>( output_part.start_y[myid_i] );
    const int spectral_start_z = static_cast<int>( output_part.start_z[myid_j] );

    if ( comm_info.myid == 0 )
    {
        fftm::examples::turbulence::create_directories( app_options.output_dir );
        std::ofstream metadata( ( app_options.output_dir + "/run.json" ).c_str() );
        if ( !metadata )
            throw std::runtime_error( "Failed to create Taylor-Green run metadata" );
        metadata << "{\n"
                 << "  \"format\": \"fftm-taylor-green-run-v1\",\n"
                 << "  \"equations\": \"incompressible Navier-Stokes\",\n"
                 << "  \"nonlinear_form\": \"rotational\",\n"
                 << "  \"dealiasing\": \"two-thirds\",\n"
                 << "  \"integrator\": \"Williamson low-storage RK3\",\n"
                 << "  \"precision\": \"" << ( sizeof( real_t ) == sizeof( float ) ? "float32" : "float64" )
                 << "\",\n"
                 << "  \"shape\": [" << app_options.nx << ", " << app_options.ny << ", " << app_options.nz
                 << "],\n"
                 << "  \"reynolds\": " << app_options.reynolds << ",\n"
                 << "  \"dt_mode\": \""
                 << ( app_options.target_cfl > 0.0 ? "adaptive_cfl" : "fixed" ) << "\",\n"
                 << "  \"dt_initial\": " << app_options.dt << ",\n"
                 << "  \"dt_min\": " << app_options.dt_min << ",\n"
                 << "  \"dt_max\": " << app_options.dt_max << ",\n"
                 << "  \"dt_growth\": " << app_options.dt_growth << ",\n"
                 << "  \"target_cfl\": " << app_options.target_cfl << ",\n"
                 << "  \"cfl_fail\": " << app_options.cfl_fail << ",\n"
                 << "  \"final_time_requested\": " << app_options.final_time << ",\n"
                 << "  \"steps_requested\": " << app_options.steps << ",\n"
                 << "  \"snapshot_period\": " << app_options.snapshot_period << ",\n"
                 << "  \"viz_period\": " << app_options.viz_period << ",\n"
                 << "  \"snapshot_size\": " << app_options.snapshot_size << ",\n"
                 << "  \"viz_snapshot_size\": " << app_options.viz_snapshot_size << ",\n"
                 << "  \"visualization_output\": \""
                 << ( app_options.viz_snapshot_size == app_options.snapshot_size ? "." : "visualization" )
                 << "\",\n"
                 << "  \"strategy\": \"" << selected.strategy_3d << "\",\n"
                 << "  \"mode\": \"" << selected.mode << "\",\n"
                 << "  \"grid\": [" << selected.grid.p1 << ", " << selected.grid.p2 << "],\n"
                 << "  \"pencil_layout\": \""
                 << fftm::autotune::value_or_empty( selected.config, "FFTM_AUTOTUNE_PENCIL_LAYOUT" ) << "\",\n"
                 << "  \"pencil_pipeline\": \""
                 << fftm::autotune::value_or_empty( selected.config, "FFTM_AUTOTUNE_PENCIL_PIPELINE" ) << "\",\n"
                 << "  \"autotune_cache\": \"" << app_options.cache_file << "\",\n"
                 << "  \"autotune_source\": \"" << selected.source << "\"\n"
                 << "}\n";
    }
    comm_info.barrier();

    fftm::examples::turbulence::taylor_green_solver<fftm_t> solver(
        distributed_fft, comm_info, log, app_options, input_start_x, input_start_y, spectral_start_y,
        spectral_start_z
    );
    solver.run( comm_info, log );
    return 0;
}

template <fftm::mpi_transpose_3d_mode Mode>
int dispatch_strategy(
    const fftm::examples::turbulence::turbulence_options &app_options,
    const fftm::autotune::selected_3d_config &selected, const scfd::communication::mpi_comm_info &comm_info,
    scfd::utils::log_mpi &log
)
{
    if ( selected.strategy_3d == "slab-pencil" )
        return run_taylor_green<fftm::strategy_3d_slab_pencil<Mode>>(
            app_options, selected, comm_info, log
        );
    if ( selected.strategy_3d == "pencil-slab" )
        return run_taylor_green<fftm::strategy_3d_pencil_slab<Mode>>(
            app_options, selected, comm_info, log
        );
    if ( selected.strategy_3d == "pencil-pencil" )
    {
        if ( selected.init_options.pencil_layout_3d == fftm::fftm_3d_pencil_layout::legacy )
        {
            return run_taylor_green<fftm::strategy_3d_pencil_pencil<Mode, false>>(
                app_options, selected, comm_info, log
            );
        }
        return run_taylor_green<fftm::strategy_3d_pencil_pencil<Mode>>(
            app_options, selected, comm_info, log
        );
    }
    throw std::logic_error( "Unsupported FFTM_AUTOTUNE_STRATEGY_3D='" + selected.strategy_3d + "'" );
}

int dispatch_mode(
    const fftm::examples::turbulence::turbulence_options &app_options,
    const fftm::autotune::selected_3d_config &selected, const scfd::communication::mpi_comm_info &comm_info,
    scfd::utils::log_mpi &log
)
{
    if ( selected.mode == "p2p-waitany" || selected.mode.empty() )
    {
        return dispatch_strategy<fftm::mpi_transpose_3d_mode::p2p_waitany>(
            app_options, selected, comm_info, log
        );
    }
    if ( selected.mode == "p2p-waitall" )
    {
        return dispatch_strategy<fftm::mpi_transpose_3d_mode::p2p_waitall>(
            app_options, selected, comm_info, log
        );
    }
    if ( selected.mode == "alltoallv" )
    {
        return dispatch_strategy<fftm::mpi_transpose_3d_mode::alltoallv>(
            app_options, selected, comm_info, log
        );
    }
    if ( selected.mode == "alltoallw" )
    {
        return dispatch_strategy<fftm::mpi_transpose_3d_mode::alltoallw>(
            app_options, selected, comm_info, log
        );
    }
    throw std::logic_error( "Unsupported FFTM_AUTOTUNE_MODE='" + selected.mode + "'" );
}

} // namespace

int main( int argc, char **argv )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    auto                          comm_info = mpi.comm_world();
    scfd::utils::log_mpi          log;

    try
    {
        scfd::utils::init_cuda_mpi( log, comm_info, 0, wrap_mpi_processes_over_gpus_enabled() );
        const auto app_options = fftm::examples::turbulence::parse_options( argc, argv );
        const auto sizes       = make_global_sizes( app_options );

        fftm::autotune::autotune_options autotune_options;
        autotune_options.cache_file        = app_options.cache_file;
        autotune_options.create_if_missing = true;
        autotune_options.mismatch_policy   = fftm::autotune::cache_mismatch_policy::error;
        autotune_options.validate_hardware = true;

        const auto selected =
            fftm::autotune::load_or_create_3d_config<runtime_api_t>( comm_info, sizes, autotune_options );
        return dispatch_mode( app_options, selected, comm_info, log );
    }
    catch ( const std::exception &error )
    {
        log.error( scfd::utils::nested_exception_to_multistring( error ) );
        return 1;
    }
}
