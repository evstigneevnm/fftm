#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <tuple>

#include <external_wrap/cufft_wrap_many.h>
#include <fftm.hpp>
#include <fftm_autotune.hpp>
#include <scfd/backend/cuda.h>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/log_mpi.h>

#include "detail/mpi_cuda_test_init.h"

namespace
{

using value_type = double;
using base_fft_t = fftm::wrap::cufft_wrap_many<value_type>;
using backend_t  = scfd::backend::cuda;
using comm_t     = scfd::communication::mpi_comm_info;
using pencil_fftm_t = fftm::fftm<
    base_fft_t, comm_t, backend_t,
    fftm::strategy_3d_pencil_pencil<fftm::mpi_transpose_3d_mode::p2p_waitany, true>>;
using slab_fftm_t = fftm::fftm<
    base_fft_t, comm_t, backend_t,
    fftm::strategy_3d_slab_pencil<fftm::mpi_transpose_3d_mode::p2p_waitany, true>>;

template <class Plan>
void run_plan_lifetime(
    const comm_t &comm, std::size_t size, fftm::transform_strategy_3d strategy,
    int grid_i, int grid_j,
    const std::shared_ptr<fftm::detail::reusable_workspace_resource> &workspace
)
{
    fftm::processor_grid grid;
    grid.init( grid_i, grid_j );
    fftm::global_sizes sizes;
    sizes.init( size, size, size );

    auto options = fftm::production_options_3d(
        strategy, comm.num_procs, true
    );
    Plan plan( comm );
    plan.set_reusable_workspace_resource( workspace );
    plan.template init<3>( grid, sizes, options );

    const auto input_sizes  = plan.get_local_input_sizes();
    const auto output_sizes = plan.get_local_output_sizes();
    typename Plan::template real_array_t<3> input;
    typename Plan::template complex_array_t<3> spectrum;
    fftm::autotune::selected_3d_config selected;
    selected.runtime_workspace = workspace;
    fftm::autotune::init_autotuned_3d_data_arrays(
        input, spectrum, selected, input_sizes, output_sizes
    );

    base_fft_t::runtime_api::memset_zero(
        input.raw_ptr(), static_cast<std::size_t>( input.size() ) * sizeof( value_type )
    );
    plan.forward( input, spectrum );
    plan.backward( spectrum, input );
    base_fft_t::runtime_api::device_synchronize();
    plan.release_resources();
    comm.barrier();
    base_fft_t::runtime_api::device_synchronize();
    comm.barrier();
}

void run_candidate_sequence(
    const comm_t &comm, std::size_t size,
    const std::shared_ptr<fftm::detail::reusable_workspace_resource> &workspace
)
{
    int pencil_grid_j = 1;
    for ( int candidate = 2; candidate * candidate <= comm.num_procs; ++candidate )
    {
        if ( comm.num_procs % candidate == 0 )
            pencil_grid_j = candidate;
    }
    const int pencil_grid_i = comm.num_procs / pencil_grid_j;

    const auto ceil_div = []( std::size_t value, std::size_t divisor ) {
        return ( value + divisor - 1 ) / divisor;
    };
    std::size_t max_spectrum_bytes = 0;
    const auto reserve_grid = [&]( int grid_i, int grid_j ) {
        const auto spectrum_shape = std::make_tuple(
            size, ceil_div( size, static_cast<std::size_t>( grid_i ) ),
            ceil_div( size / 2 + 1, static_cast<std::size_t>( grid_j ) )
        );
        max_spectrum_bytes = std::max(
            max_spectrum_bytes,
            fftm::autotune::checked_local_3d_bytes(
                spectrum_shape, sizeof( base_fft_t::complex ), "lifecycle spectrum pool"
            )
        );
    };
    reserve_grid( pencil_grid_i, pencil_grid_j );
    reserve_grid( comm.num_procs, 1 );

    workspace->acquire(
        fftm::detail::workspace_memory_type_token<base_fft_t::runtime_api::memory_type>()
    );
    try
    {
        workspace->require_spectrum_size_bytes( max_spectrum_bytes );
        workspace->release_lease();
    }
    catch ( ... )
    {
        workspace->release_lease();
        throw;
    }

    run_plan_lifetime<pencil_fftm_t>(
        comm, size, fftm::transform_strategy_3d::pencil_pencil, pencil_grid_i, pencil_grid_j, workspace
    );
    run_plan_lifetime<slab_fftm_t>(
        comm, size, fftm::transform_strategy_3d::slab_pencil, comm.num_procs, 1, workspace
    );
}

} // namespace

int main( int argc, char *argv[] )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    const comm_t                  comm = mpi.comm_world();
    scfd::utils::log_mpi log;
    fftm::test::detail::init_cuda_mpi_for_tests( log, comm );

    constexpr std::size_t problem_size = 128;
    constexpr std::size_t tolerance =
        static_cast<std::size_t>( 64 ) * static_cast<std::size_t>( 1024 ) * static_cast<std::size_t>( 1024 );
    const auto workspace = fftm::detail::make_reusable_workspace_resource<
        base_fft_t::runtime_api::memory_type>();

    // The first lifetime loads CUDA/cuFFT modules. Compare subsequent
    // lifetimes so process-wide one-time allocations do not look like leaks.
    run_candidate_sequence( comm, problem_size, workspace );
    comm.barrier();
    const auto baseline = base_fft_t::runtime_api::get_device_memory_info();

    std::size_t max_retained = 0;
    for ( int iteration = 0; iteration < 2; ++iteration )
    {
        run_candidate_sequence( comm, problem_size, workspace );
        comm.barrier();
        const auto current = base_fft_t::runtime_api::get_device_memory_info();
        if ( baseline.free_bytes_known && current.free_bytes_known && baseline.free_bytes > current.free_bytes )
            max_retained = std::max( max_retained, baseline.free_bytes - current.free_bytes );
    }

    max_retained = static_cast<std::size_t>(
        comm.all_reduce_max( static_cast<double>( max_retained ) )
    );
    if ( max_retained > tolerance )
    {
        throw std::runtime_error(
            "FFTM plan lifecycle retained " + std::to_string( max_retained / ( 1024 * 1024 ) ) +
            " MiB; allowed 64 MiB"
        );
    }

    if ( comm.myid == 0 )
        std::cout << "FFTM_RESOURCE_LIFECYCLE retained_mib=" << max_retained / ( 1024 * 1024 )
                  << " workspace_pool_mib=" << workspace->device_capacity_bytes() / ( 1024 * 1024 )
                  << " input_pool_mib=" << workspace->input_capacity_bytes() / ( 1024 * 1024 )
                  << " spectrum_pool_mib=" << workspace->spectrum_capacity_bytes() / ( 1024 * 1024 )
                  << " retained_pool_mib=" << workspace->retained_device_capacity_bytes() / ( 1024 * 1024 )
                  << '\n';
    return 0;
}
