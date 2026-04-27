#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


#include <scfd/backend/cuda.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/system_timer_event.h>

#include <external_wrap/cufft_wrap_many.h>
#include <fftm.hpp>

#include "detail/fft_benchmark_common.h"
#include "detail/fft_benchmark_options.h"
#include "detail/test_memory_profile_helpers.h"

namespace
{

using T             = double;
using base_fft_t    = fftm::wrap::cufft_wrap_many<T>;
using runtime_api_t = typename base_fft_t::runtime_api;
using backend_t     = scfd::backend::cuda;
using memory_t      = backend_t::memory_type;
using reduce_t      = backend_t::reduce_type;
using for_each_t    = backend_t::template for_each_nd_type<3, int>;
using idx_t         = scfd::static_vec::vec<int, 3>;
using rect_t        = scfd::static_vec::rect<int, 3>;
using options_t     = fftm::test::detail::fftm_3d_benchmark_options<T>;
using strategy_kind = fftm::test::detail::fftm_3d_strategy_kind;

template <class Strategy>
int run_benchmark_case(
    scfd::utils::log_mpi &log, const options_t &options, const scfd::communication::mpi_comm_info &comm_info
)
{
    using fftm_t =
        fftm::fftm<base_fft_t, scfd::communication::mpi_comm_info, backend_t, Strategy, scfd::utils::log_mpi>;
    using real_array_t  = typename fftm_t::template real_array_t<3>;
    using hat_array_t   = typename fftm_t::template complex_array_t<3>;
    using error_array_t = scfd::arrays::array_nd<T, 1, memory_t>;

    fftm::processor_grid grid;
    grid.init( options.p1, options.p2 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );

    fftm_t distributed_fft( comm_info, log );
    distributed_fft.template init<3>( grid, sizes );

    const auto  in_sizes   = distributed_fft.get_local_input_sizes();
    const auto  out_sizes  = distributed_fft.get_local_output_sizes();
    const auto &input_part = distributed_fft.input_partition();

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( grid, sizes );

    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    real_array_t work;
    hat_array_t  hat;

    work.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ) );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    const T        normalization = T( 1 ) / static_cast<T>( options.nx * options.ny * options.nz );
    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );
    T max_norm = T( 0 );

    for ( int iter = 0; iter < options.warmup; ++iter )
    {
        const unsigned long long seed = 0x3141592653589793ull + static_cast<unsigned long long>( iter );

        for_each(
            fftm::test::detail::fill_random_real_3d_functor<T, idx_t, real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), 0 },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

        runtime_api_t::device_synchronize();
        distributed_fft.forward( work, hat );
        distributed_fft.backward( hat, work );
        runtime_api_t::device_synchronize();
    }

    for ( int iter = 0; iter < options.times; ++iter )
    {
        const unsigned long long seed = 0x3141592653589793ull + static_cast<unsigned long long>( iter );

        for_each(
            fftm::test::detail::fill_random_real_3d_functor<T, idx_t, real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), 0 },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        distributed_fft.forward( work, hat );
        distributed_fft.backward( hat, work );
        runtime_api_t::device_synchronize();
        t1.record();
        wall_times.push_back( static_cast<T>( t1.elapsed_time( t0 ) ) );

        for_each(
            fftm::test::detail::scale_real_functor<T, idx_t, real_array_t>{ work, normalization },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

        for_each(
            fftm::test::detail::overwrite_with_random_diff_square_3d_functor<T, idx_t, real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), 0 },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

        const T local_diff_sq  = reduce( work.size(), work.raw_ptr(), T( 0 ) );
        const T global_diff_sq = comm_info.all_reduce_sum( local_diff_sq );
        const T diff_l2        = std::sqrt( global_diff_sq / static_cast<T>( options.nx * options.ny * options.nz ) );
        if ( diff_l2 > max_norm )
            max_norm = diff_l2;

        if ( diff_l2 > options.epsilon && comm_info.myid == 0 )
        {
            log.warning_f(
                "strategy=%s, mode=%s, iteration=%d: l2_diff=%.8e exceeded epsilon=%.8e", fftm_t::strategy_name(),
                fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_3d ), iter, diff_l2, options.epsilon
            );
        }
    }

    const auto stats = fftm::test::detail::compute_timing_statistics( wall_times );
    fftm::test::detail::log_tracked_memory_with_external_mpi(
        log, comm_info, distributed_fft, "benchmark=fftm-3d",
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( work ), fftm::test::detail::array_bytes( hat ) ) ),
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( work ), fftm::test::detail::array_bytes( hat ) ) )
    );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "benchmark=fftm-3d, strategy=%s, mode=%s, grid=(%zu,%zu), Nx=%zu, Ny=%zu, Nz=%zu, warmup=%d, times=%d: "
            "avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
            fftm_t::strategy_name(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_3d ), options.p1,
            options.p2, options.nx, options.ny, options.nz, options.warmup, options.times, stats.mean, stats.stddev
        );

        std::ostringstream row;
        row << fftm::test::detail::csv_quote( "fftm-3d" ) << ',' << comm_info.num_procs << ','
            << fftm::test::detail::csv_quote( fftm_t::strategy_name() ) << ','
            << fftm::test::detail::csv_quote( fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_3d ) ) << ','
            << options.p1 << ',' << options.p2 << ',' << 1 << ',' << options.nx << ',' << options.ny << ','
            << options.nz << ',' << 0 << ',' << options.times << ',' << options.warmup << ',' << options.epsilon
            << ',' << stats.mean << ',' << stats.stddev << ',' << max_norm << ','
            << fftm::test::detail::csv_quote( options.directory );

        fftm::test::detail::append_csv_row(
            options.directory, "benchmark_fftm_3d.csv",
            "benchmark,num_gpus,strategy,mode,p1,p2,p3,nx,ny,nz,nw,times,warmup,epsilon,avg_wall_ms,stddev_wall_ms,"
            "max_l2_diff,directory",
            row.str()
        );
    }

    return 0;
}

template <fftm::mpi_transpose_3d_mode Mode>
int dispatch_mode(
    strategy_kind strategy, scfd::utils::log_mpi &log, const options_t &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    switch ( strategy )
    {
    case strategy_kind::slab_pencil:
        return run_benchmark_case<fftm::strategy_3d_slab_pencil<Mode>>( log, options, comm_info );
    case strategy_kind::pencil_slab:
        return run_benchmark_case<fftm::strategy_3d_pencil_slab<Mode>>( log, options, comm_info );
    case strategy_kind::pencil_pencil:
        return run_benchmark_case<fftm::strategy_3d_pencil_pencil<Mode>>( log, options, comm_info );
    }
    return 1;
}

int dispatch_strategy(
    strategy_kind strategy, scfd::utils::log_mpi &log, const options_t &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    switch ( options.mode )
    {
    case fftm::mpi_transpose_3d_mode::p2p_waitall:
        return dispatch_mode<fftm::mpi_transpose_3d_mode::p2p_waitall>( strategy, log, options, comm_info );
    case fftm::mpi_transpose_3d_mode::p2p_waitany:
        return dispatch_mode<fftm::mpi_transpose_3d_mode::p2p_waitany>( strategy, log, options, comm_info );
    case fftm::mpi_transpose_3d_mode::alltoallv:
        return dispatch_mode<fftm::mpi_transpose_3d_mode::alltoallv>( strategy, log, options, comm_info );
    case fftm::mpi_transpose_3d_mode::alltoallw:
        return dispatch_mode<fftm::mpi_transpose_3d_mode::alltoallw>( strategy, log, options, comm_info );
    }
    return 1;
}

} // namespace

int main( int argc, char *argv[] )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    auto                          comm_info = mpi.comm_world();
    scfd::utils::log_mpi          log;

    try
    {
        scfd::utils::init_cuda_mpi( log, comm_info );
        const options_t options = fftm::test::detail::parse_fftm_3d_benchmark_options<T>(
            argc, argv, comm_info.num_procs, "test_benchmark_fftm_3D.bin"
        );

        if ( options.p1 * options.p2 != static_cast<std::size_t>( comm_info.num_procs ) )
            throw std::logic_error( "P1*P2 must equal the number of MPI processes" );

        if ( options.run_all )
        {
            int status = 0;
            status |= dispatch_strategy( strategy_kind::slab_pencil, log, options, comm_info );
            status |= dispatch_strategy( strategy_kind::pencil_slab, log, options, comm_info );
            status |= dispatch_strategy( strategy_kind::pencil_pencil, log, options, comm_info );
            return status;
        }

        return dispatch_strategy( options.strategy, log, options, comm_info );
    }
    catch ( const std::exception &ex )
    {
        log.error( scfd::utils::nested_exception_to_multistring( ex ) );
        return 1;
    }
}
