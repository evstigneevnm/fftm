#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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
#include "detail/mpi_cuda_test_init.h"
#include "detail/test_memory_profile_helpers.h"

namespace
{

using T             = double;
using base_fft_t    = fftm::wrap::cufft_wrap_many<T>;
using runtime_api_t = typename base_fft_t::runtime_api;
using backend_t     = scfd::backend::cuda;
using memory_t      = backend_t::memory_type;
using reduce_t      = backend_t::reduce_type;
using for_each_t    = backend_t::template for_each_nd_type<4, int>;
using idx_t         = scfd::static_vec::vec<int, 4>;
using rect_t        = scfd::static_vec::rect<int, 4>;
using options_t     = fftm::test::detail::fftm_4d_benchmark_options<T>;
using strategy_kind = fftm::test::detail::fftm_4d_strategy_kind;

std::string stage_times_4d_filename( int rank )
{
    return "fftm_4d_stage_times_r" + std::to_string( rank ) + ".csv";
}

std::string wall_times_4d_filename( int rank )
{
    return "wall_times_4d_r" + std::to_string( rank ) + ".csv";
}

const char *stage_timer_4d_strategy_name( strategy_kind strategy )
{
    switch ( strategy )
    {
    case strategy_kind::pencil_pencil:
        return "pencil-pencil";
    case strategy_kind::slab_slab:
        return "slab-slab";
    }
    return "unknown";
}

void append_4d_wall_time_rows(
    const options_t &options, int num_procs, int rank, std::size_t p1, std::size_t p2, std::size_t p3,
    const std::vector<T> &local_wall_times, const std::vector<T> &global_wall_times
)
{
    if ( local_wall_times.size() != global_wall_times.size() )
        throw std::logic_error( "4D local/global wall-time sample counts differ" );

    const std::string header =
        "source,rank,num_gpus,iteration,local_wall_ms,global_wall_ms,strategy,mode,p1,p2,p3,nx,ny,nz,nw,"
        "native_spectral_layout,native_xw_direct_layout,native_xw_chunk_mib,native_xw_chunk_window,"
        "native_xw_compact_staging,slab_native_work_area_alias,slab_native_wz_communication_layout,"
        "slab_native_wz_plan_concurrency,slab_native_wz_ready_pipeline,directory";
    std::vector<std::string> rows;
    rows.reserve( local_wall_times.size() );
    for ( std::size_t i = 0; i < local_wall_times.size(); ++i )
    {
        std::ostringstream row;
        row << fftm::test::detail::csv_quote( "fftm-4d-wall" ) << ',' << rank << ',' << num_procs << ',' << i
            << ',' << local_wall_times[i] << ',' << global_wall_times[i] << ','
            << fftm::test::detail::csv_quote( stage_timer_4d_strategy_name( options.strategy ) ) << ','
            << fftm::test::detail::csv_quote( fftm::mpi_transpose_3d_mode_name( options.mode ) ) << ',' << p1
            << ',' << p2 << ',' << p3 << ',' << options.nx << ',' << options.ny << ',' << options.nz << ','
            << options.nw << ',' << ( options.use_4d_slab_native_xw_native_spectral_layout ? 1 : 0 ) << ','
            << ( options.use_4d_native_xw_direct_layout ? 1 : 0 ) << ',' << options.native_xw_chunk_mib << ','
            << options.native_xw_chunk_window << ',' << ( options.use_4d_native_xw_compact_staging ? 1 : 0 )
            << ',' << ( options.use_4d_slab_native_work_area_alias ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_wz_communication_layout ? 1 : 0 ) << ','
            << options.slab_native_wz_plan_concurrency << ','
            << ( options.use_4d_slab_native_wz_ready_pipeline ? 1 : 0 ) << ','
            << fftm::test::detail::csv_quote( options.directory );
        rows.push_back( row.str() );
    }

    fftm::test::detail::append_csv_rows(
        options.directory, wall_times_4d_filename( rank ), header, rows
    );
}

void append_4d_stage_timer_rows(
    const options_t &options, int num_procs, int rank, int iteration, std::size_t p1, std::size_t p2,
    std::size_t p3, double wall_ms, bool slab_native_work_area_alias_effective,
    const std::vector<fftm::fftm_native_stage_timing> &rows
)
{
    if ( rows.empty() )
        return;

    const std::string header =
        "source,rank,num_gpus,iteration,stage_index,stage,stage_ms,wall_ms,strategy,mode,p1,p2,p3,"
        "nx,ny,nz,nw,fft_exec_no_sync,slab_native_xw,slab_native_xw_batched_peer_kernels,"
        "slab_native_xw_tensor_coalesced_kernels,slab_native_xw_vector4_kernels,"
        "slab_native_xw_tiled_kernels,slab_native_xw_layout_stage,native_xw_direct_layout,"
        "native_xw_chunked_transport,native_xw_chunk_mib,native_xw_chunk_window,native_xw_compact_staging,"
        "slab_native_work_area_alias,slab_native_work_area_alias_effective,slab_native_wz_communication_layout,"
        "slab_native_wz_plan_concurrency,slab_native_wz_ready_pipeline,"
        "slab_native_xw_native_spectral_layout,pencil_same_zw_peer_paired,pencil_same_zw_native_layout,"
        "pencil_degenerate_xw_slab_path,pencil_degenerate_local_transposes,"
        "pencil_degenerate_same_xw_native,pencil_degenerate_wz_sliced_z_fft,directory";

    for ( std::size_t i = 0; i < rows.size(); ++i )
    {
        std::ostringstream row;
        row << fftm::test::detail::csv_quote( "fftm-4d-stage" ) << ',' << rank << ',' << num_procs << ','
            << iteration << ',' << i << ',' << fftm::test::detail::csv_quote( rows[i].stage ) << ','
            << rows[i].ms << ',' << wall_ms << ','
            << fftm::test::detail::csv_quote( stage_timer_4d_strategy_name( options.strategy ) ) << ','
            << fftm::test::detail::csv_quote( fftm::mpi_transpose_3d_mode_name( options.mode ) ) << ','
            << p1 << ',' << p2 << ',' << p3 << ',' << options.nx << ',' << options.ny << ',' << options.nz
            << ',' << options.nw << ',' << ( options.use_fft_exec_no_sync ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_transpose ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_batched_peer_kernels ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_tensor_coalesced_kernels ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_vector4_kernels ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_tiled_kernels ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_layout_stage ? 1 : 0 ) << ','
            << ( options.use_4d_native_xw_direct_layout ? 1 : 0 ) << ','
            << ( options.use_4d_native_xw_chunked_transport ? 1 : 0 ) << ','
            << options.native_xw_chunk_mib << ',' << options.native_xw_chunk_window << ','
            << ( options.use_4d_native_xw_compact_staging ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_work_area_alias ? 1 : 0 ) << ','
            << ( slab_native_work_area_alias_effective ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_wz_communication_layout ? 1 : 0 ) << ','
            << options.slab_native_wz_plan_concurrency << ','
            << ( options.use_4d_slab_native_wz_ready_pipeline ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_native_spectral_layout ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_same_zw_peer_paired ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_same_zw_native_layout ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_degenerate_xw_slab_path ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_degenerate_local_transposes ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_degenerate_same_xw_native ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_degenerate_wz_sliced_z_fft ? 1 : 0 ) << ','
            << fftm::test::detail::csv_quote( options.directory );

        fftm::test::detail::append_csv_row(
            options.directory, stage_times_4d_filename( rank ), header, row.str()
        );
    }
}

template <class DistStrategy4D, fftm::mpi_transpose_3d_mode Mode>
int run_benchmark_case(
    strategy_kind strategy, scfd::utils::log_mpi &log, const options_t &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    using fftm_t = fftm::fftm<
        base_fft_t, scfd::communication::mpi_comm_info, backend_t, fftm::strategy_3d_pencil_pencil<Mode>,
        scfd::utils::log_mpi, DistStrategy4D>;
    using real_array_t  = typename fftm_t::template real_array_t<4>;
    using hat_array_t   = typename fftm_t::template complex_array_t<4>;
    using error_array_t = scfd::arrays::array_nd<T, 1, memory_t>;

    std::size_t                              p1 = 0, p2 = 0, p3 = 0;
    fftm::test::detail::fftm_4d_test_options base_options;
    base_options.strategy  = strategy;
    base_options.run_all   = options.run_all;
    base_options.mode      = options.mode;
    base_options.nx        = options.nx;
    base_options.ny        = options.ny;
    base_options.nz        = options.nz;
    base_options.nw        = options.nw;
    base_options.p1        = options.p1;
    base_options.p2        = options.p2;
    base_options.p3        = options.p3;
    base_options.times     = options.times;
    base_options.enable_native_stage_timers = options.enable_native_stage_timers;
    base_options.use_4d_slab_native_xw_transpose = options.use_4d_slab_native_xw_transpose;
    base_options.use_4d_slab_native_xw_batched_peer_kernels =
        options.use_4d_slab_native_xw_batched_peer_kernels;
    base_options.use_4d_slab_native_xw_tensor_coalesced_kernels =
        options.use_4d_slab_native_xw_tensor_coalesced_kernels;
    base_options.use_4d_slab_native_xw_vector4_kernels = options.use_4d_slab_native_xw_vector4_kernels;
    base_options.use_4d_slab_native_xw_tiled_kernels = options.use_4d_slab_native_xw_tiled_kernels;
    base_options.use_4d_slab_native_xw_layout_stage = options.use_4d_slab_native_xw_layout_stage;
    base_options.use_4d_native_xw_direct_layout = options.use_4d_native_xw_direct_layout;
    base_options.use_4d_native_xw_chunked_transport = options.use_4d_native_xw_chunked_transport;
    base_options.native_xw_chunk_mib = options.native_xw_chunk_mib;
    base_options.native_xw_chunk_window = options.native_xw_chunk_window;
    base_options.use_4d_native_xw_compact_staging = options.use_4d_native_xw_compact_staging;
    base_options.use_4d_slab_native_work_area_alias = options.use_4d_slab_native_work_area_alias;
    base_options.use_4d_slab_native_wz_communication_layout =
        options.use_4d_slab_native_wz_communication_layout;
    base_options.slab_native_wz_plan_concurrency = options.slab_native_wz_plan_concurrency;
    base_options.use_4d_slab_native_wz_ready_pipeline = options.use_4d_slab_native_wz_ready_pipeline;
    base_options.use_4d_pencil_same_zw_peer_paired = options.use_4d_pencil_same_zw_peer_paired;
    base_options.use_4d_pencil_same_zw_native_layout = options.use_4d_pencil_same_zw_native_layout;
    base_options.use_4d_pencil_degenerate_xw_slab_path = options.use_4d_pencil_degenerate_xw_slab_path;
    base_options.use_4d_pencil_degenerate_local_transposes =
        options.use_4d_pencil_degenerate_local_transposes;
    base_options.use_4d_pencil_degenerate_same_xw_native =
        options.use_4d_pencil_degenerate_same_xw_native;
    base_options.use_4d_pencil_degenerate_wz_sliced_z_fft =
        options.use_4d_pencil_degenerate_wz_sliced_z_fft;
    base_options.use_4d_slab_native_xw_native_spectral_layout =
        options.use_4d_slab_native_xw_native_spectral_layout;
    std::tie( p1, p2, p3 ) = fftm::test::detail::choose_grid_4d( base_options, strategy, comm_info.num_procs );

    fftm::processor_grid grid;
    grid.init( p1, p2, p3 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz, options.nw );

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( grid, sizes );

    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    fftm_t distributed_fft( comm_info, log );
    distributed_fft.template init<4>( grid, sizes, fftm::test::detail::make_fftm_init_options( options ) );

    const auto  in_sizes   = distributed_fft.get_local_input_sizes_4d();
    const auto  out_sizes  = distributed_fft.get_local_spectral_sizes_4d();
    const auto &input_part = distributed_fft.input_partition();
    const bool use_native_spectral_layout = distributed_fft.uses_native_spectral_layout_4d();

    real_array_t work;
    hat_array_t  hat;

    work.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ), std::get<3>( in_sizes ) );
    hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ), std::get<3>( out_sizes ) );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    const T        normalization = T( 1 ) / static_cast<T>( options.nx * options.ny * options.nz * options.nw );
    std::vector<T> local_wall_times;
    std::vector<T> wall_times;
    local_wall_times.reserve( static_cast<std::size_t>( options.times ) );
    wall_times.reserve( static_cast<std::size_t>( options.times ) );
    T    max_norm          = T( 0 );
    bool validation_failed = false;

    for ( int iter = 0; iter < options.warmup; ++iter )
    {
        const unsigned long long seed = 0xABCDEF0123456789ull + static_cast<unsigned long long>( iter );

        for_each(
            fftm::test::detail::fill_random_real_4d_functor<T, idx_t, real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), static_cast<int>( input_part.start_z[myid_k] ), 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        runtime_api_t::device_synchronize();
        if ( use_native_spectral_layout )
        {
            auto hat_native = distributed_fft.make_native_spectral_view_4d( hat );
            distributed_fft.forward_native_spectral_4d( work, hat_native );
            distributed_fft.backward_native_spectral_4d( hat_native, work );
        }
        else
        {
            distributed_fft.forward( work, hat );
            distributed_fft.backward( hat, work );
        }
        runtime_api_t::device_synchronize();
    }

    for ( int iter = 0; iter < options.times; ++iter )
    {
        const unsigned long long seed = 0xABCDEF0123456789ull + static_cast<unsigned long long>( iter );

        for_each(
            fftm::test::detail::fill_random_real_4d_functor<T, idx_t, real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), static_cast<int>( input_part.start_z[myid_k] ), 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        if ( options.enable_native_stage_timers )
        {
            distributed_fft.begin_native_stage_timing_iteration( iter );
        }
        t0.record();
        if ( use_native_spectral_layout )
        {
            auto hat_native = distributed_fft.make_native_spectral_view_4d( hat );
            distributed_fft.forward_native_spectral_4d( work, hat_native );
            distributed_fft.backward_native_spectral_4d( hat_native, work );
        }
        else
        {
            distributed_fft.forward( work, hat );
            distributed_fft.backward( hat, work );
        }
        runtime_api_t::device_synchronize();
        t1.record();
        const T local_wall_ms  = static_cast<T>( t1.elapsed_time( t0 ) );
        const T global_wall_ms = comm_info.all_reduce_max( local_wall_ms );
        local_wall_times.push_back( local_wall_ms );
        wall_times.push_back( global_wall_ms );
        if ( options.enable_native_stage_timers )
        {
            append_4d_stage_timer_rows(
                options, comm_info.num_procs, comm_info.myid, iter, p1, p2, p3,
                static_cast<double>( global_wall_ms ),
                distributed_fft.slab_4d_native_work_area_alias_effective(),
                distributed_fft.native_stage_timings()
            );
            distributed_fft.end_native_stage_timing_iteration();
        }

        for_each(
            fftm::test::detail::scale_real_functor<T, idx_t, real_array_t>{ work, normalization },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        for_each(
            fftm::test::detail::overwrite_with_random_diff_square_4d_functor<T, idx_t, real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), static_cast<int>( input_part.start_z[myid_k] ), 0 },
            fftm::test::detail::make_range_4d<idx_t, rect_t>( work )
        );
        for_each.wait();

        const T local_diff_sq  = reduce( work.size(), work.raw_ptr(), T( 0 ) );
        const T global_diff_sq = comm_info.all_reduce_sum( local_diff_sq );
        const T diff_l2 =
            std::sqrt( global_diff_sq / static_cast<T>( options.nx * options.ny * options.nz * options.nw ) );
        if ( diff_l2 > max_norm )
            max_norm = diff_l2;

        if ( diff_l2 > options.epsilon && comm_info.myid == 0 )
        {
            log.warning_f(
                "strategy=%s, mode=%s, iteration=%d: l2_diff=%.8e exceeded epsilon=%.8e", fftm_t::strategy_name_4d(),
                fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ), iter, diff_l2, options.epsilon
            );
        }
        if ( diff_l2 > options.epsilon )
            validation_failed = true;
    }

    append_4d_wall_time_rows(
        options, comm_info.num_procs, comm_info.myid, p1, p2, p3, local_wall_times, wall_times
    );
    const auto stats = fftm::test::detail::compute_timing_statistics( wall_times );
    fftm::test::detail::log_tracked_memory_with_external_mpi(
        log, comm_info, distributed_fft, "benchmark=fftm-4d",
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( work ), fftm::test::detail::array_bytes( hat ) ) ),
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( work ), fftm::test::detail::array_bytes( hat ) ) )
    );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "benchmark=fftm-4d, strategy=%s, mode=%s, grid=(%zu,%zu,%zu), Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu, "
            "warmup=%d, times=%d: wall_time_scope=mpi-rank-max, avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
            fftm_t::strategy_name_4d(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ), p1, p2, p3,
            options.nx, options.ny, options.nz, options.nw, options.warmup, options.times, stats.mean, stats.stddev
        );

        std::ostringstream row;
        row << fftm::test::detail::csv_quote( "fftm-4d" ) << ',' << comm_info.num_procs << ','
            << fftm::test::detail::csv_quote( fftm_t::strategy_name_4d() ) << ','
            << fftm::test::detail::csv_quote( fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_4d ) ) << ','
            << ( options.use_fft_exec_no_sync ? 1 : 0 ) << ','
            << ( options.enable_native_stage_timers ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_transpose ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_batched_peer_kernels ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_tensor_coalesced_kernels ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_vector4_kernels ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_tiled_kernels ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_xw_layout_stage ? 1 : 0 ) << ','
            << ( options.use_4d_native_xw_direct_layout ? 1 : 0 ) << ','
            << ( options.use_4d_native_xw_chunked_transport ? 1 : 0 ) << ','
            << options.native_xw_chunk_mib << ',' << options.native_xw_chunk_window << ','
            << ( options.use_4d_native_xw_compact_staging ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_work_area_alias ? 1 : 0 ) << ','
            << ( distributed_fft.slab_4d_native_work_area_alias_effective() ? 1 : 0 ) << ','
            << ( options.use_4d_slab_native_wz_communication_layout ? 1 : 0 ) << ','
            << options.slab_native_wz_plan_concurrency << ','
            << ( options.use_4d_slab_native_wz_ready_pipeline ? 1 : 0 ) << ','
            << distributed_fft.shared_fft_work_size_bytes() << ','
            << distributed_fft.shared_transpose_work_size_bytes() << ','
            << distributed_fft.shared_same_xw_work_size_bytes() << ','
            << distributed_fft.shared_native_4d_sliced_work_size_bytes() << ','
            << distributed_fft.shared_stage1_alias_size_bytes() << ','
            << distributed_fft.shared_work_size_bytes() << ','
            << ( options.use_4d_slab_native_xw_native_spectral_layout ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_same_zw_peer_paired ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_same_zw_native_layout ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_degenerate_xw_slab_path ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_degenerate_local_transposes ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_degenerate_same_xw_native ? 1 : 0 ) << ','
            << ( options.use_4d_pencil_degenerate_wz_sliced_z_fft ? 1 : 0 ) << ','
            << p1 << ',' << p2 << ',' << p3 << ',' << options.nx << ',' << options.ny << ',' << options.nz << ','
            << options.nw << ',' << options.times << ',' << options.warmup << ',' << options.epsilon << ','
            << fftm::test::detail::csv_quote( "mpi-rank-max" ) << ',' << stats.mean << ',' << stats.stddev << ','
            << max_norm << ','
            << fftm::test::detail::csv_quote( options.directory );

        fftm::test::detail::append_csv_row(
            options.directory, "benchmark_fftm_4d.csv",
            "benchmark,num_gpus,strategy,mode,fft_exec_no_sync,native_stage_timers,slab_native_xw,slab_native_xw_batched_peer_kernels,"
            "slab_native_xw_tensor_coalesced_kernels,slab_native_xw_vector4_kernels,"
            "slab_native_xw_tiled_kernels,slab_native_xw_layout_stage,native_xw_direct_layout,"
            "native_xw_chunked_transport,native_xw_chunk_mib,native_xw_chunk_window,native_xw_compact_staging,"
            "slab_native_work_area_alias,slab_native_work_area_alias_effective,slab_native_wz_communication_layout,"
            "slab_native_wz_plan_concurrency,slab_native_wz_ready_pipeline,"
            "fft_work_bytes,"
            "transpose_work_bytes,same_xw_work_bytes,sliced_z_work_bytes,stage1_alias_bytes,shared_work_bytes,"
            "slab_native_xw_native_spectral_layout,pencil_same_zw_peer_paired,pencil_same_zw_native_layout,"
            "pencil_degenerate_xw_slab_path,pencil_degenerate_local_transposes,"
            "pencil_degenerate_same_xw_native,pencil_degenerate_wz_sliced_z_fft,"
            "p1,p2,p3,nx,ny,nz,nw,times,warmup,epsilon,"
            "wall_time_scope,avg_wall_ms,stddev_wall_ms,max_l2_diff,directory",
            row.str()
        );
    }

    if ( validation_failed && comm_info.myid == 0 )
    {
        log.error_f(
            "benchmark=fftm-4d validation failed: max_l2_diff=%.8e exceeded epsilon=%.8e", max_norm,
            options.epsilon
        );
    }

    return validation_failed ? 1 : 0;
}

template <fftm::mpi_transpose_3d_mode Mode>
int dispatch_mode(
    strategy_kind strategy, scfd::utils::log_mpi &log, const options_t &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    switch ( strategy )
    {
    case strategy_kind::pencil_pencil:
        return run_benchmark_case<fftm::strategy_4d_pencil_pencil_mpi<Mode>, Mode>( strategy, log, options, comm_info );
    case strategy_kind::slab_slab:
        return run_benchmark_case<fftm::strategy_4d_slab_slab_mpi<Mode>, Mode>( strategy, log, options, comm_info );
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
        fftm::test::detail::init_cuda_mpi_for_tests( log, comm_info );
        const options_t options = fftm::test::detail::parse_fftm_4d_benchmark_options<T>(
            argc, argv, comm_info.num_procs, "test_benchmark_fftm_4D.bin"
        );

        if ( options.p1 * options.p2 * options.p3 != static_cast<std::size_t>( comm_info.num_procs ) )
            throw std::logic_error( "P1*P2*P3 must equal the number of MPI processes" );

        if ( options.run_all )
        {
            int status = 0;
            status |= dispatch_strategy( strategy_kind::pencil_pencil, log, options, comm_info );
            status |= dispatch_strategy( strategy_kind::slab_slab, log, options, comm_info );
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
