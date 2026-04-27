#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define SCFD_ARRAYS_ORDINAL_TYPE ptrdiff_t

#include <scfd/backend/cuda.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/utils/log_std.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/system_timer_event.h>

#include <external_wrap/cufft_wrap_many.h>
#include <ffts.hpp>

#include "detail/fft_benchmark_common.h"
#include "detail/fft_benchmark_options.h"

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
using options_t     = fftm::test::detail::ffts_3d_benchmark_options<T>;

int run_benchmark( scfd::utils::log_std &log, const options_t &options )
{
    using ffts_t        = fftm::ffts<base_fft_t, backend_t>;
    using real_array_t  = typename ffts_t::template real_array_t<3>;
    using hat_array_t   = typename ffts_t::template complex_array_t<3>;
    using error_array_t = scfd::arrays::array_nd<T, 1, memory_t>;

    ffts_t fft;
    fft.init( options.nx, options.ny, options.nz );

    real_array_t work;
    hat_array_t  hat;

    work.init( options.nx, options.ny, options.nz );
    hat.init( options.nx, options.ny, options.nz / 2 + 1 );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    const T        normalization = T( 1 ) / static_cast<T>( options.nx * options.ny * options.nz );
    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );
    T max_norm = T( 0 );

    for ( int iter = 0; iter < options.warmup; ++iter )
    {
        const unsigned long long seed = 0x1234ABCDEF987654ull + static_cast<unsigned long long>( iter );

        for_each(
            fftm::test::detail::fill_random_real_3d_functor<T, idx_t, real_array_t>{ work, seed, 0, 0, 0 },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

        runtime_api_t::device_synchronize();
        fft.forward( work, hat );
        fft.backward( hat, work );
        runtime_api_t::device_synchronize();
    }

    for ( int iter = 0; iter < options.times; ++iter )
    {
        const unsigned long long seed = 0x1234ABCDEF987654ull + static_cast<unsigned long long>( iter );

        for_each(
            fftm::test::detail::fill_random_real_3d_functor<T, idx_t, real_array_t>{ work, seed, 0, 0, 0 },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

        scfd::utils::system_timer_event t0, t1;
        runtime_api_t::device_synchronize();
        t0.record();
        fft.forward( work, hat );
        fft.backward( hat, work );
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
                work, seed, 0, 0, 0 },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

        const T diff_l2 = std::sqrt( reduce( work.size(), work.raw_ptr(), T( 0 ) ) / static_cast<T>( work.size() ) );
        if ( diff_l2 > max_norm )
            max_norm = diff_l2;
        if ( diff_l2 > options.epsilon )
        {
            log.warning_f( "iteration=%d: l2_diff=%.8e exceeded epsilon=%.8e", iter, diff_l2, options.epsilon );
        }
    }

    const auto stats = fftm::test::detail::compute_timing_statistics( wall_times );

    log.info_f(
        "benchmark=ffts-3d, Nx=%zu, Ny=%zu, Nz=%zu, warmup=%d, times=%d: avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
        options.nx, options.ny, options.nz, options.warmup, options.times, stats.mean, stats.stddev
    );

    std::ostringstream row;
    row << fftm::test::detail::csv_quote( "ffts-3d" ) << ',' << 1 << ',' << fftm::test::detail::csv_quote( "cufft-3d" )
        << ',' << fftm::test::detail::csv_quote( "none" ) << ',' << 1 << ',' << 1 << ',' << 1 << ',' << options.nx
        << ',' << options.ny << ',' << options.nz << ',' << 0 << ',' << options.times << ',' << options.warmup << ','
        << options.epsilon << ',' << stats.mean << ',' << stats.stddev << ',' << max_norm << ','
        << fftm::test::detail::csv_quote( options.directory );

    fftm::test::detail::append_csv_row(
        options.directory, "benchmark_ffts_3d.csv",
        "benchmark,num_gpus,strategy,mode,p1,p2,p3,nx,ny,nz,nw,times,warmup,epsilon,avg_wall_ms,stddev_wall_ms,max_l2_"
        "diff,directory",
        row.str()
    );

    return 0;
}

} // namespace

int main( int argc, char *argv[] )
{
    scfd::utils::log_std log;

    try
    {
        scfd::utils::init_cuda_persistent( log, 0 );
        const options_t options =
            fftm::test::detail::parse_ffts_3d_benchmark_options<T>( argc, argv, "test_benchmark_ffts_3D.bin" );
        return run_benchmark( log, options );
    }
    catch ( const std::exception &ex )
    {
        log.error( scfd::utils::nested_exception_to_multistring( ex ) );
        return 1;
    }
}
