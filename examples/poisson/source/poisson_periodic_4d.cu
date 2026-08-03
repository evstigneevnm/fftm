#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>

#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/system_timer_event.h>

#include <fftm.hpp>
#include <fftm_backend.hpp>

namespace
{

using real_t        = double;
using fft_backend_t = fftm::device_backend::fft<real_t>;
using runtime_api_t = typename fft_backend_t::runtime_api;
using backend_t     = fftm::device_backend::scfd_backend;
using comm_t        = scfd::communication::mpi_comm_info;
using reduce_t      = typename backend_t::reduce_type;
using for_each_t    = typename backend_t::template for_each_nd_type<4, int>;
using idx_t         = scfd::static_vec::vec<int, 4>;
using rect_t        = scfd::static_vec::rect<int, 4>;
using fftm_t        = fftm::fftm<
    fft_backend_t, comm_t, backend_t,
    fftm::strategy_3d_slab_pencil<fftm::mpi_transpose_3d_mode::alltoallv>, scfd::utils::log_mpi,
    fftm::strategy_4d_slab_slab_mpi<fftm::mpi_transpose_3d_mode::p2p_waitany>>;
using real_array_t = typename fftm_t::template real_array_t<4>;
using hat_array_t  = typename fftm_t::template complex_array_t<4>;

struct example_options
{
    std::size_t size  = 16;
    int         times = 1;
};

example_options parse_options( int argc, char **argv )
{
    if ( argc > 3 )
        throw std::logic_error( "USAGE: poisson_periodic_4d.bin [side_length] [repetitions]" );

    example_options options;
    if ( argc >= 2 )
        options.size = static_cast<std::size_t>( std::strtoull( argv[1], nullptr, 10 ) );
    if ( argc >= 3 )
        options.times = std::atoi( argv[2] );

    if ( options.size < 8 )
        throw std::logic_error( "The 4D Poisson side length must be at least 8" );
    if ( options.times <= 0 )
        throw std::logic_error( "The repetition count must be positive" );
    return options;
}

bool wrap_mpi_processes_over_gpus_enabled()
{
    const char *value = std::getenv( "FFTM_WRAP_PROCS_GPUS" );
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

template <class Array>
rect_t make_range( const Array &array )
{
    const auto size = array.size_nd();
    return rect_t(
        idx_t( 0, 0, 0, 0 ),
        idx_t(
            static_cast<int>( size[0] ), static_cast<int>( size[1] ), static_cast<int>( size[2] ),
            static_cast<int>( size[3] )
        )
    );
}

__DEVICE_TAG__ real_t manufactured_solution( real_t x, real_t y, real_t z, real_t w )
{
    return sin( x ) * cos( real_t( 2 ) * y ) * sin( real_t( 3 ) * z ) * cos( w );
}

template <class RealArray>
struct fill_rhs_functor
{
    RealArray field;
    real_t    spacing;
    real_t    x0;
    real_t    y0;
    real_t    z0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const real_t x = x0 + spacing * static_cast<real_t>( idx[0] );
        const real_t y = y0 + spacing * static_cast<real_t>( idx[1] );
        const real_t z = z0 + spacing * static_cast<real_t>( idx[2] );
        const real_t w = spacing * static_cast<real_t>( idx[3] );
        field( idx ) = real_t( 15 ) * manufactured_solution( x, y, z, w );
    }
};

template <class NativeSpectrum>
struct solve_poisson_functor
{
    NativeSpectrum spectrum;
    int            nx;
    int            ny;
    int            nz;
    int            x_start;
    int            z_start;
    int            w_start;
    int            y_start;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const int gx = x_start + idx[0];
        const int gz = z_start + idx[1];
        const int gw = w_start + idx[2];
        const int gy = y_start + idx[3];
        const int kx = gx <= nx / 2 ? gx : gx - nx;
        const int ky = gy <= ny / 2 ? gy : gy - ny;
        const int kz = gz <= nz / 2 ? gz : gz - nz;
        const int kw = gw;
        const real_t k2 = static_cast<real_t>( kx * kx + ky * ky + kz * kz + kw * kw );

        if ( k2 == real_t( 0 ) )
        {
            spectrum( idx ).x = real_t( 0 );
            spectrum( idx ).y = real_t( 0 );
            return;
        }
        spectrum( idx ).x /= k2;
        spectrum( idx ).y /= k2;
    }
};

template <class RealArray>
struct scale_functor
{
    RealArray field;
    real_t    scale;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        field( idx ) *= scale;
    }
};

template <class RealArray>
struct error_square_functor
{
    RealArray field;
    real_t    spacing;
    real_t    x0;
    real_t    y0;
    real_t    z0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const real_t x = x0 + spacing * static_cast<real_t>( idx[0] );
        const real_t y = y0 + spacing * static_cast<real_t>( idx[1] );
        const real_t z = z0 + spacing * static_cast<real_t>( idx[2] );
        const real_t w = spacing * static_cast<real_t>( idx[3] );
        const real_t error = field( idx ) - manufactured_solution( x, y, z, w );
        field( idx ) = error * error;
    }
};

template <class RealArray>
struct exact_square_functor
{
    RealArray field;
    real_t    spacing;
    real_t    x0;
    real_t    y0;
    real_t    z0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const real_t x = x0 + spacing * static_cast<real_t>( idx[0] );
        const real_t y = y0 + spacing * static_cast<real_t>( idx[1] );
        const real_t z = z0 + spacing * static_cast<real_t>( idx[2] );
        const real_t w = spacing * static_cast<real_t>( idx[3] );
        const real_t exact = manufactured_solution( x, y, z, w );
        field( idx ) = exact * exact;
    }
};

int run( const example_options &options, const comm_t &comm, scfd::utils::log_mpi &log )
{
    fftm::processor_grid grid;
    grid.init( 1, static_cast<std::size_t>( comm.num_procs ), 1 );

    fftm::global_sizes sizes;
    sizes.init( options.size, options.size, options.size, options.size );

    fftm_t transform( comm, log );
    transform.init<4>(
        grid, sizes,
        fftm::production_options_4d(
            fftm::transform_strategy_4d_mpi::slab_slab, fftm::fftm_4d_spectral_layout::native_xzwy
        )
    );

    fftm::fft_partitioning<comm_t> partitioning( comm );
    partitioning.init( grid, sizes );
    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    const auto input_sizes    = transform.get_local_input_sizes_4d();
    const auto spectral_sizes = transform.get_local_spectral_sizes_4d();
    const auto spectral_start = transform.get_local_spectral_starts_4d();
    const auto &input_part    = transform.input_partition();

    real_array_t field;
    hat_array_t  spectrum_storage;
    field.init(
        std::get<0>( input_sizes ), std::get<1>( input_sizes ), std::get<2>( input_sizes ),
        std::get<3>( input_sizes )
    );
    spectrum_storage.init(
        std::get<0>( spectral_sizes ), std::get<1>( spectral_sizes ), std::get<2>( spectral_sizes ),
        std::get<3>( spectral_sizes )
    );
    auto spectrum = transform.make_native_spectral_view_4d( spectrum_storage );

    const real_t two_pi = real_t( 6.283185307179586476925286766559 );
    const real_t spacing = two_pi / static_cast<real_t>( options.size );
    const real_t x0 = spacing * static_cast<real_t>( input_part.start_x[myid_i] );
    const real_t y0 = spacing * static_cast<real_t>( input_part.start_y[myid_j] );
    const real_t z0 = spacing * static_cast<real_t>( input_part.start_z[myid_k] );
    const real_t side = static_cast<real_t>( options.size );
    const real_t normalization = real_t( 1 ) / ( side * side * side * side );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    real_t elapsed_ms = real_t( 0 );
    for ( int iteration = 0; iteration < options.times; ++iteration )
    {
        for_each( fill_rhs_functor<real_array_t>{ field, spacing, x0, y0, z0 }, make_range( field ) );
        for_each.wait();

        runtime_api_t::device_synchronize();
        scfd::utils::system_timer_event begin, end;
        begin.record();

        transform.forward_native_spectral_4d( field, spectrum );
        for_each(
            solve_poisson_functor<decltype( spectrum )>{
                spectrum, static_cast<int>( options.size ), static_cast<int>( options.size ),
                static_cast<int>( options.size ), static_cast<int>( std::get<0>( spectral_start ) ),
                static_cast<int>( std::get<1>( spectral_start ) ),
                static_cast<int>( std::get<2>( spectral_start ) ),
                static_cast<int>( std::get<3>( spectral_start ) ) },
            make_range( spectrum )
        );
        for_each.wait();
        transform.backward_native_spectral_4d( spectrum, field );
        for_each( scale_functor<real_array_t>{ field, normalization }, make_range( field ) );
        for_each.wait();

        runtime_api_t::device_synchronize();
        end.record();
        elapsed_ms += static_cast<real_t>( end.elapsed_time( begin ) );
    }

    for_each( error_square_functor<real_array_t>{ field, spacing, x0, y0, z0 }, make_range( field ) );
    for_each.wait();
    const real_t error_sq = comm.all_reduce_sum( reduce( field.total_size(), field.raw_ptr(), real_t( 0 ) ) );

    for_each( exact_square_functor<real_array_t>{ field, spacing, x0, y0, z0 }, make_range( field ) );
    for_each.wait();
    const real_t exact_sq = comm.all_reduce_sum( reduce( field.total_size(), field.raw_ptr(), real_t( 0 ) ) );
    const real_t relative_l2 = std::sqrt( error_sq / exact_sq );

    if ( comm.myid == 0 )
    {
        log.info_f(
            "poisson_periodic_4d: size=%zux%zux%zux%zu mpi=%d strategy=slab-slab mode=p2p-waitany "
            "layout=native_xzwy avg_ms=%.6e rel_l2=%.6e",
            options.size, options.size, options.size, options.size, comm.num_procs,
            elapsed_ms / static_cast<real_t>( options.times ), relative_l2
        );
    }

    if ( relative_l2 > real_t( 1.0e-10 ) )
        throw std::logic_error( "4D Poisson relative L2 error is too large: " + std::to_string( relative_l2 ) );
    return 0;
}

} // namespace

int main( int argc, char **argv )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    const comm_t                  comm = mpi.comm_world();
    scfd::utils::log_mpi          log;

    try
    {
        fftm::device_backend::init_mpi( log, comm, 0, wrap_mpi_processes_over_gpus_enabled() );
        return run( parse_options( argc, argv ), comm, log );
    }
    catch ( const std::exception &error )
    {
        log.error( scfd::utils::nested_exception_to_multistring( error ) );
        return 1;
    }
}
