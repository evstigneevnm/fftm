#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/system_timer_event.h>

#include <fftm.hpp>
#include <fftm_autotune_measure_4d.hpp>
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

struct example_options
{
    std::size_t nx                = 16;
    std::size_t ny                = 16;
    std::size_t nz                = 16;
    std::size_t nw                = 16;
    int         times             = 1;
    int         warmup            = 0;
    int         autotune_warmup   = 1;
    int         autotune_times    = 3;
    bool        measure_autotune  = true;
    std::string cache_file        = "fftm_poisson_periodic_4d_autotune.env";
};

example_options parse_options( int argc, char **argv )
{
    if ( argc > 8 || argc == 3 || argc == 4 )
    {
        throw std::logic_error(
            "USAGE: poisson_periodic_4d_autotuned.bin [N] or "
            "[Nx Ny Nz Nw [cache_file [repetitions [warmup]]]]"
        );
    }

    example_options options;
    if ( argc == 2 )
    {
        options.nx = options.ny = options.nz = options.nw =
            static_cast<std::size_t>( std::strtoull( argv[1], nullptr, 10 ) );
    }
    else if ( argc >= 5 )
    {
        options.nx = static_cast<std::size_t>( std::strtoull( argv[1], nullptr, 10 ) );
        options.ny = static_cast<std::size_t>( std::strtoull( argv[2], nullptr, 10 ) );
        options.nz = static_cast<std::size_t>( std::strtoull( argv[3], nullptr, 10 ) );
        options.nw = static_cast<std::size_t>( std::strtoull( argv[4], nullptr, 10 ) );
    }
    if ( argc >= 6 )
        options.cache_file = argv[5];
    if ( argc >= 7 )
        options.times = std::atoi( argv[6] );
    if ( argc >= 8 )
        options.warmup = std::atoi( argv[7] );

    if ( options.nx < 8 || options.ny < 8 || options.nz < 8 || options.nw < 8 )
        throw std::logic_error( "Every 4D Poisson dimension must be at least 8" );
    if ( options.times <= 0 || options.warmup < 0 )
        throw std::logic_error( "The repetition count must be positive and warmup nonnegative" );
    if ( options.cache_file.empty() )
        throw std::logic_error( "The 4D autotune cache path must not be empty" );

    if ( const char *value = std::getenv( "FFTM_CPP_AUTOTUNE_MEASURE" ) )
    {
        if ( fftm::autotune::truthy( value ) )
            options.measure_autotune = true;
        else if ( fftm::autotune::falsey( value ) )
            options.measure_autotune = false;
        else
            throw std::logic_error( "FFTM_CPP_AUTOTUNE_MEASURE must be a boolean value" );
    }
    if ( const char *value = std::getenv( "FFTM_CPP_AUTOTUNE_WARMUP" ) )
        options.autotune_warmup = std::atoi( value );
    if ( const char *value = std::getenv( "FFTM_CPP_AUTOTUNE_TIMES" ) )
        options.autotune_times = std::atoi( value );
    if ( options.autotune_warmup < 0 || options.autotune_times <= 0 )
        throw std::logic_error( "C++ autotune requires warmup >= 0 and times > 0" );
    return options;
}

bool wrap_mpi_processes_over_gpus_enabled()
{
    const char *value = std::getenv( "FFTM_WRAP_PROCS_GPUS" );
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void apply_autotune_environment( fftm::autotune::autotune_options_4d &options )
{
    const auto assign = []( const char *name, std::string &target ) {
        if ( const char *value = std::getenv( name ) )
            target = value;
    };
    const auto assign_bool = []( const char *name, bool &target ) {
        const char *value = std::getenv( name );
        if ( value == nullptr )
            return;
        if ( fftm::autotune::truthy( value ) )
            target = true;
        else if ( fftm::autotune::falsey( value ) )
            target = false;
        else
            throw std::logic_error( std::string( name ) + " must be a boolean value" );
    };

    assign( "FFTM_CPP_AUTOTUNE_STRATEGY_4D", options.constraints_4d.strategy_4d );
    assign( "FFTM_CPP_AUTOTUNE_MODE", options.constraints_4d.mode );
    assign( "FFTM_CPP_AUTOTUNE_BACKEND_4D", options.constraints_4d.backend_4d );
    assign( "FFTM_CPP_AUTOTUNE_GRID_4D", options.constraints_4d.grid_4d );
    assign(
        "FFTM_CPP_AUTOTUNE_SPECTRAL_LAYOUT_4D",
        options.constraints_4d.spectral_layout_4d
    );
    assign(
        "FFTM_CPP_AUTOTUNE_PENCIL_PIPELINE_4D",
        options.constraints_4d.pencil_pipeline_4d
    );
    assign_bool( "FFTM_CPP_AUTOTUNE_STRICT_DEVICE_IDENTITY", options.strict_device_identity );
    assign_bool(
        "FFTM_CPP_AUTOTUNE_ALLOW_LEGACY_SIGNATURE",
        options.allow_legacy_hardware_signature
    );

    if ( const char *value = std::getenv( "FFTM_CPP_AUTOTUNE_ACCEPTED_SPECTRAL_LAYOUTS_4D" ) )
    {
        options.accepted_spectral_layouts.clear();
        for ( const auto &token : fftm::autotune::split( value, ',' ) )
        {
            if ( token.empty() )
                continue;
            options.accepted_spectral_layouts.push_back(
                fftm::autotune::parse_spectral_layout_4d( token )
            );
        }
        if ( options.accepted_spectral_layouts.empty() )
        {
            throw std::logic_error(
                "FFTM_CPP_AUTOTUNE_ACCEPTED_SPECTRAL_LAYOUTS_4D must list at least one layout"
            );
        }
    }
}

void apply_measurement_environment( fftm::autotune::measured_4d_options &options )
{
    const auto assign_bool = []( const char *name, bool &target ) {
        const char *value = std::getenv( name );
        if ( value == nullptr )
            return;
        if ( fftm::autotune::truthy( value ) )
            target = true;
        else if ( fftm::autotune::falsey( value ) )
            target = false;
        else
            throw std::logic_error( std::string( name ) + " must be a boolean value" );
    };

    assign_bool( "FFTM_CPP_AUTOTUNE_INCLUDE_SLAB_SLAB", options.include_slab_slab );
    assign_bool( "FFTM_CPP_AUTOTUNE_INCLUDE_PENCIL_PENCIL", options.include_pencil_pencil );
    assign_bool( "FFTM_CPP_AUTOTUNE_INCLUDE_P2P_WAITANY", options.include_p2p_waitany );
    assign_bool( "FFTM_CPP_AUTOTUNE_INCLUDE_ALLTOALLV", options.include_alltoallv );
    assign_bool(
        "FFTM_CPP_AUTOTUNE_INCLUDE_SLAB_CREDIT_FALLBACK",
        options.include_slab_credit_fallback
    );
    assign_bool(
        "FFTM_CPP_AUTOTUNE_CONTINUE_ON_CANDIDATE_ERROR",
        options.continue_on_candidate_error
    );
    assign_bool( "FFTM_CPP_AUTOTUNE_LOG_CANDIDATE_MEMORY", options.log_candidate_memory );
    assign_bool(
        "FFTM_CPP_AUTOTUNE_VERIFY_MEMORY_RECOVERY",
        options.verify_candidate_memory_recovery
    );
    if ( const char *value = std::getenv( "FFTM_CPP_AUTOTUNE_MEMORY_RECOVERY_TOLERANCE_MIB" ) )
    {
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull( value, &end, 10 );
        const std::size_t mib = static_cast<std::size_t>( 1024 ) * static_cast<std::size_t>( 1024 );
        if ( value[0] == '\0' || end == nullptr || *end != '\0' ||
             parsed > static_cast<unsigned long long>( std::numeric_limits<std::size_t>::max() / mib ) )
        {
            throw std::logic_error(
                "FFTM_CPP_AUTOTUNE_MEMORY_RECOVERY_TOLERANCE_MIB must be a nonnegative integer"
            );
        }
        options.candidate_memory_recovery_tolerance_bytes = static_cast<std::size_t>( parsed ) * mib;
    }
}

template <class Array>
rect_t make_range( const Array &array )
{
    const auto size = array.size_nd();
    return rect_t(
        idx_t( 0, 0, 0, 0 ),
        idx_t(
            static_cast<int>( size[0] ), static_cast<int>( size[1] ),
            static_cast<int>( size[2] ), static_cast<int>( size[3] )
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
    real_t    hx;
    real_t    hy;
    real_t    hz;
    real_t    hw;
    real_t    x0;
    real_t    y0;
    real_t    z0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const real_t x = x0 + hx * static_cast<real_t>( idx[0] );
        const real_t y = y0 + hy * static_cast<real_t>( idx[1] );
        const real_t z = z0 + hz * static_cast<real_t>( idx[2] );
        const real_t w = hw * static_cast<real_t>( idx[3] );
        field( idx ) = real_t( 15 ) * manufactured_solution( x, y, z, w );
    }
};

template <class Spectrum>
struct solve_poisson_functor
{
    Spectrum spectrum;
    int      nx;
    int      ny;
    int      nz;
    int      start0;
    int      start1;
    int      start2;
    int      start3;
    bool     native_xzwy;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const int gx = native_xzwy ? start0 + idx[0] : start3 + idx[3];
        const int gy = native_xzwy ? start3 + idx[3] : start0 + idx[0];
        const int gz = start1 + idx[1];
        const int gw = start2 + idx[2];
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
    real_t    hx;
    real_t    hy;
    real_t    hz;
    real_t    hw;
    real_t    x0;
    real_t    y0;
    real_t    z0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const real_t x = x0 + hx * static_cast<real_t>( idx[0] );
        const real_t y = y0 + hy * static_cast<real_t>( idx[1] );
        const real_t z = z0 + hz * static_cast<real_t>( idx[2] );
        const real_t w = hw * static_cast<real_t>( idx[3] );
        const real_t error = field( idx ) - manufactured_solution( x, y, z, w );
        field( idx ) = error * error;
    }
};

template <class RealArray>
struct exact_square_functor
{
    RealArray field;
    real_t    hx;
    real_t    hy;
    real_t    hz;
    real_t    hw;
    real_t    x0;
    real_t    y0;
    real_t    z0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const real_t x = x0 + hx * static_cast<real_t>( idx[0] );
        const real_t y = y0 + hy * static_cast<real_t>( idx[1] );
        const real_t z = z0 + hz * static_cast<real_t>( idx[2] );
        const real_t w = hw * static_cast<real_t>( idx[3] );
        const real_t exact = manufactured_solution( x, y, z, w );
        field( idx ) = exact * exact;
    }
};

fftm::global_sizes make_global_sizes( const example_options &options )
{
    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz, options.nw );
    return sizes;
}

template <class Strategy4D>
int run_poisson(
    const example_options &app_options, const fftm::autotune::selected_4d_config &selected,
    const comm_t &comm, scfd::utils::log_mpi &log
)
{
    using fftm_t = fftm::fftm<
        fft_backend_t, comm_t, backend_t,
        fftm::strategy_3d_slab_pencil<fftm::mpi_transpose_3d_mode::alltoallv>,
        scfd::utils::log_mpi, Strategy4D>;
    using real_array_t = typename fftm_t::template real_array_t<4>;
    using hat_array_t  = typename fftm_t::template complex_array_t<4>;

    const fftm::global_sizes sizes = make_global_sizes( app_options );
    fftm_t transform( comm, log );
    fftm::autotune::init_autotuned_4d_plan( transform, selected, sizes );

    fftm::fft_partitioning<comm_t> partitioning( comm );
    partitioning.init( selected.grid, sizes );
    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    const auto input_sizes    = transform.get_local_input_sizes_4d();
    const auto spectral_sizes = transform.get_local_spectral_sizes_4d();
    const auto spectral_start = transform.get_local_spectral_starts_4d();
    const auto &input_part    = transform.input_partition();

    real_array_t field;
    hat_array_t  spectrum_storage;
    fftm::autotune::init_autotuned_4d_data_arrays(
        field, spectrum_storage, selected, input_sizes, spectral_sizes
    );

    const real_t two_pi = real_t( 6.283185307179586476925286766559 );
    const real_t hx = two_pi / static_cast<real_t>( app_options.nx );
    const real_t hy = two_pi / static_cast<real_t>( app_options.ny );
    const real_t hz = two_pi / static_cast<real_t>( app_options.nz );
    const real_t hw = two_pi / static_cast<real_t>( app_options.nw );
    const real_t x0 = hx * static_cast<real_t>( input_part.start_x[myid_i] );
    const real_t y0 = hy * static_cast<real_t>( input_part.start_y[myid_j] );
    const real_t z0 = hz * static_cast<real_t>( input_part.start_z[myid_k] );
    const real_t normalization = real_t( 1 ) /
        ( static_cast<real_t>( app_options.nx ) * static_cast<real_t>( app_options.ny ) *
          static_cast<real_t>( app_options.nz ) * static_cast<real_t>( app_options.nw ) );
    const bool native_xzwy = transform.uses_native_spectral_layout_4d();

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    const auto fill_rhs = [&]() {
        for_each(
            fill_rhs_functor<real_array_t>{ field, hx, hy, hz, hw, x0, y0, z0 },
            make_range( field )
        );
        for_each.wait();
    };

    real_t elapsed_ms = real_t( 0 );
    for ( int iteration = 0; iteration < app_options.warmup + app_options.times; ++iteration )
    {
        fill_rhs();
        runtime_api_t::device_synchronize();
        scfd::utils::system_timer_event begin, end;
        if ( iteration >= app_options.warmup )
            begin.record();

        if ( native_xzwy )
        {
            auto spectrum = transform.make_native_spectral_view_4d( spectrum_storage );
            transform.forward_native_spectral_4d( field, spectrum );
            for_each(
                solve_poisson_functor<decltype( spectrum )>{
                    spectrum, static_cast<int>( app_options.nx ),
                    static_cast<int>( app_options.ny ), static_cast<int>( app_options.nz ),
                    static_cast<int>( std::get<0>( spectral_start ) ),
                    static_cast<int>( std::get<1>( spectral_start ) ),
                    static_cast<int>( std::get<2>( spectral_start ) ),
                    static_cast<int>( std::get<3>( spectral_start ) ), true },
                make_range( spectrum )
            );
            for_each.wait();
            transform.backward_native_spectral_4d( spectrum, field );
        }
        else
        {
            transform.forward( field, spectrum_storage );
            for_each(
                solve_poisson_functor<hat_array_t>{
                    spectrum_storage, static_cast<int>( app_options.nx ),
                    static_cast<int>( app_options.ny ), static_cast<int>( app_options.nz ),
                    static_cast<int>( std::get<0>( spectral_start ) ),
                    static_cast<int>( std::get<1>( spectral_start ) ),
                    static_cast<int>( std::get<2>( spectral_start ) ),
                    static_cast<int>( std::get<3>( spectral_start ) ), false },
                make_range( spectrum_storage )
            );
            for_each.wait();
            transform.backward( spectrum_storage, field );
        }
        for_each( scale_functor<real_array_t>{ field, normalization }, make_range( field ) );
        for_each.wait();

        runtime_api_t::device_synchronize();
        if ( iteration >= app_options.warmup )
        {
            end.record();
            elapsed_ms += static_cast<real_t>( end.elapsed_time( begin ) );
        }
    }

    for_each(
        error_square_functor<real_array_t>{ field, hx, hy, hz, hw, x0, y0, z0 },
        make_range( field )
    );
    for_each.wait();
    const real_t error_sq = comm.all_reduce_sum(
        reduce( field.total_size(), field.raw_ptr(), real_t( 0 ) )
    );

    for_each(
        exact_square_functor<real_array_t>{ field, hx, hy, hz, hw, x0, y0, z0 },
        make_range( field )
    );
    for_each.wait();
    const real_t exact_sq = comm.all_reduce_sum(
        reduce( field.total_size(), field.raw_ptr(), real_t( 0 ) )
    );
    const real_t relative_l2 = std::sqrt( error_sq / exact_sq );

    if ( comm.myid == 0 )
    {
        log.info_f(
            "poisson_periodic_4d_autotuned: size=%zux%zux%zux%zu mpi=%d strategy=%s mode=%s "
            "grid=%zux%zux%zu layout=%s backend=%s device_aware_mpi=%d cache=%s source=%s "
            "avg_ms=%.6e rel_l2=%.6e",
            app_options.nx, app_options.ny, app_options.nz, app_options.nw,
            comm.num_procs, selected.strategy_4d.c_str(), selected.mode.c_str(),
            selected.grid.p1, selected.grid.p2, selected.grid.p3,
            fftm::autotune::value_or_empty(
                selected.config, "FFTM_AUTOTUNE_SPECTRAL_LAYOUT_4D"
            ).c_str(),
            fftm::device_backend::name(), fftm::device_backend::device_aware_mpi_enabled() ? 1 : 0,
            app_options.cache_file.c_str(), selected.source.c_str(),
            elapsed_ms / static_cast<real_t>( app_options.times ), relative_l2
        );
    }

    if ( relative_l2 > real_t( 1.0e-10 ) )
    {
        throw std::logic_error(
            "4D Poisson relative L2 error is too large: " + std::to_string( relative_l2 )
        );
    }
    return 0;
}

template <fftm::mpi_transpose_3d_mode Mode>
int dispatch_strategy(
    const example_options &options, const fftm::autotune::selected_4d_config &selected,
    const comm_t &comm, scfd::utils::log_mpi &log
)
{
    if ( selected.strategy_4d == "slab-slab" )
    {
        return run_poisson<fftm::strategy_4d_slab_slab_mpi<Mode>>(
            options, selected, comm, log
        );
    }
    if ( selected.strategy_4d == "pencil-pencil" )
    {
        return run_poisson<fftm::strategy_4d_pencil_pencil_mpi<Mode>>(
            options, selected, comm, log
        );
    }
    throw std::logic_error( "Unsupported FFTM 4D strategy '" + selected.strategy_4d + "'" );
}

int dispatch_mode(
    const example_options &options, const fftm::autotune::selected_4d_config &selected,
    const comm_t &comm, scfd::utils::log_mpi &log
)
{
    if ( selected.mode == "p2p-waitany" || selected.mode.empty() )
    {
        return dispatch_strategy<fftm::mpi_transpose_3d_mode::p2p_waitany>(
            options, selected, comm, log
        );
    }
    if ( selected.mode == "alltoallv" )
    {
        return dispatch_strategy<fftm::mpi_transpose_3d_mode::alltoallv>(
            options, selected, comm, log
        );
    }
    throw std::logic_error( "Unsupported FFTM 4D mode '" + selected.mode + "'" );
}

} // namespace

int main( int argc, char **argv )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    const comm_t                  comm = mpi.comm_world();
    scfd::utils::log_mpi          log;

    try
    {
        fftm::device_backend::init_mpi(
            log, comm, 0, wrap_mpi_processes_over_gpus_enabled()
        );
        const example_options app_options = parse_options( argc, argv );
        const fftm::global_sizes sizes = make_global_sizes( app_options );

        fftm::autotune::autotune_options_4d autotune_options;
        autotune_options.cache_file = app_options.cache_file;
        autotune_options.create_if_missing = true;
        autotune_options.mismatch_policy = fftm::autotune::cache_mismatch_policy::error;
        autotune_options.validate_hardware = true;
        autotune_options.requested_spectral_layout = fftm::fftm_4d_spectral_layout::native_xzwy;
        autotune_options.accepted_spectral_layouts = {
            fftm::fftm_4d_spectral_layout::public_yzwx,
            fftm::fftm_4d_spectral_layout::native_xzwy
        };
        autotune_options.device_aware_mpi = fftm::device_backend::device_aware_mpi_enabled();
        apply_autotune_environment( autotune_options );

        fftm::autotune::selected_4d_config selected;
        if ( app_options.measure_autotune )
        {
            fftm::autotune::measured_4d_options measured_options;
            measured_options.warmup = app_options.autotune_warmup;
            measured_options.iterations = app_options.autotune_times;
            apply_measurement_environment( measured_options );
            fftm::autotune::fftm_4d_candidate_evaluator<
                fft_backend_t, comm_t, backend_t, scfd::utils::log_mpi>
                evaluator( comm, sizes, log );
            selected = fftm::autotune::load_or_measure_4d_config<runtime_api_t>(
                comm, sizes, autotune_options, measured_options, evaluator
            );
        }
        else
        {
            selected = fftm::autotune::load_or_create_4d_config<runtime_api_t>(
                comm, sizes, autotune_options
            );
        }
        return dispatch_mode( app_options, selected, comm, log );
    }
    catch ( const std::exception &error )
    {
        log.error( scfd::utils::nested_exception_to_multistring( error ) );
        return 1;
    }
}
