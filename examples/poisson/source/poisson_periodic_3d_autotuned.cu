#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/system_timer_event.h>

#include <fftm.hpp>
#include <fftm_autotune_measure.hpp>
#include <fftm_backend.hpp>

namespace
{

using real_t        = double;
using fft_backend_t = fftm::device_backend::fft<real_t>;
using runtime_api_t = typename fft_backend_t::runtime_api;
using backend_t     = fftm::device_backend::scfd_backend;
using memory_t      = typename backend_t::memory_type;
using reduce_t      = typename backend_t::reduce_type;
using for_each_t    = typename backend_t::template for_each_nd_type<3, int>;
using idx_t         = scfd::static_vec::vec<int, 3>;
using rect_t        = scfd::static_vec::rect<int, 3>;

struct example_options
{
    std::size_t nx         = 128;
    std::size_t ny         = 128;
    std::size_t nz         = 128;
    int         warmup     = 1;
    int         times      = 3;
    int         autotune_warmup = 2;
    int         autotune_times = 5;
    bool        measure_autotune = true;
    std::string cache_file = "fftm_poisson_periodic_3d_autotune.env";
};

template <class Array>
rect_t make_range_3d( const Array &array )
{
    const auto size = array.size_nd();
    return rect_t( idx_t( 0, 0, 0 ), idx_t( static_cast<int>( size[0] ), static_cast<int>( size[1] ), static_cast<int>( size[2] ) ) );
}

template <class RealArray>
struct fill_periodic_rhs_functor
{
    RealArray rhs;
    real_t    hx;
    real_t    hy;
    real_t    hz;
    real_t    x0;
    real_t    y0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const real_t x = x0 + hx * static_cast<real_t>( idx[0] );
        const real_t y = y0 + hy * static_cast<real_t>( idx[1] );
        const real_t z = hz * static_cast<real_t>( idx[2] );
        const real_t u = sin( x ) * cos( real_t( 2 ) * y ) * sin( real_t( 3 ) * z );
        rhs( idx ) = real_t( 14 ) * u; // -laplacian(u) for modes (1, 2, 3).
    }
};

template <class ComplexArray>
struct solve_negative_laplacian_functor
{
    ComplexArray spectrum;
    int          nx;
    int          ny;
    int          y_start;
    int          z_start;
    bool         spectral_xyz_layout;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const int kx = idx[0] <= nx / 2 ? idx[0] : idx[0] - nx;
        const int gy = y_start + ( spectral_xyz_layout ? idx[1] : idx[2] );
        const int gz = z_start + ( spectral_xyz_layout ? idx[2] : idx[1] );
        const int ky = gy <= ny / 2 ? gy : gy - ny;
        const int kz = gz;

        const real_t k2 = static_cast<real_t>( kx * kx + ky * ky + kz * kz );
        if ( k2 == real_t( 0 ) )
        {
            spectrum( idx ).x = real_t( 0 );
            spectrum( idx ).y = real_t( 0 );
            return;
        }
        const real_t scale = real_t( 1 ) / k2;
        spectrum( idx ).x *= scale;
        spectrum( idx ).y *= scale;
    }
};

template <class RealArray>
struct scale_real_functor
{
    RealArray field;
    real_t    scale;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        field( idx ) *= scale;
    }
};

template <class RealArray>
struct manufactured_square_functor
{
    RealArray field;
    real_t    hx;
    real_t    hy;
    real_t    hz;
    real_t    x0;
    real_t    y0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const real_t x = x0 + hx * static_cast<real_t>( idx[0] );
        const real_t y = y0 + hy * static_cast<real_t>( idx[1] );
        const real_t z = hz * static_cast<real_t>( idx[2] );
        const real_t value = sin( x ) * cos( real_t( 2 ) * y ) * sin( real_t( 3 ) * z );
        field( idx ) = value * value;
    }
};

template <class RealArray>
struct manufactured_error_square_functor
{
    RealArray field;
    real_t    hx;
    real_t    hy;
    real_t    hz;
    real_t    x0;
    real_t    y0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const real_t x = x0 + hx * static_cast<real_t>( idx[0] );
        const real_t y = y0 + hy * static_cast<real_t>( idx[1] );
        const real_t z = hz * static_cast<real_t>( idx[2] );
        const real_t exact = sin( x ) * cos( real_t( 2 ) * y ) * sin( real_t( 3 ) * z );
        const real_t diff = field( idx ) - exact;
        field( idx ) = diff * diff;
    }
};

example_options parse_options( int argc, char **argv )
{
    example_options options;
    if ( argc == 2 )
    {
        options.nx = options.ny = options.nz = static_cast<std::size_t>( std::strtoull( argv[1], nullptr, 10 ) );
    }
    else if ( argc >= 4 )
    {
        options.nx = static_cast<std::size_t>( std::strtoull( argv[1], nullptr, 10 ) );
        options.ny = static_cast<std::size_t>( std::strtoull( argv[2], nullptr, 10 ) );
        options.nz = static_cast<std::size_t>( std::strtoull( argv[3], nullptr, 10 ) );
    }
    if ( argc >= 5 )
        options.cache_file = argv[4];
    if ( argc >= 6 )
        options.times = std::atoi( argv[5] );
    if ( argc >= 7 )
        options.warmup = std::atoi( argv[6] );
    if ( options.nx == 0 || options.ny == 0 || options.nz == 0 )
        throw std::logic_error( "Grid sizes must be positive" );
    if ( options.times <= 0 )
        throw std::logic_error( "times must be positive" );
    if ( options.warmup < 0 )
        throw std::logic_error( "warmup must be nonnegative" );
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

void apply_autotune_environment( fftm::autotune::autotune_options &options )
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

    assign( "FFTM_CPP_AUTOTUNE_STRATEGY_3D", options.constraints_3d.strategy_3d );
    assign( "FFTM_CPP_AUTOTUNE_MODE", options.constraints_3d.mode );
    assign( "FFTM_CPP_AUTOTUNE_BACKEND_3D", options.constraints_3d.backend_3d );
    assign( "FFTM_CPP_AUTOTUNE_GRID_3D", options.constraints_3d.grid_3d );
    assign( "FFTM_CPP_AUTOTUNE_PENCIL_LAYOUT", options.constraints_3d.pencil_layout );
    assign_bool( "FFTM_CPP_AUTOTUNE_STRICT_DEVICE_IDENTITY", options.strict_device_identity );
    assign_bool( "FFTM_CPP_AUTOTUNE_ALLOW_LEGACY_SIGNATURE", options.allow_legacy_hardware_signature );
}

void apply_measurement_environment( fftm::autotune::measured_3d_options &options )
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

    assign_bool( "FFTM_CPP_AUTOTUNE_INCLUDE_ALLTOALLV", options.include_alltoallv );
    assign_bool(
        "FFTM_CPP_AUTOTUNE_CONTINUE_ON_CANDIDATE_ERROR", options.continue_on_candidate_error
    );
    assign_bool( "FFTM_CPP_AUTOTUNE_INCLUDE_HIGH_MEMORY_OPT0", options.include_high_memory_opt0 );
    assign_bool( "FFTM_CPP_AUTOTUNE_PRODUCTION_CANDIDATES_ONLY", options.production_candidates_only );
    assign_bool( "FFTM_CPP_AUTOTUNE_LOG_CANDIDATE_MEMORY", options.log_candidate_memory );
    assign_bool(
        "FFTM_CPP_AUTOTUNE_VERIFY_MEMORY_RECOVERY", options.verify_candidate_memory_recovery
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
    if ( const char *value = std::getenv( "FFTM_CPP_AUTOTUNE_MAX_PENCIL_GRIDS" ) )
    {
        const unsigned long parsed = std::strtoul( value, nullptr, 10 );
        if ( parsed == 0 )
            throw std::logic_error( "FFTM_CPP_AUTOTUNE_MAX_PENCIL_GRIDS must be positive" );
        options.max_pencil_grids = static_cast<std::size_t>( parsed );
    }
}

bool wrap_mpi_processes_over_gpus_enabled()
{
    const char *value = std::getenv( "FFTM_WRAP_PROCS_GPUS" );
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

fftm::global_sizes make_global_sizes( const example_options &options )
{
    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );
    return sizes;
}

template <class Strategy>
int run_poisson(
    const example_options &app_options, const fftm::autotune::selected_3d_config &selected,
    const scfd::communication::mpi_comm_info &comm_info, scfd::utils::log_mpi &log
)
{
    using fftm_t =
        fftm::fftm<fft_backend_t, scfd::communication::mpi_comm_info, backend_t, Strategy, scfd::utils::log_mpi>;
    using real_array_t = typename fftm_t::template real_array_t<3>;
    using hat_array_t  = typename fftm_t::template complex_array_t<3>;

    fftm_t distributed_fft( comm_info, log );
    fftm::fftm_init_options init_options = selected.init_options;
    fftm::autotune::selected_3d_config selected_for_plan = selected;
    selected_for_plan.init_options = init_options;
    fftm::autotune::init_autotuned_3d_plan( distributed_fft, selected_for_plan, make_global_sizes( app_options ) );

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( selected.grid, make_global_sizes( app_options ) );
    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    const auto  in_sizes    = distributed_fft.get_local_input_sizes();
    const auto  out_sizes   = distributed_fft.get_local_output_sizes();
    const auto &input_part  = distributed_fft.input_partition();
    const auto &output_part = distributed_fft.output_partition();

    real_array_t rhs;
    hat_array_t  spectrum;
    fftm::autotune::init_autotuned_3d_data_arrays(
        rhs, spectrum, selected_for_plan, in_sizes, out_sizes
    );

    const real_t two_pi = real_t( 6.283185307179586476925286766559 );
    const real_t hx     = two_pi / static_cast<real_t>( app_options.nx );
    const real_t hy     = two_pi / static_cast<real_t>( app_options.ny );
    const real_t hz     = two_pi / static_cast<real_t>( app_options.nz );
    const real_t norm   = real_t( 1 ) / static_cast<real_t>( app_options.nx * app_options.ny * app_options.nz );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    auto fill_rhs = [&]() {
        for_each(
            fill_periodic_rhs_functor<real_array_t>{
                rhs, hx, hy, hz, hx * static_cast<real_t>( input_part.start_x[myid_i] ),
                hy * static_cast<real_t>( input_part.start_y[myid_j] ) },
            make_range_3d( rhs )
        );
        for_each.wait();
    };

    fill_rhs();

    const bool spectral_xyz_layout =
        fftm_t::strategy_family_3d == fftm::transform_strategy_3d::pencil_pencil && fftm_t::strategy_3d_optimized_layout;

    real_t timed_ms = real_t( 0 );
    const int iterations = app_options.warmup + app_options.times;
    for ( int iter = 0; iter < iterations; ++iter )
    {
        fill_rhs();
        runtime_api_t::device_synchronize();
        scfd::utils::system_timer_event t0, t1;
        if ( iter >= app_options.warmup )
            t0.record();

        distributed_fft.forward( rhs, spectrum );
        for_each(
            solve_negative_laplacian_functor<hat_array_t>{
                spectrum, static_cast<int>( app_options.nx ), static_cast<int>( app_options.ny ),
                static_cast<int>( output_part.start_y[myid_i] ), static_cast<int>( output_part.start_z[myid_j] ),
                spectral_xyz_layout },
            make_range_3d( spectrum )
        );
        for_each.wait();
        distributed_fft.backward( spectrum, rhs );
        for_each( scale_real_functor<real_array_t>{ rhs, norm }, make_range_3d( rhs ) );
        for_each.wait();

        runtime_api_t::device_synchronize();
        if ( iter >= app_options.warmup )
        {
            t1.record();
            timed_ms += static_cast<real_t>( t1.elapsed_time( t0 ) );
        }
    }

    const real_t x0 = hx * static_cast<real_t>( input_part.start_x[myid_i] );
    const real_t y0 = hy * static_cast<real_t>( input_part.start_y[myid_j] );
    for_each(
        manufactured_error_square_functor<real_array_t>{ rhs, hx, hy, hz, x0, y0 },
        make_range_3d( rhs )
    );
    for_each.wait();
    const real_t error_l2_sq = comm_info.all_reduce_sum( reduce( rhs.size(), rhs.raw_ptr(), real_t( 0 ) ) );
    for_each(
        manufactured_square_functor<real_array_t>{ rhs, hx, hy, hz, x0, y0 },
        make_range_3d( rhs )
    );
    for_each.wait();
    const real_t exact_l2_sq = comm_info.all_reduce_sum( reduce( rhs.size(), rhs.raw_ptr(), real_t( 0 ) ) );
    const real_t rel_l2 = std::sqrt( error_l2_sq / exact_l2_sq );
    const real_t avg_ms = timed_ms / static_cast<real_t>( app_options.times );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "poisson_periodic_3d_autotuned: size=%zux%zux%zu mpi=%d strategy=%s mode=%s grid=%zux%zu "
            "layout=%s pipeline=%s cache=%s source=%s avg_ms=%.6e rel_l2=%.6e",
            app_options.nx, app_options.ny, app_options.nz, comm_info.num_procs, selected.strategy_3d.c_str(),
            selected.mode.c_str(), selected.grid.p1, selected.grid.p2,
            fftm::autotune::value_or_empty( selected.config, "FFTM_AUTOTUNE_PENCIL_LAYOUT" ).c_str(),
            fftm::autotune::value_or_empty( selected.config, "FFTM_AUTOTUNE_PENCIL_PIPELINE" ).c_str(),
            app_options.cache_file.c_str(), selected.source.c_str(), avg_ms, rel_l2
        );
    }

    if ( rel_l2 > real_t( 1.0e-10 ) )
        throw std::logic_error( "Poisson relative L2 error is too large: " + std::to_string( rel_l2 ) );
    return 0;
}

template <fftm::mpi_transpose_3d_mode Mode>
int dispatch_strategy(
    const example_options &app_options, const fftm::autotune::selected_3d_config &selected,
    const scfd::communication::mpi_comm_info &comm_info, scfd::utils::log_mpi &log
)
{
    if ( selected.strategy_3d == "slab-pencil" )
        return run_poisson<fftm::strategy_3d_slab_pencil<Mode>>( app_options, selected, comm_info, log );
    if ( selected.strategy_3d == "pencil-slab" )
        return run_poisson<fftm::strategy_3d_pencil_slab<Mode>>( app_options, selected, comm_info, log );
    if ( selected.strategy_3d == "pencil-pencil" )
    {
        if ( selected.init_options.pencil_layout_3d == fftm::fftm_3d_pencil_layout::legacy )
        {
            return run_poisson<fftm::strategy_3d_pencil_pencil<Mode, false>>(
                app_options, selected, comm_info, log
            );
        }
        return run_poisson<fftm::strategy_3d_pencil_pencil<Mode>>( app_options, selected, comm_info, log );
    }
    throw std::logic_error( "Unsupported FFTM_AUTOTUNE_STRATEGY_3D='" + selected.strategy_3d + "'" );
}

int dispatch_mode(
    const example_options &app_options, const fftm::autotune::selected_3d_config &selected,
    const scfd::communication::mpi_comm_info &comm_info, scfd::utils::log_mpi &log
)
{
    if ( selected.mode == "p2p-waitany" || selected.mode.empty() )
    {
        return dispatch_strategy<fftm::mpi_transpose_3d_mode::p2p_waitany>( app_options, selected, comm_info, log );
    }
    if ( selected.mode == "p2p-waitall" )
    {
        return dispatch_strategy<fftm::mpi_transpose_3d_mode::p2p_waitall>( app_options, selected, comm_info, log );
    }
    if ( selected.mode == "alltoallv" )
    {
        return dispatch_strategy<fftm::mpi_transpose_3d_mode::alltoallv>( app_options, selected, comm_info, log );
    }
    if ( selected.mode == "alltoallw" )
    {
        return dispatch_strategy<fftm::mpi_transpose_3d_mode::alltoallw>( app_options, selected, comm_info, log );
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
        fftm::device_backend::init_mpi( log, comm_info, 0, wrap_mpi_processes_over_gpus_enabled() );

        const example_options app_options = parse_options( argc, argv );
        const fftm::global_sizes sizes = make_global_sizes( app_options );

        fftm::autotune::autotune_options autotune_options;
        autotune_options.cache_file        = app_options.cache_file;
        autotune_options.create_if_missing = true;
        autotune_options.mismatch_policy   = fftm::autotune::cache_mismatch_policy::error;
        autotune_options.validate_hardware = true;
        apply_autotune_environment( autotune_options );

        fftm::autotune::selected_3d_config selected;
        if ( app_options.measure_autotune )
        {
            fftm::autotune::measured_3d_options measured_options;
            measured_options.warmup = app_options.autotune_warmup;
            measured_options.iterations = app_options.autotune_times;
            apply_measurement_environment( measured_options );
            fftm::autotune::fftm_3d_candidate_evaluator<
                fft_backend_t, scfd::communication::mpi_comm_info, backend_t, scfd::utils::log_mpi>
                evaluator( comm_info, sizes, log );
            selected = fftm::autotune::load_or_measure_3d_config<runtime_api_t>(
                comm_info, sizes, autotune_options, measured_options, evaluator
            );
        }
        else
        {
            selected = fftm::autotune::load_or_create_3d_config<runtime_api_t>(
                comm_info, sizes, autotune_options
            );
        }

        return dispatch_mode( app_options, selected, comm_info, log );
    }
    catch ( const std::exception &e )
    {
        log.error( scfd::utils::nested_exception_to_multistring( e ) );
        return 1;
    }
}
