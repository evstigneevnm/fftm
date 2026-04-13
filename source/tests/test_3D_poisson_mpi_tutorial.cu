#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include <scfd/backend/cuda.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/system_timer_event.h>

#include <external_wrap/cufft_wrap_many.h>
#include <fftm.hpp>

#include "detail/poisson_3d_problem.h"

namespace
{

using T          = double;
using base_fft_t = fftm::wrap::cufft_wrap_many<T>;
using backend_t  = scfd::backend::cuda;
using memory_t   = backend_t::memory_type;
using reduce_t   = backend_t::reduce_type;
using for_each_t = backend_t::template for_each_nd_type<3, int>;
using idx_t      = scfd::static_vec::vec<int, 3>;
using rect_t     = scfd::static_vec::rect<int, 3>;

enum class strategy_kind
{
    slab_pencil,
    pencil_slab,
    pencil_pencil
};

struct test_options
{
    strategy_kind               strategy = strategy_kind::pencil_pencil;
    fftm::mpi_transpose_3d_mode mode     = fftm::mpi_transpose_3d_mode::alltoallv;
    std::size_t                 nx       = 128;
    std::size_t                 ny       = 128;
    std::size_t                 nz       = 128;
    std::size_t                 p1       = 0;
    std::size_t                 p2       = 0;
};

std::pair<std::size_t, std::size_t> choose_pencil_grid( std::size_t num_procs )
{
    std::size_t p1 = 1;
    for ( std::size_t d = 1; d * d <= num_procs; ++d )
    {
        if ( num_procs % d == 0 )
            p1 = d;
    }
    return std::make_pair( p1, num_procs / p1 );
}

test_options parse_options( int argc, char *argv[], int num_procs )
{
    test_options options;
    int          argi = 1;

    while ( argi < argc )
    {
        const std::string arg = argv[argi];
        if ( arg == "--strategy" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --strategy" );
            const std::string value = argv[argi + 1];
            if ( value == "slab-pencil" )
                options.strategy = strategy_kind::slab_pencil;
            else if ( value == "pencil-slab" )
                options.strategy = strategy_kind::pencil_slab;
            else if ( value == "pencil-pencil" )
                options.strategy = strategy_kind::pencil_pencil;
            else
                throw std::logic_error( "Unknown strategy '" + value + "'" );
            argi += 2;
        }
        else if ( arg == "--mode" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --mode" );
            const std::string value = argv[argi + 1];
            if ( value == "p2p-waitall" )
                options.mode = fftm::mpi_transpose_3d_mode::p2p_waitall;
            else if ( value == "p2p-waitany" )
                options.mode = fftm::mpi_transpose_3d_mode::p2p_waitany;
            else if ( value == "alltoallv" )
                options.mode = fftm::mpi_transpose_3d_mode::alltoallv;
            else if ( value == "alltoallw" )
                options.mode = fftm::mpi_transpose_3d_mode::alltoallw;
            else
                throw std::logic_error( "Unknown mode '" + value + "'" );
            argi += 2;
        }
        else if ( arg == "--grid" )
        {
            if ( argi + 2 >= argc )
                throw std::logic_error( "Missing values for --grid P1 P2" );
            options.p1 = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            options.p2 = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
            argi += 3;
        }
        else
        {
            break;
        }
    }

    if ( argc - argi == 3 )
    {
        options.nx = static_cast<std::size_t>( std::strtoull( argv[argi], NULL, 10 ) );
        options.ny = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
        options.nz = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
    }
    else if ( argc != argi )
    {
        throw std::logic_error(
            "USAGE: test_3D_poisson_mpi_tutorial.bin "
            "[--strategy slab-pencil|pencil-slab|pencil-pencil] "
            "[--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw] "
            "[--grid P1 P2] [Nx Ny Nz]"
        );
    }

    if ( options.p1 == 0 || options.p2 == 0 )
    {
        const auto pencil_grid = choose_pencil_grid( static_cast<std::size_t>( num_procs ) );
        if ( options.strategy == strategy_kind::slab_pencil )
        {
            options.p1 = static_cast<std::size_t>( num_procs );
            options.p2 = 1;
        }
        else if ( options.strategy == strategy_kind::pencil_slab )
        {
            options.p1 = 1;
            options.p2 = static_cast<std::size_t>( num_procs );
        }
        else
        {
            options.p1 = pencil_grid.first;
            options.p2 = pencil_grid.second;
        }
    }

    return options;
}

template <class Array>
rect_t make_range( const Array &array )
{
    const auto sz = array.size_nd();
    return rect_t( idx_t( 0, 0, 0 ), idx_t( static_cast<int>( sz[0] ), static_cast<int>( sz[1] ), static_cast<int>( sz[2] ) ) );
}

template <class ComplexArray>
struct solve_fourier_in_place_functor
{
    ComplexArray spectrum;
    int          nx;
    int          ny;
    int          y_start;
    int          z_start;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const int kx = idx[0] <= nx / 2 ? idx[0] : idx[0] - nx;
        const int gy = y_start + idx[2];
        const int ky = gy <= ny / 2 ? gy : gy - ny;
        const int kz = z_start + idx[1];

        const T k2 = static_cast<T>( kx * kx + ky * ky + kz * kz );
        if ( k2 == T( 0 ) )
        {
            spectrum( idx ).x = T( 0 );
            spectrum( idx ).y = T( 0 );
            return;
        }

        const T scale = -T( 1 ) / k2;
        spectrum( idx ).x *= scale;
        spectrum( idx ).y *= scale;
    }
};

template <class RealArray>
struct scale_real_functor
{
    RealArray field;
    T         scale;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        field( idx ) *= scale;
    }
};

template <class RealArray>
struct square_real_functor
{
    RealArray field;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const T value = field( idx );
        field( idx ) = value * value;
    }
};

template <class Strategy>
int run_tutorial_case(
    scfd::utils::log_mpi                      &log,
    const test_options                        &options,
    const scfd::communication::mpi_comm_info  &comm_info
)
{
    using fftm_t       = fftm::fftm<base_fft_t, scfd::communication::mpi_comm_info, backend_t, Strategy, scfd::utils::log_mpi>;
    using real_array_t = typename fftm_t::template real_array_t<3>;
    using hat_array_t  = typename fftm_t::template complex_array_t<3>;

    fftm::processor_grid grid;
    grid.init( options.p1, options.p2 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( grid, sizes );
    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    fftm_t distributed_fft( comm_info, log );
    distributed_fft.template init<3>( grid, sizes );

    const auto in_sizes  = distributed_fft.get_local_input_sizes();
    const auto out_sizes = distributed_fft.get_local_output_sizes();
    const auto &input_part  = distributed_fft.input_partition();
    const auto &output_part = distributed_fft.output_partition();

    real_array_t field;
    hat_array_t  field_hat;
    field.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    field_hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ) );

    const T lx = fftm::test::detail::poisson_3d_problem<T>::domain_length();
    const T hx = lx / static_cast<T>( options.nx );
    const T hy = lx / static_cast<T>( options.ny );
    const T hz = lx / static_cast<T>( options.nz );
    const T cell_volume = hx * hy * hz;
    const T normalization = T( 1 ) / static_cast<T>( options.nx * options.ny * options.nz );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    // Step 1: fill the owned real-space chunk with the manufactured Poisson RHS.
    for_each(
        fftm::test::detail::fill_poisson_3d_rhs_functor<T, idx_t, real_array_t>{
            field,
            hx,
            hy,
            hz,
            hx * static_cast<T>( input_part.start_x[myid_i] ),
            hy * static_cast<T>( input_part.start_y[myid_j] ),
            T( 0 )
        },
        make_range( field )
    );
    for_each.wait();

    const T local_rhs_sum = reduce( field.size(), field.raw_ptr(), T( 0 ) );
    const T rhs_mean = comm_info.all_reduce_sum( local_rhs_sum ) / static_cast<T>( options.nx * options.ny * options.nz );
    if ( comm_info.myid == 0 && std::abs( rhs_mean ) > T( 1.0e-12 ) )
    {
        log.warning_f( "rhs_mean = %.8e", rhs_mean );
    }

    scfd::utils::system_timer_event t0, t1;
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    t0.record();

    // Step 2: FFT the RHS to Fourier space.
    distributed_fft.forward( field, field_hat );

    // Step 3: solve -k^2 u_hat = f_hat directly on the local spectral chunk.
    for_each(
        solve_fourier_in_place_functor<hat_array_t>{
            field_hat,
            static_cast<int>( options.nx ),
            static_cast<int>( options.ny ),
            static_cast<int>( output_part.start_y[myid_i] ),
            static_cast<int>( output_part.start_z[myid_j] )
        },
        make_range( field_hat )
    );
    for_each.wait();

    // Step 4: inverse FFT back to real space and apply CUFFT normalization.
    distributed_fft.backward( field_hat, field );
    for_each( scale_real_functor<real_array_t>{ field, normalization }, make_range( field ) );
    for_each.wait();

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    t1.record();
    const T wall_ms = static_cast<T>( t1.elapsed_time( t0 ) );

    // Step 5: reuse the solution buffer to accumulate the global L2 norm.
    for_each( square_real_functor<real_array_t>{ field }, make_range( field ) );
    for_each.wait();

    const T local_l2_sq  = reduce( field.size(), field.raw_ptr(), T( 0 ) ) * cell_volume;
    const T global_l2_sq = comm_info.all_reduce_sum( local_l2_sq );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "strategy=%s, mode=%s, Nx=%zu, Ny=%zu, Nz=%zu: solution_l2=%.8e, wall_ms=%.8e",
            fftm_t::strategy_name(),
            fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_3d ),
            options.nx,
            options.ny,
            options.nz,
            std::sqrt( global_l2_sq ),
            wall_ms
        );
    }

    return 0;
}

template <fftm::mpi_transpose_3d_mode Mode>
int run_for_strategy_kind(
    strategy_kind                              strategy,
    scfd::utils::log_mpi                      &log,
    const test_options                        &options,
    const scfd::communication::mpi_comm_info  &comm_info
)
{
    switch ( strategy )
    {
        case strategy_kind::slab_pencil:
            return run_tutorial_case<fftm::strategy_3d_slab_pencil<Mode>>( log, options, comm_info );
        case strategy_kind::pencil_slab:
            return run_tutorial_case<fftm::strategy_3d_pencil_slab<Mode>>( log, options, comm_info );
        case strategy_kind::pencil_pencil:
            return run_tutorial_case<fftm::strategy_3d_pencil_pencil<Mode>>( log, options, comm_info );
    }
    return 1;
}

int dispatch_mode(
    strategy_kind                              strategy,
    scfd::utils::log_mpi                      &log,
    const test_options                        &options,
    const scfd::communication::mpi_comm_info  &comm_info
)
{
    switch ( options.mode )
    {
        case fftm::mpi_transpose_3d_mode::p2p_waitall:
            return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::p2p_waitall>( strategy, log, options, comm_info );
        case fftm::mpi_transpose_3d_mode::p2p_waitany:
            return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::p2p_waitany>( strategy, log, options, comm_info );
        case fftm::mpi_transpose_3d_mode::alltoallv:
            return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::alltoallv>( strategy, log, options, comm_info );
        case fftm::mpi_transpose_3d_mode::alltoallw:
            return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::alltoallw>( strategy, log, options, comm_info );
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

        const test_options options = parse_options( argc, argv, comm_info.num_procs );
        if ( options.p1 * options.p2 != static_cast<std::size_t>( comm_info.num_procs ) )
            throw std::logic_error( "P1*P2 must equal the number of MPI processes" );

        return dispatch_mode( options.strategy, log, options, comm_info );
    }
    catch ( const std::exception &e )
    {
        log.error( e.what() );
        return 1;
    }
}
