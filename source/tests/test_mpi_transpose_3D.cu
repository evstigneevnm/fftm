#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI

#include <thrust/complex.h>

#include <scfd/backend/cuda.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>

#include <external_wrap/cufft_wrap.h>

#include "../detail/array_arrangers.h"
#include "../detail/mpi_transpose_3d.h"
#include "../fft_partitioning.h"

namespace
{

using T          = double;
using complex_t  = thrust::complex<T>;
using backend_t  = scfd::backend::cuda;
using runtime_api_t = fftm::wrap::cuda_runtime_api;
using memory_t   = backend_t::memory_type;
using reduce_t   = backend_t::reduce_type;
using for_each_t = backend_t::for_each_nd_type<3, int>;
using idx_t      = scfd::static_vec::vec<int, 3>;
using rect_t     = scfd::static_vec::rect<int, 3>;

using xyz_array_t  = scfd::arrays::tensor_array_nd<complex_t, 3, memory_t, scfd::arrays::custom_arranger_102_t>;
using xzy_array_t  = scfd::arrays::tensor_array_nd<complex_t, 3, memory_t, scfd::arrays::custom_arranger_201_t>;
using flag_array_t = scfd::arrays::array_nd<int, 3, memory_t, scfd::arrays::custom_arranger_201_t>;

__DEVICE_TAG__ complex_t make_value( int gx, int gy, int gz, int nx, int ny )
{
    const T real_part = static_cast<T>( 1 + gx + nx * ( gy + ny * gz ) );
    return complex_t( real_part, -real_part );
}

struct test_options
{
    fftm::mpi_transpose_3d_mode mode    = fftm::mpi_transpose_3d_mode::alltoallv;
    bool                        run_all = false;
    std::size_t                 nx      = 24;
    std::size_t                 ny      = 20;
    std::size_t                 nz      = 18;
    std::size_t                 p1      = 0;
    std::size_t                 p2      = 0;
};

std::pair<std::size_t, std::size_t> choose_default_grid( std::size_t num_procs )
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

    int argi = 1;
    while ( argi < argc )
    {
        const std::string arg = argv[argi];
        if ( arg == "--mode" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --mode" );

            const std::string mode_name = argv[argi + 1];
            if ( mode_name == "p2p-waitall" )
                options.mode = fftm::mpi_transpose_3d_mode::p2p_waitall;
            else if ( mode_name == "p2p-waitany" )
                options.mode = fftm::mpi_transpose_3d_mode::p2p_waitany;
            else if ( mode_name == "alltoallv" )
                options.mode = fftm::mpi_transpose_3d_mode::alltoallv;
            else if ( mode_name == "alltoallw" )
                options.mode = fftm::mpi_transpose_3d_mode::alltoallw;
            else if ( mode_name == "all" )
                options.run_all = true;
            else
                throw std::logic_error( "Unknown mode '" + mode_name + "'" );
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
            "USAGE: test_mpi_transpose_3D.bin [--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw|all] "
            "[--grid P1 P2] [Nx Ny Nz]"
        );
    }

    if ( options.p1 == 0 || options.p2 == 0 )
    {
        const auto grid = choose_default_grid( static_cast<std::size_t>( num_procs ) );
        options.p1      = grid.first;
        options.p2      = grid.second;
    }

    return options;
}

template <class Array>
rect_t make_range_for_array( const Array &array )
{
    const auto size = array.size_nd();
    return rect_t(
        idx_t( 0, 0, 0 ), idx_t( static_cast<int>( size[0] ), static_cast<int>( size[1] ), static_cast<int>( size[2] ) )
    );
}

template <class Idx, class Array>
struct fill_input_functor
{
    fill_input_functor( Array &_array, int _x0, int _y0, int _nx, int _ny )
        : array( _array ), x0( _x0 ), y0( _y0 ), nx( _nx ), ny( _ny )
    {
    }

    Array array;
    int   x0;
    int   y0;
    int   nx;
    int   ny;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        array( idx ) = make_value( x0 + idx[0], y0 + idx[1], idx[2], nx, ny );
    }
};

template <class Idx, class ArrayOut, class FlagArray>
struct verify_output_functor
{
    verify_output_functor( const ArrayOut &_array, FlagArray &_flags, int _x0, int _z0, int _nx, int _ny )
        : array( _array ), flags( _flags ), x0( _x0 ), z0( _z0 ), nx( _nx ), ny( _ny )
    {
    }

    ArrayOut  array;
    FlagArray flags;
    int       x0;
    int       z0;
    int       nx;
    int       ny;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const complex_t expected = make_value( x0 + idx[0], idx[2], z0 + idx[1], nx, ny );
        flags( idx )             = ( array( idx ) == expected ) ? 0 : 1;
    }
};

template <class ArrayOut>
void log_first_mismatch(
    scfd::utils::log_mpi &log, const ArrayOut &array, const flag_array_t &flags, int x0, int z0, int nx, int ny
)
{
    typename ArrayOut::view_type     out_view( array );
    typename flag_array_t::view_type flag_view( flags );
    const auto                       size = array.size_nd();

    for ( int i = 0; i < size[0]; ++i )
    {
        for ( int j = 0; j < size[1]; ++j )
        {
            for ( int k = 0; k < size[2]; ++k )
            {
                if ( flag_view( i, j, k ) == 0 )
                    continue;

                const complex_t expected = make_value( x0 + i, k, z0 + j, nx, ny );
                const complex_t actual   = out_view( i, j, k );
                log.error_f(
                    "first mismatch at local=(%d,%d,%d): actual=(%.17e, %.17e), expected=(%.17e, %.17e)", i, j, k,
                    actual.real(), actual.imag(), expected.real(), expected.imag()
                );
                out_view.release( false );
                flag_view.release( false );
                return;
            }
        }
    }

    out_view.release( false );
    flag_view.release( false );
}

int run_mode(
    scfd::utils::log_mpi &log, const test_options &options, fftm::mpi_transpose_3d_mode mode,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    using transpose_t =
        fftm::mpi_transpose_3d<
            complex_t, backend_t, scfd::communication::mpi_comm_info, scfd::utils::log_mpi, runtime_api_t>;

    fftm::processor_grid grid;
    grid.init( options.p1, options.p2 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( grid, sizes );

    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    fftm::partition input_dim, output_dim, unused_dim;
    std::tie( input_dim, output_dim, unused_dim ) = partitioning.get_partitioning_3D();

    xyz_array_t  input;
    xzy_array_t  output;
    flag_array_t flags;

    input.init( input_dim.size_x[myid_i], input_dim.size_y[myid_j], input_dim.size_z[0] );
    output.init( output_dim.size_x[myid_i], output_dim.size_z[myid_j], output_dim.size_y[0] );
    flags.init( output_dim.size_x[myid_i], output_dim.size_z[myid_j], output_dim.size_y[0] );

    for_each_t for_each;
    reduce_t   reduce;

    for_each(
        fill_input_functor<idx_t, xyz_array_t>(
            input, static_cast<int>( input_dim.start_x[myid_i] ), static_cast<int>( input_dim.start_y[myid_j] ),
            static_cast<int>( options.nx ), static_cast<int>( options.ny )
        ),
        make_range_for_array( input )
    );
    for_each.wait();

    transpose_t transpose( comm_info, log );
    transpose.init( input_dim, output_dim, myid_i, myid_j );
    comm_info.barrier();
    transpose.transpose_xyz_to_xzy( input, output, mode );
    comm_info.barrier();

    for_each(
        verify_output_functor<idx_t, xzy_array_t, flag_array_t>(
            output, flags, static_cast<int>( input_dim.start_x[myid_i] ),
            static_cast<int>( output_dim.start_z[myid_j] ), static_cast<int>( options.nx ),
            static_cast<int>( options.ny )
        ),
        make_range_for_array( output )
    );
    for_each.wait();

    const int local_errors  = reduce( flags.total_size(), flags.raw_ptr(), 0 );
    const int global_errors = comm_info.all_reduce_sum( local_errors );

    if ( global_errors != 0 && local_errors != 0 )
    {
        log_first_mismatch(
            log, output, flags, static_cast<int>( input_dim.start_x[myid_i] ),
            static_cast<int>( output_dim.start_z[myid_j] ), static_cast<int>( options.nx ),
            static_cast<int>( options.ny )
        );
    }

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "mode=%s, grid=(%zu,%zu), sizes=(%zu,%zu,%zu), global_errors=%d", fftm::mpi_transpose_3d_mode_name( mode ),
            options.p1, options.p2, options.nx, options.ny, options.nz, global_errors
        );
    }

    return ( global_errors == 0 ) ? 0 : 1;
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
        {
            throw std::logic_error( "P1*P2 must equal the number of MPI processes" );
        }

        int failed = 0;

        if ( options.run_all )
        {
            failed = std::max( failed, run_mode( log, options, fftm::mpi_transpose_3d_mode::p2p_waitall, comm_info ) );
            failed = std::max( failed, run_mode( log, options, fftm::mpi_transpose_3d_mode::p2p_waitany, comm_info ) );
            failed = std::max( failed, run_mode( log, options, fftm::mpi_transpose_3d_mode::alltoallv, comm_info ) );
            failed = std::max( failed, run_mode( log, options, fftm::mpi_transpose_3d_mode::alltoallw, comm_info ) );
        }
        else
        {
            failed = run_mode( log, options, options.mode, comm_info );
        }

        if ( comm_info.myid == 0 )
        {
            if ( failed == 0 )
                log.info( "PASSED" );
            else
                log.error( "FAILED" );
        }

        return failed;
    }
    catch ( const std::exception &e )
    {
        log.error( scfd::utils::nested_exception_to_multistring( e ) );
        return 1;
    }
}
