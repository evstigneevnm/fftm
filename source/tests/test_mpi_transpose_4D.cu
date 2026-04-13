#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>

#include <thrust/complex.h>

#include <scfd/backend/cuda.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/log_mpi.h>

#include "../detail/array_arrangers.h"
#include "../detail/direct_transpose_4d.h"
#include "../detail/mpi_transpose_4d.h"
#include "../fft_partitioning.h"

namespace
{

using T          = double;
using complex_t  = thrust::complex<T>;
using backend_t  = scfd::backend::cuda;
using memory_t   = backend_t::memory_type;
using reduce_t   = backend_t::reduce_type;
using for_each_t = backend_t::for_each_nd_type<4, int>;
using idx_t      = scfd::static_vec::vec<int, 4>;
using rect_t     = scfd::static_vec::rect<int, 4>;

using xyzw_array_t = scfd::arrays::tensor_array_nd<complex_t, 4, memory_t, scfd::arrays::custom_arranger_0123_t>;
using xywz_array_t = scfd::arrays::tensor_array_nd<complex_t, 4, memory_t, scfd::arrays::custom_arranger_0123_t>;
using xzwy_array_t = scfd::arrays::tensor_array_nd<complex_t, 4, memory_t, scfd::arrays::custom_arranger_0213_t>;
using yzwx_array_t = scfd::arrays::tensor_array_nd<complex_t, 4, memory_t, scfd::arrays::custom_arranger_3210_t>;

using xyzw_flag_t = scfd::arrays::tensor_array_nd<int, 4, memory_t, scfd::arrays::custom_arranger_0123_t>;
using xywz_flag_t = scfd::arrays::tensor_array_nd<int, 4, memory_t, scfd::arrays::custom_arranger_0123_t>;
using xzwy_flag_t = scfd::arrays::tensor_array_nd<int, 4, memory_t, scfd::arrays::custom_arranger_0213_t>;
using yzwx_flag_t = scfd::arrays::tensor_array_nd<int, 4, memory_t, scfd::arrays::custom_arranger_3210_t>;

struct test_options
{
    fftm::mpi_transpose_3d_mode mode = fftm::mpi_transpose_3d_mode::alltoallv;
    bool                        run_all = false;
    std::size_t                 nx = 12;
    std::size_t                 ny = 10;
    std::size_t                 nz = 8;
    std::size_t                 nw = 10;
    std::size_t                 p1 = 0;
    std::size_t                 p2 = 0;
    std::size_t                 p3 = 0;
};

std::tuple<std::size_t, std::size_t, std::size_t> choose_default_grid( std::size_t num_procs )
{
    std::size_t best_p1 = 1;
    std::size_t best_p2 = 1;
    std::size_t best_p3 = num_procs;
    std::size_t best_span = best_p3 - best_p1;

    for ( std::size_t p1 = 1; p1 <= num_procs; ++p1 )
    {
        if ( num_procs % p1 != 0 )
            continue;
        const std::size_t rem1 = num_procs / p1;
        for ( std::size_t p2 = 1; p2 <= rem1; ++p2 )
        {
            if ( rem1 % p2 != 0 )
                continue;
            const std::size_t p3 = rem1 / p2;
            const std::size_t max_dim = std::max( p1, std::max( p2, p3 ) );
            const std::size_t min_dim = std::min( p1, std::min( p2, p3 ) );
            const std::size_t span = max_dim - min_dim;

            if ( span < best_span )
            {
                best_span = span;
                best_p1 = p1;
                best_p2 = p2;
                best_p3 = p3;
            }
        }
    }

    return std::make_tuple( best_p1, best_p2, best_p3 );
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
            if ( argi + 3 >= argc )
                throw std::logic_error( "Missing values for --grid P1 P2 P3" );
            options.p1 = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            options.p2 = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
            options.p3 = static_cast<std::size_t>( std::strtoull( argv[argi + 3], NULL, 10 ) );
            argi += 4;
        }
        else
        {
            break;
        }
    }

    if ( argc - argi == 4 )
    {
        options.nx = static_cast<std::size_t>( std::strtoull( argv[argi], NULL, 10 ) );
        options.ny = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
        options.nz = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
        options.nw = static_cast<std::size_t>( std::strtoull( argv[argi + 3], NULL, 10 ) );
    }
    else if ( argc != argi )
    {
        throw std::logic_error(
            "USAGE: test_mpi_transpose_4D.bin [--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw|all] "
            "[--grid P1 P2 P3] [Nx Ny Nz Nw]"
        );
    }

    if ( options.p1 == 0 || options.p2 == 0 || options.p3 == 0 )
    {
        std::tie( options.p1, options.p2, options.p3 ) =
            choose_default_grid( static_cast<std::size_t>( num_procs ) );
    }

    return options;
}

template <class Array>
rect_t make_range_for_array( const Array &array )
{
    const auto size = array.size_nd();
    return rect_t(
        idx_t( 0, 0, 0, 0 ),
        idx_t(
            static_cast<int>( size[0] ),
            static_cast<int>( size[1] ),
            static_cast<int>( size[2] ),
            static_cast<int>( size[3] )
        )
    );
}

__DEVICE_TAG__ complex_t make_value( int gx, int gy, int gz, int gw, int nx, int ny, int nz )
{
    const T real_part = static_cast<T>( 1 + gx + nx * ( gy + ny * ( gz + nz * gw ) ) );
    return complex_t( real_part, -real_part );
}

template <class Idx, class Array>
struct fill_stage0_functor
{
    Array array;
    int   x0;
    int   y0;
    int   z0;
    int   nx;
    int   ny;
    int   nz;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        array( idx ) = make_value( x0 + idx[0], y0 + idx[1], z0 + idx[2], idx[3], nx, ny, nz );
    }
};

template <class Idx, class ArrayOut, class RefArray, class FlagArray>
struct verify_xywz_functor
{
    ArrayOut  array;
    RefArray  reference;
    FlagArray flags;
    int       x0;
    int       y0;
    int       w0;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const complex_t expected = reference( x0 + idx[0], y0 + idx[1], w0 + idx[2], idx[3] );
        flags( idx ) = ( array( idx ) == expected ) ? 0 : 1;
    }
};

template <class Idx, class ArrayOut, class RefArray, class FlagArray>
struct verify_xzwy_functor
{
    ArrayOut  array;
    RefArray  reference;
    FlagArray flags;
    int       x0;
    int       z0;
    int       w0;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const complex_t expected = reference( x0 + idx[0], z0 + idx[1], w0 + idx[2], idx[3] );
        flags( idx ) = ( array( idx ) == expected ) ? 0 : 1;
    }
};

template <class Idx, class ArrayOut, class RefArray, class FlagArray>
struct verify_yzwx_functor
{
    ArrayOut  array;
    RefArray  reference;
    FlagArray flags;
    int       y0;
    int       z0;
    int       w0;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const complex_t expected = reference( y0 + idx[0], z0 + idx[1], w0 + idx[2], idx[3] );
        flags( idx ) = ( array( idx ) == expected ) ? 0 : 1;
    }
};

template <class Idx, class ArrayOut, class RefArray, class FlagArray>
struct verify_xyzw_functor
{
    ArrayOut  array;
    RefArray  reference;
    FlagArray flags;
    int       x0;
    int       y0;
    int       z0;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        const complex_t expected = reference( x0 + idx[0], y0 + idx[1], z0 + idx[2], idx[3] );
        flags( idx ) = ( array( idx ) == expected ) ? 0 : 1;
    }
};

void log_first_mismatch_xywz(
    scfd::utils::log_mpi &log,
    const xywz_array_t   &array,
    const xywz_flag_t    &flags,
    const xywz_array_t   &reference,
    int                   x0,
    int                   y0,
    int                   w0
)
{
    typename xywz_array_t::view_type out_view( array );
    typename xywz_flag_t::view_type flag_view( flags );
    typename xywz_array_t::view_type ref_view( reference );
    const auto size = array.size_nd();

    for ( int i = 0; i < size[0]; ++i )
    for ( int j = 0; j < size[1]; ++j )
    for ( int k = 0; k < size[2]; ++k )
    for ( int l = 0; l < size[3]; ++l )
    {
        if ( flag_view( i, j, k, l ) == 0 )
            continue;
        const complex_t actual = out_view( i, j, k, l );
        const complex_t expected = ref_view( x0 + i, y0 + j, w0 + k, l );
        log.error_f(
            "xywz mismatch at local=(%d,%d,%d,%d): actual=(%.17e, %.17e), expected=(%.17e, %.17e)",
            i, j, k, l, actual.real(), actual.imag(), expected.real(), expected.imag()
        );
        out_view.release( false );
        flag_view.release( false );
        ref_view.release( false );
        return;
    }

    out_view.release( false );
    flag_view.release( false );
    ref_view.release( false );
}

void log_first_mismatch_xzwy(
    scfd::utils::log_mpi &log,
    const xzwy_array_t   &array,
    const xzwy_flag_t    &flags,
    const xzwy_array_t   &reference,
    int                   x0,
    int                   z0,
    int                   w0
)
{
    typename xzwy_array_t::view_type out_view( array );
    typename xzwy_flag_t::view_type flag_view( flags );
    typename xzwy_array_t::view_type ref_view( reference );
    const auto size = array.size_nd();

    for ( int i = 0; i < size[0]; ++i )
    for ( int j = 0; j < size[1]; ++j )
    for ( int k = 0; k < size[2]; ++k )
    for ( int l = 0; l < size[3]; ++l )
    {
        if ( flag_view( i, j, k, l ) == 0 )
            continue;
        const complex_t actual = out_view( i, j, k, l );
        const complex_t expected = ref_view( x0 + i, z0 + j, w0 + k, l );
        log.error_f(
            "xzwy mismatch at local=(%d,%d,%d,%d): actual=(%.17e, %.17e), expected=(%.17e, %.17e)",
            i, j, k, l, actual.real(), actual.imag(), expected.real(), expected.imag()
        );
        out_view.release( false );
        flag_view.release( false );
        ref_view.release( false );
        return;
    }

    out_view.release( false );
    flag_view.release( false );
    ref_view.release( false );
}

void log_first_mismatch_yzwx(
    scfd::utils::log_mpi &log,
    const yzwx_array_t   &array,
    const yzwx_flag_t    &flags,
    const yzwx_array_t   &reference,
    int                   y0,
    int                   z0,
    int                   w0
)
{
    typename yzwx_array_t::view_type out_view( array );
    typename yzwx_flag_t::view_type flag_view( flags );
    typename yzwx_array_t::view_type ref_view( reference );
    const auto size = array.size_nd();

    for ( int i = 0; i < size[0]; ++i )
    for ( int j = 0; j < size[1]; ++j )
    for ( int k = 0; k < size[2]; ++k )
    for ( int l = 0; l < size[3]; ++l )
    {
        if ( flag_view( i, j, k, l ) == 0 )
            continue;
        const complex_t actual = out_view( i, j, k, l );
        const complex_t expected = ref_view( y0 + i, z0 + j, w0 + k, l );
        log.error_f(
            "yzwx mismatch at local=(%d,%d,%d,%d): actual=(%.17e, %.17e), expected=(%.17e, %.17e)",
            i, j, k, l, actual.real(), actual.imag(), expected.real(), expected.imag()
        );
        out_view.release( false );
        flag_view.release( false );
        ref_view.release( false );
        return;
    }

    out_view.release( false );
    flag_view.release( false );
    ref_view.release( false );
}

void log_first_mismatch_xyzw(
    scfd::utils::log_mpi &log,
    const xyzw_array_t   &array,
    const xyzw_flag_t    &flags,
    const xyzw_array_t   &reference,
    int                   x0,
    int                   y0,
    int                   z0
)
{
    typename xyzw_array_t::view_type out_view( array );
    typename xyzw_flag_t::view_type flag_view( flags );
    typename xyzw_array_t::view_type ref_view( reference );
    const auto size = array.size_nd();

    for ( int i = 0; i < size[0]; ++i )
    for ( int j = 0; j < size[1]; ++j )
    for ( int k = 0; k < size[2]; ++k )
    for ( int l = 0; l < size[3]; ++l )
    {
        if ( flag_view( i, j, k, l ) == 0 )
            continue;
        const complex_t actual = out_view( i, j, k, l );
        const complex_t expected = ref_view( x0 + i, y0 + j, z0 + k, l );
        log.error_f(
            "xyzw mismatch at local=(%d,%d,%d,%d): actual=(%.17e, %.17e), expected=(%.17e, %.17e)",
            i, j, k, l, actual.real(), actual.imag(), expected.real(), expected.imag()
        );
        out_view.release( false );
        flag_view.release( false );
        ref_view.release( false );
        return;
    }

    out_view.release( false );
    flag_view.release( false );
    ref_view.release( false );
}

int check_xywz_stage(
    scfd::utils::log_mpi                        &log,
    const scfd::communication::mpi_comm_info    &comm_info,
    for_each_t                                  &for_each,
    reduce_t                                    &reduce,
    const xywz_array_t                          &array,
    const xywz_array_t                          &reference,
    int                                          x0,
    int                                          y0,
    int                                          w0,
    const std::string                           &stage_name,
    fftm::mpi_transpose_3d_mode                  mode,
    const test_options                          &options
)
{
    xywz_flag_t flags;
    const auto size = array.size_nd();
    flags.init( size[0], size[1], size[2], size[3] );

    for_each(
        verify_xywz_functor<idx_t, xywz_array_t, xywz_array_t, xywz_flag_t>{ array, reference, flags, x0, y0, w0 },
        make_range_for_array( array )
    );
    for_each.wait();

    const int local_errors = reduce( flags.total_size(), flags.raw_ptr(), 0 );
    const int global_errors = comm_info.all_reduce_sum( local_errors );

    if ( global_errors != 0 && local_errors != 0 )
        log_first_mismatch_xywz( log, array, flags, reference, x0, y0, w0 );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "mode=%s, stage=%s, grid=(%zu,%zu,%zu), sizes=(%zu,%zu,%zu,%zu), global_errors=%d",
            fftm::mpi_transpose_3d_mode_name( mode ),
            stage_name.c_str(),
            options.p1, options.p2, options.p3,
            options.nx, options.ny, options.nz, options.nw,
            global_errors
        );
    }

    return ( global_errors == 0 ) ? 0 : 1;
}

int check_xzwy_stage(
    scfd::utils::log_mpi                        &log,
    const scfd::communication::mpi_comm_info    &comm_info,
    for_each_t                                  &for_each,
    reduce_t                                    &reduce,
    const xzwy_array_t                          &array,
    const xzwy_array_t                          &reference,
    int                                          x0,
    int                                          z0,
    int                                          w0,
    const std::string                           &stage_name,
    fftm::mpi_transpose_3d_mode                  mode,
    const test_options                          &options
)
{
    xzwy_flag_t flags;
    const auto size = array.size_nd();
    flags.init( size[0], size[1], size[2], size[3] );

    for_each(
        verify_xzwy_functor<idx_t, xzwy_array_t, xzwy_array_t, xzwy_flag_t>{ array, reference, flags, x0, z0, w0 },
        make_range_for_array( array )
    );
    for_each.wait();

    const int local_errors = reduce( flags.total_size(), flags.raw_ptr(), 0 );
    const int global_errors = comm_info.all_reduce_sum( local_errors );

    if ( global_errors != 0 && local_errors != 0 )
        log_first_mismatch_xzwy( log, array, flags, reference, x0, z0, w0 );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "mode=%s, stage=%s, grid=(%zu,%zu,%zu), sizes=(%zu,%zu,%zu,%zu), global_errors=%d",
            fftm::mpi_transpose_3d_mode_name( mode ),
            stage_name.c_str(),
            options.p1, options.p2, options.p3,
            options.nx, options.ny, options.nz, options.nw,
            global_errors
        );
    }

    return ( global_errors == 0 ) ? 0 : 1;
}

int check_yzwx_stage(
    scfd::utils::log_mpi                        &log,
    const scfd::communication::mpi_comm_info    &comm_info,
    for_each_t                                  &for_each,
    reduce_t                                    &reduce,
    const yzwx_array_t                          &array,
    const yzwx_array_t                          &reference,
    int                                          y0,
    int                                          z0,
    int                                          w0,
    const std::string                           &stage_name,
    fftm::mpi_transpose_3d_mode                  mode,
    const test_options                          &options
)
{
    yzwx_flag_t flags;
    const auto size = array.size_nd();
    flags.init( size[0], size[1], size[2], size[3] );

    for_each(
        verify_yzwx_functor<idx_t, yzwx_array_t, yzwx_array_t, yzwx_flag_t>{ array, reference, flags, y0, z0, w0 },
        make_range_for_array( array )
    );
    for_each.wait();

    const int local_errors = reduce( flags.total_size(), flags.raw_ptr(), 0 );
    const int global_errors = comm_info.all_reduce_sum( local_errors );

    if ( global_errors != 0 && local_errors != 0 )
        log_first_mismatch_yzwx( log, array, flags, reference, y0, z0, w0 );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "mode=%s, stage=%s, grid=(%zu,%zu,%zu), sizes=(%zu,%zu,%zu,%zu), global_errors=%d",
            fftm::mpi_transpose_3d_mode_name( mode ),
            stage_name.c_str(),
            options.p1, options.p2, options.p3,
            options.nx, options.ny, options.nz, options.nw,
            global_errors
        );
    }

    return ( global_errors == 0 ) ? 0 : 1;
}

int check_xyzw_stage(
    scfd::utils::log_mpi                        &log,
    const scfd::communication::mpi_comm_info    &comm_info,
    for_each_t                                  &for_each,
    reduce_t                                    &reduce,
    const xyzw_array_t                          &array,
    const xyzw_array_t                          &reference,
    int                                          x0,
    int                                          y0,
    int                                          z0,
    const std::string                           &stage_name,
    fftm::mpi_transpose_3d_mode                  mode,
    const test_options                          &options
)
{
    xyzw_flag_t flags;
    const auto size = array.size_nd();
    flags.init( size[0], size[1], size[2], size[3] );

    for_each(
        verify_xyzw_functor<idx_t, xyzw_array_t, xyzw_array_t, xyzw_flag_t>{ array, reference, flags, x0, y0, z0 },
        make_range_for_array( array )
    );
    for_each.wait();

    const int local_errors = reduce( flags.total_size(), flags.raw_ptr(), 0 );
    const int global_errors = comm_info.all_reduce_sum( local_errors );

    if ( global_errors != 0 && local_errors != 0 )
        log_first_mismatch_xyzw( log, array, flags, reference, x0, y0, z0 );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "mode=%s, stage=%s, grid=(%zu,%zu,%zu), sizes=(%zu,%zu,%zu,%zu), global_errors=%d",
            fftm::mpi_transpose_3d_mode_name( mode ),
            stage_name.c_str(),
            options.p1, options.p2, options.p3,
            options.nx, options.ny, options.nz, options.nw,
            global_errors
        );
    }

    return ( global_errors == 0 ) ? 0 : 1;
}

int run_mode(
    scfd::utils::log_mpi                         &log,
    const test_options                           &options,
    fftm::mpi_transpose_3d_mode                   mode,
    const scfd::communication::mpi_comm_info     &comm_info
)
{
    using same_xy_t = fftm::detail::mpi_transpose_4d_same_xy<complex_t, backend_t, scfd::communication::mpi_comm_info, scfd::utils::log_mpi>;
    using same_xw_t = fftm::detail::mpi_transpose_4d_same_xw<complex_t, backend_t, scfd::communication::mpi_comm_info, scfd::utils::log_mpi>;
    using same_zw_t = fftm::detail::mpi_transpose_4d_same_zw<complex_t, backend_t, scfd::communication::mpi_comm_info, scfd::utils::log_mpi>;

    if ( options.nw % 2 != 0 )
        throw std::logic_error( "Nw must be even for the half-spectrum transpose test" );

    fftm::processor_grid grid;
    grid.init( options.p1, options.p2, options.p3 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz, options.nw );

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( grid, sizes );

    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    fftm::partition input_dim;
    fftm::partition transpose1_dim;
    fftm::partition transpose2_dim;
    fftm::partition transpose3_dim;
    std::tie( input_dim, transpose1_dim, transpose2_dim, transpose3_dim ) = partitioning.get_partitioning_4D();

    fftm::partition half_input_dim = input_dim;
    half_input_dim.size_w[0] = options.nw / 2 + 1;
    half_input_dim.compute_offsets( true );

    xyzw_array_t stage0_ref;
    xywz_array_t stage1_ref;
    xzwy_array_t stage2_ref;
    yzwx_array_t stage3_ref;

    stage0_ref.init( options.nx, options.ny, options.nz, options.nw / 2 + 1 );
    stage1_ref.init( options.nx, options.ny, options.nw / 2 + 1, options.nz );
    stage2_ref.init( options.nx, options.nz, options.nw / 2 + 1, options.ny );
    stage3_ref.init( options.ny, options.nz, options.nw / 2 + 1, options.nx );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    for_each(
        fill_stage0_functor<idx_t, xyzw_array_t>{
            stage0_ref,
            0,
            0,
            0,
            static_cast<int>( options.nx ),
            static_cast<int>( options.ny ),
            static_cast<int>( options.nz )
        },
        make_range_for_array( stage0_ref )
    );
    for_each.wait();

    fftm::detail::direct_transpose_4d reference_transpose( options.nx, options.ny, options.nz, options.nw / 2 + 1 );
    reference_transpose.xyzw_to_xywz( for_each, stage0_ref, stage1_ref );
    for_each.wait();
    reference_transpose.xywz_to_xzwy( for_each, stage1_ref, stage2_ref );
    for_each.wait();
    reference_transpose.xzwy_to_yzwx( for_each, stage2_ref, stage3_ref );
    for_each.wait();

    xyzw_array_t stage0_local;
    xywz_array_t stage1_local;
    xzwy_array_t stage2_local;
    yzwx_array_t stage3_local;
    xzwy_array_t stage2_back;
    xywz_array_t stage1_back;
    xyzw_array_t stage0_back;

    stage0_local.init(
        half_input_dim.size_x[myid_i],
        half_input_dim.size_y[myid_j],
        half_input_dim.size_z[myid_k],
        half_input_dim.size_w[0]
    );
    stage1_local.init(
        transpose1_dim.size_x[myid_i],
        transpose1_dim.size_y[myid_j],
        transpose1_dim.size_w[myid_k],
        transpose1_dim.size_z[0]
    );
    stage2_local.init(
        transpose2_dim.size_x[myid_i],
        transpose2_dim.size_z[myid_j],
        transpose2_dim.size_w[myid_k],
        transpose2_dim.size_y[0]
    );
    stage3_local.init(
        transpose3_dim.size_y[myid_i],
        transpose3_dim.size_z[myid_j],
        transpose3_dim.size_w[myid_k],
        transpose3_dim.size_x[0]
    );
    stage2_back.init(
        transpose2_dim.size_x[myid_i],
        transpose2_dim.size_z[myid_j],
        transpose2_dim.size_w[myid_k],
        transpose2_dim.size_y[0]
    );
    stage1_back.init(
        transpose1_dim.size_x[myid_i],
        transpose1_dim.size_y[myid_j],
        transpose1_dim.size_w[myid_k],
        transpose1_dim.size_z[0]
    );
    stage0_back.init(
        half_input_dim.size_x[myid_i],
        half_input_dim.size_y[myid_j],
        half_input_dim.size_z[myid_k],
        half_input_dim.size_w[0]
    );

    for_each(
        fill_stage0_functor<idx_t, xyzw_array_t>{
            stage0_local,
            static_cast<int>( half_input_dim.start_x[myid_i] ),
            static_cast<int>( half_input_dim.start_y[myid_j] ),
            static_cast<int>( half_input_dim.start_z[myid_k] ),
            static_cast<int>( options.nx ),
            static_cast<int>( options.ny ),
            static_cast<int>( options.nz )
        },
        make_range_for_array( stage0_local )
    );
    for_each.wait();

    same_xy_t same_xy( comm_info, log );
    same_xw_t same_xw( comm_info, log );
    same_zw_t same_zw( comm_info, log );

    same_xy.init( half_input_dim, transpose1_dim, myid_i, myid_j, myid_k );
    same_xw.init( transpose1_dim, transpose2_dim, myid_i, myid_j, myid_k );
    same_zw.init( transpose2_dim, transpose3_dim, myid_i, myid_j, myid_k );

    int failed = 0;

    comm_info.barrier();
    same_xy.transpose_xyzw_to_xywz( stage0_local, stage1_local, mode );
    comm_info.barrier();
    failed = std::max(
        failed,
        check_xywz_stage(
            log,
            comm_info,
            for_each,
            reduce,
            stage1_local,
            stage1_ref,
            static_cast<int>( transpose1_dim.start_x[myid_i] ),
            static_cast<int>( transpose1_dim.start_y[myid_j] ),
            static_cast<int>( transpose1_dim.start_w[myid_k] ),
            "xyzw->xywz",
            mode,
            options
        )
    );

    comm_info.barrier();
    same_xw.transpose_xywz_to_xzwy( stage1_local, stage2_local, mode );
    comm_info.barrier();
    failed = std::max(
        failed,
        check_xzwy_stage(
            log,
            comm_info,
            for_each,
            reduce,
            stage2_local,
            stage2_ref,
            static_cast<int>( transpose2_dim.start_x[myid_i] ),
            static_cast<int>( transpose2_dim.start_z[myid_j] ),
            static_cast<int>( transpose2_dim.start_w[myid_k] ),
            "xywz->xzwy",
            mode,
            options
        )
    );

    comm_info.barrier();
    same_zw.transpose_xzwy_to_yzwx( stage2_local, stage3_local, mode );
    comm_info.barrier();
    failed = std::max(
        failed,
        check_yzwx_stage(
            log,
            comm_info,
            for_each,
            reduce,
            stage3_local,
            stage3_ref,
            static_cast<int>( transpose3_dim.start_y[myid_i] ),
            static_cast<int>( transpose3_dim.start_z[myid_j] ),
            static_cast<int>( transpose3_dim.start_w[myid_k] ),
            "xzwy->yzwx",
            mode,
            options
        )
    );

    comm_info.barrier();
    same_zw.transpose_yzwx_to_xzwy( stage3_local, stage2_back, mode );
    comm_info.barrier();
    failed = std::max(
        failed,
        check_xzwy_stage(
            log,
            comm_info,
            for_each,
            reduce,
            stage2_back,
            stage2_ref,
            static_cast<int>( transpose2_dim.start_x[myid_i] ),
            static_cast<int>( transpose2_dim.start_z[myid_j] ),
            static_cast<int>( transpose2_dim.start_w[myid_k] ),
            "yzwx->xzwy",
            mode,
            options
        )
    );

    comm_info.barrier();
    same_xw.transpose_xzwy_to_xywz( stage2_back, stage1_back, mode );
    comm_info.barrier();
    failed = std::max(
        failed,
        check_xywz_stage(
            log,
            comm_info,
            for_each,
            reduce,
            stage1_back,
            stage1_ref,
            static_cast<int>( transpose1_dim.start_x[myid_i] ),
            static_cast<int>( transpose1_dim.start_y[myid_j] ),
            static_cast<int>( transpose1_dim.start_w[myid_k] ),
            "xzwy->xywz",
            mode,
            options
        )
    );

    comm_info.barrier();
    same_xy.transpose_xywz_to_xyzw( stage1_back, stage0_back, mode );
    comm_info.barrier();
    failed = std::max(
        failed,
        check_xyzw_stage(
            log,
            comm_info,
            for_each,
            reduce,
            stage0_back,
            stage0_ref,
            static_cast<int>( half_input_dim.start_x[myid_i] ),
            static_cast<int>( half_input_dim.start_y[myid_j] ),
            static_cast<int>( half_input_dim.start_z[myid_k] ),
            "xywz->xyzw",
            mode,
            options
        )
    );

    return failed;
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
        if ( options.p1 * options.p2 * options.p3 != static_cast<std::size_t>( comm_info.num_procs ) )
            throw std::logic_error( "P1*P2*P3 must equal the number of MPI processes" );

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
        log.error( e.what() );
        return 1;
    }
}
