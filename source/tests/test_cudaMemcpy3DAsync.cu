#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <thrust/complex.h>

#include <scfd/backend/cuda.h>
#include <scfd/for_each/cuda_nd.h>
#include <scfd/for_each/cuda_nd_impl.cuh>
#include <scfd/static_vec/rect.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/cuda_timer_event.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/utils/safe_call.h>

#include "detail/cuda_memcpy_3d_transposer.h"
#include "detail/manual_transpose_3d.h"

template <class Idx, class ArrayLhs, class ArrayRhs, class ArrayRes>
struct diff_functor
{
    diff_functor( const ArrayLhs &_lhs, const ArrayRhs &_rhs, ArrayRes &_res ) : lhs( _lhs ), rhs( _rhs ), res( _res )
    {
    }

    ArrayLhs lhs;
    ArrayRhs rhs;
    ArrayRes res;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        res( idx ) = lhs( idx ) - rhs( idx );
    }
};

template <class T, class Array>
T abs_sum_norm( const Array &array )
{
    using view_t = typename Array::view_type;

    view_t view( array );

    T norm = 0;
#pragma omp parallel for reduction( + : norm )
    for ( std::size_t i = 0; i < static_cast<std::size_t>( array.total_size() ); ++i )
    {
        norm += thrust::abs( view.raw_ptr()[i] );
    }

    view.release( false );
    return norm;
}

template <class Idx, class Array>
scfd::static_vec::rect<int, 3> make_range_for_array( const Array &array )
{
    const auto sz = array.size_nd();
    return scfd::static_vec::rect<int, 3>(
        Idx( 0, 0, 0 ), Idx( static_cast<int>( sz[0] ), static_cast<int>( sz[1] ), static_cast<int>( sz[2] ) )
    );
}

template <class T, class Array>
void fill_unique_values_xyz( Array &array, std::size_t nx, std::size_t ny, std::size_t nz )
{
    using complex_t = typename Array::value_type;
    using view_t    = typename Array::view_type;

    view_t view( array );

    for ( std::size_t i = 0; i < nx; ++i )
    {
        for ( std::size_t j = 0; j < ny; ++j )
        {
            for ( std::size_t k = 0; k < nz; ++k )
            {
                const T value   = static_cast<T>( 1 + i + nx * ( j + ny * k ) );
                view( i, j, k ) = complex_t( value, 0 );
            }
        }
    }

    view.release( true );
}

template <fftm::tests::detail::permutation_3d Perm, class T, class Array>
void write_pos_if_requested(
    bool write_pos_files, const std::string &filename, const Array &array, std::size_t nx, std::size_t ny,
    std::size_t nz
)
{
    if ( !write_pos_files )
    {
        return;
    }

    const auto dims = fftm::tests::detail::dims_for_permutation<Perm>( nx, ny, nz );
    fftm::tests::detail::write_out_pos_file_scal_3D_point<T>( filename, array, dims[0], dims[1], dims[2] );
}

template <
    class T, class Idx, fftm::tests::detail::permutation_3d SrcPerm, fftm::tests::detail::permutation_3d DstPerm,
    class ForEach, class Transposer, class SrcArray, class DstArray, class DiffArray>
void execute_verified_step(
    const std::string &label, const ForEach &for_each, const Transposer &cuda_transposer,
    scfd::utils::cuda_timer_event &timer_begin, scfd::utils::cuda_timer_event &timer_end, const SrcArray &src_actual,
    const SrcArray &src_reference, DstArray &dst_actual, DstArray &dst_reference, DiffArray &diff_array
)
{
    using manual_t = fftm::tests::detail::manual_transpose_3d<SrcPerm, DstPerm>;

    manual_t manual_transposer( cuda_transposer.nx(), cuda_transposer.ny(), cuda_transposer.nz() );

    timer_begin.record();
    cuda_transposer.template transpose_async<SrcPerm, DstPerm>( src_actual, dst_actual );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    timer_end.record();

    manual_transposer.transpose( for_each, src_reference, dst_reference );

    const auto diff_range = make_range_for_array<Idx>( dst_actual );
    for_each( diff_functor<Idx, DstArray, DstArray, DiffArray>( dst_actual, dst_reference, diff_array ), diff_range );
    for_each.wait();

    const T reference_norm = abs_sum_norm<T>( dst_reference );
    const T diff_norm      = abs_sum_norm<T>( diff_array );
    const T tolerance = 10 * std::numeric_limits<T>::epsilon() * ( reference_norm > T( 1 ) ? reference_norm : T( 1 ) );

    std::cout << label << ": " << fftm::tests::detail::permutation_3d_traits<SrcPerm>::label() << " -> "
              << fftm::tests::detail::permutation_3d_traits<DstPerm>::label() << ", diff=" << diff_norm
              << ", wall_ms=" << timer_end.elapsed_time( timer_begin ) << std::endl;

    if ( diff_norm > tolerance )
    {
        throw std::runtime_error( label + " failed: diff norm = " + std::to_string( diff_norm ) );
    }
}

template <class T, class Idx, class ForEach, class Array, class DiffArray>
void verify_arrays_equal(
    const std::string &label, const ForEach &for_each, const Array &lhs, const Array &rhs, DiffArray &diff_array
)
{
    const auto diff_range = make_range_for_array<Idx>( lhs );
    for_each( diff_functor<Idx, Array, Array, DiffArray>( lhs, rhs, diff_array ), diff_range );
    for_each.wait();

    const T reference_norm = abs_sum_norm<T>( rhs );
    const T diff_norm      = abs_sum_norm<T>( diff_array );
    const T tolerance = 10 * std::numeric_limits<T>::epsilon() * ( reference_norm > T( 1 ) ? reference_norm : T( 1 ) );

    std::cout << label << ": diff=" << diff_norm << std::endl;

    if ( diff_norm > tolerance )
    {
        throw std::runtime_error( label + " failed: diff norm = " + std::to_string( diff_norm ) );
    }
}

int main( int argc, char const *argv[] )
{
    static const int dim = 3;

    using T          = double;
    using complex_t  = thrust::complex<T>;
    using idx_t      = scfd::static_vec::vec<int, dim>;
    using backend_t  = scfd::backend::cuda;
    using memory_t   = backend_t::memory_type;
    using for_each_t = backend_t::for_each_nd_type<dim>;
    using timer_t    = scfd::utils::cuda_timer_event;
    using perm_t     = fftm::tests::detail::permutation_3d;

    using xyz_array_t = fftm::tests::detail::permuted_tensor_3d_t<complex_t, memory_t, perm_t::xyz>;
    using xzy_array_t = fftm::tests::detail::permuted_tensor_3d_t<complex_t, memory_t, perm_t::xzy>;
    using zxy_array_t = fftm::tests::detail::permuted_tensor_3d_t<complex_t, memory_t, perm_t::zxy>;
    using zyx_array_t = fftm::tests::detail::permuted_tensor_3d_t<complex_t, memory_t, perm_t::zyx>;
    using yzx_array_t = fftm::tests::detail::permuted_tensor_3d_t<complex_t, memory_t, perm_t::yzx>;
    using yxz_array_t = fftm::tests::detail::permuted_tensor_3d_t<complex_t, memory_t, perm_t::yxz>;

    std::size_t nx              = 20;
    std::size_t ny              = 30;
    std::size_t nz              = 50;
    bool        write_pos_files = false;

    int argi = 1;
    if ( ( argi < argc ) && ( std::string( argv[argi] ) == "--write-pos" ) )
    {
        write_pos_files = true;
        ++argi;
    }

    if ( argc - argi == 3 )
    {
        nx = static_cast<std::size_t>( std::strtoull( argv[argi], NULL, 10 ) );
        ny = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
        nz = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
    }
    else if ( argc != argi )
    {
        throw std::logic_error( "USAGE: test_cudaMemcpy3DAsync.bin [--write-pos] [Nx Ny Nz]" );
    }

    scfd::utils::init_cuda_persistent();

    for_each_t for_each;
    timer_t    timer_begin;
    timer_t    timer_end;

    fftm::tests::detail::cuda_memcpy_3d_transposer<complex_t> cuda_transposer( nx, ny, nz );

    xyz_array_t xyz_initial;
    xyz_array_t xyz_roundtrip_actual;
    xyz_array_t xyz_roundtrip_reference;
    xyz_array_t xyz_diff;

    xzy_array_t xzy_actual;
    xzy_array_t xzy_reference;
    xzy_array_t xzy_diff;

    zxy_array_t zxy_actual;
    zxy_array_t zxy_reference;
    zxy_array_t zxy_diff;

    zyx_array_t zyx_actual;
    zyx_array_t zyx_reference;
    zyx_array_t zyx_diff;

    yzx_array_t yzx_actual;
    yzx_array_t yzx_reference;
    yzx_array_t yzx_diff;

    yxz_array_t yxz_actual;
    yxz_array_t yxz_reference;
    yxz_array_t yxz_diff;

    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::xyz>( xyz_initial, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::xyz>( xyz_roundtrip_actual, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::xyz>( xyz_roundtrip_reference, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::xyz>( xyz_diff, nx, ny, nz ) );

    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::xzy>( xzy_actual, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::xzy>( xzy_reference, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::xzy>( xzy_diff, nx, ny, nz ) );

    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::zxy>( zxy_actual, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::zxy>( zxy_reference, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::zxy>( zxy_diff, nx, ny, nz ) );

    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::zyx>( zyx_actual, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::zyx>( zyx_reference, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::zyx>( zyx_diff, nx, ny, nz ) );

    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::yzx>( yzx_actual, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::yzx>( yzx_reference, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::yzx>( yzx_diff, nx, ny, nz ) );

    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::yxz>( yxz_actual, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::yxz>( yxz_reference, nx, ny, nz ) );
    SCFD_SAFE_CALL( fftm::tests::detail::init_permuted_array<perm_t::yxz>( yxz_diff, nx, ny, nz ) );

    fill_unique_values_xyz<T>( xyz_initial, nx, ny, nz );

    execute_verified_step<T, idx_t, perm_t::xyz, perm_t::xzy>(
        "Step 1", for_each, cuda_transposer, timer_begin, timer_end, xyz_initial, xyz_initial, xzy_actual,
        xzy_reference, xzy_diff
    );

    execute_verified_step<T, idx_t, perm_t::xzy, perm_t::zxy>(
        "Step 2", for_each, cuda_transposer, timer_begin, timer_end, xzy_actual, xzy_reference, zxy_actual,
        zxy_reference, zxy_diff
    );

    execute_verified_step<T, idx_t, perm_t::zxy, perm_t::zyx>(
        "Step 3", for_each, cuda_transposer, timer_begin, timer_end, zxy_actual, zxy_reference, zyx_actual,
        zyx_reference, zyx_diff
    );

    execute_verified_step<T, idx_t, perm_t::zyx, perm_t::yzx>(
        "Step 4", for_each, cuda_transposer, timer_begin, timer_end, zyx_actual, zyx_reference, yzx_actual,
        yzx_reference, yzx_diff
    );

    execute_verified_step<T, idx_t, perm_t::yzx, perm_t::yxz>(
        "Step 5", for_each, cuda_transposer, timer_begin, timer_end, yzx_actual, yzx_reference, yxz_actual,
        yxz_reference, yxz_diff
    );

    execute_verified_step<T, idx_t, perm_t::yxz, perm_t::xyz>(
        "Step 6", for_each, cuda_transposer, timer_begin, timer_end, yxz_actual, yxz_reference, xyz_roundtrip_actual,
        xyz_roundtrip_reference, xyz_diff
    );

    verify_arrays_equal<T, idx_t>( "Round-trip", for_each, xyz_roundtrip_actual, xyz_initial, xyz_diff );

    write_pos_if_requested<perm_t::xyz, T>( write_pos_files, "step0_xyz_actual.pos", xyz_initial, nx, ny, nz );
    write_pos_if_requested<perm_t::xzy, T>( write_pos_files, "step1_xzy_actual.pos", xzy_actual, nx, ny, nz );
    write_pos_if_requested<perm_t::xzy, T>( write_pos_files, "step1_xzy_manual.pos", xzy_reference, nx, ny, nz );
    write_pos_if_requested<perm_t::zxy, T>( write_pos_files, "step2_zxy_actual.pos", zxy_actual, nx, ny, nz );
    write_pos_if_requested<perm_t::zxy, T>( write_pos_files, "step2_zxy_manual.pos", zxy_reference, nx, ny, nz );
    write_pos_if_requested<perm_t::zyx, T>( write_pos_files, "step3_zyx_actual.pos", zyx_actual, nx, ny, nz );
    write_pos_if_requested<perm_t::zyx, T>( write_pos_files, "step3_zyx_manual.pos", zyx_reference, nx, ny, nz );
    write_pos_if_requested<perm_t::yzx, T>( write_pos_files, "step4_yzx_actual.pos", yzx_actual, nx, ny, nz );
    write_pos_if_requested<perm_t::yzx, T>( write_pos_files, "step4_yzx_manual.pos", yzx_reference, nx, ny, nz );
    write_pos_if_requested<perm_t::yxz, T>( write_pos_files, "step5_yxz_actual.pos", yxz_actual, nx, ny, nz );
    write_pos_if_requested<perm_t::yxz, T>( write_pos_files, "step5_yxz_manual.pos", yxz_reference, nx, ny, nz );
    write_pos_if_requested<perm_t::xyz, T>( write_pos_files, "step6_xyz_actual.pos", xyz_roundtrip_actual, nx, ny, nz );
    write_pos_if_requested<perm_t::xyz, T>(
        write_pos_files, "step6_xyz_manual.pos", xyz_roundtrip_reference, nx, ny, nz
    );

    return EXIT_SUCCESS;
}
