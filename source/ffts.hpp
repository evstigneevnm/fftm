#ifndef __FFTM_FFTS_HPP__
#define __FFTM_FFTS_HPP__

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/utils/cuda_safe_call.h>

#include "detail/array_arrangers.h"
#include "detail/cuda_memcpy_4d_slab_transposer.h"
#include "detail/direct_transpose_4d.h"
#include "fft_direction.h"

namespace fftm
{

enum class transpose_backend
{
    direct,
    memcpy
};

inline const char *transpose_backend_name( transpose_backend backend )
{
    switch ( backend )
    {
        case transpose_backend::direct:
            return "direct";
        case transpose_backend::memcpy:
            return "memcpy";
    }

    return "unknown";
}

struct ffts_init_options
{
    transpose_backend backend = transpose_backend::direct;
};

namespace detail
{

template <class Real, class Complex, class Memory, std::size_t Dim>
struct ffts_array_traits;

template <class Real, class Complex, class Memory>
struct ffts_array_traits<Real, Complex, Memory, 2>
{
    using real_array_t    = scfd::arrays::tensor_array_nd<Real, 2, Memory, scfd::arrays::custom_arranger_10_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 2, Memory, scfd::arrays::custom_arranger_10_t>;
};

template <class Real, class Complex, class Memory>
struct ffts_array_traits<Real, Complex, Memory, 3>
{
    using real_array_t    = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_210_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_210_t>;
};

template <class Real, class Complex, class Memory>
struct ffts_array_traits<Real, Complex, Memory, 4>
{
    using real_array_t         = scfd::arrays::tensor_array_nd<Real, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using xyzw_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using xywz_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using xzwy_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0213_t>;
    using yzwx_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using complex_array_t      = yzwx_complex_array_t;
};

} // namespace detail

template <class BaseFFT, class Backend>
class ffts
{
public:
    using real     = typename BaseFFT::real;
    using complex  = typename BaseFFT::complex;
    using memory_t = typename Backend::memory_type;

    template <std::size_t Dim>
    using real_array_t = typename detail::ffts_array_traits<real, complex, memory_t, Dim>::real_array_t;

    template <std::size_t Dim>
    using complex_array_t = typename detail::ffts_array_traits<real, complex, memory_t, Dim>::complex_array_t;

    ffts()
        : init_done_( false )
        , dim_( 0 )
        , transpose_backend_( transpose_backend::direct )
        , nx_( 0 )
        , ny_( 0 )
        , nz_( 0 )
        , nw_( 0 )
        , ny_half_( 0 )
        , nz_half_( 0 )
        , nw_half_( 0 )
    {
        for_each_4d_.block_size = 128;
    }

    template <std::size_t Dim>
    void init( const std::array<std::size_t, Dim> &grid, const ffts_init_options &options = ffts_init_options() )
    {
        init_array_( grid, options, typename std::integral_constant<std::size_t, Dim>::type() );
    }

    void init( std::size_t nx, std::size_t ny )
    {
        ensure_can_init_();
        nx_      = nx;
        ny_      = ny;
        ny_half_ = ny_ / 2 + 1;
        dim_     = 2;

        add_2d_plans_();
        base_fft_.activate();
        init_done_ = true;
    }

    void init( std::size_t nx, std::size_t ny, std::size_t nz )
    {
        ensure_can_init_();
        nx_      = nx;
        ny_      = ny;
        nz_      = nz;
        nz_half_ = nz_ / 2 + 1;
        dim_     = 3;

        add_3d_plans_();
        base_fft_.activate();
        init_done_ = true;
    }

    void init( std::size_t nx, std::size_t ny, std::size_t nz, std::size_t nw, const ffts_init_options &options = ffts_init_options() )
    {
        ensure_can_init_();
        nx_                = nx;
        ny_                = ny;
        nz_                = nz;
        nw_                = nw;
        nw_half_           = nw_ / 2 + 1;
        dim_               = 4;
        transpose_backend_ = options.backend;

        xyzw_stage_.init( nx_, ny_, nz_, nw_half_ );
        xywz_stage_.init( nx_, ny_, nw_half_, nz_ );
        xzwy_stage_.init( nx_, nz_, nw_half_, ny_ );
        work_hat_ .init( ny_, nz_, nw_half_, nx_ );

        direct_transposer_.reset( new detail::direct_transpose_4d( nx_, ny_, nz_, nw_half_ ) );
        memcpy_transposer_.reset( new detail::cuda_memcpy_4d_slab_transposer<complex>( nx_, ny_, nz_, nw_half_ ) );

        add_4d_plans_();
        base_fft_.activate();
        init_done_ = true;
    }

    bool is_initialized() const
    {
        return init_done_;
    }

    std::size_t dimension() const
    {
        ensure_initialized_();
        return dim_;
    }

    transpose_backend get_transpose_backend() const
    {
        ensure_initialized_();
        if ( dim_ != 4 )
        {
            throw std::logic_error( "ffts::get_transpose_backend is only available after 4D initialization." );
        }
        return transpose_backend_;
    }

    void forward( const real_array_t<2> &in, complex_array_t<2> &out )
    {
        ensure_dimension_( 2 );
        base_fft_.template exec<real_array_t<2>, complex_array_t<2>>( "forward_2d", in, out );
    }

    void backward( const complex_array_t<2> &in, real_array_t<2> &out )
    {
        ensure_dimension_( 2 );
        base_fft_.template exec<complex_array_t<2>, real_array_t<2>>( "inverse_2d", in, out );
    }

    void forward( const real_array_t<3> &in, complex_array_t<3> &out )
    {
        ensure_dimension_( 3 );
        base_fft_.template exec<real_array_t<3>, complex_array_t<3>>( "forward_3d", in, out );
    }

    void backward( const complex_array_t<3> &in, real_array_t<3> &out )
    {
        ensure_dimension_( 3 );
        base_fft_.template exec<complex_array_t<3>, real_array_t<3>>( "inverse_3d", in, out );
    }

    void forward( const real_array_t<4> &in, complex_array_t<4> &out )
    {
        ensure_dimension_( 4 );

        base_fft_.template exec<real_array_t<4>, xyzw_complex_array_t>( "forward_w", in, xyzw_stage_ );
        transpose_xyzw_to_xywz_( xyzw_stage_, xywz_stage_ );

        base_fft_.template exec<xywz_complex_array_t, xywz_complex_array_t>( "forward_z", xywz_stage_, xywz_stage_ );
        transpose_xywz_to_xzwy_( xywz_stage_, xzwy_stage_ );

        base_fft_.template exec<xzwy_complex_array_t, xzwy_complex_array_t>( "forward_y", xzwy_stage_, xzwy_stage_ );
        transpose_xzwy_to_yzwx_( xzwy_stage_, out );

        base_fft_.template exec<complex_array_t<4>, complex_array_t<4>>( "forward_x", out, out );
    }

    void backward( const complex_array_t<4> &in, real_array_t<4> &out )
    {
        ensure_dimension_( 4 );

        copy_spectral_field_( in, work_hat_ );
        base_fft_.template exec<complex_array_t<4>, complex_array_t<4>>( "inverse_x", work_hat_, work_hat_ );
        transpose_yzwx_to_xzwy_( work_hat_, xzwy_stage_ );

        base_fft_.template exec<xzwy_complex_array_t, xzwy_complex_array_t>( "inverse_y", xzwy_stage_, xzwy_stage_ );
        transpose_xzwy_to_xywz_( xzwy_stage_, xywz_stage_ );

        base_fft_.template exec<xywz_complex_array_t, xywz_complex_array_t>( "inverse_z", xywz_stage_, xywz_stage_ );
        transpose_xywz_to_xyzw_( xywz_stage_, xyzw_stage_ );

        base_fft_.template exec<xyzw_complex_array_t, real_array_t<4>>( "inverse_w", xyzw_stage_, out );
    }

private:
    using backend_t     = Backend;
    using for_each_4d_t = typename backend_t::template for_each_nd_type<4, int>;

    using traits_4d_t           = detail::ffts_array_traits<real, complex, memory_t, 4>;
    using xyzw_complex_array_t  = typename traits_4d_t::xyzw_complex_array_t;
    using xywz_complex_array_t  = typename traits_4d_t::xywz_complex_array_t;
    using xzwy_complex_array_t  = typename traits_4d_t::xzwy_complex_array_t;
    using yzwx_complex_array_t  = typename traits_4d_t::yzwx_complex_array_t;
    using direct_transposer_t   = detail::direct_transpose_4d;
    using memcpy_transposer_t   = detail::cuda_memcpy_4d_slab_transposer<complex>;

    void ensure_can_init_() const
    {
        if ( init_done_ )
        {
            throw std::logic_error( "ffts::init can only be called once per instance." );
        }
    }

    void ensure_initialized_() const
    {
        if ( !init_done_ )
        {
            throw std::logic_error( "ffts: call init() before using the transform." );
        }
    }

    void ensure_dimension_( std::size_t expected_dim ) const
    {
        ensure_initialized_();
        if ( dim_ != expected_dim )
        {
            throw std::logic_error(
                "ffts: initialized for " + std::to_string( dim_ ) + "D but used as " + std::to_string( expected_dim ) + "D."
            );
        }
    }

    void add_2d_plans_()
    {
        base_fft_.template add_plan_2D<fftm::direction::R2C>(
            "forward_2d",
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            1,
            static_cast<long long int>( nx_ * ny_ ),
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_half_ ),
            1,
            static_cast<long long int>( nx_ * ny_half_ ),
            1
        );

        base_fft_.template add_plan_2D<fftm::direction::C2R>(
            "inverse_2d",
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_half_ ),
            1,
            static_cast<long long int>( nx_ * ny_half_ ),
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            1,
            static_cast<long long int>( nx_ * ny_ ),
            1
        );
    }

    void add_3d_plans_()
    {
        base_fft_.template add_plan_3D<fftm::direction::R2C>(
            "forward_3d",
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_ ),
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_ ),
            1,
            static_cast<long long int>( nx_ * ny_ * nz_ ),
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_half_ ),
            1,
            static_cast<long long int>( nx_ * ny_ * nz_half_ ),
            1
        );

        base_fft_.template add_plan_3D<fftm::direction::C2R>(
            "inverse_3d",
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_ ),
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_half_ ),
            1,
            static_cast<long long int>( nx_ * ny_ * nz_half_ ),
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_ ),
            1,
            static_cast<long long int>( nx_ * ny_ * nz_ ),
            1
        );
    }

    void add_4d_plans_()
    {
        const long long int real_batch = static_cast<long long int>( nx_ * ny_ * nz_ );
        const long long int z_batch    = static_cast<long long int>( nx_ * ny_ * nw_half_ );
        const long long int y_batch    = static_cast<long long int>( nx_ * nz_ * nw_half_ );
        const long long int x_batch    = static_cast<long long int>( ny_ * nz_ * nw_half_ );

        const long long int w_stride = static_cast<long long int>( nx_ * ny_ * nz_ );
        const long long int z_stride = static_cast<long long int>( nx_ * ny_ * nw_half_ );
        const long long int y_stride = static_cast<long long int>( nx_ * nz_ * nw_half_ );

        base_fft_.template add_plan_1D<fftm::direction::R2C>(
            "forward_w",
            static_cast<long long int>( nw_ ),
            1,
            w_stride,
            1,
            1,
            w_stride,
            1,
            real_batch
        );

        base_fft_.template add_plan_1D<fftm::direction::C2R>(
            "inverse_w",
            static_cast<long long int>( nw_ ),
            1,
            w_stride,
            1,
            1,
            w_stride,
            1,
            real_batch
        );

        base_fft_.template add_plan_1D<fftm::direction::C2CF>(
            "forward_z",
            static_cast<long long int>( nz_ ),
            1,
            z_stride,
            1,
            1,
            z_stride,
            1,
            z_batch
        );

        base_fft_.template add_plan_1D<fftm::direction::C2CB>(
            "inverse_z",
            static_cast<long long int>( nz_ ),
            1,
            z_stride,
            1,
            1,
            z_stride,
            1,
            z_batch
        );

        base_fft_.template add_plan_1D<fftm::direction::C2CF>(
            "forward_y",
            static_cast<long long int>( ny_ ),
            1,
            y_stride,
            1,
            1,
            y_stride,
            1,
            y_batch
        );

        base_fft_.template add_plan_1D<fftm::direction::C2CB>(
            "inverse_y",
            static_cast<long long int>( ny_ ),
            1,
            y_stride,
            1,
            1,
            y_stride,
            1,
            y_batch
        );

        base_fft_.template add_plan_1D<fftm::direction::C2CF>(
            "forward_x",
            static_cast<long long int>( nx_ ),
            1,
            1,
            static_cast<long long int>( nx_ ),
            1,
            1,
            static_cast<long long int>( nx_ ),
            x_batch
        );

        base_fft_.template add_plan_1D<fftm::direction::C2CB>(
            "inverse_x",
            static_cast<long long int>( nx_ ),
            1,
            1,
            static_cast<long long int>( nx_ ),
            1,
            1,
            static_cast<long long int>( nx_ ),
            x_batch
        );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_xywz_( const SrcArray &src, DstArray &dst )
    {
        if ( transpose_backend_ == transpose_backend::direct )
        {
            direct_transposer_->xyzw_to_xywz( for_each_4d_, src, dst );
        }
        else
        {
            memcpy_transposer_->xyzw_to_xywz( for_each_4d_, src, dst );
        }
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xzwy_( const SrcArray &src, DstArray &dst )
    {
        if ( transpose_backend_ == transpose_backend::direct )
        {
            direct_transposer_->xywz_to_xzwy( for_each_4d_, src, dst );
        }
        else
        {
            memcpy_transposer_->xywz_to_xzwy( for_each_4d_, src, dst );
        }
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_yzwx_( const SrcArray &src, DstArray &dst )
    {
        if ( transpose_backend_ == transpose_backend::direct )
        {
            direct_transposer_->xzwy_to_yzwx( for_each_4d_, src, dst );
        }
        else
        {
            memcpy_transposer_->xzwy_to_yzwx( for_each_4d_, src, dst );
        }
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_xzwy_( const SrcArray &src, DstArray &dst )
    {
        if ( transpose_backend_ == transpose_backend::direct )
        {
            direct_transposer_->yzwx_to_xzwy( for_each_4d_, src, dst );
        }
        else
        {
            memcpy_transposer_->yzwx_to_xzwy( for_each_4d_, src, dst );
        }
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_xywz_( const SrcArray &src, DstArray &dst )
    {
        if ( transpose_backend_ == transpose_backend::direct )
        {
            direct_transposer_->xzwy_to_xywz( for_each_4d_, src, dst );
        }
        else
        {
            memcpy_transposer_->xzwy_to_xywz( for_each_4d_, src, dst );
        }
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xyzw_( const SrcArray &src, DstArray &dst )
    {
        if ( transpose_backend_ == transpose_backend::direct )
        {
            direct_transposer_->xywz_to_xyzw( for_each_4d_, src, dst );
        }
        else
        {
            memcpy_transposer_->xywz_to_xyzw( for_each_4d_, src, dst );
        }
    }

    void copy_spectral_field_( const complex_array_t<4> &src, complex_array_t<4> &dst )
    {
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dst.raw_ptr(),
                src.raw_ptr(),
                sizeof( complex ) * static_cast<std::size_t>( src.total_size() ),
                cudaMemcpyDeviceToDevice
            )
        );
    }

    void init_array_(
        const std::array<std::size_t, 2> &grid,
        const ffts_init_options          &,
        std::integral_constant<std::size_t, 2>
    )
    {
        init( grid[0], grid[1] );
    }

    void init_array_(
        const std::array<std::size_t, 3> &grid,
        const ffts_init_options          &,
        std::integral_constant<std::size_t, 3>
    )
    {
        init( grid[0], grid[1], grid[2] );
    }

    void init_array_(
        const std::array<std::size_t, 4> &grid,
        const ffts_init_options          &options,
        std::integral_constant<std::size_t, 4>
    )
    {
        init( grid[0], grid[1], grid[2], grid[3], options );
    }

    BaseFFT base_fft_;

    bool             init_done_;
    std::size_t      dim_;
    transpose_backend transpose_backend_;

    std::size_t nx_;
    std::size_t ny_;
    std::size_t nz_;
    std::size_t nw_;
    std::size_t ny_half_;
    std::size_t nz_half_;
    std::size_t nw_half_;

    for_each_4d_t for_each_4d_;

    std::unique_ptr<direct_transposer_t> direct_transposer_;
    std::unique_ptr<memcpy_transposer_t> memcpy_transposer_;

    xyzw_complex_array_t xyzw_stage_;
    xywz_complex_array_t xywz_stage_;
    xzwy_complex_array_t xzwy_stage_;
    yzwx_complex_array_t work_hat_;
};

} // namespace fftm

#endif // __FFTM_FFTS_HPP__
