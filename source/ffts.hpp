#ifndef __FFTM_FFTS_HPP__
#define __FFTM_FFTS_HPP__

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <scfd/arrays/tensor_array_nd.h>

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

enum class transform_strategy_4d
{
    pencil_pencil,
    slab_slab
};

template <transpose_backend Backend = transpose_backend::direct>
struct strategy_4d_pencil_pencil
{
};

template <transpose_backend Backend = transpose_backend::direct>
struct strategy_4d_slab_slab
{
};

struct ffts_init_options
{
};

namespace detail
{

template <class Strategy4D>
struct ffts_4d_strategy_traits;

template <transpose_backend Backend>
struct ffts_4d_strategy_traits<strategy_4d_pencil_pencil<Backend>>
{
    static constexpr transform_strategy_4d family  = transform_strategy_4d::pencil_pencil;
    static constexpr transpose_backend     backend = Backend;

    static const char *name()
    {
        return backend == transpose_backend::direct ? "pencil-pencil-direct" : "pencil-pencil-memcpy";
    }
};

template <transpose_backend Backend>
struct ffts_4d_strategy_traits<strategy_4d_slab_slab<Backend>>
{
    static constexpr transform_strategy_4d family  = transform_strategy_4d::slab_slab;
    static constexpr transpose_backend     backend = Backend;

    static const char *name()
    {
        return backend == transpose_backend::direct ? "slab-slab-direct" : "slab-slab-memcpy";
    }
};

template <
    class       Real,
    class       Complex,
    class       Memory,
    std::size_t Dim,
    class       Strategy4D = strategy_4d_pencil_pencil<transpose_backend::direct>
>
struct ffts_array_traits;

template <class Real, class Complex, class Memory, class Strategy4D>
struct ffts_array_traits<Real, Complex, Memory, 2, Strategy4D>
{
    using real_array_t    = scfd::arrays::tensor_array_nd<Real, 2, Memory, scfd::arrays::custom_arranger_10_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 2, Memory, scfd::arrays::custom_arranger_10_t>;
};

template <class Real, class Complex, class Memory, class Strategy4D>
struct ffts_array_traits<Real, Complex, Memory, 3, Strategy4D>
{
    using real_array_t    = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_210_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_210_t>;
};

template <class Real, class Complex, class Memory, transpose_backend Backend>
struct ffts_array_traits<Real, Complex, Memory, 4, strategy_4d_pencil_pencil<Backend>>
{
    using real_array_t         = scfd::arrays::tensor_array_nd<Real, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using xyzw_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using xywz_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using xzwy_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0213_t>;
    using yzwx_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using stage0_complex_array_t = xyzw_complex_array_t;
    using stage1_complex_array_t = xywz_complex_array_t;
    using stage2_complex_array_t = xzwy_complex_array_t;
    using complex_array_t      = yzwx_complex_array_t;
};

template <class Real, class Complex, class Memory, transpose_backend Backend>
struct ffts_array_traits<Real, Complex, Memory, 4, strategy_4d_slab_slab<Backend>>
{
    using real_array_t         = scfd::arrays::tensor_array_nd<Real, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using xyzw_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using zwxy_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_1032_t>;
    using yzwx_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_2103_t>;
    using stage0_complex_array_t = xyzw_complex_array_t;
    using stage1_complex_array_t = zwxy_complex_array_t;
    using stage2_complex_array_t = zwxy_complex_array_t;
    using complex_array_t      = yzwx_complex_array_t;
};

} // namespace detail

template <
    class BaseFFT,
    class Backend,
    class Strategy4D = strategy_4d_pencil_pencil<transpose_backend::direct>
>
class ffts
{
public:
    using real     = typename BaseFFT::real;
    using complex  = typename BaseFFT::complex;
    using memory_t = typename Backend::memory_type;
    using runtime_api = typename BaseFFT::runtime_api;
    using strategy_4d_t = Strategy4D;

    template <std::size_t Dim>
    using real_array_t = typename detail::ffts_array_traits<real, complex, memory_t, Dim, Strategy4D>::real_array_t;

    template <std::size_t Dim>
    using complex_array_t = typename detail::ffts_array_traits<real, complex, memory_t, Dim, Strategy4D>::complex_array_t;

    static constexpr transform_strategy_4d strategy_family_4d = detail::ffts_4d_strategy_traits<Strategy4D>::family;
    static constexpr transpose_backend transpose_backend_4d   = detail::ffts_4d_strategy_traits<Strategy4D>::backend;

    static const char *strategy_name()
    {
        return detail::ffts_4d_strategy_traits<Strategy4D>::name();
    }

    ffts()
        : init_done_( false )
        , dim_( 0 )
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
        (void)options;
        ensure_can_init_();
        nx_      = nx;
        ny_      = ny;
        nz_      = nz;
        nw_      = nw;
        nw_half_ = nw_ / 2 + 1;
        dim_     = 4;

        stage0_.init( nx_, ny_, nz_, nw_half_ );
        work_hat_.init( ny_, nz_, nw_half_, nx_ );

        init_4d_storage_( strategy_family_tag() );
        add_4d_plans_( strategy_family_tag() );
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
        forward_4d_( strategy_family_tag(), in, out );
    }

    void backward( const complex_array_t<4> &in, real_array_t<4> &out )
    {
        ensure_dimension_( 4 );

        copy_spectral_field_( in, work_hat_ );
        backward_4d_( strategy_family_tag(), in, out );
    }

private:
    using backend_t     = Backend;
    using runtime_api_t = typename BaseFFT::runtime_api;
    using for_each_4d_t = typename backend_t::template for_each_nd_type<4, int>;
    using strategy_family_tag   = std::integral_constant<transform_strategy_4d, strategy_family_4d>;
    using transpose_backend_tag = std::integral_constant<transpose_backend, transpose_backend_4d>;

    using traits_4d_t            = detail::ffts_array_traits<real, complex, memory_t, 4, Strategy4D>;
    using stage0_complex_array_t = typename traits_4d_t::stage0_complex_array_t;
    using stage1_complex_array_t = typename traits_4d_t::stage1_complex_array_t;
    using stage2_complex_array_t = typename traits_4d_t::stage2_complex_array_t;
    using direct_transposer_t    = detail::direct_transpose_4d;
    using memcpy_transposer_t    = detail::cuda_memcpy_4d_slab_transposer<complex, runtime_api_t>;

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
        base_fft_.template add_plan_2D<::fftm::direction::R2C>(
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

        base_fft_.template add_plan_2D<::fftm::direction::C2R>(
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
        base_fft_.template add_plan_3D<::fftm::direction::R2C>(
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

        base_fft_.template add_plan_3D<::fftm::direction::C2R>(
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

    void add_4d_plans_( std::integral_constant<transform_strategy_4d, transform_strategy_4d::pencil_pencil> )
    {
        const long long int real_batch = static_cast<long long int>( nx_ * ny_ * nz_ );
        const long long int z_batch    = static_cast<long long int>( nx_ * ny_ * nw_half_ );
        const long long int y_batch    = static_cast<long long int>( nx_ * nz_ * nw_half_ );
        const long long int x_batch    = static_cast<long long int>( ny_ * nz_ * nw_half_ );

        const long long int w_stride = static_cast<long long int>( nx_ * ny_ * nz_ );
        const long long int z_stride = static_cast<long long int>( nx_ * ny_ * nw_half_ );
        const long long int y_stride = static_cast<long long int>( nx_ * nz_ * nw_half_ );

        base_fft_.template add_plan_1D<::fftm::direction::R2C>(
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

        base_fft_.template add_plan_1D<::fftm::direction::C2R>(
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

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
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

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
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

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
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

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
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

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
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

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
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

    void add_4d_plans_( std::integral_constant<transform_strategy_4d, transform_strategy_4d::slab_slab> )
    {
        const long long int zw_batch = static_cast<long long int>( nx_ * ny_ );
        const long long int xy_batch = static_cast<long long int>( nz_ * nw_half_ );
        const long long int xy_stride = static_cast<long long int>( nz_ * nw_half_ );

        base_fft_.template add_plan_2D<::fftm::direction::R2C>(
            "forward_zw",
            static_cast<long long int>( nz_ ),
            static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ),
            static_cast<long long int>( nw_ ),
            1,
            static_cast<long long int>( nz_ * nw_ ),
            static_cast<long long int>( nz_ ),
            static_cast<long long int>( nw_half_ ),
            1,
            static_cast<long long int>( nz_ * nw_half_ ),
            zw_batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2R>(
            "inverse_zw",
            static_cast<long long int>( nz_ ),
            static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ),
            static_cast<long long int>( nw_half_ ),
            1,
            static_cast<long long int>( nz_ * nw_half_ ),
            static_cast<long long int>( nz_ ),
            static_cast<long long int>( nw_ ),
            1,
            static_cast<long long int>( nz_ * nw_ ),
            zw_batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2CF>(
            "forward_xy",
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            xy_stride,
            1,
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            xy_stride,
            1,
            xy_batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2CB>(
            "inverse_xy",
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            xy_stride,
            1,
            static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ),
            xy_stride,
            1,
            xy_batch
        );
    }

    void init_4d_storage_( std::integral_constant<transform_strategy_4d, transform_strategy_4d::pencil_pencil> )
    {
        stage1_.init( nx_, ny_, nw_half_, nz_ );
        stage2_.init( nx_, nz_, nw_half_, ny_ );
        direct_transposer_.reset( new detail::direct_transpose_4d( nx_, ny_, nz_, nw_half_ ) );
        init_memcpy_transposer_( transpose_backend_tag() );
    }

    void init_4d_storage_( std::integral_constant<transform_strategy_4d, transform_strategy_4d::slab_slab> )
    {
        stage1_.init( nz_, nw_half_, nx_, ny_ );
        direct_transposer_.reset( new detail::direct_transpose_4d( nx_, ny_, nz_, nw_half_ ) );
        init_memcpy_transposer_( transpose_backend_tag() );
    }

    void init_memcpy_transposer_( std::integral_constant<transpose_backend, transpose_backend::direct> )
    {
    }

    void init_memcpy_transposer_( std::integral_constant<transpose_backend, transpose_backend::memcpy> )
    {
        memcpy_transposer_.reset( new detail::cuda_memcpy_4d_slab_transposer<complex, runtime_api_t>( nx_, ny_, nz_, nw_half_ ) );
    }

    void forward_4d_(
        std::integral_constant<transform_strategy_4d, transform_strategy_4d::pencil_pencil>,
        const real_array_t<4> &in,
        complex_array_t<4>    &out
    )
    {
        base_fft_.template exec<real_array_t<4>, stage0_complex_array_t>( "forward_w", in, stage0_ );
        transpose_xyzw_to_xywz_( stage0_, stage1_ );

        base_fft_.template exec<stage1_complex_array_t, stage1_complex_array_t>( "forward_z", stage1_, stage1_ );
        transpose_xywz_to_xzwy_( stage1_, stage2_ );

        base_fft_.template exec<stage2_complex_array_t, stage2_complex_array_t>( "forward_y", stage2_, stage2_ );
        transpose_xzwy_to_yzwx_( stage2_, out );

        base_fft_.template exec<complex_array_t<4>, complex_array_t<4>>( "forward_x", out, out );
    }

    void forward_4d_(
        std::integral_constant<transform_strategy_4d, transform_strategy_4d::slab_slab>,
        const real_array_t<4> &in,
        complex_array_t<4>    &out
    )
    {
        base_fft_.template exec<real_array_t<4>, stage0_complex_array_t>( "forward_zw", in, stage0_ );
        transpose_xyzw_to_zwxy_( stage0_, stage1_ );
        base_fft_.template exec<stage1_complex_array_t, stage1_complex_array_t>( "forward_xy", stage1_, stage1_ );
        transpose_zwxy_to_yzwx_( stage1_, out );
    }

    void backward_4d_(
        std::integral_constant<transform_strategy_4d, transform_strategy_4d::pencil_pencil>,
        const complex_array_t<4> &,
        real_array_t<4>          &out
    )
    {
        base_fft_.template exec<complex_array_t<4>, complex_array_t<4>>( "inverse_x", work_hat_, work_hat_ );
        transpose_yzwx_to_xzwy_( work_hat_, stage2_ );

        base_fft_.template exec<stage2_complex_array_t, stage2_complex_array_t>( "inverse_y", stage2_, stage2_ );
        transpose_xzwy_to_xywz_( stage2_, stage1_ );

        base_fft_.template exec<stage1_complex_array_t, stage1_complex_array_t>( "inverse_z", stage1_, stage1_ );
        transpose_xywz_to_xyzw_( stage1_, stage0_ );

        base_fft_.template exec<stage0_complex_array_t, real_array_t<4>>( "inverse_w", stage0_, out );
    }

    void backward_4d_(
        std::integral_constant<transform_strategy_4d, transform_strategy_4d::slab_slab>,
        const complex_array_t<4> &,
        real_array_t<4>          &out
    )
    {
        transpose_yzwx_to_zwxy_( work_hat_, stage1_ );
        base_fft_.template exec<stage1_complex_array_t, stage1_complex_array_t>( "inverse_xy", stage1_, stage1_ );
        transpose_zwxy_to_xyzw_( stage1_, stage0_ );
        base_fft_.template exec<stage0_complex_array_t, real_array_t<4>>( "inverse_zw", stage0_, out );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_xywz_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        direct_transposer_->xyzw_to_xywz( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_xywz_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        memcpy_transposer_->xyzw_to_xywz( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xzwy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        direct_transposer_->xywz_to_xzwy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xzwy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        memcpy_transposer_->xywz_to_xzwy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_yzwx_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        direct_transposer_->xzwy_to_yzwx( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_yzwx_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        memcpy_transposer_->xzwy_to_yzwx( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_xzwy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        direct_transposer_->yzwx_to_xzwy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_xzwy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        memcpy_transposer_->yzwx_to_xzwy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_xywz_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        direct_transposer_->xzwy_to_xywz( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_xywz_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        memcpy_transposer_->xzwy_to_xywz( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xyzw_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        direct_transposer_->xywz_to_xyzw( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xyzw_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        memcpy_transposer_->xywz_to_xyzw( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_zwxy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        direct_transposer_->xyzw_to_zwxy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_zwxy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        memcpy_transposer_->xyzw_to_zwxy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_xyzw_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        direct_transposer_->zwxy_to_xyzw( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_xyzw_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        memcpy_transposer_->zwxy_to_xyzw( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_yzwx_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        direct_transposer_->zwxy_to_yzwx( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_yzwx_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        memcpy_transposer_->zwxy_to_yzwx( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_zwxy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        direct_transposer_->yzwx_to_zwxy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_zwxy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>,
        const SrcArray &src,
        DstArray &dst
    )
    {
        memcpy_transposer_->yzwx_to_zwxy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_xywz_( const SrcArray &src, DstArray &dst )
    {
        transpose_xyzw_to_xywz_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xzwy_( const SrcArray &src, DstArray &dst )
    {
        transpose_xywz_to_xzwy_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_yzwx_( const SrcArray &src, DstArray &dst )
    {
        transpose_xzwy_to_yzwx_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_xzwy_( const SrcArray &src, DstArray &dst )
    {
        transpose_yzwx_to_xzwy_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_xywz_( const SrcArray &src, DstArray &dst )
    {
        transpose_xzwy_to_xywz_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xyzw_( const SrcArray &src, DstArray &dst )
    {
        transpose_xywz_to_xyzw_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_zwxy_( const SrcArray &src, DstArray &dst )
    {
        transpose_xyzw_to_zwxy_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_xyzw_( const SrcArray &src, DstArray &dst )
    {
        transpose_zwxy_to_xyzw_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_yzwx_( const SrcArray &src, DstArray &dst )
    {
        transpose_zwxy_to_yzwx_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_zwxy_( const SrcArray &src, DstArray &dst )
    {
        transpose_yzwx_to_zwxy_backend_( transpose_backend_tag(), src, dst );
    }

    void copy_spectral_field_( const complex_array_t<4> &src, complex_array_t<4> &dst )
    {
        runtime_api_t::memcpy(
            dst.raw_ptr(),
            src.raw_ptr(),
            sizeof( complex ) * static_cast<std::size_t>( src.total_size() ),
            runtime_api_t::device_to_device_kind()
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

    bool        init_done_;
    std::size_t dim_;

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

    stage0_complex_array_t stage0_;
    stage1_complex_array_t stage1_;
    stage2_complex_array_t stage2_;
    complex_array_t<4>     work_hat_;
};

} // namespace fftm

#endif // __FFTM_FFTS_HPP__
