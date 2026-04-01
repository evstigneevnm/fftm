#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/backend/cuda.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/utils/log_std.h>
#include <scfd/utils/scalar_traits.h>
#include <scfd/utils/system_timer_event.h>

#include <external_wrap/cufft_wrap_many.h>

#include "detail/cuda_memcpy_4d_slab_transposer.h"
#include "detail/poisson_fft_test_common.h"

template <class T>
class poisson_4d_fft_case
{
public:
    using results_t = std::pair<std::pair<T, T>, T>;

    poisson_4d_fft_case( std::size_t nx, std::size_t ny, std::size_t nz, std::size_t nw )
        : nx_( nx )
        , ny_( ny )
        , nz_( nz )
        , nw_( nw )
        , nw_half_( nw / 2 + 1 )
        , hx_( domain_length() / static_cast<T>( nx_ ) )
        , hy_( domain_length() / static_cast<T>( ny_ ) )
        , hz_( domain_length() / static_cast<T>( nz_ ) )
        , hw_( domain_length() / static_cast<T>( nw_ ) )
        , cell_volume_( hx_ * hy_ * hz_ * hw_ )
        , real_range_( idx_t( 0, 0, 0, 0 ), idx_t( to_int( nx_ ), to_int( ny_ ), to_int( nz_ ), to_int( nw_ ) ) )
        , spectral_range_yzwx_(
              idx_t( 0, 0, 0, 0 ),
              idx_t( to_int( ny_ ), to_int( nz_ ), to_int( nw_half_ ), to_int( nx_ ) )
          )
        , transposer_( nx_, ny_, nz_, nw_half_ )
    {
        if ( nx_ < 2 || ny_ < 2 || nz_ < 2 || nw_ < 2 )
        {
            throw std::logic_error( "test_4D_poisson: all grid dimensions must be at least 2." );
        }

        allocate_arrays();
        init_fft();
        for_each_.block_size = 128;
        fill_problem_data();
    }

    results_t run()
    {
        const T wall_time_ms = solve();
        return { validate(), wall_time_ms };
    }

    T rhs_mean() const
    {
        return reduce_( rhs_.total_size(), rhs_.raw_ptr(), T( 0 ) ) / static_cast<T>( rhs_.total_size() );
    }

private:
    static constexpr int dim = 4;

    using backend_t  = scfd::backend::cuda;
    using memory_t   = typename backend_t::memory_type;
    using for_each_t = typename backend_t::template for_each_nd_type<dim, int>;
    using reduce_t   = typename backend_t::reduce_type;
    using base_fft_t = fftm::wrap::cufft_wrap_many<T>;
    using complex_t  = typename base_fft_t::complex;
    using idx_t      = scfd::static_vec::vec<int, dim>;
    using range_t    = scfd::static_vec::rect<int, dim>;

    using real_array_t = scfd::arrays::tensor_array_nd<T, dim, memory_t, scfd::arrays::custom_arranger_0213_t>;
    using xyzw_complex_array_t =
        scfd::arrays::tensor_array_nd<complex_t, dim, memory_t, scfd::arrays::custom_arranger_0213_t>;
    using xywz_complex_array_t =
        scfd::arrays::tensor_array_nd<complex_t, dim, memory_t, scfd::arrays::custom_arranger_3012_t>;
    using xzwy_complex_array_t =
        scfd::arrays::tensor_array_nd<complex_t, dim, memory_t, scfd::arrays::custom_arranger_1023_t>;
    using yzwx_complex_array_t =
        scfd::arrays::tensor_array_nd<complex_t, dim, memory_t, scfd::arrays::custom_arranger_3120_t>;

public:
    struct fill_problem_functor
    {
        real_array_t rhs;
        real_array_t exact_solution;
        real_array_t exact_dx;
        real_array_t exact_dy;
        real_array_t exact_dz;
        real_array_t exact_dw;
        T hx;
        T hy;
        T hz;
        T hw;

        __device__ __host__ void operator()( const idx_t &idx )
        {
            const T x  = hx * static_cast<T>( idx[0] );
            const T y  = hy * static_cast<T>( idx[1] );
            const T z  = hz * static_cast<T>( idx[2] );
            const T w  = hw * static_cast<T>( idx[3] );
            const T pi = scfd::utils::scalar_traits<T>::pi();

            const T dx = x - pi;
            const T dy = y - pi;
            const T dz = z - pi;
            const T dw = w - pi;
            const T exponent = -( dx * dx + dy * dy + dz * dz + dw * dw );

            T gaussian;
#ifndef __CUDA_ARCH__
            gaussian = std::exp( exponent );
#else
            gaussian = ::exp( exponent );
#endif

            const T sin_x = scfd::utils::scalar_traits<T>::sin( x );
            const T cos_x = scfd::utils::scalar_traits<T>::cos( x );
            const T sin_y = scfd::utils::scalar_traits<T>::sin( y );
            const T cos_y = scfd::utils::scalar_traits<T>::cos( y );
            const T sin_z = scfd::utils::scalar_traits<T>::sin( z );
            const T cos_z = scfd::utils::scalar_traits<T>::cos( z );
            const T sin_w = scfd::utils::scalar_traits<T>::sin( w );
            const T cos_w = scfd::utils::scalar_traits<T>::cos( w );

            exact_solution( idx ) = T( 100 ) * gaussian * sin_x * sin_y * sin_z * sin_w;
            exact_dx( idx ) =
                T( 100 ) * gaussian * ( cos_x - T( 2 ) * dx * sin_x ) * sin_y * sin_z * sin_w;
            exact_dy( idx ) =
                T( 100 ) * gaussian * sin_x * ( cos_y - T( 2 ) * dy * sin_y ) * sin_z * sin_w;
            exact_dz( idx ) =
                T( 100 ) * gaussian * sin_x * sin_y * ( cos_z - T( 2 ) * dz * sin_z ) * sin_w;
            exact_dw( idx ) =
                T( 100 ) * gaussian * sin_x * sin_y * sin_z * ( cos_w - T( 2 ) * dw * sin_w );

            rhs( idx ) = T( 100 ) * gaussian *
                         ( ( T( 4 ) * dx * dx + T( 4 ) * dy * dy + T( 4 ) * dz * dz + T( 4 ) * dw * dw - T( 12 ) ) *
                               sin_x * sin_y * sin_z * sin_w -
                           T( 4 ) * dx * cos_x * sin_y * sin_z * sin_w -
                           T( 4 ) * dy * sin_x * cos_y * sin_z * sin_w -
                           T( 4 ) * dz * sin_x * sin_y * cos_z * sin_w -
                           T( 4 ) * dw * sin_x * sin_y * sin_z * cos_w );
        }
    };

    struct solve_fourier_functor
    {
        yzwx_complex_array_t field_hat;
        int nx;
        int ny;
        int nz;

        __device__ __host__ void operator()( const idx_t &idx )
        {
            const int ky = idx[0] <= ny / 2 ? idx[0] : idx[0] - ny;
            const int kz = idx[1] <= nz / 2 ? idx[1] : idx[1] - nz;
            const int kw = idx[2];
            const int kx = idx[3] <= nx / 2 ? idx[3] : idx[3] - nx;

            const T kx_t = static_cast<T>( kx );
            const T ky_t = static_cast<T>( ky );
            const T kz_t = static_cast<T>( kz );
            const T kw_t = static_cast<T>( kw );
            const T k2   = kx_t * kx_t + ky_t * ky_t + kz_t * kz_t + kw_t * kw_t;

            if ( k2 == T( 0 ) )
            {
                field_hat( idx ).x = T( 0 );
                field_hat( idx ).y = T( 0 );
                return;
            }

            const T scale = -T( 1 ) / k2;
            field_hat( idx ).x *= scale;
            field_hat( idx ).y *= scale;
        }
    };

    struct derivative_spectrum_functor
    {
        yzwx_complex_array_t solution_hat;
        yzwx_complex_array_t field_hat;
        int axis;
        int nx;
        int ny;
        int nz;

        __device__ __host__ void operator()( const idx_t &idx )
        {
            const int ky = idx[0] <= ny / 2 ? idx[0] : idx[0] - ny;
            const int kz = idx[1] <= nz / 2 ? idx[1] : idx[1] - nz;
            const int kw = idx[2];
            const int kx = idx[3] <= nx / 2 ? idx[3] : idx[3] - nx;

            T wave_number = T( 0 );
            if ( axis == 0 )
            {
                wave_number = static_cast<T>( kx );
            }
            else if ( axis == 1 )
            {
                wave_number = static_cast<T>( ky );
            }
            else if ( axis == 2 )
            {
                wave_number = static_cast<T>( kz );
            }
            else
            {
                wave_number = static_cast<T>( kw );
            }

            const T real_part = solution_hat( idx ).x;
            const T imag_part = solution_hat( idx ).y;

            field_hat( idx ).x = -wave_number * imag_part;
            field_hat( idx ).y = wave_number * real_part;
        }
    };

    struct scale_real_functor
    {
        real_array_t field;
        T scale;

        __device__ __host__ void operator()( const idx_t &idx )
        {
            field( idx ) *= scale;
        }
    };

    struct error_fields_functor
    {
        real_array_t numerical_solution;
        real_array_t numerical_dx;
        real_array_t numerical_dy;
        real_array_t numerical_dz;
        real_array_t numerical_dw;
        real_array_t exact_solution;
        real_array_t exact_dx;
        real_array_t exact_dy;
        real_array_t exact_dz;
        real_array_t exact_dw;
        real_array_t solution_error_sq;
        real_array_t gradient_error_sq;

        __device__ __host__ void operator()( const idx_t &idx )
        {
            const T solution_diff = numerical_solution( idx ) - exact_solution( idx );
            const T dx_diff       = numerical_dx( idx ) - exact_dx( idx );
            const T dy_diff       = numerical_dy( idx ) - exact_dy( idx );
            const T dz_diff       = numerical_dz( idx ) - exact_dz( idx );
            const T dw_diff       = numerical_dw( idx ) - exact_dw( idx );

            solution_error_sq( idx ) = solution_diff * solution_diff;
            gradient_error_sq( idx ) = dx_diff * dx_diff + dy_diff * dy_diff + dz_diff * dz_diff + dw_diff * dw_diff;
        }
    };

private:
    static T domain_length()
    {
        return T( 2 ) * scfd::utils::scalar_traits<T>::pi();
    }

    static int to_int( std::size_t value )
    {
        return static_cast<int>( value );
    }

    T normalization_factor() const
    {
        return T( 1 ) / static_cast<T>( nx_ * ny_ * nz_ * nw_ );
    }

    void allocate_arrays()
    {
        rhs_.init( nx_, ny_, nz_, nw_ );
        exact_solution_.init( nx_, ny_, nz_, nw_ );
        exact_dx_.init( nx_, ny_, nz_, nw_ );
        exact_dy_.init( nx_, ny_, nz_, nw_ );
        exact_dz_.init( nx_, ny_, nz_, nw_ );
        exact_dw_.init( nx_, ny_, nz_, nw_ );

        numerical_solution_.init( nx_, ny_, nz_, nw_ );
        numerical_dx_.init( nx_, ny_, nz_, nw_ );
        numerical_dy_.init( nx_, ny_, nz_, nw_ );
        numerical_dz_.init( nx_, ny_, nz_, nw_ );
        numerical_dw_.init( nx_, ny_, nz_, nw_ );

        solution_error_sq_.init( nx_, ny_, nz_, nw_ );
        gradient_error_sq_.init( nx_, ny_, nz_, nw_ );

        xyzw_stage_.init( nx_, ny_, nz_, nw_half_ );
        xywz_stage_.init( nx_, ny_, nw_half_, nz_ );
        xzwy_stage_.init( nx_, nz_, nw_half_, ny_ );
        solution_hat_.init( ny_, nz_, nw_half_, nx_ );
        work_hat_.init( ny_, nz_, nw_half_, nx_ );
    }

    void init_fft()
    {
        const long long int real_batch = static_cast<long long int>( nx_ * ny_ * nz_ );
        const long long int z_batch    = static_cast<long long int>( nx_ * ny_ * nw_half_ );
        const long long int y_batch    = static_cast<long long int>( nx_ * nz_ * nw_half_ );
        const long long int x_batch    = static_cast<long long int>( ny_ * nz_ * nw_half_ );

        const long long int w_stride = static_cast<long long int>( nx_ * ny_ * nz_ );
        const long long int y_stride = static_cast<long long int>( nx_ * nz_ * nw_half_ );

        fft_.template add_plan_1D<fftm::direction::R2C>(
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

        fft_.template add_plan_1D<fftm::direction::C2R>(
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

        fft_.template add_plan_1D<fftm::direction::C2CF>(
            "forward_z",
            static_cast<long long int>( nz_ ),
            1,
            1,
            static_cast<long long int>( nz_ ),
            1,
            1,
            static_cast<long long int>( nz_ ),
            z_batch
        );

        fft_.template add_plan_1D<fftm::direction::C2CB>(
            "inverse_z",
            static_cast<long long int>( nz_ ),
            1,
            1,
            static_cast<long long int>( nz_ ),
            1,
            1,
            static_cast<long long int>( nz_ ),
            z_batch
        );

        fft_.template add_plan_1D<fftm::direction::C2CF>(
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

        fft_.template add_plan_1D<fftm::direction::C2CB>(
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

        fft_.template add_plan_1D<fftm::direction::C2CF>(
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

        fft_.template add_plan_1D<fftm::direction::C2CB>(
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

        fft_.activate();
    }

    void fill_problem_data()
    {
        for_each_(
            fill_problem_functor{
                rhs_,
                exact_solution_,
                exact_dx_,
                exact_dy_,
                exact_dz_,
                exact_dw_,
                hx_,
                hy_,
                hz_,
                hw_,
            },
            real_range_
        );
        for_each_.wait();
    }

    void forward_to_spectral()
    {
        fft_.template exec<real_array_t, xyzw_complex_array_t>( "forward_w", rhs_, xyzw_stage_ );
        transposer_.xyzw_to_xywz( xyzw_stage_, xywz_stage_ );

        fft_.template exec<xywz_complex_array_t, xywz_complex_array_t>( "forward_z", xywz_stage_, xywz_stage_ );
        transposer_.xywz_to_xzwy( xywz_stage_, xzwy_stage_ );

        fft_.template exec<xzwy_complex_array_t, xzwy_complex_array_t>( "forward_y", xzwy_stage_, xzwy_stage_ );
        transposer_.xzwy_to_yzwx( xzwy_stage_, solution_hat_ );

        fft_.template exec<yzwx_complex_array_t, yzwx_complex_array_t>( "forward_x", solution_hat_, solution_hat_ );
    }

    void solve_in_fourier_space()
    {
        for_each_(
            solve_fourier_functor{
                solution_hat_,
                to_int( nx_ ),
                to_int( ny_ ),
                to_int( nz_ ),
            },
            spectral_range_yzwx_
        );
        for_each_.wait();
    }

    void copy_spectral_field( const yzwx_complex_array_t &src, yzwx_complex_array_t &dst )
    {
        CUDA_SAFE_CALL(
            cudaMemcpy(
                dst.raw_ptr(),
                src.raw_ptr(),
                sizeof( complex_t ) * static_cast<std::size_t>( src.total_size() ),
                cudaMemcpyDeviceToDevice
            )
        );
    }

    void scale_real_field( real_array_t &field, T scale )
    {
        for_each_( scale_real_functor{ field, scale }, real_range_ );
        for_each_.wait();
    }

    void inverse_complex_field( const yzwx_complex_array_t &field_hat, real_array_t &out_real )
    {
        copy_spectral_field( field_hat, work_hat_ );

        fft_.template exec<yzwx_complex_array_t, yzwx_complex_array_t>( "inverse_x", work_hat_, work_hat_ );
        transposer_.yzwx_to_xzwy( work_hat_, xzwy_stage_ );

        fft_.template exec<xzwy_complex_array_t, xzwy_complex_array_t>( "inverse_y", xzwy_stage_, xzwy_stage_ );
        transposer_.xzwy_to_xywz( xzwy_stage_, xywz_stage_ );

        fft_.template exec<xywz_complex_array_t, xywz_complex_array_t>( "inverse_z", xywz_stage_, xywz_stage_ );
        transposer_.xywz_to_xyzw( xywz_stage_, xyzw_stage_ );

        fft_.template exec<xyzw_complex_array_t, real_array_t>( "inverse_w", xyzw_stage_, out_real );
        scale_real_field( out_real, normalization_factor() );
    }

    void build_derivative_spectrum( int axis )
    {
        for_each_(
            derivative_spectrum_functor{
                solution_hat_,
                work_hat_,
                axis,
                to_int( nx_ ),
                to_int( ny_ ),
                to_int( nz_ ),
            },
            spectral_range_yzwx_
        );
        for_each_.wait();
    }

    T solve()
    {
        scfd::utils::system_timer_event t_begin, t_end;

        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        t_begin.record();

        forward_to_spectral();
        solve_in_fourier_space();
        inverse_complex_field( solution_hat_, numerical_solution_ );

        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        t_end.record();

        return static_cast<T>( t_end.elapsed_time( t_begin ) );
    }

    std::pair<T, T> validate()
    {
        build_derivative_spectrum( 0 );
        inverse_complex_field( work_hat_, numerical_dx_ );

        build_derivative_spectrum( 1 );
        inverse_complex_field( work_hat_, numerical_dy_ );

        build_derivative_spectrum( 2 );
        inverse_complex_field( work_hat_, numerical_dz_ );

        build_derivative_spectrum( 3 );
        inverse_complex_field( work_hat_, numerical_dw_ );

        for_each_(
            error_fields_functor{
                numerical_solution_,
                numerical_dx_,
                numerical_dy_,
                numerical_dz_,
                numerical_dw_,
                exact_solution_,
                exact_dx_,
                exact_dy_,
                exact_dz_,
                exact_dw_,
                solution_error_sq_,
                gradient_error_sq_,
            },
            real_range_
        );
        for_each_.wait();

        const T l2_sq =
            reduce_( solution_error_sq_.total_size(), solution_error_sq_.raw_ptr(), T( 0 ) ) * cell_volume_;
        const T h1_sq =
            reduce_( gradient_error_sq_.total_size(), gradient_error_sq_.raw_ptr(), T( 0 ) ) * cell_volume_;

        return { std::sqrt( l2_sq ), std::sqrt( h1_sq ) };
    }

    std::size_t nx_;
    std::size_t ny_;
    std::size_t nz_;
    std::size_t nw_;
    std::size_t nw_half_;

    T hx_;
    T hy_;
    T hz_;
    T hw_;
    T cell_volume_;

    range_t real_range_;
    range_t spectral_range_yzwx_;

    for_each_t for_each_;
    reduce_t   reduce_;
    base_fft_t fft_;
    fftm::tests::detail::cuda_memcpy_4d_slab_transposer<complex_t> transposer_;

    real_array_t rhs_;
    real_array_t exact_solution_;
    real_array_t exact_dx_;
    real_array_t exact_dy_;
    real_array_t exact_dz_;
    real_array_t exact_dw_;

    real_array_t numerical_solution_;
    real_array_t numerical_dx_;
    real_array_t numerical_dy_;
    real_array_t numerical_dz_;
    real_array_t numerical_dw_;

    real_array_t solution_error_sq_;
    real_array_t gradient_error_sq_;

    xyzw_complex_array_t xyzw_stage_;
    xywz_complex_array_t xywz_stage_;
    xzwy_complex_array_t xzwy_stage_;
    yzwx_complex_array_t solution_hat_;
    yzwx_complex_array_t work_hat_;
};

template <class T>
using results_4d_t = typename poisson_4d_fft_case<T>::results_t;

std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>> parse_grids_4d( int argc, char *argv[] )
{
    if ( argc == 1 )
    {
        return {
            { 8, 8, 8, 8 },
            { 16, 16, 16, 16 },
            { 32, 32, 32, 32 },
        };
    }

    if ( ( ( argc - 1 ) % 4 ) != 0 )
    {
        throw std::logic_error( "USAGE: test_4D_poisson.bin Nx1 Ny1 Nz1 Nw1 [Nx2 Ny2 Nz2 Nw2 ...]" );
    }

    std::vector<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>> grids;
    grids.reserve( ( argc - 1 ) / 4 );
    for ( int arg_i = 1; arg_i < argc; arg_i += 4 )
    {
        grids.emplace_back(
            static_cast<std::size_t>( std::stoul( argv[arg_i] ) ),
            static_cast<std::size_t>( std::stoul( argv[arg_i + 1] ) ),
            static_cast<std::size_t>( std::stoul( argv[arg_i + 2] ) ),
            static_cast<std::size_t>( std::stoul( argv[arg_i + 3] ) )
        );
    }
    return grids;
}

int main( int argc, char *argv[] )
{
    scfd::utils::log_std log;

    try
    {
        using T      = double;
        using grid_t = std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>;
        using run_t  = std::pair<grid_t, results_4d_t<T>>;

        scfd::utils::init_cuda_persistent( log, 0 );

        const auto grids = parse_grids_4d( argc, argv );

        std::vector<run_t> runs;
        runs.reserve( grids.size() );

        for ( const auto &grid : grids )
        {
            const auto nx = std::get<0>( grid );
            const auto ny = std::get<1>( grid );
            const auto nz = std::get<2>( grid );
            const auto nw = std::get<3>( grid );

            poisson_4d_fft_case<T> poisson_case( nx, ny, nz, nw );
            const T                rhs_mean = poisson_case.rhs_mean();

            if ( std::abs( rhs_mean ) > T( 1.0e-12 ) )
            {
                log.warning_f(
                    "rhs_mean = %.8e for Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu; the periodic FFT solve removes the zero Fourier mode,"
                    " so the manufactured rhs should have zero mean.",
                    rhs_mean,
                    nx,
                    ny,
                    nz,
                    nw
                );
            }

            runs.push_back( { grid, poisson_case.run() } );
        }

        for ( const auto &run : runs )
        {
            const auto &grid = run.first;
            const auto &res  = run.second;

            log.info_f(
                "Nx=%zu, Ny=%zu, Nz=%zu, Nw=%zu: L2=%.8e, H1=%.8e, wall_ms=%.8e",
                std::get<0>( grid ),
                std::get<1>( grid ),
                std::get<2>( grid ),
                std::get<3>( grid ),
                res.first.first,
                res.first.second,
                res.second
            );
        }

        return 0;
    }
    catch ( const std::exception &e )
    {
        log.error( e.what() );
        return 1;
    }
}
