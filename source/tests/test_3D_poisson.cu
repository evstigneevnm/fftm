#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <scfd/backend/cuda.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/utils/log_std.h>
#include <scfd/utils/scalar_traits.h>
#include <scfd/utils/system_timer_event.h>

#include <external_wrap/cufft_wrap_many.h>
#include <ffts.hpp>

#include "detail/poisson_fft_test_common.h"

template <class FFTS>
class poisson_3d_fft_case
{
public:
    using T         = typename FFTS::real;
    using results_t = std::pair<std::pair<T, T>, T>;

    explicit poisson_3d_fft_case( std::size_t nx, std::size_t ny, std::size_t nz )
        : nx_( nx )
        , ny_( ny )
        , nz_( nz )
        , nz_c_( nz / 2 + 1 )
        , hx_( domain_length() / static_cast<T>( nx_ ) )
        , hy_( domain_length() / static_cast<T>( ny_ ) )
        , hz_( domain_length() / static_cast<T>( nz_ ) )
        , cell_volume_( hx_ * hy_ * hz_ )
        , real_range_( idx_t( 0, 0, 0 ), idx_t( to_int( nx_ ), to_int( ny_ ), to_int( nz_ ) ) )
        , spectral_range_( idx_t( 0, 0, 0 ), idx_t( to_int( nx_ ), to_int( ny_ ), to_int( nz_c_ ) ) )
    {
        if ( nx_ < 2 || ny_ < 2 || nz_ < 2 )
        {
            throw std::logic_error( "test_3D_poisson: all grid dimensions must be at least 2." );
        }

        ffts_.init( nx_, ny_, nz_ );
        allocate_arrays();
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
        return reduce_( rhs_.size(), rhs_.raw_ptr(), T( 0 ) ) / static_cast<T>( rhs_.size() );
    }

    void write_gmsh_outputs( const std::string &prefix ) const
    {
        const T length = domain_length();
        io::write_out_pos_file_scal_3D_hex( prefix + "_rhs.pos", rhs_, length, length, length );
        io::write_out_pos_file_scal_3D_hex( prefix + "_solution.pos", numerical_solution_, length, length, length );
        io::write_out_pos_file_scal_3D_hex( prefix + "_exact.pos", exact_solution_, length, length, length );
    }

private:
    static constexpr int dim = 3;

    using backend_t       = scfd::backend::cuda;
    using for_each_t      = typename backend_t::template for_each_nd_type<dim, int>;
    using reduce_t        = typename backend_t::reduce_type;
    using idx_t           = scfd::static_vec::vec<int, dim>;
    using range_t         = scfd::static_vec::rect<int, dim>;
    using real_array_t    = typename FFTS::template real_array_t<3>;
    using complex_array_t = typename FFTS::template complex_array_t<3>;

public:
    struct fill_problem_functor
    {
        real_array_t rhs;
        real_array_t exact_solution;
        real_array_t exact_dx;
        real_array_t exact_dy;
        real_array_t exact_dz;
        T hx;
        T hy;
        T hz;

        __device__ __host__ void operator()( const idx_t &idx )
        {
            const T x  = hx * static_cast<T>( idx[0] );
            const T y  = hy * static_cast<T>( idx[1] );
            const T z  = hz * static_cast<T>( idx[2] );
            const T pi = scfd::utils::scalar_traits<T>::pi();

            const T dx = x - pi;
            const T dy = y - pi;
            const T dz = z - pi;
            const T exponent = -( dx * dx + dy * dy + dz * dz );

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

            exact_solution( idx ) = T( 100 ) * gaussian * sin_x * sin_y * sin_z;
            exact_dx( idx )       = T( 100 ) * gaussian * ( cos_x - T( 2 ) * dx * sin_x ) * sin_y * sin_z;
            exact_dy( idx )       = T( 100 ) * gaussian * sin_x * ( cos_y - T( 2 ) * dy * sin_y ) * sin_z;
            exact_dz( idx )       = T( 100 ) * gaussian * sin_x * sin_y * ( cos_z - T( 2 ) * dz * sin_z );

            rhs( idx ) = T( 100 ) * gaussian *
                         ( ( T( 4 ) * dx * dx + T( 4 ) * dy * dy + T( 4 ) * dz * dz - T( 9 ) ) *
                               sin_x * sin_y * sin_z -
                           T( 4 ) * dx * cos_x * sin_y * sin_z -
                           T( 4 ) * dy * sin_x * cos_y * sin_z -
                           T( 4 ) * dz * sin_x * sin_y * cos_z );
        }
    };

    struct solve_fourier_functor
    {
        complex_array_t rhs_hat;
        complex_array_t solution_hat;
        int nx;
        int ny;

        __device__ __host__ void operator()( const idx_t &idx )
        {
            const int kx = idx[0] <= nx / 2 ? idx[0] : idx[0] - nx;
            const int ky = idx[1] <= ny / 2 ? idx[1] : idx[1] - ny;
            const int kz = idx[2];

            const T kx_t = static_cast<T>( kx );
            const T ky_t = static_cast<T>( ky );
            const T kz_t = static_cast<T>( kz );
            const T k2   = kx_t * kx_t + ky_t * ky_t + kz_t * kz_t;

            if ( k2 == T( 0 ) )
            {
                solution_hat( idx ).x = T( 0 );
                solution_hat( idx ).y = T( 0 );
                return;
            }

            const T scale = -T( 1 ) / k2;
            solution_hat( idx ).x = scale * rhs_hat( idx ).x;
            solution_hat( idx ).y = scale * rhs_hat( idx ).y;
        }
    };

    struct derivative_spectra_functor
    {
        complex_array_t solution_hat;
        complex_array_t dx_hat;
        complex_array_t dy_hat;
        complex_array_t dz_hat;
        int nx;
        int ny;

        __device__ __host__ void operator()( const idx_t &idx )
        {
            const int kx = idx[0] <= nx / 2 ? idx[0] : idx[0] - nx;
            const int ky = idx[1] <= ny / 2 ? idx[1] : idx[1] - ny;
            const int kz = idx[2];

            const T kx_t = static_cast<T>( kx );
            const T ky_t = static_cast<T>( ky );
            const T kz_t = static_cast<T>( kz );

            const T real_part = solution_hat( idx ).x;
            const T imag_part = solution_hat( idx ).y;

            dx_hat( idx ).x = -kx_t * imag_part;
            dx_hat( idx ).y = kx_t * real_part;

            dy_hat( idx ).x = -ky_t * imag_part;
            dy_hat( idx ).y = ky_t * real_part;

            dz_hat( idx ).x = -kz_t * imag_part;
            dz_hat( idx ).y = kz_t * real_part;
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
        real_array_t exact_solution;
        real_array_t exact_dx;
        real_array_t exact_dy;
        real_array_t exact_dz;
        real_array_t solution_error_sq;
        real_array_t gradient_error_sq;

        __device__ __host__ void operator()( const idx_t &idx )
        {
            const T solution_diff = numerical_solution( idx ) - exact_solution( idx );
            const T dx_diff       = numerical_dx( idx ) - exact_dx( idx );
            const T dy_diff       = numerical_dy( idx ) - exact_dy( idx );
            const T dz_diff       = numerical_dz( idx ) - exact_dz( idx );

            solution_error_sq( idx ) = solution_diff * solution_diff;
            gradient_error_sq( idx ) = dx_diff * dx_diff + dy_diff * dy_diff + dz_diff * dz_diff;
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
        return T( 1 ) / static_cast<T>( nx_ * ny_ * nz_ );
    }

    void allocate_arrays()
    {
        rhs_.init( nx_, ny_, nz_ );
        exact_solution_.init( nx_, ny_, nz_ );
        exact_dx_.init( nx_, ny_, nz_ );
        exact_dy_.init( nx_, ny_, nz_ );
        exact_dz_.init( nx_, ny_, nz_ );

        numerical_solution_.init( nx_, ny_, nz_ );
        numerical_dx_.init( nx_, ny_, nz_ );
        numerical_dy_.init( nx_, ny_, nz_ );
        numerical_dz_.init( nx_, ny_, nz_ );

        solution_error_sq_.init( nx_, ny_, nz_ );
        gradient_error_sq_.init( nx_, ny_, nz_ );

        rhs_hat_.init( nx_, ny_, nz_c_ );
        solution_hat_.init( nx_, ny_, nz_c_ );
        dx_hat_.init( nx_, ny_, nz_c_ );
        dy_hat_.init( nx_, ny_, nz_c_ );
        dz_hat_.init( nx_, ny_, nz_c_ );
    }

    void fill_problem_data()
    {
        for_each_(
            fill_problem_functor{ rhs_, exact_solution_, exact_dx_, exact_dy_, exact_dz_, hx_, hy_, hz_ },
            real_range_
        );
        for_each_.wait();
    }

    void solve_in_fourier_space()
    {
        for_each_(
            solve_fourier_functor{ rhs_hat_, solution_hat_, to_int( nx_ ), to_int( ny_ ) },
            spectral_range_
        );
        for_each_.wait();
    }

    void build_derivative_spectra()
    {
        for_each_(
            derivative_spectra_functor{ solution_hat_, dx_hat_, dy_hat_, dz_hat_, to_int( nx_ ), to_int( ny_ ) },
            spectral_range_
        );
        for_each_.wait();
    }

    void scale_real_field( real_array_t &field, T scale )
    {
        for_each_( scale_real_functor{ field, scale }, real_range_ );
        for_each_.wait();
    }

    T solve()
    {
        scfd::utils::system_timer_event t_begin, t_end;

        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        t_begin.record();

        ffts_.forward( rhs_, rhs_hat_ );
        solve_in_fourier_space();
        ffts_.backward( solution_hat_, numerical_solution_ );
        scale_real_field( numerical_solution_, normalization_factor() );

        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        t_end.record();

        return static_cast<T>( t_end.elapsed_time( t_begin ) );
    }

    std::pair<T, T> validate()
    {
        build_derivative_spectra();

        ffts_.backward( dx_hat_, numerical_dx_ );
        ffts_.backward( dy_hat_, numerical_dy_ );
        ffts_.backward( dz_hat_, numerical_dz_ );

        scale_real_field( numerical_dx_, normalization_factor() );
        scale_real_field( numerical_dy_, normalization_factor() );
        scale_real_field( numerical_dz_, normalization_factor() );

        for_each_(
            error_fields_functor{
                numerical_solution_,
                numerical_dx_,
                numerical_dy_,
                numerical_dz_,
                exact_solution_,
                exact_dx_,
                exact_dy_,
                exact_dz_,
                solution_error_sq_,
                gradient_error_sq_,
            },
            real_range_
        );
        for_each_.wait();

        const T l2_sq = reduce_( solution_error_sq_.size(), solution_error_sq_.raw_ptr(), T( 0 ) ) * cell_volume_;
        const T h1_sq = reduce_( gradient_error_sq_.size(), gradient_error_sq_.raw_ptr(), T( 0 ) ) * cell_volume_;

        return { std::sqrt( l2_sq ), std::sqrt( h1_sq ) };
    }

    FFTS ffts_;

    std::size_t nx_;
    std::size_t ny_;
    std::size_t nz_;
    std::size_t nz_c_;

    T hx_;
    T hy_;
    T hz_;
    T cell_volume_;

    range_t real_range_;
    range_t spectral_range_;

    for_each_t for_each_;
    reduce_t reduce_;

    real_array_t rhs_;
    real_array_t exact_solution_;
    real_array_t exact_dx_;
    real_array_t exact_dy_;
    real_array_t exact_dz_;

    real_array_t numerical_solution_;
    real_array_t numerical_dx_;
    real_array_t numerical_dy_;
    real_array_t numerical_dz_;

    real_array_t solution_error_sq_;
    real_array_t gradient_error_sq_;

    complex_array_t rhs_hat_;
    complex_array_t solution_hat_;
    complex_array_t dx_hat_;
    complex_array_t dy_hat_;
    complex_array_t dz_hat_;
};

template <class FFTS>
using results_t = typename poisson_3d_fft_case<FFTS>::results_t;

std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> parse_grids( int argc, char *argv[] )
{
    if ( argc == 1 )
    {
        return {
            { 16, 16, 16 },
            { 32, 32, 32 },
            { 64, 64, 64 },
        };
    }

    if ( ( ( argc - 1 ) % 3 ) != 0 )
    {
        throw std::logic_error( "USAGE: test_3D_poisson.bin Nx1 Ny1 Nz1 [Nx2 Ny2 Nz2 ...]" );
    }

    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> grids;
    grids.reserve( ( argc - 1 ) / 3 );
    for ( int arg_i = 1; arg_i < argc; arg_i += 3 )
    {
        grids.emplace_back(
            static_cast<std::size_t>( std::stoul( argv[arg_i] ) ),
            static_cast<std::size_t>( std::stoul( argv[arg_i + 1] ) ),
            static_cast<std::size_t>( std::stoul( argv[arg_i + 2] ) )
        );
    }
    return grids;
}

int main( int argc, char *argv[] )
{
    scfd::utils::log_std log;

    try
    {
        using base_fft_t = fftm::wrap::cufft_wrap_many<double>;
        using ffts_t     = fftm::ffts<base_fft_t, scfd::backend::cuda>;
        using grid_t     = std::tuple<std::size_t, std::size_t, std::size_t>;
        using run_t      = std::pair<grid_t, results_t<ffts_t>>;

        scfd::utils::init_cuda_persistent( log, 0 );

        const auto grids = parse_grids( argc, argv );

        std::vector<run_t> runs;
        runs.reserve( grids.size() );

        for ( const auto &grid : grids )
        {
            const auto nx = std::get<0>( grid );
            const auto ny = std::get<1>( grid );
            const auto nz = std::get<2>( grid );

            poisson_3d_fft_case<ffts_t> poisson_case( nx, ny, nz );
            const double                rhs_mean = poisson_case.rhs_mean();

            if ( std::abs( rhs_mean ) > 1.0e-12 )
            {
                log.warning_f(
                    "rhs_mean = %.8e for Nx=%zu, Ny=%zu, Nz=%zu; the periodic FFT solve removes the zero Fourier mode,"
                    " so the manufactured rhs should have zero mean.",
                    rhs_mean,
                    nx,
                    ny,
                    nz
                );
            }

            runs.push_back( { grid, poisson_case.run() } );
            poisson_case.write_gmsh_outputs(
                "poisson_3d_" + std::to_string( nx ) + "x" + std::to_string( ny ) + "x" + std::to_string( nz )
            );
        }

        for ( const auto &run : runs )
        {
            const auto &grid = run.first;
            const auto &res  = run.second;

            log.info_f(
                "Nx=%zu, Ny=%zu, Nz=%zu: L2=%.8e, H1=%.8e, wall_ms=%.8e",
                std::get<0>( grid ),
                std::get<1>( grid ),
                std::get<2>( grid ),
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
