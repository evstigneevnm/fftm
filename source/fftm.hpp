#ifndef __FFTM_FFTM_HPP__
#define __FFTM_FFTM_HPP__

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_mpi.h>

#include "detail/array_arrangers.h"
#include "detail/direct_transpose_4d.h"
#include "detail/mpi_transpose_3d.h"
#include "detail/mpi_transpose_4d.h"
#include "fft_direction.h"
#include "fft_partitioning.h"

namespace fftm
{

enum class transform_strategy_3d
{
    slab_pencil,
    pencil_slab,
    pencil_pencil
};

enum class transform_strategy_4d_mpi
{
    pencil_pencil,
    slab_slab
};

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv>
struct strategy_3d_slab_pencil
{
};

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv>
struct strategy_3d_pencil_slab
{
};

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv>
struct strategy_3d_pencil_pencil
{
};

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv>
struct strategy_4d_pencil_pencil_mpi
{
};

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv>
struct strategy_4d_slab_slab_mpi
{
};

struct fftm_init_options
{
};

namespace detail
{

template <class Strategy3D>
struct fftm_3d_strategy_traits;

template <mpi_transpose_3d_mode Mode>
struct fftm_3d_strategy_traits<strategy_3d_slab_pencil<Mode>>
{
    static constexpr transform_strategy_3d family = transform_strategy_3d::slab_pencil;
    static constexpr mpi_transpose_3d_mode mode   = Mode;

    static const char *name()
    {
        return "slab-pencil";
    }
};

template <mpi_transpose_3d_mode Mode>
struct fftm_3d_strategy_traits<strategy_3d_pencil_slab<Mode>>
{
    static constexpr transform_strategy_3d family = transform_strategy_3d::pencil_slab;
    static constexpr mpi_transpose_3d_mode mode   = Mode;

    static const char *name()
    {
        return "pencil-slab";
    }
};

template <mpi_transpose_3d_mode Mode>
struct fftm_3d_strategy_traits<strategy_3d_pencil_pencil<Mode>>
{
    static constexpr transform_strategy_3d family = transform_strategy_3d::pencil_pencil;
    static constexpr mpi_transpose_3d_mode mode   = Mode;

    static const char *name()
    {
        return "pencil-pencil";
    }
};

template <class Strategy4D>
struct fftm_4d_strategy_traits;

template <mpi_transpose_3d_mode Mode>
struct fftm_4d_strategy_traits<strategy_4d_pencil_pencil_mpi<Mode>>
{
    static constexpr transform_strategy_4d_mpi family = transform_strategy_4d_mpi::pencil_pencil;
    static constexpr mpi_transpose_3d_mode     mode   = Mode;

    static const char *name()
    {
        return "pencil-pencil";
    }
};

template <mpi_transpose_3d_mode Mode>
struct fftm_4d_strategy_traits<strategy_4d_slab_slab_mpi<Mode>>
{
    static constexpr transform_strategy_4d_mpi family = transform_strategy_4d_mpi::slab_slab;
    static constexpr mpi_transpose_3d_mode     mode   = Mode;

    static const char *name()
    {
        return "slab-slab";
    }
};

template <class Real, class Complex, class Memory, class Strategy3D>
struct fftm_3d_array_traits;

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_3d_array_traits<Real, Complex, Memory, strategy_3d_slab_pencil<Mode>>
{
    using real_array_t           = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage0_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using stage1_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using complex_array_t        = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using x_fft_complex_array_t  = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
};

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_3d_array_traits<Real, Complex, Memory, strategy_3d_pencil_slab<Mode>>
{
    using real_array_t           = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage0_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage1_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using complex_array_t        = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using x_fft_complex_array_t  = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
};

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_3d_array_traits<Real, Complex, Memory, strategy_3d_pencil_pencil<Mode>>
{
    using real_array_t           = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage0_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage1_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using complex_array_t        = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using x_fft_complex_array_t  = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
};

template <class Real, class Complex, class Memory, class Strategy4D>
struct fftm_4d_array_traits;

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_4d_array_traits<Real, Complex, Memory, strategy_4d_pencil_pencil_mpi<Mode>>
{
    using real_array_t           = scfd::arrays::tensor_array_nd<Real, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using stage0_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using stage1_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using stage2_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0213_t>;
    using complex_array_t        = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
};

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_4d_array_traits<Real, Complex, Memory, strategy_4d_slab_slab_mpi<Mode>>
{
    using real_array_t           = scfd::arrays::tensor_array_nd<Real, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using stage0_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using stage1_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using stage2_complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0213_t>;
    using complex_array_t        = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_2103_t>;
};

template <class ArrayIn, class ArrayOut, class Idx>
struct fftm_copy_same_indices_functor
{
    ArrayIn  in;
    ArrayOut out;

    __DEVICE_TAG__ void operator()( const Idx &idx ) const
    {
        out( idx ) = in( idx );
    }
};

} // namespace detail

template <
    class BaseFFT,
    class MPIComm,
    class Backend,
    class Strategy3D = strategy_3d_pencil_pencil<mpi_transpose_3d_mode::alltoallv>,
    class Log        = scfd::utils::log_mpi,
    class Strategy4D = strategy_4d_pencil_pencil_mpi<mpi_transpose_3d_mode::alltoallv>
>
class fftm
{
public:
    using real         = typename BaseFFT::real;
    using complex      = typename BaseFFT::complex;
    using runtime_api_t = typename BaseFFT::runtime_api;
    using memory_t     = typename Backend::memory_type;
    using partition_t  = ::fftm::partition;
    using strategy_3d_t = Strategy3D;
    using strategy_4d_t = Strategy4D;

    template <std::size_t Dim>
    using real_array_t = typename std::conditional<
        Dim == 3,
        typename detail::fftm_3d_array_traits<real, complex, memory_t, Strategy3D>::real_array_t,
        typename std::conditional<
            Dim == 4,
            typename detail::fftm_4d_array_traits<real, complex, memory_t, Strategy4D>::real_array_t,
            void
        >::type
    >::type;

    template <std::size_t Dim>
    using complex_array_t = typename std::conditional<
        Dim == 3,
        typename detail::fftm_3d_array_traits<real, complex, memory_t, Strategy3D>::complex_array_t,
        typename std::conditional<
            Dim == 4,
            typename detail::fftm_4d_array_traits<real, complex, memory_t, Strategy4D>::complex_array_t,
            void
        >::type
    >::type;

    static constexpr transform_strategy_3d     strategy_family_3d = detail::fftm_3d_strategy_traits<Strategy3D>::family;
    static constexpr mpi_transpose_3d_mode     transpose_mode_3d  = detail::fftm_3d_strategy_traits<Strategy3D>::mode;
    static constexpr transform_strategy_4d_mpi strategy_family_4d = detail::fftm_4d_strategy_traits<Strategy4D>::family;
    static constexpr mpi_transpose_3d_mode     transpose_mode_4d  = detail::fftm_4d_strategy_traits<Strategy4D>::mode;

    static const char *strategy_name()
    {
        return detail::fftm_3d_strategy_traits<Strategy3D>::name();
    }

    static const char *strategy_name_4d()
    {
        return detail::fftm_4d_strategy_traits<Strategy4D>::name();
    }

    explicit fftm( const MPIComm &mpi, const Log &log = Log() )
        : mpi_( mpi )
        , log_( log )
        , partitioning_( mpi )
        , same_x_( mpi, log )
        , same_z_( mpi, log )
        , same_xy_( mpi, log )
        , same_xw_( mpi, log )
        , same_zw_( mpi, log )
    {
        for_each_3d_.block_size = 128;
        for_each_4d_.block_size = 128;
    }

    template <std::size_t Dim>
    void init( const processor_grid &pg, const global_sizes &gs, const fftm_init_options &options = fftm_init_options() )
    {
        init_( std::integral_constant<std::size_t, Dim>(), pg, gs, options );
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

    std::tuple<std::size_t, std::size_t, std::size_t> get_local_input_sizes() const
    {
        ensure_dimension_( 3 );
        return std::make_tuple(
            input_dim_.size_x[myid_i_],
            input_dim_.size_y[myid_j_],
            input_dim_.size_z[0]
        );
    }

    std::tuple<std::size_t, std::size_t, std::size_t> get_local_output_sizes() const
    {
        ensure_dimension_( 3 );
        return std::make_tuple(
            output_dim_.size_x[0],
            output_dim_.size_z[myid_j_],
            output_dim_.size_y[myid_i_]
        );
    }

    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> get_local_input_sizes_4d() const
    {
        ensure_dimension_( 4 );
        return std::make_tuple(
            input_dim_.size_x[myid_i_],
            input_dim_.size_y[myid_j_],
            input_dim_.size_z[myid_k_],
            input_dim_.size_w[0]
        );
    }

    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> get_local_output_sizes_4d() const
    {
        ensure_dimension_( 4 );
        return std::make_tuple(
            output_dim_.size_y[myid_i_],
            output_dim_.size_z[myid_j_],
            output_dim_.size_w[myid_k_],
            output_dim_.size_x[0]
        );
    }

    const partition_t &input_partition() const
    {
        ensure_initialized_();
        return input_dim_;
    }

    const partition_t &output_partition() const
    {
        ensure_initialized_();
        return output_dim_;
    }

    void forward( const real_array_t<3> &in, complex_array_t<3> &out )
    {
        ensure_dimension_( 3 );
        forward_3d_( strategy_family_3d_tag(), in, out );
    }

    void backward( const complex_array_t<3> &in, real_array_t<3> &out )
    {
        ensure_dimension_( 3 );
        backward_3d_( strategy_family_3d_tag(), in, out );
    }

    void forward( const real_array_t<4> &in, complex_array_t<4> &out )
    {
        ensure_dimension_( 4 );
        forward_4d_( strategy_family_4d_tag(), in, out );
    }

    void backward( const complex_array_t<4> &in, real_array_t<4> &out )
    {
        ensure_dimension_( 4 );
        backward_4d_( strategy_family_4d_tag(), in, out );
    }

private:
    using strategy_family_3d_tag = std::integral_constant<transform_strategy_3d, strategy_family_3d>;
    using strategy_family_4d_tag = std::integral_constant<transform_strategy_4d_mpi, strategy_family_4d>;

    using traits_3d_t = detail::fftm_3d_array_traits<real, complex, memory_t, Strategy3D>;
    using traits_4d_t = detail::fftm_4d_array_traits<real, complex, memory_t, Strategy4D>;

    using idx_3d_t  = scfd::static_vec::vec<int, 3>;
    using rect_3d_t = scfd::static_vec::rect<int, 3>;
    using idx_4d_t  = scfd::static_vec::vec<int, 4>;
    using rect_4d_t = scfd::static_vec::rect<int, 4>;

    using real_array3_t       = typename traits_3d_t::real_array_t;
    using stage0_complex3_t   = typename traits_3d_t::stage0_complex_array_t;
    using complex_array3_t    = typename traits_3d_t::complex_array_t;
    using x_fft_complex3_t    = typename traits_3d_t::x_fft_complex_array_t;
    using stage1_complex3_t   = typename std::conditional<
        strategy_family_3d == transform_strategy_3d::pencil_pencil,
        typename traits_3d_t::stage1_complex_array_t,
        complex_array3_t
    >::type;

    using real_array4_t       = typename traits_4d_t::real_array_t;
    using stage0_complex4_t   = typename traits_4d_t::stage0_complex_array_t;
    using stage1_complex4_t   = typename traits_4d_t::stage1_complex_array_t;
    using stage2_complex4_t   = typename traits_4d_t::stage2_complex_array_t;
    using complex_array4_t    = typename traits_4d_t::complex_array_t;

    using partitioning_t      = fft_partitioning<MPIComm>;
    using same_x_t            = ::fftm::mpi_transpose_3d<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_z_t            = ::fftm::mpi_transpose_3d_same_z<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_xy_t           = ::fftm::detail::mpi_transpose_4d_same_xy<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_xw_t           = ::fftm::detail::mpi_transpose_4d_same_xw<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_zw_t           = ::fftm::detail::mpi_transpose_4d_same_zw<complex, Backend, MPIComm, Log, runtime_api_t>;
    using for_each_3d_t       = typename Backend::template for_each_nd_type<3, int>;
    using for_each_4d_t       = typename Backend::template for_each_nd_type<4, int>;

    void ensure_can_init_() const
    {
        if ( init_done_ )
            throw std::logic_error( "fftm::init can only be called once per instance." );
    }

    void ensure_initialized_() const
    {
        if ( !init_done_ )
            throw std::logic_error( "fftm: call init() before using the transform." );
    }

    void ensure_dimension_( std::size_t expected_dim ) const
    {
        ensure_initialized_();
        if ( dim_ != expected_dim )
        {
            throw std::logic_error(
                "fftm: initialized for " + std::to_string( dim_ ) + "D but used as " + std::to_string( expected_dim ) + "D."
            );
        }
    }

    void require_grid_p1_is_one_() const
    {
        if ( partitioning_.get_process_grid().p1 != 1 )
            throw std::logic_error( "This strategy currently requires p1 == 1." );
    }

    void require_grid_p2_is_one_() const
    {
        if ( partitioning_.get_process_grid().p2 != 1 )
            throw std::logic_error( "This strategy currently requires p2 == 1." );
    }

    void require_grid_p3_is_one_() const
    {
        if ( partitioning_.get_process_grid().p3 != 1 )
            throw std::logic_error( "This 4D strategy currently requires p3 == 1." );
    }

    void init_( std::integral_constant<std::size_t, 3>, const processor_grid &pg, const global_sizes &gs, const fftm_init_options &options )
    {
        (void)options;
        ensure_can_init_();
        if ( gs.is_4D() )
            throw std::logic_error( "fftm::init<3> received 4D sizes." );

        partitioning_.init( pg, gs );
        std::tie( myid_i_, myid_j_, myid_k_ ) = partitioning_.get_my_grid();
        std::tie( input_dim_, transpose1_dim_, transpose2_dim_ ) = partitioning_.get_partitioning_3D();

        nx_      = gs.Nx;
        ny_      = gs.Ny;
        nz_      = gs.Nz;
        nw_      = 0;
        nz_half_ = nz_ / 2 + 1;
        nw_half_ = 0;
        dim_     = 3;

        half_input_dim_           = input_dim_;
        half_input_dim_.size_z[0] = nz_half_;
        half_input_dim_.compute_offsets( false );

        init_strategy_( strategy_family_3d_tag() );
        add_plans_( strategy_family_3d_tag() );
        base_fft_.activate();
        init_done_ = true;
    }

    void init_( std::integral_constant<std::size_t, 4>, const processor_grid &pg, const global_sizes &gs, const fftm_init_options &options )
    {
        (void)options;
        ensure_can_init_();
        if ( !gs.is_4D() )
            throw std::logic_error( "fftm::init<4> requires 4D global sizes." );

        partitioning_.init( pg, gs );
        std::tie( myid_i_, myid_j_, myid_k_ ) = partitioning_.get_my_grid();
        std::tie( input_dim_, transpose1_dim_, transpose2_dim_, transpose3_dim_ ) = partitioning_.get_partitioning_4D();

        nx_      = gs.Nx;
        ny_      = gs.Ny;
        nz_      = gs.Nz;
        nw_      = gs.Nw;
        nz_half_ = 0;
        nw_half_ = nw_ / 2 + 1;
        dim_     = 4;

        half_input_dim_           = input_dim_;
        half_input_dim_.size_w[0] = nw_half_;
        half_input_dim_.compute_offsets( true );

        init_4d_strategy_( strategy_family_4d_tag() );
        add_4d_plans_( strategy_family_4d_tag() );
        base_fft_.activate();
        init_done_ = true;
    }

    void add_plan_z_r2c_( const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size )
    {
        const long long int stride = x_size * y_size;
        const long long int batch  = x_size * y_size;

        base_fft_.template add_plan_1D<::fftm::direction::R2C>(
            forward_name,
            static_cast<long long int>( nz_ ),
            1,
            stride,
            1,
            1,
            x_size * y_size,
            1,
            batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2R>(
            inverse_name,
            static_cast<long long int>( nz_ ),
            1,
            stride,
            1,
            1,
            x_size * y_size,
            1,
            batch
        );
    }

    void add_plan_y_c2c_( const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int z_size )
    {
        const long long int batch = x_size * z_size;
        const long long int idist = static_cast<long long int>( ny_ );

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name,
            static_cast<long long int>( ny_ ),
            1,
            1,
            idist,
            1,
            1,
            idist,
            batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name,
            static_cast<long long int>( ny_ ),
            1,
            1,
            idist,
            1,
            1,
            idist,
            batch
        );
    }

    void add_plan_x_c2c_( const std::string &forward_name, const std::string &inverse_name, long long int y_size, long long int z_size )
    {
        const long long int batch  = y_size * z_size;
        const long long int stride = y_size * z_size;

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name,
            static_cast<long long int>( nx_ ),
            1,
            stride,
            1,
            1,
            stride,
            1,
            batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name,
            static_cast<long long int>( nx_ ),
            1,
            stride,
            1,
            1,
            stride,
            1,
            batch
        );
    }

    void add_plan_w_r2c_( const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size, long long int z_size )
    {
        const long long int stride = x_size * y_size * z_size;
        const long long int batch  = x_size * y_size * z_size;

        base_fft_.template add_plan_1D<::fftm::direction::R2C>(
            forward_name,
            static_cast<long long int>( nw_ ),
            1,
            stride,
            1,
            1,
            stride,
            1,
            batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2R>(
            inverse_name,
            static_cast<long long int>( nw_ ),
            1,
            stride,
            1,
            1,
            stride,
            1,
            batch
        );
    }

    void add_plan_z_c2c_4d_( const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size, long long int w_size )
    {
        const long long int stride = x_size * y_size * w_size;
        const long long int batch  = x_size * y_size * w_size;

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name,
            static_cast<long long int>( nz_ ),
            1,
            stride,
            1,
            1,
            stride,
            1,
            batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name,
            static_cast<long long int>( nz_ ),
            1,
            stride,
            1,
            1,
            stride,
            1,
            batch
        );
    }

    void add_plan_y_c2c_4d_( const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int z_size, long long int w_size )
    {
        const long long int stride = x_size * z_size * w_size;
        const long long int batch  = x_size * z_size * w_size;

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name,
            static_cast<long long int>( ny_ ),
            1,
            stride,
            1,
            1,
            stride,
            1,
            batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name,
            static_cast<long long int>( ny_ ),
            1,
            stride,
            1,
            1,
            stride,
            1,
            batch
        );
    }

    void add_plan_x_c2c_4d_( const std::string &forward_name, const std::string &inverse_name, long long int y_size, long long int z_size, long long int w_size )
    {
        const long long int batch = y_size * z_size * w_size;

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name,
            static_cast<long long int>( nx_ ),
            1,
            1,
            static_cast<long long int>( nx_ ),
            1,
            1,
            static_cast<long long int>( nx_ ),
            batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name,
            static_cast<long long int>( nx_ ),
            1,
            1,
            static_cast<long long int>( nx_ ),
            1,
            1,
            static_cast<long long int>( nx_ ),
            batch
        );
    }

    void add_plan_zw_r2c_( const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size )
    {
        const long long int batch = x_size * y_size;

        base_fft_.template add_plan_2D<::fftm::direction::R2C>(
            forward_name,
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
            batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2R>(
            inverse_name,
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
            batch
        );
    }

    void add_plan_xy_c2c_4d_( const std::string &forward_name, const std::string &inverse_name, long long int z_size, long long int w_size )
    {
        const long long int batch    = z_size * w_size;
        const long long int xy_stride = z_size * w_size;

        base_fft_.template add_plan_2D<::fftm::direction::C2CF>(
            forward_name,
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
            batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2CB>(
            inverse_name,
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
            batch
        );
    }

    void init_strategy_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil> )
    {
        require_grid_p2_is_one_();

        output_dim_ = transpose2_dim_;

        stage0_3d_.init( input_dim_.size_x[myid_i_], nz_half_, ny_ );
        work_hat_3d_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
        x_fft_stage_3d_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
        same_z_.init( half_input_dim_, output_dim_, myid_i_, 0 );
    }

    void init_strategy_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab> )
    {
        require_grid_p1_is_one_();

        output_dim_ = transpose2_dim_;

        stage0_3d_.init( input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], nz_half_ );
        work_hat_3d_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
        x_fft_stage_3d_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
        same_x_.init( half_input_dim_, transpose1_dim_, myid_i_, myid_j_ );
    }

    void init_strategy_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil> )
    {
        output_dim_ = transpose2_dim_;

        stage0_3d_.init( input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], nz_half_ );
        stage1_3d_.init( input_dim_.size_x[myid_i_], transpose1_dim_.size_z[myid_j_], ny_ );
        work_hat_3d_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
        x_fft_stage_3d_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );

        same_x_.init( half_input_dim_, transpose1_dim_, myid_i_, myid_j_ );
        same_z_.init( transpose1_dim_, output_dim_, myid_i_, myid_j_ );
    }

    void init_4d_strategy_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil> )
    {
        output_dim_ = transpose3_dim_;

        stage0_4d_.init(
            input_dim_.size_x[myid_i_],
            input_dim_.size_y[myid_j_],
            input_dim_.size_z[myid_k_],
            nw_half_
        );
        stage1_4d_.init(
            transpose1_dim_.size_x[myid_i_],
            transpose1_dim_.size_y[myid_j_],
            transpose1_dim_.size_w[myid_k_],
            transpose1_dim_.size_z[0]
        );
        stage2_4d_.init(
            transpose2_dim_.size_x[myid_i_],
            transpose2_dim_.size_z[myid_j_],
            transpose2_dim_.size_w[myid_k_],
            transpose2_dim_.size_y[0]
        );
        work_hat_4d_.init(
            output_dim_.size_y[myid_i_],
            output_dim_.size_z[myid_j_],
            output_dim_.size_w[myid_k_],
            output_dim_.size_x[0]
        );

        same_xy_.init( half_input_dim_, transpose1_dim_, myid_i_, myid_j_, myid_k_ );
        same_xw_.init( transpose1_dim_, transpose2_dim_, myid_i_, myid_j_, myid_k_ );
        same_zw_.init( transpose2_dim_, output_dim_, myid_i_, myid_j_, myid_k_ );
    }

    void init_4d_strategy_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab> )
    {
        require_grid_p1_is_one_();
        require_grid_p3_is_one_();

        output_dim_ = transpose3_dim_;

        stage0_4d_.init(
            input_dim_.size_x[myid_i_],
            input_dim_.size_y[myid_j_],
            input_dim_.size_z[myid_k_],
            nw_half_
        );
        stage1_4d_.init(
            transpose1_dim_.size_x[myid_i_],
            transpose1_dim_.size_y[myid_j_],
            transpose1_dim_.size_w[myid_k_],
            transpose1_dim_.size_z[0]
        );
        stage2_4d_.init(
            transpose2_dim_.size_x[myid_i_],
            transpose2_dim_.size_z[myid_j_],
            transpose2_dim_.size_w[myid_k_],
            transpose2_dim_.size_y[0]
        );
        work_hat_4d_.init(
            output_dim_.size_y[myid_i_],
            output_dim_.size_z[myid_j_],
            output_dim_.size_w[myid_k_],
            output_dim_.size_x[0]
        );

        same_xw_.init( transpose1_dim_, transpose2_dim_, myid_i_, myid_j_, myid_k_ );
    }

    void add_plans_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil> )
    {
        add_plan_z_r2c_( "forward_z", "inverse_z", input_dim_.size_x[myid_i_], ny_ );
        add_plan_y_c2c_( "forward_y", "inverse_y", input_dim_.size_x[myid_i_], nz_half_ );
        add_plan_x_c2c_( "forward_x", "inverse_x", output_dim_.size_y[myid_i_], nz_half_ );
    }

    void add_plans_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab> )
    {
        add_plan_z_r2c_( "forward_z", "inverse_z", nx_, input_dim_.size_y[myid_j_] );
        add_plan_y_c2c_( "forward_y", "inverse_y", nx_, output_dim_.size_z[myid_j_] );
        add_plan_x_c2c_( "forward_x", "inverse_x", ny_, output_dim_.size_z[myid_j_] );
    }

    void add_plans_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil> )
    {
        add_plan_z_r2c_( "forward_z", "inverse_z", input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_] );
        add_plan_y_c2c_( "forward_y", "inverse_y", input_dim_.size_x[myid_i_], transpose1_dim_.size_z[myid_j_] );
        add_plan_x_c2c_( "forward_x", "inverse_x", output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_] );
    }

    void add_4d_plans_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil> )
    {
        add_plan_w_r2c_(
            "forward_w",
            "inverse_w",
            input_dim_.size_x[myid_i_],
            input_dim_.size_y[myid_j_],
            input_dim_.size_z[myid_k_]
        );
        add_plan_z_c2c_4d_(
            "forward_z",
            "inverse_z",
            transpose1_dim_.size_x[myid_i_],
            transpose1_dim_.size_y[myid_j_],
            transpose1_dim_.size_w[myid_k_]
        );
        add_plan_y_c2c_4d_(
            "forward_y",
            "inverse_y",
            transpose2_dim_.size_x[myid_i_],
            transpose2_dim_.size_z[myid_j_],
            transpose2_dim_.size_w[myid_k_]
        );
        add_plan_x_c2c_4d_(
            "forward_x",
            "inverse_x",
            output_dim_.size_y[myid_i_],
            output_dim_.size_z[myid_j_],
            output_dim_.size_w[myid_k_]
        );
    }

    void add_4d_plans_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab> )
    {
        add_plan_zw_r2c_(
            "forward_zw",
            "inverse_zw",
            input_dim_.size_x[myid_i_],
            input_dim_.size_y[myid_j_]
        );
        add_plan_xy_c2c_4d_(
            "forward_xy",
            "inverse_xy",
            output_dim_.size_z[myid_j_],
            output_dim_.size_w[myid_k_]
        );
    }

    void forward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil>, const real_array3_t &in, complex_array3_t &out )
    {
        base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_z", in, stage0_3d_ );
        base_fft_.template exec<stage0_complex3_t, stage0_complex3_t>( "forward_y", stage0_3d_, stage0_3d_ );
        same_z_.transpose_x_to_y( stage0_3d_, work_hat_3d_, transpose_mode_3d );
        reorder_x_stage_( work_hat_3d_, x_fft_stage_3d_ );
        base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "forward_x", x_fft_stage_3d_, x_fft_stage_3d_ );
        reorder_x_stage_( x_fft_stage_3d_, out );
    }

    void backward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil>, const complex_array3_t &in, real_array3_t &out )
    {
        copy_device_buffer_( in, work_hat_3d_ );
        reorder_x_stage_( work_hat_3d_, x_fft_stage_3d_ );
        base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "inverse_x", x_fft_stage_3d_, x_fft_stage_3d_ );
        reorder_x_stage_( x_fft_stage_3d_, work_hat_3d_ );
        same_z_.transpose_y_to_x( work_hat_3d_, stage0_3d_, transpose_mode_3d );
        base_fft_.template exec<stage0_complex3_t, stage0_complex3_t>( "inverse_y", stage0_3d_, stage0_3d_ );
        base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out );
    }

    void forward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab>, const real_array3_t &in, complex_array3_t &out )
    {
        base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_z", in, stage0_3d_ );
        same_x_.transpose_xyz_to_xzy( stage0_3d_, work_hat_3d_, transpose_mode_3d );
        base_fft_.template exec<complex_array3_t, complex_array3_t>( "forward_y", work_hat_3d_, work_hat_3d_ );
        reorder_x_stage_( work_hat_3d_, x_fft_stage_3d_ );
        base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "forward_x", x_fft_stage_3d_, x_fft_stage_3d_ );
        reorder_x_stage_( x_fft_stage_3d_, out );
    }

    void backward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab>, const complex_array3_t &in, real_array3_t &out )
    {
        copy_device_buffer_( in, work_hat_3d_ );
        reorder_x_stage_( work_hat_3d_, x_fft_stage_3d_ );
        base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "inverse_x", x_fft_stage_3d_, x_fft_stage_3d_ );
        reorder_x_stage_( x_fft_stage_3d_, work_hat_3d_ );
        base_fft_.template exec<complex_array3_t, complex_array3_t>( "inverse_y", work_hat_3d_, work_hat_3d_ );
        same_x_.transpose_xzy_to_xyz( work_hat_3d_, stage0_3d_, transpose_mode_3d );
        base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out );
    }

    void forward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil>, const real_array3_t &in, complex_array3_t &out )
    {
        base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_z", in, stage0_3d_ );
        same_x_.transpose_xyz_to_xzy( stage0_3d_, stage1_3d_, transpose_mode_3d );
        base_fft_.template exec<stage1_complex3_t, stage1_complex3_t>( "forward_y", stage1_3d_, stage1_3d_ );
        same_z_.transpose_x_to_y( stage1_3d_, work_hat_3d_, transpose_mode_3d );
        reorder_x_stage_( work_hat_3d_, x_fft_stage_3d_ );
        base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "forward_x", x_fft_stage_3d_, x_fft_stage_3d_ );
        reorder_x_stage_( x_fft_stage_3d_, out );
    }

    void backward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil>, const complex_array3_t &in, real_array3_t &out )
    {
        copy_device_buffer_( in, work_hat_3d_ );
        reorder_x_stage_( work_hat_3d_, x_fft_stage_3d_ );
        base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "inverse_x", x_fft_stage_3d_, x_fft_stage_3d_ );
        reorder_x_stage_( x_fft_stage_3d_, work_hat_3d_ );
        same_z_.transpose_y_to_x( work_hat_3d_, stage1_3d_, transpose_mode_3d );
        base_fft_.template exec<stage1_complex3_t, stage1_complex3_t>( "inverse_y", stage1_3d_, stage1_3d_ );
        same_x_.transpose_xzy_to_xyz( stage1_3d_, stage0_3d_, transpose_mode_3d );
        base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out );
    }

    void forward_4d_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil>, const real_array4_t &in, complex_array4_t &out )
    {
        base_fft_.template exec<real_array4_t, stage0_complex4_t>( "forward_w", in, stage0_4d_ );
        same_xy_.transpose_xyzw_to_xywz( stage0_4d_, stage1_4d_, transpose_mode_4d );
        base_fft_.template exec<stage1_complex4_t, stage1_complex4_t>( "forward_z", stage1_4d_, stage1_4d_ );
        same_xw_.transpose_xywz_to_xzwy( stage1_4d_, stage2_4d_, transpose_mode_4d );
        base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>( "forward_y", stage2_4d_, stage2_4d_ );
        same_zw_.transpose_xzwy_to_yzwx( stage2_4d_, out, transpose_mode_4d );
        base_fft_.template exec<complex_array4_t, complex_array4_t>( "forward_x", out, out );
    }

    void backward_4d_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil>, const complex_array4_t &in, real_array4_t &out )
    {
        copy_device_buffer_( in, work_hat_4d_ );
        base_fft_.template exec<complex_array4_t, complex_array4_t>( "inverse_x", work_hat_4d_, work_hat_4d_ );
        same_zw_.transpose_yzwx_to_xzwy( work_hat_4d_, stage2_4d_, transpose_mode_4d );
        base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>( "inverse_y", stage2_4d_, stage2_4d_ );
        same_xw_.transpose_xzwy_to_xywz( stage2_4d_, stage1_4d_, transpose_mode_4d );
        base_fft_.template exec<stage1_complex4_t, stage1_complex4_t>( "inverse_z", stage1_4d_, stage1_4d_ );
        same_xy_.transpose_xywz_to_xyzw( stage1_4d_, stage0_4d_, transpose_mode_4d );
        base_fft_.template exec<stage0_complex4_t, real_array4_t>( "inverse_w", stage0_4d_, out );
    }

    void forward_4d_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab>, const real_array4_t &in, complex_array4_t &out )
    {
        base_fft_.template exec<real_array4_t, stage0_complex4_t>( "forward_zw", in, stage0_4d_ );
        transpose_local_4d_<0, 1, 3, 2>( stage0_4d_, stage1_4d_ );
        same_xw_.transpose_xywz_to_xzwy( stage1_4d_, stage2_4d_, transpose_mode_4d );
        transpose_local_4d_<3, 1, 2, 0>( stage2_4d_, out );
        base_fft_.template exec<complex_array4_t, complex_array4_t>( "forward_xy", out, out );
    }

    void backward_4d_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab>, const complex_array4_t &in, real_array4_t &out )
    {
        copy_device_buffer_( in, work_hat_4d_ );
        base_fft_.template exec<complex_array4_t, complex_array4_t>( "inverse_xy", work_hat_4d_, work_hat_4d_ );
        transpose_local_4d_<3, 1, 2, 0>( work_hat_4d_, stage2_4d_ );
        same_xw_.transpose_xzwy_to_xywz( stage2_4d_, stage1_4d_, transpose_mode_4d );
        transpose_local_4d_<0, 1, 3, 2>( stage1_4d_, stage0_4d_ );
        base_fft_.template exec<stage0_complex4_t, real_array4_t>( "inverse_zw", stage0_4d_, out );
    }

    template <class Array>
    rect_3d_t make_range_( const Array &array ) const
    {
        const auto sz = array.size_nd();
        return rect_3d_t(
            idx_3d_t( 0, 0, 0 ),
            idx_3d_t( static_cast<int>( sz[0] ), static_cast<int>( sz[1] ), static_cast<int>( sz[2] ) )
        );
    }

    template <class Array>
    rect_4d_t make_range_4d_( const Array &array ) const
    {
        const auto sz = array.size_nd();
        return rect_4d_t(
            idx_4d_t( 0, 0, 0, 0 ),
            idx_4d_t(
                static_cast<int>( sz[0] ),
                static_cast<int>( sz[1] ),
                static_cast<int>( sz[2] ),
                static_cast<int>( sz[3] )
            )
        );
    }

    template <class ArrayIn, class ArrayOut>
    void reorder_x_stage_( const ArrayIn &in, ArrayOut &out )
    {
        for_each_3d_(
            detail::fftm_copy_same_indices_functor<ArrayIn, ArrayOut, idx_3d_t>{ in, out },
            make_range_( out )
        );
        for_each_3d_.wait();
    }

    template <int DstAxis0, int DstAxis1, int DstAxis2, int DstAxis3, class ArrayIn, class ArrayOut>
    void transpose_local_4d_( const ArrayIn &in, ArrayOut &out )
    {
        const auto in_size  = in.size_nd();
        const auto out_size = out.size_nd();
        const std::size_t expected[4] = {
            static_cast<std::size_t>( in_size[DstAxis0] ),
            static_cast<std::size_t>( in_size[DstAxis1] ),
            static_cast<std::size_t>( in_size[DstAxis2] ),
            static_cast<std::size_t>( in_size[DstAxis3] )
        };

        if ( static_cast<std::size_t>( out_size[0] ) != expected[0] ||
             static_cast<std::size_t>( out_size[1] ) != expected[1] ||
             static_cast<std::size_t>( out_size[2] ) != expected[2] ||
             static_cast<std::size_t>( out_size[3] ) != expected[3] )
        {
            throw std::logic_error( "fftm local 4D transpose destination shape mismatch" );
        }

        for_each_4d_(
            detail::direct_transpose_4d_functor<idx_4d_t, ArrayIn, ArrayOut, DstAxis0, DstAxis1, DstAxis2, DstAxis3>( in, out ),
            make_range_4d_( in )
        );
        for_each_4d_.wait();
    }

    template <class ArrayIn, class ArrayOut>
    void copy_device_buffer_( const ArrayIn &in, ArrayOut &out )
    {
        if ( in.size() != out.size() )
            throw std::logic_error( "fftm device copy requires equal total sizes" );

        runtime_api_t::memcpy(
            out.raw_ptr(),
            in.raw_ptr(),
            in.size() * sizeof( typename ArrayIn::value_type ),
            runtime_api_t::device_to_device_kind()
        );
    }

private:
    BaseFFT          base_fft_;
    MPIComm          mpi_;
    Log              log_;
    partitioning_t   partitioning_;
    same_x_t         same_x_;
    same_z_t         same_z_;
    same_xy_t        same_xy_;
    same_xw_t        same_xw_;
    same_zw_t        same_zw_;

    bool             init_done_ = false;
    std::size_t      dim_       = 0;
    int              myid_i_    = 0;
    int              myid_j_    = 0;
    int              myid_k_    = 0;

    std::size_t      nx_        = 0;
    std::size_t      ny_        = 0;
    std::size_t      nz_        = 0;
    std::size_t      nw_        = 0;
    std::size_t      nz_half_   = 0;
    std::size_t      nw_half_   = 0;

    partition_t      input_dim_;
    partition_t      half_input_dim_;
    partition_t      transpose1_dim_;
    partition_t      transpose2_dim_;
    partition_t      transpose3_dim_;
    partition_t      output_dim_;

    stage0_complex3_t stage0_3d_;
    stage1_complex3_t stage1_3d_;
    complex_array3_t  work_hat_3d_;
    x_fft_complex3_t  x_fft_stage_3d_;
    for_each_3d_t     for_each_3d_;

    stage0_complex4_t stage0_4d_;
    stage1_complex4_t stage1_4d_;
    stage2_complex4_t stage2_4d_;
    complex_array4_t  work_hat_4d_;
    for_each_4d_t     for_each_4d_;
};

} // namespace fftm

#endif // __FFTM_FFTM_HPP__
