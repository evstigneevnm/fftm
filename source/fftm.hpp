#ifndef __FFTM_FFTM_HPP__
#define __FFTM_FFTM_HPP__

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_mpi.h>

#include "detail/array_arrangers.h"
#include "detail/mpi_transpose_3d.h"
#include "detail/mpi_transpose_3d_same_z.h"
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
    class Log        = scfd::utils::log_mpi
>
class fftm
{
public:
    using real       = typename BaseFFT::real;
    using complex    = typename BaseFFT::complex;
    using memory_t   = typename Backend::memory_type;
    using partition_t = ::fftm::partition;
    using strategy_3d_t = Strategy3D;

    template <std::size_t Dim>
    using real_array_t = typename std::conditional<
        Dim == 3,
        typename detail::fftm_3d_array_traits<real, complex, memory_t, Strategy3D>::real_array_t,
        void
    >::type;

    template <std::size_t Dim>
    using complex_array_t = typename std::conditional<
        Dim == 3,
        typename detail::fftm_3d_array_traits<real, complex, memory_t, Strategy3D>::complex_array_t,
        void
    >::type;

    static constexpr transform_strategy_3d strategy_family_3d = detail::fftm_3d_strategy_traits<Strategy3D>::family;
    static constexpr mpi_transpose_3d_mode transpose_mode_3d  = detail::fftm_3d_strategy_traits<Strategy3D>::mode;

    static const char *strategy_name()
    {
        return detail::fftm_3d_strategy_traits<Strategy3D>::name();
    }

    explicit fftm( const MPIComm &mpi, const Log &log = Log() )
        : mpi_( mpi )
        , log_( log )
        , partitioning_( mpi )
        , same_x_( mpi, log )
        , same_z_( mpi, log )
    {
        for_each_3d_.block_size = 128;
    }

    template <std::size_t Dim>
    void init( const processor_grid &pg, const global_sizes &gs, const fftm_init_options &options = fftm_init_options() )
    {
        static_assert( Dim == 3, "fftm currently implements only the 3D path." );
        (void)options;
        ensure_can_init_();
        if ( gs.is_4D() )
        {
            throw std::logic_error( "fftm currently implements only the 3D path." );
        }

        partitioning_.init( pg, gs );
        std::tie( myid_i_, myid_j_, myid_k_ ) = partitioning_.get_my_grid();
        std::tie( input_dim_, transpose1_dim_, transpose2_dim_ ) = partitioning_.get_partitioning_3D();

        nx_      = gs.Nx;
        ny_      = gs.Ny;
        nz_      = gs.Nz;
        nz_half_ = nz_ / 2 + 1;

        half_input_dim_           = input_dim_;
        half_input_dim_.size_z[0] = nz_half_;
        half_input_dim_.start_z.clear();
        half_input_dim_.compute_offsets( false );

        init_strategy_( strategy_family_tag() );
        add_plans_( strategy_family_tag() );
        base_fft_.activate();
        init_done_ = true;
    }

    bool is_initialized() const
    {
        return init_done_;
    }

    std::tuple<std::size_t, std::size_t, std::size_t> get_local_input_sizes() const
    {
        ensure_initialized_();
        return std::make_tuple(
            input_dim_.size_x[myid_i_],
            input_dim_.size_y[myid_j_],
            input_dim_.size_z[0]
        );
    }

    std::tuple<std::size_t, std::size_t, std::size_t> get_local_output_sizes() const
    {
        ensure_initialized_();
        return std::make_tuple(
            output_dim_.size_x[0],
            output_dim_.size_z[myid_j_],
            output_dim_.size_y[myid_i_]
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

    template <class ArrayIn, class ArrayOut>
    void forward( const ArrayIn &in, ArrayOut &out )
    {
        ensure_initialized_();
        forward_3d_( strategy_family_tag(), in, out );
    }

    template <class ArrayIn, class ArrayOut>
    void backward( const ArrayIn &in, ArrayOut &out )
    {
        ensure_initialized_();
        backward_3d_( strategy_family_tag(), in, out );
    }

private:
    using strategy_family_tag = std::integral_constant<transform_strategy_3d, strategy_family_3d>;
    using traits_3d_t         = detail::fftm_3d_array_traits<real, complex, memory_t, Strategy3D>;
    using idx_3d_t            = scfd::static_vec::vec<int, 3>;
    using rect_3d_t           = scfd::static_vec::rect<int, 3>;

    using real_array3_t       = typename traits_3d_t::real_array_t;
    using stage0_complex_t    = typename traits_3d_t::stage0_complex_array_t;
    using complex_array3_t    = typename traits_3d_t::complex_array_t;
    using x_fft_complex_t     = typename traits_3d_t::x_fft_complex_array_t;
    using stage1_complex_t    = typename std::conditional<
        strategy_family_3d == transform_strategy_3d::pencil_pencil,
        typename traits_3d_t::stage1_complex_array_t,
        complex_array3_t
    >::type;

    using partitioning_t      = fft_partitioning<MPIComm>;
    using same_x_t            = ::fftm::mpi_transpose_3d<complex, Backend, MPIComm, Log>;
    using same_z_t            = ::fftm::mpi_transpose_3d_same_z<complex, Backend, MPIComm, Log>;
    using for_each_3d_t       = typename Backend::template for_each_nd_type<3, int>;

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

    void require_grid_p1_is_one_() const
    {
        if ( partitioning_.get_process_grid().p1 != 1 )
        {
            throw std::logic_error( "This 3D strategy currently requires p1 == 1." );
        }
    }

    void require_grid_p2_is_one_() const
    {
        if ( partitioning_.get_process_grid().p2 != 1 )
        {
            throw std::logic_error( "This 3D strategy currently requires p2 == 1." );
        }
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
        const long long int batch = y_size * z_size;
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

    void init_strategy_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil> )
    {
        require_grid_p2_is_one_();

        output_dim_ = transpose2_dim_;

        stage0_.init( input_dim_.size_x[myid_i_], nz_half_, ny_ );
        work_hat_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
        x_fft_stage_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
        same_z_.init( half_input_dim_, output_dim_, myid_i_, 0 );
    }

    void init_strategy_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab> )
    {
        require_grid_p1_is_one_();

        output_dim_ = transpose2_dim_;

        stage0_.init( input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], nz_half_ );
        work_hat_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
        x_fft_stage_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
        same_x_.init( half_input_dim_, transpose1_dim_, myid_i_, myid_j_ );
    }

    void init_strategy_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil> )
    {
        output_dim_ = transpose2_dim_;

        stage0_.init( input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], nz_half_ );
        stage1_.init( input_dim_.size_x[myid_i_], transpose1_dim_.size_z[myid_j_], ny_ );
        work_hat_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
        x_fft_stage_.init( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );

        same_x_.init( half_input_dim_, transpose1_dim_, myid_i_, myid_j_ );
        same_z_.init( transpose1_dim_, output_dim_, myid_i_, myid_j_ );
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

    template <class ArrayIn, class ArrayOut>
    void forward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil>, const ArrayIn &in, ArrayOut &out )
    {
        base_fft_.template exec<ArrayIn, stage0_complex_t>( "forward_z", in, stage0_ );
        base_fft_.template exec<stage0_complex_t, stage0_complex_t>( "forward_y", stage0_, stage0_ );
        same_z_.transpose_x_to_y( stage0_, work_hat_, transpose_mode_3d );
        reorder_x_stage_( work_hat_, x_fft_stage_ );
        base_fft_.template exec<x_fft_complex_t, x_fft_complex_t>( "forward_x", x_fft_stage_, x_fft_stage_ );
        reorder_x_stage_( x_fft_stage_, out );
    }

    template <class ArrayIn, class ArrayOut>
    void backward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil>, const ArrayIn &in, ArrayOut &out )
    {
        CUDA_SAFE_CALL( cudaMemcpy(
            work_hat_.raw_ptr(),
            in.raw_ptr(),
            in.size() * sizeof( complex ),
            cudaMemcpyDeviceToDevice
        ) );
        reorder_x_stage_( work_hat_, x_fft_stage_ );
        base_fft_.template exec<x_fft_complex_t, x_fft_complex_t>( "inverse_x", x_fft_stage_, x_fft_stage_ );
        reorder_x_stage_( x_fft_stage_, work_hat_ );
        same_z_.transpose_y_to_x( work_hat_, stage0_, transpose_mode_3d );
        base_fft_.template exec<stage0_complex_t, stage0_complex_t>( "inverse_y", stage0_, stage0_ );
        base_fft_.template exec<stage0_complex_t, ArrayOut>( "inverse_z", stage0_, out );
    }

    template <class ArrayIn, class ArrayOut>
    void forward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab>, const ArrayIn &in, ArrayOut &out )
    {
        base_fft_.template exec<ArrayIn, stage0_complex_t>( "forward_z", in, stage0_ );
        same_x_.transpose_xyz_to_xzy( stage0_, work_hat_, transpose_mode_3d );
        base_fft_.template exec<complex_array3_t, complex_array3_t>( "forward_y", work_hat_, work_hat_ );
        reorder_x_stage_( work_hat_, x_fft_stage_ );
        base_fft_.template exec<x_fft_complex_t, x_fft_complex_t>( "forward_x", x_fft_stage_, x_fft_stage_ );
        reorder_x_stage_( x_fft_stage_, out );
    }

    template <class ArrayIn, class ArrayOut>
    void backward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab>, const ArrayIn &in, ArrayOut &out )
    {
        CUDA_SAFE_CALL( cudaMemcpy(
            work_hat_.raw_ptr(),
            in.raw_ptr(),
            in.size() * sizeof( complex ),
            cudaMemcpyDeviceToDevice
        ) );
        reorder_x_stage_( work_hat_, x_fft_stage_ );
        base_fft_.template exec<x_fft_complex_t, x_fft_complex_t>( "inverse_x", x_fft_stage_, x_fft_stage_ );
        reorder_x_stage_( x_fft_stage_, work_hat_ );
        base_fft_.template exec<complex_array3_t, complex_array3_t>( "inverse_y", work_hat_, work_hat_ );
        same_x_.transpose_xzy_to_xyz( work_hat_, stage0_, transpose_mode_3d );
        base_fft_.template exec<stage0_complex_t, ArrayOut>( "inverse_z", stage0_, out );
    }

    template <class ArrayIn, class ArrayOut>
    void forward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil>, const ArrayIn &in, ArrayOut &out )
    {
        base_fft_.template exec<ArrayIn, stage0_complex_t>( "forward_z", in, stage0_ );
        same_x_.transpose_xyz_to_xzy( stage0_, stage1_, transpose_mode_3d );
        base_fft_.template exec<stage1_complex_t, stage1_complex_t>( "forward_y", stage1_, stage1_ );
        same_z_.transpose_x_to_y( stage1_, work_hat_, transpose_mode_3d );
        reorder_x_stage_( work_hat_, x_fft_stage_ );
        base_fft_.template exec<x_fft_complex_t, x_fft_complex_t>( "forward_x", x_fft_stage_, x_fft_stage_ );
        reorder_x_stage_( x_fft_stage_, out );
    }

    template <class ArrayIn, class ArrayOut>
    void backward_3d_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil>, const ArrayIn &in, ArrayOut &out )
    {
        CUDA_SAFE_CALL( cudaMemcpy(
            work_hat_.raw_ptr(),
            in.raw_ptr(),
            in.size() * sizeof( complex ),
            cudaMemcpyDeviceToDevice
        ) );
        reorder_x_stage_( work_hat_, x_fft_stage_ );
        base_fft_.template exec<x_fft_complex_t, x_fft_complex_t>( "inverse_x", x_fft_stage_, x_fft_stage_ );
        reorder_x_stage_( x_fft_stage_, work_hat_ );
        same_z_.transpose_y_to_x( work_hat_, stage1_, transpose_mode_3d );
        base_fft_.template exec<stage1_complex_t, stage1_complex_t>( "inverse_y", stage1_, stage1_ );
        same_x_.transpose_xzy_to_xyz( stage1_, stage0_, transpose_mode_3d );
        base_fft_.template exec<stage0_complex_t, ArrayOut>( "inverse_z", stage0_, out );
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

    template <class ArrayIn, class ArrayOut>
    void reorder_x_stage_( const ArrayIn &in, ArrayOut &out )
    {
        for_each_3d_(
            detail::fftm_copy_same_indices_functor<ArrayIn, ArrayOut, idx_3d_t>{ in, out },
            make_range_( out )
        );
        for_each_3d_.wait();
    }

private:
    BaseFFT         base_fft_;
    MPIComm         mpi_;
    Log             log_;
    partitioning_t  partitioning_;
    same_x_t        same_x_;
    same_z_t        same_z_;

    bool            init_done_ = false;
    int             myid_i_    = 0;
    int             myid_j_    = 0;
    int             myid_k_    = 0;

    std::size_t     nx_        = 0;
    std::size_t     ny_        = 0;
    std::size_t     nz_        = 0;
    std::size_t     nz_half_   = 0;

    partition_t     input_dim_;
    partition_t     half_input_dim_;
    partition_t     transpose1_dim_;
    partition_t     transpose2_dim_;
    partition_t     output_dim_;

    stage0_complex_t stage0_;
    stage1_complex_t stage1_;
    complex_array3_t work_hat_;
    x_fft_complex_t  x_fft_stage_;
    for_each_3d_t    for_each_3d_;
};

} // namespace fftm

#endif // __FFTM_FFTM_HPP__
