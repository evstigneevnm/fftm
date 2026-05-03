#ifndef __FFTM_FFTM_HPP__
#define __FFTM_FFTM_HPP__

#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/memory/shared_buffer.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/safe_call.h>

#include "detail/array_arrangers.h"
#include "detail/direct_transpose_4d.h"
#include "detail/memory_profile_utils.h"
#include "detail/mpi_transpose_3d.h"
#include "detail/mpi_transpose_4d.h"
#include "fft_direction.h"
#include "fft_partitioning.h"
#include "profiling.h"

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

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv, bool UseOptimized = true>
struct strategy_3d_slab_pencil
{
};

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv, bool UseOptimized = true>
struct strategy_3d_pencil_slab
{
};

template <mpi_transpose_3d_mode Mode = mpi_transpose_3d_mode::alltoallv, bool UseOptimized = true>
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
    std::string profiling_key                    = "fftm_prof";
    std::string memory_profiling_key             = "fftm_mem";
    bool        use_optimized                    = true;
    bool        use_direct_backward_receive      = false;
    bool        direct_p2p_cuda_aware            = true;
    bool        use_p2p_send_thread              = false;
    bool        use_p2p_byte_transfer            = false;
    bool        print_profile_summary_on_destroy = true;
    bool        print_profile_totals_on_destroy  = true;
    bool        print_memory_profile_on_destroy  = true;
    bool        print_memory_totals_on_destroy   = true;
};

namespace detail
{

template <class Strategy3D>
struct fftm_3d_strategy_traits;

template <mpi_transpose_3d_mode Mode, bool UseOptimized>
struct fftm_3d_strategy_traits<strategy_3d_slab_pencil<Mode, UseOptimized>>
{
    static constexpr transform_strategy_3d family = transform_strategy_3d::slab_pencil;
    static constexpr mpi_transpose_3d_mode mode   = Mode;
    static constexpr bool optimized_layout        = UseOptimized;

    static const char *name()
    {
        return "slab-pencil";
    }
};

template <mpi_transpose_3d_mode Mode, bool UseOptimized>
struct fftm_3d_strategy_traits<strategy_3d_pencil_slab<Mode, UseOptimized>>
{
    static constexpr transform_strategy_3d family = transform_strategy_3d::pencil_slab;
    static constexpr mpi_transpose_3d_mode mode   = Mode;
    static constexpr bool optimized_layout        = UseOptimized;

    static const char *name()
    {
        return "pencil-slab";
    }
};

template <mpi_transpose_3d_mode Mode, bool UseOptimized>
struct fftm_3d_strategy_traits<strategy_3d_pencil_pencil<Mode, UseOptimized>>
{
    static constexpr transform_strategy_3d family = transform_strategy_3d::pencil_pencil;
    static constexpr mpi_transpose_3d_mode mode   = Mode;
    static constexpr bool optimized_layout        = UseOptimized;

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
struct fftm_3d_array_traits<Real, Complex, Memory, strategy_3d_slab_pencil<Mode, false>>
{
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage0_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using stage1_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using x_fft_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
};

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_3d_array_traits<Real, Complex, Memory, strategy_3d_slab_pencil<Mode, true>>
{
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_210_t>;
    using stage0_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
    using stage1_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
    using x_fft_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
};

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode, bool UseOptimized>
struct fftm_3d_array_traits<Real, Complex, Memory, strategy_3d_pencil_slab<Mode, UseOptimized>>
{
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage0_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage1_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using x_fft_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
};

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_3d_array_traits<Real, Complex, Memory, strategy_3d_pencil_pencil<Mode, false>>
{
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage0_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage1_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using x_fft_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
};

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_3d_array_traits<Real, Complex, Memory, strategy_3d_pencil_pencil<Mode, true>>
{
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    using stage0_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    // The local Y FFT needs Y-fast storage. A temporary X-fast view is bound
    // separately for the second redistribution and final X FFT.
    using stage1_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
    using x_fft_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_120_t>;
};

template <class Real, class Complex, class Memory, class Strategy4D>
struct fftm_4d_array_traits;

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_4d_array_traits<Real, Complex, Memory, strategy_4d_pencil_pencil_mpi<Mode>>
{
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using stage0_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using stage1_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using stage2_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0213_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
};

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_4d_array_traits<Real, Complex, Memory, strategy_4d_slab_slab_mpi<Mode>>
{
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using stage0_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using stage1_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using stage2_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0213_t>;
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_2103_t>;
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
    class BaseFFT, class MPIComm, class Backend,
    class Strategy3D = strategy_3d_slab_pencil<mpi_transpose_3d_mode::alltoallv>, class Log = scfd::utils::log_mpi,
    class Strategy4D = strategy_4d_pencil_pencil_mpi<mpi_transpose_3d_mode::alltoallv>>
class fftm
{
public:
    using backend_type  = Backend;
    using base_fft_type = BaseFFT;
    using mpi_comm_type = MPIComm;
    using real          = typename BaseFFT::real;
    using complex       = typename BaseFFT::complex;
    using runtime_api_t = typename BaseFFT::runtime_api;
    using memory_t      = typename Backend::memory_type;
    using partition_t   = ::fftm::partition;
    using strategy_3d_t = Strategy3D;
    using strategy_4d_t = Strategy4D;
    using memory_profiler_t        = fftm_memory_profiler;
    using memory_profile_bytes_t   = typename memory_profiler_t::bytes_type;
    using memory_profile_buckets_t = detail::memory_profile_buckets<memory_profile_bytes_t>;

    template <std::size_t Dim>
    using real_array_t = typename std::conditional<
        Dim == 3, typename detail::fftm_3d_array_traits<real, complex, memory_t, Strategy3D>::real_array_t,
        typename std::conditional<
            Dim == 4, typename detail::fftm_4d_array_traits<real, complex, memory_t, Strategy4D>::real_array_t,
            void>::type>::type;

    template <std::size_t Dim>
    using complex_array_t = typename std::conditional<
        Dim == 3, typename detail::fftm_3d_array_traits<real, complex, memory_t, Strategy3D>::complex_array_t,
        typename std::conditional<
            Dim == 4, typename detail::fftm_4d_array_traits<real, complex, memory_t, Strategy4D>::complex_array_t,
            void>::type>::type;

    static constexpr transform_strategy_3d     strategy_family_3d = detail::fftm_3d_strategy_traits<Strategy3D>::family;
    static constexpr mpi_transpose_3d_mode     transpose_mode_3d  = detail::fftm_3d_strategy_traits<Strategy3D>::mode;
    static constexpr bool strategy_3d_optimized_layout = detail::fftm_3d_strategy_traits<Strategy3D>::optimized_layout;
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
        : mpi_( mpi ), log_( log ), partitioning_( mpi ), same_x_( mpi, log ), same_z_( mpi, log ),
          same_xy_( mpi, log ), same_xw_( mpi, log ), same_zw_( mpi, log )
    {
        for_each_3d_.block_size = 128;
        for_each_4d_.block_size = 128;
        same_x_.use_external_work_area();
        same_z_.use_external_work_area();
        same_xy_.use_external_work_area();
        same_xw_.use_external_work_area();
        same_zw_.use_external_work_area();
        same_x_.use_external_host_work_area();
        same_z_.use_external_host_work_area();
        same_xy_.use_external_host_work_area();
        same_xw_.use_external_host_work_area();
        same_zw_.use_external_host_work_area();
    }

    ~fftm()
    {
        try
        {
            log_profile_on_destroy_();
            log_memory_profile_on_destroy_();
        }
        catch ( ... )
        {
        }
    }

    template <std::size_t Dim>
    void
    init( const processor_grid &pg, const global_sizes &gs, const fftm_init_options &options = fftm_init_options() )
    {
        SCFD_SAFE_CALL( init_( std::integral_constant<std::size_t, Dim>(), pg, gs, options ) );
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
        return std::make_tuple( input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], input_dim_.size_z[0] );
    }

    std::tuple<std::size_t, std::size_t, std::size_t> get_local_output_sizes() const
    {
        ensure_dimension_( 3 );
        return std::make_tuple( output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_] );
    }

    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> get_local_input_sizes_4d() const
    {
        ensure_dimension_( 4 );
        return std::make_tuple(
            input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], input_dim_.size_z[myid_k_], input_dim_.size_w[0]
        );
    }

    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> get_local_output_sizes_4d() const
    {
        ensure_dimension_( 4 );
        return std::make_tuple(
            output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_], output_dim_.size_w[myid_k_], output_dim_.size_x[0]
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
        SCFD_SAFE_CALL( forward_3d_( strategy_family_3d_tag(), in, out ) );
    }

    // Destructive inverse: the spectral input is reused as a work buffer to avoid a full-domain copy.
    void backward( complex_array_t<3> &in, real_array_t<3> &out )
    {
        ensure_dimension_( 3 );
        SCFD_SAFE_CALL( backward_3d_( strategy_family_3d_tag(), in, out ) );
    }

    void forward( const real_array_t<4> &in, complex_array_t<4> &out )
    {
        ensure_dimension_( 4 );
        SCFD_SAFE_CALL( forward_4d_( strategy_family_4d_tag(), in, out ) );
    }

    // Destructive inverse: the spectral input is reused as a work buffer to avoid a full-domain copy.
    void backward( complex_array_t<4> &in, real_array_t<4> &out )
    {
        ensure_dimension_( 4 );
        SCFD_SAFE_CALL( backward_4d_( strategy_family_4d_tag(), in, out ) );
    }

    void log_profile()
    {
        if ( profiler_.enabled() )
        {
            profiler_.log_print( log_ );
        }
    }

    void log_profile_totals()
    {
        if ( profiler_.enabled() )
        {
            profiler_.log_print_totals( log_ );
        }
    }

    void log_memory_profile()
    {
        if ( memory_profiler_.enabled() )
        {
            log_memory_profile_summary_();
        }
    }

    void log_memory_totals()
    {
        if ( memory_profiler_.enabled() )
        {
            log_memory_profile_totals_();
        }
    }

    bool is_memory_profiling_enabled() const
    {
        return memory_profiler_.enabled();
    }

    memory_profile_buckets_t get_memory_profile_buckets() const
    {
        return collect_memory_profile_buckets_();
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

    using real_array3_t     = typename traits_3d_t::real_array_t;
    using stage0_complex3_t = typename traits_3d_t::stage0_complex_array_t;
    using complex_array3_t  = typename traits_3d_t::complex_array_t;
    using x_fft_complex3_t  = typename traits_3d_t::x_fft_complex_array_t;
    using stage1_complex3_t = typename std::conditional<
        strategy_family_3d == transform_strategy_3d::pencil_pencil, typename traits_3d_t::stage1_complex_array_t,
        complex_array3_t>::type;

    using real_array4_t     = typename traits_4d_t::real_array_t;
    using stage0_complex4_t = typename traits_4d_t::stage0_complex_array_t;
    using stage1_complex4_t = typename traits_4d_t::stage1_complex_array_t;
    using stage2_complex4_t = typename traits_4d_t::stage2_complex_array_t;
    using complex_array4_t  = typename traits_4d_t::complex_array_t;

    using partitioning_t   = fft_partitioning<MPIComm>;
    using same_x_t         = ::fftm::mpi_transpose_3d<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_z_t         = ::fftm::mpi_transpose_3d_same_z<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_xy_t        = ::fftm::detail::mpi_transpose_4d_same_xy<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_xw_t        = ::fftm::detail::mpi_transpose_4d_same_xw<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_zw_t        = ::fftm::detail::mpi_transpose_4d_same_zw<complex, Backend, MPIComm, Log, runtime_api_t>;
    using for_each_3d_t    = typename Backend::template for_each_nd_type<3, int>;
    using for_each_4d_t    = typename Backend::template for_each_nd_type<4, int>;
    using complex_buffer_t = scfd::arrays::array_nd<complex, 1, memory_t>;
    using shared_buffer_t  = scfd::memory::shared_buffer<memory_t>;
    using host_shared_buffer_t       = scfd::memory::shared_buffer<typename memory_t::host_memory_type>;
    using profiler_t                 = fftm_profiler;
    using optional_profiler_t        = optional_profiler<profiler_t>;
    using optional_memory_profiler_t = optional_memory_profiler<memory_profiler_t>;

    typename optional_profiler_t::scoped_ticker profile_scope_( const std::string &name )
    {
        return profiler_.scoped_tic( name );
    }

    static typename memory_profiler_t::bytes_type bytes_of_elems_( std::size_t elems, std::size_t elem_size )
    {
        return static_cast<typename memory_profiler_t::bytes_type>( elems ) *
               static_cast<typename memory_profiler_t::bytes_type>( elem_size );
    }

    static std::size_t align_up_( std::size_t value, std::size_t alignment )
    {
        return ( value + alignment - 1 ) / alignment * alignment;
    }

    void *shared_work_ptr_( std::size_t offset = 0 ) const
    {
        return static_cast<void *>( static_cast<char *>( shared_work_buffer_.naive_ptr() ) + offset );
    }

    void *shared_host_work_ptr_( std::size_t offset = 0 ) const
    {
        return static_cast<void *>( static_cast<char *>( shared_host_work_buffer_.naive_ptr() ) + offset );
    }

    void activate_shared_work_area_()
    {
        const std::size_t fft_work_size       = base_fft_.activate_work_size();
        const std::size_t transpose_work_size = std::max(
            std::max( same_x_.get_work_size_bytes(), same_z_.get_work_size_bytes() ),
            std::max(
                same_xy_.get_work_size_bytes(),
                std::max( same_xw_.get_work_size_bytes(), same_zw_.get_work_size_bytes() )
            )
        );
        shared_work_size_ = align_up_( std::max( fft_work_size, transpose_work_size ), 256 );
        shared_work_buffer_.require_size_bytes( shared_work_size_ );
        shared_work_buffer_.activate();

        const std::size_t transpose_host_work_size = std::max(
            std::max( same_x_.get_host_work_size_bytes(), same_z_.get_host_work_size_bytes() ),
            std::max(
                same_xy_.get_host_work_size_bytes(),
                std::max( same_xw_.get_host_work_size_bytes(), same_zw_.get_host_work_size_bytes() )
            )
        );
        shared_host_work_size_ = align_up_( transpose_host_work_size, 256 );
        shared_host_work_buffer_.require_size_bytes( shared_host_work_size_ );
        if ( shared_host_work_size_ != 0 )
        {
            shared_host_work_buffer_.activate();
        }

        void *work_area = shared_work_ptr_();
        base_fft_.set_external_work_area( work_area );
        same_x_.set_external_work_area( work_area );
        same_z_.set_external_work_area( work_area );
        same_xy_.set_external_work_area( work_area );
        same_xw_.set_external_work_area( work_area );
        same_zw_.set_external_work_area( work_area );
        if ( shared_host_work_size_ != 0 )
        {
            void *host_work_area = shared_host_work_ptr_();
            same_x_.set_external_host_work_area( host_work_area );
            same_z_.set_external_host_work_area( host_work_area );
            same_xy_.set_external_host_work_area( host_work_area );
            same_xw_.set_external_host_work_area( host_work_area );
            same_zw_.set_external_host_work_area( host_work_area );
        }

        update_memory_profile_for_current_dim_();
    }

    void configure_profiling_( const fftm_init_options &options )
    {
        init_options_ = options;
        if ( !options.profiling_key.empty() )
        {
            profiler_.enable( options.profiling_key );
        }
        else
        {
            profiler_.disable();
        }
        configure_memory_profiling_( options );
        same_x_.set_profiler( profiler_.native_ptr() );
        same_z_.set_profiler( profiler_.native_ptr() );
        same_xy_.set_profiler( profiler_.native_ptr() );
        same_xw_.set_profiler( profiler_.native_ptr() );
        same_zw_.set_profiler( profiler_.native_ptr() );
        same_x_.set_direct_transfer_options( options.use_direct_backward_receive, options.direct_p2p_cuda_aware );
        same_z_.set_direct_transfer_options( options.use_direct_backward_receive, options.direct_p2p_cuda_aware );
        same_x_.set_p2p_send_thread_enabled( options.use_p2p_send_thread );
        same_z_.set_p2p_send_thread_enabled( options.use_p2p_send_thread );
        same_x_.set_p2p_byte_transfer_enabled( options.use_p2p_byte_transfer );
        same_z_.set_p2p_byte_transfer_enabled( options.use_p2p_byte_transfer );
    }

    void configure_memory_profiling_( const fftm_init_options &options )
    {
        if ( !options.memory_profiling_key.empty() )
        {
            memory_profiler_.enable( options.memory_profiling_key );
        }
        else
        {
            memory_profiler_.disable();
        }

        base_fft_.set_memory_profiler( memory_profiler_.native_ptr(), "fftm/base_fft" );
        same_x_.set_memory_profiler( memory_profiler_.native_ptr(), "fftm/transpose_3d_same_x" );
        same_z_.set_memory_profiler( memory_profiler_.native_ptr(), "fftm/transpose_3d_same_z" );
        same_xy_.set_memory_profiler( memory_profiler_.native_ptr(), "fftm/transpose_4d_same_xy" );
        same_xw_.set_memory_profiler( memory_profiler_.native_ptr(), "fftm/transpose_4d_same_xw" );
        same_zw_.set_memory_profiler( memory_profiler_.native_ptr(), "fftm/transpose_4d_same_zw" );
        update_memory_profile_for_current_dim_();
    }

    void log_profile_on_destroy_()
    {
        if ( !profiler_.enabled() )
        {
            return;
        }
        if ( init_options_.print_profile_summary_on_destroy )
        {
            profiler_.log_print( log_ );
        }
        if ( init_options_.print_profile_totals_on_destroy )
        {
            profiler_.log_print_totals( log_ );
        }
    }

    void log_memory_profile_on_destroy_()
    {
        if ( !memory_profiler_.enabled() )
        {
            return;
        }
        if ( init_options_.print_memory_profile_on_destroy )
        {
            log_memory_profile_summary_();
        }
        if ( init_options_.print_memory_totals_on_destroy )
        {
            log_memory_profile_totals_();
        }
    }

    void log_memory_profile_summary_()
    {
        memory_profiler_t *profiler = memory_profiler_.native_ptr();
        if ( profiler == nullptr )
        {
            return;
        }

        const memory_profile_buckets_t local_buckets = collect_memory_profile_buckets_();
        std::stringstream ss;
        ss << "Memory profile (MPI reduced):";
        for ( const auto &item : profiler->entries() )
        {
            const auto current_local = item.second.current_bytes;
            const auto peak_local    = item.second.peak_bytes;
            const auto current_sum   = mpi_.all_reduce_sum( current_local );
            const auto current_max   = mpi_.all_reduce_max( current_local );
            const auto peak_sum      = mpi_.all_reduce_sum( peak_local );
            const auto peak_max      = mpi_.all_reduce_max( peak_local );

            if ( mpi_.myid == 0 )
            {
                const double proc_count = static_cast<double>( mpi_.num_procs );
                ss << "\n  " << item.first << ": current(sum/max/avg)=" << current_sum << "/" << current_max << "/"
                   << static_cast<typename memory_profiler_t::bytes_type>( current_sum / mpi_.num_procs ) << " B ("
                   << std::fixed << std::setprecision( 3 ) << bytes_to_mib_( current_sum ) << "/"
                   << bytes_to_mib_( current_max ) << "/"
                   << bytes_to_mib_( static_cast<typename memory_profiler_t::bytes_type>( current_sum / mpi_.num_procs ) )
                   << " MiB)"
                   << ", peak(sum/max/avg)=" << peak_sum << "/" << peak_max << "/"
                   << static_cast<typename memory_profiler_t::bytes_type>( peak_sum / mpi_.num_procs ) << " B ("
                   << bytes_to_mib_( peak_sum ) << "/" << bytes_to_mib_( peak_max ) << "/"
                   << bytes_to_mib_( static_cast<typename memory_profiler_t::bytes_type>( peak_sum / mpi_.num_procs ) )
                   << " MiB)";
                (void)proc_count;
            }
        }

        ss << "\nMemory profile categories (MPI reduced):";
        append_memory_profile_bucket_lines_mpi_( ss, local_buckets, true );
        if ( mpi_.myid == 0 )
        {
            log_.info( ss.str() );
        }
    }

    void log_memory_profile_totals_()
    {
        memory_profiler_t *profiler = memory_profiler_.native_ptr();
        if ( profiler == nullptr )
        {
            return;
        }

        const memory_profile_buckets_t local_buckets = collect_memory_profile_buckets_();
        std::stringstream              ss;
        ss << "Memory profile totals (MPI reduced):";
        append_memory_profile_total_lines_mpi_( ss, local_buckets, true );

        if ( mpi_.myid == 0 )
        {
            log_.info( ss.str() );
        }
    }

    static double bytes_to_mib_( memory_profile_bytes_t bytes )
    {
        return static_cast<double>( bytes ) / ( 1024.0 * 1024.0 );
    }

    memory_profile_buckets_t collect_memory_profile_buckets_() const
    {
        memory_profile_buckets_t       buckets;
        const memory_profiler_t       *profiler = memory_profiler_.native_ptr();
        if ( profiler == nullptr )
        {
            return buckets;
        }

        for ( const auto &item : profiler->entries() )
        {
            buckets.add( item.first, item.second.current_bytes, item.second.peak_bytes );
        }
        return buckets;
    }

    void append_memory_profile_line_mpi_(
        std::stringstream &ss, const char *name, memory_profile_bytes_t current_local, memory_profile_bytes_t peak_local
    ) const
    {
        const auto current_sum = mpi_.all_reduce_sum( current_local );
        const auto current_max = mpi_.all_reduce_max( current_local );
        const auto peak_sum    = mpi_.all_reduce_sum( peak_local );
        const auto peak_max    = mpi_.all_reduce_max( peak_local );

        if ( mpi_.myid != 0 )
        {
            return;
        }

        ss << "\n  " << name << ": current(sum/max/avg)=" << current_sum << "/" << current_max << "/"
           << static_cast<memory_profile_bytes_t>( current_sum / mpi_.num_procs ) << " B ("
           << std::fixed << std::setprecision( 3 ) << bytes_to_mib_( current_sum ) << "/" << bytes_to_mib_( current_max )
           << "/" << bytes_to_mib_( static_cast<memory_profile_bytes_t>( current_sum / mpi_.num_procs ) ) << " MiB)"
           << ", peak(sum/max/avg)=" << peak_sum << "/" << peak_max << "/"
           << static_cast<memory_profile_bytes_t>( peak_sum / mpi_.num_procs ) << " B (" << bytes_to_mib_( peak_sum )
           << "/" << bytes_to_mib_( peak_max ) << "/"
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( peak_sum / mpi_.num_procs ) ) << " MiB)";
    }

    void append_memory_profile_bucket_lines_mpi_(
        std::stringstream &ss, const memory_profile_buckets_t &local_buckets, bool include_external_test
    ) const
    {
        append_memory_profile_line_mpi_(
            ss, detail::memory_profile_bucket_name( detail::memory_profile_bucket::device ),
            local_buckets.get( detail::memory_profile_bucket::device ).current,
            local_buckets.get( detail::memory_profile_bucket::device ).peak
        );
        append_memory_profile_line_mpi_(
            ss, detail::memory_profile_bucket_name( detail::memory_profile_bucket::host_pinned ),
            local_buckets.get( detail::memory_profile_bucket::host_pinned ).current,
            local_buckets.get( detail::memory_profile_bucket::host_pinned ).peak
        );
        if ( include_external_test )
        {
            append_memory_profile_line_mpi_(
                ss, detail::memory_profile_bucket_name( detail::memory_profile_bucket::external_test ),
                local_buckets.get( detail::memory_profile_bucket::external_test ).current,
                local_buckets.get( detail::memory_profile_bucket::external_test ).peak
            );
        }

        const auto other = local_buckets.get( detail::memory_profile_bucket::other );
        append_memory_profile_line_mpi_(
            ss, detail::memory_profile_bucket_name( detail::memory_profile_bucket::other ), other.current, other.peak
        );
    }

    void append_memory_profile_total_lines_mpi_(
        std::stringstream &ss, const memory_profile_buckets_t &local_buckets, bool include_external_test
    ) const
    {
        append_memory_profile_line_mpi_( ss, "total", local_buckets.total().current, local_buckets.total().peak );
        append_memory_profile_bucket_lines_mpi_( ss, local_buckets, include_external_test );
    }

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
                "fftm: initialized for " + std::to_string( dim_ ) + "D but used as " + std::to_string( expected_dim ) +
                "D."
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

    void init_(
        std::integral_constant<std::size_t, 3>, const processor_grid &pg, const global_sizes &gs,
        const fftm_init_options &options
    )
    {
        ensure_can_init_();
        if ( gs.is_4D() )
            throw std::logic_error( "fftm::init<3> received 4D sizes." );
        configure_profiling_( options );

        FFTM_PROFILE_SCOPED_TIC( "fftm::init<3>" );
        SCFD_SAFE_CALL( partitioning_.init( pg, gs ) );
        SCFD_SAFE_CALL( std::tie( myid_i_, myid_j_, myid_k_ ) = partitioning_.get_my_grid() );
        SCFD_SAFE_CALL(
            std::tie( input_dim_, transpose1_dim_, transpose2_dim_ ) = partitioning_.get_partitioning_3D()
        );

        nx_      = gs.Nx;
        ny_      = gs.Ny;
        nz_      = gs.Nz;
        nw_      = 0;
        nz_half_ = nz_ / 2 + 1;
        nw_half_ = 0;
        dim_     = 3;

        half_input_dim_           = input_dim_;
        half_input_dim_.size_z[0] = nz_half_;
        SCFD_SAFE_CALL( half_input_dim_.compute_offsets( false ) );

        SCFD_SAFE_CALL( init_strategy_( strategy_family_3d_tag() ) );
        SCFD_SAFE_CALL( add_plans_( strategy_family_3d_tag() ) );
        SCFD_SAFE_CALL( activate_shared_work_area_() );
        init_done_ = true;
    }

    void init_(
        std::integral_constant<std::size_t, 4>, const processor_grid &pg, const global_sizes &gs,
        const fftm_init_options &options
    )
    {
        ensure_can_init_();
        if ( !gs.is_4D() )
            throw std::logic_error( "fftm::init<4> requires 4D global sizes." );
        configure_profiling_( options );

        FFTM_PROFILE_SCOPED_TIC( "fftm::init<4>" );
        SCFD_SAFE_CALL( partitioning_.init( pg, gs ) );
        SCFD_SAFE_CALL( std::tie( myid_i_, myid_j_, myid_k_ ) = partitioning_.get_my_grid() );
        SCFD_SAFE_CALL(
            std::tie( input_dim_, transpose1_dim_, transpose2_dim_, transpose3_dim_ ) =
                partitioning_.get_partitioning_4D()
        );

        nx_      = gs.Nx;
        ny_      = gs.Ny;
        nz_      = gs.Nz;
        nw_      = gs.Nw;
        nz_half_ = 0;
        nw_half_ = nw_ / 2 + 1;
        dim_     = 4;

        half_input_dim_           = input_dim_;
        half_input_dim_.size_w[0] = nw_half_;
        SCFD_SAFE_CALL( half_input_dim_.compute_offsets( true ) );

        SCFD_SAFE_CALL( init_4d_strategy_( strategy_family_4d_tag() ) );
        SCFD_SAFE_CALL( add_4d_plans_( strategy_family_4d_tag() ) );
        SCFD_SAFE_CALL( activate_shared_work_area_() );
        init_done_ = true;
    }

    void add_plan_z_r2c_legacy_y_fast_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size
    )
    {
        const long long int stride = x_size * y_size;
        const long long int batch  = x_size * y_size;

        base_fft_.template add_plan_1D<::fftm::direction::R2C>(
            forward_name, static_cast<long long int>( nz_ ), 1, stride, 1, 1, x_size * y_size, 1, batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2R>(
            inverse_name, static_cast<long long int>( nz_ ), 1, stride, 1, 1, x_size * y_size, 1, batch
        );
    }

    void add_plan_yz_r2c_slab_optimized_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size
    )
    {
        // Egger-style slab path: each local [y,z] plane is contiguous, so the
        // first transform is a true batched 2D transform with unit stride.
        const long long int real_dist    = static_cast<long long int>( ny_ * nz_ );
        const long long int complex_dist = static_cast<long long int>( ny_ * nz_half_ );

        base_fft_.template add_plan_2D<::fftm::direction::R2C>(
            forward_name, static_cast<long long int>( ny_ ), static_cast<long long int>( nz_ ),
            static_cast<long long int>( ny_ ), static_cast<long long int>( nz_ ), 1, real_dist,
            static_cast<long long int>( ny_ ), static_cast<long long int>( nz_half_ ), 1, complex_dist, x_size
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2R>(
            inverse_name, static_cast<long long int>( ny_ ), static_cast<long long int>( nz_ ),
            static_cast<long long int>( ny_ ), static_cast<long long int>( nz_half_ ), 1, complex_dist,
            static_cast<long long int>( ny_ ), static_cast<long long int>( nz_ ), 1, real_dist, x_size
        );
    }

    void add_plan_y_c2c_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int z_size
    )
    {
        const long long int batch = x_size * z_size;
        const long long int idist = static_cast<long long int>( ny_ );

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name, static_cast<long long int>( ny_ ), 1, 1, idist, 1, 1, idist, batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name, static_cast<long long int>( ny_ ), 1, 1, idist, 1, 1, idist, batch
        );
    }

    void add_plan_x_c2c_(
        const std::string &forward_name, const std::string &inverse_name, long long int y_size, long long int z_size
    )
    {
        const long long int batch  = y_size * z_size;
        const long long int stride = y_size * z_size;

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name, static_cast<long long int>( nx_ ), 1, stride, 1, 1, stride, 1, batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name, static_cast<long long int>( nx_ ), 1, stride, 1, 1, stride, 1, batch
        );
    }

    void add_plan_xy_c2c_pencil_slab_(
        const std::string &forward_name, const std::string &inverse_name, long long int z_size
    )
    {
        // Pencil-slab owns full X and full Y after the first redistribution.
        // The current xzy storage keeps each [x,y] plane contiguous for a
        // fixed local z, so this replaces two 1D FFTs plus reorder copies with
        // a single batched 2D C2C transform.
        const long long int dist  = static_cast<long long int>( nx_ * ny_ );
        const long long int batch = z_size;

        base_fft_.template add_plan_2D<::fftm::direction::C2CF>(
            forward_name, static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), 1, dist,
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), 1, dist, batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2CB>(
            inverse_name, static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), 1, dist,
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), 1, dist, batch
        );
    }

    void add_plan_w_r2c_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size,
        long long int z_size
    )
    {
        const long long int stride = x_size * y_size * z_size;
        const long long int batch  = x_size * y_size * z_size;

        base_fft_.template add_plan_1D<::fftm::direction::R2C>(
            forward_name, static_cast<long long int>( nw_ ), 1, stride, 1, 1, stride, 1, batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2R>(
            inverse_name, static_cast<long long int>( nw_ ), 1, stride, 1, 1, stride, 1, batch
        );
    }

    void add_plan_z_c2c_4d_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size,
        long long int w_size
    )
    {
        const long long int stride = x_size * y_size * w_size;
        const long long int batch  = x_size * y_size * w_size;

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name, static_cast<long long int>( nz_ ), 1, stride, 1, 1, stride, 1, batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name, static_cast<long long int>( nz_ ), 1, stride, 1, 1, stride, 1, batch
        );
    }

    void add_plan_y_c2c_4d_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int z_size,
        long long int w_size
    )
    {
        const long long int stride = x_size * z_size * w_size;
        const long long int batch  = x_size * z_size * w_size;

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name, static_cast<long long int>( ny_ ), 1, stride, 1, 1, stride, 1, batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name, static_cast<long long int>( ny_ ), 1, stride, 1, 1, stride, 1, batch
        );
    }

    void add_plan_x_c2c_4d_(
        const std::string &forward_name, const std::string &inverse_name, long long int y_size, long long int z_size,
        long long int w_size
    )
    {
        const long long int batch = y_size * z_size * w_size;

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name, static_cast<long long int>( nx_ ), 1, 1, static_cast<long long int>( nx_ ), 1, 1,
            static_cast<long long int>( nx_ ), batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name, static_cast<long long int>( nx_ ), 1, 1, static_cast<long long int>( nx_ ), 1, 1,
            static_cast<long long int>( nx_ ), batch
        );
    }

    void add_plan_zw_r2c_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size
    )
    {
        const long long int batch = x_size * y_size;

        base_fft_.template add_plan_2D<::fftm::direction::R2C>(
            forward_name, static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ), 1,
            static_cast<long long int>( nz_ * nw_ ), static_cast<long long int>( nz_ ),
            static_cast<long long int>( nw_half_ ), 1, static_cast<long long int>( nz_ * nw_half_ ), batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2R>(
            inverse_name, static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_half_ ), 1,
            static_cast<long long int>( nz_ * nw_half_ ), static_cast<long long int>( nz_ ),
            static_cast<long long int>( nw_ ), 1, static_cast<long long int>( nz_ * nw_ ), batch
        );
    }

    void add_plan_xy_c2c_4d_(
        const std::string &forward_name, const std::string &inverse_name, long long int z_size, long long int w_size
    )
    {
        const long long int batch     = z_size * w_size;
        const long long int xy_stride = z_size * w_size;

        base_fft_.template add_plan_2D<::fftm::direction::C2CF>(
            forward_name, static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), xy_stride, 1,
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), xy_stride, 1, batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2CB>(
            inverse_name, static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), xy_stride, 1,
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), xy_stride, 1, batch
        );
    }

    void init_strategy_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::init_strategy_3d_slab_pencil" );
        require_grid_p2_is_one_();

        output_dim_ = transpose2_dim_;

        SCFD_SAFE_CALL( init_shared_stage0_xfft_3d_(
            input_dim_.size_x[myid_i_], nz_half_, ny_, output_dim_.size_x[0], output_dim_.size_z[myid_j_],
            output_dim_.size_y[myid_i_]
        ) );
        SCFD_SAFE_CALL( same_z_.init( half_input_dim_, output_dim_, myid_i_, 0 ) );
    }

    void init_strategy_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::init_strategy_3d_pencil_slab" );
        require_grid_p1_is_one_();

        output_dim_ = transpose2_dim_;

        SCFD_SAFE_CALL( init_shared_stage0_xfft_3d_(
            input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], nz_half_, output_dim_.size_x[0],
            output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_]
        ) );
        SCFD_SAFE_CALL( same_x_.init( half_input_dim_, transpose1_dim_, myid_i_, myid_j_ ) );
    }

    void init_strategy_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::init_strategy_3d_pencil_pencil" );
        output_dim_ = transpose2_dim_;

        SCFD_SAFE_CALL( init_shared_stage0_xfft_3d_(
            input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], nz_half_, output_dim_.size_x[0],
            output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_], input_dim_.size_x[myid_i_],
            transpose1_dim_.size_z[myid_j_], ny_
        ) );
        SCFD_SAFE_CALL( init_owned_stage1_3d_( input_dim_.size_x[myid_i_], transpose1_dim_.size_z[myid_j_], ny_ ) );

        SCFD_SAFE_CALL( same_x_.init( half_input_dim_, transpose1_dim_, myid_i_, myid_j_ ) );
        SCFD_SAFE_CALL( same_z_.init( transpose1_dim_, output_dim_, myid_i_, myid_j_ ) );
    }

    void
    init_4d_strategy_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::init_strategy_4d_pencil_pencil" );
        output_dim_ = transpose3_dim_;

        SCFD_SAFE_CALL( init_shared_stage0_stage2_4d_(
            input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], input_dim_.size_z[myid_k_], nw_half_,
            transpose2_dim_.size_x[myid_i_], transpose2_dim_.size_z[myid_j_], transpose2_dim_.size_w[myid_k_],
            transpose2_dim_.size_y[0]
        ) );
        SCFD_SAFE_CALL( init_owned_stage1_4d_(
            transpose1_dim_.size_x[myid_i_], transpose1_dim_.size_y[myid_j_], transpose1_dim_.size_w[myid_k_],
            transpose1_dim_.size_z[0]
        ) );

        SCFD_SAFE_CALL( same_xy_.init( half_input_dim_, transpose1_dim_, myid_i_, myid_j_, myid_k_ ) );
        SCFD_SAFE_CALL( same_xw_.init( transpose1_dim_, transpose2_dim_, myid_i_, myid_j_, myid_k_ ) );
        SCFD_SAFE_CALL( same_zw_.init( transpose2_dim_, output_dim_, myid_i_, myid_j_, myid_k_ ) );
        update_memory_profile_4d_();
    }

    void init_4d_strategy_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::init_strategy_4d_slab_slab" );
        require_grid_p1_is_one_();
        require_grid_p3_is_one_();

        output_dim_ = transpose3_dim_;

        SCFD_SAFE_CALL( init_shared_stage0_stage2_4d_(
            input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], input_dim_.size_z[myid_k_], nw_half_,
            transpose2_dim_.size_x[myid_i_], transpose2_dim_.size_z[myid_j_], transpose2_dim_.size_w[myid_k_],
            transpose2_dim_.size_y[0]
        ) );
        SCFD_SAFE_CALL( init_owned_stage1_4d_(
            transpose1_dim_.size_x[myid_i_], transpose1_dim_.size_y[myid_j_], transpose1_dim_.size_w[myid_k_],
            transpose1_dim_.size_z[0]
        ) );

        SCFD_SAFE_CALL( same_xw_.init( transpose1_dim_, transpose2_dim_, myid_i_, myid_j_, myid_k_ ) );
        update_memory_profile_4d_();
    }

    void add_plans_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::add_plans_3d_slab_pencil" );
        SCFD_SAFE_CALL( add_plans_3d_slab_pencil_layout_( std::integral_constant<bool, strategy_3d_optimized_layout>() ) );
    }

    void add_plans_3d_slab_pencil_layout_( std::false_type )
    {
        SCFD_SAFE_CALL( add_plan_z_r2c_legacy_y_fast_( "forward_z", "inverse_z", input_dim_.size_x[myid_i_], ny_ ) );
        SCFD_SAFE_CALL( add_plan_y_c2c_( "forward_y", "inverse_y", input_dim_.size_x[myid_i_], nz_half_ ) );
        SCFD_SAFE_CALL( add_plan_x_c2c_( "forward_x", "inverse_x", output_dim_.size_y[myid_i_], nz_half_ ) );
    }

    void add_plans_3d_slab_pencil_layout_( std::true_type )
    {
        SCFD_SAFE_CALL( add_plan_yz_r2c_slab_optimized_(
            "forward_yz", "inverse_yz", static_cast<long long int>( input_dim_.size_x[myid_i_] )
        ) );
        SCFD_SAFE_CALL( add_plan_x_c2c_( "forward_x", "inverse_x", output_dim_.size_y[myid_i_], nz_half_ ) );
    }

    void add_plans_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::add_plans_3d_pencil_slab" );
        SCFD_SAFE_CALL(
            add_plans_3d_pencil_slab_layout_( std::integral_constant<bool, strategy_3d_optimized_layout>() )
        );
    }

    void add_plans_3d_pencil_slab_layout_( std::false_type )
    {
        SCFD_SAFE_CALL( add_plan_z_r2c_legacy_y_fast_( "forward_z", "inverse_z", nx_, input_dim_.size_y[myid_j_] ) );
        SCFD_SAFE_CALL( add_plan_y_c2c_( "forward_y", "inverse_y", nx_, output_dim_.size_z[myid_j_] ) );
        SCFD_SAFE_CALL( add_plan_x_c2c_( "forward_x", "inverse_x", ny_, output_dim_.size_z[myid_j_] ) );
    }

    void add_plans_3d_pencil_slab_layout_( std::true_type )
    {
        if ( !init_options_.use_optimized )
        {
            SCFD_SAFE_CALL( add_plans_3d_pencil_slab_layout_( std::false_type() ) );
            return;
        }

        SCFD_SAFE_CALL( add_plan_z_r2c_legacy_y_fast_( "forward_z", "inverse_z", nx_, input_dim_.size_y[myid_j_] ) );
        SCFD_SAFE_CALL(
            add_plan_xy_c2c_pencil_slab_( "forward_xy", "inverse_xy", output_dim_.size_z[myid_j_] )
        );
    }

    void add_plans_( std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::add_plans_3d_pencil_pencil" );
        SCFD_SAFE_CALL(
            add_plans_3d_pencil_pencil_layout_( std::integral_constant<bool, strategy_3d_optimized_layout>() )
        );
    }

    void add_plans_3d_pencil_pencil_layout_( std::false_type )
    {
        SCFD_SAFE_CALL(
            add_plan_z_r2c_legacy_y_fast_( "forward_z", "inverse_z", input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_] )
        );
        SCFD_SAFE_CALL(
            add_plan_y_c2c_( "forward_y", "inverse_y", input_dim_.size_x[myid_i_], transpose1_dim_.size_z[myid_j_] )
        );
        SCFD_SAFE_CALL(
            add_plan_x_c2c_( "forward_x", "inverse_x", output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_] )
        );
    }

    void add_plans_3d_pencil_pencil_layout_( std::true_type )
    {
        if ( !init_options_.use_optimized )
        {
            throw std::logic_error(
                "fftm 3D pencil-pencil optimized layout was selected at compile time; use "
                "strategy_3d_pencil_pencil<Mode, false> for the legacy runtime path."
            );
        }

        SCFD_SAFE_CALL( add_plans_3d_pencil_pencil_layout_( std::false_type() ) );
    }

    void add_4d_plans_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::add_plans_4d_pencil_pencil" );
        SCFD_SAFE_CALL( add_plan_w_r2c_(
            "forward_w", "inverse_w", input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], input_dim_.size_z[myid_k_]
        ) );
        SCFD_SAFE_CALL( add_plan_z_c2c_4d_(
            "forward_z", "inverse_z", transpose1_dim_.size_x[myid_i_], transpose1_dim_.size_y[myid_j_],
            transpose1_dim_.size_w[myid_k_]
        ) );
        SCFD_SAFE_CALL( add_plan_y_c2c_4d_(
            "forward_y", "inverse_y", transpose2_dim_.size_x[myid_i_], transpose2_dim_.size_z[myid_j_],
            transpose2_dim_.size_w[myid_k_]
        ) );
        SCFD_SAFE_CALL( add_plan_x_c2c_4d_(
            "forward_x", "inverse_x", output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_],
            output_dim_.size_w[myid_k_]
        ) );
    }

    void add_4d_plans_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::add_plans_4d_slab_slab" );
        SCFD_SAFE_CALL(
            add_plan_zw_r2c_( "forward_zw", "inverse_zw", input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_] )
        );
        SCFD_SAFE_CALL(
            add_plan_xy_c2c_4d_( "forward_xy", "inverse_xy", output_dim_.size_z[myid_j_], output_dim_.size_w[myid_k_] )
        );
    }

    void forward_3d_(
        std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil>, const real_array3_t &in,
        complex_array3_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::forward_3d_slab_pencil" );
        SCFD_SAFE_CALL( forward_3d_slab_pencil_layout_(
            std::integral_constant<bool, strategy_3d_optimized_layout>(), in, out
        ) );
    }

    void forward_3d_slab_pencil_layout_( std::false_type, const real_array3_t &in, complex_array3_t &out )
    {
        SCFD_SAFE_CALL( base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_z", in, stage0_3d_ ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage0_complex3_t, stage0_complex3_t>( "forward_y", stage0_3d_, stage0_3d_ )
        );
        SCFD_SAFE_CALL( same_z_.transpose_x_to_y( stage0_3d_, out, transpose_mode_3d ) );
        SCFD_SAFE_CALL( reorder_x_stage_( out, x_fft_stage_3d_ ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "forward_x", x_fft_stage_3d_, x_fft_stage_3d_ )
        );
        SCFD_SAFE_CALL( reorder_x_stage_( x_fft_stage_3d_, out ) );
    }

    void forward_3d_slab_pencil_layout_( std::true_type, const real_array3_t &in, complex_array3_t &out )
    {
        if ( !init_options_.use_optimized )
        {
            throw std::logic_error(
                "fftm 3D slab-pencil optimized layout was selected at compile time; use "
                "strategy_3d_slab_pencil<Mode, false> for the legacy runtime path."
            );
        }
        SCFD_SAFE_CALL( base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_yz", in, stage0_3d_ ) );
        SCFD_SAFE_CALL( same_z_.transpose_x_to_y_optimized_layout( stage0_3d_, out, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, complex_array3_t>( "forward_x", out, out ) );
    }

    void backward_3d_(
        std::integral_constant<transform_strategy_3d, transform_strategy_3d::slab_pencil>, complex_array3_t &in,
        real_array3_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::backward_3d_slab_pencil" );
        SCFD_SAFE_CALL( backward_3d_slab_pencil_layout_(
            std::integral_constant<bool, strategy_3d_optimized_layout>(), in, out
        ) );
    }

    void backward_3d_slab_pencil_layout_( std::false_type, complex_array3_t &in, real_array3_t &out )
    {
        SCFD_SAFE_CALL( reorder_x_stage_( in, x_fft_stage_3d_ ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "inverse_x", x_fft_stage_3d_, x_fft_stage_3d_ )
        );
        SCFD_SAFE_CALL( reorder_x_stage_( x_fft_stage_3d_, in ) );
        SCFD_SAFE_CALL( same_z_.transpose_y_to_x( in, stage0_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage0_complex3_t, stage0_complex3_t>( "inverse_y", stage0_3d_, stage0_3d_ )
        );
        SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out ) );
    }

    void backward_3d_slab_pencil_layout_( std::true_type, complex_array3_t &in, real_array3_t &out )
    {
        if ( !init_options_.use_optimized )
        {
            throw std::logic_error(
                "fftm 3D slab-pencil optimized layout was selected at compile time; use "
                "strategy_3d_slab_pencil<Mode, false> for the legacy runtime path."
            );
        }
        SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, complex_array3_t>( "inverse_x", in, in ) );
        SCFD_SAFE_CALL( same_z_.transpose_y_to_x_optimized_layout( in, stage0_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_yz", stage0_3d_, out ) );
    }

    void forward_3d_(
        std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab>, const real_array3_t &in,
        complex_array3_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::forward_3d_pencil_slab" );
        SCFD_SAFE_CALL( forward_3d_pencil_slab_layout_(
            std::integral_constant<bool, strategy_3d_optimized_layout>(), in, out
        ) );
    }

    void forward_3d_pencil_slab_layout_( std::false_type, const real_array3_t &in, complex_array3_t &out )
    {
        SCFD_SAFE_CALL( base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_z", in, stage0_3d_ ) );
        SCFD_SAFE_CALL( same_x_.transpose_xyz_to_xzy( stage0_3d_, out, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, complex_array3_t>( "forward_y", out, out ) );
        SCFD_SAFE_CALL( reorder_x_stage_( out, x_fft_stage_3d_ ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "forward_x", x_fft_stage_3d_, x_fft_stage_3d_ )
        );
        SCFD_SAFE_CALL( reorder_x_stage_( x_fft_stage_3d_, out ) );
    }

    void forward_3d_pencil_slab_layout_( std::true_type, const real_array3_t &in, complex_array3_t &out )
    {
        if ( !init_options_.use_optimized )
        {
            SCFD_SAFE_CALL( forward_3d_pencil_slab_layout_( std::false_type(), in, out ) );
            return;
        }

        SCFD_SAFE_CALL( base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_z", in, stage0_3d_ ) );
        SCFD_SAFE_CALL( same_x_.transpose_xyz_to_xzy( stage0_3d_, out, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, complex_array3_t>( "forward_xy", out, out ) );
    }

    void backward_3d_(
        std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_slab>, complex_array3_t &in,
        real_array3_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::backward_3d_pencil_slab" );
        SCFD_SAFE_CALL( backward_3d_pencil_slab_layout_(
            std::integral_constant<bool, strategy_3d_optimized_layout>(), in, out
        ) );
    }

    void backward_3d_pencil_slab_layout_( std::false_type, complex_array3_t &in, real_array3_t &out )
    {
        SCFD_SAFE_CALL( reorder_x_stage_( in, x_fft_stage_3d_ ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "inverse_x", x_fft_stage_3d_, x_fft_stage_3d_ )
        );
        SCFD_SAFE_CALL( reorder_x_stage_( x_fft_stage_3d_, in ) );
        SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, complex_array3_t>( "inverse_y", in, in ) );
        SCFD_SAFE_CALL( same_x_.transpose_xzy_to_xyz( in, stage0_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out ) );
    }

    void backward_3d_pencil_slab_layout_( std::true_type, complex_array3_t &in, real_array3_t &out )
    {
        if ( !init_options_.use_optimized )
        {
            SCFD_SAFE_CALL( backward_3d_pencil_slab_layout_( std::false_type(), in, out ) );
            return;
        }

        SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, complex_array3_t>( "inverse_xy", in, in ) );
        SCFD_SAFE_CALL( same_x_.transpose_xzy_to_xyz( in, stage0_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out ) );
    }

    void forward_3d_(
        std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil>, const real_array3_t &in,
        complex_array3_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::forward_3d_pencil_pencil" );
        SCFD_SAFE_CALL( forward_3d_pencil_pencil_layout_(
            std::integral_constant<bool, strategy_3d_optimized_layout>(), in, out
        ) );
    }

    void forward_3d_pencil_pencil_layout_( std::false_type, const real_array3_t &in, complex_array3_t &out )
    {
        SCFD_SAFE_CALL( base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_z", in, stage0_3d_ ) );
        SCFD_SAFE_CALL( same_x_.transpose_xyz_to_xzy( stage0_3d_, stage1_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage1_complex3_t, stage1_complex3_t>( "forward_y", stage1_3d_, stage1_3d_ )
        );
        SCFD_SAFE_CALL( same_z_.transpose_x_to_y( stage1_3d_, out, transpose_mode_3d ) );
        SCFD_SAFE_CALL( reorder_x_stage_( out, x_fft_stage_3d_ ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "forward_x", x_fft_stage_3d_, x_fft_stage_3d_ )
        );
        SCFD_SAFE_CALL( reorder_x_stage_( x_fft_stage_3d_, out ) );
    }

    void forward_3d_pencil_pencil_layout_( std::true_type, const real_array3_t &in, complex_array3_t &out )
    {
        if ( !init_options_.use_optimized )
        {
            throw std::logic_error(
                "fftm 3D pencil-pencil optimized layout was selected at compile time; use "
                "strategy_3d_pencil_pencil<Mode, false> for the legacy runtime path."
            );
        }

        SCFD_SAFE_CALL( base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_z", in, stage0_3d_ ) );
        SCFD_SAFE_CALL( same_x_.transpose_xyz_to_xzy( stage0_3d_, stage1_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage1_complex3_t, stage1_complex3_t>( "forward_y", stage1_3d_, stage1_3d_ )
        );
        SCFD_SAFE_CALL( reorder_stage1_to_xfast_( stage1_3d_, stage1_xfast_3d_ ) );
        SCFD_SAFE_CALL( same_z_.transpose_x_to_y_optimized_layout( stage1_xfast_3d_, out, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, complex_array3_t>( "forward_x", out, out ) );
    }

    void backward_3d_(
        std::integral_constant<transform_strategy_3d, transform_strategy_3d::pencil_pencil>, complex_array3_t &in,
        real_array3_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::backward_3d_pencil_pencil" );
        SCFD_SAFE_CALL( backward_3d_pencil_pencil_layout_(
            std::integral_constant<bool, strategy_3d_optimized_layout>(), in, out
        ) );
    }

    void backward_3d_pencil_pencil_layout_( std::false_type, complex_array3_t &in, real_array3_t &out )
    {
        SCFD_SAFE_CALL( reorder_x_stage_( in, x_fft_stage_3d_ ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<x_fft_complex3_t, x_fft_complex3_t>( "inverse_x", x_fft_stage_3d_, x_fft_stage_3d_ )
        );
        SCFD_SAFE_CALL( reorder_x_stage_( x_fft_stage_3d_, in ) );
        SCFD_SAFE_CALL( same_z_.transpose_y_to_x( in, stage1_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage1_complex3_t, stage1_complex3_t>( "inverse_y", stage1_3d_, stage1_3d_ )
        );
        SCFD_SAFE_CALL( same_x_.transpose_xzy_to_xyz( stage1_3d_, stage0_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out ) );
    }

    void backward_3d_pencil_pencil_layout_( std::true_type, complex_array3_t &in, real_array3_t &out )
    {
        if ( !init_options_.use_optimized )
        {
            throw std::logic_error(
                "fftm 3D pencil-pencil optimized layout was selected at compile time; use "
                "strategy_3d_pencil_pencil<Mode, false> for the legacy runtime path."
            );
        }

        SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, complex_array3_t>( "inverse_x", in, in ) );
        SCFD_SAFE_CALL( same_z_.transpose_y_to_x_optimized_layout( in, stage1_xfast_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL( reorder_stage1_from_xfast_( stage1_xfast_3d_, stage1_3d_ ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage1_complex3_t, stage1_complex3_t>( "inverse_y", stage1_3d_, stage1_3d_ )
        );
        SCFD_SAFE_CALL( same_x_.transpose_xzy_to_xyz( stage1_3d_, stage0_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out ) );
    }

    void forward_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil>,
        const real_array4_t &in, complex_array4_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::forward_4d_pencil_pencil" );
        SCFD_SAFE_CALL( base_fft_.template exec<real_array4_t, stage0_complex4_t>( "forward_w", in, stage0_4d_ ) );
        SCFD_SAFE_CALL( same_xy_.transpose_xyzw_to_xywz( stage0_4d_, stage1_4d_, transpose_mode_4d ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage1_complex4_t, stage1_complex4_t>( "forward_z", stage1_4d_, stage1_4d_ )
        );
        SCFD_SAFE_CALL( same_xw_.transpose_xywz_to_xzwy( stage1_4d_, stage2_4d_, transpose_mode_4d ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>( "forward_y", stage2_4d_, stage2_4d_ )
        );
        SCFD_SAFE_CALL( same_zw_.transpose_xzwy_to_yzwx( stage2_4d_, out, transpose_mode_4d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<complex_array4_t, complex_array4_t>( "forward_x", out, out ) );
    }

    void backward_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil>,
        complex_array4_t &in, real_array4_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::backward_4d_pencil_pencil" );
        SCFD_SAFE_CALL( base_fft_.template exec<complex_array4_t, complex_array4_t>( "inverse_x", in, in ) );
        SCFD_SAFE_CALL( same_zw_.transpose_yzwx_to_xzwy( in, stage2_4d_, transpose_mode_4d ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>( "inverse_y", stage2_4d_, stage2_4d_ )
        );
        SCFD_SAFE_CALL( same_xw_.transpose_xzwy_to_xywz( stage2_4d_, stage1_4d_, transpose_mode_4d ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage1_complex4_t, stage1_complex4_t>( "inverse_z", stage1_4d_, stage1_4d_ )
        );
        SCFD_SAFE_CALL( same_xy_.transpose_xywz_to_xyzw( stage1_4d_, stage0_4d_, transpose_mode_4d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex4_t, real_array4_t>( "inverse_w", stage0_4d_, out ) );
    }

    void forward_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab>,
        const real_array4_t &in, complex_array4_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::forward_4d_slab_slab" );
        SCFD_SAFE_CALL( base_fft_.template exec<real_array4_t, stage0_complex4_t>( "forward_zw", in, stage0_4d_ ) );
        SCFD_SAFE_CALL( transpose_local_4d_<0, 1, 3, 2>( stage0_4d_, stage1_4d_ ) );
        SCFD_SAFE_CALL( same_xw_.transpose_xywz_to_xzwy( stage1_4d_, stage2_4d_, transpose_mode_4d ) );
        SCFD_SAFE_CALL( transpose_local_4d_<3, 1, 2, 0>( stage2_4d_, out ) );
        SCFD_SAFE_CALL( base_fft_.template exec<complex_array4_t, complex_array4_t>( "forward_xy", out, out ) );
    }

    void backward_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab>, complex_array4_t &in,
        real_array4_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::backward_4d_slab_slab" );
        SCFD_SAFE_CALL( base_fft_.template exec<complex_array4_t, complex_array4_t>( "inverse_xy", in, in ) );
        SCFD_SAFE_CALL( transpose_local_4d_<3, 1, 2, 0>( in, stage2_4d_ ) );
        SCFD_SAFE_CALL( same_xw_.transpose_xzwy_to_xywz( stage2_4d_, stage1_4d_, transpose_mode_4d ) );
        SCFD_SAFE_CALL( transpose_local_4d_<0, 1, 3, 2>( stage1_4d_, stage0_4d_ ) );
        SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex4_t, real_array4_t>( "inverse_zw", stage0_4d_, out ) );
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
            idx_4d_t( 0, 0, 0, 0 ), idx_4d_t(
                                        static_cast<int>( sz[0] ), static_cast<int>( sz[1] ), static_cast<int>( sz[2] ),
                                        static_cast<int>( sz[3] )
                                    )
        );
    }

    template <class ArrayIn, class ArrayOut>
    void reorder_x_stage_( const ArrayIn &in, ArrayOut &out )
    {
        SCFD_SAFE_CALL( for_each_3d_(
            detail::fftm_copy_same_indices_functor<ArrayIn, ArrayOut, idx_3d_t>{ in, out }, make_range_( out )
        ) );
        SCFD_SAFE_CALL( for_each_3d_.wait() );
    }

    template <class ArrayIn, class ArrayOut>
    void reorder_stage1_to_xfast_( const ArrayIn &in, ArrayOut &out )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::reorder_stage1_to_xfast" );
        SCFD_SAFE_CALL( reorder_x_stage_( in, out ) );
    }

    template <class ArrayIn, class ArrayOut>
    void reorder_stage1_from_xfast_( const ArrayIn &in, ArrayOut &out )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::reorder_stage1_from_xfast" );
        SCFD_SAFE_CALL( reorder_x_stage_( in, out ) );
    }

    template <int DstAxis0, int DstAxis1, int DstAxis2, int DstAxis3, class ArrayIn, class ArrayOut>
    void transpose_local_4d_( const ArrayIn &in, ArrayOut &out )
    {
        const auto        in_size     = in.size_nd();
        const auto        out_size    = out.size_nd();
        const std::size_t expected[4] = {
            static_cast<std::size_t>( in_size[DstAxis0] ), static_cast<std::size_t>( in_size[DstAxis1] ),
            static_cast<std::size_t>( in_size[DstAxis2] ), static_cast<std::size_t>( in_size[DstAxis3] ) };

        if ( static_cast<std::size_t>( out_size[0] ) != expected[0] ||
             static_cast<std::size_t>( out_size[1] ) != expected[1] ||
             static_cast<std::size_t>( out_size[2] ) != expected[2] ||
             static_cast<std::size_t>( out_size[3] ) != expected[3] )
        {
            throw std::logic_error( "fftm local 4D transpose destination shape mismatch" );
        }

        SCFD_SAFE_CALL( for_each_4d_(
            detail::direct_transpose_4d_functor<idx_4d_t, ArrayIn, ArrayOut, DstAxis0, DstAxis1, DstAxis2, DstAxis3>(
                in, out
            ),
            make_range_4d_( in )
        ) );
        SCFD_SAFE_CALL( for_each_4d_.wait() );
    }

    void init_shared_stage0_xfft_3d_(
        std::size_t stage0_d0, std::size_t stage0_d1, std::size_t stage0_d2, std::size_t xfft_d0, std::size_t xfft_d1,
        std::size_t xfft_d2, std::size_t stage1_xfast_d0 = 0, std::size_t stage1_xfast_d1 = 0,
        std::size_t stage1_xfast_d2 = 0
    )
    {
        const std::size_t stage0_size       = stage0_d0 * stage0_d1 * stage0_d2;
        const std::size_t xfft_size         = xfft_d0 * xfft_d1 * xfft_d2;
        const std::size_t stage1_xfast_size = stage1_xfast_d0 * stage1_xfast_d1 * stage1_xfast_d2;

        SCFD_SAFE_CALL( scratch_stage0_xfft_3d_.init( std::max( std::max( stage0_size, xfft_size ), stage1_xfast_size ) ) );
        SCFD_SAFE_CALL(
            stage0_3d_.init_by_raw_data( scratch_stage0_xfft_3d_.raw_ptr(), stage0_d0, stage0_d1, stage0_d2 )
        );
        SCFD_SAFE_CALL( x_fft_stage_3d_.init_by_raw_data( scratch_stage0_xfft_3d_.raw_ptr(), xfft_d0, xfft_d1, xfft_d2 )
        );
        if ( stage1_xfast_size != 0 )
        {
            SCFD_SAFE_CALL( stage1_xfast_3d_.init_by_raw_data(
                scratch_stage0_xfft_3d_.raw_ptr(), stage1_xfast_d0, stage1_xfast_d1, stage1_xfast_d2
            ) );
        }
        update_memory_profile_3d_();
    }

    void init_owned_stage1_3d_( std::size_t d0, std::size_t d1, std::size_t d2 )
    {
        SCFD_SAFE_CALL( scratch_stage1_3d_.init( d0 * d1 * d2 ) );
        SCFD_SAFE_CALL( stage1_3d_.init_by_raw_data( scratch_stage1_3d_.raw_ptr(), d0, d1, d2 ) );
        update_memory_profile_3d_();
    }

    void init_shared_stage0_stage2_4d_(
        std::size_t stage0_d0, std::size_t stage0_d1, std::size_t stage0_d2, std::size_t stage0_d3,
        std::size_t stage2_d0, std::size_t stage2_d1, std::size_t stage2_d2, std::size_t stage2_d3
    )
    {
        const std::size_t stage0_size = stage0_d0 * stage0_d1 * stage0_d2 * stage0_d3;
        const std::size_t stage2_size = stage2_d0 * stage2_d1 * stage2_d2 * stage2_d3;

        SCFD_SAFE_CALL( scratch_stage0_stage2_4d_.init( std::max( stage0_size, stage2_size ) ) );
        SCFD_SAFE_CALL( stage0_4d_.init_by_raw_data(
            scratch_stage0_stage2_4d_.raw_ptr(), stage0_d0, stage0_d1, stage0_d2, stage0_d3
        ) );
        SCFD_SAFE_CALL( stage2_4d_.init_by_raw_data(
            scratch_stage0_stage2_4d_.raw_ptr(), stage2_d0, stage2_d1, stage2_d2, stage2_d3
        ) );
        update_memory_profile_4d_();
    }

    void init_owned_stage1_4d_( std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3 )
    {
        SCFD_SAFE_CALL( scratch_stage1_4d_.init( d0 * d1 * d2 * d3 ) );
        SCFD_SAFE_CALL( stage1_4d_.init_by_raw_data( scratch_stage1_4d_.raw_ptr(), d0, d1, d2, d3 ) );
        update_memory_profile_4d_();
    }

    void update_memory_profile_for_current_dim_()
    {
        if ( dim_ == 3 )
        {
            update_memory_profile_3d_();
        }
        else if ( dim_ == 4 )
        {
            update_memory_profile_4d_();
        }
        else
        {
            update_memory_profile_3d_();
            update_memory_profile_4d_();
        }
    }

    void update_memory_profile_3d_()
    {
        if ( !memory_profiler_.enabled() )
        {
            return;
        }

        memory_profiler_t *profiler = memory_profiler_.native_ptr();
        profiler->set_bytes(
            "fftm/shared_work_buffer", static_cast<memory_profiler_t::bytes_type>( shared_work_buffer_.get_work_size() )
        );
        profiler->set_bytes(
            "fftm/shared_host_work_buffer",
            static_cast<memory_profiler_t::bytes_type>( shared_host_work_buffer_.get_work_size() )
        );
        profiler->set_bytes(
            "fftm/scratch_stage0_xfft_3d",
            bytes_of_elems_( static_cast<std::size_t>( scratch_stage0_xfft_3d_.size() ), sizeof( complex ) )
        );
        profiler->set_bytes(
            "fftm/scratch_stage1_3d",
            bytes_of_elems_( static_cast<std::size_t>( scratch_stage1_3d_.size() ), sizeof( complex ) )
        );
        profiler->set_bytes( "fftm/scratch_work_hat_3d", 0 );
        profiler->set_bytes( "fftm/stage0_4d", 0 );
        profiler->set_bytes( "fftm/stage1_4d", 0 );
        profiler->set_bytes( "fftm/stage2_4d", 0 );
        profiler->set_bytes( "fftm/work_hat_4d", 0 );
        profiler->set_bytes( "fftm/scratch_stage0_stage2_4d", 0 );
        profiler->set_bytes( "fftm/scratch_stage1_4d", 0 );
    }

    void update_memory_profile_4d_()
    {
        if ( !memory_profiler_.enabled() )
        {
            return;
        }

        memory_profiler_t *profiler = memory_profiler_.native_ptr();
        profiler->set_bytes(
            "fftm/shared_work_buffer", static_cast<memory_profiler_t::bytes_type>( shared_work_buffer_.get_work_size() )
        );
        profiler->set_bytes(
            "fftm/shared_host_work_buffer",
            static_cast<memory_profiler_t::bytes_type>( shared_host_work_buffer_.get_work_size() )
        );
        profiler->set_bytes( "fftm/scratch_stage0_xfft_3d", 0 );
        profiler->set_bytes( "fftm/scratch_stage1_3d", 0 );
        profiler->set_bytes( "fftm/scratch_work_hat_3d", 0 );
        profiler->set_bytes( "fftm/stage0_4d", 0 );
        profiler->set_bytes( "fftm/stage1_4d", 0 );
        profiler->set_bytes( "fftm/stage2_4d", 0 );
        profiler->set_bytes( "fftm/work_hat_4d", 0 );
        profiler->set_bytes(
            "fftm/scratch_stage0_stage2_4d",
            bytes_of_elems_( static_cast<std::size_t>( scratch_stage0_stage2_4d_.size() ), sizeof( complex ) )
        );
        profiler->set_bytes(
            "fftm/scratch_stage1_4d",
            bytes_of_elems_( static_cast<std::size_t>( scratch_stage1_4d_.size() ), sizeof( complex ) )
        );
    }

private:
    BaseFFT                    base_fft_;
    MPIComm                    mpi_;
    Log                        log_;
    fftm_init_options          init_options_;
    optional_profiler_t        profiler_;
    optional_memory_profiler_t memory_profiler_;
    partitioning_t             partitioning_;
    shared_buffer_t            shared_work_buffer_;
    host_shared_buffer_t       shared_host_work_buffer_;
    std::size_t                shared_work_size_      = 0;
    std::size_t                shared_host_work_size_ = 0;
    same_x_t                   same_x_;
    same_z_t                   same_z_;
    same_xy_t                  same_xy_;
    same_xw_t                  same_xw_;
    same_zw_t                  same_zw_;

    bool        init_done_ = false;
    std::size_t dim_       = 0;
    int         myid_i_    = 0;
    int         myid_j_    = 0;
    int         myid_k_    = 0;

    std::size_t nx_      = 0;
    std::size_t ny_      = 0;
    std::size_t nz_      = 0;
    std::size_t nw_      = 0;
    std::size_t nz_half_ = 0;
    std::size_t nw_half_ = 0;

    partition_t input_dim_;
    partition_t half_input_dim_;
    partition_t transpose1_dim_;
    partition_t transpose2_dim_;
    partition_t transpose3_dim_;
    partition_t output_dim_;

    complex_buffer_t  scratch_stage0_xfft_3d_;
    complex_buffer_t  scratch_stage1_3d_;
    complex_buffer_t  scratch_stage0_stage2_4d_;
    complex_buffer_t  scratch_stage1_4d_;
    stage0_complex3_t stage0_3d_;
    stage1_complex3_t stage1_3d_;
    x_fft_complex3_t  x_fft_stage_3d_;
    x_fft_complex3_t  stage1_xfast_3d_;
    for_each_3d_t     for_each_3d_;

    stage0_complex4_t stage0_4d_;
    stage1_complex4_t stage1_4d_;
    stage2_complex4_t stage2_4d_;
    for_each_4d_t     for_each_4d_;
};

} // namespace fftm

#endif // __FFTM_FFTM_HPP__
