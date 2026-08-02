#ifndef __FFTM_FFTM_HPP__
#define __FFTM_FFTM_HPP__

#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/safe_call.h>
#include <scfd/utils/system_timer_event.h>

#include "fftm_options.hpp"
#include "detail/array_arrangers.h"
#include "detail/direct_transpose_4d.h"
#include "detail/memory_profile_utils.h"
#include "detail/mpi_transpose_3d.h"
#include "detail/mpi_transpose_3d_pencil_pencil_reference_owned.h"
#include "detail/mpi_transpose_3d_pencil_pencil_pipeline.h"
#include "detail/mpi_transpose_4d.h"
#include "detail/shared_workspace_buffer.h"
#include "fft_direction.h"
#include "fft_partitioning.h"
#include "profiling.h"

namespace fftm
{

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

template <class Real, class Complex, class Memory, mpi_transpose_3d_mode Mode>
struct fftm_3d_array_traits<Real, Complex, Memory, strategy_3d_pencil_slab<Mode, false>>
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
struct fftm_3d_array_traits<Real, Complex, Memory, strategy_3d_pencil_slab<Mode, true>>
{
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_210_t>;
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
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 3, Memory, scfd::arrays::custom_arranger_210_t>;
    using stage0_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_102_t>;
    // The local Y FFT reads Y-fast storage and writes an X-fast temporary.
    // This mirrors reference's pencil path and avoids explicit reorder kernels
    // before and after the second redistribution.
    using stage1_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_201_t>;
    // Optimized pencil-pencil stores the public spectral tensor in reference's
    // z-fast (x,y,z) physical order.
    using complex_array_t = scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_210_t>;
    using x_fft_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 3, Memory, scfd::arrays::custom_arranger_012_t>;
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

    using native_spectral_array4_t =
        typename detail::fftm_4d_array_traits<real, complex, memory_t, Strategy4D>::stage2_complex_array_t;

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
          pencil_pencil_pipeline_( base_fft_, same_x_, same_z_, profiler_ ),
          pencil_pencil_reference_owned_( base_fft_, mpi, log, profiler_ ), same_xy_( mpi, log ), same_xw_( mpi, log ),
          same_zw_( mpi, log )
    {
        for_each_3d_.block_size = 128;
        for_each_4d_.block_size = 128;
        same_x_.use_external_work_area();
        same_z_.use_external_work_area();
        pencil_pencil_reference_owned_.use_external_work_area();
        same_xy_.use_external_work_area();
        same_xw_.use_external_work_area();
        same_zw_.use_external_work_area();
        same_x_.use_external_host_work_area();
        same_z_.use_external_host_work_area();
        pencil_pencil_reference_owned_.use_external_host_work_area();
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

        try
        {
            release_resources();
        }
        catch ( ... )
        {
        }
    }

    void release_resources()
    {
        runtime_api_t::device_synchronize();
        pencil_pencil_reference_owned_.quiesce_for_resource_release();
        base_fft_.release();

        release_owned_array_( scratch_stage0_xfft_3d_ );
        release_owned_array_( scratch_stage0_default_z_3d_ );
        release_owned_array_( scratch_stage1_3d_ );
        release_owned_array_( scratch_stage1_xfast_3d_ );
        release_owned_array_( scratch_stage0_stage2_4d_ );
        release_owned_array_( scratch_stage1_4d_ );

        if ( reusable_workspace_resource_ )
        {
            if ( reusable_workspace_lease_active_ )
            {
                reusable_workspace_resource_->release_lease();
                reusable_workspace_lease_active_ = false;
            }
        }
        else
        {
            shared_work_buffer_.release();
            shared_host_work_buffer_.release();
        }
        init_done_ = false;
    }

    void set_reusable_workspace_resource(
        const std::shared_ptr<detail::reusable_workspace_resource> &resource
    )
    {
        if ( init_done_ || reusable_workspace_lease_active_ )
            throw std::logic_error( "FFTM reusable workspace must be set before plan initialization" );
        reusable_workspace_resource_ = resource;
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
        if ( strategy_family_3d == transform_strategy_3d::pencil_pencil && strategy_3d_optimized_layout )
        {
            return std::make_tuple(
                output_dim_.size_x[0], output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_]
            );
        }
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
        return get_local_spectral_sizes_4d( fftm_4d_spectral_layout::public_yzwx );
    }

    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> get_local_spectral_sizes_4d() const
    {
        ensure_dimension_( 4 );
        return get_local_spectral_sizes_4d( spectral_layout_4d() );
    }

    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
    get_local_spectral_sizes_4d( fftm_4d_spectral_layout layout ) const
    {
        ensure_dimension_( 4 );
        if ( layout == fftm_4d_spectral_layout::native_xzwy )
        {
            ensure_native_spectral_layout_supported_();
            if ( strategy_family_4d == transform_strategy_4d_mpi::pencil_pencil )
            {
                return std::make_tuple(
                    output_dim_.size_x[0], output_dim_.size_z[myid_j_], output_dim_.size_w[myid_k_],
                    output_dim_.size_y[myid_i_]
                );
            }
            return std::make_tuple(
                transpose2_dim_.size_x[myid_i_], transpose2_dim_.size_z[myid_j_],
                transpose2_dim_.size_w[myid_k_], transpose2_dim_.size_y[0]
            );
        }
        return std::make_tuple(
            output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_], output_dim_.size_w[myid_k_], output_dim_.size_x[0]
        );
    }

    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> get_local_spectral_starts_4d() const
    {
        ensure_dimension_( 4 );
        return get_local_spectral_starts_4d( spectral_layout_4d() );
    }

    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
    get_local_spectral_starts_4d( fftm_4d_spectral_layout layout ) const
    {
        ensure_dimension_( 4 );
        if ( layout == fftm_4d_spectral_layout::native_xzwy )
        {
            ensure_native_spectral_layout_supported_();
            if ( strategy_family_4d == transform_strategy_4d_mpi::pencil_pencil )
            {
                return std::make_tuple(
                    output_dim_.start_x[0], output_dim_.start_z[myid_j_], output_dim_.start_w[myid_k_],
                    output_dim_.start_y[myid_i_]
                );
            }
            return std::make_tuple(
                transpose2_dim_.start_x[myid_i_], transpose2_dim_.start_z[myid_j_],
                transpose2_dim_.start_w[myid_k_], transpose2_dim_.start_y[0]
            );
        }
        return std::make_tuple(
            output_dim_.start_y[myid_i_], output_dim_.start_z[myid_j_], output_dim_.start_w[myid_k_],
            output_dim_.start_x[0]
        );
    }

    fftm_4d_spectral_layout spectral_layout_4d() const
    {
        ensure_dimension_( 4 );
        return init_options_.spectral_layout_4d;
    }

    bool uses_native_spectral_layout_4d() const
    {
        return spectral_layout_4d() == fftm_4d_spectral_layout::native_xzwy;
    }

    native_spectral_array4_t make_native_spectral_view_4d( complex_array_t<4> &array ) const
    {
        ensure_dimension_( 4 );
        ensure_native_spectral_layout_supported_();
        return native_spectral_view_4d_( array );
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

    void forward_native_spectral_4d( const real_array_t<4> &in, native_spectral_array4_t &out )
    {
        ensure_dimension_( 4 );
        ensure_native_spectral_layout_supported_();
        SCFD_SAFE_CALL( forward_native_spectral_4d_( strategy_family_4d_tag(), in, out ) );
    }

    // Destructive inverse: the spectral input is reused as a work buffer to avoid a full-domain copy.
    void backward( complex_array_t<4> &in, real_array_t<4> &out )
    {
        ensure_dimension_( 4 );
        SCFD_SAFE_CALL( backward_4d_( strategy_family_4d_tag(), in, out ) );
    }

    void backward_native_spectral_4d( native_spectral_array4_t &in, real_array_t<4> &out )
    {
        ensure_dimension_( 4 );
        ensure_native_spectral_layout_supported_();
        SCFD_SAFE_CALL( backward_native_spectral_4d_( strategy_family_4d_tag(), in, out ) );
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

    void begin_native_stage_timing_iteration( int iteration )
    {
        if ( dim_ == 4 )
        {
            native_4d_stage_timing_iteration_ = iteration;
            native_4d_stage_timings_.clear();
            native_4d_stage_timing_active_ = init_options_.diagnostics.enable_native_stage_timers;
        }
        pencil_pencil_reference_owned_.begin_native_stage_timing_iteration( iteration );
    }

    void end_native_stage_timing_iteration()
    {
        native_4d_stage_timing_active_ = false;
        pencil_pencil_reference_owned_.end_native_stage_timing_iteration();
    }

    const std::vector<fftm_native_stage_timing> &native_stage_timings() const
    {
        if ( dim_ == 4 )
        {
            return native_4d_stage_timings_;
        }
        return pencil_pencil_reference_owned_.native_stage_timings();
    }

    bool native_opt0_compact_y_workarea_effective() const
    {
        return native_opt0_compact_y_workarea_enabled_();
    }

    bool native_opt0_auto_compact_y_workarea_effective() const
    {
        return native_opt0_auto_compact_y_workarea_;
    }

    bool native_opt0_default_z_scratch_aliased() const
    {
        return native_opt0_default_z_scratch_aliased_;
    }

    bool slab_4d_native_work_area_alias_effective() const
    {
        return slab_4d_native_work_area_alias_effective_;
    }

    std::size_t shared_fft_work_size_bytes() const
    {
        return shared_fft_work_size_;
    }

    std::size_t shared_transpose_work_size_bytes() const
    {
        return shared_transpose_work_size_;
    }

    std::size_t shared_same_xw_work_size_bytes() const
    {
        return shared_same_xw_work_size_;
    }

    std::size_t shared_native_4d_sliced_work_size_bytes() const
    {
        return shared_native_4d_sliced_work_size_;
    }

    std::size_t shared_stage1_alias_size_bytes() const
    {
        return shared_stage1_alias_size_;
    }

    std::size_t shared_work_size_bytes() const
    {
        return shared_work_size_;
    }

    void run_native_opt0_y_same_buffer_microbench(
        complex_array_t<3> &spectral_workspace, std::size_t iterations, std::size_t warmup
    )
    {
        ensure_dimension_( 3 );
        if ( strategy_family_3d != transform_strategy_3d::pencil_pencil )
        {
            throw std::logic_error( "native opt0 Y microbenchmark requires pencil-pencil strategy" );
        }
        if ( !native_opt0_default_z_layout_enabled_() )
        {
            throw std::logic_error( "native opt0 Y microbenchmark requires --use-native-opt0-default-z-layout" );
        }
        const bool have_named_y_plans =
            !native_opt0_forward_y_plan_names_.empty() && !native_opt0_inverse_y_plan_names_.empty();
        const bool have_bundle_y_plans =
            native_opt0_y_plan_bundle_enabled_() &&
            native_opt0_y_plan_array_bundle_ != base_fft_type::invalid_c2c_plan_array_id();
        if ( native_opt0_y_plan_offsets_.empty() || ( !have_named_y_plans && !have_bundle_y_plans ) )
        {
            throw std::logic_error( "native opt0 Y microbenchmark requires initialized native opt0 Y plans" );
        }

        if ( native_opt0_reference_y_buffer_topology_enabled_() )
        {
            SCFD_SAFE_CALL( bind_native_opt0_reference_forward_views_( spectral_workspace ) );
        }
        else
        {
            SCFD_SAFE_CALL( bind_native_opt0_zfast_3d_views_() );
        }

        const auto diagnostic_raw_bundle = make_native_opt0_y_diagnostic_raw_bundle_();
        complex   *production_backward_y_output_ptr =
            native_opt0_reference_y_buffer_topology_enabled_() ? native_opt0_reference_backward_y_output_ptr_() : nullptr;
        pencil_pencil_reference_owned_.dump_local_fft_plan_descriptors();
        SCFD_SAFE_CALL( pencil_pencil_reference_owned_.run_native_opt0_y_same_buffer_microbench(
            base_fft_, native_opt0_forward_y_plan_names_, native_opt0_inverse_y_plan_names_,
            native_opt0_y_plan_offsets_, stage1_zfast_3d_, stage1_yfft_zfast_3d_, diagnostic_raw_bundle,
            production_backward_y_output_ptr, iterations, warmup
        ) );
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
    using zfast_complex3_t =
        scfd::arrays::tensor_array_nd<complex, 3, memory_t, scfd::arrays::custom_arranger_210_t>;
    using output_zfast_complex3_t = zfast_complex3_t;
    using stage1_complex3_t = typename std::conditional<
        strategy_family_3d == transform_strategy_3d::pencil_pencil, typename traits_3d_t::stage1_complex_array_t,
        complex_array3_t>::type;

    using real_array4_t     = typename traits_4d_t::real_array_t;
    using stage0_complex4_t = typename traits_4d_t::stage0_complex_array_t;
    using stage1_complex4_t = typename traits_4d_t::stage1_complex_array_t;
    using stage2_complex4_t = typename traits_4d_t::stage2_complex_array_t;
    using complex_array4_t  = typename traits_4d_t::complex_array_t;
    using slab_wz_communication_complex4_t =
        scfd::arrays::tensor_array_nd<complex, 4, memory_t, scfd::arrays::custom_arranger_0321_t>;

    using partitioning_t             = fft_partitioning<MPIComm>;
    using profiler_t                 = fftm_profiler;
    using optional_profiler_t        = optional_profiler<profiler_t>;
    using optional_memory_profiler_t = optional_memory_profiler<memory_profiler_t>;
    using same_x_t                   = ::fftm::mpi_transpose_3d<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_z_t                   = ::fftm::mpi_transpose_3d_same_z<complex, Backend, MPIComm, Log, runtime_api_t>;
    using pencil_pencil_pipeline_t =
        ::fftm::detail::mpi_transpose_3d_pencil_pencil_pipeline<BaseFFT, same_x_t, same_z_t, optional_profiler_t>;
    using pencil_pencil_reference_owned_t = ::fftm::detail::mpi_transpose_3d_pencil_pencil_reference_owned<
        BaseFFT, complex, Backend, MPIComm, Log, runtime_api_t, optional_profiler_t>;
    using pencil_pencil_plan_state_t = ::fftm::detail::mpi_transpose_3d_pencil_pencil_plan_state;
    using same_xy_t        = ::fftm::detail::mpi_transpose_4d_same_xy<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_xw_t        = ::fftm::detail::mpi_transpose_4d_same_xw<complex, Backend, MPIComm, Log, runtime_api_t>;
    using same_zw_t        = ::fftm::detail::mpi_transpose_4d_same_zw<complex, Backend, MPIComm, Log, runtime_api_t>;
    using for_each_3d_t    = typename Backend::template for_each_nd_type<3, int>;
    using for_each_4d_t    = typename Backend::template for_each_nd_type<4, int>;
    using complex_buffer_t = scfd::arrays::array_nd<complex, 1, memory_t>;
    using shared_buffer_t = detail::shared_workspace_buffer<memory_t>;
    using host_shared_buffer_t = detail::shared_workspace_buffer<typename memory_t::host_memory_type>;

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
        void *base = reusable_workspace_resource_ ? reusable_workspace_resource_->device_ptr()
                                                   : shared_work_buffer_.naive_ptr();
        return static_cast<void *>( static_cast<char *>( base ) + offset );
    }

    void *shared_host_work_ptr_( std::size_t offset = 0 ) const
    {
        void *base = reusable_workspace_resource_ ? reusable_workspace_resource_->host_ptr()
                                                   : shared_host_work_buffer_.naive_ptr();
        return static_cast<void *>( static_cast<char *>( base ) + offset );
    }

    std::size_t add_reserve_bytes_( std::size_t allocation_bytes, std::size_t reserve_bytes, const char *what ) const
    {
        if ( reserve_bytes > std::numeric_limits<std::size_t>::max() - allocation_bytes )
        {
            std::ostringstream ss;
            ss << what << " overflows size_t";
            throw std::overflow_error( ss.str() );
        }
        return allocation_bytes + reserve_bytes;
    }

    std::size_t native_opt0_reference_domain_bytes_() const
    {
        return pencil_pencil_reference_owned_.native_opt0_reference_domain_size_bytes();
    }

    complex *native_opt0_reference_slot_ptr_( std::size_t slot ) const
    {
        const std::size_t offset = slot * native_opt0_reference_domain_bytes_();
        return reinterpret_cast<complex *>( shared_work_ptr_( offset ) );
    }

    complex *native_opt0_reference_backward_y_output_ptr_() const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        return reinterpret_cast<complex *>( shared_work_ptr_( native_opt0_reference_backward_y_output_offset_bytes_() ) );
#else
        return native_opt0_reference_slot_ptr_( 0 );
#endif
    }

    std::size_t native_opt0_reference_workspace_slot_offset_( bool compact_y_workarea ) const
    {
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( native_opt0_reference_y_buffer_topology_enabled_() && compact_y_workarea )
            return 1;
        return 3;
#else
        (void)compact_y_workarea;
        return 1;
#endif
    }

    std::size_t native_opt0_y_reference_workspace_total_bytes_( bool compact_y_workarea ) const
    {
        if ( !native_opt0_default_z_layout_enabled_() )
            return 0;
        const std::size_t domain_bytes = native_opt0_reference_domain_bytes_();
        const std::size_t slot_offset = native_opt0_reference_workspace_slot_offset_( compact_y_workarea );
        if ( domain_bytes != 0 && slot_offset > std::numeric_limits<std::size_t>::max() / domain_bytes )
            throw std::overflow_error( "native opt0 Y reference workarea offset overflows size_t" );
        const std::size_t reference_work_offset = slot_offset * domain_bytes;
        const std::size_t fft_work_span = native_opt0_reference_fft_work_span_();
        if ( reference_work_offset > std::numeric_limits<std::size_t>::max() - fft_work_span )
            throw std::overflow_error( "native opt0 Y reference workspace total overflows size_t" );
        const std::size_t total = reference_work_offset + fft_work_span;
        return align_up_( total, 256 );
    }

    std::size_t native_opt0_y_reference_workspace_total_bytes_() const
    {
        return native_opt0_y_reference_workspace_total_bytes_( native_opt0_compact_y_workarea_enabled_() );
    }

    std::size_t native_opt0_reference_backward_y_output_offset_bytes_( bool compact_y_workarea ) const
    {
        const std::size_t work_total = native_opt0_y_reference_workspace_total_bytes_( compact_y_workarea );
#ifdef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( compact_y_workarea )
        {
            const std::size_t domain_bytes = native_opt0_reference_domain_bytes_();
            if ( domain_bytes > std::numeric_limits<std::size_t>::max() / 3 )
                throw std::overflow_error( "native opt0 compact backward-Y output offset overflows size_t" );
            return std::max( work_total, 3 * domain_bytes );
        }
#endif
        return work_total;
    }

    std::size_t native_opt0_reference_backward_y_output_offset_bytes_() const
    {
        return native_opt0_reference_backward_y_output_offset_bytes_( native_opt0_compact_y_workarea_enabled_() );
    }

    std::size_t native_opt0_y_parallel_work_size_( bool compact_y_workarea ) const
    {
        std::size_t total = native_opt0_y_reference_workspace_total_bytes_( compact_y_workarea );
        if ( native_opt0_reference_y_buffer_topology_enabled_() )
        {
            const std::size_t backward_output_offset =
                native_opt0_reference_backward_y_output_offset_bytes_( compact_y_workarea );
            const std::size_t domain_bytes = native_opt0_reference_domain_bytes_();
            if ( backward_output_offset > std::numeric_limits<std::size_t>::max() - domain_bytes )
                throw std::overflow_error( "native opt0 compact Y workspace size overflows size_t" );
            total = std::max( total, backward_output_offset + domain_bytes );
        }
        return align_up_( total, 256 );
    }

    std::size_t native_opt0_y_parallel_work_size_() const
    {
        return native_opt0_y_parallel_work_size_( native_opt0_compact_y_workarea_enabled_() );
    }

    std::size_t native_opt0_y_plan_work_stride_() const
    {
        if ( native_opt0_y_plan_bundle_enabled_() )
        {
            if ( native_opt0_y_plan_array_bundle_ == base_fft_type::invalid_c2c_plan_array_id() )
                return 0;
            return base_fft_.c2c_plan_array_work_size( native_opt0_y_plan_array_bundle_ );
        }
        if ( native_opt0_forward_y_plan_names_.size() != native_opt0_inverse_y_plan_names_.size() )
            throw std::logic_error( "native opt0 y-plan forward/inverse count mismatch" );

        std::size_t work_stride = 0;
        for ( std::size_t i = 0; i < native_opt0_forward_y_plan_names_.size(); ++i )
        {
            const std::size_t forward_work = base_fft_.work_size( native_opt0_forward_y_plan_names_[i] );
            const std::size_t inverse_work = base_fft_.work_size( native_opt0_inverse_y_plan_names_[i] );
            work_stride = std::max( work_stride, std::max( forward_work, inverse_work ) );
        }
        return work_stride;
    }

    std::size_t native_opt0_reference_fft_work_span_() const
    {
        const std::size_t y_work_stride = native_opt0_y_plan_work_stride_();
        const std::size_t y_plan_count  = native_opt0_y_plan_bundle_enabled_() &&
                                          native_opt0_y_plan_array_bundle_ != base_fft_type::invalid_c2c_plan_array_id()
                                              ? base_fft_.c2c_plan_array_size( native_opt0_y_plan_array_bundle_ )
                                              : native_opt0_forward_y_plan_names_.size();
        if ( y_work_stride != 0 && y_plan_count > std::numeric_limits<std::size_t>::max() / y_work_stride )
            throw std::overflow_error( "native opt0 y-plan workspace span overflows size_t" );

        std::size_t span = native_opt0_reference_domain_bytes_();
        span = std::max( span, base_fft_.work_size( "forward_z" ) );
        span = std::max( span, base_fft_.work_size( "inverse_z" ) );
        span = std::max( span, base_fft_.work_size( "forward_x" ) );
        span = std::max( span, base_fft_.work_size( "inverse_x" ) );
        span = std::max( span, y_plan_count * y_work_stride );
        return span;
    }

    void bind_native_opt0_y_parallel_fft_plans_()
    {
        if ( !native_opt0_default_z_layout_enabled_() )
            return;
        const std::size_t reference_work_offset =
            pencil_pencil_reference_owned_.native_opt0_y_reference_work_base_offset_bytes();
        void *reference_work_area = shared_work_ptr_( reference_work_offset );
        base_fft_.set_work_area( "forward_z", reference_work_area );
        base_fft_.set_work_area( "inverse_z", reference_work_area );
        base_fft_.set_work_area( "forward_x", reference_work_area );
        base_fft_.set_work_area( "inverse_x", reference_work_area );
        if ( native_opt0_y_plan_bundle_enabled_() )
        {
            SCFD_SAFE_CALL( pencil_pencil_reference_owned_.bind_native_opt0_y_plan_array_bundle(
                base_fft_, native_opt0_y_plan_array_bundle_, native_opt0_y_plan_work_stride_()
            ) );
        }
        else
        {
            SCFD_SAFE_CALL( pencil_pencil_reference_owned_.bind_native_opt0_y_parallel_fft_plans(
                base_fft_, native_opt0_forward_y_plan_names_, native_opt0_inverse_y_plan_names_,
                native_opt0_y_plan_work_stride_()
            ) );
        }
    }

    std::size_t c2c_plan_array_work_span_( typename base_fft_type::c2c_plan_array_id_t bundle_id ) const
    {
        if ( bundle_id == base_fft_type::invalid_c2c_plan_array_id() )
            return 0;
        const std::size_t count  = base_fft_.c2c_plan_array_size( bundle_id );
        const std::size_t stride = base_fft_.c2c_plan_array_work_size( bundle_id );
        if ( count != 0 && stride > std::numeric_limits<std::size_t>::max() / count )
            throw std::overflow_error( "C2C plan-array work span overflows size_t" );
        return count * stride;
    }

    std::size_t native_4d_degen_wz_sliced_z_fft_work_size_() const
    {
        if ( !pencil_4d_degenerate_wz_sliced_z_fft_enabled_() )
            return 0;
        return std::max(
            c2c_plan_array_work_span_( native_4d_degen_wz_forward_z_plan_array_bundle_ ),
            c2c_plan_array_work_span_( native_4d_degen_wz_inverse_z_plan_array_bundle_ )
        );
    }

    void bind_native_4d_degen_wz_sliced_z_fft_plans_()
    {
        if ( !pencil_4d_degenerate_wz_sliced_z_fft_enabled_() )
            return;
        if ( native_4d_degen_wz_forward_z_plan_array_bundle_ == base_fft_type::invalid_c2c_plan_array_id() ||
             native_4d_degen_wz_inverse_z_plan_array_bundle_ == base_fft_type::invalid_c2c_plan_array_id() )
        {
            throw std::logic_error( "fftm 4D degenerate WZ sliced-Z plan array was not created" );
        }
        base_fft_.bind_c2c_plan_array_work_areas(
            native_4d_degen_wz_forward_z_plan_array_bundle_, shared_work_ptr_(),
            base_fft_.c2c_plan_array_work_size( native_4d_degen_wz_forward_z_plan_array_bundle_ )
        );
        base_fft_.bind_c2c_plan_array_work_areas(
            native_4d_degen_wz_inverse_z_plan_array_bundle_, shared_work_ptr_(),
            base_fft_.c2c_plan_array_work_size( native_4d_degen_wz_inverse_z_plan_array_bundle_ )
        );
    }

    void forward_4d_degen_wz_sliced_z_fft_()
    {
        base_fft_.exec_c2c_plan_array_direction_dual_offsets_no_sync(
            native_4d_degen_wz_forward_z_plan_array_bundle_, ::fftm::direction::C2CF,
            native_4d_degen_wz_forward_z_input_offsets_, native_4d_degen_wz_forward_z_output_offsets_,
            stage0_4d_.raw_ptr(), stage1_4d_.raw_ptr()
        );
        base_fft_.synchronize_c2c_plan_array_streams( native_4d_degen_wz_forward_z_plan_array_bundle_ );
    }

    void inverse_4d_degen_wz_sliced_z_fft_()
    {
        base_fft_.exec_c2c_plan_array_direction_dual_offsets_no_sync(
            native_4d_degen_wz_inverse_z_plan_array_bundle_, ::fftm::direction::C2CB,
            native_4d_degen_wz_inverse_z_input_offsets_, native_4d_degen_wz_inverse_z_output_offsets_,
            stage1_4d_.raw_ptr(), stage0_4d_.raw_ptr()
        );
        base_fft_.synchronize_c2c_plan_array_streams( native_4d_degen_wz_inverse_z_plan_array_bundle_ );
    }

    void release_native_opt0_default_z_scratch_if_aliased_()
    {
        if ( !native_opt0_default_z_scratch_aliasing_enabled_() || scratch_stage0_default_z_3d_.is_free() )
            return;

        const std::size_t released_bytes =
            bytes_of_elems_( static_cast<std::size_t>( scratch_stage0_default_z_3d_.size() ), sizeof( complex ) );
        scratch_stage0_default_z_3d_.free();
        native_opt0_default_z_scratch_aliased_ = true;

        if ( init_options_.reporting.verbose )
        {
            std::ostringstream ss;
            ss << "FFTM native opt0 default-Z scratch aliased into shared workspace: released="
               << bytes_to_mib_( static_cast<memory_profile_bytes_t>( released_bytes ) )
               << " MiB";
            log_.info( ss.str() );
        }
    }

    void maybe_enable_native_opt0_auto_compact_y_workarea_(
        std::size_t fft_work_size, std::size_t transpose_work_size
    )
    {
        if ( !native_opt0_reference_y_buffer_topology_enabled_() ||
             !init_options_.execution.use_native_opt0_memory_feasibility_guard ||
             init_options_.execution.use_native_opt0_compact_y_workarea ||
             native_opt0_auto_compact_y_workarea_ )
        {
            return;
        }

        const auto mem_info = runtime_api_t::get_device_memory_info();
        if ( !mem_info.free_bytes_known )
        {
            return;
        }

        const std::size_t reserve_bytes = init_options_.execution.native_opt0_memory_feasibility_reserve_bytes;
        const std::size_t noncompact_workspace_bytes = align_up_(
            std::max( std::max( fft_work_size, transpose_work_size ), native_opt0_y_parallel_work_size_( false ) ),
            256
        );
        const std::size_t compact_workspace_bytes = align_up_(
            std::max( std::max( fft_work_size, transpose_work_size ), native_opt0_y_parallel_work_size_( true ) ),
            256
        );
        const std::size_t noncompact_required_bytes =
            add_reserve_bytes_( noncompact_workspace_bytes, reserve_bytes, "native opt0 noncompact requirement" );
        const std::size_t compact_required_bytes =
            add_reserve_bytes_( compact_workspace_bytes, reserve_bytes, "native opt0 compact requirement" );

        const std::size_t reusable_capacity = reusable_workspace_resource_
                                                  ? reusable_workspace_resource_->device_capacity_bytes()
                                                  : 0;
        if ( reusable_capacity >= noncompact_workspace_bytes )
            return;
        const bool reusable_compact_fit =
            reusable_capacity != 0 && reusable_capacity >= compact_workspace_bytes;

        if ( !reusable_compact_fit &&
             ( mem_info.free_bytes >= noncompact_required_bytes || mem_info.free_bytes < compact_required_bytes ) )
        {
            return;
        }

        native_opt0_auto_compact_y_workarea_ = true;
        pencil_pencil_reference_owned_.set_native_opt0_compact_y_workarea_enabled( true );

        if ( init_options_.reporting.verbose )
        {
            std::ostringstream ss;
            ss << "FFTM native opt0 auto-compact Y workspace enabled: free="
               << bytes_to_mib_( static_cast<memory_profile_bytes_t>( mem_info.free_bytes ) )
               << " MiB, noncompact_required="
               << bytes_to_mib_( static_cast<memory_profile_bytes_t>( noncompact_required_bytes ) )
               << " MiB, compact_required="
               << bytes_to_mib_( static_cast<memory_profile_bytes_t>( compact_required_bytes ) )
               << " MiB, saved="
               << bytes_to_mib_(
                      static_cast<memory_profile_bytes_t>( noncompact_workspace_bytes - compact_workspace_bytes )
                  )
               << " MiB";
            log_.info( ss.str() );
        }
    }

    void activate_shared_work_area_()
    {
        ensure_slab_4d_native_work_area_alias_supported_();

        const std::size_t fft_work_size        = base_fft_.activate_work_size();
        const bool        only_same_xw_4d_work = only_same_xw_4d_transpose_workspace_active_();
        const std::size_t transpose_work_size  = only_same_xw_4d_work
                                                      ? same_xw_.get_work_size_bytes()
                                                      : std::max(
                                                            std::max(
                                                                std::max(
                                                                    same_x_.get_work_size_bytes(),
                                                                    same_z_.get_work_size_bytes()
                                                                ),
                                                                pencil_pencil_reference_owned_.get_work_size_bytes()
                                                            ),
                                                            std::max(
                                                                same_xy_.get_work_size_bytes(),
                                                                std::max(
                                                                    same_xw_.get_work_size_bytes(),
                                                                    same_zw_.get_work_size_bytes()
                                                                )
                                                            )
                                                        );
        maybe_enable_native_opt0_auto_compact_y_workarea_( fft_work_size, transpose_work_size );

        const std::size_t native_opt0_y_work_size = native_opt0_y_parallel_work_size_();
        const std::size_t native_4d_degen_wz_work_size = native_4d_degen_wz_sliced_z_fft_work_size_();
        shared_fft_work_size_             = fft_work_size;
        shared_transpose_work_size_       = transpose_work_size;
        shared_same_xw_work_size_         = same_xw_.get_work_size_bytes();
        shared_native_4d_sliced_work_size_ = native_4d_degen_wz_work_size;
        shared_stage1_alias_size_ = stage1_4d_alias_pending_
                                        ? stage1_4d_size_ * sizeof( complex )
                                        : 0;
        shared_work_size_ = align_up_(
            std::max(
                std::max( fft_work_size, transpose_work_size ),
                std::max(
                    std::max( native_opt0_y_work_size, native_4d_degen_wz_work_size ),
                    shared_stage1_alias_size_
                )
            ),
            256
        );
        preflight_native_opt0_shared_work_allocation_( shared_work_size_ );
        if ( reusable_workspace_resource_ )
        {
            reusable_workspace_resource_->acquire( detail::workspace_memory_type_token<memory_t>() );
            reusable_workspace_lease_active_ = true;
            reusable_workspace_resource_->require_device_size_bytes( shared_work_size_ );
            reusable_workspace_resource_->activate_device();
        }
        else
        {
            shared_work_buffer_.require_size_bytes( shared_work_size_ );
            shared_work_buffer_.activate();
        }

        const std::size_t transpose_host_work_size = only_same_xw_4d_work
                                                          ? same_xw_.get_host_work_size_bytes()
                                                          : std::max(
                                                                std::max(
                                                                    std::max(
                                                                        same_x_.get_host_work_size_bytes(),
                                                                        same_z_.get_host_work_size_bytes()
                                                                    ),
                                                                    pencil_pencil_reference_owned_.get_host_work_size_bytes()
                                                                ),
                                                                std::max(
                                                                    same_xy_.get_host_work_size_bytes(),
                                                                    std::max(
                                                                        same_xw_.get_host_work_size_bytes(),
                                                                        same_zw_.get_host_work_size_bytes()
                                                                    )
                                                                )
                                                            );
        shared_host_work_size_ = align_up_( transpose_host_work_size, 256 );
        if ( reusable_workspace_resource_ )
        {
            reusable_workspace_resource_->require_host_size_bytes( shared_host_work_size_ );
            reusable_workspace_resource_->activate_host();
        }
        else
        {
            shared_host_work_buffer_.require_size_bytes( shared_host_work_size_ );
            if ( shared_host_work_size_ != 0 )
                shared_host_work_buffer_.activate();
        }

        void *work_area = shared_work_ptr_();
        if ( stage1_4d_alias_pending_ )
        {
            SCFD_SAFE_CALL( stage1_4d_.init_by_raw_data(
                reinterpret_cast<complex *>( work_area ), stage1_4d_d0_, stage1_4d_d1_, stage1_4d_d2_,
                stage1_4d_d3_
            ) );
            slab_4d_native_work_area_alias_effective_ = true;
        }
        base_fft_.set_external_work_area( work_area );
        same_x_.set_external_work_area( work_area );
        same_z_.set_external_work_area( work_area );
        pencil_pencil_reference_owned_.set_external_work_area( work_area );
        if ( !only_same_xw_4d_work )
        {
            same_xy_.set_external_work_area( work_area );
        }
        same_xw_.set_external_work_area( work_area );
        if ( !only_same_xw_4d_work )
        {
            same_zw_.set_external_work_area( work_area );
        }
        release_native_opt0_default_z_scratch_if_aliased_();
        SCFD_SAFE_CALL( bind_native_opt0_y_parallel_fft_plans_() );
        SCFD_SAFE_CALL( bind_native_4d_degen_wz_sliced_z_fft_plans_() );
        log_shared_workspace_sizes_();
        if ( shared_host_work_size_ != 0 )
        {
            void *host_work_area = shared_host_work_ptr_();
            same_x_.set_external_host_work_area( host_work_area );
            same_z_.set_external_host_work_area( host_work_area );
            pencil_pencil_reference_owned_.set_external_host_work_area( host_work_area );
            if ( !only_same_xw_4d_work )
            {
                same_xy_.set_external_host_work_area( host_work_area );
            }
            same_xw_.set_external_host_work_area( host_work_area );
            if ( !only_same_xw_4d_work )
            {
                same_zw_.set_external_host_work_area( host_work_area );
            }
        }

        update_memory_profile_for_current_dim_();
    }

    bool only_same_xw_4d_transpose_workspace_active_() const
    {
        if ( dim_ != 4 || !init_options_.execution.use_4d_native_xw_direct_layout )
        {
            return false;
        }
        if ( strategy_family_4d == transform_strategy_4d_mpi::slab_slab )
        {
            return true;
        }
        return strategy_family_4d == transform_strategy_4d_mpi::pencil_pencil &&
               pencil_4d_degenerate_same_xw_native_enabled_();
    }

    void ensure_slab_4d_native_work_area_alias_supported_() const
    {
        if ( !init_options_.execution.use_4d_slab_native_work_area_alias )
            return;
        if ( dim_ != 4 || strategy_family_4d != transform_strategy_4d_mpi::slab_slab ||
             !slab_4d_native_xw_native_spectral_layout_enabled_() ||
             !init_options_.execution.use_4d_native_xw_direct_layout )
        {
            throw std::logic_error(
                "FFTM 4D slab work-area alias requires slab-slab, native-xzwy spectral layout, "
                "native slab XW transpose, and native XW direct layout."
            );
        }
        if ( !stage1_4d_alias_pending_ )
        {
            throw std::logic_error( "FFTM 4D slab work-area alias was not prepared during stage initialization." );
        }
    }

    void log_shared_workspace_sizes_()
    {
        if ( dim_ != 4 || !init_options_.reporting.verbose )
            return;
        std::ostringstream ss;
        ss << "FFTM 4D shared workspace: fft=" << shared_fft_work_size_ << " B ("
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( shared_fft_work_size_ ) )
           << " MiB), transpose=" << shared_transpose_work_size_ << " B ("
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( shared_transpose_work_size_ ) )
           << " MiB), same_xw=" << shared_same_xw_work_size_ << " B ("
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( shared_same_xw_work_size_ ) )
           << " MiB), sliced_z=" << shared_native_4d_sliced_work_size_ << " B ("
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( shared_native_4d_sliced_work_size_ ) )
           << " MiB), stage1_alias=" << shared_stage1_alias_size_ << " B ("
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( shared_stage1_alias_size_ ) )
           << " MiB), allocated=" << shared_work_size_ << " B ("
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( shared_work_size_ ) )
           << " MiB), slab_alias_effective="
           << ( slab_4d_native_work_area_alias_effective_ ? 1 : 0 );
        log_.info( ss.str() );
    }

    void preflight_native_opt0_shared_work_allocation_( std::size_t allocation_bytes ) const
    {
        if ( !native_opt0_reference_y_buffer_topology_enabled_() ||
             !init_options_.execution.use_native_opt0_memory_feasibility_guard )
        {
            return;
        }

        const auto mem_info = runtime_api_t::get_device_memory_info();
        if ( !mem_info.free_bytes_known )
        {
            return;
        }

        const std::size_t reserve_bytes = init_options_.execution.native_opt0_memory_feasibility_reserve_bytes;
        const std::size_t reusable_capacity = reusable_workspace_resource_
                                                  ? reusable_workspace_resource_->device_capacity_bytes()
                                                  : 0;
        if ( reusable_capacity != 0 && reusable_capacity < allocation_bytes )
        {
            throw std::runtime_error(
                "FFTM reusable device workspace is smaller than the requested native opt0 workspace; "
                "evaluate the largest-memory candidate first"
            );
        }
        const std::size_t allocation_needed = reusable_capacity >= allocation_bytes ? 0 : allocation_bytes;
        const std::size_t required_bytes =
            add_reserve_bytes_( allocation_needed, reserve_bytes, "native opt0 memory feasibility guard" );
        if ( mem_info.free_bytes >= required_bytes )
        {
            return;
        }

        const std::size_t domain_bytes = native_opt0_reference_domain_bytes_();
        const std::size_t reference_work_bytes = native_opt0_y_reference_workspace_total_bytes_();
        const std::size_t compact_workspace_bytes = native_opt0_y_parallel_work_size_( true );
        const std::size_t compact_required_bytes =
            compact_workspace_bytes > std::numeric_limits<std::size_t>::max() - reserve_bytes
                ? std::numeric_limits<std::size_t>::max()
                : compact_workspace_bytes + reserve_bytes;

        std::ostringstream ss;
        ss << "FFTM native opt0 memory feasibility guard failed before allocating shared workspace: free="
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( mem_info.free_bytes ) )
           << " MiB, required="
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( required_bytes ) )
           << " MiB, shared_workspace="
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( allocation_bytes ) )
           << " MiB, reusable_capacity="
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( reusable_capacity ) )
           << " MiB, reserve="
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( reserve_bytes ) )
           << " MiB, domain_slot="
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( domain_bytes ) )
           << " MiB, y_reference_workspace="
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( reference_work_bytes ) )
           << " MiB, compact_shared_workspace="
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( compact_workspace_bytes ) )
           << " MiB, compact_required="
           << bytes_to_mib_( static_cast<memory_profile_bytes_t>( compact_required_bytes ) )
           << " MiB. ";
        if ( !native_opt0_compact_y_workarea_enabled_() && mem_info.free_bytes >= compact_required_bytes )
        {
            ss << "Compact native opt0 should fit this device memory. If this message is reached, auto-compact "
                  "was not applicable for the current init sequence; explicitly set "
                  "FFTM_USE_NATIVE_OPT0_COMPACT_Y_WORKAREA=1 to force the compact layout.";
        }
        else
        {
            ss << "Use pencil layout opt1 for this GPU count/size, or explicitly try compact native opt0 only when "
                  "it fits the target device memory.";
        }
        throw std::runtime_error( ss.str() );
    }

    static bool is_reference_owned_pencil_pipeline_( fftm_3d_pencil_pipeline pipeline )
    {
        return pipeline == fftm_3d_pencil_pipeline::reference || pipeline == fftm_3d_pencil_pipeline::reference_parity;
    }

    static bool is_reference_parity_pencil_pipeline_( fftm_3d_pencil_pipeline pipeline )
    {
        return pipeline == fftm_3d_pencil_pipeline::reference_parity;
    }

    ::fftm::detail::pencil_pencil_plan_layout native_pencil_pencil_plan_layout_() const
    {
        if ( init_options_.pencil_layout_3d == fftm_3d_pencil_layout::opt1 )
        {
            return ::fftm::detail::pencil_pencil_plan_layout::opt1;
        }
        if ( init_options_.pencil_layout_3d == fftm_3d_pencil_layout::opt0 )
        {
            return ::fftm::detail::pencil_pencil_plan_layout::opt0;
        }
        if ( init_options_.pencil_layout_3d == fftm_3d_pencil_layout::auto_select && strategy_3d_optimized_layout )
        {
            const processor_grid pg = partitioning_.get_process_grid();
            return pg.p1 > pg.p2 ? ::fftm::detail::pencil_pencil_plan_layout::opt0
                                  : ::fftm::detail::pencil_pencil_plan_layout::opt1;
        }
        return ::fftm::detail::pencil_pencil_plan_layout::opt0;
    }

    void init_native_pencil_pencil_plan_state_()
    {
        processor_grid grid = partitioning_.get_process_grid();
        global_sizes   sizes;
        sizes.init( nx_, ny_, nz_ );
        pencil_pencil_plan_state_.init(
            grid, sizes, mpi_.myid, mpi_.num_procs, native_pencil_pencil_plan_layout_(), sizeof( complex )
        );

        if ( pencil_pencil_plan_state_.rank_i() != myid_i_ || pencil_pencil_plan_state_.rank_j() != myid_j_ )
            throw std::logic_error( "fftm native pencil plan-state rank mismatch" );
    }

    static fftm_init_options normalize_init_options_( fftm_init_options options )
    {
        if ( options.diagnostics.use_4d_slab_native_xw_native_spectral_layout )
        {
            options.spectral_layout_4d = fftm_4d_spectral_layout::native_xzwy;
        }
        if ( native_opt0_diagnostic_variant_requested_( options ) &&
             !options.diagnostics.allow_native_opt0_diagnostic_variants )
        {
            throw std::logic_error(
                "FFTM native opt0 diagnostic Y executor variants require "
                "fftm_init_options::diagnostics.allow_native_opt0_diagnostic_variants=true. "
                "Use the production native opt0 path without reference/bundle/context diagnostic flags."
            );
        }
        if ( options.diagnostics.use_fft_exec_no_sync && !options.diagnostics.allow_native_opt0_diagnostic_variants )
        {
            throw std::logic_error(
                "FFTM generic FFT exec no-sync is diagnostic-only. It can race later MPI/copy stages unless every "
                "consumer has an explicit stage-boundary synchronization. Use the native opt0 Y no-sync path for "
                "production, or set fftm_init_options::diagnostics.allow_native_opt0_diagnostic_variants=true "
                "for diagnostics."
            );
        }
        if ( options.diagnostics.use_3d_deferred_send_completion &&
             ( strategy_family_3d != transform_strategy_3d::pencil_pencil ||
               transpose_mode_3d != mpi_transpose_3d_mode::p2p_waitany ||
               !strategy_3d_optimized_layout ||
               !options.use_optimized ||
               !is_reference_parity_pencil_pipeline_( options.pencil_pipeline_3d ) ) )
        {
            throw std::logic_error(
                "FFTM deferred 3D send completion requires the optimized native pencil-pencil strategy, "
                "p2p-waitany, and reference-parity."
            );
        }
        if ( options.diagnostics.use_4d_slab_native_xw_layout_stage &&
             options.spectral_layout_4d == fftm_4d_spectral_layout::native_xzwy )
        {
            throw std::logic_error(
                "FFTM 4D slab native XW layout-stage and native-spectral-layout are mutually exclusive."
            );
        }
        if ( options.execution.use_4d_pencil_degenerate_local_transposes &&
             options.spectral_layout_4d != fftm_4d_spectral_layout::native_xzwy )
        {
            throw std::logic_error(
                "FFTM 4D degenerate-pencil local transposes require spectral_layout_4d=native_xzwy."
            );
        }
        if ( options.execution.use_4d_pencil_degenerate_same_xw_native &&
             !options.execution.use_4d_pencil_degenerate_local_transposes )
        {
            throw std::logic_error(
                "FFTM 4D degenerate-pencil native same_xw requires use_4d_pencil_degenerate_local_transposes=true."
            );
        }
        if ( options.execution.use_4d_native_xw_direct_layout &&
             options.spectral_layout_4d != fftm_4d_spectral_layout::native_xzwy )
        {
            throw std::logic_error(
                "FFTM 4D native XW direct layout requires spectral_layout_4d=native_xzwy."
            );
        }
        if ( options.execution.use_4d_native_xw_chunked_transport && !options.execution.use_4d_native_xw_direct_layout )
        {
            throw std::logic_error(
                "FFTM 4D native XW chunked transport requires use_4d_native_xw_direct_layout=true."
            );
        }
        if ( options.execution.use_4d_native_xw_chunked_transport && options.execution.native_xw_chunk_bytes == 0 )
        {
            throw std::logic_error( "FFTM 4D native XW chunked transport requires a nonzero chunk size." );
        }
        if ( options.execution.native_xw_chunk_window != 0 && !options.execution.use_4d_native_xw_chunked_transport )
        {
            throw std::logic_error( "FFTM 4D native XW chunk window requires chunked transport." );
        }
        if ( options.execution.use_4d_native_xw_compact_staging &&
             ( !options.execution.use_4d_native_xw_direct_layout ||
               !options.execution.use_4d_native_xw_chunked_transport ||
               options.execution.native_xw_chunk_window == 0 ) )
        {
            throw std::logic_error(
                "FFTM 4D native XW compact staging requires direct layout, chunked transport, and a bounded chunk window."
            );
        }
        if ( options.execution.use_4d_slab_native_wz_communication_layout &&
             ( !options.execution.use_4d_slab_native_xw_transpose ||
               options.spectral_layout_4d != fftm_4d_spectral_layout::native_xzwy ||
               !options.execution.use_4d_native_xw_direct_layout ||
               !options.execution.use_4d_native_xw_chunked_transport ||
               options.execution.native_xw_chunk_window == 0 ) )
        {
            throw std::logic_error(
                "FFTM 4D slab WZ communication layout requires native slab XW, native-xzwy spectral layout, "
                "direct layout, chunked transport, and a bounded chunk window."
            );
        }
        if ( options.execution.slab_native_wz_plan_concurrency == 0 )
        {
            throw std::logic_error( "FFTM 4D slab WZ plan concurrency must be positive." );
        }
        if ( ( options.execution.slab_native_wz_plan_concurrency != 1 ||
               options.execution.use_4d_slab_native_wz_ready_pipeline ) &&
             !options.execution.use_4d_slab_native_wz_communication_layout )
        {
            throw std::logic_error(
                "FFTM 4D slab WZ plan concurrency/pipeline options require the WZ communication layout."
            );
        }
        if ( is_reference_parity_pencil_pipeline_( options.pencil_pipeline_3d ) )
        {
            options.execution.use_direct_backward_receive = false;
            options.execution.direct_p2p_cuda_aware       = true;
            options.diagnostics.use_p2p_send_thread         = false;
            options.execution.use_p2p_byte_transfer       = true;
            options.execution.use_persistent_p2p          = false;
            options.diagnostics.use_ready_p2p_send          = false;
        }
        return options;
    }

    static bool native_opt0_diagnostic_variant_requested_( const fftm_init_options &options )
    {
        return options.diagnostics.use_native_opt0_reference_y_plan_lifecycle ||
               options.diagnostics.use_native_opt0_reference_y_plan_bundle ||
               options.diagnostics.use_native_opt0_raw_y_plan_bundle ||
               options.diagnostics.use_native_opt0_y_plan_bundle_stream_first ||
               options.diagnostics.use_native_opt0_raw_y_plan_bundle_reference_streams ||
               options.diagnostics.use_native_opt0_reference_local_plan_context;
    }

    static void native_4d_stage_timing_callback_( void *context, const char *stage, double ms )
    {
        static_cast<fftm *>( context )->record_native_4d_stage_timing_( stage, ms );
    }

    void configure_profiling_( const fftm_init_options &options )
    {
        init_options_ = normalize_init_options_( options );
        native_opt0_auto_compact_y_workarea_ = false;
        native_opt0_default_z_scratch_aliased_ = false;
        if ( !init_options_.reporting.profiling_key.empty() )
        {
            profiler_.enable( init_options_.reporting.profiling_key );
        }
        else
        {
            profiler_.disable();
        }
        configure_memory_profiling_( init_options_ );
        slab_4d_native_work_area_alias_effective_ = false;
        stage1_4d_alias_pending_                   = false;
        shared_fft_work_size_                     = 0;
        shared_transpose_work_size_               = 0;
        shared_same_xw_work_size_                 = 0;
        shared_native_4d_sliced_work_size_         = 0;
        shared_stage1_alias_size_                  = 0;
        same_x_.set_profiler( profiler_.native_ptr() );
        same_z_.set_profiler( profiler_.native_ptr() );
        pencil_pencil_reference_owned_.set_profiler( profiler_.native_ptr() );
        same_xy_.set_profiler( profiler_.native_ptr() );
        same_xw_.set_profiler( profiler_.native_ptr() );
        same_zw_.set_profiler( profiler_.native_ptr() );
        same_xy_.set_stage_timing_callback(
            this, &fftm::native_4d_stage_timing_callback_, "4d/transpose/same_xy"
        );
        same_xw_.set_stage_timing_callback(
            this, &fftm::native_4d_stage_timing_callback_, "4d/transpose/same_xw"
        );
        same_xw_.set_slab_native_batched_peer_kernels_enabled(
            init_options_.diagnostics.use_4d_slab_native_xw_batched_peer_kernels
        );
        same_xw_.set_slab_native_tensor_coalesced_kernels_enabled(
            init_options_.diagnostics.use_4d_slab_native_xw_tensor_coalesced_kernels
        );
        same_xw_.set_slab_native_vector4_kernels_enabled(
            init_options_.diagnostics.use_4d_slab_native_xw_vector4_kernels
        );
        same_xw_.set_slab_native_tiled_kernels_enabled(
            init_options_.diagnostics.use_4d_slab_native_xw_tiled_kernels
        );
        same_xw_.set_native_xzwy_direct_layout_enabled( init_options_.execution.use_4d_native_xw_direct_layout );
        same_xw_.set_native_xzwy_chunked_transport(
            init_options_.execution.use_4d_native_xw_chunked_transport, init_options_.execution.native_xw_chunk_bytes,
            init_options_.execution.native_xw_chunk_window
        );
        same_xw_.set_native_xzwy_compact_staging_enabled(
            init_options_.execution.use_4d_native_xw_compact_staging
        );
        same_xw_.set_slab_native_wz_communication_layout_enabled(
            slab_4d_native_wz_communication_layout_enabled_()
        );
        same_zw_.set_peer_paired_p2p_enabled( init_options_.diagnostics.use_4d_pencil_same_zw_peer_paired );
        same_zw_.set_native_message_layout_enabled( pencil_4d_native_zw_message_layout_enabled_() );
        same_zw_.set_stage_timing_callback(
            this, &fftm::native_4d_stage_timing_callback_, "4d/transpose/same_zw"
        );
        same_x_.set_direct_transfer_options(
            init_options_.execution.use_direct_backward_receive, init_options_.execution.direct_p2p_cuda_aware
        );
        same_z_.set_direct_transfer_options(
            init_options_.execution.use_direct_backward_receive, init_options_.execution.direct_p2p_cuda_aware
        );
        pencil_pencil_reference_owned_.set_direct_transfer_options(
            init_options_.execution.use_direct_backward_receive, init_options_.execution.direct_p2p_cuda_aware
        );
        same_x_.set_p2p_send_thread_enabled( init_options_.diagnostics.use_p2p_send_thread );
        same_z_.set_p2p_send_thread_enabled( init_options_.diagnostics.use_p2p_send_thread );
        pencil_pencil_reference_owned_.set_p2p_send_thread_enabled( init_options_.diagnostics.use_p2p_send_thread );
        same_x_.set_p2p_byte_transfer_enabled( init_options_.execution.use_p2p_byte_transfer );
        same_z_.set_p2p_byte_transfer_enabled( init_options_.execution.use_p2p_byte_transfer );
        pencil_pencil_reference_owned_.set_p2p_byte_transfer_enabled( init_options_.execution.use_p2p_byte_transfer );
        same_x_.set_persistent_p2p_enabled( init_options_.execution.use_persistent_p2p );
        same_z_.set_persistent_p2p_enabled( init_options_.execution.use_persistent_p2p );
	        pencil_pencil_reference_owned_.set_persistent_p2p_enabled( init_options_.execution.use_persistent_p2p );
	        pencil_pencil_reference_owned_.set_ready_p2p_send_enabled( init_options_.diagnostics.use_ready_p2p_send );
	        pencil_pencil_reference_owned_.set_schedule_dump_enabled( init_options_.diagnostics.print_pencil_schedule );
        pencil_pencil_reference_owned_.set_direct_forward_byte_receive_enabled(
            init_options_.diagnostics.use_direct_forward_byte_receive
        );
        pencil_pencil_reference_owned_.set_stable_forward_byte_send_buffer_enabled(
            init_options_.diagnostics.use_stable_forward_byte_send_buffer
        );
        pencil_pencil_reference_owned_.set_ready_stable_forward_byte_send_buffer_enabled(
            init_options_.diagnostics.use_ready_stable_forward_byte_send_buffer
        );
        pencil_pencil_reference_owned_.set_contiguous_forward_byte_send_enabled(
            init_options_.diagnostics.use_contiguous_forward_byte_send
        );
        pencil_pencil_reference_owned_.set_physical_forward_peer_exchange_enabled(
            init_options_.diagnostics.use_physical_forward_peer_exchange
        );
        pencil_pencil_reference_owned_.set_contiguous_forward_send_mode(
            init_options_.diagnostics.contiguous_forward_send_mode
        );
        pencil_pencil_reference_owned_.set_contiguous_forward_send_chunk_bytes(
            init_options_.diagnostics.contiguous_forward_send_chunk_bytes
        );
        pencil_pencil_reference_owned_.set_large_count_p2p_transport( init_options_.execution.large_count_p2p_transport );
        pencil_pencil_reference_owned_.set_large_count_datatype_cache_enabled(
            init_options_.diagnostics.use_large_count_datatype_cache
        );
        pencil_pencil_reference_owned_.set_native_backward_second_peer_loop_enabled(
            init_options_.diagnostics.use_native_backward_second_peer_loop
        );
        pencil_pencil_reference_owned_.set_deferred_send_completion_enabled(
            init_options_.diagnostics.use_3d_deferred_send_completion
        );
        pencil_pencil_reference_owned_.set_local_fft_diagnostics(
            init_options_.diagnostics.enable_local_fft_diagnostics, init_options_.diagnostics.local_fft_diagnostics_directory,
            init_options_.diagnostics.local_fft_diagnostics_label
        );
        pencil_pencil_reference_owned_.set_native_stage_timers_enabled( init_options_.diagnostics.enable_native_stage_timers );
        pencil_pencil_reference_owned_.set_native_opt0_shared_y_plan_handles_enabled(
            init_options_.execution.use_native_opt0_shared_y_plan_handles
        );
        pencil_pencil_reference_owned_.set_native_opt0_y_group_device_sync_enabled(
            init_options_.execution.use_native_opt0_y_group_device_sync
        );
        pencil_pencil_reference_owned_.set_native_opt0_y_no_sync_exec_enabled(
            init_options_.execution.use_native_opt0_y_no_sync_exec
        );
        pencil_pencil_reference_owned_.set_native_opt0_raw_y_plan_array_executor_enabled(
            init_options_.execution.use_native_opt0_raw_y_plan_array_executor
        );
        pencil_pencil_reference_owned_.set_native_opt0_reference_y_plan_lifecycle_enabled(
            init_options_.diagnostics.use_native_opt0_reference_y_plan_lifecycle
        );
        pencil_pencil_reference_owned_.set_native_opt0_reference_y_plan_bundle_enabled(
            init_options_.diagnostics.use_native_opt0_reference_y_plan_bundle
        );
        pencil_pencil_reference_owned_.set_native_opt0_raw_y_plan_bundle_enabled(
            init_options_.diagnostics.use_native_opt0_raw_y_plan_bundle
        );
        pencil_pencil_reference_owned_.set_native_opt0_y_plan_bundle_stream_first_enabled(
            init_options_.diagnostics.use_native_opt0_y_plan_bundle_stream_first
        );
        pencil_pencil_reference_owned_.set_native_opt0_raw_y_plan_bundle_reference_streams_enabled(
            init_options_.diagnostics.use_native_opt0_raw_y_plan_bundle_reference_streams
        );
        pencil_pencil_reference_owned_.set_native_opt0_reference_local_plan_context_enabled(
            init_options_.diagnostics.use_native_opt0_reference_local_plan_context
        );
        const auto selected_pencil_layout = native_pencil_pencil_plan_layout_();
        pencil_pencil_reference_owned_.set_pencil_layout_selector(
            selected_pencil_layout == ::fftm::detail::pencil_pencil_plan_layout::opt0 ? 1 : 2
        );
        pencil_pencil_reference_owned_.set_reference_parity_enabled(
            is_reference_parity_pencil_pipeline_( init_options_.pencil_pipeline_3d )
        );
    }

    void configure_memory_profiling_( const fftm_init_options &options )
    {
        if ( !options.reporting.memory_profiling_key.empty() )
        {
            memory_profiler_.enable( options.reporting.memory_profiling_key );
        }
        else
        {
            memory_profiler_.disable();
        }

        base_fft_.set_memory_profiler( memory_profiler_.native_ptr(), "fftm/base_fft" );
        base_fft_.set_hot_exec_no_sync_enabled( init_options_.diagnostics.use_fft_exec_no_sync );
        same_x_.set_memory_profiler( memory_profiler_.native_ptr(), "fftm/transpose_3d_same_x" );
        same_z_.set_memory_profiler( memory_profiler_.native_ptr(), "fftm/transpose_3d_same_z" );
        pencil_pencil_reference_owned_.set_memory_profiler(
            memory_profiler_.native_ptr(), "fftm/transpose_3d_pencil_pencil_reference_owned"
        );
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
        if ( init_options_.reporting.print_profile_summary_on_destroy )
        {
            profiler_.log_print( log_ );
        }
        if ( init_options_.reporting.print_profile_totals_on_destroy )
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
        if ( init_options_.reporting.print_memory_profile_on_destroy )
        {
            log_memory_profile_summary_();
        }
        if ( init_options_.reporting.print_memory_totals_on_destroy )
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

    void record_native_4d_stage_timing_( const char *stage, double ms )
    {
        if ( !init_options_.diagnostics.enable_native_stage_timers || !native_4d_stage_timing_active_ )
        {
            return;
        }

        fftm_native_stage_timing row;
        row.stage = stage;
        row.ms    = ms;
        native_4d_stage_timings_.push_back( row );
    }

    template <class Fn>
    void time_native_4d_stage_( const char *stage, Fn fn )
    {
        if ( !init_options_.diagnostics.enable_native_stage_timers || !native_4d_stage_timing_active_ )
        {
            fn();
            return;
        }

        runtime_api_t::device_synchronize();
        scfd::utils::system_timer_event begin, end;
        begin.record();
        fn();
        runtime_api_t::device_synchronize();
        end.record();
        record_native_4d_stage_timing_( stage, end.elapsed_time( begin ) );
    }

    template <class Fn>
    void time_native_4d_stage_( const std::string &stage, Fn fn )
    {
        time_native_4d_stage_( stage.c_str(), fn );
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

    void ensure_native_spectral_layout_supported_() const
    {
        if ( strategy_family_4d == transform_strategy_4d_mpi::slab_slab && !slab_4d_native_xw_transpose_enabled_() )
        {
            throw std::logic_error( "FFTM 4D native-xzwy spectral layout requires native slab XW transpose." );
        }
        if ( strategy_family_4d != transform_strategy_4d_mpi::slab_slab &&
             strategy_family_4d != transform_strategy_4d_mpi::pencil_pencil )
        {
            throw std::logic_error( "FFTM 4D native-xzwy spectral layout is not supported by this strategy." );
        }
    }

    void ensure_4d_native_xw_direct_layout_supported_() const
    {
        if ( !init_options_.execution.use_4d_native_xw_direct_layout )
            return;
        if ( transpose_mode_4d != mpi_transpose_3d_mode::p2p_waitany )
        {
            throw std::logic_error( "FFTM 4D native XW direct layout requires p2p-waitany mode." );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        throw std::logic_error( "FFTM 4D native XW direct layout requires CUDA-aware MPI." );
#endif
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

    bool native_opt0_default_z_layout_enabled_() const
    {
        if ( !init_options_.execution.use_native_opt0_default_z_layout )
        {
            return false;
        }
        if ( strategy_family_3d != transform_strategy_3d::pencil_pencil || !strategy_3d_optimized_layout )
        {
            return false;
        }
        if ( !init_options_.use_optimized ||
             native_pencil_pencil_plan_layout_() != ::fftm::detail::pencil_pencil_plan_layout::opt0 )
        {
            return false;
        }
        if ( !is_reference_owned_pencil_pipeline_( init_options_.pencil_pipeline_3d ) )
        {
            return false;
        }
        const processor_grid pg = partitioning_.get_process_grid();
        return pg.p1 > 1;
    }

    bool native_opt0_reference_y_buffer_topology_enabled_() const
    {
        return native_opt0_default_z_layout_enabled_() &&
               init_options_.execution.use_native_opt0_reference_y_buffer_topology;
    }

    bool native_opt0_reference_scratch_aliasing_enabled_() const
    {
        return native_opt0_reference_y_buffer_topology_enabled_();
    }

    bool native_opt0_default_z_scratch_aliasing_enabled_() const
    {
        return native_opt0_reference_y_buffer_topology_enabled_();
    }

    bool slab_4d_native_xw_transpose_enabled_() const
    {
        return init_options_.execution.use_4d_slab_native_xw_transpose;
    }

    bool slab_4d_native_xw_layout_stage_enabled_() const
    {
        return slab_4d_native_xw_transpose_enabled_() && init_options_.diagnostics.use_4d_slab_native_xw_layout_stage;
    }

    bool slab_4d_native_xw_native_spectral_layout_enabled_() const
    {
        return slab_4d_native_xw_transpose_enabled_() &&
               strategy_family_4d == transform_strategy_4d_mpi::slab_slab &&
               init_options_.spectral_layout_4d == fftm_4d_spectral_layout::native_xzwy;
    }

    bool slab_4d_native_wz_communication_layout_enabled_() const
    {
        return slab_4d_native_xw_native_spectral_layout_enabled_() &&
               init_options_.execution.use_4d_slab_native_wz_communication_layout;
    }

    std::size_t slab_4d_native_wz_plan_concurrency_() const
    {
        if ( !slab_4d_native_wz_communication_layout_enabled_() )
            return 1;
        return std::max<std::size_t>(
            1, std::min<std::size_t>( init_options_.execution.slab_native_wz_plan_concurrency,
                                     input_dim_.size_y[myid_j_] )
        );
    }

    bool slab_4d_native_wz_ready_pipeline_enabled_() const
    {
        return slab_4d_native_wz_communication_layout_enabled_() &&
               init_options_.execution.use_4d_slab_native_wz_ready_pipeline;
    }

    bool pencil_4d_native_zw_native_spectral_layout_enabled_() const
    {
        return strategy_family_4d == transform_strategy_4d_mpi::pencil_pencil &&
               init_options_.spectral_layout_4d == fftm_4d_spectral_layout::native_xzwy;
    }

    bool pencil_4d_native_zw_message_layout_enabled_() const
    {
        return pencil_4d_native_zw_native_spectral_layout_enabled_() &&
               init_options_.execution.use_4d_pencil_same_zw_native_layout;
    }

    bool pencil_4d_degenerate_xw_slab_path_enabled_() const
    {
        if ( !pencil_4d_native_zw_native_spectral_layout_enabled_() ||
             init_options_.execution.use_4d_pencil_degenerate_local_transposes ||
             !init_options_.diagnostics.use_4d_pencil_degenerate_xw_slab_path || !slab_4d_native_xw_transpose_enabled_() )
        {
            return false;
        }
        const processor_grid pg = partitioning_.get_process_grid();
        return pg.p1 == 1 && pg.p3 == 1;
    }

    bool pencil_4d_degenerate_local_transposes_enabled_() const
    {
        if ( !pencil_4d_native_zw_native_spectral_layout_enabled_() ||
             !init_options_.execution.use_4d_pencil_degenerate_local_transposes )
        {
            return false;
        }
        const processor_grid pg = partitioning_.get_process_grid();
        return pg.p1 == 1 && pg.p3 == 1;
    }

    bool pencil_4d_degenerate_same_xw_native_enabled_() const
    {
        return pencil_4d_degenerate_local_transposes_enabled_() &&
               init_options_.execution.use_4d_pencil_degenerate_same_xw_native;
    }

    bool pencil_4d_degenerate_wz_sliced_z_fft_enabled_() const
    {
        return pencil_4d_degenerate_local_transposes_enabled_() &&
               init_options_.execution.use_4d_pencil_degenerate_wz_sliced_z_fft;
    }

    bool native_opt0_compact_y_workarea_enabled_() const
    {
        return native_opt0_reference_y_buffer_topology_enabled_() &&
               ( init_options_.execution.use_native_opt0_compact_y_workarea || native_opt0_auto_compact_y_workarea_ );
    }

    bool native_opt0_tight_y_plan_sequence_enabled_() const
    {
        return native_opt0_default_z_layout_enabled_() &&
               init_options_.execution.use_native_opt0_tight_y_plan_sequence;
    }

    bool native_opt0_y_group_device_sync_enabled_() const
    {
        return native_opt0_default_z_layout_enabled_() &&
               init_options_.execution.use_native_opt0_y_group_device_sync;
    }

    bool native_opt0_shared_y_plan_handles_enabled_() const
    {
        return native_opt0_default_z_layout_enabled_() &&
               init_options_.execution.use_native_opt0_shared_y_plan_handles;
    }

    bool native_opt0_raw_y_plan_array_executor_enabled_() const
    {
        return native_opt0_default_z_layout_enabled_() &&
               init_options_.execution.use_native_opt0_raw_y_plan_array_executor;
    }

    bool native_opt0_reference_y_plan_bundle_enabled_() const
    {
        return native_opt0_default_z_layout_enabled_() &&
               init_options_.diagnostics.use_native_opt0_reference_y_plan_bundle;
    }

    bool native_opt0_raw_y_plan_bundle_enabled_() const
    {
        return native_opt0_default_z_layout_enabled_() &&
               init_options_.diagnostics.use_native_opt0_raw_y_plan_bundle;
    }

    bool native_opt0_y_plan_bundle_enabled_() const
    {
        return native_opt0_reference_y_plan_bundle_enabled_() || native_opt0_raw_y_plan_bundle_enabled_() ||
               native_opt0_reference_local_plan_context_enabled_();
    }

    bool native_opt0_y_plan_bundle_stream_first_enabled_() const
    {
        return native_opt0_y_plan_bundle_enabled_() && init_options_.diagnostics.use_native_opt0_y_plan_bundle_stream_first;
    }

    bool native_opt0_raw_y_plan_bundle_reference_streams_enabled_() const
    {
        return native_opt0_raw_y_plan_bundle_enabled_() &&
               init_options_.diagnostics.use_native_opt0_raw_y_plan_bundle_reference_streams;
    }

    bool native_opt0_reference_local_plan_context_enabled_() const
    {
        return native_opt0_default_z_layout_enabled_() &&
               is_reference_owned_pencil_pipeline_( init_options_.pencil_pipeline_3d ) &&
               init_options_.diagnostics.use_native_opt0_reference_local_plan_context;
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
        if ( init_options_.diagnostics.enable_local_fft_diagnostics )
        {
            SCFD_SAFE_CALL( pencil_pencil_reference_owned_.dump_local_fft_plan_descriptors() );
        }
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

    void add_plan_z_r2c_z_fast_to_y_fast_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size
    )
    {
        // Optimized pencil input uses Z-fast real storage:
        //   real(x,y,z)   -> z + Nz * (y + Ny_local * x)
        // The R2C output keeps the existing Y-fast half-spectrum layout:
        //   half(x,y,kz) -> y + Ny_local * (x + X * kz)
        // This matches the reference pencil plan shape and removes the large
        // strided-Z access pattern from the first local transform.
        const long long int batch       = x_size * y_size;
        const long long int output_step = x_size * y_size;

        base_fft_.template add_plan_1D<::fftm::direction::R2C>(
            forward_name, static_cast<long long int>( nz_ ), 1, 1, static_cast<long long int>( nz_ ), output_step,
            output_step, 1, batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2R>(
            inverse_name, static_cast<long long int>( nz_ ), output_step, output_step, 1, 1, 1,
            static_cast<long long int>( nz_ ), batch
        );
    }

    void add_plan_z_r2c_default_layout_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size
    )
    {
        const long long int batch = x_size * y_size;

        base_fft_.template add_plan_1D_default<::fftm::direction::R2C>(
            forward_name, static_cast<long long int>( nz_ ), batch
        );

        base_fft_.template add_plan_1D_default<::fftm::direction::C2R>(
            inverse_name, static_cast<long long int>( nz_ ), batch
        );
    }

    void add_plan_yz_r2c_slab_optimized_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size
    )
    {
        // reference-style slab path: each local [y,z] plane is contiguous, so the
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

    void add_plan_y_c2c_y_fast_to_x_fast_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int z_size
    )
    {
        // Input stage1 layout is y + Ny * (x + X * z).
        // Forward output is x + X * (z + Z * y), so the following X FFT
        // can read unit-stride X lines. The inverse plan performs the
        // reverse layout conversion while applying inverse Y FFTs.
        const long long int batch       = x_size * z_size;
        const long long int input_dist  = static_cast<long long int>( ny_ );
        const long long int output_step = x_size * z_size;

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name, static_cast<long long int>( ny_ ), 1, 1, input_dist, output_step, output_step, 1, batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name, static_cast<long long int>( ny_ ), output_step, output_step, 1, 1, 1, input_dist, batch
        );
    }

    void add_plan_y_c2c_native_opt0_z_fast_(
        const std::string &forward_prefix, const std::string &inverse_prefix, long long int x_size,
        long long int z_size
    )
    {
        native_opt0_forward_y_plan_names_.clear();
        native_opt0_inverse_y_plan_names_.clear();
        native_opt0_y_plan_offsets_.clear();
        native_opt0_y_plan_array_bundle_ = base_fft_type::invalid_c2c_plan_array_id();

        const long long int num_plans = std::min( x_size, z_size );
        const long long int batch     = std::max( x_size, z_size );
        const long long int nembed    = ( x_size <= z_size ) ? 1 : z_size * static_cast<long long int>( ny_ );
        const long long int offset    = ( x_size <= z_size ) ? z_size * static_cast<long long int>( ny_ ) : 1;

        if ( native_opt0_y_plan_bundle_enabled_() )
        {
            native_opt0_y_plan_offsets_.reserve( static_cast<std::size_t>( num_plans ) );
            for ( long long int i = 0; i < num_plans; ++i )
            {
                native_opt0_y_plan_offsets_.push_back( static_cast<std::size_t>( i * offset ) );
            }
            if ( native_opt0_raw_y_plan_bundle_enabled_() )
            {
                if ( native_opt0_raw_y_plan_bundle_reference_streams_enabled_() )
                {
                    const auto stream_handles =
                        pencil_pencil_reference_owned_.template native_opt0_y_stream_handles<base_fft_type>(
                            native_opt0_y_plan_offsets_.size()
                        );
                    native_opt0_y_plan_array_bundle_ = base_fft_.make_raw_c2c_plan_array_1D_with_streams(
                        static_cast<long long int>( ny_ ), nembed, z_size, nembed, nembed, z_size, nembed, batch,
                        native_opt0_y_plan_offsets_, stream_handles
                    );
                }
                else
                {
                    native_opt0_y_plan_array_bundle_ = base_fft_.make_raw_c2c_plan_array_1D(
                        static_cast<long long int>( ny_ ), nembed, z_size, nembed, nembed, z_size, nembed, batch,
                        native_opt0_y_plan_offsets_
                    );
                }
            }
            else if ( native_opt0_reference_local_plan_context_enabled_() )
            {
                const auto stream_handles =
                    pencil_pencil_reference_owned_.template native_opt0_y_stream_handles<base_fft_type>(
                        native_opt0_y_plan_offsets_.size()
                    );
                const long long int z_batch =
                    x_size * static_cast<long long int>( input_dim_.size_y[myid_j_] );
                const long long int x_batch =
                    static_cast<long long int>( output_dim_.size_y[myid_i_] * output_dim_.size_z[myid_j_] );
                native_opt0_y_plan_array_bundle_ = base_fft_.make_reference_opt0_local_plan_context_1D(
                    static_cast<long long int>( nz_ ), z_batch,
                    static_cast<long long int>( ny_ ), nembed, z_size, nembed, nembed, z_size, nembed, batch,
                    static_cast<long long int>( nx_ ), 1, x_batch, 1, 1, x_batch, 1, x_batch,
                    native_opt0_y_plan_offsets_, stream_handles
                );
            }
            else
            {
                native_opt0_y_plan_array_bundle_ = base_fft_.make_c2c_plan_array_1D(
                    static_cast<long long int>( ny_ ), nembed, z_size, nembed, nembed, z_size, nembed, batch,
                    native_opt0_y_plan_offsets_
                );
            }
            return;
        }

        for ( long long int i = 0; i < num_plans; ++i )
        {
            const std::string suffix       = std::to_string( i );
            const std::string forward_name = forward_prefix + "_" + suffix;
            const std::string inverse_name = inverse_prefix + "_" + suffix;

            base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
                forward_name, static_cast<long long int>( ny_ ), nembed, z_size, nembed, nembed, z_size, nembed,
                batch
            );
            if ( native_opt0_shared_y_plan_handles_enabled_() )
            {
                native_opt0_forward_y_plan_names_.push_back( forward_name );
                native_opt0_inverse_y_plan_names_.push_back( forward_name );
                native_opt0_y_plan_offsets_.push_back( static_cast<std::size_t>( i * offset ) );
                continue;
            }
            base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
                inverse_name, static_cast<long long int>( ny_ ), nembed, z_size, nembed, nembed, z_size, nembed,
                batch
            );

            native_opt0_forward_y_plan_names_.push_back( forward_name );
            native_opt0_inverse_y_plan_names_.push_back( inverse_name );
            native_opt0_y_plan_offsets_.push_back( static_cast<std::size_t>( i * offset ) );
        }
    }

    typename base_fft_type::c2c_plan_array_id_t make_native_opt0_y_diagnostic_raw_bundle_()
    {
        if ( !native_opt0_default_z_layout_enabled_() || native_opt0_y_plan_offsets_.empty() )
        {
            return base_fft_type::invalid_c2c_plan_array_id();
        }

        const long long int x_size = static_cast<long long int>( input_dim_.size_x[myid_i_] );
        const long long int z_size = static_cast<long long int>( transpose1_dim_.size_z[myid_j_] );
        const long long int nembed = ( x_size <= z_size ) ? 1 : z_size * static_cast<long long int>( ny_ );
        const long long int batch  = std::max( x_size, z_size );
        return base_fft_.make_raw_c2c_plan_array_1D_diagnostic(
            static_cast<long long int>( ny_ ), nembed, z_size, nembed, nembed, z_size, nembed, batch,
            native_opt0_y_plan_offsets_
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

    void add_plan_x_c2c_x_fast_to_z_fast_(
        const std::string &forward_name, const std::string &inverse_name, long long int y_size, long long int z_size
    )
    {
        // Forward input is x + Nx * (z + Z * y). The existing public output
        // layout remains z + Z * (y + Y * x). This keeps compatibility while
        // making the expensive X FFT unit-stride.
        const long long int batch       = y_size * z_size;
        const long long int input_dist  = static_cast<long long int>( nx_ );
        const long long int output_step = y_size * z_size;

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            forward_name, static_cast<long long int>( nx_ ), 1, 1, input_dist, output_step, output_step, 1, batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            inverse_name, static_cast<long long int>( nx_ ), output_step, output_step, 1, 1, 1, input_dist, batch
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

    void add_plan_z_c2c_4d_xyzw_xywz_wslice_(
        long long int x_size, long long int y_size, long long int z_size, long long int w_size
    )
    {
        native_4d_degen_wz_forward_z_input_offsets_.clear();
        native_4d_degen_wz_forward_z_output_offsets_.clear();
        native_4d_degen_wz_inverse_z_input_offsets_.clear();
        native_4d_degen_wz_inverse_z_output_offsets_.clear();
        native_4d_degen_wz_forward_z_plan_array_bundle_ = base_fft_type::invalid_c2c_plan_array_id();
        native_4d_degen_wz_inverse_z_plan_array_bundle_ = base_fft_type::invalid_c2c_plan_array_id();

        const long long int xy = x_size * y_size;
        if ( xy <= 0 || z_size <= 0 || w_size <= 0 )
            throw std::logic_error( "fftm 4D degenerate WZ sliced-Z plan dimensions must be positive" );

        std::vector<std::size_t> dummy_offsets;
        dummy_offsets.reserve( static_cast<std::size_t>( w_size ) );
        native_4d_degen_wz_forward_z_input_offsets_.reserve( static_cast<std::size_t>( w_size ) );
        native_4d_degen_wz_forward_z_output_offsets_.reserve( static_cast<std::size_t>( w_size ) );
        native_4d_degen_wz_inverse_z_input_offsets_.reserve( static_cast<std::size_t>( w_size ) );
        native_4d_degen_wz_inverse_z_output_offsets_.reserve( static_cast<std::size_t>( w_size ) );

        for ( long long int w = 0; w < w_size; ++w )
        {
            dummy_offsets.push_back( 0 );
            const std::size_t xyzw_offset = static_cast<std::size_t>( xy * z_size * w );
            const std::size_t xywz_offset = static_cast<std::size_t>( xy * w );
            native_4d_degen_wz_forward_z_input_offsets_.push_back( xyzw_offset );
            native_4d_degen_wz_forward_z_output_offsets_.push_back( xywz_offset );
            native_4d_degen_wz_inverse_z_input_offsets_.push_back( xywz_offset );
            native_4d_degen_wz_inverse_z_output_offsets_.push_back( xyzw_offset );
        }

        native_4d_degen_wz_forward_z_plan_array_bundle_ = base_fft_.make_c2c_plan_array_1D(
            z_size, 1, xy, 1, 1, xy * w_size, 1, xy, dummy_offsets
        );
        native_4d_degen_wz_inverse_z_plan_array_bundle_ = base_fft_.make_c2c_plan_array_1D(
            z_size, 1, xy * w_size, 1, 1, xy, 1, xy, dummy_offsets
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

    void add_plan_zw_r2c_slab_communication_layout_( long long int x_size, long long int y_size )
    {
        const std::string forward_name = "forward_zw_slab_communication";
        const std::string inverse_name = "inverse_zw_slab_communication";
        const long long int real_x_distance = y_size * static_cast<long long int>( nz_ * nw_ );

        base_fft_.template add_plan_2D<::fftm::direction::R2C>(
            forward_name, static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ), 1, real_x_distance,
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_half_ ), x_size, 1, x_size
        );
        base_fft_.template add_plan_2D<::fftm::direction::C2R>(
            inverse_name, static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_half_ ), x_size, 1,
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ), 1, real_x_distance, x_size
        );

        std::vector<std::string> forward_names( static_cast<std::size_t>( y_size ), forward_name );
        std::vector<std::string> inverse_names( static_cast<std::size_t>( y_size ), inverse_name );
        slab_wz_communication_forward_input_offsets_.resize( static_cast<std::size_t>( y_size ) );
        slab_wz_communication_forward_output_offsets_.resize( static_cast<std::size_t>( y_size ) );
        slab_wz_communication_inverse_input_offsets_.resize( static_cast<std::size_t>( y_size ) );
        slab_wz_communication_inverse_output_offsets_.resize( static_cast<std::size_t>( y_size ) );
        for ( std::size_t y = 0; y < static_cast<std::size_t>( y_size ); ++y )
        {
            const std::size_t real_offset = y * nz_ * nw_;
            const std::size_t complex_offset = y * static_cast<std::size_t>( x_size ) * nz_ * nw_half_;
            slab_wz_communication_forward_input_offsets_[y]  = real_offset;
            slab_wz_communication_forward_output_offsets_[y] = complex_offset;
            slab_wz_communication_inverse_input_offsets_[y]  = complex_offset;
            slab_wz_communication_inverse_output_offsets_[y] = real_offset;
        }
        slab_wz_communication_forward_sequence_ = base_fft_.make_plan_sequence( forward_names );
        slab_wz_communication_inverse_sequence_ = base_fft_.make_plan_sequence( inverse_names );
        slab_wz_communication_plan_bundle_ = base_fft_.make_r2c_c2r_plan_bundle_2D(
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ), 1, real_x_distance,
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_half_ ), x_size, 1, x_size,
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_half_ ), x_size, 1,
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ), 1, real_x_distance, x_size,
            slab_4d_native_wz_plan_concurrency_()
        );
    }

    void add_plan_zw_r2c_pencil_xy_fast_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int y_size
    )
    {
        const long long int xy_stride = x_size * y_size;
        const long long int batch     = x_size * y_size;

        base_fft_.template add_plan_2D<::fftm::direction::R2C>(
            forward_name, static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ), xy_stride, 1,
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_half_ ), xy_stride, 1, batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2R>(
            inverse_name, static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_half_ ), xy_stride, 1,
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ), xy_stride, 1, batch
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

    void add_plan_xy_c2c_4d_xzwy_native_(
        const std::string &forward_name, const std::string &inverse_name, long long int x_size, long long int z_size,
        long long int w_size
    )
    {
        const long long int batch       = z_size * w_size;
        const long long int y_stride    = x_size * z_size * w_size;
        const long long int batch_dist  = x_size;

        base_fft_.template add_plan_2D<::fftm::direction::C2CF>(
            forward_name, static_cast<long long int>( ny_ ), x_size,
            static_cast<long long int>( ny_ ), y_stride, 1, batch_dist,
            static_cast<long long int>( ny_ ), y_stride, 1, batch_dist, batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2CB>(
            inverse_name, static_cast<long long int>( ny_ ), x_size,
            static_cast<long long int>( ny_ ), y_stride, 1, batch_dist,
            static_cast<long long int>( ny_ ), y_stride, 1, batch_dist, batch
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
        SCFD_SAFE_CALL( init_native_pencil_pencil_plan_state_() );

        SCFD_SAFE_CALL(
            init_strategy_3d_pencil_pencil_layout_( std::integral_constant<bool, strategy_3d_optimized_layout>() )
        );
    }

    void init_strategy_3d_pencil_pencil_layout_( std::false_type )
    {
        SCFD_SAFE_CALL( init_shared_stage0_xfft_3d_(
            input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], nz_half_, output_dim_.size_x[0],
            output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_]
        ) );
        SCFD_SAFE_CALL( init_owned_stage1_3d_( input_dim_.size_x[myid_i_], transpose1_dim_.size_z[myid_j_], ny_ ) );

        SCFD_SAFE_CALL( same_x_.init( half_input_dim_, transpose1_dim_, myid_i_, myid_j_ ) );
        SCFD_SAFE_CALL( same_z_.init( transpose1_dim_, output_dim_, myid_i_, myid_j_ ) );
        if ( is_reference_owned_pencil_pipeline_( init_options_.pencil_pipeline_3d ) )
        {
            if ( is_reference_parity_pencil_pipeline_( init_options_.pencil_pipeline_3d ) )
            {
                SCFD_SAFE_CALL( pencil_pencil_reference_owned_.init(
                    transpose_mode_3d, pencil_pencil_plan_state_, half_input_dim_, transpose1_dim_, output_dim_,
                    myid_i_, myid_j_, init_options_.execution.use_persistent_p2p
                ) );
            }
            else
            {
                SCFD_SAFE_CALL( pencil_pencil_reference_owned_.init(
                    transpose_mode_3d, half_input_dim_, transpose1_dim_, output_dim_, myid_i_, myid_j_,
                    init_options_.execution.use_persistent_p2p
                ) );
            }
        }
        else if ( init_options_.pencil_pipeline_3d != fftm_3d_pencil_pipeline::staged )
        {
            SCFD_SAFE_CALL( pencil_pencil_pipeline_.init( transpose_mode_3d, init_options_.execution.use_persistent_p2p ) );
        }
    }

    void init_strategy_3d_pencil_pencil_layout_( std::true_type )
    {
        if ( !init_options_.use_optimized || init_options_.pencil_layout_3d == fftm_3d_pencil_layout::legacy )
        {
            throw std::logic_error(
                "fftm 3D pencil-pencil optimized layout was selected at compile time; use "
                "strategy_3d_pencil_pencil<Mode, false> for the legacy runtime path."
            );
        }

        if ( !native_opt0_reference_scratch_aliasing_enabled_() )
        {
            SCFD_SAFE_CALL( init_shared_stage0_xfft_3d_(
                input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], nz_half_, output_dim_.size_x[0],
                output_dim_.size_z[myid_j_], output_dim_.size_y[myid_i_]
            ) );
        }
        if ( native_opt0_default_z_layout_enabled_() )
        {
            SCFD_SAFE_CALL( init_stage0_default_z_3d_(
                input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], nz_half_
            ) );
        }
        if ( !native_opt0_reference_scratch_aliasing_enabled_() )
        {
            SCFD_SAFE_CALL( init_owned_stage1_xfast_3d_(
                input_dim_.size_x[myid_i_], transpose1_dim_.size_z[myid_j_], ny_
            ) );
        }

        if ( native_opt0_default_z_layout_enabled_() )
        {
            if ( !native_opt0_reference_scratch_aliasing_enabled_() )
            {
                SCFD_SAFE_CALL( init_owned_stage1_3d_(
                    input_dim_.size_x[myid_i_], transpose1_dim_.size_z[myid_j_], ny_
                ) );
                SCFD_SAFE_CALL( bind_native_opt0_zfast_3d_views_() );
            }
        }
        else
        {
            const std::size_t output_capacity =
                output_dim_.size_x[0] * output_dim_.size_z[myid_j_] * output_dim_.size_y[myid_i_];
            SCFD_SAFE_CALL( init_stage1_3d_(
                input_dim_.size_x[myid_i_], transpose1_dim_.size_z[myid_j_], ny_, output_capacity
            ) );
        }

        SCFD_SAFE_CALL( same_x_.init( half_input_dim_, transpose1_dim_, myid_i_, myid_j_ ) );
        SCFD_SAFE_CALL( same_z_.init( transpose1_dim_, output_dim_, myid_i_, myid_j_ ) );
        if ( is_reference_owned_pencil_pipeline_( init_options_.pencil_pipeline_3d ) )
        {
            pencil_pencil_reference_owned_.set_native_opt0_default_z_layout_enabled(
                native_opt0_default_z_layout_enabled_()
            );
            pencil_pencil_reference_owned_.set_native_opt0_reference_y_buffer_topology_enabled(
                native_opt0_reference_y_buffer_topology_enabled_()
            );
            pencil_pencil_reference_owned_.set_native_opt0_compact_y_workarea_enabled(
                native_opt0_compact_y_workarea_enabled_()
            );
            pencil_pencil_reference_owned_.set_native_opt0_tight_y_plan_sequence_enabled(
                native_opt0_tight_y_plan_sequence_enabled_()
            );
            const auto selected_pencil_layout = native_pencil_pencil_plan_layout_();
            pencil_pencil_reference_owned_.set_pencil_layout_selector(
                selected_pencil_layout == ::fftm::detail::pencil_pencil_plan_layout::opt0 ? 1 : 2
            );
            if ( is_reference_parity_pencil_pipeline_( init_options_.pencil_pipeline_3d ) )
            {
                SCFD_SAFE_CALL( pencil_pencil_reference_owned_.init(
                    transpose_mode_3d, pencil_pencil_plan_state_, half_input_dim_, transpose1_dim_, output_dim_,
                    myid_i_, myid_j_, init_options_.execution.use_persistent_p2p
                ) );
            }
            else
            {
                SCFD_SAFE_CALL( pencil_pencil_reference_owned_.init(
                    transpose_mode_3d, half_input_dim_, transpose1_dim_, output_dim_, myid_i_, myid_j_,
                    init_options_.execution.use_persistent_p2p
                ) );
            }
        }
        else if ( init_options_.pencil_pipeline_3d != fftm_3d_pencil_pipeline::staged )
        {
            SCFD_SAFE_CALL( pencil_pencil_pipeline_.init( transpose_mode_3d, init_options_.execution.use_persistent_p2p ) );
        }
    }

    void
    init_4d_strategy_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::init_strategy_4d_pencil_pencil" );
        if ( init_options_.execution.use_4d_slab_native_wz_communication_layout )
        {
            throw std::logic_error( "FFTM 4D slab WZ communication layout is not valid for pencil-pencil" );
        }
        ensure_4d_native_xw_direct_layout_supported_();
        if ( init_options_.execution.use_4d_pencil_degenerate_local_transposes )
        {
            const processor_grid grid = partitioning_.get_process_grid();
            if ( grid.p1 != 1 || grid.p3 != 1 )
            {
                throw std::logic_error(
                    "FFTM 4D degenerate-pencil local transposes require a 1xP2x1 processor grid."
                );
            }
        }
        if ( init_options_.execution.use_4d_native_xw_direct_layout &&
             !pencil_4d_degenerate_same_xw_native_enabled_() )
        {
            throw std::logic_error(
                "FFTM 4D pencil native XW direct layout requires native-xzwy spectral layout, a 1xP2x1 "
                "grid, degenerate local transposes, and the degenerate native same_xw path."
            );
        }
        output_dim_ = transpose3_dim_;

        SCFD_SAFE_CALL( same_xw_.init( transpose1_dim_, transpose2_dim_, myid_i_, myid_j_, myid_k_ ) );
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
        SCFD_SAFE_CALL( same_zw_.init( transpose2_dim_, output_dim_, myid_i_, myid_j_, myid_k_ ) );
        update_memory_profile_4d_();
    }

    void init_4d_strategy_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::init_strategy_4d_slab_slab" );
        ensure_4d_native_xw_direct_layout_supported_();
        if ( init_options_.execution.use_4d_slab_native_wz_communication_layout &&
             transpose_mode_4d != mpi_transpose_3d_mode::p2p_waitany )
        {
            throw std::logic_error( "FFTM 4D slab WZ communication layout requires p2p-waitany" );
        }
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        if ( init_options_.execution.use_4d_slab_native_wz_communication_layout )
        {
            throw std::logic_error( "FFTM 4D slab WZ communication layout requires CUDA-aware MPI" );
        }
#endif
        if ( init_options_.execution.use_4d_native_xw_direct_layout && !slab_4d_native_xw_transpose_enabled_() )
        {
            throw std::logic_error(
                "FFTM 4D slab native XW direct layout requires use_4d_slab_native_xw_transpose=true."
            );
        }
        if ( init_options_.execution.use_4d_native_xw_direct_layout &&
             !slab_4d_native_xw_native_spectral_layout_enabled_() )
        {
            throw std::logic_error(
                "FFTM 4D slab native XW direct layout requires the native-xzwy spectral layout."
            );
        }
        require_grid_p1_is_one_();
        require_grid_p3_is_one_();

        output_dim_ = transpose3_dim_;

        SCFD_SAFE_CALL( same_xw_.init( transpose1_dim_, transpose2_dim_, myid_i_, myid_j_, myid_k_ ) );
        SCFD_SAFE_CALL( init_shared_stage0_stage2_4d_(
            input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], input_dim_.size_z[myid_k_], nw_half_,
            transpose2_dim_.size_x[myid_i_], transpose2_dim_.size_z[myid_j_], transpose2_dim_.size_w[myid_k_],
            transpose2_dim_.size_y[0]
        ) );
        SCFD_SAFE_CALL( init_owned_stage1_4d_(
            transpose1_dim_.size_x[myid_i_], transpose1_dim_.size_y[myid_j_], transpose1_dim_.size_w[myid_k_],
            transpose1_dim_.size_z[0]
        ) );

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

        SCFD_SAFE_CALL( add_plan_z_r2c_z_fast_to_y_fast_( "forward_z", "inverse_z", nx_, input_dim_.size_y[myid_j_] ) );
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
        if ( !init_options_.use_optimized || init_options_.pencil_layout_3d == fftm_3d_pencil_layout::legacy )
        {
            throw std::logic_error(
                "fftm 3D pencil-pencil optimized layout was selected at compile time; use "
                "strategy_3d_pencil_pencil<Mode, false> for the legacy runtime path."
            );
        }

        if ( is_reference_owned_pencil_pipeline_( init_options_.pencil_pipeline_3d ) &&
             partitioning_.get_process_grid().p2 == 1 && !native_opt0_default_z_layout_enabled_() )
        {
            SCFD_SAFE_CALL( add_plan_yz_r2c_slab_optimized_(
                "forward_yz_pencil_pencil_p2_degenerate", "inverse_yz_pencil_pencil_p2_degenerate",
                static_cast<long long int>( input_dim_.size_x[myid_i_] )
            ) );
            SCFD_SAFE_CALL( add_plan_x_c2c_(
                "forward_x", "inverse_x", output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_]
            ) );
            return;
        }

        if ( native_opt0_default_z_layout_enabled_() )
        {
            if ( native_opt0_reference_local_plan_context_enabled_() )
            {
                SCFD_SAFE_CALL( add_plan_y_c2c_native_opt0_z_fast_(
                    "forward_y_native_opt0_zfast", "inverse_y_native_opt0_zfast",
                    static_cast<long long int>( input_dim_.size_x[myid_i_] ),
                    static_cast<long long int>( transpose1_dim_.size_z[myid_j_] )
                ) );
                SCFD_SAFE_CALL( add_plan_z_r2c_default_layout_(
                    "forward_z", "inverse_z", input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_]
                ) );
            }
            else
            {
                SCFD_SAFE_CALL( add_plan_z_r2c_default_layout_(
                    "forward_z", "inverse_z", input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_]
                ) );
                SCFD_SAFE_CALL( add_plan_y_c2c_native_opt0_z_fast_(
                    "forward_y_native_opt0_zfast", "inverse_y_native_opt0_zfast",
                    static_cast<long long int>( input_dim_.size_x[myid_i_] ),
                    static_cast<long long int>( transpose1_dim_.size_z[myid_j_] )
                ) );
            }
            SCFD_SAFE_CALL( add_plan_x_c2c_(
                "forward_x", "inverse_x", output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_]
            ) );
            return;
        }
        else
        {
            SCFD_SAFE_CALL( add_plan_z_r2c_z_fast_to_y_fast_(
                "forward_z", "inverse_z", input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_]
            ) );
        }
        SCFD_SAFE_CALL( add_plan_y_c2c_y_fast_to_x_fast_(
            "forward_y", "inverse_y", input_dim_.size_x[myid_i_], transpose1_dim_.size_z[myid_j_]
        ) );
        SCFD_SAFE_CALL( add_plan_x_c2c_x_fast_to_z_fast_(
            "forward_x", "inverse_x", output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_]
        ) );
    }

    void add_4d_plans_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::add_plans_4d_pencil_pencil" );
        if ( pencil_4d_degenerate_xw_slab_path_enabled_() )
        {
            SCFD_SAFE_CALL( add_plan_zw_r2c_pencil_xy_fast_(
                "forward_zw_pencil_degenerate_xw_slab", "inverse_zw_pencil_degenerate_xw_slab",
                input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_]
            ) );
            SCFD_SAFE_CALL( add_plan_xy_c2c_4d_xzwy_native_(
                "forward_xy_xzwy_native", "inverse_xy_xzwy_native", transpose2_dim_.size_x[myid_i_],
                transpose2_dim_.size_z[myid_j_], transpose2_dim_.size_w[myid_k_]
            ) );
            return;
        }
        SCFD_SAFE_CALL( add_plan_w_r2c_(
            "forward_w", "inverse_w", input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], input_dim_.size_z[myid_k_]
        ) );
        SCFD_SAFE_CALL( add_plan_z_c2c_4d_(
            "forward_z", "inverse_z", transpose1_dim_.size_x[myid_i_], transpose1_dim_.size_y[myid_j_],
            transpose1_dim_.size_w[myid_k_]
        ) );
        if ( pencil_4d_degenerate_wz_sliced_z_fft_enabled_() )
        {
            SCFD_SAFE_CALL( add_plan_z_c2c_4d_xyzw_xywz_wslice_(
                input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], input_dim_.size_z[myid_k_],
                transpose1_dim_.size_w[myid_k_]
            ) );
        }
        SCFD_SAFE_CALL( add_plan_y_c2c_4d_(
            "forward_y", "inverse_y", transpose2_dim_.size_x[myid_i_], transpose2_dim_.size_z[myid_j_],
            transpose2_dim_.size_w[myid_k_]
        ) );
        SCFD_SAFE_CALL( add_plan_x_c2c_4d_(
            "forward_x", "inverse_x", output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_],
            output_dim_.size_w[myid_k_]
        ) );
        if ( pencil_4d_native_zw_native_spectral_layout_enabled_() )
        {
            SCFD_SAFE_CALL( add_plan_x_c2c_4d_(
                "forward_x_xzwy_native", "inverse_x_xzwy_native", output_dim_.size_y[myid_i_],
                output_dim_.size_z[myid_j_], output_dim_.size_w[myid_k_]
            ) );
        }
    }

    void add_4d_plans_( std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab> )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::add_plans_4d_slab_slab" );
        SCFD_SAFE_CALL(
            add_plan_zw_r2c_( "forward_zw", "inverse_zw", input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_] )
        );
        if ( slab_4d_native_wz_communication_layout_enabled_() )
        {
            SCFD_SAFE_CALL( add_plan_zw_r2c_slab_communication_layout_(
                static_cast<long long int>( input_dim_.size_x[myid_i_] ),
                static_cast<long long int>( input_dim_.size_y[myid_j_] )
            ) );
        }
        SCFD_SAFE_CALL(
            add_plan_xy_c2c_4d_( "forward_xy", "inverse_xy", output_dim_.size_z[myid_j_], output_dim_.size_w[myid_k_] )
        );
        if ( slab_4d_native_xw_native_spectral_layout_enabled_() )
        {
            SCFD_SAFE_CALL( add_plan_xy_c2c_4d_xzwy_native_(
                "forward_xy_xzwy_native", "inverse_xy_xzwy_native", transpose2_dim_.size_x[myid_i_],
                transpose2_dim_.size_z[myid_j_], transpose2_dim_.size_w[myid_k_]
            ) );
        }
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
        SCFD_SAFE_CALL( same_x_.transpose_xyz_to_xzy_optimized_layout( stage0_3d_, out, transpose_mode_3d ) );
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
        SCFD_SAFE_CALL( same_x_.transpose_xzy_to_xyz_optimized_layout( in, stage0_3d_, transpose_mode_3d ) );
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
        if ( !init_options_.use_optimized || init_options_.pencil_layout_3d == fftm_3d_pencil_layout::legacy )
        {
            throw std::logic_error(
                "fftm 3D pencil-pencil optimized layout was selected at compile time; use "
                "strategy_3d_pencil_pencil<Mode, false> for the legacy runtime path."
            );
        }

        if ( init_options_.pencil_pipeline_3d != fftm_3d_pencil_pipeline::staged )
        {
            if ( is_reference_owned_pencil_pipeline_( init_options_.pencil_pipeline_3d ) )
            {
                if ( partitioning_.get_process_grid().p2 == 1 && !native_opt0_default_z_layout_enabled_() )
                {
                    SCFD_SAFE_CALL( bind_stage1_3d_to_io_buffer_( out, "forward pencil-pencil output" ) );
                    FFTM_PROFILE_SCOPED_TIC( "fftm::forward_3d_pencil_pencil_reference_p2_degenerate" );
                    stage0_complex3_t stage0_slab_view(
                        stage0_3d_.raw_ptr(), input_dim_.size_x[myid_i_], nz_half_, ny_
                    );
                    SCFD_SAFE_CALL( base_fft_.template exec<real_array3_t, stage0_complex3_t>(
                        "forward_yz_pencil_pencil_p2_degenerate", in, stage0_slab_view
                    ) );
                    SCFD_SAFE_CALL(
                        same_z_.transpose_x_to_y_optimized_layout( stage0_slab_view, out, transpose_mode_3d )
                    );
                    SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, complex_array3_t>( "forward_x", out, out ) );
                    return;
                }
                if ( partitioning_.get_process_grid().p1 == 1 )
                {
                    SCFD_SAFE_CALL( bind_stage1_3d_to_io_buffer_( out, "forward pencil-pencil output" ) );
                    FFTM_PROFILE_SCOPED_TIC( "fftm::forward_3d_pencil_pencil_reference_p1_degenerate" );
                    SCFD_SAFE_CALL(
                        base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_z", in, stage0_3d_ )
                    );
                    SCFD_SAFE_CALL(
                        same_x_.transpose_xyz_to_xzy_optimized_layout( stage0_3d_, stage1_3d_, transpose_mode_3d )
                    );
                    SCFD_SAFE_CALL( base_fft_.template exec<stage1_complex3_t, x_fft_complex3_t>(
                        "forward_y", stage1_3d_, stage1_xfast_3d_
                    ) );
                    SCFD_SAFE_CALL( copy_stage1_xfast_to_xfft_3d_() );
                    SCFD_SAFE_CALL(
                        base_fft_.template exec<x_fft_complex3_t, complex_array3_t>( "forward_x", x_fft_stage_3d_, out )
                    );
                    return;
                }
                if ( native_opt0_default_z_layout_enabled_() )
                {
                    FFTM_PROFILE_SCOPED_TIC( "fftm::forward_3d_pencil_pencil_native_opt0_zfast" );
                    if ( native_opt0_reference_y_buffer_topology_enabled_() )
                    {
                        SCFD_SAFE_CALL( bind_native_opt0_reference_forward_views_( out ) );
                    }
                    SCFD_SAFE_CALL( pencil_pencil_reference_owned_.exec_local_fft(
                        "reference_owned/forward_z_fft", "forward_z", in, stage0_default_z_3d_
                    ) );
                    if ( native_opt0_reference_y_buffer_topology_enabled_() )
                    {
                        SCFD_SAFE_CALL( pencil_pencil_reference_owned_.template forward_native_opt0_zfast_reference_buffers<
                            zfast_complex3_t, zfast_complex3_t, zfast_complex3_t, output_zfast_complex3_t,
                            complex_array3_t>(
                            stage0_default_z_3d_, stage1_zfast_3d_, stage1_yfft_zfast_3d_, x_fft_zfast_3d_, out,
                            native_opt0_forward_y_plan_names_, native_opt0_y_plan_offsets_, transpose_mode_3d
                        ) );
                    }
                    else
                    {
                        SCFD_SAFE_CALL( pencil_pencil_reference_owned_.template forward_native_opt0_zfast<
                            zfast_complex3_t, zfast_complex3_t, zfast_complex3_t, complex_array3_t>(
                            stage0_default_z_3d_, stage1_zfast_3d_, stage1_yfft_zfast_3d_, out,
                            native_opt0_forward_y_plan_names_, native_opt0_y_plan_offsets_, transpose_mode_3d
                        ) );
                    }
                    return;
                }
                SCFD_SAFE_CALL( bind_stage1_3d_to_io_buffer_( out, "forward pencil-pencil output" ) );
                SCFD_SAFE_CALL( pencil_pencil_reference_owned_.template forward<
                    real_array3_t, stage0_complex3_t, stage1_complex3_t, x_fft_complex3_t, x_fft_complex3_t,
                    complex_array3_t>(
                    in, stage0_3d_, stage1_3d_, stage1_xfast_3d_, x_fft_stage_3d_, out, transpose_mode_3d
                ) );
            }
            else
            {
                SCFD_SAFE_CALL( bind_stage1_3d_to_io_buffer_( out, "forward pencil-pencil output" ) );
                SCFD_SAFE_CALL( pencil_pencil_pipeline_.template forward<
                    real_array3_t, stage0_complex3_t, stage1_complex3_t, x_fft_complex3_t, x_fft_complex3_t,
                    complex_array3_t>(
                    in, stage0_3d_, stage1_3d_, stage1_xfast_3d_, x_fft_stage_3d_, out, transpose_mode_3d
                ) );
            }
            return;
        }

        SCFD_SAFE_CALL( bind_stage1_3d_to_io_buffer_( out, "forward pencil-pencil output" ) );
        SCFD_SAFE_CALL( base_fft_.template exec<real_array3_t, stage0_complex3_t>( "forward_z", in, stage0_3d_ ) );
        SCFD_SAFE_CALL( same_x_.transpose_xyz_to_xzy_optimized_layout( stage0_3d_, stage1_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<stage1_complex3_t, x_fft_complex3_t>( "forward_y", stage1_3d_, stage1_xfast_3d_ )
        );
        SCFD_SAFE_CALL(
            same_z_.transpose_x_to_y_xfast_layout( stage1_xfast_3d_, x_fft_stage_3d_, transpose_mode_3d )
        );
        SCFD_SAFE_CALL( base_fft_.template exec<x_fft_complex3_t, complex_array3_t>( "forward_x", x_fft_stage_3d_, out ) );
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
        SCFD_SAFE_CALL( same_x_.transpose_xzy_to_xyz_optimized_layout( stage1_3d_, stage0_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out ) );
    }

    void backward_3d_pencil_pencil_layout_( std::true_type, complex_array3_t &in, real_array3_t &out )
    {
        if ( !init_options_.use_optimized || init_options_.pencil_layout_3d == fftm_3d_pencil_layout::legacy )
        {
            throw std::logic_error(
                "fftm 3D pencil-pencil optimized layout was selected at compile time; use "
                "strategy_3d_pencil_pencil<Mode, false> for the legacy runtime path."
            );
        }

        if ( init_options_.pencil_pipeline_3d != fftm_3d_pencil_pipeline::staged )
        {
            if ( is_reference_owned_pencil_pipeline_( init_options_.pencil_pipeline_3d ) )
            {
                if ( partitioning_.get_process_grid().p2 == 1 && !native_opt0_default_z_layout_enabled_() )
                {
                    SCFD_SAFE_CALL( bind_stage1_3d_to_io_buffer_( in, "backward pencil-pencil spectral input" ) );
                    FFTM_PROFILE_SCOPED_TIC( "fftm::backward_3d_pencil_pencil_reference_p2_degenerate" );
                    stage0_complex3_t stage0_slab_view(
                        stage0_3d_.raw_ptr(), input_dim_.size_x[myid_i_], nz_half_, ny_
                    );
                    SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, complex_array3_t>( "inverse_x", in, in ) );
                    /*
                     * The reference pipeline's datatype-direct variant is meant for
                     * the owned non-degenerate pencil schedule.  For degenerate
                     * shortcut routes, CUDA-aware strided receive datatypes in
                     * the generic transpose can stall on some MPI stacks; keep
                     * these shortcut paths on the packed value transfer.
                     */
                    same_z_.set_direct_transfer_options( false, init_options_.execution.direct_p2p_cuda_aware );
                    try
                    {
                        SCFD_SAFE_CALL(
                            same_z_.transpose_y_to_x_optimized_layout( in, stage0_slab_view, transpose_mode_3d )
                        );
                    }
                    catch ( ... )
                    {
                        same_z_.set_direct_transfer_options(
                            init_options_.execution.use_direct_backward_receive, init_options_.execution.direct_p2p_cuda_aware
                        );
                        throw;
                    }
                    same_z_.set_direct_transfer_options(
                        init_options_.execution.use_direct_backward_receive, init_options_.execution.direct_p2p_cuda_aware
                    );
                    SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex3_t, real_array3_t>(
                        "inverse_yz_pencil_pencil_p2_degenerate", stage0_slab_view, out
                    ) );
                    return;
                }
                if ( partitioning_.get_process_grid().p1 == 1 )
                {
                    SCFD_SAFE_CALL( bind_stage1_3d_to_io_buffer_( in, "backward pencil-pencil spectral input" ) );
                    FFTM_PROFILE_SCOPED_TIC( "fftm::backward_3d_pencil_pencil_reference_p1_degenerate" );
                    SCFD_SAFE_CALL(
                        base_fft_.template exec<complex_array3_t, x_fft_complex3_t>( "inverse_x", in, x_fft_stage_3d_ )
                    );
                    SCFD_SAFE_CALL( copy_xfft_to_stage1_xfast_3d_() );
                    SCFD_SAFE_CALL( base_fft_.template exec<x_fft_complex3_t, stage1_complex3_t>(
                        "inverse_y", stage1_xfast_3d_, stage1_3d_
                    ) );
                    /*
                     * See the p2-degenerate note above: avoid direct datatype
                     * receives in the generic same-X shortcut and reserve that
                     * variant for the owned pencil pipeline.
                     */
                    same_x_.set_direct_transfer_options( false, init_options_.execution.direct_p2p_cuda_aware );
                    try
                    {
                        SCFD_SAFE_CALL( same_x_.transpose_xzy_to_xyz_optimized_layout(
                            stage1_3d_, stage0_3d_, transpose_mode_3d
                        ) );
                    }
                    catch ( ... )
                    {
                        same_x_.set_direct_transfer_options(
                            init_options_.execution.use_direct_backward_receive, init_options_.execution.direct_p2p_cuda_aware
                        );
                        throw;
                    }
                    same_x_.set_direct_transfer_options(
                        init_options_.execution.use_direct_backward_receive, init_options_.execution.direct_p2p_cuda_aware
                    );
                    SCFD_SAFE_CALL(
                        base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out )
                    );
                    return;
                }
                if ( native_opt0_default_z_layout_enabled_() )
                {
                    FFTM_PROFILE_SCOPED_TIC( "fftm::backward_3d_pencil_pencil_native_opt0_zfast" );
                    if ( native_opt0_reference_y_buffer_topology_enabled_() )
                    {
                        SCFD_SAFE_CALL( bind_native_opt0_reference_backward_views_( in ) );
                    }
                    SCFD_SAFE_CALL( pencil_pencil_reference_owned_.template backward_native_opt0_zfast<
                        complex_array3_t, output_zfast_complex3_t, zfast_complex3_t, zfast_complex3_t,
                        zfast_complex3_t>(
                        in, x_fft_zfast_3d_, stage1_yfft_zfast_3d_, stage1_zfast_3d_, stage0_default_z_3d_,
                        native_opt0_inverse_y_plan_names_, native_opt0_y_plan_offsets_, transpose_mode_3d
                    ) );
                    SCFD_SAFE_CALL(
                        pencil_pencil_reference_owned_.exec_backward_z_with_deferred_send( stage0_default_z_3d_, out )
                    );
                    return;
                }
                SCFD_SAFE_CALL( bind_stage1_3d_to_io_buffer_( in, "backward pencil-pencil spectral input" ) );
                SCFD_SAFE_CALL( pencil_pencil_reference_owned_.template backward<
                    complex_array3_t, x_fft_complex3_t, x_fft_complex3_t, stage1_complex3_t, stage0_complex3_t,
                    real_array3_t>(
                    in, x_fft_stage_3d_, stage1_xfast_3d_, stage1_3d_, stage0_3d_, out, transpose_mode_3d
                ) );
            }
            else
            {
                SCFD_SAFE_CALL( bind_stage1_3d_to_io_buffer_( in, "backward pencil-pencil spectral input" ) );
                SCFD_SAFE_CALL( pencil_pencil_pipeline_.template backward<
                    complex_array3_t, x_fft_complex3_t, x_fft_complex3_t, stage1_complex3_t, stage0_complex3_t,
                    real_array3_t>(
                    in, x_fft_stage_3d_, stage1_xfast_3d_, stage1_3d_, stage0_3d_, out, transpose_mode_3d
                ) );
            }
            return;
        }

        SCFD_SAFE_CALL( base_fft_.template exec<complex_array3_t, x_fft_complex3_t>( "inverse_x", in, x_fft_stage_3d_ ) );
        SCFD_SAFE_CALL(
            same_z_.transpose_y_to_x_xfast_layout( x_fft_stage_3d_, stage1_xfast_3d_, transpose_mode_3d )
        );
        SCFD_SAFE_CALL( bind_stage1_3d_to_io_buffer_( in, "backward pencil-pencil spectral input" ) );
        SCFD_SAFE_CALL(
            base_fft_.template exec<x_fft_complex3_t, stage1_complex3_t>( "inverse_y", stage1_xfast_3d_, stage1_3d_ )
        );
        SCFD_SAFE_CALL( same_x_.transpose_xzy_to_xyz_optimized_layout( stage1_3d_, stage0_3d_, transpose_mode_3d ) );
        SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex3_t, real_array3_t>( "inverse_z", stage0_3d_, out ) );
    }

    void forward_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil>,
        const real_array4_t &in, complex_array4_t &out
    )
    {
        if ( pencil_4d_native_zw_native_spectral_layout_enabled_() )
        {
            stage2_complex4_t out_native = pencil_4d_native_zw_native_spectral_view_( out );
            SCFD_SAFE_CALL( forward_native_spectral_4d_(
                std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil>(), in,
                out_native
            ) );
            return;
        }

        FFTM_PROFILE_SCOPED_TIC( "fftm::forward_4d_pencil_pencil" );
        time_native_4d_stage_( "4d/pencil/forward_w_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<real_array4_t, stage0_complex4_t>( "forward_w", in, stage0_4d_ ) );
        } );
        time_native_4d_stage_( "4d/pencil/forward_same_xy", [&]() {
            SCFD_SAFE_CALL( same_xy_.transpose_xyzw_to_xywz( stage0_4d_, stage1_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/forward_z_fft", [&]() {
            SCFD_SAFE_CALL(
                base_fft_.template exec<stage1_complex4_t, stage1_complex4_t>( "forward_z", stage1_4d_, stage1_4d_ )
            );
        } );
        time_native_4d_stage_( "4d/pencil/forward_same_xw", [&]() {
            SCFD_SAFE_CALL( same_xw_.transpose_xywz_to_xzwy( stage1_4d_, stage2_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/forward_y_fft", [&]() {
            SCFD_SAFE_CALL(
                base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>( "forward_y", stage2_4d_, stage2_4d_ )
            );
        } );
        time_native_4d_stage_( "4d/pencil/forward_same_zw", [&]() {
            SCFD_SAFE_CALL( same_zw_.transpose_xzwy_to_yzwx( stage2_4d_, out, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/forward_x_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<complex_array4_t, complex_array4_t>( "forward_x", out, out ) );
        } );
    }

    void backward_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil>,
        complex_array4_t &in, real_array4_t &out
    )
    {
        if ( pencil_4d_native_zw_native_spectral_layout_enabled_() )
        {
            stage2_complex4_t in_native = pencil_4d_native_zw_native_spectral_view_( in );
            SCFD_SAFE_CALL( backward_native_spectral_4d_(
                std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil>(),
                in_native, out
            ) );
            return;
        }

        FFTM_PROFILE_SCOPED_TIC( "fftm::backward_4d_pencil_pencil" );
        time_native_4d_stage_( "4d/pencil/backward_x_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<complex_array4_t, complex_array4_t>( "inverse_x", in, in ) );
        } );
        time_native_4d_stage_( "4d/pencil/backward_same_zw", [&]() {
            SCFD_SAFE_CALL( same_zw_.transpose_yzwx_to_xzwy( in, stage2_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/backward_y_fft", [&]() {
            SCFD_SAFE_CALL(
                base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>( "inverse_y", stage2_4d_, stage2_4d_ )
            );
        } );
        time_native_4d_stage_( "4d/pencil/backward_same_xw", [&]() {
            SCFD_SAFE_CALL( same_xw_.transpose_xzwy_to_xywz( stage2_4d_, stage1_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/backward_z_fft", [&]() {
            SCFD_SAFE_CALL(
                base_fft_.template exec<stage1_complex4_t, stage1_complex4_t>( "inverse_z", stage1_4d_, stage1_4d_ )
            );
        } );
        time_native_4d_stage_( "4d/pencil/backward_same_xy", [&]() {
            SCFD_SAFE_CALL( same_xy_.transpose_xywz_to_xyzw( stage1_4d_, stage0_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/backward_w_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex4_t, real_array4_t>( "inverse_w", stage0_4d_, out ) );
        } );
    }

    void forward_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab>,
        const real_array4_t &in, complex_array4_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::forward_4d_slab_slab" );
        if ( slab_4d_native_wz_communication_layout_enabled_() )
        {
            stage2_complex4_t out_native = slab_4d_native_xw_native_spectral_view_( out );
            SCFD_SAFE_CALL( forward_native_spectral_4d_(
                std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab>(), in,
                out_native
            ) );
            return;
        }
        time_native_4d_stage_( "4d/slab/forward_zw_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<real_array4_t, stage0_complex4_t>( "forward_zw", in, stage0_4d_ ) );
        } );
        if ( slab_4d_native_xw_transpose_enabled_() )
        {
            if ( slab_4d_native_xw_native_spectral_layout_enabled_() )
            {
                stage2_complex4_t out_native = slab_4d_native_xw_native_spectral_view_( out );
                SCFD_SAFE_CALL( forward_native_spectral_4d_from_stage0_( out_native ) );
                return;
            }
            else if ( slab_4d_native_xw_layout_stage_enabled_() )
            {
                time_native_4d_stage_( "4d/slab/forward_same_xw_native_xzwy", [&]() {
                    SCFD_SAFE_CALL(
                        same_xw_.transpose_xyzw_to_xzwy_slab_native( stage0_4d_, stage2_4d_, transpose_mode_4d )
                    );
                } );
                time_native_4d_stage_( "4d/slab/forward_local_yzwx_from_native_xzwy", [&]() {
                    SCFD_SAFE_CALL( transpose_local_4d_<3, 1, 2, 0>( stage2_4d_, out ) );
                } );
            }
            else
            {
                time_native_4d_stage_( "4d/slab/forward_same_xw_native", [&]() {
                    SCFD_SAFE_CALL(
                        same_xw_.transpose_xyzw_to_yzwx_slab_native( stage0_4d_, out, transpose_mode_4d )
                    );
                } );
            }
        }
        else
        {
            time_native_4d_stage_( "4d/slab/forward_local_xywz", [&]() {
                SCFD_SAFE_CALL( transpose_local_4d_<0, 1, 3, 2>( stage0_4d_, stage1_4d_ ) );
            } );
            time_native_4d_stage_( "4d/slab/forward_same_xw", [&]() {
                SCFD_SAFE_CALL( same_xw_.transpose_xywz_to_xzwy( stage1_4d_, stage2_4d_, transpose_mode_4d ) );
            } );
            time_native_4d_stage_( "4d/slab/forward_local_yzwx", [&]() {
                SCFD_SAFE_CALL( transpose_local_4d_<3, 1, 2, 0>( stage2_4d_, out ) );
            } );
        }
        time_native_4d_stage_( "4d/slab/forward_xy_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<complex_array4_t, complex_array4_t>( "forward_xy", out, out ) );
        } );
    }

    void backward_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab>, complex_array4_t &in,
        real_array4_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::backward_4d_slab_slab" );
        if ( slab_4d_native_xw_native_spectral_layout_enabled_() )
        {
            stage2_complex4_t in_native = slab_4d_native_xw_native_spectral_view_( in );
            SCFD_SAFE_CALL( backward_native_spectral_4d_to_real_( in_native, out ) );
            return;
        }
        time_native_4d_stage_( "4d/slab/backward_xy_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<complex_array4_t, complex_array4_t>( "inverse_xy", in, in ) );
        } );
        if ( slab_4d_native_xw_transpose_enabled_() )
        {
            if ( slab_4d_native_xw_layout_stage_enabled_() )
            {
                time_native_4d_stage_( "4d/slab/backward_local_xzwy_for_native", [&]() {
                    SCFD_SAFE_CALL( transpose_local_4d_<3, 1, 2, 0>( in, stage2_4d_ ) );
                } );
                time_native_4d_stage_( "4d/slab/backward_same_xw_native_xzwy", [&]() {
                    SCFD_SAFE_CALL(
                        same_xw_.transpose_xzwy_to_xyzw_slab_native( stage2_4d_, stage0_4d_, transpose_mode_4d )
                    );
                } );
            }
            else
            {
                time_native_4d_stage_( "4d/slab/backward_same_xw_native", [&]() {
                    SCFD_SAFE_CALL(
                        same_xw_.transpose_yzwx_to_xyzw_slab_native( in, stage0_4d_, transpose_mode_4d )
                    );
                } );
            }
        }
        else
        {
            time_native_4d_stage_( "4d/slab/backward_local_xzwy", [&]() {
                SCFD_SAFE_CALL( transpose_local_4d_<3, 1, 2, 0>( in, stage2_4d_ ) );
            } );
            time_native_4d_stage_( "4d/slab/backward_same_xw", [&]() {
                SCFD_SAFE_CALL( same_xw_.transpose_xzwy_to_xywz( stage2_4d_, stage1_4d_, transpose_mode_4d ) );
            } );
            time_native_4d_stage_( "4d/slab/backward_local_xyzw", [&]() {
                SCFD_SAFE_CALL( transpose_local_4d_<0, 1, 3, 2>( stage1_4d_, stage0_4d_ ) );
            } );
        }
        time_native_4d_stage_( "4d/slab/backward_zw_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex4_t, real_array4_t>( "inverse_zw", stage0_4d_, out ) );
        } );
    }

    void forward_native_spectral_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab>,
        const real_array4_t &in, stage2_complex4_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::forward_4d_slab_slab_native_spectral" );
        if ( slab_4d_native_wz_communication_layout_enabled_() )
        {
            if ( slab_4d_native_wz_ready_pipeline_enabled_() )
            {
                ensure_slab_wz_communication_plan_bundle_();
                const std::size_t lanes = base_fft_.r2c_c2r_plan_bundle_lane_count(
                    slab_wz_communication_plan_bundle_
                );
                time_native_4d_stage_( "4d/slab/forward_wz_same_xw_ready_pipeline", [&]() {
                    SCFD_SAFE_CALL( same_xw_.transpose_xyzw_wz_communication_to_xzwy_slab_native_pipelined(
                        stage0_wz_communication_4d_, out, transpose_mode_4d, lanes,
                        [&]( int plane, std::size_t lane ) {
                            launch_forward_slab_wz_communication_plane_(
                                in, static_cast<std::size_t>( plane ), lane
                            );
                        },
                        [&]( std::size_t lane ) {
                            return base_fft_.r2c_c2r_plan_bundle_lane_ready(
                                slab_wz_communication_plan_bundle_, lane
                            );
                        },
                        [&]() {
                            base_fft_.synchronize_r2c_c2r_plan_bundle( slab_wz_communication_plan_bundle_ );
                        }
                    ) );
                } );
            }
            else
            {
                time_native_4d_stage_( "4d/slab/forward_zw_fft_wz_communication", [&]() {
                    SCFD_SAFE_CALL( execute_forward_slab_wz_communication_fft_( in ) );
                } );
                time_native_4d_stage_( "4d/slab/forward_same_xw_wz_communication", [&]() {
                    SCFD_SAFE_CALL( same_xw_.transpose_xyzw_wz_communication_to_xzwy_slab_native(
                        stage0_wz_communication_4d_, out, transpose_mode_4d
                    ) );
                } );
            }
            time_native_4d_stage_( "4d/slab/forward_xy_fft_xzwy_native", [&]() {
                SCFD_SAFE_CALL( base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>(
                    "forward_xy_xzwy_native", out, out
                ) );
            } );
            return;
        }
        time_native_4d_stage_( "4d/slab/forward_zw_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<real_array4_t, stage0_complex4_t>( "forward_zw", in, stage0_4d_ ) );
        } );
        SCFD_SAFE_CALL( forward_native_spectral_4d_from_stage0_( out ) );
    }

    void forward_native_spectral_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil>,
        const real_array4_t &in, stage2_complex4_t &out
    )
    {
        if ( pencil_4d_degenerate_xw_slab_path_enabled_() )
        {
            FFTM_PROFILE_SCOPED_TIC( "fftm::forward_4d_pencil_pencil_degenerate_xw_slab_native_spectral" );
            time_native_4d_stage_( "4d/pencil/degen_xw/forward_zw_fft", [&]() {
                SCFD_SAFE_CALL( base_fft_.template exec<real_array4_t, stage0_complex4_t>(
                    "forward_zw_pencil_degenerate_xw_slab", in, stage0_4d_
                ) );
            } );
            SCFD_SAFE_CALL( forward_native_spectral_4d_from_stage0_( out, "4d/pencil/degen_xw" ) );
            return;
        }
        if ( pencil_4d_degenerate_local_transposes_enabled_() )
        {
            FFTM_PROFILE_SCOPED_TIC( "fftm::forward_4d_pencil_pencil_degenerate_local_native_spectral" );
            time_native_4d_stage_( "4d/pencil/degen_local/forward_w_fft", [&]() {
                SCFD_SAFE_CALL(
                    base_fft_.template exec<real_array4_t, stage0_complex4_t>( "forward_w", in, stage0_4d_ )
                );
            } );
            if ( pencil_4d_degenerate_wz_sliced_z_fft_enabled_() )
            {
                time_native_4d_stage_( "4d/pencil/degen_local/forward_z_fft_sliced_xyzw_to_xywz", [&]() {
                    SCFD_SAFE_CALL( forward_4d_degen_wz_sliced_z_fft_() );
                } );
            }
            else
            {
                time_native_4d_stage_( "4d/pencil/degen_local/forward_same_xy_local", [&]() {
                    SCFD_SAFE_CALL( transpose_local_4d_<0, 1, 3, 2>( stage0_4d_, stage1_4d_ ) );
                } );
                time_native_4d_stage_( "4d/pencil/degen_local/forward_z_fft", [&]() {
                    SCFD_SAFE_CALL( base_fft_.template exec<stage1_complex4_t, stage1_complex4_t>(
                        "forward_z", stage1_4d_, stage1_4d_
                    ) );
                } );
            }
            time_native_4d_stage_( "4d/pencil/degen_local/forward_same_xw", [&]() {
                if ( pencil_4d_degenerate_same_xw_native_enabled_() )
                {
                    SCFD_SAFE_CALL( same_xw_.transpose_xywz_to_xzwy_degenerate_pencil_native(
                        stage1_4d_, out, transpose_mode_4d
                    ) );
                }
                else
                {
                    SCFD_SAFE_CALL( same_xw_.transpose_xywz_to_xzwy( stage1_4d_, out, transpose_mode_4d ) );
                }
            } );
            time_native_4d_stage_( "4d/pencil/degen_local/forward_y_fft", [&]() {
                SCFD_SAFE_CALL( base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>(
                    "forward_y", out, out
                ) );
            } );
            time_native_4d_stage_( "4d/pencil/degen_local/forward_same_zw_alias", [&]() {} );
            time_native_4d_stage_( "4d/pencil/degen_local/forward_x_fft_xzwy", [&]() {
                SCFD_SAFE_CALL( base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>(
                    "forward_x_xzwy_native", out, out
                ) );
            } );
            return;
        }

        FFTM_PROFILE_SCOPED_TIC( "fftm::forward_4d_pencil_pencil_native_spectral" );
        time_native_4d_stage_( "4d/pencil/native/forward_w_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<real_array4_t, stage0_complex4_t>( "forward_w", in, stage0_4d_ ) );
        } );
        time_native_4d_stage_( "4d/pencil/native/forward_same_xy", [&]() {
            SCFD_SAFE_CALL( same_xy_.transpose_xyzw_to_xywz( stage0_4d_, stage1_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/native/forward_z_fft", [&]() {
            SCFD_SAFE_CALL(
                base_fft_.template exec<stage1_complex4_t, stage1_complex4_t>( "forward_z", stage1_4d_, stage1_4d_ )
            );
        } );
        time_native_4d_stage_( "4d/pencil/native/forward_same_xw", [&]() {
            SCFD_SAFE_CALL( same_xw_.transpose_xywz_to_xzwy( stage1_4d_, stage2_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/native/forward_y_fft", [&]() {
            SCFD_SAFE_CALL(
                base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>( "forward_y", stage2_4d_, stage2_4d_ )
            );
        } );
        time_native_4d_stage_( "4d/pencil/native/forward_same_zw_xzwy", [&]() {
            SCFD_SAFE_CALL( same_zw_.transpose_xzwy_to_xzwy_native( stage2_4d_, out, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/native/forward_x_fft_xzwy", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>(
                "forward_x_xzwy_native", out, out
            ) );
        } );
    }

    void backward_native_spectral_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::slab_slab>, stage2_complex4_t &in,
        real_array4_t &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "fftm::backward_4d_slab_slab_native_spectral" );
        SCFD_SAFE_CALL( backward_native_spectral_4d_to_real_( in, out ) );
    }

    void backward_native_spectral_4d_(
        std::integral_constant<transform_strategy_4d_mpi, transform_strategy_4d_mpi::pencil_pencil>,
        stage2_complex4_t &in, real_array4_t &out
    )
    {
        if ( pencil_4d_degenerate_xw_slab_path_enabled_() )
        {
            FFTM_PROFILE_SCOPED_TIC( "fftm::backward_4d_pencil_pencil_degenerate_xw_slab_native_spectral" );
            SCFD_SAFE_CALL( backward_native_spectral_4d_to_real_(
                in, out, "4d/pencil/degen_xw", "inverse_zw_pencil_degenerate_xw_slab"
            ) );
            return;
        }
        if ( pencil_4d_degenerate_local_transposes_enabled_() )
        {
            FFTM_PROFILE_SCOPED_TIC( "fftm::backward_4d_pencil_pencil_degenerate_local_native_spectral" );
            time_native_4d_stage_( "4d/pencil/degen_local/backward_x_fft_xzwy", [&]() {
                SCFD_SAFE_CALL( base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>(
                    "inverse_x_xzwy_native", in, in
                ) );
            } );
            time_native_4d_stage_( "4d/pencil/degen_local/backward_same_zw_alias", [&]() {} );
            time_native_4d_stage_( "4d/pencil/degen_local/backward_y_fft", [&]() {
                SCFD_SAFE_CALL( base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>(
                    "inverse_y", in, in
                ) );
            } );
            time_native_4d_stage_( "4d/pencil/degen_local/backward_same_xw", [&]() {
                if ( pencil_4d_degenerate_same_xw_native_enabled_() )
                {
                    SCFD_SAFE_CALL( same_xw_.transpose_xzwy_to_xywz_degenerate_pencil_native(
                        in, stage1_4d_, transpose_mode_4d
                    ) );
                }
                else
                {
                    SCFD_SAFE_CALL( same_xw_.transpose_xzwy_to_xywz( in, stage1_4d_, transpose_mode_4d ) );
                }
            } );
            if ( pencil_4d_degenerate_wz_sliced_z_fft_enabled_() )
            {
                time_native_4d_stage_( "4d/pencil/degen_local/backward_z_fft_sliced_xywz_to_xyzw", [&]() {
                    SCFD_SAFE_CALL( inverse_4d_degen_wz_sliced_z_fft_() );
                } );
            }
            else
            {
                time_native_4d_stage_( "4d/pencil/degen_local/backward_z_fft", [&]() {
                    SCFD_SAFE_CALL( base_fft_.template exec<stage1_complex4_t, stage1_complex4_t>(
                        "inverse_z", stage1_4d_, stage1_4d_
                    ) );
                } );
                time_native_4d_stage_( "4d/pencil/degen_local/backward_same_xy_local", [&]() {
                    SCFD_SAFE_CALL( transpose_local_4d_<0, 1, 3, 2>( stage1_4d_, stage0_4d_ ) );
                } );
            }
            time_native_4d_stage_( "4d/pencil/degen_local/backward_w_fft", [&]() {
                SCFD_SAFE_CALL(
                    base_fft_.template exec<stage0_complex4_t, real_array4_t>( "inverse_w", stage0_4d_, out )
                );
            } );
            return;
        }

        FFTM_PROFILE_SCOPED_TIC( "fftm::backward_4d_pencil_pencil_native_spectral" );
        time_native_4d_stage_( "4d/pencil/native/backward_x_fft_xzwy", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>(
                "inverse_x_xzwy_native", in, in
            ) );
        } );
        time_native_4d_stage_( "4d/pencil/native/backward_same_zw_xzwy", [&]() {
            SCFD_SAFE_CALL( same_zw_.transpose_xzwy_native_to_xzwy( in, stage2_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/native/backward_y_fft", [&]() {
            SCFD_SAFE_CALL(
                base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>( "inverse_y", stage2_4d_, stage2_4d_ )
            );
        } );
        time_native_4d_stage_( "4d/pencil/native/backward_same_xw", [&]() {
            SCFD_SAFE_CALL( same_xw_.transpose_xzwy_to_xywz( stage2_4d_, stage1_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/native/backward_z_fft", [&]() {
            SCFD_SAFE_CALL(
                base_fft_.template exec<stage1_complex4_t, stage1_complex4_t>( "inverse_z", stage1_4d_, stage1_4d_ )
            );
        } );
        time_native_4d_stage_( "4d/pencil/native/backward_same_xy", [&]() {
            SCFD_SAFE_CALL( same_xy_.transpose_xywz_to_xyzw( stage1_4d_, stage0_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( "4d/pencil/native/backward_w_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex4_t, real_array4_t>( "inverse_w", stage0_4d_, out ) );
        } );
    }

    void forward_native_spectral_4d_from_stage0_( stage2_complex4_t &out, const char *stage_prefix = "4d/slab" )
    {
        time_native_4d_stage_( std::string( stage_prefix ) + "/forward_same_xw_native_xzwy", [&]() {
            SCFD_SAFE_CALL( same_xw_.transpose_xyzw_to_xzwy_slab_native( stage0_4d_, out, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( std::string( stage_prefix ) + "/forward_xy_fft_xzwy_native", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>(
                "forward_xy_xzwy_native", out, out
            ) );
        } );
    }

    void backward_native_spectral_4d_to_real_(
        stage2_complex4_t &in, real_array4_t &out, const char *stage_prefix = "4d/slab",
        const char *inverse_zw_plan = "inverse_zw"
    )
    {
        time_native_4d_stage_( std::string( stage_prefix ) + "/backward_xy_fft_xzwy_native", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<stage2_complex4_t, stage2_complex4_t>(
                "inverse_xy_xzwy_native", in, in
            ) );
        } );
        if ( slab_4d_native_wz_communication_layout_enabled_() )
        {
            if ( slab_4d_native_wz_ready_pipeline_enabled_() )
            {
                ensure_slab_wz_communication_plan_bundle_();
                const std::size_t lanes = base_fft_.r2c_c2r_plan_bundle_lane_count(
                    slab_wz_communication_plan_bundle_
                );
                time_native_4d_stage_( std::string( stage_prefix ) + "/backward_same_xw_wz_ready_pipeline", [&]() {
                    SCFD_SAFE_CALL( same_xw_.transpose_xzwy_to_xyzw_wz_communication_slab_native_pipelined(
                        in, stage0_wz_communication_4d_, transpose_mode_4d, lanes,
                        [&]( int plane, std::size_t lane ) {
                            launch_inverse_slab_wz_communication_plane_(
                                out, static_cast<std::size_t>( plane ), lane
                            );
                        },
                        [&]( std::size_t lane ) {
                            return base_fft_.r2c_c2r_plan_bundle_lane_ready(
                                slab_wz_communication_plan_bundle_, lane
                            );
                        },
                        [&]() {
                            base_fft_.synchronize_r2c_c2r_plan_bundle( slab_wz_communication_plan_bundle_ );
                        }
                    ) );
                } );
            }
            else
            {
                time_native_4d_stage_( std::string( stage_prefix ) + "/backward_same_xw_wz_communication", [&]() {
                    SCFD_SAFE_CALL( same_xw_.transpose_xzwy_to_xyzw_wz_communication_slab_native(
                        in, stage0_wz_communication_4d_, transpose_mode_4d
                    ) );
                } );
                time_native_4d_stage_( std::string( stage_prefix ) + "/backward_zw_fft_wz_communication", [&]() {
                    SCFD_SAFE_CALL( execute_inverse_slab_wz_communication_fft_( out ) );
                } );
            }
            return;
        }
        time_native_4d_stage_( std::string( stage_prefix ) + "/backward_same_xw_native_xzwy", [&]() {
            SCFD_SAFE_CALL( same_xw_.transpose_xzwy_to_xyzw_slab_native( in, stage0_4d_, transpose_mode_4d ) );
        } );
        time_native_4d_stage_( std::string( stage_prefix ) + "/backward_zw_fft", [&]() {
            SCFD_SAFE_CALL( base_fft_.template exec<stage0_complex4_t, real_array4_t>(
                inverse_zw_plan, stage0_4d_, out
            ) );
        } );
    }

    void execute_forward_slab_wz_communication_fft_( const real_array4_t &in )
    {
        ensure_slab_wz_communication_plan_bundle_();
        const std::size_t lanes = base_fft_.r2c_c2r_plan_bundle_lane_count(
            slab_wz_communication_plan_bundle_
        );
        for ( std::size_t y = 0; y < slab_wz_communication_forward_input_offsets_.size(); ++y )
            launch_forward_slab_wz_communication_plane_( in, y, y % lanes );
        base_fft_.synchronize_r2c_c2r_plan_bundle( slab_wz_communication_plan_bundle_ );
    }

    void execute_inverse_slab_wz_communication_fft_( real_array4_t &out )
    {
        ensure_slab_wz_communication_plan_bundle_();
        const std::size_t lanes = base_fft_.r2c_c2r_plan_bundle_lane_count(
            slab_wz_communication_plan_bundle_
        );
        for ( std::size_t y = 0; y < slab_wz_communication_inverse_input_offsets_.size(); ++y )
            launch_inverse_slab_wz_communication_plane_( out, y, y % lanes );
        base_fft_.synchronize_r2c_c2r_plan_bundle( slab_wz_communication_plan_bundle_ );
    }

    void ensure_slab_wz_communication_plan_bundle_() const
    {
        if ( slab_wz_communication_plan_bundle_ == base_fft_type::invalid_r2c_c2r_plan_bundle_id() )
            throw std::logic_error( "FFTM 4D slab WZ communication plan bundle is not initialized" );
    }

    void launch_forward_slab_wz_communication_plane_(
        const real_array4_t &in, std::size_t y, std::size_t lane
    )
    {
        base_fft_.exec_r2c_c2r_plan_bundle_forward_lane_no_sync(
            slab_wz_communication_plan_bundle_, lane, slab_wz_communication_forward_input_offsets_.at( y ),
            slab_wz_communication_forward_output_offsets_.at( y ), in.raw_ptr(),
            stage0_wz_communication_4d_.raw_ptr()
        );
    }

    void launch_inverse_slab_wz_communication_plane_( real_array4_t &out, std::size_t y, std::size_t lane )
    {
        base_fft_.exec_r2c_c2r_plan_bundle_inverse_lane_no_sync(
            slab_wz_communication_plan_bundle_, lane, slab_wz_communication_inverse_input_offsets_.at( y ),
            slab_wz_communication_inverse_output_offsets_.at( y ), stage0_wz_communication_4d_.raw_ptr(), out.raw_ptr()
        );
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

    void copy_default_z_to_stage0_yfast_()
    {
        auto scope = profiler_.scoped_tic( "fftm::native_opt0_default_z_to_yfast" );
        SCFD_SAFE_CALL( reorder_x_stage_( stage0_default_z_3d_, stage0_3d_ ) );
    }

    void copy_stage0_yfast_to_default_z_()
    {
        auto scope = profiler_.scoped_tic( "fftm::native_opt0_yfast_to_default_z" );
        SCFD_SAFE_CALL( reorder_x_stage_( stage0_3d_, stage0_default_z_3d_ ) );
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

    stage2_complex4_t slab_4d_native_xw_native_spectral_view_( complex_array4_t &array ) const
    {
        const std::size_t required_size = transpose2_dim_.size_x[myid_i_] * transpose2_dim_.size_z[myid_j_] *
                                          transpose2_dim_.size_w[myid_k_] * transpose2_dim_.size_y[0];
        if ( static_cast<std::size_t>( array.size() ) < required_size )
        {
            throw std::logic_error( "FFTM 4D native spectral buffer is too small for xzwy view" );
        }
        return stage2_complex4_t(
            array.raw_ptr(), transpose2_dim_.size_x[myid_i_], transpose2_dim_.size_z[myid_j_],
            transpose2_dim_.size_w[myid_k_], transpose2_dim_.size_y[0]
        );
    }

    stage2_complex4_t pencil_4d_native_zw_native_spectral_view_( complex_array4_t &array ) const
    {
        const std::size_t required_size = output_dim_.size_x[0] * output_dim_.size_z[myid_j_] *
                                          output_dim_.size_w[myid_k_] * output_dim_.size_y[myid_i_];
        if ( static_cast<std::size_t>( array.size() ) < required_size )
        {
            throw std::logic_error( "FFTM 4D native spectral buffer is too small for pencil xzwy view" );
        }
        return stage2_complex4_t(
            array.raw_ptr(), output_dim_.size_x[0], output_dim_.size_z[myid_j_],
            output_dim_.size_w[myid_k_], output_dim_.size_y[myid_i_]
        );
    }

    stage2_complex4_t native_spectral_view_4d_( complex_array4_t &array ) const
    {
        if ( strategy_family_4d == transform_strategy_4d_mpi::pencil_pencil )
        {
            return pencil_4d_native_zw_native_spectral_view_( array );
        }
        return slab_4d_native_xw_native_spectral_view_( array );
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

    void init_stage0_default_z_3d_( std::size_t d0, std::size_t d1, std::size_t d2 )
    {
        SCFD_SAFE_CALL( scratch_stage0_default_z_3d_.init( d0 * d1 * d2 ) );
        bind_native_opt0_stage0_default_z_view_( scratch_stage0_default_z_3d_.raw_ptr() );
        update_memory_profile_3d_();
    }

    void bind_native_opt0_stage0_default_z_view_( complex *ptr )
    {
        stage0_default_z_3d_ = zfast_complex3_t(
            ptr, input_dim_.size_x[myid_i_], input_dim_.size_y[myid_j_], nz_half_
        );
    }

    complex *native_opt0_forward_default_z_storage_ptr_() const
    {
        if ( native_opt0_default_z_scratch_aliasing_enabled_() )
            return native_opt0_reference_backward_y_output_ptr_();
        return scratch_stage0_default_z_3d_.raw_ptr();
    }

    complex *native_opt0_backward_default_z_storage_ptr_() const
    {
        if ( native_opt0_default_z_scratch_aliasing_enabled_() )
            return native_opt0_reference_slot_ptr_( 0 );
        return scratch_stage0_default_z_3d_.raw_ptr();
    }

    void bind_native_opt0_zfast_3d_views_()
    {
        SCFD_SAFE_CALL( stage1_zfast_3d_.init_by_raw_data(
            scratch_stage1_3d_.raw_ptr(), input_dim_.size_x[myid_i_], ny_, transpose1_dim_.size_z[myid_j_]
        ) );
        SCFD_SAFE_CALL( stage1_yfft_zfast_3d_.init_by_raw_data(
            scratch_stage1_xfast_3d_.raw_ptr(), input_dim_.size_x[myid_i_], ny_, transpose1_dim_.size_z[myid_j_]
        ) );
        SCFD_SAFE_CALL( x_fft_zfast_3d_.init_by_raw_data(
            scratch_stage0_xfft_3d_.raw_ptr(), output_dim_.size_x[0], output_dim_.size_y[myid_i_],
            output_dim_.size_z[myid_j_]
        ) );
    }

    void bind_native_opt0_reference_forward_views_( complex_array3_t &out )
    {
        complex *temp_slot = native_opt0_reference_slot_ptr_( 0 );
        const std::size_t zfast_stage_size =
            input_dim_.size_x[myid_i_] * ny_ * transpose1_dim_.size_z[myid_j_];
        if ( static_cast<std::size_t>( out.size() ) < zfast_stage_size )
        {
            throw std::logic_error( "native opt0 reference forward output is too small for aliased Y/X stage" );
        }
        bind_native_opt0_stage0_default_z_view_( native_opt0_forward_default_z_storage_ptr_() );
        stage1_zfast_3d_ = zfast_complex3_t(
            temp_slot, input_dim_.size_x[myid_i_], ny_, transpose1_dim_.size_z[myid_j_]
        );
        stage1_yfft_zfast_3d_ = zfast_complex3_t(
            out.raw_ptr(), input_dim_.size_x[myid_i_], ny_, transpose1_dim_.size_z[myid_j_]
        );
        x_fft_zfast_3d_ = output_zfast_complex3_t(
            temp_slot, output_dim_.size_x[0], output_dim_.size_y[myid_i_], output_dim_.size_z[myid_j_]
        );
    }

    void bind_native_opt0_reference_backward_views_( complex_array3_t &in )
    {
        complex *temp_slot = native_opt0_reference_slot_ptr_( 0 );
        complex *backward_y_slot = native_opt0_reference_backward_y_output_ptr_();
        const std::size_t zfast_stage_size =
            input_dim_.size_x[myid_i_] * ny_ * transpose1_dim_.size_z[myid_j_];
        if ( static_cast<std::size_t>( in.size() ) < zfast_stage_size )
        {
            throw std::logic_error( "native opt0 reference backward input is too small for aliased Y stage" );
        }
        bind_native_opt0_stage0_default_z_view_( native_opt0_backward_default_z_storage_ptr_() );
        x_fft_zfast_3d_ = output_zfast_complex3_t(
            temp_slot, output_dim_.size_x[0], output_dim_.size_y[myid_i_],
            output_dim_.size_z[myid_j_]
        );
        stage1_yfft_zfast_3d_ = zfast_complex3_t(
            in.raw_ptr(), input_dim_.size_x[myid_i_], ny_, transpose1_dim_.size_z[myid_j_]
        );
        stage1_zfast_3d_ = zfast_complex3_t(
            backward_y_slot, input_dim_.size_x[myid_i_], ny_, transpose1_dim_.size_z[myid_j_]
        );
    }

    void init_shared_stage0_stage1_xfast_3d_(
        std::size_t stage0_d0, std::size_t stage0_d1, std::size_t stage0_d2, std::size_t stage1_xfast_d0,
        std::size_t stage1_xfast_d1, std::size_t stage1_xfast_d2
    )
    {
        const std::size_t stage0_size       = stage0_d0 * stage0_d1 * stage0_d2;
        const std::size_t stage1_xfast_size = stage1_xfast_d0 * stage1_xfast_d1 * stage1_xfast_d2;

        SCFD_SAFE_CALL( scratch_stage0_xfft_3d_.init( std::max( stage0_size, stage1_xfast_size ) ) );
        SCFD_SAFE_CALL(
            stage0_3d_.init_by_raw_data( scratch_stage0_xfft_3d_.raw_ptr(), stage0_d0, stage0_d1, stage0_d2 )
        );
        SCFD_SAFE_CALL( stage1_xfast_3d_.init_by_raw_data(
            scratch_stage0_xfft_3d_.raw_ptr(), stage1_xfast_d0, stage1_xfast_d1, stage1_xfast_d2
        ) );
        update_memory_profile_3d_();
    }

    void init_owned_stage1_3d_( std::size_t d0, std::size_t d1, std::size_t d2 )
    {
        SCFD_SAFE_CALL( scratch_stage1_3d_.init( d0 * d1 * d2 ) );
        SCFD_SAFE_CALL( stage1_3d_.init_by_raw_data( scratch_stage1_3d_.raw_ptr(), d0, d1, d2 ) );
        update_memory_profile_3d_();
    }

    void init_owned_stage1_xfast_3d_( std::size_t d0, std::size_t d1, std::size_t d2 )
    {
        SCFD_SAFE_CALL( scratch_stage1_xfast_3d_.init( d0 * d1 * d2 ) );
        SCFD_SAFE_CALL( stage1_xfast_3d_.init_by_raw_data( scratch_stage1_xfast_3d_.raw_ptr(), d0, d1, d2 ) );
        update_memory_profile_3d_();
    }

    void init_stage1_3d_( std::size_t d0, std::size_t d1, std::size_t d2, std::size_t io_capacity_elems )
    {
        stage1_3d_d0_          = d0;
        stage1_3d_d1_          = d1;
        stage1_3d_d2_          = d2;
        stage1_3d_size_        = d0 * d1 * d2;
        stage1_3d_uses_io_buffer_ = ( io_capacity_elems >= stage1_3d_size_ );

        if ( !stage1_3d_uses_io_buffer_ )
        {
            SCFD_SAFE_CALL( init_owned_stage1_3d_( d0, d1, d2 ) );
            return;
        }
        update_memory_profile_3d_();
    }

    template <class Array>
    void bind_stage1_3d_to_io_buffer_( Array &array, const char *what )
    {
        if ( !stage1_3d_uses_io_buffer_ )
        {
            return;
        }

        if ( static_cast<std::size_t>( array.size() ) < stage1_3d_size_ )
        {
            throw std::logic_error(
                std::string( "fftm 3D pencil-pencil " ) + what + " is too small for stage1 workspace alias"
            );
        }

        stage1_3d_ = stage1_complex3_t( array.raw_ptr(), stage1_3d_d0_, stage1_3d_d1_, stage1_3d_d2_ );
    }

    void copy_stage1_xfast_to_xfft_3d_()
    {
        if ( stage1_xfast_3d_.size() != x_fft_stage_3d_.size() )
            throw std::logic_error( "fftm 3D pencil-pencil degenerate X-stage copy size mismatch" );

        runtime_api_t::memcpy(
            x_fft_stage_3d_.raw_ptr(), stage1_xfast_3d_.raw_ptr(),
            static_cast<std::size_t>( x_fft_stage_3d_.size() ) * sizeof( complex ),
            runtime_api_t::device_to_device_kind()
        );
    }

    void copy_xfft_to_stage1_xfast_3d_()
    {
        if ( stage1_xfast_3d_.size() != x_fft_stage_3d_.size() )
            throw std::logic_error( "fftm 3D pencil-pencil degenerate inverse X-stage copy size mismatch" );

        runtime_api_t::memcpy(
            stage1_xfast_3d_.raw_ptr(), x_fft_stage_3d_.raw_ptr(),
            static_cast<std::size_t>( x_fft_stage_3d_.size() ) * sizeof( complex ),
            runtime_api_t::device_to_device_kind()
        );
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
        SCFD_SAFE_CALL( stage0_wz_communication_4d_.init_by_raw_data(
            scratch_stage0_stage2_4d_.raw_ptr(), stage0_d0, stage0_d1, stage0_d2, stage0_d3
        ) );
        SCFD_SAFE_CALL( stage2_4d_.init_by_raw_data(
            scratch_stage0_stage2_4d_.raw_ptr(), stage2_d0, stage2_d1, stage2_d2, stage2_d3
        ) );
        update_memory_profile_4d_();
    }

    void init_owned_stage1_4d_( std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3 )
    {
        stage1_4d_d0_   = d0;
        stage1_4d_d1_   = d1;
        stage1_4d_d2_   = d2;
        stage1_4d_d3_   = d3;
        stage1_4d_size_ = d0 * d1 * d2 * d3;
        stage1_4d_alias_pending_ = init_options_.execution.use_4d_slab_native_work_area_alias;
        if ( !stage1_4d_alias_pending_ )
        {
            SCFD_SAFE_CALL( scratch_stage1_4d_.init( stage1_4d_size_ ) );
            SCFD_SAFE_CALL( stage1_4d_.init_by_raw_data( scratch_stage1_4d_.raw_ptr(), d0, d1, d2, d3 ) );
        }
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
            "fftm/shared_work_buffer", static_cast<memory_profiler_t::bytes_type>( shared_work_size_ )
        );
        profiler->set_bytes(
            "fftm/shared_host_work_buffer",
            static_cast<memory_profiler_t::bytes_type>( shared_host_work_size_ )
        );
        profiler->set_bytes(
            "fftm/scratch_stage0_xfft_3d",
            bytes_of_elems_( static_cast<std::size_t>( scratch_stage0_xfft_3d_.size() ), sizeof( complex ) )
        );
        profiler->set_bytes(
            "fftm/scratch_stage1_3d",
            bytes_of_elems_( static_cast<std::size_t>( scratch_stage1_3d_.size() ), sizeof( complex ) )
        );
        profiler->set_bytes(
            "fftm/scratch_stage1_xfast_3d",
            bytes_of_elems_( static_cast<std::size_t>( scratch_stage1_xfast_3d_.size() ), sizeof( complex ) )
        );
        profiler->set_bytes(
            "fftm/scratch_stage0_default_z_3d",
            scratch_stage0_default_z_3d_.is_free()
                ? 0
                : bytes_of_elems_( static_cast<std::size_t>( scratch_stage0_default_z_3d_.size() ), sizeof( complex ) )
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
            "fftm/shared_work_buffer", static_cast<memory_profiler_t::bytes_type>( shared_work_size_ )
        );
        profiler->set_bytes(
            "fftm/shared_host_work_buffer",
            static_cast<memory_profiler_t::bytes_type>( shared_host_work_size_ )
        );
        profiler->set_bytes( "fftm/scratch_stage0_xfft_3d", 0 );
        profiler->set_bytes( "fftm/scratch_stage1_3d", 0 );
        profiler->set_bytes( "fftm/scratch_stage1_xfast_3d", 0 );
        profiler->set_bytes( "fftm/scratch_stage0_default_z_3d", 0 );
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
    template <class Array>
    static void release_owned_array_( Array &array )
    {
        if ( !array.is_free() && array.is_own() )
            array.free();
    }

    MPIComm                    mpi_;
    Log                        log_;
    fftm_init_options          init_options_;
    optional_profiler_t        profiler_;
    optional_memory_profiler_t memory_profiler_;
    partitioning_t             partitioning_;

    // Owners must precede every execution object that refers to their storage.
    // C++ destroys members in reverse declaration order, so transpose state and
    // FFT plans are torn down before these buffers release their allocations.
    shared_buffer_t            shared_work_buffer_;
    host_shared_buffer_t       shared_host_work_buffer_;
    std::shared_ptr<detail::reusable_workspace_resource> reusable_workspace_resource_;
    bool                       reusable_workspace_lease_active_ = false;
    complex_buffer_t           scratch_stage0_xfft_3d_;
    complex_buffer_t           scratch_stage0_default_z_3d_;
    complex_buffer_t           scratch_stage1_3d_;
    complex_buffer_t           scratch_stage1_xfast_3d_;
    complex_buffer_t           scratch_stage0_stage2_4d_;
    complex_buffer_t           scratch_stage1_4d_;

    BaseFFT                    base_fft_;
    std::size_t                shared_work_size_      = 0;
    std::size_t                shared_host_work_size_ = 0;
    std::size_t                shared_fft_work_size_ = 0;
    std::size_t                shared_transpose_work_size_ = 0;
    std::size_t                shared_same_xw_work_size_ = 0;
    std::size_t                shared_native_4d_sliced_work_size_ = 0;
    std::size_t                shared_stage1_alias_size_ = 0;
    bool                       slab_4d_native_work_area_alias_effective_ = false;
    bool                       native_opt0_auto_compact_y_workarea_ = false;
    bool                       native_opt0_default_z_scratch_aliased_ = false;
    bool                       native_4d_stage_timing_active_ = false;
    int                        native_4d_stage_timing_iteration_ = -1;
	    same_x_t                   same_x_;
    same_z_t                   same_z_;
    pencil_pencil_pipeline_t   pencil_pencil_pipeline_;
    pencil_pencil_reference_owned_t pencil_pencil_reference_owned_;
    pencil_pencil_plan_state_t pencil_pencil_plan_state_;
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

    std::size_t       stage1_4d_d0_ = 0;
    std::size_t       stage1_4d_d1_ = 0;
    std::size_t       stage1_4d_d2_ = 0;
    std::size_t       stage1_4d_d3_ = 0;
    std::size_t       stage1_4d_size_ = 0;
    bool              stage1_4d_alias_pending_ = false;
    stage0_complex3_t stage0_3d_;
    zfast_complex3_t  stage0_default_z_3d_;
    zfast_complex3_t  stage1_zfast_3d_;
    zfast_complex3_t  stage1_yfft_zfast_3d_;
    output_zfast_complex3_t x_fft_zfast_3d_;
	    stage1_complex3_t stage1_3d_;
    x_fft_complex3_t  x_fft_stage_3d_;
    x_fft_complex3_t  stage1_xfast_3d_;
    std::vector<std::string> native_opt0_forward_y_plan_names_;
	    std::vector<std::string> native_opt0_inverse_y_plan_names_;
	    std::vector<std::size_t> native_opt0_y_plan_offsets_;
    std::vector<std::size_t> native_4d_degen_wz_forward_z_input_offsets_;
    std::vector<std::size_t> native_4d_degen_wz_forward_z_output_offsets_;
    std::vector<std::size_t> native_4d_degen_wz_inverse_z_input_offsets_;
    std::vector<std::size_t> native_4d_degen_wz_inverse_z_output_offsets_;
    std::vector<std::size_t> slab_wz_communication_forward_input_offsets_;
    std::vector<std::size_t> slab_wz_communication_forward_output_offsets_;
    std::vector<std::size_t> slab_wz_communication_inverse_input_offsets_;
    std::vector<std::size_t> slab_wz_communication_inverse_output_offsets_;
    std::vector<fftm_native_stage_timing> native_4d_stage_timings_;
    typename base_fft_type::plan_sequence_id_t slab_wz_communication_forward_sequence_ =
        base_fft_type::invalid_plan_sequence_id();
    typename base_fft_type::plan_sequence_id_t slab_wz_communication_inverse_sequence_ =
        base_fft_type::invalid_plan_sequence_id();
    typename base_fft_type::r2c_c2r_plan_bundle_id_t slab_wz_communication_plan_bundle_ =
        base_fft_type::invalid_r2c_c2r_plan_bundle_id();
    typename base_fft_type::c2c_plan_array_id_t native_opt0_y_plan_array_bundle_ =
        base_fft_type::invalid_c2c_plan_array_id();
    typename base_fft_type::c2c_plan_array_id_t native_4d_degen_wz_forward_z_plan_array_bundle_ =
        base_fft_type::invalid_c2c_plan_array_id();
    typename base_fft_type::c2c_plan_array_id_t native_4d_degen_wz_inverse_z_plan_array_bundle_ =
        base_fft_type::invalid_c2c_plan_array_id();
    std::size_t       stage1_3d_d0_             = 0;
    std::size_t       stage1_3d_d1_             = 0;
    std::size_t       stage1_3d_d2_             = 0;
    std::size_t       stage1_3d_size_           = 0;
    bool              stage1_3d_uses_io_buffer_ = false;
    for_each_3d_t     for_each_3d_;

    stage0_complex4_t stage0_4d_;
    slab_wz_communication_complex4_t stage0_wz_communication_4d_;
    stage1_complex4_t stage1_4d_;
    stage2_complex4_t stage2_4d_;
    for_each_4d_t     for_each_4d_;
};

} // namespace fftm

#endif // __FFTM_FFTM_HPP__
