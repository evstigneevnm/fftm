#ifndef __FFTM_FFTS_HPP__
#define __FFTM_FFTS_HPP__

#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/memory/shared_buffer.h>

#include "detail/array_arrangers.h"
#include "detail/cuda_memcpy_4d_slab_transposer.h"
#include "detail/direct_transpose_4d.h"
#include "detail/memory_profile_utils.h"
#include "fft_direction.h"
#include "profiling.h"

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
    std::string profiling_key                   = "ffts_prof";
    std::string memory_profiling_key            = "ffts_mem";
    bool        print_profile_summary_on_destroy = true;
    bool        print_profile_totals_on_destroy  = true;
    bool        print_memory_profile_on_destroy = true;
    bool        print_memory_totals_on_destroy  = true;
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
    class Real, class Complex, class Memory, std::size_t Dim,
    class Strategy4D = strategy_4d_pencil_pencil<transpose_backend::direct>>
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
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using xyzw_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using xywz_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0123_t>;
    using xzwy_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_0213_t>;
    using yzwx_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using stage0_complex_array_t = xyzw_complex_array_t;
    using stage1_complex_array_t = xywz_complex_array_t;
    using stage2_complex_array_t = xzwy_complex_array_t;
    using complex_array_t        = yzwx_complex_array_t;
};

template <class Real, class Complex, class Memory, transpose_backend Backend>
struct ffts_array_traits<Real, Complex, Memory, 4, strategy_4d_slab_slab<Backend>>
{
    using real_array_t = scfd::arrays::tensor_array_nd<Real, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using xyzw_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_3210_t>;
    using zwxy_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_1032_t>;
    using yzwx_complex_array_t =
        scfd::arrays::tensor_array_nd<Complex, 4, Memory, scfd::arrays::custom_arranger_2103_t>;
    using stage0_complex_array_t = xyzw_complex_array_t;
    using stage1_complex_array_t = zwxy_complex_array_t;
    using stage2_complex_array_t = zwxy_complex_array_t;
    using complex_array_t        = yzwx_complex_array_t;
};

} // namespace detail

template <class BaseFFT, class Backend, class Strategy4D = strategy_4d_pencil_pencil<transpose_backend::direct>>
class ffts
{
public:
    using backend_type               = Backend;
    using base_fft_type              = BaseFFT;
    using real                       = typename BaseFFT::real;
    using complex                    = typename BaseFFT::complex;
    using memory_t                   = typename Backend::memory_type;
    using runtime_api                = typename BaseFFT::runtime_api;
    using strategy_4d_t              = Strategy4D;
    using profiler_t                 = ffts_profiler;
    using optional_profiler_t        = optional_profiler<profiler_t>;
    using memory_profiler_t          = ffts_memory_profiler;
    using optional_memory_profiler_t = optional_memory_profiler<memory_profiler_t>;
    using memory_profile_bytes_t     = typename memory_profiler_t::bytes_type;
    using memory_profile_buckets_t   = detail::memory_profile_buckets<memory_profile_bytes_t>;

    template <std::size_t Dim>
    using real_array_t = typename detail::ffts_array_traits<real, complex, memory_t, Dim, Strategy4D>::real_array_t;

    template <std::size_t Dim>
    using complex_array_t =
        typename detail::ffts_array_traits<real, complex, memory_t, Dim, Strategy4D>::complex_array_t;

    static constexpr transform_strategy_4d strategy_family_4d   = detail::ffts_4d_strategy_traits<Strategy4D>::family;
    static constexpr transpose_backend     transpose_backend_4d = detail::ffts_4d_strategy_traits<Strategy4D>::backend;

    static const char *strategy_name()
    {
        return detail::ffts_4d_strategy_traits<Strategy4D>::name();
    }

    ffts()
        : init_done_( false ), dim_( 0 ), nx_( 0 ), ny_( 0 ), nz_( 0 ), nw_( 0 ), ny_half_( 0 ), nz_half_( 0 ),
          nw_half_( 0 )
    {
        for_each_4d_.block_size = 128;
    }

    ~ffts()
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
    void init( const std::array<std::size_t, Dim> &grid, const ffts_init_options &options = ffts_init_options() )
    {
        init_array_( grid, options, typename std::integral_constant<std::size_t, Dim>::type() );
    }

    void init( std::size_t nx, std::size_t ny, const ffts_init_options &options = ffts_init_options() )
    {
        ensure_can_init_();
        configure_profiling_( options );
        FFTM_PROFILE_SCOPED_TIC( "ffts::init<2>" );
        nx_      = nx;
        ny_      = ny;
        ny_half_ = ny_ / 2 + 1;
        dim_     = 2;

        add_2d_plans_();
        activate_shared_work_area_();
        update_memory_profile_();
        init_done_ = true;
    }

    void init( std::size_t nx, std::size_t ny, std::size_t nz, const ffts_init_options &options = ffts_init_options() )
    {
        ensure_can_init_();
        configure_profiling_( options );
        FFTM_PROFILE_SCOPED_TIC( "ffts::init<3>" );
        nx_      = nx;
        ny_      = ny;
        nz_      = nz;
        nz_half_ = nz_ / 2 + 1;
        dim_     = 3;

        add_3d_plans_();
        activate_shared_work_area_();
        update_memory_profile_();
        init_done_ = true;
    }

    void init(
        std::size_t nx, std::size_t ny, std::size_t nz, std::size_t nw,
        const ffts_init_options &options = ffts_init_options()
    )
    {
        ensure_can_init_();
        configure_profiling_( options );
        FFTM_PROFILE_SCOPED_TIC( "ffts::init<4>" );
        nx_      = nx;
        ny_      = ny;
        nz_      = nz;
        nw_      = nw;
        nw_half_ = nw_ / 2 + 1;
        dim_     = 4;

        init_4d_storage_( strategy_family_tag() );
        add_4d_plans_( strategy_family_tag() );
        activate_shared_work_area_();
        update_memory_profile_();
        init_done_ = true;
    }

    bool is_initialized() const
    {
        return init_done_;
    }

    void print_memory_profile( std::ostream &out ) const
    {
        if ( memory_profiler_.enabled() )
        {
            print_memory_profile_summary_( out );
        }
    }

    void print_profile( std::ostream &out ) const
    {
        if ( profiler_.enabled() )
        {
            ostream_log_t log( out );
            profiler_.log_print( log );
        }
    }

    void print_profile_totals( std::ostream &out ) const
    {
        if ( profiler_.enabled() )
        {
            ostream_log_t log( out );
            profiler_.log_print_totals( log );
        }
    }

    void print_memory_totals( std::ostream &out ) const
    {
        if ( memory_profiler_.enabled() )
        {
            print_memory_profile_totals_( out );
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

    std::size_t dimension() const
    {
        ensure_initialized_();
        return dim_;
    }

    void forward( const real_array_t<2> &in, complex_array_t<2> &out )
    {
        ensure_dimension_( 2 );
        FFTM_PROFILE_SCOPED_TIC( "ffts::forward_2d" );
        base_fft_.template exec<real_array_t<2>, complex_array_t<2>>( "forward_2d", in, out );
    }

    void backward( const complex_array_t<2> &in, real_array_t<2> &out )
    {
        ensure_dimension_( 2 );
        FFTM_PROFILE_SCOPED_TIC( "ffts::backward_2d" );
        base_fft_.template exec<complex_array_t<2>, real_array_t<2>>( "inverse_2d", in, out );
    }

    void forward( const real_array_t<3> &in, complex_array_t<3> &out )
    {
        ensure_dimension_( 3 );
        FFTM_PROFILE_SCOPED_TIC( "ffts::forward_3d" );
        base_fft_.template exec<real_array_t<3>, complex_array_t<3>>( "forward_3d", in, out );
    }

    void backward( const complex_array_t<3> &in, real_array_t<3> &out )
    {
        ensure_dimension_( 3 );
        FFTM_PROFILE_SCOPED_TIC( "ffts::backward_3d" );
        base_fft_.template exec<complex_array_t<3>, real_array_t<3>>( "inverse_3d", in, out );
    }

    void forward( const real_array_t<4> &in, complex_array_t<4> &out )
    {
        ensure_dimension_( 4 );
        FFTM_PROFILE_SCOPED_TIC( "ffts::forward_4d" );
        forward_4d_( strategy_family_tag(), in, out );
    }

    // Destructive inverse: the spectral input is reused as a work buffer to avoid a full-domain copy.
    void backward( complex_array_t<4> &in, real_array_t<4> &out )
    {
        ensure_dimension_( 4 );
        FFTM_PROFILE_SCOPED_TIC( "ffts::backward_4d" );
        backward_4d_( strategy_family_tag(), in, out );
    }

private:
    using backend_t             = Backend;
    using runtime_api_t         = typename BaseFFT::runtime_api;
    using for_each_4d_t         = typename backend_t::template for_each_nd_type<4, int>;
    using strategy_family_tag   = std::integral_constant<transform_strategy_4d, strategy_family_4d>;
    using transpose_backend_tag = std::integral_constant<transpose_backend, transpose_backend_4d>;

    using traits_4d_t            = detail::ffts_array_traits<real, complex, memory_t, 4, Strategy4D>;
    using stage0_complex_array_t = typename traits_4d_t::stage0_complex_array_t;
    using stage1_complex_array_t = typename traits_4d_t::stage1_complex_array_t;
    using stage2_complex_array_t = typename traits_4d_t::stage2_complex_array_t;
    using direct_transposer_t    = detail::direct_transpose_4d;
    using memcpy_transposer_t    = detail::cuda_memcpy_4d_slab_transposer<complex, runtime_api_t>;
    using shared_buffer_t        = scfd::memory::shared_buffer<memory_t>;

    struct ostream_log_t
    {
        explicit ostream_log_t( std::ostream &out_ ) : out( out_ )
        {
        }

        void info( const std::string &message )
        {
            out << message;
            if ( message.empty() || message.back() != '\n' )
            {
                out << '\n';
            }
        }

        std::ostream &out;
    };

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

    void *shared_work_ptr_() const
    {
        return static_cast<void *>( shared_work_buffer_.naive_ptr() );
    }

    void activate_shared_work_area_()
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::activate_shared_work_area" );
        shared_work_size_ = align_up_( base_fft_.activate_work_size(), 256 );
        shared_work_buffer_.require_size_bytes( shared_work_size_ );
        shared_work_buffer_.activate();
        base_fft_.set_external_work_area( shared_work_ptr_() );
        update_memory_profile_();
    }

    void configure_profiling_( const ffts_init_options &options )
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
    }

    void configure_memory_profiling_( const ffts_init_options &options )
    {
        if ( !options.memory_profiling_key.empty() )
        {
            memory_profiler_.enable( options.memory_profiling_key );
        }
        else
        {
            memory_profiler_.disable();
        }
        base_fft_.set_memory_profiler( memory_profiler_.native_ptr(), "ffts/base_fft" );
        update_memory_profile_();
    }

    void log_profile_on_destroy_() const
    {
        if ( !profiler_.enabled() )
        {
            return;
        }

        if ( init_options_.print_profile_summary_on_destroy )
        {
            ostream_log_t log( std::cout );
            profiler_.log_print( log );
        }
        if ( init_options_.print_profile_totals_on_destroy )
        {
            ostream_log_t log( std::cout );
            profiler_.log_print_totals( log );
        }
    }

    void update_memory_profile_()
    {
        if ( !memory_profiler_.enabled() )
        {
            return;
        }

        memory_profiler_t *profiler = memory_profiler_.native_ptr();
        profiler->set_bytes(
            "ffts/shared_work_buffer", static_cast<memory_profiler_t::bytes_type>( shared_work_buffer_.get_work_size() )
        );
        profiler->set_bytes(
            "ffts/scratch_stage0_stage2_4d",
            bytes_of_elems_( static_cast<std::size_t>( scratch_stage0_stage2_4d_.size() ), sizeof( complex ) )
        );
        profiler->set_bytes(
            "ffts/scratch_stage1_4d",
            bytes_of_elems_( static_cast<std::size_t>( scratch_stage1_4d_.size() ), sizeof( complex ) )
        );
        profiler->set_bytes( "ffts/stage0_4d", 0 );
        profiler->set_bytes( "ffts/stage1_4d", 0 );
        profiler->set_bytes( "ffts/stage2_4d", 0 );
        profiler->set_bytes( "ffts/work_hat_4d", 0 );
    }

    void log_memory_profile_on_destroy_() const
    {
        if ( !memory_profiler_.enabled() )
        {
            return;
        }

        if ( init_options_.print_memory_profile_on_destroy )
        {
            print_memory_profile_summary_( std::cout );
        }
        if ( init_options_.print_memory_totals_on_destroy )
        {
            print_memory_profile_totals_( std::cout );
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

    static void append_memory_profile_line_(
        std::ostream &out, const char *name, memory_profile_bytes_t current, memory_profile_bytes_t peak
    )
    {
        out << "\n  " << name << ": current(sum/max/avg)=" << current << "/" << current << "/" << current << " B ("
            << std::fixed << std::setprecision( 3 ) << bytes_to_mib_( current ) << "/" << bytes_to_mib_( current ) << "/"
            << bytes_to_mib_( current ) << " MiB)"
            << ", peak(sum/max/avg)=" << peak << "/" << peak << "/" << peak << " B (" << bytes_to_mib_( peak ) << "/"
            << bytes_to_mib_( peak ) << "/" << bytes_to_mib_( peak ) << " MiB)";
    }

    static void append_memory_profile_bucket_lines_(
        std::ostream &out, const memory_profile_buckets_t &buckets, bool include_external_test
    )
    {
        append_memory_profile_line_(
            out, detail::memory_profile_bucket_name( detail::memory_profile_bucket::device ),
            buckets.get( detail::memory_profile_bucket::device ).current,
            buckets.get( detail::memory_profile_bucket::device ).peak
        );
        append_memory_profile_line_(
            out, detail::memory_profile_bucket_name( detail::memory_profile_bucket::host_pinned ),
            buckets.get( detail::memory_profile_bucket::host_pinned ).current,
            buckets.get( detail::memory_profile_bucket::host_pinned ).peak
        );
        if ( include_external_test )
        {
            append_memory_profile_line_(
                out, detail::memory_profile_bucket_name( detail::memory_profile_bucket::external_test ),
                buckets.get( detail::memory_profile_bucket::external_test ).current,
                buckets.get( detail::memory_profile_bucket::external_test ).peak
            );
        }

        const auto other = buckets.get( detail::memory_profile_bucket::other );
        if ( other.current != 0 || other.peak != 0 )
        {
            append_memory_profile_line_(
                out, detail::memory_profile_bucket_name( detail::memory_profile_bucket::other ), other.current, other.peak
            );
        }
    }

    void print_memory_profile_summary_( std::ostream &out ) const
    {
        const memory_profiler_t *profiler = memory_profiler_.native_ptr();
        if ( profiler == nullptr )
        {
            return;
        }

        out << "Memory profile:";
        for ( const auto &item : profiler->entries() )
        {
            append_memory_profile_line_( out, item.first.c_str(), item.second.current_bytes, item.second.peak_bytes );
        }

        out << "\nMemory profile categories:";
        append_memory_profile_bucket_lines_( out, collect_memory_profile_buckets_(), true );
        out << '\n';
    }

    void print_memory_profile_totals_( std::ostream &out ) const
    {
        const memory_profile_buckets_t buckets = collect_memory_profile_buckets_();
        out << "Memory profile totals:";
        append_memory_profile_line_( out, "total", buckets.total().current, buckets.total().peak );
        append_memory_profile_bucket_lines_( out, buckets, true );
        out << '\n';
    }

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
                "ffts: initialized for " + std::to_string( dim_ ) + "D but used as " + std::to_string( expected_dim ) +
                "D."
            );
        }
    }

    void add_2d_plans_()
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::add_plans_2d" );
        base_fft_.template add_plan_2D<::fftm::direction::R2C>(
            "forward_2d", static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), 1,
            static_cast<long long int>( nx_ * ny_ ), static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_half_ ), 1, static_cast<long long int>( nx_ * ny_half_ ), 1
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2R>(
            "inverse_2d", static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_half_ ), 1,
            static_cast<long long int>( nx_ * ny_half_ ), static_cast<long long int>( nx_ ),
            static_cast<long long int>( ny_ ), 1, static_cast<long long int>( nx_ * ny_ ), 1
        );
    }

    void add_3d_plans_()
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::add_plans_3d" );
        base_fft_.template add_plan_3D<::fftm::direction::R2C>(
            "forward_3d", static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_ ), 1, static_cast<long long int>( nx_ * ny_ * nz_ ),
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_half_ ), 1, static_cast<long long int>( nx_ * ny_ * nz_half_ ), 1
        );

        base_fft_.template add_plan_3D<::fftm::direction::C2R>(
            "inverse_3d", static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nz_half_ ), 1, static_cast<long long int>( nx_ * ny_ * nz_half_ ),
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), static_cast<long long int>( nz_ ), 1,
            static_cast<long long int>( nx_ * ny_ * nz_ ), 1
        );
    }

    void add_4d_plans_( std::integral_constant<transform_strategy_4d, transform_strategy_4d::pencil_pencil> )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::add_plans_4d_pencil_pencil" );
        const long long int real_batch = static_cast<long long int>( nx_ * ny_ * nz_ );
        const long long int z_batch    = static_cast<long long int>( nx_ * ny_ * nw_half_ );
        const long long int y_batch    = static_cast<long long int>( nx_ * nz_ * nw_half_ );
        const long long int x_batch    = static_cast<long long int>( ny_ * nz_ * nw_half_ );

        const long long int w_stride = static_cast<long long int>( nx_ * ny_ * nz_ );
        const long long int z_stride = static_cast<long long int>( nx_ * ny_ * nw_half_ );
        const long long int y_stride = static_cast<long long int>( nx_ * nz_ * nw_half_ );

        base_fft_.template add_plan_1D<::fftm::direction::R2C>(
            "forward_w", static_cast<long long int>( nw_ ), 1, w_stride, 1, 1, w_stride, 1, real_batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2R>(
            "inverse_w", static_cast<long long int>( nw_ ), 1, w_stride, 1, 1, w_stride, 1, real_batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            "forward_z", static_cast<long long int>( nz_ ), 1, z_stride, 1, 1, z_stride, 1, z_batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            "inverse_z", static_cast<long long int>( nz_ ), 1, z_stride, 1, 1, z_stride, 1, z_batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            "forward_y", static_cast<long long int>( ny_ ), 1, y_stride, 1, 1, y_stride, 1, y_batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            "inverse_y", static_cast<long long int>( ny_ ), 1, y_stride, 1, 1, y_stride, 1, y_batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CF>(
            "forward_x", static_cast<long long int>( nx_ ), 1, 1, static_cast<long long int>( nx_ ), 1, 1,
            static_cast<long long int>( nx_ ), x_batch
        );

        base_fft_.template add_plan_1D<::fftm::direction::C2CB>(
            "inverse_x", static_cast<long long int>( nx_ ), 1, 1, static_cast<long long int>( nx_ ), 1, 1,
            static_cast<long long int>( nx_ ), x_batch
        );
    }

    void add_4d_plans_( std::integral_constant<transform_strategy_4d, transform_strategy_4d::slab_slab> )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::add_plans_4d_slab_slab" );
        const long long int zw_batch  = static_cast<long long int>( nx_ * ny_ );
        const long long int xy_batch  = static_cast<long long int>( nz_ * nw_half_ );
        const long long int xy_stride = static_cast<long long int>( nz_ * nw_half_ );

        base_fft_.template add_plan_2D<::fftm::direction::R2C>(
            "forward_zw", static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ), 1,
            static_cast<long long int>( nz_ * nw_ ), static_cast<long long int>( nz_ ),
            static_cast<long long int>( nw_half_ ), 1, static_cast<long long int>( nz_ * nw_half_ ), zw_batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2R>(
            "inverse_zw", static_cast<long long int>( nz_ ), static_cast<long long int>( nw_ ),
            static_cast<long long int>( nz_ ), static_cast<long long int>( nw_half_ ), 1,
            static_cast<long long int>( nz_ * nw_half_ ), static_cast<long long int>( nz_ ),
            static_cast<long long int>( nw_ ), 1, static_cast<long long int>( nz_ * nw_ ), zw_batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2CF>(
            "forward_xy", static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), xy_stride, 1,
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), xy_stride, 1, xy_batch
        );

        base_fft_.template add_plan_2D<::fftm::direction::C2CB>(
            "inverse_xy", static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ),
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), xy_stride, 1,
            static_cast<long long int>( nx_ ), static_cast<long long int>( ny_ ), xy_stride, 1, xy_batch
        );
    }

    void init_4d_storage_( std::integral_constant<transform_strategy_4d, transform_strategy_4d::pencil_pencil> )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::init_4d_storage_pencil_pencil" );
        init_shared_stage0_stage2_4d_( nx_, ny_, nz_, nw_half_, nx_, nz_, nw_half_, ny_ );
        init_owned_stage1_4d_( nx_, ny_, nw_half_, nz_ );
        direct_transposer_.reset( new detail::direct_transpose_4d( nx_, ny_, nz_, nw_half_ ) );
        init_memcpy_transposer_( transpose_backend_tag() );
        update_memory_profile_();
    }

    void init_4d_storage_( std::integral_constant<transform_strategy_4d, transform_strategy_4d::slab_slab> )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::init_4d_storage_slab_slab" );
        init_shared_stage0_stage2_4d_( nx_, ny_, nz_, nw_half_, 0, 0, 0, 0 );
        init_owned_stage1_4d_( nz_, nw_half_, nx_, ny_ );
        direct_transposer_.reset( new detail::direct_transpose_4d( nx_, ny_, nz_, nw_half_ ) );
        init_memcpy_transposer_( transpose_backend_tag() );
        update_memory_profile_();
    }

    void init_memcpy_transposer_( std::integral_constant<transpose_backend, transpose_backend::direct> )
    {
    }

    void init_memcpy_transposer_( std::integral_constant<transpose_backend, transpose_backend::memcpy> )
    {
        memcpy_transposer_.reset(
            new detail::cuda_memcpy_4d_slab_transposer<complex, runtime_api_t>( nx_, ny_, nz_, nw_half_ )
        );
    }

    void forward_4d_(
        std::integral_constant<transform_strategy_4d, transform_strategy_4d::pencil_pencil>, const real_array_t<4> &in,
        complex_array_t<4> &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::forward_4d_pencil_pencil" );
        base_fft_.template exec<real_array_t<4>, stage0_complex_array_t>( "forward_w", in, stage0_ );
        transpose_xyzw_to_xywz_( stage0_, stage1_ );

        base_fft_.template exec<stage1_complex_array_t, stage1_complex_array_t>( "forward_z", stage1_, stage1_ );
        transpose_xywz_to_xzwy_( stage1_, stage2_ );

        base_fft_.template exec<stage2_complex_array_t, stage2_complex_array_t>( "forward_y", stage2_, stage2_ );
        transpose_xzwy_to_yzwx_( stage2_, out );

        base_fft_.template exec<complex_array_t<4>, complex_array_t<4>>( "forward_x", out, out );
    }

    void forward_4d_(
        std::integral_constant<transform_strategy_4d, transform_strategy_4d::slab_slab>, const real_array_t<4> &in,
        complex_array_t<4> &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::forward_4d_slab_slab" );
        base_fft_.template exec<real_array_t<4>, stage0_complex_array_t>( "forward_zw", in, stage0_ );
        transpose_xyzw_to_zwxy_( stage0_, stage1_ );
        base_fft_.template exec<stage1_complex_array_t, stage1_complex_array_t>( "forward_xy", stage1_, stage1_ );
        transpose_zwxy_to_yzwx_( stage1_, out );
    }

    void backward_4d_(
        std::integral_constant<transform_strategy_4d, transform_strategy_4d::pencil_pencil>, complex_array_t<4> &in,
        real_array_t<4> &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::backward_4d_pencil_pencil" );
        base_fft_.template exec<complex_array_t<4>, complex_array_t<4>>( "inverse_x", in, in );
        transpose_yzwx_to_xzwy_( in, stage2_ );

        base_fft_.template exec<stage2_complex_array_t, stage2_complex_array_t>( "inverse_y", stage2_, stage2_ );
        transpose_xzwy_to_xywz_( stage2_, stage1_ );

        base_fft_.template exec<stage1_complex_array_t, stage1_complex_array_t>( "inverse_z", stage1_, stage1_ );
        transpose_xywz_to_xyzw_( stage1_, stage0_ );

        base_fft_.template exec<stage0_complex_array_t, real_array_t<4>>( "inverse_w", stage0_, out );
    }

    void backward_4d_(
        std::integral_constant<transform_strategy_4d, transform_strategy_4d::slab_slab>, complex_array_t<4> &in,
        real_array_t<4> &out
    )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::backward_4d_slab_slab" );
        transpose_yzwx_to_zwxy_( in, stage1_ );
        base_fft_.template exec<stage1_complex_array_t, stage1_complex_array_t>( "inverse_xy", stage1_, stage1_ );
        transpose_zwxy_to_xyzw_( stage1_, stage0_ );
        base_fft_.template exec<stage0_complex_array_t, real_array_t<4>>( "inverse_zw", stage0_, out );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_xywz_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>, const SrcArray &src, DstArray &dst
    )
    {
        direct_transposer_->xyzw_to_xywz( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_xywz_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>, const SrcArray &src, DstArray &dst
    )
    {
        memcpy_transposer_->xyzw_to_xywz( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xzwy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>, const SrcArray &src, DstArray &dst
    )
    {
        direct_transposer_->xywz_to_xzwy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xzwy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>, const SrcArray &src, DstArray &dst
    )
    {
        memcpy_transposer_->xywz_to_xzwy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_yzwx_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>, const SrcArray &src, DstArray &dst
    )
    {
        direct_transposer_->xzwy_to_yzwx( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_yzwx_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>, const SrcArray &src, DstArray &dst
    )
    {
        memcpy_transposer_->xzwy_to_yzwx( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_xzwy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>, const SrcArray &src, DstArray &dst
    )
    {
        direct_transposer_->yzwx_to_xzwy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_xzwy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>, const SrcArray &src, DstArray &dst
    )
    {
        memcpy_transposer_->yzwx_to_xzwy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_xywz_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>, const SrcArray &src, DstArray &dst
    )
    {
        direct_transposer_->xzwy_to_xywz( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_xywz_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>, const SrcArray &src, DstArray &dst
    )
    {
        memcpy_transposer_->xzwy_to_xywz( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xyzw_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>, const SrcArray &src, DstArray &dst
    )
    {
        direct_transposer_->xywz_to_xyzw( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xyzw_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>, const SrcArray &src, DstArray &dst
    )
    {
        memcpy_transposer_->xywz_to_xyzw( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_zwxy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>, const SrcArray &src, DstArray &dst
    )
    {
        direct_transposer_->xyzw_to_zwxy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_zwxy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>, const SrcArray &src, DstArray &dst
    )
    {
        memcpy_transposer_->xyzw_to_zwxy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_xyzw_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>, const SrcArray &src, DstArray &dst
    )
    {
        direct_transposer_->zwxy_to_xyzw( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_xyzw_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>, const SrcArray &src, DstArray &dst
    )
    {
        memcpy_transposer_->zwxy_to_xyzw( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_yzwx_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>, const SrcArray &src, DstArray &dst
    )
    {
        direct_transposer_->zwxy_to_yzwx( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_yzwx_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>, const SrcArray &src, DstArray &dst
    )
    {
        memcpy_transposer_->zwxy_to_yzwx( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_zwxy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::direct>, const SrcArray &src, DstArray &dst
    )
    {
        direct_transposer_->yzwx_to_zwxy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_zwxy_backend_(
        std::integral_constant<transpose_backend, transpose_backend::memcpy>, const SrcArray &src, DstArray &dst
    )
    {
        memcpy_transposer_->yzwx_to_zwxy( for_each_4d_, src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_xywz_( const SrcArray &src, DstArray &dst )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::transpose_xyzw_to_xywz" );
        transpose_xyzw_to_xywz_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xzwy_( const SrcArray &src, DstArray &dst )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::transpose_xywz_to_xzwy" );
        transpose_xywz_to_xzwy_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_yzwx_( const SrcArray &src, DstArray &dst )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::transpose_xzwy_to_yzwx" );
        transpose_xzwy_to_yzwx_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_xzwy_( const SrcArray &src, DstArray &dst )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::transpose_yzwx_to_xzwy" );
        transpose_yzwx_to_xzwy_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xzwy_to_xywz_( const SrcArray &src, DstArray &dst )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::transpose_xzwy_to_xywz" );
        transpose_xzwy_to_xywz_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xywz_to_xyzw_( const SrcArray &src, DstArray &dst )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::transpose_xywz_to_xyzw" );
        transpose_xywz_to_xyzw_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_xyzw_to_zwxy_( const SrcArray &src, DstArray &dst )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::transpose_xyzw_to_zwxy" );
        transpose_xyzw_to_zwxy_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_xyzw_( const SrcArray &src, DstArray &dst )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::transpose_zwxy_to_xyzw" );
        transpose_zwxy_to_xyzw_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_zwxy_to_yzwx_( const SrcArray &src, DstArray &dst )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::transpose_zwxy_to_yzwx" );
        transpose_zwxy_to_yzwx_backend_( transpose_backend_tag(), src, dst );
    }

    template <class SrcArray, class DstArray>
    void transpose_yzwx_to_zwxy_( const SrcArray &src, DstArray &dst )
    {
        FFTM_PROFILE_SCOPED_TIC( "ffts::transpose_yzwx_to_zwxy" );
        transpose_yzwx_to_zwxy_backend_( transpose_backend_tag(), src, dst );
    }

    void init_shared_stage0_stage2_4d_(
        std::size_t stage0_d0, std::size_t stage0_d1, std::size_t stage0_d2, std::size_t stage0_d3,
        std::size_t stage2_d0, std::size_t stage2_d1, std::size_t stage2_d2, std::size_t stage2_d3
    )
    {
        const std::size_t stage0_size = stage0_d0 * stage0_d1 * stage0_d2 * stage0_d3;
        const std::size_t stage2_size = stage2_d0 * stage2_d1 * stage2_d2 * stage2_d3;

        scratch_stage0_stage2_4d_.init( std::max( stage0_size, stage2_size ) );
        stage0_.init_by_raw_data( scratch_stage0_stage2_4d_.raw_ptr(), stage0_d0, stage0_d1, stage0_d2, stage0_d3 );
        if ( stage2_size != 0 )
        {
            stage2_.init_by_raw_data( scratch_stage0_stage2_4d_.raw_ptr(), stage2_d0, stage2_d1, stage2_d2, stage2_d3 );
        }
    }

    void init_owned_stage1_4d_( std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3 )
    {
        scratch_stage1_4d_.init( d0 * d1 * d2 * d3 );
        stage1_.init_by_raw_data( scratch_stage1_4d_.raw_ptr(), d0, d1, d2, d3 );
    }

    void
    init_array_( const std::array<std::size_t, 2> &grid, const ffts_init_options &options, std::integral_constant<std::size_t, 2> )
    {
        init( grid[0], grid[1], options );
    }

    void
    init_array_( const std::array<std::size_t, 3> &grid, const ffts_init_options &options, std::integral_constant<std::size_t, 3> )
    {
        init( grid[0], grid[1], grid[2], options );
    }

    void
    init_array_( const std::array<std::size_t, 4> &grid, const ffts_init_options &options, std::integral_constant<std::size_t, 4> )
    {
        init( grid[0], grid[1], grid[2], grid[3], options );
    }

    BaseFFT                    base_fft_;
    ffts_init_options          init_options_;
    optional_profiler_t        profiler_;
    optional_memory_profiler_t memory_profiler_;
    shared_buffer_t            shared_work_buffer_;
    std::size_t                shared_work_size_ = 0;

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

    scfd::arrays::array_nd<complex, 1, memory_t> scratch_stage0_stage2_4d_;
    scfd::arrays::array_nd<complex, 1, memory_t> scratch_stage1_4d_;
    stage0_complex_array_t                       stage0_;
    stage1_complex_array_t                       stage1_;
    stage2_complex_array_t                       stage2_;
};

} // namespace fftm

#endif // __FFTM_FFTS_HPP__
