#ifndef __FFTM_CUFFT_WRAP_H__
#define __FFTM_CUFFT_WRAP_H__

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <scfd/backend/cuda.h>
#include <scfd/memory/cuda.h>
#include <scfd/utils/cuda_stream_wrap.h>
#include <scfd/utils/todo.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/cufft_safe_call.h>

#include "fft_direction.h"

namespace fftm
{
namespace wrap
{

inline void cufft_check_no_sync( cufftResult status, const char *expr )
{
    if ( status != CUFFT_SUCCESS )
    {
        throw std::runtime_error(
            std::string( "CUFFT_NO_SYNC_CHECK: " ) + expr + " failed: " +
            std::string( _cufftGetErrorEnum( status ) )
        );
    }
}

struct fft_plan_descriptor
{
    direction                         dir = direction::R2C;
    int                               cufft_type = 0;
    int                               rank = 0;
    bool                              default_layout = false;
    std::array<long long int, 3>      n = {{ 0, 0, 0 }};
    std::array<long long int, 3>      inembed = {{ 0, 0, 0 }};
    long long int                     istride = 0;
    long long int                     idist = 0;
    std::array<long long int, 3>      onembed = {{ 0, 0, 0 }};
    long long int                     ostride = 0;
    long long int                     odist = 0;
    long long int                     batch = 0;
    std::size_t                       work_size = 0;
    std::uintptr_t                    work_area_token = 0;
    std::uintptr_t                    stream_token = 0;
};

inline const char *fft_direction_name( direction dir )
{
    switch ( dir )
    {
    case direction::R2C:
        return "R2C";
    case direction::C2R:
        return "C2R";
    case direction::C2CF:
        return "C2CF";
    case direction::C2CB:
        return "C2CB";
    }
    return "unknown";
}

struct cuda_runtime_api
{
    using memory_type        = scfd::memory::cuda_device;
    using device_memory_info_type = typename scfd::backend::cuda::device_memory_info_type;
    using stream_wrap        = scfd::utils::cuda_stream_wrap;
    using stream_t           = cudaStream_t;
    using memcpy_kind_t      = cudaMemcpyKind;
    using memcpy_3d_params_t = cudaMemcpy3DParms;
    using host_func_t        = cudaHostFn_t;
    using event_t            = cudaEvent_t;
    using pos_t              = cudaPos;
    using pitched_ptr_t      = cudaPitchedPtr;
    using extent_t           = cudaExtent;

    static pos_t make_pos( size_t x, size_t y, size_t z )
    {
        return make_cudaPos( x, y, z );
    }

    static pitched_ptr_t make_pitched_ptr( void *ptr, size_t pitch, size_t xsz, size_t ysz )
    {
        return make_cudaPitchedPtr( ptr, pitch, xsz, ysz );
    }

    static pitched_ptr_t make_pitched_ptr( const void *ptr, size_t pitch, size_t xsz, size_t ysz )
    {
        return make_cudaPitchedPtr( const_cast<void *>( ptr ), pitch, xsz, ysz );
    }

    static extent_t make_extent( size_t width, size_t height, size_t depth )
    {
        return make_cudaExtent( width, height, depth );
    }

    static constexpr memcpy_kind_t device_to_device_kind()
    {
        return cudaMemcpyDeviceToDevice;
    }

    static constexpr memcpy_kind_t device_to_host_kind()
    {
        return cudaMemcpyDeviceToHost;
    }

    static constexpr memcpy_kind_t host_to_device_kind()
    {
        return cudaMemcpyHostToDevice;
    }

    static void memcpy_3d_async( memcpy_3d_params_t *params, stream_t stream )
    {
        CUDA_SAFE_CALL( cudaMemcpy3DAsync( params, stream ) );
    }

    static void memcpy_async( void *dst, const void *src, size_t bytes, memcpy_kind_t kind, stream_t stream )
    {
        CUDA_SAFE_CALL( cudaMemcpyAsync( dst, src, bytes, kind, stream ) );
    }

    static void memcpy( void *dst, const void *src, size_t bytes, memcpy_kind_t kind )
    {
        CUDA_SAFE_CALL( cudaMemcpy( dst, src, bytes, kind ) );
    }

    static void device_synchronize()
    {
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    }

    static int get_device()
    {
        int device = 0;
        CUDA_SAFE_CALL( cudaGetDevice( &device ) );
        return device;
    }

    static device_memory_info_type get_device_memory_info()
    {
        return scfd::backend::cuda::get_device_memory_info();
    }

    static void set_device( int device )
    {
        CUDA_SAFE_CALL( cudaSetDevice( device ) );
    }

    static void stream_synchronize( stream_t stream )
    {
        CUDA_SAFE_CALL( cudaStreamSynchronize( stream ) );
    }

    static stream_t default_stream()
    {
        return stream_t{};
    }

    static event_t event_create()
    {
        event_t event = nullptr;
        CUDA_SAFE_CALL( cudaEventCreate( &event ) );
        return event;
    }

    static void event_destroy( event_t event )
    {
        if ( event != nullptr )
            CUDA_SAFE_CALL( cudaEventDestroy( event ) );
    }

    static void event_record( event_t event, stream_t stream )
    {
        CUDA_SAFE_CALL( cudaEventRecord( event, stream ) );
    }

    static void event_synchronize( event_t event )
    {
        CUDA_SAFE_CALL( cudaEventSynchronize( event ) );
    }

    static double event_elapsed_time_ms( event_t start, event_t stop )
    {
        float ms = 0.0f;
        CUDA_SAFE_CALL( cudaEventElapsedTime( &ms, start, stop ) );
        return static_cast<double>( ms );
    }

    static bool stream_ready( stream_t stream )
    {
        const cudaError_t err = cudaStreamQuery( stream );
        if ( err == cudaSuccess )
            return true;
        if ( err == cudaErrorNotReady )
            return false;
        CUDA_SAFE_CALL( err );
        return false;
    }

    static void launch_host_func( stream_t stream, host_func_t func, void *data )
    {
        CUDA_SAFE_CALL( cudaLaunchHostFunc( stream, func, data ) );
    }
};

namespace cufft
{
namespace detail
{


template <typename T>
struct fft_complex;

template <>
struct fft_complex<float>
{
    using complex = cufftComplex;
};

template <>
struct fft_complex<double>
{
    using complex = cufftDoubleComplex;
};

template <typename T>
struct fft_r2c;

template <>
struct fft_r2c<float>
{
    static constexpr cufftType type = CUFFT_R2C;
    using in_type                   = float;
    using out_type                  = cufftComplex;

    static cufftResult exec( cufftHandle &plan, in_type *in, out_type *out )
    {
        return cufftExecR2C( plan, in, out );
    }
};

template <>
struct fft_r2c<double>
{
    static constexpr cufftType type = CUFFT_D2Z;
    using in_type                   = double;
    using out_type                  = cufftDoubleComplex;

    static cufftResult exec( cufftHandle &plan, in_type *in, out_type *out )
    {
        return cufftExecD2Z( plan, in, out );
    }
};

// C2R
template <typename T>
struct fft_c2r;

template <>
struct fft_c2r<float>
{
    static constexpr cufftType type = CUFFT_C2R;
    using in_type                   = cufftComplex;
    using out_type                  = float;

    static cufftResult exec( cufftHandle &plan, in_type *in, out_type *out )
    {
        return cufftExecC2R( plan, in, out );
    }
};

template <>
struct fft_c2r<double>
{
    static constexpr cufftType type = CUFFT_Z2D;
    using in_type                   = cufftDoubleComplex;
    using out_type                  = double;

    static cufftResult exec( cufftHandle &plan, in_type *in, out_type *out )
    {
        return cufftExecZ2D( plan, in, out );
    }
};

// C2C forward
template <typename T>
struct fft_c2cf;

template <>
struct fft_c2cf<float>
{
    static constexpr cufftType type = CUFFT_C2C;
    using in_type                   = cufftComplex;
    using out_type                  = cufftComplex;

    static cufftResult exec( cufftHandle &plan, in_type *in, out_type *out )
    {
        return cufftExecC2C( plan, in, out, CUFFT_FORWARD );
    }
};

template <>
struct fft_c2cf<double>
{
    static constexpr cufftType type = CUFFT_Z2Z;
    using in_type                   = cufftDoubleComplex;
    using out_type                  = cufftDoubleComplex;

    static cufftResult exec( cufftHandle &plan, in_type *in, out_type *out )
    {
        return cufftExecZ2Z( plan, in, out, CUFFT_FORWARD );
    }
};

// C2C backward
template <typename T>
struct fft_c2cb;

template <>
struct fft_c2cb<float>
{
    static constexpr cufftType type = CUFFT_C2C;
    using in_type                   = cufftComplex;
    using out_type                  = cufftComplex;

    static cufftResult exec( cufftHandle &plan, in_type *in, out_type *out )
    {
        return cufftExecC2C( plan, in, out, CUFFT_INVERSE );
    }
};

template <>
struct fft_c2cb<double>
{
    static constexpr cufftType type = CUFFT_Z2Z;
    using in_type                   = cufftDoubleComplex;
    using out_type                  = cufftDoubleComplex;

    static cufftResult exec( cufftHandle &plan, in_type *in, out_type *out )
    {
        return cufftExecZ2Z( plan, in, out, CUFFT_INVERSE );
    }
};

template <typename T>
struct fft_c2c_direct_raw;

template <>
struct fft_c2c_direct_raw<float>
{
    static cufftResult exec( cufftHandle plan, void *in, void *out, int direction )
    {
        return cufftExecC2C(
            plan, static_cast<cufftComplex *>( in ), static_cast<cufftComplex *>( out ), direction
        );
    }
};

template <>
struct fft_c2c_direct_raw<double>
{
    static cufftResult exec( cufftHandle plan, void *in, void *out, int direction )
    {
        return cufftExecZ2Z(
            plan, static_cast<cufftDoubleComplex *>( in ), static_cast<cufftDoubleComplex *>( out ), direction
        );
    }
};


template <typename T, direction D>
struct fft_traits;

template <typename T>
struct fft_traits<T, direction::R2C> : fft_r2c<T>
{
};

template <typename T>
struct fft_traits<T, direction::C2R> : fft_c2r<T>
{
};

template <typename T>
struct fft_traits<T, direction::C2CF> : fft_c2cf<T>
{
};

template <typename T>
struct fft_traits<T, direction::C2CB> : fft_c2cb<T>
{
};


}


template <typename T>
class fft_base
{
public:
    using plan_descriptor = fft_plan_descriptor;
    using complex     = typename detail::fft_complex<T>::complex;
    using memory_type = typename cuda_runtime_api::memory_type;
    using runtime_api = cuda_runtime_api;
    virtual ~fft_base()
    {
    }
    // virtual cufftResult exec(void* in, void* out) = 0; //check error here
    virtual void        exec( void *in, void *out )      = 0;
    virtual void        exec_no_sync( void *in, void *out ) = 0;
    virtual void        exec_direction( direction exec_dir, void *in, void *out ) = 0;
    virtual void        exec_direction_no_sync( direction exec_dir, void *in, void *out ) = 0;
    virtual direction   get_direction() const            = 0;
    virtual plan_descriptor descriptor() const            = 0;
    virtual std::size_t get_work_size() const            = 0;
    virtual void        set_work_area( void *work_area ) = 0;
    virtual void        set_stream( typename runtime_api::stream_t stream ) = 0;
    virtual void recreate_with_stream_and_work_area(
        typename runtime_api::stream_t stream, void *work_area
    ) = 0;
};


template <typename T, ::fftm::direction D>
class fft : public fft_base<T>
{
    using traits = detail::fft_traits<T, D>;

public:
    using base_t   = fft_base<T>;
    using in_type  = typename traits::in_type;
    using out_type = typename traits::out_type;
    using opaque_plan_handle_t = cufftHandle;
    struct minimal_reference_c2c_plan_array_t
    {
        std::vector<cufftHandle> handles;
        std::vector<cudaStream_t> streams;
        std::vector<std::size_t>  offsets;
        fft_plan_descriptor       descriptor;
        std::size_t               work_stride_bytes = 0;
    };
    using minimal_reference_c2c_plan_array_handle_t = minimal_reference_c2c_plan_array_t *;


    explicit fft(
        long long int n, long long int inembed, long long int istride, long long int idist, long long int onembed,
        long long int ostride, long long int odist, long long int batch
    )
        : fft( std::array<long long int, 1>{ n }, std::array<long long int, 1>{ inembed }, istride, idist,
               std::array<long long int, 1>{ onembed }, ostride, odist, batch )
    {
    }

    explicit fft(
        long long int n0, long long int n1, long long int inembed0, long long int inembed1, long long int istride,
        long long int idist, long long int onembed0, long long int onembed1, long long int ostride, long long int odist,
        long long int batch
    )
        : fft( std::array<long long int, 2>{ n0, n1 }, std::array<long long int, 2>{ inembed0, inembed1 }, istride,
               idist, std::array<long long int, 2>{ onembed0, onembed1 }, ostride, odist, batch )
    {
    }

    explicit fft(
        long long int n0, long long int n1, long long int n2, long long int inembed0, long long int inembed1,
        long long int inembed2, long long int istride, long long int idist, long long int onembed0,
        long long int onembed1, long long int onembed2, long long int ostride, long long int odist, long long int batch
    )
        : fft( std::array<long long int, 3>{ n0, n1, n2 }, std::array<long long int, 3>{ inembed0, inembed1, inembed2 },
               istride, idist, std::array<long long int, 3>{ onembed0, onembed1, onembed2 }, ostride, odist, batch )
    {
    }

    template <std::size_t Rank>
    explicit fft(
        const std::array<long long int, Rank> &n, const std::array<long long int, Rank> &inembed, long long int istride,
        long long int idist, const std::array<long long int, Rank> &onembed, long long int ostride, long long int odist,
        long long int batch
    )
        : handle_( 0 ), work_size( 0 )
    {
        static_assert( Rank >= 1 && Rank <= 3, "cufft::fft supports only 1D, 2D, and 3D plans" );

        auto n_local       = n;
        auto inembed_local = inembed;
        auto onembed_local = onembed;

        descriptor_.dir        = D;
        descriptor_.cufft_type = static_cast<int>( get_fft_direction( D ) );
        descriptor_.rank       = static_cast<int>( Rank );
        descriptor_.istride    = istride;
        descriptor_.idist      = idist;
        descriptor_.ostride    = ostride;
        descriptor_.odist      = odist;
        descriptor_.batch      = batch;
        for ( std::size_t i = 0; i < Rank; ++i )
        {
            descriptor_.n[i]       = n[i];
            descriptor_.inembed[i] = inembed[i];
            descriptor_.onembed[i] = onembed[i];
        }

        create_plan(
            static_cast<int>( Rank ), n_local.data(), inembed_local.data(), istride, idist, onembed_local.data(),
            ostride, odist, batch
        );
    }

    template <std::size_t Rank>
    explicit fft( const std::array<long long int, Rank> &n, long long int batch )
        : handle_( 0 ), work_size( 0 )
    {
        static_assert( Rank >= 1 && Rank <= 3, "cufft::fft supports only 1D, 2D, and 3D plans" );

        auto n_local = n;

        descriptor_.dir            = D;
        descriptor_.cufft_type     = static_cast<int>( get_fft_direction( D ) );
        descriptor_.rank           = static_cast<int>( Rank );
        descriptor_.default_layout = true;
        descriptor_.batch          = batch;
        for ( std::size_t i = 0; i < Rank; ++i )
        {
            descriptor_.n[i] = n[i];
        }

        create_plan(
            static_cast<int>( Rank ), n_local.data(), nullptr, 0, 0, nullptr, 0, 0, batch
        );
    }

    ~fft() override
    {
        if ( handle_ != 0 )
        {
            cufftDestroy( handle_ );
        }
    }

    virtual std::size_t get_work_size() const override
    {
        return work_size;
    }


    virtual void set_work_area( void *work_area ) override
    {
        descriptor_.work_area_token = reinterpret_cast<std::uintptr_t>( work_area );
        CUFFT_SAFE_CALL( cufftSetWorkArea( handle_, work_area ) );
    }

    virtual void set_stream( typename base_t::runtime_api::stream_t stream ) override
    {
        descriptor_.stream_token = reinterpret_cast<std::uintptr_t>( stream );
        CUFFT_SAFE_CALL( cufftSetStream( handle_, stream ) );
    }

    virtual void recreate_with_stream_and_work_area(
        typename base_t::runtime_api::stream_t stream, void *work_area
    ) override
    {
        if ( handle_ != 0 )
        {
            CUFFT_SAFE_CALL( cufftDestroy( handle_ ) );
            handle_ = 0;
        }

        auto n_local       = descriptor_.n;
        auto inembed_local = descriptor_.inembed;
        auto onembed_local = descriptor_.onembed;
        if ( descriptor_.default_layout )
        {
            create_plan(
                descriptor_.rank, n_local.data(), nullptr, 0, 0, nullptr, 0, 0, descriptor_.batch
            );
        }
        else
        {
            create_plan(
                descriptor_.rank, n_local.data(), inembed_local.data(), descriptor_.istride, descriptor_.idist,
                onembed_local.data(), descriptor_.ostride, descriptor_.odist, descriptor_.batch
            );
        }

        set_stream( stream );
        set_work_area( work_area );
    }


    void exec_typed( void *in, void *out )
    {
        CUFFT_SAFE_CALL( traits::exec( handle_, static_cast<in_type *>( in ), static_cast<out_type *>( out ) ) );
    }

    void exec_typed_no_sync( void *in, void *out )
    {
        cufft_check_no_sync(
            traits::exec( handle_, static_cast<in_type *>( in ), static_cast<out_type *>( out ) ),
            "cufft_wrap::fft::exec_typed_no_sync"
        );
    }

    void exec_typed_direction( direction exec_dir, void *in, void *out )
    {
        if ( exec_dir == D )
        {
            exec_typed( in, out );
            return;
        }
        if ( ( D == direction::C2CF || D == direction::C2CB ) && exec_dir == direction::C2CF )
        {
            CUFFT_SAFE_CALL( detail::fft_c2cf<T>::exec(
                handle_, static_cast<typename detail::fft_c2cf<T>::in_type *>( in ),
                static_cast<typename detail::fft_c2cf<T>::out_type *>( out )
            ) );
            return;
        }
        if ( ( D == direction::C2CF || D == direction::C2CB ) && exec_dir == direction::C2CB )
        {
            CUFFT_SAFE_CALL( detail::fft_c2cb<T>::exec(
                handle_, static_cast<typename detail::fft_c2cb<T>::in_type *>( in ),
                static_cast<typename detail::fft_c2cb<T>::out_type *>( out )
            ) );
            return;
        }
        throw std::logic_error( "cufft_wrap::fft::exec_typed_direction: incompatible execution direction" );
    }

    void exec_typed_direction_no_sync( direction exec_dir, void *in, void *out )
    {
        if ( exec_dir == D )
        {
            exec_typed_no_sync( in, out );
            return;
        }
        if ( ( D == direction::C2CF || D == direction::C2CB ) && exec_dir == direction::C2CF )
        {
            cufft_check_no_sync(
                detail::fft_c2cf<T>::exec(
                    handle_, static_cast<typename detail::fft_c2cf<T>::in_type *>( in ),
                    static_cast<typename detail::fft_c2cf<T>::out_type *>( out )
                ),
                "cufft_wrap::fft::exec_typed_direction_no_sync(C2CF)"
            );
            return;
        }
        if ( ( D == direction::C2CF || D == direction::C2CB ) && exec_dir == direction::C2CB )
        {
            cufft_check_no_sync(
                detail::fft_c2cb<T>::exec(
                    handle_, static_cast<typename detail::fft_c2cb<T>::in_type *>( in ),
                    static_cast<typename detail::fft_c2cb<T>::out_type *>( out )
                ),
                "cufft_wrap::fft::exec_typed_direction_no_sync(C2CB)"
            );
            return;
        }
        throw std::logic_error( "cufft_wrap::fft::exec_typed_direction_no_sync: incompatible execution direction" );
    }

    opaque_plan_handle_t opaque_plan_handle() const
    {
        return handle_;
    }

    static opaque_plan_handle_t create_empty_opaque_plan()
    {
        opaque_plan_handle_t handle = 0;
        CUFFT_SAFE_CALL( cufftCreate( &handle ) );
        try
        {
            CUFFT_SAFE_CALL( cufftSetAutoAllocation( handle, 0 ) );
        }
        catch ( ... )
        {
            cufftDestroy( handle );
            throw;
        }
        return handle;
    }

    static std::size_t make_opaque_1d_default(
        opaque_plan_handle_t plan, long long int n, long long int batch
    )
    {
        long long int n_local[1]  = { n };
        std::size_t   work_size_out = 0;
        CUFFT_SAFE_CALL( cufftMakePlanMany64(
            plan, 1, n_local, nullptr, 0, 0, nullptr, 0, 0, get_fft_direction_static_( D ), batch,
            &work_size_out
        ) );
        return work_size_out;
    }

    static std::size_t make_opaque_1d(
        opaque_plan_handle_t plan, long long int n, long long int inembed, long long int istride,
        long long int idist, long long int onembed, long long int ostride, long long int odist,
        long long int batch
    )
    {
        long long int n_local[1]       = { n };
        long long int inembed_local[1] = { inembed };
        long long int onembed_local[1] = { onembed };
        std::size_t   work_size_out    = 0;
        CUFFT_SAFE_CALL( cufftMakePlanMany64(
            plan, 1, n_local, inembed_local, istride, idist, onembed_local, ostride, odist,
            get_fft_direction_static_( D ), batch, &work_size_out
        ) );
        return work_size_out;
    }

    static opaque_plan_handle_t create_opaque_c2c_1d(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        std::size_t &work_size_out
    )
    {
        return create_opaque_c2c_1d_with_optional_stream(
            n, inembed, istride, idist, onembed, ostride, odist, batch, work_size_out, nullptr
        );
    }

    static opaque_plan_handle_t create_opaque_c2c_1d_with_stream(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        std::size_t &work_size_out, typename base_t::runtime_api::stream_t stream
    )
    {
        return create_opaque_c2c_1d_with_optional_stream(
            n, inembed, istride, idist, onembed, ostride, odist, batch, work_size_out, stream
        );
    }

    static opaque_plan_handle_t create_direct_raw_reference_c2c_1d_with_stream(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        std::size_t &work_size_out, typename base_t::runtime_api::stream_t stream
    )
    {
        opaque_plan_handle_t handle = 0;
        long long int        n_arr[1]       = { n };
        long long int        inembed_arr[1] = { inembed };
        long long int        onembed_arr[1] = { onembed };
        work_size_out                         = 0;

        CUFFT_SAFE_CALL( cufftCreate( &handle ) );
        try
        {
            CUFFT_SAFE_CALL( cufftSetAutoAllocation( handle, 0 ) );
            CUFFT_SAFE_CALL( cufftMakePlanMany64(
                handle, 1, n_arr, inembed_arr, istride, idist, onembed_arr, ostride, odist,
                detail::fft_c2cf<T>::type, batch, &work_size_out
            ) );
            CUFFT_SAFE_CALL( cufftSetStream( handle, stream ) );
        }
        catch ( ... )
        {
            cufftDestroy( handle );
            throw;
        }
        return handle;
    }

    static minimal_reference_c2c_plan_array_handle_t create_minimal_reference_c2c_plan_array_1d(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        const std::vector<std::size_t> &offsets
    )
    {
        auto *bundle = new minimal_reference_c2c_plan_array_t();
        bundle->offsets = offsets;
        bundle->handles.reserve( offsets.size() );
        bundle->streams.reserve( offsets.size() );
        bundle->descriptor.dir            = ::fftm::direction::C2CF;
        bundle->descriptor.cufft_type     = detail::fft_c2cf<T>::type;
        bundle->descriptor.rank           = 1;
        bundle->descriptor.default_layout = false;
        bundle->descriptor.n[0]           = n;
        bundle->descriptor.inembed[0]     = inembed;
        bundle->descriptor.istride        = istride;
        bundle->descriptor.idist          = idist;
        bundle->descriptor.onembed[0]     = onembed;
        bundle->descriptor.ostride        = ostride;
        bundle->descriptor.odist          = odist;
        bundle->descriptor.batch          = batch;

        long long int n_arr[1]       = { n };
        long long int inembed_arr[1] = { inembed };
        long long int onembed_arr[1] = { onembed };
        try
        {
            for ( std::size_t i = 0; i < offsets.size(); ++i )
            {
                cudaStream_t stream = nullptr;
                CUDA_SAFE_CALL( cudaStreamCreate( &stream ) );
                bundle->streams.push_back( stream );

                cufftHandle handle = 0;
                CUFFT_SAFE_CALL( cufftCreate( &handle ) );
                bundle->handles.push_back( handle );
                CUFFT_SAFE_CALL( cufftSetAutoAllocation( handle, 0 ) );
                std::size_t work_size = 0;
                CUFFT_SAFE_CALL( cufftMakePlanMany64(
                    handle, 1, n_arr, inembed_arr, istride, idist, onembed_arr, ostride, odist,
                    detail::fft_c2cf<T>::type, batch, &work_size
                ) );
                CUFFT_SAFE_CALL( cufftSetStream( handle, stream ) );
                bundle->work_stride_bytes = std::max( bundle->work_stride_bytes, work_size );
            }
            bundle->descriptor.work_size = bundle->work_stride_bytes;
        }
        catch ( ... )
        {
            destroy_minimal_reference_c2c_plan_array_noexcept( bundle );
            throw;
        }
        return bundle;
    }

private:
    static opaque_plan_handle_t create_opaque_c2c_1d_with_optional_stream(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        std::size_t &work_size_out, typename base_t::runtime_api::stream_t stream
    )
    {
        opaque_plan_handle_t handle = 0;
        long long int        n_local[1]       = { n };
        long long int        inembed_local[1] = { inembed };
        long long int        onembed_local[1] = { onembed };
        work_size_out                         = 0;

        CUFFT_SAFE_CALL( cufftCreate( &handle ) );
        try
        {
            CUFFT_SAFE_CALL( cufftSetAutoAllocation( handle, 0 ) );
            CUFFT_SAFE_CALL( cufftMakePlanMany64(
                handle, 1, n_local, inembed_local, istride, idist, onembed_local, ostride, odist,
                detail::fft_c2cf<T>::type, batch, &work_size_out
            ) );
            if ( stream != nullptr )
            {
                CUFFT_SAFE_CALL( cufftSetStream( handle, stream ) );
            }
        }
        catch ( ... )
        {
            cufftDestroy( handle );
            throw;
        }
        return handle;
    }

public:
    static void destroy_opaque_plan_noexcept( opaque_plan_handle_t plan )
    {
        if ( plan != 0 )
            cufftDestroy( plan );
    }

    static void destroy_minimal_reference_c2c_plan_array_noexcept(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        if ( bundle == nullptr )
            return;
        for ( cufftHandle handle : bundle->handles )
        {
            if ( handle != 0 )
                cufftDestroy( handle );
        }
        for ( cudaStream_t stream : bundle->streams )
        {
            if ( stream != nullptr )
                cudaStreamDestroy( stream );
        }
        delete bundle;
    }

    static std::size_t minimal_reference_c2c_plan_array_size(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        return bundle == nullptr ? 0 : bundle->handles.size();
    }

    static std::size_t minimal_reference_c2c_plan_array_work_size(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        return bundle == nullptr ? 0 : bundle->work_stride_bytes;
    }

    static fft_plan_descriptor minimal_reference_c2c_plan_array_descriptor(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        if ( bundle == nullptr )
            throw std::logic_error( "cufft_wrap::fft::minimal_reference_c2c_plan_array_descriptor: null bundle" );
        return bundle->descriptor;
    }

    static std::uintptr_t minimal_reference_c2c_plan_array_first_handle_token(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        return bundle == nullptr || bundle->handles.empty() ? 0 : static_cast<std::uintptr_t>( bundle->handles.front() );
    }

    static std::uintptr_t minimal_reference_c2c_plan_array_last_handle_token(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        return bundle == nullptr || bundle->handles.empty() ? 0 : static_cast<std::uintptr_t>( bundle->handles.back() );
    }

    static std::uintptr_t minimal_reference_c2c_plan_array_first_stream_token(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        return bundle == nullptr || bundle->streams.empty()
                   ? 0
                   : reinterpret_cast<std::uintptr_t>( bundle->streams.front() );
    }

    static std::uintptr_t minimal_reference_c2c_plan_array_last_stream_token(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        return bundle == nullptr || bundle->streams.empty()
                   ? 0
                   : reinterpret_cast<std::uintptr_t>( bundle->streams.back() );
    }

    static void bind_minimal_reference_c2c_plan_array_work_areas(
        minimal_reference_c2c_plan_array_handle_t bundle, void *base_work_area, std::size_t work_stride_bytes
    )
    {
        if ( bundle == nullptr )
            throw std::logic_error( "cufft_wrap::fft::bind_minimal_reference_c2c_plan_array_work_areas: null bundle" );
        char *raw = static_cast<char *>( base_work_area );
        const std::size_t stride = std::max( work_stride_bytes, bundle->work_stride_bytes );
        for ( std::size_t i = 0; i < bundle->handles.size(); ++i )
        {
            CUFFT_SAFE_CALL( cufftSetWorkArea( bundle->handles[i], static_cast<void *>( raw + i * stride ) ) );
        }
    }

    static void synchronize_minimal_reference_c2c_plan_array_streams(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        if ( bundle == nullptr )
            throw std::logic_error( "cufft_wrap::fft::synchronize_minimal_reference_c2c_plan_array_streams: null bundle" );
        for ( cudaStream_t stream : bundle->streams )
        {
            CUDA_SAFE_CALL( cudaStreamSynchronize( stream ) );
        }
    }

    static void set_opaque_stream( opaque_plan_handle_t plan, typename base_t::runtime_api::stream_t stream )
    {
        CUFFT_SAFE_CALL( cufftSetStream( plan, stream ) );
    }

    static void set_opaque_work_area( opaque_plan_handle_t plan, void *work_area )
    {
        CUFFT_SAFE_CALL( cufftSetWorkArea( plan, work_area ) );
    }

    static void set_direct_raw_work_area( opaque_plan_handle_t plan, void *work_area )
    {
        CUFFT_SAFE_CALL( cufftSetWorkArea( plan, work_area ) );
    }

    static void exec_opaque_c2c( opaque_plan_handle_t plan, direction exec_dir, void *in, void *out )
    {
        if ( exec_dir == direction::C2CF )
        {
            CUFFT_SAFE_CALL( detail::fft_c2cf<T>::exec(
                plan, static_cast<typename detail::fft_c2cf<T>::in_type *>( in ),
                static_cast<typename detail::fft_c2cf<T>::out_type *>( out )
            ) );
            return;
        }
        if ( exec_dir == direction::C2CB )
        {
            CUFFT_SAFE_CALL( detail::fft_c2cb<T>::exec(
                plan, static_cast<typename detail::fft_c2cb<T>::in_type *>( in ),
                static_cast<typename detail::fft_c2cb<T>::out_type *>( out )
            ) );
            return;
        }
        throw std::logic_error( "cufft_wrap::fft::exec_opaque_c2c: incompatible execution direction" );
    }

    static void exec_opaque_c2c_no_sync( opaque_plan_handle_t plan, direction exec_dir, void *in, void *out )
    {
        if ( exec_dir == direction::C2CF )
        {
            cufft_check_no_sync(
                detail::fft_c2cf<T>::exec(
                    plan, static_cast<typename detail::fft_c2cf<T>::in_type *>( in ),
                    static_cast<typename detail::fft_c2cf<T>::out_type *>( out )
                ),
                "cufft_wrap::fft::exec_opaque_c2c_no_sync(C2CF)"
            );
            return;
        }
        if ( exec_dir == direction::C2CB )
        {
            cufft_check_no_sync(
                detail::fft_c2cb<T>::exec(
                    plan, static_cast<typename detail::fft_c2cb<T>::in_type *>( in ),
                    static_cast<typename detail::fft_c2cb<T>::out_type *>( out )
                ),
                "cufft_wrap::fft::exec_opaque_c2c_no_sync(C2CB)"
            );
            return;
        }
        throw std::logic_error( "cufft_wrap::fft::exec_opaque_c2c_no_sync: incompatible execution direction" );
    }

    static void exec_direct_raw_c2c( opaque_plan_handle_t plan, direction exec_dir, void *in, void *out )
    {
        if ( exec_dir == direction::C2CF )
        {
            CUFFT_SAFE_CALL( detail::fft_c2c_direct_raw<T>::exec( plan, in, out, CUFFT_FORWARD ) );
            return;
        }
        if ( exec_dir == direction::C2CB )
        {
            CUFFT_SAFE_CALL( detail::fft_c2c_direct_raw<T>::exec( plan, in, out, CUFFT_INVERSE ) );
            return;
        }
        throw std::logic_error( "cufft_wrap::fft::exec_direct_raw_c2c: incompatible execution direction" );
    }

    static void exec_direct_raw_c2c_no_sync( opaque_plan_handle_t plan, direction exec_dir, void *in, void *out )
    {
        if ( exec_dir == direction::C2CF )
        {
            cufft_check_no_sync(
                detail::fft_c2c_direct_raw<T>::exec( plan, in, out, CUFFT_FORWARD ),
                "cufft_wrap::fft::exec_direct_raw_c2c_no_sync(C2CF)"
            );
            return;
        }
        if ( exec_dir == direction::C2CB )
        {
            cufft_check_no_sync(
                detail::fft_c2c_direct_raw<T>::exec( plan, in, out, CUFFT_INVERSE ),
                "cufft_wrap::fft::exec_direct_raw_c2c_no_sync(C2CB)"
            );
            return;
        }
        throw std::logic_error( "cufft_wrap::fft::exec_direct_raw_c2c_no_sync: incompatible execution direction" );
    }

    static void exec_minimal_reference_c2c_plan_array(
        minimal_reference_c2c_plan_array_handle_t bundle, direction exec_dir, void *in, void *out
    )
    {
        if ( bundle == nullptr )
            throw std::logic_error( "cufft_wrap::fft::exec_minimal_reference_c2c_plan_array: null bundle" );
        int cufft_direction = 0;
        if ( exec_dir == direction::C2CF )
            cufft_direction = CUFFT_FORWARD;
        else if ( exec_dir == direction::C2CB )
            cufft_direction = CUFFT_INVERSE;
        else
            throw std::logic_error(
                "cufft_wrap::fft::exec_minimal_reference_c2c_plan_array: incompatible execution direction"
            );
        if ( bundle->offsets.size() != bundle->handles.size() )
            throw std::logic_error( "cufft_wrap::fft::exec_minimal_reference_c2c_plan_array: offset count mismatch" );
        for ( std::size_t i = 0; i < bundle->handles.size(); ++i )
        {
            CUFFT_SAFE_CALL( detail::fft_c2c_direct_raw<T>::exec(
                bundle->handles[i],
                static_cast<void *>( static_cast<typename base_t::complex *>( in ) + bundle->offsets[i] ),
                static_cast<void *>( static_cast<typename base_t::complex *>( out ) + bundle->offsets[i] ),
                cufft_direction
            ) );
        }
    }

    static void exec_minimal_reference_c2c_plan_array_no_sync(
        minimal_reference_c2c_plan_array_handle_t bundle, direction exec_dir, void *in, void *out
    )
    {
        if ( bundle == nullptr )
            throw std::logic_error( "cufft_wrap::fft::exec_minimal_reference_c2c_plan_array_no_sync: null bundle" );
        int cufft_direction = 0;
        if ( exec_dir == direction::C2CF )
            cufft_direction = CUFFT_FORWARD;
        else if ( exec_dir == direction::C2CB )
            cufft_direction = CUFFT_INVERSE;
        else
            throw std::logic_error(
                "cufft_wrap::fft::exec_minimal_reference_c2c_plan_array_no_sync: incompatible execution direction"
            );
        if ( bundle->offsets.size() != bundle->handles.size() )
            throw std::logic_error(
                "cufft_wrap::fft::exec_minimal_reference_c2c_plan_array_no_sync: offset count mismatch"
            );
        for ( std::size_t i = 0; i < bundle->handles.size(); ++i )
        {
            cufft_check_no_sync(
                detail::fft_c2c_direct_raw<T>::exec(
                    bundle->handles[i],
                    static_cast<void *>( static_cast<typename base_t::complex *>( in ) + bundle->offsets[i] ),
                    static_cast<void *>( static_cast<typename base_t::complex *>( out ) + bundle->offsets[i] ),
                    cufft_direction
                ),
                "cufft_wrap::fft::exec_minimal_reference_c2c_plan_array_no_sync"
            );
        }
    }

    static void exec_minimal_reference_c2c_plan_array_repeated(
        minimal_reference_c2c_plan_array_handle_t bundle, direction exec_dir, void *in, void *out,
        std::size_t repeats
    )
    {
        for ( std::size_t repeat = 0; repeat < repeats; ++repeat )
        {
            exec_minimal_reference_c2c_plan_array( bundle, exec_dir, in, out );
        }
    }

    static void exec_minimal_reference_c2c_plan_array_no_sync_repeated(
        minimal_reference_c2c_plan_array_handle_t bundle, direction exec_dir, void *in, void *out,
        std::size_t repeats
    )
    {
        for ( std::size_t repeat = 0; repeat < repeats; ++repeat )
        {
            exec_minimal_reference_c2c_plan_array_no_sync( bundle, exec_dir, in, out );
        }
    }

    virtual void exec( void *in, void *out ) override
    {
        exec_typed( in, out );
    }

    virtual void exec_no_sync( void *in, void *out ) override
    {
        exec_typed_no_sync( in, out );
    }

    virtual void exec_direction( direction exec_dir, void *in, void *out ) override
    {
        exec_typed_direction( exec_dir, in, out );
    }

    virtual void exec_direction_no_sync( direction exec_dir, void *in, void *out ) override
    {
        exec_typed_direction_no_sync( exec_dir, in, out );
    }

    virtual direction get_direction() const override
    {
        return D;
    }

    virtual fft_plan_descriptor descriptor() const override
    {
        return descriptor_;
    }

private:
    cufftHandle handle_;
    std::size_t work_size;
    fft_plan_descriptor descriptor_;

    void create_plan(
        int rank, long long int *n, long long int *inembed, long long int istride, long long int idist,
        long long int *onembed, long long int ostride, long long int odist, long long int batch
    )
    {
        CUFFT_SAFE_CALL( cufftCreate( &handle_ ) );

        try
        {
            CUFFT_SAFE_CALL( cufftSetAutoAllocation( handle_, 0 ) );
            CUFFT_SAFE_CALL( cufftMakePlanMany64(
                handle_, rank, n, inembed, istride, idist, onembed, ostride, odist, get_fft_direction( D ), batch,
                &work_size
            ) );
            descriptor_.work_size = work_size;
        }
        catch ( ... )
        {
            cufftDestroy( handle_ );
            handle_ = 0;
            throw;
        }
    }

    cufftType get_fft_direction( direction dir )
    {
        return get_fft_direction_static_( dir );
    }

    static cufftType get_fft_direction_static_( direction dir )
    {
        switch ( dir )
        {
        case direction::R2C:
            return detail::fft_r2c<T>::type;
        case direction::C2R:
            return detail::fft_c2r<T>::type;
        case direction::C2CF:
            return detail::fft_c2cf<T>::type;
        case direction::C2CB:
            return detail::fft_c2cb<T>::type;
        }

        return detail::fft_c2cf<T>::type; //to avoid warning
    }
};

}
}
}


#endif // __FFTM_CUFFT_WRAP_H__
