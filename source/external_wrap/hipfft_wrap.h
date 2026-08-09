#ifndef __FFTM_HIPFFT_WRAP_H__
#define __FFTM_HIPFFT_WRAP_H__

#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <hipfft/hipfft.h>
#include <scfd/backend/hip.h>
#include <scfd/memory/hip.h>
#include <scfd/utils/todo.h>
#include <scfd/utils/hip_safe_call.h>

#include "fft_plan_descriptor.h"
#include "hip_stream_wrap.h"
#include "hipfft_safe_call.h"
#include "runtime_api_types.h"
#include "../detail/runtime_hardware_identity.h"

namespace fftm
{
namespace wrap
{

inline void hipfft_check_no_sync( hipfftResult status, const char *expr )
{
    if ( status != HIPFFT_SUCCESS )
    {
        throw std::runtime_error(
            std::string( "HIPFFT_NO_SYNC_CHECK: " ) + expr + " failed: " +
            std::string( ::fftm::wrap::detail::hipfft_error_string( status ) )
        );
    }
}

struct hip_runtime_api
{
    using memory_type        = scfd::memory::hip_device;
    using device_memory_info_type = typename scfd::backend::hip::device_memory_info_type;
    using stream_wrap        = ::fftm::wrap::hip_stream_wrap;
    using stream_t           = hipStream_t;
    using memcpy_kind_t      = memory_copy_kind;
    using memcpy_3d_params_t = memory_copy_3d_params;
    using host_func_t        = hipHostFn_t;
    using event_t            = hipEvent_t;
    using pos_t              = memory_position_3d;
    using pitched_ptr_t      = pitched_memory_pointer;
    using extent_t           = memory_extent_3d;

    static pos_t make_pos( size_t x, size_t y, size_t z )
    {
        return pos_t{ x, y, z };
    }

    static pitched_ptr_t make_pitched_ptr( void *ptr, size_t pitch, size_t xsz, size_t ysz )
    {
        return pitched_ptr_t{ ptr, pitch, xsz, ysz };
    }

    static pitched_ptr_t make_pitched_ptr( const void *ptr, size_t pitch, size_t xsz, size_t ysz )
    {
        return pitched_ptr_t{ const_cast<void *>( ptr ), pitch, xsz, ysz };
    }

    static extent_t make_extent( size_t width, size_t height, size_t depth )
    {
        return extent_t{ width, height, depth };
    }

    static constexpr memcpy_kind_t device_to_device_kind()
    {
        return memcpy_kind_t::device_to_device;
    }

    static constexpr memcpy_kind_t device_to_host_kind()
    {
        return memcpy_kind_t::device_to_host;
    }

    static constexpr memcpy_kind_t host_to_device_kind()
    {
        return memcpy_kind_t::host_to_device;
    }

    static void memcpy_3d_async( memcpy_3d_params_t *params, stream_t stream )
    {
        hipMemcpy3DParms native_params = {};
        native_params.srcPos = make_hipPos(
            params->source_position.x, params->source_position.y, params->source_position.z
        );
        native_params.srcPtr = make_hipPitchedPtr(
            params->source.pointer, params->source.pitch, params->source.x_size, params->source.y_size
        );
        native_params.dstPos = make_hipPos(
            params->destination_position.x, params->destination_position.y, params->destination_position.z
        );
        native_params.dstPtr = make_hipPitchedPtr(
            params->destination.pointer, params->destination.pitch, params->destination.x_size,
            params->destination.y_size
        );
        native_params.extent = make_hipExtent( params->extent.width, params->extent.height, params->extent.depth );
        native_params.kind   = native_copy_kind( params->kind );
        HIP_SAFE_CALL( hipMemcpy3DAsync( &native_params, stream ) );
    }

    static void memcpy_async( void *dst, const void *src, size_t bytes, memcpy_kind_t kind, stream_t stream )
    {
        HIP_SAFE_CALL( hipMemcpyAsync( dst, src, bytes, native_copy_kind( kind ), stream ) );
    }

    static void memcpy( void *dst, const void *src, size_t bytes, memcpy_kind_t kind )
    {
        HIP_SAFE_CALL( hipMemcpy( dst, src, bytes, native_copy_kind( kind ) ) );
    }

    static void memset_zero( void *dst, size_t bytes )
    {
        HIP_SAFE_CALL( hipMemset( dst, 0, bytes ) );
    }

    static void device_synchronize()
    {
        HIP_SAFE_CALL( hipDeviceSynchronize() );
    }

    static int get_device()
    {
        int device = 0;
        HIP_SAFE_CALL( hipGetDevice( &device ) );
        return device;
    }

    static device_memory_info_type get_device_memory_info()
    {
        return scfd::backend::hip::get_device_memory_info();
    }

    static ::fftm::detail::runtime_hardware_identity get_hardware_identity()
    {
        ::fftm::detail::runtime_hardware_identity result;
        result.backend = "hip";

        const int device = get_device();
        hipDeviceProp_t properties{};
        HIP_SAFE_CALL( hipGetDeviceProperties( &properties, device ) );
        result.device_name       = properties.name;
        result.architecture      = properties.gcnArchName;
        result.total_memory_bytes = static_cast<std::size_t>( properties.totalGlobalMem );
        result.total_memory_known = true;

        char pci_bus_id[32] = {};
        HIP_SAFE_CALL( hipDeviceGetPCIBusId( pci_bus_id, static_cast<int>( sizeof( pci_bus_id ) ), device ) );
        result.pci_bus_id = pci_bus_id;

#if defined( HIP_VERSION_MAJOR ) && HIP_VERSION_MAJOR >= 5
        hipUUID device_uuid{};
        HIP_SAFE_CALL( hipDeviceGetUuid( &device_uuid, device ) );
        std::ostringstream uuid;
        uuid << std::hex << std::setfill( '0' );
        for ( unsigned char byte : device_uuid.bytes )
            uuid << std::setw( 2 ) << static_cast<unsigned int>( byte );
        result.device_uuid = uuid.str();
#endif

        HIP_SAFE_CALL( hipRuntimeGetVersion( &result.runtime_version ) );
        HIP_SAFE_CALL( hipDriverGetVersion( &result.driver_version ) );
        return result;
    }

    static void set_device( int device )
    {
        HIP_SAFE_CALL( hipSetDevice( device ) );
    }

    static void stream_synchronize( stream_t stream )
    {
        HIP_SAFE_CALL( hipStreamSynchronize( stream ) );
    }

    static stream_t create_nonblocking_stream()
    {
        stream_t stream = nullptr;
        HIP_SAFE_CALL( hipStreamCreateWithFlags( &stream, hipStreamNonBlocking ) );
        return stream;
    }

    static void destroy_stream( stream_t stream )
    {
        if ( stream != nullptr )
            HIP_SAFE_CALL( hipStreamDestroy( stream ) );
    }

    static stream_t default_stream()
    {
        return stream_t{};
    }

    static event_t event_create()
    {
        event_t event = nullptr;
        HIP_SAFE_CALL( hipEventCreate( &event ) );
        return event;
    }

    static void event_destroy( event_t event )
    {
        if ( event != nullptr )
            HIP_SAFE_CALL( hipEventDestroy( event ) );
    }

    static void event_record( event_t event, stream_t stream )
    {
        HIP_SAFE_CALL( hipEventRecord( event, stream ) );
    }

    static void event_synchronize( event_t event )
    {
        HIP_SAFE_CALL( hipEventSynchronize( event ) );
    }

    static double event_elapsed_time_ms( event_t start, event_t stop )
    {
        float ms = 0.0f;
        HIP_SAFE_CALL( hipEventElapsedTime( &ms, start, stop ) );
        return static_cast<double>( ms );
    }

    static bool stream_ready( stream_t stream )
    {
        const hipError_t err = hipStreamQuery( stream );
        if ( err == hipSuccess )
            return true;
        if ( err == hipErrorNotReady )
            return false;
        HIP_SAFE_CALL( err );
        return false;
    }

    static void launch_host_func( stream_t stream, host_func_t func, void *data )
    {
        HIP_SAFE_CALL( hipLaunchHostFunc( stream, func, data ) );
    }

private:
    static constexpr hipMemcpyKind native_copy_kind( memcpy_kind_t kind )
    {
        return kind == memcpy_kind_t::device_to_device
                   ? hipMemcpyDeviceToDevice
                   : ( kind == memcpy_kind_t::device_to_host ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice );
    }
};

namespace hipfft
{
namespace detail
{

template <class Handle>
inline typename std::enable_if<std::is_pointer<Handle>::value, std::uintptr_t>::type
opaque_handle_token( Handle handle )
{
    return reinterpret_cast<std::uintptr_t>( handle );
}

template <class Handle>
inline typename std::enable_if<std::is_integral<Handle>::value, std::uintptr_t>::type
opaque_handle_token( Handle handle )
{
    return static_cast<std::uintptr_t>( handle );
}


template <typename T>
struct fft_complex;

template <>
struct fft_complex<float>
{
    using complex = ::fftm::wrap::complex_value<float>;
    static_assert( sizeof( complex ) == sizeof( hipfftComplex ), "neutral and hipFFT complex sizes differ" );
    static_assert( alignof( complex ) == alignof( hipfftComplex ), "neutral and hipFFT complex alignments differ" );
};

template <>
struct fft_complex<double>
{
    using complex = ::fftm::wrap::complex_value<double>;
    static_assert( sizeof( complex ) == sizeof( hipfftDoubleComplex ), "neutral and hipFFT complex sizes differ" );
    static_assert(
        alignof( complex ) == alignof( hipfftDoubleComplex ), "neutral and hipFFT complex alignments differ"
    );
};

template <typename T>
struct fft_r2c;

template <>
struct fft_r2c<float>
{
    static constexpr hipfftType type = HIPFFT_R2C;
    using in_type                   = float;
    using out_type                  = hipfftComplex;

    static hipfftResult exec( hipfftHandle &plan, in_type *in, out_type *out )
    {
        return hipfftExecR2C( plan, in, out );
    }
};

template <>
struct fft_r2c<double>
{
    static constexpr hipfftType type = HIPFFT_D2Z;
    using in_type                   = double;
    using out_type                  = hipfftDoubleComplex;

    static hipfftResult exec( hipfftHandle &plan, in_type *in, out_type *out )
    {
        return hipfftExecD2Z( plan, in, out );
    }
};

// C2R
template <typename T>
struct fft_c2r;

template <>
struct fft_c2r<float>
{
    static constexpr hipfftType type = HIPFFT_C2R;
    using in_type                   = hipfftComplex;
    using out_type                  = float;

    static hipfftResult exec( hipfftHandle &plan, in_type *in, out_type *out )
    {
        return hipfftExecC2R( plan, in, out );
    }
};

template <>
struct fft_c2r<double>
{
    static constexpr hipfftType type = HIPFFT_Z2D;
    using in_type                   = hipfftDoubleComplex;
    using out_type                  = double;

    static hipfftResult exec( hipfftHandle &plan, in_type *in, out_type *out )
    {
        return hipfftExecZ2D( plan, in, out );
    }
};

// C2C forward
template <typename T>
struct fft_c2cf;

template <>
struct fft_c2cf<float>
{
    static constexpr hipfftType type = HIPFFT_C2C;
    using in_type                   = hipfftComplex;
    using out_type                  = hipfftComplex;

    static hipfftResult exec( hipfftHandle &plan, in_type *in, out_type *out )
    {
        return hipfftExecC2C( plan, in, out, HIPFFT_FORWARD );
    }
};

template <>
struct fft_c2cf<double>
{
    static constexpr hipfftType type = HIPFFT_Z2Z;
    using in_type                   = hipfftDoubleComplex;
    using out_type                  = hipfftDoubleComplex;

    static hipfftResult exec( hipfftHandle &plan, in_type *in, out_type *out )
    {
        return hipfftExecZ2Z( plan, in, out, HIPFFT_FORWARD );
    }
};

// C2C backward
template <typename T>
struct fft_c2cb;

template <>
struct fft_c2cb<float>
{
    static constexpr hipfftType type = HIPFFT_C2C;
    using in_type                   = hipfftComplex;
    using out_type                  = hipfftComplex;

    static hipfftResult exec( hipfftHandle &plan, in_type *in, out_type *out )
    {
        return hipfftExecC2C( plan, in, out, HIPFFT_BACKWARD );
    }
};

template <>
struct fft_c2cb<double>
{
    static constexpr hipfftType type = HIPFFT_Z2Z;
    using in_type                   = hipfftDoubleComplex;
    using out_type                  = hipfftDoubleComplex;

    static hipfftResult exec( hipfftHandle &plan, in_type *in, out_type *out )
    {
        return hipfftExecZ2Z( plan, in, out, HIPFFT_BACKWARD );
    }
};

template <typename T>
struct fft_c2c_direct_raw;

template <>
struct fft_c2c_direct_raw<float>
{
    static hipfftResult exec( hipfftHandle plan, void *in, void *out, int direction )
    {
        return hipfftExecC2C(
            plan, static_cast<hipfftComplex *>( in ), static_cast<hipfftComplex *>( out ), direction
        );
    }
};

template <>
struct fft_c2c_direct_raw<double>
{
    static hipfftResult exec( hipfftHandle plan, void *in, void *out, int direction )
    {
        return hipfftExecZ2Z(
            plan, static_cast<hipfftDoubleComplex *>( in ), static_cast<hipfftDoubleComplex *>( out ), direction
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
    using memory_type = typename hip_runtime_api::memory_type;
    using runtime_api = hip_runtime_api;
    virtual ~fft_base()
    {
    }
    // virtual hipfftResult exec(void* in, void* out) = 0; //check error here
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
    using opaque_plan_handle_t = hipfftHandle;
    struct minimal_reference_c2c_plan_array_t
    {
        std::vector<hipfftHandle> handles;
        std::vector<hipStream_t> streams;
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
        static_assert( Rank >= 1 && Rank <= 3, "hipfft::fft supports only 1D, 2D, and 3D plans" );

        auto n_local       = n;
        auto inembed_local = inembed;
        auto onembed_local = onembed;

        descriptor_.dir        = D;
        descriptor_.backend_type = static_cast<int>( get_fft_direction( D ) );
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
        static_assert( Rank >= 1 && Rank <= 3, "hipfft::fft supports only 1D, 2D, and 3D plans" );

        auto n_local = n;

        descriptor_.dir            = D;
        descriptor_.backend_type   = static_cast<int>( get_fft_direction( D ) );
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
            hipfftDestroy( handle_ );
        }
    }

    virtual std::size_t get_work_size() const override
    {
        return work_size;
    }


    virtual void set_work_area( void *work_area ) override
    {
        descriptor_.work_area_token = reinterpret_cast<std::uintptr_t>( work_area );
        FFTM_HIPFFT_SAFE_CALL( hipfftSetWorkArea( handle_, work_area ) );
    }

    virtual void set_stream( typename base_t::runtime_api::stream_t stream ) override
    {
        descriptor_.stream_token = reinterpret_cast<std::uintptr_t>( stream );
        FFTM_HIPFFT_SAFE_CALL( hipfftSetStream( handle_, stream ) );
    }

    virtual void recreate_with_stream_and_work_area(
        typename base_t::runtime_api::stream_t stream, void *work_area
    ) override
    {
        if ( handle_ != 0 )
        {
            FFTM_HIPFFT_SAFE_CALL( hipfftDestroy( handle_ ) );
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
        FFTM_HIPFFT_SAFE_CALL( traits::exec( handle_, static_cast<in_type *>( in ), static_cast<out_type *>( out ) ) );
    }

    void exec_typed_no_sync( void *in, void *out )
    {
        hipfft_check_no_sync(
            traits::exec( handle_, static_cast<in_type *>( in ), static_cast<out_type *>( out ) ),
            "hipfft_wrap::fft::exec_typed_no_sync"
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
            FFTM_HIPFFT_SAFE_CALL( detail::fft_c2cf<T>::exec(
                handle_, static_cast<typename detail::fft_c2cf<T>::in_type *>( in ),
                static_cast<typename detail::fft_c2cf<T>::out_type *>( out )
            ) );
            return;
        }
        if ( ( D == direction::C2CF || D == direction::C2CB ) && exec_dir == direction::C2CB )
        {
            FFTM_HIPFFT_SAFE_CALL( detail::fft_c2cb<T>::exec(
                handle_, static_cast<typename detail::fft_c2cb<T>::in_type *>( in ),
                static_cast<typename detail::fft_c2cb<T>::out_type *>( out )
            ) );
            return;
        }
        throw std::logic_error( "hipfft_wrap::fft::exec_typed_direction: incompatible execution direction" );
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
            hipfft_check_no_sync(
                detail::fft_c2cf<T>::exec(
                    handle_, static_cast<typename detail::fft_c2cf<T>::in_type *>( in ),
                    static_cast<typename detail::fft_c2cf<T>::out_type *>( out )
                ),
                "hipfft_wrap::fft::exec_typed_direction_no_sync(C2CF)"
            );
            return;
        }
        if ( ( D == direction::C2CF || D == direction::C2CB ) && exec_dir == direction::C2CB )
        {
            hipfft_check_no_sync(
                detail::fft_c2cb<T>::exec(
                    handle_, static_cast<typename detail::fft_c2cb<T>::in_type *>( in ),
                    static_cast<typename detail::fft_c2cb<T>::out_type *>( out )
                ),
                "hipfft_wrap::fft::exec_typed_direction_no_sync(C2CB)"
            );
            return;
        }
        throw std::logic_error( "hipfft_wrap::fft::exec_typed_direction_no_sync: incompatible execution direction" );
    }

    opaque_plan_handle_t opaque_plan_handle() const
    {
        return handle_;
    }

    static opaque_plan_handle_t create_empty_opaque_plan()
    {
        opaque_plan_handle_t handle = 0;
        FFTM_HIPFFT_SAFE_CALL( hipfftCreate( &handle ) );
        try
        {
            FFTM_HIPFFT_SAFE_CALL( hipfftSetAutoAllocation( handle, 0 ) );
        }
        catch ( ... )
        {
            hipfftDestroy( handle );
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
        FFTM_HIPFFT_SAFE_CALL( hipfftMakePlanMany64(
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
        FFTM_HIPFFT_SAFE_CALL( hipfftMakePlanMany64(
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

        FFTM_HIPFFT_SAFE_CALL( hipfftCreate( &handle ) );
        try
        {
            FFTM_HIPFFT_SAFE_CALL( hipfftSetAutoAllocation( handle, 0 ) );
            FFTM_HIPFFT_SAFE_CALL( hipfftMakePlanMany64(
                handle, 1, n_arr, inembed_arr, istride, idist, onembed_arr, ostride, odist,
                detail::fft_c2cf<T>::type, batch, &work_size_out
            ) );
            FFTM_HIPFFT_SAFE_CALL( hipfftSetStream( handle, stream ) );
        }
        catch ( ... )
        {
            hipfftDestroy( handle );
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
        bundle->descriptor.backend_type   = detail::fft_c2cf<T>::type;
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
                hipStream_t stream = nullptr;
                HIP_SAFE_CALL( hipStreamCreate( &stream ) );
                bundle->streams.push_back( stream );

                hipfftHandle handle = 0;
                FFTM_HIPFFT_SAFE_CALL( hipfftCreate( &handle ) );
                bundle->handles.push_back( handle );
                FFTM_HIPFFT_SAFE_CALL( hipfftSetAutoAllocation( handle, 0 ) );
                std::size_t work_size = 0;
                FFTM_HIPFFT_SAFE_CALL( hipfftMakePlanMany64(
                    handle, 1, n_arr, inembed_arr, istride, idist, onembed_arr, ostride, odist,
                    detail::fft_c2cf<T>::type, batch, &work_size
                ) );
                FFTM_HIPFFT_SAFE_CALL( hipfftSetStream( handle, stream ) );
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

        FFTM_HIPFFT_SAFE_CALL( hipfftCreate( &handle ) );
        try
        {
            FFTM_HIPFFT_SAFE_CALL( hipfftSetAutoAllocation( handle, 0 ) );
            FFTM_HIPFFT_SAFE_CALL( hipfftMakePlanMany64(
                handle, 1, n_local, inembed_local, istride, idist, onembed_local, ostride, odist,
                detail::fft_c2cf<T>::type, batch, &work_size_out
            ) );
            if ( stream != nullptr )
            {
                FFTM_HIPFFT_SAFE_CALL( hipfftSetStream( handle, stream ) );
            }
        }
        catch ( ... )
        {
            hipfftDestroy( handle );
            throw;
        }
        return handle;
    }

public:
    static void destroy_opaque_plan_noexcept( opaque_plan_handle_t plan )
    {
        if ( plan != 0 )
            hipfftDestroy( plan );
    }

    static void destroy_minimal_reference_c2c_plan_array_noexcept(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        if ( bundle == nullptr )
            return;
        for ( hipfftHandle handle : bundle->handles )
        {
            if ( handle != 0 )
                hipfftDestroy( handle );
        }
        for ( hipStream_t stream : bundle->streams )
        {
            if ( stream != nullptr )
                hipStreamDestroy( stream );
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
            throw std::logic_error( "hipfft_wrap::fft::minimal_reference_c2c_plan_array_descriptor: null bundle" );
        return bundle->descriptor;
    }

    static std::uintptr_t minimal_reference_c2c_plan_array_first_handle_token(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        return bundle == nullptr || bundle->handles.empty() ? 0 : detail::opaque_handle_token( bundle->handles.front() );
    }

    static std::uintptr_t minimal_reference_c2c_plan_array_last_handle_token(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        return bundle == nullptr || bundle->handles.empty() ? 0 : detail::opaque_handle_token( bundle->handles.back() );
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
            throw std::logic_error( "hipfft_wrap::fft::bind_minimal_reference_c2c_plan_array_work_areas: null bundle" );
        char *raw = static_cast<char *>( base_work_area );
        const std::size_t stride = std::max( work_stride_bytes, bundle->work_stride_bytes );
        for ( std::size_t i = 0; i < bundle->handles.size(); ++i )
        {
            FFTM_HIPFFT_SAFE_CALL( hipfftSetWorkArea( bundle->handles[i], static_cast<void *>( raw + i * stride ) ) );
        }
    }

    static void synchronize_minimal_reference_c2c_plan_array_streams(
        minimal_reference_c2c_plan_array_handle_t bundle
    )
    {
        if ( bundle == nullptr )
            throw std::logic_error( "hipfft_wrap::fft::synchronize_minimal_reference_c2c_plan_array_streams: null bundle" );
        for ( hipStream_t stream : bundle->streams )
        {
            HIP_SAFE_CALL( hipStreamSynchronize( stream ) );
        }
    }

    static void set_opaque_stream( opaque_plan_handle_t plan, typename base_t::runtime_api::stream_t stream )
    {
        FFTM_HIPFFT_SAFE_CALL( hipfftSetStream( plan, stream ) );
    }

    static void set_opaque_work_area( opaque_plan_handle_t plan, void *work_area )
    {
        FFTM_HIPFFT_SAFE_CALL( hipfftSetWorkArea( plan, work_area ) );
    }

    static void set_direct_raw_work_area( opaque_plan_handle_t plan, void *work_area )
    {
        FFTM_HIPFFT_SAFE_CALL( hipfftSetWorkArea( plan, work_area ) );
    }

    static void exec_opaque_c2c( opaque_plan_handle_t plan, direction exec_dir, void *in, void *out )
    {
        if ( exec_dir == direction::C2CF )
        {
            FFTM_HIPFFT_SAFE_CALL( detail::fft_c2cf<T>::exec(
                plan, static_cast<typename detail::fft_c2cf<T>::in_type *>( in ),
                static_cast<typename detail::fft_c2cf<T>::out_type *>( out )
            ) );
            return;
        }
        if ( exec_dir == direction::C2CB )
        {
            FFTM_HIPFFT_SAFE_CALL( detail::fft_c2cb<T>::exec(
                plan, static_cast<typename detail::fft_c2cb<T>::in_type *>( in ),
                static_cast<typename detail::fft_c2cb<T>::out_type *>( out )
            ) );
            return;
        }
        throw std::logic_error( "hipfft_wrap::fft::exec_opaque_c2c: incompatible execution direction" );
    }

    static void exec_opaque_c2c_no_sync( opaque_plan_handle_t plan, direction exec_dir, void *in, void *out )
    {
        if ( exec_dir == direction::C2CF )
        {
            hipfft_check_no_sync(
                detail::fft_c2cf<T>::exec(
                    plan, static_cast<typename detail::fft_c2cf<T>::in_type *>( in ),
                    static_cast<typename detail::fft_c2cf<T>::out_type *>( out )
                ),
                "hipfft_wrap::fft::exec_opaque_c2c_no_sync(C2CF)"
            );
            return;
        }
        if ( exec_dir == direction::C2CB )
        {
            hipfft_check_no_sync(
                detail::fft_c2cb<T>::exec(
                    plan, static_cast<typename detail::fft_c2cb<T>::in_type *>( in ),
                    static_cast<typename detail::fft_c2cb<T>::out_type *>( out )
                ),
                "hipfft_wrap::fft::exec_opaque_c2c_no_sync(C2CB)"
            );
            return;
        }
        throw std::logic_error( "hipfft_wrap::fft::exec_opaque_c2c_no_sync: incompatible execution direction" );
    }

    static void exec_direct_raw_c2c( opaque_plan_handle_t plan, direction exec_dir, void *in, void *out )
    {
        if ( exec_dir == direction::C2CF )
        {
            FFTM_HIPFFT_SAFE_CALL( detail::fft_c2c_direct_raw<T>::exec( plan, in, out, HIPFFT_FORWARD ) );
            return;
        }
        if ( exec_dir == direction::C2CB )
        {
            FFTM_HIPFFT_SAFE_CALL( detail::fft_c2c_direct_raw<T>::exec( plan, in, out, HIPFFT_BACKWARD ) );
            return;
        }
        throw std::logic_error( "hipfft_wrap::fft::exec_direct_raw_c2c: incompatible execution direction" );
    }

    static void exec_direct_raw_c2c_no_sync( opaque_plan_handle_t plan, direction exec_dir, void *in, void *out )
    {
        if ( exec_dir == direction::C2CF )
        {
            hipfft_check_no_sync(
                detail::fft_c2c_direct_raw<T>::exec( plan, in, out, HIPFFT_FORWARD ),
                "hipfft_wrap::fft::exec_direct_raw_c2c_no_sync(C2CF)"
            );
            return;
        }
        if ( exec_dir == direction::C2CB )
        {
            hipfft_check_no_sync(
                detail::fft_c2c_direct_raw<T>::exec( plan, in, out, HIPFFT_BACKWARD ),
                "hipfft_wrap::fft::exec_direct_raw_c2c_no_sync(C2CB)"
            );
            return;
        }
        throw std::logic_error( "hipfft_wrap::fft::exec_direct_raw_c2c_no_sync: incompatible execution direction" );
    }

    static void exec_minimal_reference_c2c_plan_array(
        minimal_reference_c2c_plan_array_handle_t bundle, direction exec_dir, void *in, void *out
    )
    {
        if ( bundle == nullptr )
            throw std::logic_error( "hipfft_wrap::fft::exec_minimal_reference_c2c_plan_array: null bundle" );
        int hipfft_direction = 0;
        if ( exec_dir == direction::C2CF )
            hipfft_direction = HIPFFT_FORWARD;
        else if ( exec_dir == direction::C2CB )
            hipfft_direction = HIPFFT_BACKWARD;
        else
            throw std::logic_error(
                "hipfft_wrap::fft::exec_minimal_reference_c2c_plan_array: incompatible execution direction"
            );
        if ( bundle->offsets.size() != bundle->handles.size() )
            throw std::logic_error( "hipfft_wrap::fft::exec_minimal_reference_c2c_plan_array: offset count mismatch" );
        for ( std::size_t i = 0; i < bundle->handles.size(); ++i )
        {
            FFTM_HIPFFT_SAFE_CALL( detail::fft_c2c_direct_raw<T>::exec(
                bundle->handles[i],
                static_cast<void *>( static_cast<typename base_t::complex *>( in ) + bundle->offsets[i] ),
                static_cast<void *>( static_cast<typename base_t::complex *>( out ) + bundle->offsets[i] ),
                hipfft_direction
            ) );
        }
    }

    static void exec_minimal_reference_c2c_plan_array_no_sync(
        minimal_reference_c2c_plan_array_handle_t bundle, direction exec_dir, void *in, void *out
    )
    {
        if ( bundle == nullptr )
            throw std::logic_error( "hipfft_wrap::fft::exec_minimal_reference_c2c_plan_array_no_sync: null bundle" );
        int hipfft_direction = 0;
        if ( exec_dir == direction::C2CF )
            hipfft_direction = HIPFFT_FORWARD;
        else if ( exec_dir == direction::C2CB )
            hipfft_direction = HIPFFT_BACKWARD;
        else
            throw std::logic_error(
                "hipfft_wrap::fft::exec_minimal_reference_c2c_plan_array_no_sync: incompatible execution direction"
            );
        if ( bundle->offsets.size() != bundle->handles.size() )
            throw std::logic_error(
                "hipfft_wrap::fft::exec_minimal_reference_c2c_plan_array_no_sync: offset count mismatch"
            );
        for ( std::size_t i = 0; i < bundle->handles.size(); ++i )
        {
            hipfft_check_no_sync(
                detail::fft_c2c_direct_raw<T>::exec(
                    bundle->handles[i],
                    static_cast<void *>( static_cast<typename base_t::complex *>( in ) + bundle->offsets[i] ),
                    static_cast<void *>( static_cast<typename base_t::complex *>( out ) + bundle->offsets[i] ),
                    hipfft_direction
                ),
                "hipfft_wrap::fft::exec_minimal_reference_c2c_plan_array_no_sync"
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
    hipfftHandle handle_;
    std::size_t work_size;
    fft_plan_descriptor descriptor_;

    void create_plan(
        int rank, long long int *n, long long int *inembed, long long int istride, long long int idist,
        long long int *onembed, long long int ostride, long long int odist, long long int batch
    )
    {
        FFTM_HIPFFT_SAFE_CALL( hipfftCreate( &handle_ ) );

        try
        {
            FFTM_HIPFFT_SAFE_CALL( hipfftSetAutoAllocation( handle_, 0 ) );
            FFTM_HIPFFT_SAFE_CALL( hipfftMakePlanMany64(
                handle_, rank, n, inembed, istride, idist, onembed, ostride, odist, get_fft_direction( D ), batch,
                &work_size
            ) );
            descriptor_.work_size = work_size;
        }
        catch ( ... )
        {
            hipfftDestroy( handle_ );
            handle_ = 0;
            throw;
        }
    }

    hipfftType get_fft_direction( direction dir )
    {
        return get_fft_direction_static_( dir );
    }

    static hipfftType get_fft_direction_static_( direction dir )
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


#endif // __FFTM_HIPFFT_WRAP_H__
