#ifndef __FFTM_CUFFT_WRAP_H__
#define __FFTM_CUFFT_WRAP_H__

#include <array>
#include <stdexcept>
#include <type_traits>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>
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

struct cuda_runtime_api
{
    using memory_type        = scfd::memory::cuda_device;
    using stream_wrap        = scfd::utils::cuda_stream_wrap;
    using stream_t           = cudaStream_t;
    using memcpy_kind_t      = cudaMemcpyKind;
    using memcpy_3d_params_t = cudaMemcpy3DParms;
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

    static void stream_synchronize( stream_t stream )
    {
        CUDA_SAFE_CALL( cudaStreamSynchronize( stream ) );
    }
};

namespace cufft
{
namespace detail
{



template<typename T>
struct fft_complex;

template<>
struct fft_complex<float>
{
    using complex = cufftComplex;
};

template<>
struct fft_complex<double>
{
    using complex = cufftDoubleComplex;
};

template<typename T> 
struct fft_r2c;

template<>
struct fft_r2c<float>
{
    static constexpr cufftType type = CUFFT_R2C;
    using in_type  = float;
    using out_type = cufftComplex;

    static cufftResult exec(cufftHandle& plan, in_type* in, out_type* out)
    {
        return cufftExecR2C(plan, in, out);
    }
};

template<>
struct fft_r2c<double>
{
    static constexpr cufftType type = CUFFT_D2Z;
    using in_type  = double;
    using out_type = cufftDoubleComplex;

    static cufftResult exec(cufftHandle& plan, in_type* in, out_type* out)
    {
        return cufftExecD2Z(plan, in, out);
    }
};

// C2R
template<typename T> 
struct fft_c2r;

template<>
struct fft_c2r<float>
{
    static constexpr cufftType type = CUFFT_C2R;
    using in_type  = cufftComplex;
    using out_type = float;

    static cufftResult exec(cufftHandle& plan, in_type* in, out_type* out)
    {
        return cufftExecC2R(plan, in, out);
    }
};

template<>
struct fft_c2r<double>
{
    static constexpr cufftType type = CUFFT_Z2D;
    using in_type  = cufftDoubleComplex;
    using out_type = double;

    static cufftResult exec(cufftHandle& plan, in_type* in, out_type* out)
    {
        return cufftExecZ2D(plan, in, out);
    }
};

// C2C forward
template<typename T> 
struct fft_c2cf;

template<>
struct fft_c2cf<float>
{
    static constexpr cufftType type = CUFFT_C2C;
    using in_type  = cufftComplex;
    using out_type = cufftComplex;

    static cufftResult exec(cufftHandle& plan, in_type* in, out_type* out)
    {
        return cufftExecC2C(plan, in, out, CUFFT_FORWARD);
    }
};

template<>
struct fft_c2cf<double>
{
    static constexpr cufftType type = CUFFT_Z2Z;
    using in_type  = cufftDoubleComplex;
    using out_type = cufftDoubleComplex;

    static cufftResult exec(cufftHandle& plan, in_type* in, out_type* out)
    {
        return cufftExecZ2Z(plan, in, out, CUFFT_FORWARD);
    }
};

// C2C backward
template<typename T> 
struct fft_c2cb;

template<>
struct fft_c2cb<float>
{
    static constexpr cufftType type = CUFFT_C2C;
    using in_type  = cufftComplex;
    using out_type = cufftComplex;

    static cufftResult exec(cufftHandle& plan, in_type* in, out_type* out)
    {
        return cufftExecC2C(plan, in, out, CUFFT_INVERSE);
    }
};

template<>
struct fft_c2cb<double>
{
    static constexpr cufftType type = CUFFT_Z2Z;
    using in_type  = cufftDoubleComplex;
    using out_type = cufftDoubleComplex;

    static cufftResult exec(cufftHandle& plan, in_type* in, out_type* out)
    {
        return cufftExecZ2Z(plan, in, out, CUFFT_INVERSE);
    }
};


template<typename T, direction D>
struct fft_traits;

template<typename T>
struct fft_traits<T, direction::R2C> : fft_r2c<T> {};

template<typename T>
struct fft_traits<T, direction::C2R> : fft_c2r<T> {};

template<typename T>
struct fft_traits<T, direction::C2CF> : fft_c2cf<T> {};

template<typename T>
struct fft_traits<T, direction::C2CB> : fft_c2cb<T> {};


}


template<typename T>
class fft_base
{
public:
    using complex = typename detail::fft_complex<T>::complex;
    using memory_type = typename cuda_runtime_api::memory_type;
    using runtime_api = cuda_runtime_api;
    virtual ~fft_base() {}
    // virtual cufftResult exec(void* in, void* out) = 0; //check error here
    virtual void exec(void* in, void* out) = 0;
    virtual direction get_direction() const = 0;
    virtual std::size_t get_work_size() const = 0;
    virtual void set_work_area(void* work_area) = 0;
};


template<typename T, ::fftm::direction D>
class fft : public fft_base<T>
{
    using traits = detail::fft_traits<T, D>;

public:
    using base_t   = fft_base<T>;
    using in_type  = typename traits::in_type;
    using out_type = typename traits::out_type;


    explicit fft(long long int n, long long int inembed, long long int istride, long long int idist, long long int onembed, long long int ostride, long long int odist, long long int batch)
    : fft(std::array<long long int, 1>{n},
          std::array<long long int, 1>{inembed},
          istride,
          idist,
          std::array<long long int, 1>{onembed},
          ostride,
          odist,
          batch)
    {}

    explicit fft(long long int n0, long long int n1, long long int inembed0, long long int inembed1, long long int istride, long long int idist, long long int onembed0, long long int onembed1, long long int ostride, long long int odist, long long int batch)
    : fft(std::array<long long int, 2>{n0, n1},
          std::array<long long int, 2>{inembed0, inembed1},
          istride,
          idist,
          std::array<long long int, 2>{onembed0, onembed1},
          ostride,
          odist,
          batch)
    {}

    explicit fft(long long int n0, long long int n1, long long int n2, long long int inembed0, long long int inembed1, long long int inembed2, long long int istride, long long int idist, long long int onembed0, long long int onembed1, long long int onembed2, long long int ostride, long long int odist, long long int batch)
    : fft(std::array<long long int, 3>{n0, n1, n2},
          std::array<long long int, 3>{inembed0, inembed1, inembed2},
          istride,
          idist,
          std::array<long long int, 3>{onembed0, onembed1, onembed2},
          ostride,
          odist,
          batch)
    {}

    template<std::size_t Rank>
    explicit fft(const std::array<long long int, Rank>& n, const std::array<long long int, Rank>& inembed, long long int istride, long long int idist, const std::array<long long int, Rank>& onembed, long long int ostride, long long int odist, long long int batch)
    : handle_(0),
      work_size(0)
    {
        static_assert(Rank >= 1 && Rank <= 3, "cufft::fft supports only 1D, 2D, and 3D plans");

        auto n_local = n;
        auto inembed_local = inembed;
        auto onembed_local = onembed;

        create_plan(static_cast<int>(Rank),
                    n_local.data(),
                    inembed_local.data(),
                    istride,
                    idist,
                    onembed_local.data(),
                    ostride,
                    odist,
                    batch);
    }

    ~fft() override
    {
        if (handle_ != 0)
        {
            cufftDestroy(handle_);
        }
    }

    virtual std::size_t get_work_size() const override
    {
        return work_size;
    }


    virtual void set_work_area(void* work_area) override
    {
        CUFFT_SAFE_CALL( cufftSetWorkArea(handle_, work_area) );
    }


    virtual void exec(void* in, void* out) override
    {
        CUFFT_SAFE_CALL( traits::exec(handle_, static_cast<in_type*>(in), static_cast<out_type*>(out) ) );
    }

    virtual direction get_direction() const override
    {
        return D;
    }

private:

    cufftHandle handle_;
    std::size_t work_size;

    void create_plan(int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, long long int batch)
    {
        CUFFT_SAFE_CALL(cufftCreate(&handle_));

        try
        {
            CUFFT_SAFE_CALL( cufftSetAutoAllocation(handle_, 0) );
            CUFFT_SAFE_CALL(cufftMakePlanMany64(
                handle_,
                rank,
                n,
                inembed,
                istride,
                idist,
                onembed,
                ostride,
                odist,
                get_fft_direction(D),
                batch,
                &work_size) );
        }
        catch(...)
        {
            cufftDestroy(handle_);
            handle_ = 0;
            throw;
        }
    }

    cufftType get_fft_direction(direction dir)
    {
        switch(dir)
        {
            case direction::R2C:
                return detail::fft_r2c<T>::type;
                break;
            case direction::C2R:
                return detail::fft_c2r<T>::type;
                break;
            case direction::C2CF:
                return detail::fft_c2cf<T>::type;
                break;
            case direction::C2CB:
                return detail::fft_c2cb<T>::type;
                break;
        }
        
        return detail::fft_c2cf<T>::type; //to avoid warning
    }

};

}
}
}


#endif // __FFTM_CUFFT_WRAP_H__
