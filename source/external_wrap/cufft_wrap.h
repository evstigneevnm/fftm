#ifndef __FFTM_CUFFT_WRAP_H__
#define __FFTM_CUFFT_WRAP_H__

#include <stdexcept>
#include <cufft.h>
#include <cuda.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/cufft_safe_call.h>
#include "fft_direction.h"

namespace fftm{
namespace wrap{

namespace detail{

template<typename T>
struct complex_type
{
};

template<>
struct complex_type<float>
{
    using type = cufftComplex ;
};

template<>
struct complex_type<double>
{
    using type = cufftDoubleComplex;
};

template<typename T>
struct fft_c2r{};

template<typename T>
struct fft_r2c{};

template<typename T>
struct fft_c2c{};

template<>
struct fft_c2r<float>
{
    static const cufftType type = CUFFT_C2R;
};

template<>
struct fft_c2r<double>
{
    static const cufftType type = CUFFT_Z2D;
};
template<>
struct fft_r2c<float>
{
    static const cufftType type = CUFFT_R2C;
};
template<>
struct fft_r2c<double>
{
    static const cufftType type = CUFFT_D2Z;
};
template<>
struct fft_c2c<float>
{
    static const cufftType type = CUFFT_C2C;
};
template<>
struct fft_c2c<double>
{
    static const cufftType type = CUFFT_Z2Z;
};

}

template<class T>
class cufft_wrap
{
public:
    using real = T;
    using complex = typename detail::complex_type<T>::type;
    

    cufft_wrap():
    handle_(0),
    plan_created(false),
    work_size(0)
    {
        CUFFT_SAFE_CALL(cufftCreate(&handle_));
        CUFFT_SAFE_CALL( cufftSetAutoAllocation(handle_, 0) );
    }
    ~cufft_wrap()
    {
        cufftDestroy(handle_); 
    }

    //move constructor for exmplace construction
    cufft_wrap(cufft_wrap&& other):
    handle_(std::exchange(other.handle_, 0)),
    plan_created(std::exchange(other.plan_created, false)),
    work_size(std::exchange(other.work_size, 0))
    {}



    void plan1D_create( long long int n, long long int inembed, long long int istride, long long int idist, long long int onembed, long long int ostride, long long int odist, direction type, long long int batch)
    {
        if(!plan_created)
        {
        //cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t *workSize);

        // plan[In] – cufftHandle returned by cufftCreate.
        // rank[In] – Dimensionality of the transform (1, 2, or 3).
        // n[In] – Array of size rank, describing the size of each dimension. For multiple GPUs and rank equal to 1, the sizes must be a power of 2. For multiple GPUs and rank equal to 2 or 3, the sizes must be factorable into primes less than or equal to 127.
        // inembed[In] – Pointer of size rank that indicates the storage dimensions of the input data in memory. If set to NULL all other advanced data layout parameters are ignored.
        // istride[In] – Indicates the distance between two successive input elements in the least significant (i.e., innermost) dimension.
        // idist[In] – Indicates the distance between the first element of two consecutive signals in a batch of the input data.
        // onembed[In] – Pointer of size rank that indicates the storage dimensions of the output data in memory. If set to NULL all other advanced data layout parameters are ignored.
        // ostride[In] – Indicates the distance between two successive output elements in the output array in the least significant (i.e., innermost) dimension.
        // odist[In] – Indicates the distance between the first element of two consecutive signals in a batch of the output data.
        // type[In] – The transform data type (e.g., CUFFT_R2C for single precision real to complex). For 2 GPUs this must be a complex to complex transform.
        // batch[In] – Batch size for this transform.
        // *workSize[In] – Pointer to the size(s), in bytes, of the work areas. For example for two GPUs worksize must be declared to have two elements.
        // *workSize[Out] – Pointer to the size(s) of the work areas.
            CUFFT_SAFE_CALL(cufftMakePlanMany64(
                handle_, 1, &n, 
                &inembed, istride, idist, 
                &onembed, ostride, odist, 
                get_fft_direction(type), batch, &work_size) );
            
            plan_created = true;
        }
        else
        {
            throw std::logic_error("cufft_wrap::plan_create called after the plan has already being created.");
        }
    }
    
    void set_work_area(T* work_area)
    {
        CUFFT_SAFE_CALL( cufftSetWorkArea(handle_, work_area) );
    }

    std::size_t get_work_size() const
    {
        return work_size;
    }

private:
    cufftHandle handle_;
    bool plan_created;
    std::size_t work_size;

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
            case direction::C2C:
                return detail::fft_c2c<T>::type;
                break;
        }
        return detail::fft_c2c<T>::type;
    }



};

}}


#endif // __FFTM_CUFFT_WRAP_H__