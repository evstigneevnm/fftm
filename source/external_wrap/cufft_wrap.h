#ifndef __FFTM_CUFFT_WRAP_H__
#define __FFTM_CUFFT_WRAP_H__


#include <cufft.h>
#include <cuda.h>
#include <>

namespace fft{
namespace wrap{

namespace detail{

template<typename T>
struct complex_type
{
};

template<>
struct complex_type<float>
{
    typedef cufftComplex type;
};

template<>
struct complex_type<double>
{
    typedef cufftDoubleComplex type;
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
}
template<>
struct fft_c2r<double>
{
    static const cufftType type = CUFFT_Z2D;
}
template<>
struct fft_r2r<float>
{
    static const cufftType type = CUFFT_C2R;
}
template<>
struct fft_r2r<double>
{
    static const cufftType type = CUFFT_Z2D;
}
template<>
struct fft_c2c<float>
{
    static const cufftType type = CUFFT_C2C;
}
template<>
struct fft_c2c<double>
{
    static const cufftType type = CUFFT_Z2Z;
}


}

template<class T>
class cufft_wrap
{
public:
    using complex = typename detail::complex_type<T>::type;
    using real = T;

    cufft_wrap()
    {
        cufftCreate(&handle);
        CUFFT_CALL(cufftSetAutoAllocation(handle, 0));
    }



    ~cufft_wrap()
    {

    }
    
private:
    cufftHandle handle;

};

}}


#endif // __FFTM_CUFFT_WRAP_H__