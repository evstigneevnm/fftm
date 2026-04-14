#ifndef __FFTM_CUFFT_WRAP_MANY_H__
#define __FFTM_CUFFT_WRAP_MANY_H__

#include "fft_wrap_many.h"
#include "cufft_wrap.h"

namespace fftm
{
namespace wrap
{

template <class T>
using cufft_wrap_many = fft_wrap_many<cufft::fft, T>;

} // namespace wrap
} // namespace fftm

#endif
