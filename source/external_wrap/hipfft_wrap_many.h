#ifndef FFTM_HIPFFT_WRAP_MANY_H
#define FFTM_HIPFFT_WRAP_MANY_H

#include "fft_wrap_many.h"
#include "hipfft_wrap.h"

namespace fftm
{
namespace wrap
{

template <class T>
using hipfft_wrap_many = fft_wrap_many<hipfft::fft, T>;

} // namespace wrap
} // namespace fftm

#endif // FFTM_HIPFFT_WRAP_MANY_H
