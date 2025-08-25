#include <memory>
#include <utility>
#include "external_wrap/cufft_wrap.h"

namespace fft
{

template <class T, class BaseFFT>
class fftm
{
public:
    fftm(std::shared_ptr<BaseFFT> base_fft):
    base_fft_(base_fft)
    {}
    ~fftm()
    {}
    
private:
    


};



}