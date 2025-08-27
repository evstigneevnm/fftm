#include <memory>
#include <external_wrap/cufft_wrap.h>
#include <fftm.hpp>


int main(int argc, char const *argv[])
{
    using base_fft_t = fftm::wrap::cufft_wrap<double>;
    using fftm_t = fftm::fftm<base_fft_t>;

    auto base_fft = std::make_shared<base_fft_t>();
    auto fftm = std::make_shared<fftm_t>(base_fft);
    fftm->init_test();

    return 0;
}