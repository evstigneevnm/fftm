#ifndef __FFTM_FFTM_HPP__
#define __FFTM_FFTM_HPP__

#include <iostream>
#include <memory>
#include <utility>

namespace fftm
{

template <class BaseFFT>
class fftm
{
public:
    fftm(std::shared_ptr<BaseFFT> &base_fft):
    base_fft_(base_fft)
    {}
    ~fftm()
    {}

    void init_test()
    {
        
//         pidx:1, pidx_i:0, pidx_j:1, batch[0]:175104, batch[1]:87552, batch[2]:87552
// pidx:1, pidx_i:0, pidx_j:1, n[0]:1024, n[1]:1024, n[2]:1024
// pidx:1, pidx_i:0, pidx_j:1, inembed[0]:1, inembed[1]:1, inembed[2]:1
// pidx:1, pidx_i:0, pidx_j:1, onembed[0]:87552, onembed[1]:87552, onembed[2]:175104
// pidx:1, pidx_i:0, pidx_j:1, idist[0]:1024, idist[1]:1024, idist[2]:1024
// pidx:1, pidx_i:0, pidx_j:1, odist[0]:1, odist[1]:1, odist[2]:1

// cufftMakePlanMany64(planR2C, 1, &n[2], //plan, rank, *n
            // &inembed[2], inembed[2], idist[2], //*inembed, istride, idist
            // &onembed[2], onembed[2], odist[2], //*onembed, ostride, odist
            // cuFFT<T>::R2Ctype, batch[0], &ws_r2c)

        long long int n = 1024;
        long long int inembed = 1;
        long long int istride = 1;
        long long int idist = 1024;
        long long int onembed = 87552;
        long long int ostride = 87552;
        long long int odist = 1; 
        long long int batch = 175104;

        base_fft_->add_plan_1D("test_R2C",n, inembed, istride, idist, onembed, ostride, odist, direction::R2C, batch );
        base_fft_->add_plan_1D("test_C2R",n, inembed, istride, idist, onembed, ostride, odist, direction::C2R, batch );
        base_fft_->add_plan_1D("test_C2C",n, inembed, istride, idist, onembed, ostride, odist, direction::C2C, batch );
        base_fft_->activate();

        try
        {
            base_fft_->add_plan_1D("test_R2C",n, inembed, istride, idist, onembed, ostride, odist, direction::R2C, batch );
        }
        catch(const std::logic_error& e)
        {
            std::cout << "test logic 1: " << e.what() << std::endl;
        }
        try
        {
            base_fft_->activate();
        }
        catch(const std::logic_error& e)
        {
            std::cout << "test logic 2: " << e.what() << std::endl;
        }
        auto wsize = base_fft_->get_work_size();
        std::cout << "work_size: " << wsize*1.0e-9 << "GB." << std::endl;

    }
    
private:
    std::shared_ptr<BaseFFT> base_fft_;


};



}

#endif // __FFTM_FFTM_HPP__