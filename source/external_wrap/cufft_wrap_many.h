#ifndef __FFTM_CUFFT_WRAP_MANY_H__
#define __FFTM_CUFFT_WRAP_MANY_H__

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <scfd/utils/cuda_safe_call.h>
#include "fft_direction.h"
#include "cufft_wrap.h"


namespace fftm{
namespace wrap{


template<class T>
class cufft_wrap_many
{
private:
    using wrap_t = cufft_wrap<T>;
public:
    using complex = typename wrap_t::complex;

    cufft_wrap_many():
    work_area_(nullptr),
    activated_(false)
    {}
    ~cufft_wrap_many()
    {
        if(activated_)
        {
            cudaFree(work_area_);
        }
    }
        
    void add_plan_1D(const std::string& name, long long int n, long long int inembed, long long int istride, long long int idist, long long int onembed, long long int ostride, long long int odist, direction type, long long int batch)
    {
        if(activated_)
        {
            throw std::logic_error("cufft_wrap_many::add_plan_1D: cannot add new plans after the wrap was activated.");
        }
        container_.emplace(name, wrap_t{});
        container_[name].plan1D_create(n, inembed, istride, idist, onembed, ostride, odist, type, batch);
    }

    void activate() 
    {
        if(activated_)
        {
            throw std::logic_error("cufft_wrap_many::activate: cannot activate again, the container is already activated.");
        }
        std::vector<std::size_t> work_sizes;
        for(auto &el: container_)
        {
            work_sizes.push_back( el.second.get_work_size() );
        }
        auto max_work_size = std::max_element(work_sizes.cbegin(), work_sizes.cend());
        
        CUDA_SAFE_CALL( cudaMalloc((void**)&work_area_, (*max_work_size)*sizeof(T) ) );

        activated_ = true;

    }



private:
    T* work_area_;
    bool activated_;
    std::map<std::string, wrap_t> container_;

};



}
}

#endif // __FFTM_CUFFT_WRAP_MANY_H__