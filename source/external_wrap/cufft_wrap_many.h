#ifndef __FFTM_CUFFT_WRAP_MANY_H__
#define __FFTM_CUFFT_WRAP_MANY_H__

#include <utility>
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

template<class T, ::fftm::direction D>
class cufft_wrap_many
{
private:
    using wrap_t = cufft_wrap<T, D>;
public:
    using real = T;
    using complex = typename wrap_t::complex;

    cufft_wrap_many():
    work_area_(nullptr),
    work_area_size_(0),
    activated_(false)
    {}
    ~cufft_wrap_many()
    {
        if(activated_)
        {
            cudaFree(work_area_);
        }
    }
        



    void add_plan_1D(const std::string& name, long long int n, long long int inembed, long long int istride, long long int idist, long long int onembed, long long int ostride, long long int odist, long long int batch)
    {
        if(activated_)
        {
            throw std::logic_error("cufft_wrap_many::add_plan_1D: cannot add new plans after the wrap was activated.");
        }
        container_.emplace(name, wrap_t{});
        container_.at(name).plan1D_create(n, inembed, istride, idist, onembed, ostride, odist, type, batch);
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
        work_area_size_ = *max_work_size;
        
        CUDA_SAFE_CALL( cudaMalloc( &work_area_, work_area_size_ ) );

        for(auto &el: container_)
        {
            el.second.set_work_area(work_area_);
        }

        activated_ = true;

    }

    std::size_t get_work_size() const
    {
        return work_area_size_;
    }


    void exec(const std::string& name)
    {

    }


private:
    void* work_area_;
    bool activated_;
    std::map<std::string, wrap_t> container_;
    std::size_t work_area_size_; //in bytes!!!

};



}
}

#endif // __FFTM_CUFFT_WRAP_MANY_H__