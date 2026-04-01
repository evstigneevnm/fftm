#ifndef __FFTM_CUFFT_WRAP_MANY_H__
#define __FFTM_CUFFT_WRAP_MANY_H__

#include <array>
#include <utility>
#include <memory>
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
    using wrap_t = cufft::fft_base<T>;
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
        


    template<::fftm::direction D, std::size_t Rank>
    void add_plan(const std::string& name, const std::array<long long int, Rank>& n, const std::array<long long int, Rank>& inembed, long long int istride, long long int idist, const std::array<long long int, Rank>& onembed, long long int ostride, long long int odist, long long int batch)
    {
        if(activated_)
        {
            throw std::logic_error("cufft_wrap_many::add_plan: cannot add new plans after the wrap was activated.");
        }
        container_.emplace(name, std::make_unique<cufft::fft<T, D>>(n, inembed, istride, idist, onembed, ostride, odist, batch));
    }


    template<::fftm::direction D>
    void add_plan_1D(const std::string& name, long long int n, long long int inembed, long long int istride, long long int idist, long long int onembed, long long int ostride, long long int odist, long long int batch)
    {
        add_plan<D>(name,
                    std::array<long long int, 1>{n},
                    std::array<long long int, 1>{inembed},
                    istride,
                    idist,
                    std::array<long long int, 1>{onembed},
                    ostride,
                    odist,
                    batch);
    }


    template<::fftm::direction D>
    void add_plan_2D(const std::string& name, long long int n0, long long int n1, long long int inembed0, long long int inembed1, long long int istride, long long int idist, long long int onembed0, long long int onembed1, long long int ostride, long long int odist, long long int batch)
    {
        add_plan<D>(name,
                    std::array<long long int, 2>{n0, n1},
                    std::array<long long int, 2>{inembed0, inembed1},
                    istride,
                    idist,
                    std::array<long long int, 2>{onembed0, onembed1},
                    ostride,
                    odist,
                    batch);
    }


    template<::fftm::direction D>
    void add_plan_3D(const std::string& name, long long int n0, long long int n1, long long int n2, long long int inembed0, long long int inembed1, long long int inembed2, long long int istride, long long int idist, long long int onembed0, long long int onembed1, long long int onembed2, long long int ostride, long long int odist, long long int batch)
    {
        add_plan<D>(name,
                    std::array<long long int, 3>{n0, n1, n2},
                    std::array<long long int, 3>{inembed0, inembed1, inembed2},
                    istride,
                    idist,
                    std::array<long long int, 3>{onembed0, onembed1, onembed2},
                    ostride,
                    odist,
                    batch);
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
            work_sizes.push_back( el.second->get_work_size() );
        }
        auto max_work_size = std::max_element(work_sizes.cbegin(), work_sizes.cend());
        work_area_size_ = *max_work_size;
        
        CUDA_SAFE_CALL( cudaMalloc( &work_area_, work_area_size_ ) );

        for(auto &el: container_)
        {
            el.second->set_work_area(work_area_);
        }

        activated_ = true;

    }

    std::size_t get_work_size() const
    {
        return work_area_size_;
    }

    template<class ArrayIn, class ArrayOut>
    void exec(const std::string& name, const ArrayIn& in, ArrayOut& out)
    {
        container_[name]->exec(in.raw_ptr(), out.raw_ptr());
    }


private:
    void* work_area_;
    bool activated_;
    std::size_t work_area_size_; //in bytes!!!
    std::map<std::string, std::unique_ptr<wrap_t> > container_;

};



}
}

#endif // __FFTM_CUFFT_WRAP_MANY_H__
