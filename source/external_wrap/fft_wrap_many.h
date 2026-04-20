#ifndef __FFTM_FFT_WRAP_MANY_H__
#define __FFTM_FFT_WRAP_MANY_H__

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <scfd/arrays/array_nd.h>

#include "fft_direction.h"
#include "../profiling.h"

namespace fftm
{
namespace wrap
{

template <template <typename, ::fftm::direction> class BaseFFTWrap, class T>
class fft_wrap_many
{
private:
    using wrap_t       = typename BaseFFTWrap<T, ::fftm::direction::C2CF>::base_t;
    using memory_t     = typename wrap_t::memory_type;
    using ordinal_type = scfd::arrays::ordinal_type;
    using work_array_t = scfd::arrays::array_nd<char, 1, memory_t>;

public:
    using real              = T;
    using complex           = typename wrap_t::complex;
    using runtime_api       = typename wrap_t::runtime_api;
    using memory_profiler_t = ::fftm::fftm_memory_profiler;

    fft_wrap_many() : work_area_size_( 0 ), activated_( false )
    {
    }

    template <::fftm::direction D, std::size_t Rank>
    void add_plan(
        const std::string &name, const std::array<long long int, Rank> &n,
        const std::array<long long int, Rank> &inembed, long long int istride, long long int idist,
        const std::array<long long int, Rank> &onembed, long long int ostride, long long int odist, long long int batch
    )
    {
        if ( activated_ )
        {
            throw std::logic_error( "fft_wrap_many::add_plan: cannot add new plans after activation." );
        }

        container_.emplace(
            name, std::make_unique<BaseFFTWrap<T, D>>( n, inembed, istride, idist, onembed, ostride, odist, batch )
        );
    }

    template <::fftm::direction D>
    void add_plan_1D(
        const std::string &name, long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch
    )
    {
        add_plan<D>(
            name, std::array<long long int, 1>{ n }, std::array<long long int, 1>{ inembed }, istride, idist,
            std::array<long long int, 1>{ onembed }, ostride, odist, batch
        );
    }

    template <::fftm::direction D>
    void add_plan_2D(
        const std::string &name, long long int n0, long long int n1, long long int inembed0, long long int inembed1,
        long long int istride, long long int idist, long long int onembed0, long long int onembed1,
        long long int ostride, long long int odist, long long int batch
    )
    {
        add_plan<D>(
            name, std::array<long long int, 2>{ n0, n1 }, std::array<long long int, 2>{ inembed0, inembed1 }, istride,
            idist, std::array<long long int, 2>{ onembed0, onembed1 }, ostride, odist, batch
        );
    }

    template <::fftm::direction D>
    void add_plan_3D(
        const std::string &name, long long int n0, long long int n1, long long int n2, long long int inembed0,
        long long int inembed1, long long int inembed2, long long int istride, long long int idist,
        long long int onembed0, long long int onembed1, long long int onembed2, long long int ostride,
        long long int odist, long long int batch
    )
    {
        add_plan<D>(
            name, std::array<long long int, 3>{ n0, n1, n2 },
            std::array<long long int, 3>{ inembed0, inembed1, inembed2 }, istride, idist,
            std::array<long long int, 3>{ onembed0, onembed1, onembed2 }, ostride, odist, batch
        );
    }

    void activate( std::size_t additional_size )
    {
        if ( activated_ )
        {
            throw std::logic_error( "fft_wrap_many::activate: cannot activate again." );
        }

        work_area_size_ = 0;
        for ( const auto &el : container_ )
        {
            work_area_size_ = std::max( work_area_size_, el.second->get_work_size() );
        }

        if ( work_area_size_ > 0 )
        {
            work_area_.init( ordinal_cast_( work_area_size_ ) );
            for ( auto &el : container_ )
            {
                el.second->set_work_area( static_cast<void *>( work_area_.raw_ptr() ) );
            }
        }

        update_memory_profile_();
        activated_ = true;
    }

    std::size_t get_work_size() const
    {
        return work_area_size_;
    }

    void set_memory_profiler( memory_profiler_t *profiler, const std::string &prefix )
    {
        memory_profiler_       = profiler;
        memory_profile_prefix_ = prefix;
        update_memory_profile_();
    }

    template <class ArrayIn, class ArrayOut>
    void exec( const std::string &name, const ArrayIn &in, ArrayOut &out )
    {
        container_.at( name )->exec( in.raw_ptr(), out.raw_ptr() );
    }

private:
    ordinal_type ordinal_cast_( std::size_t value ) const
    {
        const auto max_value = static_cast<std::size_t>( std::numeric_limits<ordinal_type>::max() );
        if ( value > max_value )
        {
            throw std::overflow_error(
                "fft_wrap_many::activate: work area size exceeds scfd::arrays::ordinal_type range. "
                "Rebuild with a wider SCFD_ARRAYS_ORDINAL_TYPE."
            );
        }
        return static_cast<ordinal_type>( value );
    }

    void update_memory_profile_()
    {
        if ( memory_profiler_ == nullptr || memory_profile_prefix_.empty() )
        {
            return;
        }

        memory_profiler_->set_bytes(
            memory_profile_prefix_ + "/work_area", static_cast<memory_profiler_t::bytes_type>( work_area_size_ )
        );
    }

    work_array_t                                   work_area_;
    std::size_t                                    work_area_size_;
    bool                                           activated_;
    std::map<std::string, std::unique_ptr<wrap_t>> container_;
    memory_profiler_t                             *memory_profiler_ = nullptr;
    std::string                                    memory_profile_prefix_;
};

} // namespace wrap
} // namespace fftm

#endif
