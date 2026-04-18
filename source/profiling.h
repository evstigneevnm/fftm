#ifndef __FFTM_PROFILING_H__
#define __FFTM_PROFILING_H__

#include <memory>
#include <string>

#include <scfd/utils/mpi_timer_event.h>
#include <scfd/utils/system_timer_event.h>

#include "common/profiler.h"

namespace fftm
{

using fftm_profiler = scfd::utils::profiler<scfd::utils::mpi_timer_event>;
using ffts_profiler = scfd::utils::profiler<scfd::utils::system_timer_event>;

template <class Profiler>
class optional_profiler
{
public:
    class scoped_ticker
    {
    public:
        scoped_ticker()
            : profiler_( nullptr )
        {
        }

        scoped_ticker( Profiler *profiler, const std::string &name )
            : profiler_( profiler )
        {
            if ( profiler_ != nullptr )
            {
                profiler_->tic( name );
            }
        }

        ~scoped_ticker()
        {
            if ( profiler_ != nullptr )
            {
                profiler_->toc();
            }
        }

        scoped_ticker( const scoped_ticker & ) = delete;
        scoped_ticker &operator=( const scoped_ticker & ) = delete;

        scoped_ticker( scoped_ticker &&other ) noexcept
            : profiler_( other.profiler_ )
        {
            other.profiler_ = nullptr;
        }

        scoped_ticker &operator=( scoped_ticker &&other ) noexcept
        {
            if ( this != &other )
            {
                if ( profiler_ != nullptr )
                {
                    profiler_->toc();
                }
                profiler_       = other.profiler_;
                other.profiler_ = nullptr;
            }
            return *this;
        }

    private:
        Profiler *profiler_;
    };

    optional_profiler() = default;

    void enable( const std::string &name )
    {
        profiler_.reset( new Profiler( name ) );
    }

    void disable()
    {
        profiler_.reset();
    }

    bool enabled() const
    {
        return profiler_ != nullptr;
    }

    scoped_ticker scoped_tic( const std::string &name )
    {
        return scoped_ticker( profiler_.get(), name );
    }

    template <class Log>
    void log_print( Log &log ) const
    {
        if ( profiler_ != nullptr )
        {
            profiler_->log_print( log );
        }
    }

    template <class Log>
    void log_print_totals( Log &log ) const
    {
        if ( profiler_ != nullptr )
        {
            profiler_->log_print_totals( log );
        }
    }

private:
    std::unique_ptr<Profiler> profiler_;
};

#define __FFTM_PROFILE_CONCAT_IMPL( a, b ) a##b
#define __FFTM_PROFILE_CONCAT( a, b ) __FFTM_PROFILE_CONCAT_IMPL( a, b )
#define FFTM_PROFILE_SCOPED_TIC( name ) auto __FFTM_PROFILE_CONCAT( _fftm_profile_scope_, __LINE__ ) = this->profile_scope_( name )

} // namespace fftm

#endif
