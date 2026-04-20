#ifndef __FFTM_PROFILING_H__
#define __FFTM_PROFILING_H__

#include <memory>
#include <string>

#include <scfd/utils/mpi_timer_event.h>
#include <scfd/utils/system_timer_event.h>

#include "common/memory_profiler.h"
#include "common/profiler.h"

namespace fftm
{

using fftm_profiler        = scfd::utils::profiler<scfd::utils::mpi_timer_event>;
using ffts_profiler        = scfd::utils::profiler<scfd::utils::system_timer_event>;
using fftm_memory_profiler = memory_profiler;
using ffts_memory_profiler = memory_profiler;

template <class Profiler>
class profile_scope
{
public:
    profile_scope( Profiler *profiler, const std::string &name ) : profiler_( profiler )
    {
        if ( profiler_ != nullptr )
        {
            profiler_->tic( name );
        }
    }

    ~profile_scope()
    {
        if ( profiler_ != nullptr )
        {
            profiler_->toc();
        }
    }

    profile_scope( const profile_scope & )            = delete;
    profile_scope &operator=( const profile_scope & ) = delete;

    profile_scope( profile_scope &&other ) noexcept : profiler_( other.profiler_ )
    {
        other.profiler_ = nullptr;
    }

    profile_scope &operator=( profile_scope &&other ) noexcept
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

template <class Profiler>
class optional_profiler
{
public:
    using scoped_ticker = profile_scope<Profiler>;

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

    Profiler *native_ptr()
    {
        return profiler_.get();
    }

    const Profiler *native_ptr() const
    {
        return profiler_.get();
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

template <class Profiler>
class optional_memory_profiler
{
public:
    optional_memory_profiler() = default;

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

    Profiler *native_ptr()
    {
        return profiler_.get();
    }

    const Profiler *native_ptr() const
    {
        return profiler_.get();
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
#define FFTM_PROFILE_SCOPED_TIC( name )                                                                                \
    auto __FFTM_PROFILE_CONCAT( _fftm_profile_scope_, __LINE__ ) = this->profile_scope_( name )

} // namespace fftm

#endif
