#ifndef FFTM_HIP_STREAM_WRAP_H
#define FFTM_HIP_STREAM_WRAP_H

#include <cassert>
#include <utility>

#include <hip/hip_runtime.h>

#include <scfd/utils/hip_safe_call.h>

namespace fftm
{
namespace wrap
{

class hip_stream_wrap
{
public:
    explicit hip_stream_wrap( bool initialize = false )
        : initialized_( false ), stream_( nullptr )
    {
        if ( initialize )
            init();
    }

    hip_stream_wrap( const hip_stream_wrap & ) = delete;
    hip_stream_wrap &operator=( const hip_stream_wrap & ) = delete;

    hip_stream_wrap( hip_stream_wrap &&other ) noexcept
        : initialized_( other.initialized_ ), stream_( other.stream_ )
    {
        other.initialized_ = false;
        other.stream_      = nullptr;
    }

    hip_stream_wrap &operator=( hip_stream_wrap &&other )
    {
        if ( this != &other )
        {
            free();
            initialized_       = other.initialized_;
            stream_            = other.stream_;
            other.initialized_ = false;
            other.stream_      = nullptr;
        }
        return *this;
    }

    ~hip_stream_wrap()
    {
        free();
    }

    bool is_inited() const
    {
        return initialized_;
    }

    hipStream_t stream() const
    {
        assert( initialized_ );
        return stream_;
    }

    void init()
    {
        assert( !initialized_ );
        HIP_SAFE_CALL( hipStreamCreate( &stream_ ) );
        initialized_ = true;
    }

    void free()
    {
        if ( !initialized_ )
            return;
        initialized_ = false;
        HIP_SAFE_CALL( hipStreamDestroy( stream_ ) );
        stream_ = nullptr;
    }

private:
    bool        initialized_;
    hipStream_t stream_;
};

} // namespace wrap
} // namespace fftm

#endif // FFTM_HIP_STREAM_WRAP_H
