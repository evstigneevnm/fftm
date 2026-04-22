#ifndef __FFTM_DETAIL_MEMORY_PROFILE_UTILS_H__
#define __FFTM_DETAIL_MEMORY_PROFILE_UTILS_H__

#include <array>
#include <cstddef>
#include <string>

namespace fftm
{
namespace detail
{

enum class memory_profile_bucket
{
    device = 0,
    host_pinned,
    external_test,
    other,
    count_
};

inline const char *memory_profile_bucket_name( memory_profile_bucket bucket )
{
    switch ( bucket )
    {
    case memory_profile_bucket::device:
        return "device";
    case memory_profile_bucket::host_pinned:
        return "host_pinned";
    case memory_profile_bucket::external_test:
        return "external_test";
    case memory_profile_bucket::other:
        return "other";
    case memory_profile_bucket::count_:
        break;
    }
    return "unknown";
}

inline memory_profile_bucket classify_memory_profile_key( const std::string &key )
{
    if ( key.rfind( "external/", 0 ) == 0 )
    {
        return memory_profile_bucket::external_test;
    }
    if ( key.find( "host_" ) != std::string::npos || key.find( "shared_host_work_buffer" ) != std::string::npos )
    {
        return memory_profile_bucket::host_pinned;
    }
    if ( key.rfind( "fftm/", 0 ) == 0 || key.rfind( "ffts/", 0 ) == 0 )
    {
        return memory_profile_bucket::device;
    }
    return memory_profile_bucket::other;
}

template <class Bytes>
struct memory_profile_totals
{
    Bytes current = 0;
    Bytes peak    = 0;
};

template <class Bytes>
class memory_profile_buckets
{
public:
    void add( const std::string &key, Bytes current, Bytes peak )
    {
        auto &slot   = data_[static_cast<std::size_t>( classify_memory_profile_key( key ) )];
        slot.current += current;
        slot.peak += peak;
    }

    const memory_profile_totals<Bytes> &get( memory_profile_bucket bucket ) const
    {
        return data_[static_cast<std::size_t>( bucket )];
    }

    memory_profile_totals<Bytes> total() const
    {
        memory_profile_totals<Bytes> result;
        for ( std::size_t i = 0; i < static_cast<std::size_t>( memory_profile_bucket::count_ ); ++i )
        {
            result.current += data_[i].current;
            result.peak += data_[i].peak;
        }
        return result;
    }

private:
    std::array<memory_profile_totals<Bytes>, static_cast<std::size_t>( memory_profile_bucket::count_ )> data_ = {};
};

} // namespace detail
} // namespace fftm

#endif
