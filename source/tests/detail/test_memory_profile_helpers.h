#ifndef __FFTM_TESTS_DETAIL_TEST_MEMORY_PROFILE_HELPERS_H__
#define __FFTM_TESTS_DETAIL_TEST_MEMORY_PROFILE_HELPERS_H__

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <type_traits>

#include <detail/memory_profile_utils.h>

namespace fftm
{
namespace test
{
namespace detail
{

template <class T>
inline std::uint64_t bytes_of_elems( std::size_t elems )
{
    return static_cast<std::uint64_t>( elems ) * static_cast<std::uint64_t>( sizeof( T ) );
}

template <class Array>
inline std::uint64_t array_bytes( const Array &array )
{
    return bytes_of_elems<typename Array::value_type>( static_cast<std::size_t>( array.total_size() ) );
}

template <class... Bytes>
inline std::uint64_t sum_bytes( Bytes... bytes )
{
    std::uint64_t total = 0;
    using swallow        = int[];
    (void)swallow{ 0, ( total += static_cast<std::uint64_t>( bytes ), 0 )... };
    return total;
}

template <class Comm, class Bytes>
inline void append_memory_profile_line_mpi(
    std::stringstream &ss, const Comm &comm, const char *name, Bytes current_local, Bytes peak_local
)
{
    const auto current_sum = comm.all_reduce_sum( current_local );
    const auto current_max = comm.all_reduce_max( current_local );
    const auto peak_sum    = comm.all_reduce_sum( peak_local );
    const auto peak_max    = comm.all_reduce_max( peak_local );

    if ( comm.myid != 0 )
    {
        return;
    }

    auto bytes_to_mib = []( Bytes bytes ) -> double { return static_cast<double>( bytes ) / ( 1024.0 * 1024.0 ); };

    ss << "\n  " << name << ": current(sum/max/avg)=" << current_sum << "/" << current_max << "/"
       << static_cast<Bytes>( current_sum / comm.num_procs ) << " B (" << std::fixed << std::setprecision( 3 )
       << bytes_to_mib( current_sum ) << "/" << bytes_to_mib( current_max ) << "/"
       << bytes_to_mib( static_cast<Bytes>( current_sum / comm.num_procs ) ) << " MiB)"
       << ", peak(sum/max/avg)=" << peak_sum << "/" << peak_max << "/"
       << static_cast<Bytes>( peak_sum / comm.num_procs ) << " B (" << bytes_to_mib( peak_sum ) << "/"
       << bytes_to_mib( peak_max ) << "/" << bytes_to_mib( static_cast<Bytes>( peak_sum / comm.num_procs ) )
       << " MiB)";
}

template <class FFT, class Log, class Comm>
inline void log_tracked_memory_categories_mpi(
    Log &log, const Comm &comm, const char *label, typename FFT::memory_profile_bytes_t internal_device_current,
    typename FFT::memory_profile_bytes_t internal_device_peak, typename FFT::memory_profile_bytes_t internal_host_current,
    typename FFT::memory_profile_bytes_t internal_host_peak, typename FFT::memory_profile_bytes_t external_device_current,
    typename FFT::memory_profile_bytes_t external_device_peak, typename FFT::memory_profile_bytes_t internal_other_current = 0,
    typename FFT::memory_profile_bytes_t internal_other_peak = 0
)
{
    using bytes_t = typename FFT::memory_profile_bytes_t;

    const auto tracked_device_cur = static_cast<bytes_t>( internal_device_current + external_device_current );
    const auto tracked_device_pk  = static_cast<bytes_t>( internal_device_peak + external_device_peak );
    const auto tracked_total_cur  =
        static_cast<bytes_t>( tracked_device_cur + internal_host_current + internal_other_current );
    const auto tracked_total_pk =
        static_cast<bytes_t>( tracked_device_pk + internal_host_peak + internal_other_peak );

    std::stringstream ss;
    ss << label << " tracked memory incl. external/test-owned (MPI reduced):";
    append_memory_profile_line_mpi( ss, comm, "internal_device", internal_device_current, internal_device_peak );
    append_memory_profile_line_mpi( ss, comm, "host_pinned_internal", internal_host_current, internal_host_peak );
    append_memory_profile_line_mpi( ss, comm, "external_test_owned", external_device_current, external_device_peak );
    append_memory_profile_line_mpi( ss, comm, "tracked_device_total", tracked_device_cur, tracked_device_pk );

    if ( internal_other_current != 0 || internal_other_peak != 0 )
    {
        append_memory_profile_line_mpi( ss, comm, "other_internal", internal_other_current, internal_other_peak );
    }

    append_memory_profile_line_mpi( ss, comm, "tracked_total", tracked_total_cur, tracked_total_pk );

    if ( comm.myid == 0 )
    {
        log.info( ss.str() );
    }
}

template <class FFT, class Log, class Comm>
inline void log_tracked_memory_with_external_mpi(
    Log &log, const Comm &comm, const FFT &fft, const char *label, typename FFT::memory_profile_bytes_t external_device_current,
    typename FFT::memory_profile_bytes_t external_device_peak
)
{
    if ( !fft.is_memory_profiling_enabled() )
    {
        return;
    }

    const auto buckets = fft.get_memory_profile_buckets();
    log_tracked_memory_categories_mpi<FFT>(
        log, comm, label, buckets.get( ::fftm::detail::memory_profile_bucket::device ).current,
        buckets.get( ::fftm::detail::memory_profile_bucket::device ).peak,
        buckets.get( ::fftm::detail::memory_profile_bucket::host_pinned ).current,
        buckets.get( ::fftm::detail::memory_profile_bucket::host_pinned ).peak, external_device_current,
        external_device_peak, buckets.get( ::fftm::detail::memory_profile_bucket::other ).current,
        buckets.get( ::fftm::detail::memory_profile_bucket::other ).peak
    );
}

} // namespace detail
} // namespace test
} // namespace fftm

#endif
