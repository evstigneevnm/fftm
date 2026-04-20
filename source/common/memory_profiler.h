#ifndef __FFTM_COMMON_MEMORY_PROFILER_H__
#define __FFTM_COMMON_MEMORY_PROFILER_H__

#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <sstream>
#include <string>

namespace fftm
{

class memory_profiler
{
public:
    using bytes_type = unsigned long long;

    struct entry
    {
        bytes_type current_bytes = 0;
        bytes_type peak_bytes    = 0;
    };

    memory_profiler() : name_( "Memory profile" )
    {
    }

    explicit memory_profiler( const std::string &name ) : name_( name )
    {
    }

    void set_bytes( const std::string &key, bytes_type bytes )
    {
        entry &item        = entries_[key];
        item.current_bytes = bytes;
        if ( item.peak_bytes < bytes )
        {
            item.peak_bytes = bytes;
        }
    }

    void clear_bytes( const std::string &key )
    {
        set_bytes( key, 0 );
    }

    const std::map<std::string, entry> &entries() const
    {
        return entries_;
    }

    bytes_type total_current_bytes() const
    {
        bytes_type total = 0;
        for ( const auto &item : entries_ )
        {
            total += item.second.current_bytes;
        }
        return total;
    }

    bytes_type total_peak_bytes() const
    {
        bytes_type total = 0;
        for ( const auto &item : entries_ )
        {
            total += item.second.peak_bytes;
        }
        return total;
    }

    void print( std::ostream &out ) const
    {
        ios_saver saver( out );
        out << name_ << ":" << std::endl;
        for ( const auto &item : entries_ )
        {
            out << "  " << item.first << ": current=" << item.second.current_bytes << " B"
                << " (" << std::fixed << std::setprecision( 3 ) << bytes_to_mib_( item.second.current_bytes ) << " MiB)"
                << ", peak=" << item.second.peak_bytes << " B"
                << " (" << std::fixed << std::setprecision( 3 ) << bytes_to_mib_( item.second.peak_bytes ) << " MiB)"
                << std::endl;
        }
    }

    void print_totals( std::ostream &out ) const
    {
        ios_saver saver( out );
        out << name_ << " totals:"
            << " current=" << total_current_bytes() << " B"
            << " (" << std::fixed << std::setprecision( 3 ) << bytes_to_mib_( total_current_bytes() ) << " MiB)"
            << ", peak=" << total_peak_bytes() << " B"
            << " (" << std::fixed << std::setprecision( 3 ) << bytes_to_mib_( total_peak_bytes() ) << " MiB)"
            << std::endl;
    }

    template <class Log>
    void log_print( Log &log ) const
    {
        std::stringstream ss;
        print( ss );
        log.info( ss.str() );
    }

    template <class Log>
    void log_print_totals( Log &log ) const
    {
        std::stringstream ss;
        print_totals( ss );
        log.info( ss.str() );
    }

private:
    struct ios_saver
    {
        std::ios_base          &s;
        std::ios_base::fmtflags f;
        std::streamsize         p;

        explicit ios_saver( std::ios_base &stream ) : s( stream ), f( stream.flags() ), p( stream.precision() )
        {
        }

        ~ios_saver()
        {
            s.flags( f );
            s.precision( p );
        }
    };

    double bytes_to_mib_( bytes_type bytes ) const
    {
        return static_cast<double>( bytes ) / ( 1024.0 * 1024.0 );
    }

    std::string                  name_;
    std::map<std::string, entry> entries_;
};

} // namespace fftm

#endif
