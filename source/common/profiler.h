#ifndef __FFTM_COMMON_PROFILER_H__
#define __FFTM_COMMON_PROFILER_H__

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <scfd/utils/manual_init_singleton.h>
#include <scfd/utils/system_timer_event.h>

namespace scfd
{
namespace utils
{

template <class Event = system_timer_event, unsigned SHIFT_WIDTH = 2>
class profiler : public manual_init_singleton<profiler<Event, SHIFT_WIDTH>>
{
public:
    typedef double delta_type;

    profiler()
        : name( "Profile" )
    {
        init();
    }

    explicit profiler( const std::string &name_ )
        : name( name_ )
    {
        init();
    }

    void tic( const std::string &interval_name )
    {
        if ( stack.back()->children.find( interval_name ) == stack.back()->children.end() )
        {
            std::size_t next_call_index                        = stack.back()->children.size();
            stack.back()->children[interval_name].call_index = next_call_index;
        }
        stack.back()->children[interval_name].begin_event.record();
        stack.push_back( &stack.back()->children[interval_name] );
    }

    delta_type toc( const std::string & = "" )
    {
        profile_unit *top = stack.back();
        stack.pop_back();

        Event      current_event;
        current_event.record();
        delta_type delta = current_event.elapsed_time( top->begin_event );

        top->length += delta;
        root.length = current_event.elapsed_time( root.begin_event );

        return delta;
    }

    void reset()
    {
        stack.clear();
        root.length = 0;
        root.children.clear();

        stack.push_back( &root );
        root.begin_event.record();
    }

    struct scoped_ticker
    {
        profiler &prof;

        explicit scoped_ticker( profiler &prof_ )
            : prof( prof_ )
        {
        }

        ~scoped_ticker()
        {
            prof.toc();
        }
    };

    scoped_ticker scoped_tic( const std::string &interval_name )
    {
        tic( interval_name );
        return scoped_ticker( *this );
    }

    template <class Log>
    void log_print( Log &log )
    {
        std::stringstream sstream;
        print( sstream );
        log.info( sstream.str() );
    }

    template <class Log>
    void log_print_totals( Log &log )
    {
        std::stringstream sstream;
        print_totals( sstream );
        log.info( sstream.str() );
    }

    friend std::ostream &operator<<( std::ostream &out, profiler &prof )
    {
        prof.print( out );
        return out;
    }

private:
    struct profile_unit
    {
        profile_unit()
            : length( 0 )
            , call_index( 0 )
        {
        }

        delta_type children_time() const
        {
            delta_type s = delta_type();
            for ( typename std::map<std::string, profile_unit>::const_iterator c = children.begin(); c != children.end();
                  ++c )
            {
                s += c->second.length;
            }
            return s;
        }

        std::size_t total_width( const std::string &unit_name, int level ) const
        {
            std::size_t w = unit_name.size() + static_cast<std::size_t>( level );
            for ( typename std::map<std::string, profile_unit>::const_iterator c = children.begin(); c != children.end();
                  ++c )
            {
                w = std::max( w, c->second.total_width( c->first, level + SHIFT_WIDTH ) );
            }
            return w;
        }

        void print( std::ostream &out, const std::string &unit_name, int level, delta_type total, std::size_t width ) const
        {
            using namespace std;

            out << "[" << setw( level ) << "";
            print_line( out, unit_name, length, 100 * length / total, width - static_cast<std::size_t>( level ) );

            if ( children.size() )
            {
                delta_type val  = length - children_time();
                double     perc = 100.0 * val / total;

                if ( perc > 1e-1 )
                {
                    out << "[" << setw( level + SHIFT_WIDTH ) << "";
                    print_line(
                        out,
                        "self",
                        val,
                        perc,
                        width - static_cast<std::size_t>( level + SHIFT_WIDTH )
                    );
                }
            }

            std::map<std::size_t, std::pair<std::string, const profile_unit *>> children_sorted;
            for ( typename std::map<std::string, profile_unit>::const_iterator c = children.begin(); c != children.end();
                  ++c )
            {
                children_sorted[c->second.call_index] = std::make_pair( c->first, &c->second );
            }
            for ( typename std::map<std::size_t, std::pair<std::string, const profile_unit *>>::const_iterator c =
                      children_sorted.begin();
                  c != children_sorted.end();
                  ++c )
            {
                c->second.second->print( out, c->second.first, level + SHIFT_WIDTH, total, width );
            }
        }

        void add_to_totals( std::map<std::string, delta_type> &total_lengths ) const
        {
            for ( typename std::map<std::string, profile_unit>::const_iterator c = children.begin(); c != children.end();
                  ++c )
            {
                if ( total_lengths.find( c->first ) == total_lengths.end() )
                {
                    total_lengths[c->first] = 0.;
                }
                total_lengths[c->first] += c->second.length;
                c->second.add_to_totals( total_lengths );
            }
        }

        void print_line(
            std::ostream      &out,
            const std::string &unit_name,
            delta_type         time,
            double             perc,
            std::size_t        width
        ) const
        {
            using namespace std;

            out << unit_name << ":" << setw( static_cast<int>( width ) - static_cast<int>( unit_name.size() ) ) << ""
                << setw( 10 ) << fixed << setprecision( 3 ) << time << " " << Event::units() << "] (" << fixed
                << setprecision( 2 ) << setw( 6 ) << perc << "%)" << endl;
        }

        Event                        begin_event;
        delta_type                   length;
        std::size_t                  call_index;
        std::map<std::string, profile_unit> children;
    };

    struct ios_saver
    {
        std::ios_base          &s;
        std::ios_base::fmtflags f;
        std::streamsize         p;

        explicit ios_saver( std::ios_base &stream )
            : s( stream )
            , f( stream.flags() )
            , p( stream.precision() )
        {
        }

        ~ios_saver()
        {
            s.flags( f );
            s.precision( p );
        }
    };

    std::string               name;
    profile_unit              root;
    std::vector<profile_unit *> stack;

    void init()
    {
        stack.reserve( 128 );
        stack.push_back( &root );
        root.begin_event.record();
    }

    void print( std::ostream &out )
    {
        out << "Profile breakdown:" << std::endl;
        if ( stack.back() != &root )
        {
            out << "Warning! Profile is incomplete." << std::endl;
        }
        ios_saver ss( out );
        root.print( out, name, 0, root.length, root.total_width( name, 0 ) );
        out << std::endl;
    }

    void print_totals( std::ostream &out )
    {
        out << "Profile summarize:" << std::endl;
        if ( stack.back() != &root )
        {
            out << "Warning! Profile is incomplete." << std::endl;
        }
        std::map<std::string, delta_type> total_lengths;
        root.add_to_totals( total_lengths );
        delta_type total = root.length;
        std::size_t width = root.total_width( name, 0 );
        for ( typename std::map<std::string, delta_type>::const_iterator c = total_lengths.begin();
              c != total_lengths.end();
              ++c )
        {
            out << "[";
            root.print_line( out, c->first, c->second, 100 * c->second / total, width );
        }
        out << std::endl;
    }
};

} // namespace utils
} // namespace scfd

#endif
