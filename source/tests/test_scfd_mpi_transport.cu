// Focused SCFD CUDA-aware MPI transport diagnostic.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/arrays/array_nd.h>
#include <scfd/backend/backend.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/log_mpi.h>

namespace
{

using byte_t = unsigned char;
using comm_t = scfd::communication::mpi_comm_info;
using request_t = comm_t::request_type;
using device_memory_t = scfd::backend::memory;
using host_memory_t = typename device_memory_t::host_memory_type;
using device_array_t = scfd::arrays::array_nd<byte_t, 1, device_memory_t>;
using host_array_t = scfd::arrays::array_nd<byte_t, 1, host_memory_t>;

enum class transport_t
{
    device,
    pinned,
    staged
};

struct options_t
{
    std::vector<int> active_pairs{ 1, 2, 4, 8 };
    std::vector<int> message_mib{ 64, 256, 512, 1024 };
    std::vector<transport_t> transports{ transport_t::device, transport_t::pinned, transport_t::staged };
    int iterations = 10;
    int warmup = 3;
    std::string output = "scfd_mpi_transport.csv";
};

struct topology_t
{
    int node_count = 0;
    int local_rank = -1;
    int local_size = 0;
    int node_leader = -1;
    int pair_index = -1;
    int peer = -1;
    int available_pairs = 0;
};

const char *transport_name( transport_t transport )
{
    switch ( transport )
    {
    case transport_t::device:
        return "device";
    case transport_t::pinned:
        return "pinned";
    case transport_t::staged:
        return "staged";
    }
    return "unknown";
}

transport_t parse_transport( const std::string &value )
{
    if ( value == "device" )
        return transport_t::device;
    if ( value == "pinned" )
        return transport_t::pinned;
    if ( value == "staged" )
        return transport_t::staged;
    throw std::runtime_error( "invalid transport: " + value );
}

std::vector<std::string> split_csv( const std::string &value )
{
    std::vector<std::string> result;
    std::stringstream stream( value );
    std::string item;
    while ( std::getline( stream, item, ',' ) )
    {
        if ( !item.empty() )
            result.push_back( item );
    }
    if ( result.empty() )
        throw std::runtime_error( "empty comma-separated option" );
    return result;
}

std::vector<int> parse_int_csv( const std::string &value )
{
    std::vector<int> result;
    for ( const auto &item : split_csv( value ) )
    {
        const int parsed = std::stoi( item );
        if ( parsed <= 0 )
            throw std::runtime_error( "list values must be positive: " + item );
        result.push_back( parsed );
    }
    return result;
}

bool env_enabled( const char *name )
{
    const char *value = std::getenv( name );
    return value && value[0] && value[0] != '0';
}

const char *env_or_empty( const char *name )
{
    const char *value = std::getenv( name );
    return value ? value : "";
}

void print_usage( const char *program )
{
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "  --active-pairs 1,2,4,8\n"
        << "  --message-mib 64,256,512,1024\n"
        << "  --transports device,pinned,staged\n"
        << "  --iterations N\n"
        << "  --warmup N\n"
        << "  --output FILE\n";
}

options_t parse_options( int argc, char **argv )
{
    options_t options;
    for ( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        auto require_value = [&]( const char *name ) -> std::string {
            if ( i + 1 >= argc )
                throw std::runtime_error( std::string( "missing value after " ) + name );
            return argv[++i];
        };
        if ( arg == "--active-pairs" )
        {
            options.active_pairs = parse_int_csv( require_value( "--active-pairs" ) );
        }
        else if ( arg == "--message-mib" )
        {
            options.message_mib = parse_int_csv( require_value( "--message-mib" ) );
        }
        else if ( arg == "--transports" )
        {
            options.transports.clear();
            for ( const auto &item : split_csv( require_value( "--transports" ) ) )
                options.transports.push_back( parse_transport( item ) );
        }
        else if ( arg == "--iterations" )
        {
            options.iterations = std::stoi( require_value( "--iterations" ) );
        }
        else if ( arg == "--warmup" )
        {
            options.warmup = std::stoi( require_value( "--warmup" ) );
        }
        else if ( arg == "--output" )
        {
            options.output = require_value( "--output" );
        }
        else if ( arg == "--help" || arg == "-h" )
        {
            print_usage( argv[0] );
            std::exit( 0 );
        }
        else
        {
            throw std::runtime_error( "unknown option: " + arg );
        }
    }
    if ( options.iterations <= 0 || options.warmup < 0 )
        throw std::runtime_error( "iterations must be positive and warmup must be nonnegative" );
    return options;
}

std::string samples_path( const std::string &summary_path )
{
    const std::string suffix = ".csv";
    if ( summary_path.size() >= suffix.size() &&
         summary_path.compare( summary_path.size() - suffix.size(), suffix.size(), suffix ) == 0 )
    {
        return summary_path.substr( 0, summary_path.size() - suffix.size() ) + ".samples.csv";
    }
    return summary_path + ".samples.csv";
}

topology_t build_topology( const comm_t &comm )
{
    topology_t topology;
    auto local_comm = comm.split_type( MPI_COMM_TYPE_SHARED );
    const auto local_info = local_comm.info();
    topology.local_rank = local_info.myid;
    topology.local_size = local_info.num_procs;
    topology.node_leader = local_info.all_reduce_min( comm.myid );
    local_comm.free();

    std::vector<int> leaders( comm.num_procs, -1 );
    std::vector<int> local_ranks( comm.num_procs, -1 );
    comm.all_gather( &topology.node_leader, 1, leaders.data(), 1 );
    comm.all_gather( &topology.local_rank, 1, local_ranks.data(), 1 );

    std::set<int> unique_leaders( leaders.begin(), leaders.end() );
    topology.node_count = static_cast<int>( unique_leaders.size() );
    if ( topology.node_count == 2 )
    {
        topology.available_pairs = comm.all_reduce_min( topology.local_size );
        for ( int rank = 0; rank < comm.num_procs; ++rank )
        {
            if ( leaders[rank] != topology.node_leader && local_ranks[rank] == topology.local_rank )
            {
                topology.peer = rank;
                break;
            }
        }
        topology.pair_index = topology.local_rank;
    }
    else if ( topology.node_count == 1 && comm.num_procs >= 2 && comm.num_procs % 2 == 0 )
    {
        topology.available_pairs = comm.num_procs / 2;
        if ( comm.myid < topology.available_pairs )
        {
            topology.pair_index = comm.myid;
            topology.peer = comm.myid + topology.available_pairs;
        }
        else
        {
            topology.pair_index = comm.myid - topology.available_pairs;
            topology.peer = comm.myid - topology.available_pairs;
        }
    }
    else
    {
        throw std::runtime_error( "transport diagnostic requires one even-sized node or exactly two nodes" );
    }
    if ( topology.peer < 0 )
        throw std::runtime_error( "failed to find a matching peer rank" );
    return topology;
}

double mean( const std::vector<double> &values )
{
    return std::accumulate( values.begin(), values.end(), 0.0 ) / static_cast<double>( values.size() );
}

double standard_deviation( const std::vector<double> &values, double average )
{
    double sum = 0.0;
    for ( double value : values )
    {
        const double delta = value - average;
        sum += delta * delta;
    }
    return std::sqrt( sum / static_cast<double>( values.size() ) );
}

byte_t pattern_for_rank( int rank )
{
    return static_cast<byte_t>( ( 37 * rank + 11 ) & 0xff );
}

bool verify_received(
    transport_t transport,
    const topology_t &topology,
    std::size_t message_bytes,
    const device_array_t &device_recv,
    const host_array_t &host_recv
)
{
    const std::array<std::size_t, 6> offsets{
        0,
        1,
        message_bytes / 3,
        message_bytes / 2,
        message_bytes - 2,
        message_bytes - 1
    };
    const byte_t expected = pattern_for_rank( topology.peer );
    for ( std::size_t offset : offsets )
    {
        byte_t actual = 0;
        if ( transport == transport_t::pinned )
        {
            actual = host_recv.raw_ptr()[offset];
        }
        else
        {
            device_memory_t::copy_to_host( 1, device_recv.raw_ptr() + offset, &actual );
        }
        if ( actual != expected )
            return false;
    }
    return true;
}

void initialize_send_buffer(
    transport_t transport,
    int rank,
    std::size_t max_bytes,
    device_array_t &device_send,
    host_array_t &host_send
)
{
    const byte_t pattern = pattern_for_rank( rank );
    if ( transport == transport_t::device )
    {
        host_array_t initializer;
        initializer.init( static_cast<std::ptrdiff_t>( max_bytes ) );
        std::memset( initializer.raw_ptr(), pattern, max_bytes );
        device_memory_t::copy_from_host( max_bytes, initializer.raw_ptr(), device_send.raw_ptr() );
        initializer.free();
    }
    else
    {
        std::memset( host_send.raw_ptr(), pattern, max_bytes );
        if ( transport == transport_t::staged )
            device_memory_t::copy_from_host( max_bytes, host_send.raw_ptr(), device_send.raw_ptr() );
    }
}

} // namespace

int main( int argc, char **argv )
{
    try
    {
        scfd::communication::mpi_wrap mpi( argc, argv );
        auto comm = mpi.comm_world();
        scfd::utils::log_mpi log;
        const options_t options = parse_options( argc, argv );

        const bool wrap_devices = env_enabled( "FFTM_WRAP_PROCS_GPUS" );
        const int device_id = scfd::backend::runtime::init_device( log, comm, 0, wrap_devices );
        const topology_t topology = build_topology( comm );

        const int requested_max_pairs = *std::max_element( options.active_pairs.begin(), options.active_pairs.end() );
        if ( requested_max_pairs > topology.available_pairs )
        {
            throw std::runtime_error(
                "requested active-pair count exceeds available pairs: requested " +
                std::to_string( requested_max_pairs ) + ", available " +
                std::to_string( topology.available_pairs )
            );
        }

        const int max_message_mib = *std::max_element( options.message_mib.begin(), options.message_mib.end() );
        const std::size_t max_message_bytes =
            static_cast<std::size_t>( max_message_mib ) * 1024u * 1024u;
        if ( max_message_bytes > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
            throw std::runtime_error( "message size exceeds the MPI int-count limit" );

        std::cerr
            << "SCFD_MPI_TOPOLOGY"
            << " rank=" << comm.myid << "/" << comm.num_procs
            << " node_leader=" << topology.node_leader
            << " local_rank=" << topology.local_rank << "/" << topology.local_size
            << " pair_index=" << topology.pair_index
            << " peer=" << topology.peer
            << " device=" << device_id
            << "\n";

        std::ofstream summary;
        std::ofstream samples;
        int output_ready = 1;
        if ( comm.myid == 0 )
        {
            summary.open( options.output );
            samples.open( samples_path( options.output ) );
            if ( !summary || !samples )
            {
                output_ready = 0;
            }
            else
            {
                summary
                    << "transport,active_pairs,message_mib,message_bytes,iterations,warmup,"
                    << "mean_ms,stddev_ms,min_ms,max_ms,one_way_per_rank_GBps,"
                    << "bidirectional_per_pair_GBps,aggregate_bidirectional_GBps,"
                    << "valid,node_count,ranks_per_node\n";
                samples
                    << "transport,active_pairs,message_mib,iteration,global_ms\n";
                std::cerr
                    << "SCFD_MPI_ENV"
                    << " UCX_TLS=\"" << env_or_empty( "UCX_TLS" ) << "\""
                    << " UCX_NET_DEVICES=\"" << env_or_empty( "UCX_NET_DEVICES" ) << "\""
                    << " UCX_RNDV_SCHEME=\"" << env_or_empty( "UCX_RNDV_SCHEME" ) << "\""
                    << " UCX_PROTO_INFO=\"" << env_or_empty( "UCX_PROTO_INFO" ) << "\""
                    << " OMPI_MCA_pml=\"" << env_or_empty( "OMPI_MCA_pml" ) << "\""
                    << "\n";
            }
        }
        output_ready = comm.all_reduce_min( output_ready );
        if ( !output_ready )
            throw std::runtime_error( "failed to open output CSV files" );

        for ( transport_t transport : options.transports )
        {
            device_array_t device_send;
            device_array_t device_recv;
            host_array_t host_send;
            host_array_t host_recv;

            if ( transport != transport_t::pinned )
            {
                device_send.init( static_cast<std::ptrdiff_t>( max_message_bytes ) );
                device_recv.init( static_cast<std::ptrdiff_t>( max_message_bytes ) );
            }
            if ( transport != transport_t::device )
            {
                host_send.init( static_cast<std::ptrdiff_t>( max_message_bytes ) );
                host_recv.init( static_cast<std::ptrdiff_t>( max_message_bytes ) );
            }
            initialize_send_buffer(
                transport,
                comm.myid,
                max_message_bytes,
                device_send,
                host_send
            );
            scfd::backend::runtime::synchronize();
            comm.barrier();

            for ( int active_pairs : options.active_pairs )
            {
                const bool active = topology.pair_index < active_pairs;
                for ( int message_mib : options.message_mib )
                {
                    const std::size_t message_bytes =
                        static_cast<std::size_t>( message_mib ) * 1024u * 1024u;
                    const int message_count = static_cast<int>( message_bytes );
                    std::vector<double> measured_ms;
                    measured_ms.reserve( options.iterations );

                    for ( int iteration = -options.warmup; iteration < options.iterations; ++iteration )
                    {
                        scfd::backend::runtime::synchronize();
                        comm.barrier();
                        const double start = comm.wtime();
                        if ( active )
                        {
                            if ( transport == transport_t::staged )
                            {
                                device_memory_t::copy_to_host(
                                    message_bytes,
                                    device_send.raw_ptr(),
                                    host_send.raw_ptr()
                                );
                            }

                            std::array<request_t, 2> requests;
                            byte_t *recv_ptr =
                                transport == transport_t::device
                                    ? device_recv.raw_ptr()
                                    : host_recv.raw_ptr();
                            const byte_t *send_ptr =
                                transport == transport_t::device
                                    ? device_send.raw_ptr()
                                    : host_send.raw_ptr();
                            const int tag = 1000 + message_mib;
                            comm.irecv( recv_ptr, message_count, topology.peer, tag, requests[0] );
                            comm.isend( send_ptr, message_count, topology.peer, tag, requests[1] );
                            comm.waitall( static_cast<int>( requests.size() ), requests.data() );

                            if ( transport == transport_t::staged )
                            {
                                device_memory_t::copy_from_host(
                                    message_bytes,
                                    host_recv.raw_ptr(),
                                    device_recv.raw_ptr()
                                );
                            }
                        }
                        const double local_ms = 1000.0 * ( comm.wtime() - start );
                        const double global_ms = comm.all_reduce_max( local_ms );
                        if ( iteration >= 0 )
                            measured_ms.push_back( global_ms );
                    }

                    bool valid = true;
                    if ( active )
                    {
                        valid = verify_received(
                            transport,
                            topology,
                            message_bytes,
                            device_recv,
                            host_recv
                        );
                    }
                    const int globally_valid = comm.all_reduce_min( valid ? 1 : 0 );
                    const double average_ms = mean( measured_ms );
                    const double stddev_ms = standard_deviation( measured_ms, average_ms );
                    const auto minmax = std::minmax_element( measured_ms.begin(), measured_ms.end() );
                    const double one_way_GBps =
                        static_cast<double>( message_bytes ) / ( average_ms * 1.0e6 );
                    const double bidirectional_pair_GBps = 2.0 * one_way_GBps;
                    const double aggregate_GBps =
                        static_cast<double>( active_pairs ) * bidirectional_pair_GBps;

                    if ( comm.myid == 0 )
                    {
                        summary
                            << transport_name( transport ) << ","
                            << active_pairs << ","
                            << message_mib << ","
                            << message_bytes << ","
                            << options.iterations << ","
                            << options.warmup << ","
                            << std::setprecision( 12 ) << average_ms << ","
                            << stddev_ms << ","
                            << *minmax.first << ","
                            << *minmax.second << ","
                            << one_way_GBps << ","
                            << bidirectional_pair_GBps << ","
                            << aggregate_GBps << ","
                            << globally_valid << ","
                            << topology.node_count << ","
                            << topology.local_size << "\n";
                        for ( std::size_t sample = 0; sample < measured_ms.size(); ++sample )
                        {
                            samples
                                << transport_name( transport ) << ","
                                << active_pairs << ","
                                << message_mib << ","
                                << sample << ","
                                << std::setprecision( 12 ) << measured_ms[sample] << "\n";
                        }
                        summary.flush();
                        samples.flush();
                        std::cout
                            << "SCFD_MPI_RESULT"
                            << " transport=" << transport_name( transport )
                            << " pairs=" << active_pairs
                            << " message_mib=" << message_mib
                            << " mean_ms=" << average_ms
                            << " one_way_per_rank_GBps=" << one_way_GBps
                            << " aggregate_bidirectional_GBps=" << aggregate_GBps
                            << " valid=" << globally_valid
                            << "\n";
                    }
                    if ( !globally_valid )
                        throw std::runtime_error( "received payload validation failed" );
                }
            }
            comm.barrier();
        }

        if ( comm.myid == 0 )
            std::cout << "SCFD_MPI_TRANSPORT PASSED\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << "SCFD_MPI_TRANSPORT ERROR: " << error.what() << "\n";
        return 1;
    }
}
