#include "node_fft_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

#include <scfd/communication/mpi_comm.h>
#include <scfd/communication/mpi_wrap.h>

namespace
{

using fftm::experiments::node_hybrid::device_segment_view;
using fftm::experiments::node_hybrid::node_fft_backend;
using fftm::experiments::node_hybrid::shape_3d;
using comm_t = scfd::communication::mpi_comm_info;
using request_t = comm_t::request_type;

enum class exchange_mode
{
    none,
    ring
};

struct options
{
    shape_3d      shape{ 64, 64, 64 };
    int           gpus_per_process = 0;
    int           warmup = 2;
    int           iterations = 5;
    std::size_t   chunk_mib = 512;
    double        epsilon = 1.0e-11;
    bool          validate = true;
    exchange_mode exchange = exchange_mode::none;
    std::string   output = "node_hybrid_cufft_xt.csv";
    std::string   label = "unspecified";
};

struct topology
{
    int node_count = 0;
    int node_index = 0;
    int local_rank = 0;
    int local_size = 0;
    int source_peer = -1;
    int destination_peer = -1;
    std::vector<int> leaders;
    std::vector<int> local_ranks;
    std::vector<int> unique_leaders;
};

struct exchange_timing
{
    double receive_post_ms = 0.0;
    double send_post_ms = 0.0;
    double wait_ms = 0.0;
    std::uint64_t bytes = 0;
    std::size_t chunks = 0;
};

struct iteration_sample
{
    int iteration = 0;
    double forward_ms = 0.0;
    double receive_post_ms = 0.0;
    double send_post_ms = 0.0;
    double exchange_wait_ms = 0.0;
    double inverse_ms = 0.0;
    double total_ms = 0.0;
};

using clock_type = std::chrono::steady_clock;

double elapsed_ms( clock_type::time_point start, clock_type::time_point stop )
{
    return std::chrono::duration<double, std::milli>( stop - start ).count();
}

std::vector<std::string> split( const std::string &value, char delimiter )
{
    std::vector<std::string> result;
    std::stringstream        stream( value );
    std::string              item;
    while ( std::getline( stream, item, delimiter ) )
    {
        if ( !item.empty() )
            result.push_back( item );
    }
    return result;
}

std::size_t parse_positive_size( const std::string &value, const char *name )
{
    std::size_t consumed = 0;
    const auto parsed = std::stoull( value, &consumed );
    if ( consumed != value.size() || parsed == 0 )
        throw std::invalid_argument( std::string( name ) + " must be positive" );
    return static_cast<std::size_t>( parsed );
}

int parse_nonnegative_int( const std::string &value, const char *name )
{
    std::size_t consumed = 0;
    const long parsed = std::stol( value, &consumed );
    if ( consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max() )
        throw std::invalid_argument( std::string( name ) + " must be a nonnegative integer" );
    return static_cast<int>( parsed );
}

shape_3d parse_shape( const std::string &value )
{
    const auto items = split( value, ',' );
    if ( items.size() != 3 )
        throw std::invalid_argument( "--shape requires X,Y,Z" );
    return shape_3d{
        parse_positive_size( items[0], "shape X" ),
        parse_positive_size( items[1], "shape Y" ),
        parse_positive_size( items[2], "shape Z" )
    };
}

exchange_mode parse_exchange_mode( const std::string &value )
{
    if ( value == "none" )
        return exchange_mode::none;
    if ( value == "ring" )
        return exchange_mode::ring;
    throw std::invalid_argument( "--exchange must be none or ring" );
}

const char *exchange_mode_name( exchange_mode mode )
{
    return mode == exchange_mode::ring ? "ring" : "none";
}

void print_usage( const char *program )
{
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --size N                    Cubic transform size\n"
        << "  --shape X,Y,Z               Explicit transform dimensions\n"
        << "  --gpus-per-process N        Visible GPUs used by one cuFFT Xt plan; 0 means all\n"
        << "  --exchange none|ring        Exchange shuffled spectra between matching node ranks\n"
        << "  --chunk-mib N               MPI_BYTE chunk size, at most 2047 MiB\n"
        << "  --warmup N\n"
        << "  --iterations N\n"
        << "  --epsilon VALUE\n"
        << "  --output FILE\n"
        << "  --label TEXT\n"
        << "  --no-validate\n";
}

options parse_options( int argc, char **argv )
{
    options result;
    for ( int index = 1; index < argc; ++index )
    {
        const std::string argument = argv[index];
        auto require_value = [&]( const char *name ) -> std::string {
            if ( index + 1 >= argc )
                throw std::invalid_argument( std::string( "missing value after " ) + name );
            return argv[++index];
        };

        if ( argument == "--size" )
        {
            const std::size_t size = parse_positive_size( require_value( "--size" ), "size" );
            result.shape = shape_3d{ size, size, size };
        }
        else if ( argument == "--shape" )
        {
            result.shape = parse_shape( require_value( "--shape" ) );
        }
        else if ( argument == "--gpus-per-process" )
        {
            result.gpus_per_process =
                parse_nonnegative_int( require_value( "--gpus-per-process" ), "gpus per process" );
        }
        else if ( argument == "--exchange" )
        {
            result.exchange = parse_exchange_mode( require_value( "--exchange" ) );
        }
        else if ( argument == "--chunk-mib" )
        {
            result.chunk_mib = parse_positive_size( require_value( "--chunk-mib" ), "chunk MiB" );
        }
        else if ( argument == "--warmup" )
        {
            result.warmup = parse_nonnegative_int( require_value( "--warmup" ), "warmup" );
        }
        else if ( argument == "--iterations" )
        {
            result.iterations = parse_nonnegative_int( require_value( "--iterations" ), "iterations" );
        }
        else if ( argument == "--epsilon" )
        {
            result.epsilon = std::stod( require_value( "--epsilon" ) );
        }
        else if ( argument == "--output" )
        {
            result.output = require_value( "--output" );
        }
        else if ( argument == "--label" )
        {
            result.label = require_value( "--label" );
        }
        else if ( argument == "--no-validate" )
        {
            result.validate = false;
        }
        else if ( argument == "--help" || argument == "-h" )
        {
            print_usage( argv[0] );
            std::exit( 0 );
        }
        else
        {
            throw std::invalid_argument( "unknown option: " + argument );
        }
    }

    if ( result.iterations <= 0 )
        throw std::invalid_argument( "iterations must be positive" );
    if ( result.chunk_mib == 0 || result.chunk_mib > 2047 )
        throw std::invalid_argument( "chunk MiB must be in [1, 2047]" );
    if ( !( result.epsilon > 0.0 ) )
        throw std::invalid_argument( "epsilon must be positive" );
    return result;
}

topology build_topology( const comm_t &comm )
{
    topology result;
    auto     local_comm = comm.split_type( MPI_COMM_TYPE_SHARED );
    const auto local = local_comm.info();
    result.local_rank = local.myid;
    result.local_size = local.num_procs;
    const int leader = local.all_reduce_min( comm.myid );
    local_comm.free();

    result.leaders.resize( static_cast<std::size_t>( comm.num_procs ), -1 );
    result.local_ranks.resize( static_cast<std::size_t>( comm.num_procs ), -1 );
    comm.all_gather( &leader, 1, result.leaders.data(), 1 );
    comm.all_gather( &result.local_rank, 1, result.local_ranks.data(), 1 );

    std::set<int> unique( result.leaders.begin(), result.leaders.end() );
    result.unique_leaders.assign( unique.begin(), unique.end() );
    result.node_count = static_cast<int>( result.unique_leaders.size() );
    const auto leader_position =
        std::find( result.unique_leaders.begin(), result.unique_leaders.end(), leader );
    if ( leader_position == result.unique_leaders.end() )
        throw std::logic_error( "failed to locate node leader" );
    result.node_index =
        static_cast<int>( std::distance( result.unique_leaders.begin(), leader_position ) );

    const int minimum_local_size = comm.all_reduce_min( result.local_size );
    const int maximum_local_size = comm.all_reduce_max( result.local_size );
    if ( minimum_local_size != maximum_local_size )
        throw std::runtime_error( "node-local MPI process counts differ" );

    auto rank_for = [&]( int node_index, int local_rank ) {
        const int target_leader = result.unique_leaders[static_cast<std::size_t>( node_index )];
        for ( int rank = 0; rank < comm.num_procs; ++rank )
        {
            if ( result.leaders[static_cast<std::size_t>( rank )] == target_leader &&
                 result.local_ranks[static_cast<std::size_t>( rank )] == local_rank )
            {
                return rank;
            }
        }
        throw std::runtime_error( "missing matching node-local peer rank" );
    };

    if ( result.node_count > 1 )
    {
        const int previous = ( result.node_index + result.node_count - 1 ) % result.node_count;
        const int next = ( result.node_index + 1 ) % result.node_count;
        result.source_peer = rank_for( previous, result.local_rank );
        result.destination_peer = rank_for( next, result.local_rank );
    }
    return result;
}

int rank_after_ring_rounds( const topology &topo, int rounds )
{
    if ( topo.node_count <= 1 )
        throw std::logic_error( "ring source requested for a single-node topology" );
    const int shift = rounds % topo.node_count;
    const int source_node = ( topo.node_index + topo.node_count - shift ) % topo.node_count;
    const int source_leader = topo.unique_leaders[static_cast<std::size_t>( source_node )];
    for ( std::size_t rank = 0; rank < topo.leaders.size(); ++rank )
    {
        if ( topo.leaders[rank] == source_leader && topo.local_ranks[rank] == topo.local_rank )
            return static_cast<int>( rank );
    }
    throw std::runtime_error( "failed to resolve final ring source rank" );
}

exchange_timing exchange_spectrum(
    const comm_t &comm, const topology &topo, node_fft_backend &backend, std::size_t chunk_bytes
)
{
    const auto send_segments = backend.send_segments();
    const auto receive_segments = backend.receive_segments();
    if ( send_segments.size() != receive_segments.size() )
        throw std::runtime_error( "send/receive descriptor segment-count mismatch" );

    std::vector<request_t> receive_requests;
    std::vector<request_t> send_requests;
    std::vector<std::size_t> chunks_per_segment( send_segments.size(), 0 );
    std::size_t total_chunks = 0;
    std::uint64_t total_bytes = 0;
    for ( std::size_t segment = 0; segment < send_segments.size(); ++segment )
    {
        if ( send_segments[segment].bytes != receive_segments[segment].bytes )
            throw std::runtime_error( "send/receive descriptor segment-size mismatch" );
        const std::size_t chunks =
            ( send_segments[segment].bytes + chunk_bytes - 1 ) / chunk_bytes;
        chunks_per_segment[segment] = chunks;
        total_chunks += chunks;
        total_bytes += static_cast<std::uint64_t>( send_segments[segment].bytes );
    }
    if ( total_chunks > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
        throw std::overflow_error( "too many MPI chunks" );

    receive_requests.resize( total_chunks );
    send_requests.resize( total_chunks );

    const auto receive_post_start = clock_type::now();
    std::size_t request_index = 0;
    int tag = 100;
    for ( std::size_t segment = 0; segment < receive_segments.size(); ++segment )
    {
        const auto &view = receive_segments[segment];
        backend.select_device( view.device_ordinal );
        for ( std::size_t chunk = 0; chunk < chunks_per_segment[segment]; ++chunk )
        {
            const std::size_t offset = chunk * chunk_bytes;
            const std::size_t bytes = std::min( chunk_bytes, view.bytes - offset );
            comm.irecv<char>(
                static_cast<char *>( view.data ) + offset, static_cast<int>( bytes ),
                topo.source_peer, tag++, receive_requests[request_index++]
            );
        }
    }
    const auto receive_post_stop = clock_type::now();

    const auto send_post_start = clock_type::now();
    request_index = 0;
    tag = 100;
    for ( std::size_t segment = 0; segment < send_segments.size(); ++segment )
    {
        const auto &view = send_segments[segment];
        backend.select_device( view.device_ordinal );
        for ( std::size_t chunk = 0; chunk < chunks_per_segment[segment]; ++chunk )
        {
            const std::size_t offset = chunk * chunk_bytes;
            const std::size_t bytes = std::min( chunk_bytes, view.bytes - offset );
            comm.isend<char>(
                static_cast<const char *>( view.data ) + offset, static_cast<int>( bytes ),
                topo.destination_peer, tag++, send_requests[request_index++]
            );
        }
    }
    const auto send_post_stop = clock_type::now();

    const auto wait_start = clock_type::now();
    if ( !receive_requests.empty() )
        comm.waitall( static_cast<int>( receive_requests.size() ), receive_requests.data() );
    if ( !send_requests.empty() )
        comm.waitall( static_cast<int>( send_requests.size() ), send_requests.data() );
    const auto wait_stop = clock_type::now();

    backend.accept_received_spectrum();
    return exchange_timing{
        elapsed_ms( receive_post_start, receive_post_stop ),
        elapsed_ms( send_post_start, send_post_stop ),
        elapsed_ms( wait_start, wait_stop ),
        total_bytes,
        total_chunks
    };
}

double mean( const std::vector<double> &values )
{
    return std::accumulate( values.begin(), values.end(), 0.0 ) /
           static_cast<double>( values.size() );
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

void write_results(
    const options &opts, const topology &topo, const node_fft_backend &backend,
    const std::vector<iteration_sample> &samples, const exchange_timing &last_exchange,
    const fftm::experiments::node_hybrid::validation_result &validation,
    double plan_ms, double initialize_ms
)
{
    std::vector<double> totals;
    totals.reserve( samples.size() );
    for ( const auto &sample : samples )
        totals.push_back( sample.total_ms );
    const double average = mean( totals );
    const double deviation = standard_deviation( totals, average );
    const double minimum = *std::min_element( totals.begin(), totals.end() );

    std::ofstream summary( opts.output.c_str(), std::ios::out | std::ios::trunc );
    if ( !summary )
        throw std::runtime_error( "failed to open summary output: " + opts.output );
    summary
        << "label,backend,exchange,nodes,ranks,ranks_per_node,gpus_per_process,"
           "nx,ny,nz,logical_real_bytes,allocated_data_bytes,forward_work_bytes,"
           "inverse_work_bytes,chunk_mib,chunks_per_rank,exchange_bytes_per_rank,"
           "warmup,iterations,plan_ms,initialize_ms,avg_pair_ms,min_pair_ms,"
           "stddev_pair_ms,relative_l2,max_abs,valid\n";
    summary << std::setprecision( 12 )
            << opts.label << ',' << backend.name() << ',' << exchange_mode_name( opts.exchange ) << ','
            << topo.node_count << ',' << topo.leaders.size() << ',' << topo.local_size << ','
            << backend.device_count() << ','
            << opts.shape.x << ',' << opts.shape.y << ',' << opts.shape.z << ','
            << backend.logical_real_bytes() << ',' << backend.allocated_data_bytes() << ','
            << backend.forward_work_bytes() << ',' << backend.inverse_work_bytes() << ','
            << opts.chunk_mib << ',' << last_exchange.chunks << ',' << last_exchange.bytes << ','
            << opts.warmup << ',' << opts.iterations << ',' << plan_ms << ',' << initialize_ms << ','
            << average << ',' << minimum << ',' << deviation << ','
            << validation.relative_l2 << ',' << validation.max_abs << ','
            << ( validation.relative_l2 <= opts.epsilon ? 1 : 0 ) << '\n';

    std::ofstream sample_output( samples_path( opts.output ).c_str(), std::ios::out | std::ios::trunc );
    if ( !sample_output )
        throw std::runtime_error( "failed to open sample output" );
    sample_output
        << "iteration,forward_ms,receive_post_ms,send_post_ms,exchange_wait_ms,"
           "inverse_ms,total_ms\n";
    sample_output << std::setprecision( 12 );
    for ( const auto &sample : samples )
    {
        sample_output << sample.iteration << ',' << sample.forward_ms << ','
                      << sample.receive_post_ms << ',' << sample.send_post_ms << ','
                      << sample.exchange_wait_ms << ',' << sample.inverse_ms << ','
                      << sample.total_ms << '\n';
    }
}

int run( int argc, char **argv, const comm_t &comm )
{
    const options  opts = parse_options( argc, argv );
    const topology topo = build_topology( comm );
    if ( opts.exchange == exchange_mode::ring && topo.node_count < 2 )
        throw std::invalid_argument( "ring exchange requires at least two nodes" );

    const int available_devices =
        fftm::experiments::node_hybrid::available_cuda_device_count();
    const int requested_devices =
        opts.gpus_per_process == 0 ? available_devices : opts.gpus_per_process;
    if ( requested_devices < 2 || requested_devices > available_devices )
        throw std::invalid_argument( "gpus per process must select between 2 and all visible GPUs" );
    std::vector<int> devices( static_cast<std::size_t>( requested_devices ) );
    std::iota( devices.begin(), devices.end(), 0 );

    if ( comm.myid == 0 )
    {
        std::cout << "NODE_HYBRID_SETUP label=" << opts.label
                  << " shape=" << opts.shape.x << 'x' << opts.shape.y << 'x' << opts.shape.z
                  << " nodes=" << topo.node_count
                  << " ranks_per_node=" << topo.local_size
                  << " gpus_per_process=" << requested_devices
                  << " exchange=" << exchange_mode_name( opts.exchange ) << std::endl;
    }

    const auto plan_start = clock_type::now();
    auto backend =
        fftm::experiments::node_hybrid::make_cuda_cufft_xt_backend( opts.shape, devices );
    backend->synchronize();
    const auto plan_stop = clock_type::now();
    const double plan_ms = comm.all_reduce_max( elapsed_ms( plan_start, plan_stop ) );

    const std::uint64_t initial_seed = static_cast<std::uint64_t>( comm.myid + 1 );
    if ( comm.myid == 0 )
        std::cout << "NODE_HYBRID_INITIALIZE logical_bytes="
                  << backend->logical_real_bytes() << std::endl;
    const auto initialize_start = clock_type::now();
    backend->initialize_input( initial_seed );
    const auto initialize_stop = clock_type::now();
    const double initialize_ms =
        comm.all_reduce_max( elapsed_ms( initialize_start, initialize_stop ) );

    const std::size_t chunk_bytes = opts.chunk_mib * 1024ULL * 1024ULL;
    std::vector<iteration_sample> samples;
    samples.reserve( static_cast<std::size_t>( opts.iterations ) );
    exchange_timing last_exchange;

    const int total_iterations = opts.warmup + opts.iterations;
    for ( int iteration = 0; iteration < total_iterations; ++iteration )
    {
        comm.barrier();
        backend->synchronize();
        const auto total_start = clock_type::now();

        const auto forward_start = clock_type::now();
        backend->forward();
        backend->synchronize();
        const auto forward_stop = clock_type::now();

        exchange_timing exchange;
        if ( opts.exchange == exchange_mode::ring )
            exchange = exchange_spectrum( comm, topo, *backend, chunk_bytes );

        const auto inverse_start = clock_type::now();
        backend->inverse();
        backend->synchronize();
        const auto inverse_stop = clock_type::now();
        const auto total_stop = inverse_stop;
        backend->normalize_inverse_output();
        backend->synchronize();

        std::array<double, 6> local_times{
            elapsed_ms( forward_start, forward_stop ),
            exchange.receive_post_ms,
            exchange.send_post_ms,
            exchange.wait_ms,
            elapsed_ms( inverse_start, inverse_stop ),
            elapsed_ms( total_start, total_stop )
        };
        std::array<double, 6> maximum_times{};
        comm.all_reduce_max( local_times.data(), maximum_times.data(), static_cast<int>( local_times.size() ) );

        if ( iteration >= opts.warmup )
        {
            samples.push_back( iteration_sample{
                iteration - opts.warmup,
                maximum_times[0],
                maximum_times[1],
                maximum_times[2],
                maximum_times[3],
                maximum_times[4],
                maximum_times[5]
            } );
            last_exchange = exchange;
        }

        if ( comm.myid == 0 )
        {
            std::cout << "NODE_HYBRID_ITERATION index=" << iteration
                      << " measured=" << ( iteration >= opts.warmup ? 1 : 0 )
                      << " total_ms=" << maximum_times[5] << std::endl;
        }
    }

    fftm::experiments::node_hybrid::validation_result validation;
    if ( opts.validate )
    {
        backend->initialize_input( initial_seed );
        comm.barrier();
        backend->forward();
        backend->synchronize();
        if ( opts.exchange == exchange_mode::ring )
            exchange_spectrum( comm, topo, *backend, chunk_bytes );
        backend->inverse();
        backend->synchronize();

        const int ring_rounds = opts.exchange == exchange_mode::ring ? 1 : 0;
        const int expected_rank =
            opts.exchange == exchange_mode::ring ? rank_after_ring_rounds( topo, ring_rounds ) : comm.myid;
        const std::uint64_t expected_seed = static_cast<std::uint64_t>( expected_rank + 1 );
        if ( comm.myid == 0 )
            std::cout << "NODE_HYBRID_VALIDATE expected_source_shift=" << ring_rounds << std::endl;
        const auto local_validation = backend->validate_input( expected_seed );
        validation.relative_l2 = comm.all_reduce_max( local_validation.relative_l2 );
        validation.max_abs = comm.all_reduce_max( local_validation.max_abs );
    }

    if ( comm.myid == 0 )
    {
        write_results(
            opts, topo, *backend, samples, last_exchange, validation, plan_ms, initialize_ms
        );
        std::cout << std::setprecision( 12 )
                  << "NODE_HYBRID_RESULT label=" << opts.label
                  << " relative_l2=" << validation.relative_l2
                  << " max_abs=" << validation.max_abs
                  << " valid=" << ( !opts.validate || validation.relative_l2 <= opts.epsilon ? 1 : 0 )
                  << " output=" << opts.output << std::endl;
    }

    return !opts.validate || validation.relative_l2 <= opts.epsilon ? 0 : 2;
}

}

int main( int argc, char **argv )
{
    for ( int index = 1; index < argc; ++index )
    {
        const std::string argument = argv[index];
        if ( argument == "--help" || argument == "-h" )
        {
            print_usage( argv[0] );
            return 0;
        }
    }

    try
    {
        scfd::communication::mpi_wrap mpi( argc, argv );
        try
        {
            return run( argc, argv, mpi.comm_world() );
        }
        catch ( const std::exception &error )
        {
            const auto comm = mpi.comm_world();
            std::cerr << "NODE_HYBRID_ERROR rank=" << comm.myid
                      << " message=" << error.what() << std::endl;
            MPI_Abort( comm.comm, 1 );
            return 1;
        }
    }
    catch ( const std::exception &error )
    {
        std::cerr << "NODE_HYBRID_ERROR rank=uninitialized message=" << error.what() << std::endl;
        return 1;
    }
}
