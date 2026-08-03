#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <mpi.h>
#include <thrust/complex.h>

#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>

#include <external_wrap/cufft_wrap.h>

#include "detail/mpi_cuda_test_init.h"

namespace
{

using T             = double;
using complex_t     = thrust::complex<T>;
using runtime_api_t = fftm::wrap::cuda_runtime_api;

struct partition_1d
{
    std::vector<std::size_t> size;
    std::vector<std::size_t> start;

    void init( std::size_t n, std::size_t p )
    {
        size.assign( p, n / p );
        for ( std::size_t i = 0; i < n % p; ++i )
            ++size[i];

        start.assign( p, 0 );
        for ( std::size_t i = 1; i < p; ++i )
            start[i] = start[i - 1] + size[i - 1];
    }
};

struct test_options
{
    std::size_t nx     = 128;
    std::size_t ny     = 128;
    std::size_t nz     = 128;
    std::size_t p1     = 0;
    std::size_t p2     = 0;
    int         warmup = 3;
    int         times  = 5;
    std::size_t chunk_mib = 1024;
    std::string method = "sync";
    std::string pattern = "first-forward";
    int         egger_opt = 0;
    std::string large_count_transport = "hindexed";
};

struct device_buffer
{
    void       *ptr   = NULL;
    std::size_t bytes = 0;

    ~device_buffer()
    {
        reset();
    }

    device_buffer() = default;
    device_buffer( const device_buffer & ) = delete;
    device_buffer &operator=( const device_buffer & ) = delete;

    void allocate( std::size_t nbytes )
    {
        reset();
        bytes = nbytes;
        runtime_api_t::memory_type::malloc( &ptr, bytes );
    }

    void reset()
    {
        if ( ptr != NULL )
        {
            runtime_api_t::memory_type::free( ptr );
            ptr = NULL;
            bytes = 0;
        }
    }

    complex_t *as_complex()
    {
        return static_cast<complex_t *>( ptr );
    }
};

std::pair<std::size_t, std::size_t> choose_default_grid( std::size_t num_procs )
{
    std::size_t p1 = 1;
    for ( std::size_t d = 1; d * d <= num_procs; ++d )
    {
        if ( num_procs % d == 0 )
            p1 = d;
    }
    return std::make_pair( p1, num_procs / p1 );
}

std::size_t parse_size( const char *s )
{
    return static_cast<std::size_t>( std::strtoull( s, NULL, 10 ) );
}

test_options parse_options( int argc, char *argv[], int num_procs )
{
    test_options opt;
    int          argi = 1;
    while ( argi < argc )
    {
        const std::string arg = argv[argi];
        if ( arg == "--grid" )
        {
            if ( argi + 2 >= argc )
                throw std::logic_error( "Missing values for --grid P1 P2" );
            opt.p1 = parse_size( argv[argi + 1] );
            opt.p2 = parse_size( argv[argi + 2] );
            argi += 3;
        }
        else if ( arg == "--method" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --method" );
            opt.method = argv[argi + 1];
            if ( opt.method != "sync" && opt.method != "streams" )
                throw std::logic_error( "Unknown --method '" + opt.method + "'" );
            argi += 2;
        }
        else if ( arg == "--pattern" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --pattern" );
            opt.pattern = argv[argi + 1];
            if ( opt.pattern != "first-forward" && opt.pattern != "first-backward" &&
                 opt.pattern != "second-forward" && opt.pattern != "second-backward" )
                throw std::logic_error( "Unknown --pattern '" + opt.pattern + "'" );
            argi += 2;
        }
        else if ( arg == "--egger-opt" || arg == "--opt" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --egger-opt" );
            opt.egger_opt = std::atoi( argv[argi + 1] );
            if ( opt.egger_opt != 0 && opt.egger_opt != 1 )
                throw std::logic_error( "--egger-opt must be 0 or 1" );
            argi += 2;
        }
        else if ( arg == "--large-count-transport" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --large-count-transport" );
            opt.large_count_transport = argv[argi + 1];
            if ( opt.large_count_transport != "hindexed" && opt.large_count_transport != "chunked" )
                throw std::logic_error(
                    "Unknown --large-count-transport '" + opt.large_count_transport + "'"
                );
            argi += 2;
        }
        else if ( arg == "--warmup" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --warmup" );
            opt.warmup = std::atoi( argv[argi + 1] );
            argi += 2;
        }
        else if ( arg == "--times" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --times" );
            opt.times = std::atoi( argv[argi + 1] );
            argi += 2;
        }
        else if ( arg == "--chunk-mib" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --chunk-mib" );
            opt.chunk_mib = parse_size( argv[argi + 1] );
            argi += 2;
        }
        else
        {
            break;
        }
    }

    if ( argc - argi == 3 )
    {
        opt.nx = parse_size( argv[argi] );
        opt.ny = parse_size( argv[argi + 1] );
        opt.nz = parse_size( argv[argi + 2] );
    }
    else if ( argc != argi )
    {
        throw std::logic_error(
            "USAGE: test_egger_pencil_comm_pattern_3D.bin [--method sync|streams] "
            "[--pattern first-forward|first-backward|second-forward|second-backward] [--grid P1 P2] "
            "[--egger-opt 0|1] [--large-count-transport hindexed|chunked] "
            "[--warmup N] [--times N] [--chunk-mib N] [Nx Ny Nz]"
        );
    }

    if ( opt.warmup < 0 || opt.times <= 0 )
        throw std::logic_error( "--warmup must be non-negative and --times must be positive" );
    if ( opt.chunk_mib == 0 )
        throw std::logic_error( "--chunk-mib must be positive" );

    if ( opt.p1 == 0 || opt.p2 == 0 )
    {
        const auto grid = choose_default_grid( static_cast<std::size_t>( num_procs ) );
        opt.p1          = grid.first;
        opt.p2          = grid.second;
    }

    if ( opt.p1 * opt.p2 != static_cast<std::size_t>( num_procs ) )
        throw std::logic_error( "--grid P1 P2 must match MPI process count" );

    return opt;
}

complex_t make_value( std::size_t gx, std::size_t gy, std::size_t gz, std::size_t nx, std::size_t ny )
{
    const T value = static_cast<T>( 1 + gx + nx * ( gy + ny * gz ) );
    return complex_t( value, -value );
}

int mpi_count( std::size_t bytes, const char *label )
{
    if ( bytes > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
        throw std::runtime_error( std::string( label ) + " exceeds MPI int count range" );
    return static_cast<int>( bytes );
}

std::size_t mpi_chunk_bytes( const test_options &opt )
{
    const std::size_t requested = opt.chunk_mib * std::size_t( 1024 ) * std::size_t( 1024 );
    const std::size_t int_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
    return std::max<std::size_t>( 1, std::min( requested, int_limit ) );
}

void check_mpi( int result, const char *operation )
{
    if ( result == MPI_SUCCESS )
        return;

    char error[MPI_MAX_ERROR_STRING] = {};
    int  length                      = 0;
    MPI_Error_string( result, error, &length );
    throw std::runtime_error( std::string( operation ) + " failed: " + std::string( error, error + length ) );
}

MPI_Datatype make_large_byte_datatype( std::size_t bytes )
{
    if ( bytes == 0 )
        return MPI_BYTE;

    const std::size_t max_block = std::size_t( 1 ) << 30;
    const std::size_t blocks    = ( bytes + max_block - 1 ) / max_block;
    if ( blocks > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
        throw std::runtime_error( "large-count hindexed datatype has too many blocks" );

    std::vector<int>       block_lengths( blocks );
    std::vector<MPI_Aint>  displacements( blocks );
    std::size_t            offset = 0;
    for ( std::size_t i = 0; i < blocks; ++i )
    {
        const std::size_t block = std::min( max_block, bytes - offset );
        block_lengths[i]       = static_cast<int>( block );
        displacements[i]       = static_cast<MPI_Aint>( offset );
        offset += block;
    }

    MPI_Datatype type = MPI_DATATYPE_NULL;
    check_mpi(
        MPI_Type_create_hindexed(
            static_cast<int>( blocks ), block_lengths.data(), displacements.data(), MPI_BYTE, &type
        ),
        "MPI_Type_create_hindexed"
    );
    check_mpi( MPI_Type_commit( &type ), "MPI_Type_commit" );
    return type;
}

void post_irecv_hindexed(
    void *dst, std::size_t bytes, int source, int tag, MPI_Comm comm, std::vector<MPI_Request> &requests
)
{
    requests.assign( 1, MPI_REQUEST_NULL );
    const std::size_t int_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
    if ( bytes <= int_limit )
    {
        check_mpi(
            MPI_Irecv( dst, mpi_count( bytes, "hindexed recv bytes" ), MPI_BYTE, source, tag, comm, &requests[0] ),
            "MPI_Irecv"
        );
        return;
    }

    MPI_Datatype large_type = make_large_byte_datatype( bytes );
    check_mpi( MPI_Irecv( dst, 1, large_type, source, tag, comm, &requests[0] ), "MPI_Irecv large hindexed" );
    check_mpi( MPI_Type_free( &large_type ), "MPI_Type_free" );
}

void post_isend_hindexed(
    const void *src, std::size_t bytes, int dest, int tag, MPI_Comm comm, std::vector<MPI_Request> &requests
)
{
    requests.assign( 1, MPI_REQUEST_NULL );
    const std::size_t int_limit = static_cast<std::size_t>( std::numeric_limits<int>::max() );
    if ( bytes <= int_limit )
    {
        check_mpi(
            MPI_Isend(
                const_cast<void *>( src ), mpi_count( bytes, "hindexed send bytes" ), MPI_BYTE, dest, tag, comm,
                &requests[0]
            ),
            "MPI_Isend"
        );
        return;
    }

    MPI_Datatype large_type = make_large_byte_datatype( bytes );
    check_mpi(
        MPI_Isend( const_cast<void *>( src ), 1, large_type, dest, tag, comm, &requests[0] ),
        "MPI_Isend large hindexed"
    );
    check_mpi( MPI_Type_free( &large_type ), "MPI_Type_free" );
}

void post_irecv_chunks(
    void *dst, std::size_t bytes, int source, int tag, MPI_Comm comm, std::size_t chunk_bytes,
    std::vector<MPI_Request> &requests
)
{
    requests.clear();
    char       *base   = static_cast<char *>( dst );
    std::size_t offset = 0;
    while ( offset < bytes )
    {
        const std::size_t nbytes = std::min( chunk_bytes, bytes - offset );
        requests.push_back( MPI_REQUEST_NULL );
        MPI_Irecv(
            base + offset, mpi_count( nbytes, "chunked recv bytes" ), MPI_BYTE, source, tag, comm, &requests.back()
        );
        offset += nbytes;
    }
}

void post_isend_chunks(
    const void *src, std::size_t bytes, int dest, int tag, MPI_Comm comm, std::size_t chunk_bytes,
    std::vector<MPI_Request> &requests
)
{
    requests.clear();
    const char *base   = static_cast<const char *>( src );
    std::size_t offset = 0;
    while ( offset < bytes )
    {
        const std::size_t nbytes = std::min( chunk_bytes, bytes - offset );
        requests.push_back( MPI_REQUEST_NULL );
        MPI_Isend(
            const_cast<char *>( base + offset ), mpi_count( nbytes, "chunked send bytes" ), MPI_BYTE, dest, tag, comm,
            &requests.back()
        );
        offset += nbytes;
    }
}

void post_irecv(
    const test_options &opt, void *dst, std::size_t bytes, int source, int tag, MPI_Comm comm,
    std::size_t chunk_bytes, std::vector<MPI_Request> &requests
)
{
    if ( opt.large_count_transport == "hindexed" )
        post_irecv_hindexed( dst, bytes, source, tag, comm, requests );
    else
        post_irecv_chunks( dst, bytes, source, tag, comm, chunk_bytes, requests );
}

void post_isend(
    const test_options &opt, const void *src, std::size_t bytes, int dest, int tag, MPI_Comm comm,
    std::size_t chunk_bytes, std::vector<MPI_Request> &requests
)
{
    if ( opt.large_count_transport == "hindexed" )
        post_isend_hindexed( src, bytes, dest, tag, comm, requests );
    else
        post_isend_chunks( src, bytes, dest, tag, comm, chunk_bytes, requests );
}

bool testall_group( std::vector<MPI_Request> &requests )
{
    if ( requests.empty() )
        return true;
    int flag = 0;
    MPI_Testall( static_cast<int>( requests.size() ), requests.data(), &flag, MPI_STATUSES_IGNORE );
    return flag != 0;
}

void waitall_group( std::vector<MPI_Request> &requests )
{
    if ( requests.empty() )
        return;
    MPI_Waitall( static_cast<int>( requests.size() ), requests.data(), MPI_STATUSES_IGNORE );
}

void waitall_groups( std::vector<std::vector<MPI_Request>> &groups )
{
    for ( auto &group : groups )
        waitall_group( group );
}

struct callback_state
{
    std::mutex              mutex;
    std::condition_variable cv;
    std::vector<int>        ready;

    void reset()
    {
        std::lock_guard<std::mutex> lock( mutex );
        ready.clear();
    }
};

struct callback_payload
{
    callback_state *state = NULL;
    int             peer  = 0;
};

void CUDART_CB send_ready_callback( void *data )
{
    callback_payload *payload = static_cast<callback_payload *>( data );
    {
        std::lock_guard<std::mutex> lock( payload->state->mutex );
        payload->state->ready.push_back( payload->peer );
    }
    payload->state->cv.notify_one();
}

struct send_thread_params
{
    callback_state       *state;
    const test_options   *options;
    complex_t            *send_ptr;
    std::vector<int>      comm_order;
    std::vector<std::size_t> send_offset;
    std::vector<std::size_t> send_bytes;
    int                   local_tag;
    MPI_Comm              comm;
    std::size_t           chunk_bytes;
    std::vector<std::vector<MPI_Request>> *send_req;
    int                   device;
};

void send_thread_func( send_thread_params params )
{
    runtime_api_t::set_device( params.device );

    for ( std::size_t i = 0; i < params.comm_order.size(); ++i )
    {
        int peer_j = -1;
        {
            std::unique_lock<std::mutex> lock( params.state->mutex );
            params.state->cv.wait( lock, [&params] { return !params.state->ready.empty(); } );
            peer_j = params.state->ready.back();
            params.state->ready.pop_back();
        }

        post_isend(
            *params.options,
            &params.send_ptr[params.send_offset[peer_j]], params.send_bytes[peer_j], peer_j, params.local_tag,
            params.comm, params.chunk_bytes, ( *params.send_req )[peer_j]
        );
    }
}

struct dims3
{
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t z = 0;
};

std::size_t index3( std::size_t x, std::size_t y, std::size_t z, const dims3 &dims )
{
    return ( x * dims.y + y ) * dims.z + z;
}

void copy_block_xyz(
    const complex_t *src, complex_t *dst, const dims3 &src_dims, const dims3 &dst_dims,
    std::size_t src_x, std::size_t src_y, std::size_t src_z,
    std::size_t dst_x, std::size_t dst_y, std::size_t dst_z,
    std::size_t count_x, std::size_t count_y, std::size_t count_z,
    typename runtime_api_t::stream_t stream
)
{
    typename runtime_api_t::memcpy_3d_params_t params = {};
    params.source_position = runtime_api_t::make_pos( src_z * sizeof( complex_t ), src_y, src_x );
    params.source = runtime_api_t::make_pitched_ptr(
        const_cast<complex_t *>( src ), src_dims.z * sizeof( complex_t ), src_dims.z, src_dims.y
    );
    params.destination_position = runtime_api_t::make_pos( dst_z * sizeof( complex_t ), dst_y, dst_x );
    params.destination = runtime_api_t::make_pitched_ptr( dst, dst_dims.z * sizeof( complex_t ), dst_dims.z, dst_dims.y );
    params.extent = runtime_api_t::make_extent( count_z * sizeof( complex_t ), count_y, count_x );
    params.kind   = runtime_api_t::device_to_device_kind();
    runtime_api_t::memcpy_3d_async( &params, stream );
}

void copy_pitched_xyz(
    const complex_t *src, complex_t *dst,
    std::size_t src_pitch_elems, std::size_t src_height_elems,
    std::size_t dst_pitch_elems, std::size_t dst_height_elems,
    std::size_t src_x, std::size_t src_y, std::size_t src_z,
    std::size_t dst_x, std::size_t dst_y, std::size_t dst_z,
    std::size_t count_x, std::size_t count_y, std::size_t count_z,
    typename runtime_api_t::stream_t stream
)
{
    typename runtime_api_t::memcpy_3d_params_t params = {};
    params.source_position = runtime_api_t::make_pos( src_x * sizeof( complex_t ), src_y, src_z );
    params.source = runtime_api_t::make_pitched_ptr(
        const_cast<complex_t *>( src ), src_pitch_elems * sizeof( complex_t ), src_pitch_elems, src_height_elems
    );
    params.destination_position = runtime_api_t::make_pos( dst_x * sizeof( complex_t ), dst_y, dst_z );
    params.destination = runtime_api_t::make_pitched_ptr(
        dst, dst_pitch_elems * sizeof( complex_t ), dst_pitch_elems, dst_height_elems
    );
    params.extent = runtime_api_t::make_extent( count_x * sizeof( complex_t ), count_y, count_z );
    params.kind   = runtime_api_t::device_to_device_kind();
    runtime_api_t::memcpy_3d_async( &params, stream );
}

enum class comm_pattern
{
    first_forward,
    first_backward,
    second_forward,
    second_backward
};

comm_pattern parse_pattern_name( const std::string &name )
{
    if ( name == "first-forward" )
        return comm_pattern::first_forward;
    if ( name == "first-backward" )
        return comm_pattern::first_backward;
    if ( name == "second-forward" )
        return comm_pattern::second_forward;
    if ( name == "second-backward" )
        return comm_pattern::second_backward;
    throw std::logic_error( "Unknown communication pattern '" + name + "'" );
}

bool pattern_has_async_pack( comm_pattern pattern, int egger_opt )
{
    if ( egger_opt == 1 )
        return pattern == comm_pattern::first_backward || pattern == comm_pattern::second_backward;
    return pattern != comm_pattern::second_backward;
}

bool egger_waitall_receive_pattern( comm_pattern pattern, int egger_opt )
{
    if ( egger_opt == 1 )
        return pattern == comm_pattern::first_backward || pattern == comm_pattern::second_backward;
    return pattern == comm_pattern::second_forward;
}

std::size_t egger_unpack_stream( comm_pattern pattern, int local_rank, std::size_t comm_size, int peer )
{
    if ( pattern == comm_pattern::first_forward || pattern == comm_pattern::first_backward ||
         pattern == comm_pattern::second_forward )
    {
        return static_cast<std::size_t>(
            ( local_rank + static_cast<int>( comm_size ) - peer ) % static_cast<int>( comm_size )
        );
    }
    return static_cast<std::size_t>( peer );
}

double mean( const std::vector<double> &v )
{
    return std::accumulate( v.begin(), v.end(), 0.0 ) / static_cast<double>( v.size() );
}

double stddev( const std::vector<double> &v, double m )
{
    double acc = 0.0;
    for ( double x : v )
        acc += ( x - m ) * ( x - m );
    return std::sqrt( acc / static_cast<double>( v.size() ) );
}

int run_test( scfd::utils::log_mpi &log, const scfd::communication::mpi_comm_info &comm_info, const test_options &opt )
{
    const comm_pattern pattern = parse_pattern_name( opt.pattern );
    const int          pidx    = comm_info.myid;
    const int          pidx_i  = pidx / static_cast<int>( opt.p2 );
    const int          pidx_j  = pidx % static_cast<int>( opt.p2 );

    partition_1d part_x, part_y_input, part_y_output, part_z;
    const std::size_t nz_complex = opt.nz / 2 + 1;
    part_x.init( opt.nx, opt.p1 );
    part_y_input.init( opt.ny, opt.p2 );
    part_y_output.init( opt.ny, opt.p1 );
    part_z.init( nz_complex, opt.p2 );

    MPI_Comm comm1 = MPI_COMM_NULL;
    MPI_Comm comm2 = MPI_COMM_NULL;
    MPI_Comm_split( MPI_COMM_WORLD, pidx_i, pidx_j, &comm1 );
    MPI_Comm_split( MPI_COMM_WORLD, pidx_j, pidx_i, &comm2 );

    const bool        first_comm = pattern == comm_pattern::first_forward || pattern == comm_pattern::first_backward;
    MPI_Comm          active_comm = first_comm ? comm1 : comm2;
    const std::size_t comm_size   = first_comm ? opt.p2 : opt.p1;
    const int         local_rank  = first_comm ? pidx_j : pidx_i;

    std::vector<int> comm_order;
    for ( std::size_t p = 1; p < comm_size; ++p )
        comm_order.push_back( static_cast<int>( ( local_rank + static_cast<int>( p ) ) % static_cast<int>( comm_size ) ) );

    const std::size_t x_size       = part_x.size[pidx_i];
    const std::size_t y_input_size = part_y_input.size[pidx_j];
    const std::size_t y_output_size = part_y_output.size[pidx_i];
    const std::size_t z_size       = part_z.size[pidx_j];

    dims3 input_dims;
    dims3 output_dims;
    switch ( pattern )
    {
        case comm_pattern::first_forward:
            input_dims  = { x_size, y_input_size, nz_complex };
            output_dims = { x_size, opt.ny, z_size };
            break;
        case comm_pattern::first_backward:
            input_dims  = { x_size, opt.ny, z_size };
            output_dims = { x_size, y_input_size, nz_complex };
            break;
        case comm_pattern::second_forward:
            input_dims  = { x_size, opt.ny, z_size };
            output_dims = { opt.nx, y_output_size, z_size };
            break;
        case comm_pattern::second_backward:
            input_dims  = { opt.nx, y_output_size, z_size };
            output_dims = { x_size, opt.ny, z_size };
            break;
    }

    const std::size_t input_elems  = input_dims.x * input_dims.y * input_dims.z;
    const std::size_t output_elems = output_dims.x * output_dims.y * output_dims.z;
    const std::size_t work_elems   = std::max( input_elems, output_elems );

    std::vector<complex_t> h_input( input_elems );
    for ( std::size_t ix = 0; ix < input_dims.x; ++ix )
    {
        for ( std::size_t iy = 0; iy < input_dims.y; ++iy )
        {
            for ( std::size_t iz = 0; iz < input_dims.z; ++iz )
            {
                std::size_t gx = 0, gy = 0, gz = 0;
                switch ( pattern )
                {
                    case comm_pattern::first_forward:
                        gx = part_x.start[pidx_i] + ix;
                        gy = part_y_input.start[pidx_j] + iy;
                        gz = iz;
                        break;
                    case comm_pattern::first_backward:
                    case comm_pattern::second_forward:
                        gx = part_x.start[pidx_i] + ix;
                        gy = iy;
                        gz = part_z.start[pidx_j] + iz;
                        break;
                    case comm_pattern::second_backward:
                        gx = ix;
                        gy = part_y_output.start[pidx_i] + iy;
                        gz = part_z.start[pidx_j] + iz;
                        break;
                }
                h_input[index3( ix, iy, iz, input_dims )] = make_value( gx, gy, gz, opt.nx, opt.ny );
            }
        }
    }

    device_buffer input_d, output_d, send_d, recv_d;
    input_d.allocate( input_elems * sizeof( complex_t ) );
    output_d.allocate( output_elems * sizeof( complex_t ) );
    send_d.allocate( work_elems * sizeof( complex_t ) );
    recv_d.allocate( work_elems * sizeof( complex_t ) );

    runtime_api_t::memcpy_async(
        input_d.as_complex(), h_input.data(), input_elems * sizeof( complex_t ), runtime_api_t::host_to_device_kind(), 0
    );
    runtime_api_t::device_synchronize();

    const std::size_t stream_count = std::max( opt.p1, opt.p2 );
    std::vector<typename runtime_api_t::stream_wrap> streams( stream_count );
    for ( auto &stream : streams )
        stream.init();

    std::vector<callback_payload> callbacks( stream_count );
    callback_state                cb_state;
    for ( std::size_t p = 0; p < stream_count; ++p )
    {
        callbacks[p].state = &cb_state;
        callbacks[p].peer  = static_cast<int>( p );
    }

    std::vector<double> wall_ms;
    wall_ms.reserve( static_cast<std::size_t>( opt.times ) );

    const int         device      = runtime_api_t::get_device();
    const std::size_t chunk_bytes = mpi_chunk_bytes( opt );
    const bool        use_stream_thread =
        opt.method == "streams" && pattern_has_async_pack( pattern, opt.egger_opt );

    std::size_t total_send_bytes = 0;
    std::size_t total_recv_bytes = 0;

    for ( int iter = 0; iter < opt.warmup + opt.times; ++iter )
    {
        std::vector<std::vector<MPI_Request>> send_req( comm_size );
        std::vector<std::vector<MPI_Request>> recv_req( comm_size );
        std::vector<std::size_t>              send_offset( comm_size, 0 );
        std::vector<std::size_t>              send_bytes( comm_size, 0 );
        std::vector<std::size_t>              recv_offset( comm_size, 0 );
        std::vector<std::size_t>              recv_bytes( comm_size, 0 );
        cb_state.reset();

        MPI_Barrier( MPI_COMM_WORLD );
        const double t0 = MPI_Wtime();

        for ( int peer : comm_order )
        {
            bool send_from_input = false;
            switch ( pattern )
            {
                case comm_pattern::first_forward:
                {
                    recv_offset[peer] = x_size * part_y_input.start[peer] * z_size;
                    recv_bytes[peer]  = sizeof( complex_t ) * x_size * part_y_input.size[peer] * z_size;
                    post_irecv(
                        opt, &recv_d.as_complex()[recv_offset[peer]], recv_bytes[peer], peer, peer, active_comm,
                        chunk_bytes,
                        recv_req[peer]
                    );

                    if ( opt.egger_opt == 1 )
                    {
                        send_from_input  = true;
                        send_offset[peer] = part_z.start[peer] * y_input_size * x_size;
                        send_bytes[peer]  = sizeof( complex_t ) * part_z.size[peer] * y_input_size * x_size;
                    }
                    else
                    {
                        send_offset[peer] = x_size * y_input_size * part_z.start[peer];
                        send_bytes[peer]  = sizeof( complex_t ) * x_size * y_input_size * part_z.size[peer];
                        const dims3 compact = { x_size, y_input_size, part_z.size[peer] };
                        copy_block_xyz(
                            input_d.as_complex(), &send_d.as_complex()[send_offset[peer]], input_dims, compact,
                            0, 0, part_z.start[peer], 0, 0, 0,
                            x_size, y_input_size, part_z.size[peer], streams[peer].stream()
                        );
                    }
                    break;
                }
                case comm_pattern::first_backward:
                {
                    recv_offset[peer] = opt.egger_opt == 1
                                            ? part_z.start[peer] * y_input_size * x_size
                                            : x_size * y_input_size * part_z.start[peer];
                    recv_bytes[peer]  = sizeof( complex_t ) * x_size * y_input_size * part_z.size[peer];
                    post_irecv(
                        opt,
                        &( opt.egger_opt == 1 ? output_d.as_complex() : recv_d.as_complex() )[recv_offset[peer]],
                        recv_bytes[peer], peer, peer, active_comm, chunk_bytes,
                        recv_req[peer]
                    );

                    send_offset[peer] = x_size * part_y_input.start[peer] * z_size;
                    send_bytes[peer]  = sizeof( complex_t ) * x_size * part_y_input.size[peer] * z_size;
                    if ( opt.egger_opt == 1 )
                    {
                        copy_pitched_xyz(
                            input_d.as_complex(), &send_d.as_complex()[send_offset[peer]],
                            opt.ny, x_size, part_y_input.size[peer], x_size,
                            part_y_input.start[peer], 0, 0, 0, 0, 0,
                            part_y_input.size[peer], x_size, z_size, streams[peer].stream()
                        );
                    }
                    else
                    {
                        const dims3 compact = { x_size, part_y_input.size[peer], z_size };
                        copy_block_xyz(
                            input_d.as_complex(), &send_d.as_complex()[send_offset[peer]], input_dims, compact,
                            0, part_y_input.start[peer], 0, 0, 0, 0,
                            x_size, part_y_input.size[peer], z_size, streams[peer].stream()
                        );
                    }
                    break;
                }
                case comm_pattern::second_forward:
                {
                    recv_offset[peer] = index3( part_x.start[peer], 0, 0, output_dims );
                    recv_bytes[peer]  = sizeof( complex_t ) * part_x.size[peer] * y_output_size * z_size;
                    if ( opt.egger_opt == 1 )
                        recv_offset[peer] = part_x.start[peer] * z_size * y_output_size;
                    post_irecv(
                        opt,
                        &( opt.egger_opt == 1 ? recv_d.as_complex() : output_d.as_complex() )[recv_offset[peer]],
                        recv_bytes[peer], peer, peer, active_comm,
                        chunk_bytes, recv_req[peer]
                    );

                    if ( opt.egger_opt == 1 )
                    {
                        send_from_input  = true;
                        send_offset[peer] = x_size * z_size * part_y_output.start[peer];
                        send_bytes[peer]  = sizeof( complex_t ) * x_size * z_size * part_y_output.size[peer];
                    }
                    else
                    {
                        send_offset[peer] = x_size * part_y_output.start[peer] * z_size;
                        send_bytes[peer]  = sizeof( complex_t ) * x_size * part_y_output.size[peer] * z_size;
                        const dims3 compact = { x_size, part_y_output.size[peer], z_size };
                        copy_block_xyz(
                            input_d.as_complex(), &send_d.as_complex()[send_offset[peer]], input_dims, compact,
                            0, part_y_output.start[peer], 0, 0, 0, 0,
                            x_size, part_y_output.size[peer], z_size, streams[peer].stream()
                        );
                    }
                    break;
                }
                case comm_pattern::second_backward:
                {
                    recv_offset[peer] = x_size * part_y_output.start[peer] * z_size;
                    recv_bytes[peer]  = sizeof( complex_t ) * x_size * part_y_output.size[peer] * z_size;
                    post_irecv(
                        opt,
                        &( opt.egger_opt == 1 ? output_d.as_complex() : recv_d.as_complex() )[recv_offset[peer]],
                        recv_bytes[peer], peer, peer, active_comm, chunk_bytes,
                        recv_req[peer]
                    );

                    send_offset[peer] = opt.egger_opt == 1
                                            ? part_x.start[peer] * z_size * y_output_size
                                            : index3( part_x.start[peer], 0, 0, input_dims );
                    send_bytes[peer] = sizeof( complex_t ) * part_x.size[peer] * y_output_size * z_size;
                    if ( opt.egger_opt == 1 )
                    {
                        copy_pitched_xyz(
                            input_d.as_complex(), &send_d.as_complex()[send_offset[peer]],
                            opt.nx, z_size, part_x.size[peer], z_size,
                            part_x.start[peer], 0, 0, 0, 0, 0,
                            part_x.size[peer], z_size, y_output_size, streams[peer].stream()
                        );
                    }
                    else
                    {
                        send_from_input = true;
                    }
                    break;
                }
            }

            if ( use_stream_thread )
            {
                runtime_api_t::launch_host_func(
                    streams[peer].stream(), &send_ready_callback, static_cast<void *>( &callbacks[peer] )
                );
            }
            else
            {
                if ( pattern_has_async_pack( pattern, opt.egger_opt ) )
                    runtime_api_t::device_synchronize();
                post_isend(
                    opt,
                    &( send_from_input ? input_d.as_complex() : send_d.as_complex() )[send_offset[peer]],
                    send_bytes[peer], peer, local_rank, active_comm,
                    chunk_bytes, send_req[peer]
                );
            }
        }

        std::thread send_thread;
        if ( use_stream_thread )
        {
            send_thread_params params;
            params.state       = &cb_state;
            params.options     = &opt;
            params.send_ptr    = send_d.as_complex();
            params.comm_order  = comm_order;
            params.send_offset = send_offset;
            params.send_bytes  = send_bytes;
            params.local_tag   = local_rank;
            params.comm        = active_comm;
            params.chunk_bytes = chunk_bytes;
            params.send_req    = &send_req;
            params.device      = device;
            send_thread        = std::thread( &send_thread_func, params );
        }

        switch ( pattern )
        {
            case comm_pattern::first_forward:
                if ( opt.egger_opt == 1 )
                    copy_pitched_xyz(
                        input_d.as_complex(), output_d.as_complex(),
                        y_input_size, x_size, opt.ny, x_size,
                        0, 0, part_z.start[pidx_j], part_y_input.start[pidx_j], 0, 0,
                        y_input_size, x_size, z_size, streams[local_rank].stream()
                    );
                else
                    copy_block_xyz(
                        input_d.as_complex(), output_d.as_complex(), input_dims, output_dims,
                        0, 0, part_z.start[pidx_j], 0, part_y_input.start[pidx_j], 0,
                        x_size, y_input_size, z_size, streams[local_rank].stream()
                    );
                break;
            case comm_pattern::first_backward:
                if ( opt.egger_opt == 1 )
                    copy_pitched_xyz(
                        input_d.as_complex(), output_d.as_complex(),
                        opt.ny, x_size, y_input_size, x_size,
                        part_y_input.start[pidx_j], 0, 0, 0, 0, part_z.start[pidx_j],
                        y_input_size, x_size, z_size, streams[local_rank].stream()
                    );
                else
                    copy_block_xyz(
                        input_d.as_complex(), output_d.as_complex(), input_dims, output_dims,
                        0, part_y_input.start[pidx_j], 0, 0, 0, part_z.start[pidx_j],
                        x_size, y_input_size, z_size, streams[local_rank].stream()
                    );
                break;
            case comm_pattern::second_forward:
                if ( opt.egger_opt == 1 )
                    copy_pitched_xyz(
                        input_d.as_complex(), output_d.as_complex(),
                        x_size, z_size, opt.nx, z_size,
                        0, 0, part_y_output.start[pidx_i], part_x.start[pidx_i], 0, 0,
                        x_size, z_size, y_output_size, streams[local_rank].stream()
                    );
                else
                    copy_block_xyz(
                        input_d.as_complex(), output_d.as_complex(), input_dims, output_dims,
                        0, part_y_output.start[pidx_i], 0, part_x.start[pidx_i], 0, 0,
                        x_size, y_output_size, z_size, streams[local_rank].stream()
                    );
                break;
            case comm_pattern::second_backward:
                if ( opt.egger_opt == 1 )
                    copy_pitched_xyz(
                        input_d.as_complex(), output_d.as_complex(),
                        opt.nx, z_size, x_size, z_size,
                        part_x.start[pidx_i], 0, 0, 0, 0, part_y_output.start[pidx_i],
                        x_size, z_size, y_output_size, streams[local_rank].stream()
                    );
                else
                    copy_block_xyz(
                        input_d.as_complex(), output_d.as_complex(), input_dims, output_dims,
                        part_x.start[pidx_i], 0, 0, 0, part_y_output.start[pidx_i], 0,
                        x_size, y_output_size, z_size, streams[local_rank].stream()
                    );
                break;
        }

        auto unpack_peer = [&]( int peer ) {
            const std::size_t stream_idx = egger_unpack_stream( pattern, local_rank, comm_size, peer );
            switch ( pattern )
            {
                case comm_pattern::first_forward:
                {
                    if ( opt.egger_opt == 1 )
                    {
                        copy_pitched_xyz(
                            &recv_d.as_complex()[recv_offset[peer]], output_d.as_complex(),
                            part_y_input.size[peer], x_size, opt.ny, x_size,
                            0, 0, 0, part_y_input.start[peer], 0, 0,
                            part_y_input.size[peer], x_size, z_size, streams[stream_idx].stream()
                        );
                    }
                    else
                    {
                        const dims3 compact = { x_size, part_y_input.size[peer], z_size };
                        copy_block_xyz(
                            &recv_d.as_complex()[recv_offset[peer]], output_d.as_complex(), compact, output_dims,
                            0, 0, 0, 0, part_y_input.start[peer], 0,
                            x_size, part_y_input.size[peer], z_size, streams[stream_idx].stream()
                        );
                    }
                    break;
                }
                case comm_pattern::first_backward:
                {
                    if ( opt.egger_opt == 0 )
                    {
                        const dims3 compact = { x_size, y_input_size, part_z.size[peer] };
                        copy_block_xyz(
                            &recv_d.as_complex()[recv_offset[peer]], output_d.as_complex(), compact, output_dims,
                            0, 0, 0, 0, 0, part_z.start[peer],
                            x_size, y_input_size, part_z.size[peer], streams[stream_idx].stream()
                        );
                    }
                    break;
                }
                case comm_pattern::second_forward:
                {
                    if ( opt.egger_opt == 1 )
                    {
                        copy_pitched_xyz(
                            &recv_d.as_complex()[recv_offset[peer]], output_d.as_complex(),
                            part_x.size[peer], z_size, opt.nx, z_size,
                            0, 0, 0, part_x.start[peer], 0, 0,
                            part_x.size[peer], z_size, y_output_size, streams[stream_idx].stream()
                        );
                    }
                    break;
                }
                case comm_pattern::second_backward:
                {
                    if ( opt.egger_opt == 0 )
                    {
                        const dims3 compact = { x_size, part_y_output.size[peer], z_size };
                        copy_block_xyz(
                            &recv_d.as_complex()[recv_offset[peer]], output_d.as_complex(), compact, output_dims,
                            0, 0, 0, 0, part_y_output.start[peer], 0,
                            x_size, part_y_output.size[peer], z_size, streams[stream_idx].stream()
                        );
                    }
                    break;
                }
            }
        };

        if ( egger_waitall_receive_pattern( pattern, opt.egger_opt ) )
        {
            waitall_groups( recv_req );
        }
        else if ( opt.large_count_transport == "hindexed" )
        {
            std::vector<MPI_Request> waitany_requests( comm_size, MPI_REQUEST_NULL );
            for ( int peer : comm_order )
                waitany_requests[peer] = recv_req[peer].empty() ? MPI_REQUEST_NULL : recv_req[peer][0];

            std::size_t remaining_recvs = comm_order.size();
            while ( remaining_recvs > 0 )
            {
                int peer = MPI_UNDEFINED;
                check_mpi(
                    MPI_Waitany(
                        static_cast<int>( waitany_requests.size() ), waitany_requests.data(), &peer, MPI_STATUSES_IGNORE
                    ),
                    "MPI_Waitany"
                );
                if ( peer == MPI_UNDEFINED )
                    break;
                if ( !recv_req[peer].empty() )
                    recv_req[peer][0] = MPI_REQUEST_NULL;
                --remaining_recvs;
                unpack_peer( peer );
            }
        }
        else
        {
            std::size_t       remaining_recvs = comm_order.size();
            std::vector<char> recv_done( comm_size, 0 );
            while ( remaining_recvs > 0 )
            {
                for ( int peer : comm_order )
                {
                    if ( recv_done[peer] )
                        continue;
                    if ( !testall_group( recv_req[peer] ) )
                        continue;

                    recv_done[peer] = 1;
                    --remaining_recvs;
                    unpack_peer( peer );
                }
            }
        }

        runtime_api_t::device_synchronize();
        if ( use_stream_thread )
            send_thread.join();
        waitall_groups( send_req );

        MPI_Barrier( MPI_COMM_WORLD );
        const double t1 = MPI_Wtime();
        if ( iter >= opt.warmup )
            wall_ms.push_back( ( t1 - t0 ) * 1000.0 );

        if ( iter == 0 )
        {
            total_send_bytes = 0;
            total_recv_bytes = 0;
            for ( int peer : comm_order )
            {
                total_send_bytes += send_bytes[peer];
                total_recv_bytes += recv_bytes[peer];
            }
        }
    }

    std::vector<complex_t> h_output( output_elems );
    runtime_api_t::memcpy(
        h_output.data(), output_d.as_complex(), output_elems * sizeof( complex_t ), runtime_api_t::device_to_host_kind()
    );

    std::size_t local_errors = 0;
    if ( opt.egger_opt == 0 )
    {
        for ( std::size_t ix = 0; ix < output_dims.x; ++ix )
        {
            for ( std::size_t iy = 0; iy < output_dims.y; ++iy )
            {
                for ( std::size_t iz = 0; iz < output_dims.z; ++iz )
                {
                    std::size_t gx = 0, gy = 0, gz = 0;
                    switch ( pattern )
                    {
                        case comm_pattern::first_forward:
                        case comm_pattern::second_backward:
                            gx = part_x.start[pidx_i] + ix;
                            gy = iy;
                            gz = part_z.start[pidx_j] + iz;
                            break;
                        case comm_pattern::first_backward:
                            gx = part_x.start[pidx_i] + ix;
                            gy = part_y_input.start[pidx_j] + iy;
                            gz = iz;
                            break;
                        case comm_pattern::second_forward:
                            gx = ix;
                            gy = part_y_output.start[pidx_i] + iy;
                            gz = part_z.start[pidx_j] + iz;
                            break;
                    }

                    const complex_t expected = make_value( gx, gy, gz, opt.nx, opt.ny );
                    const complex_t actual   = h_output[index3( ix, iy, iz, output_dims )];
                    if ( actual != expected )
                    {
                        ++local_errors;
                        if ( local_errors == 1 )
                        {
                            log.error_f(
                                "first mismatch pattern=%s, local=(%zu,%zu,%zu), global=(%zu,%zu,%zu), "
                                "actual=(%.17e,%.17e), expected=(%.17e,%.17e)",
                                opt.pattern.c_str(), ix, iy, iz, gx, gy, gz, actual.real(), actual.imag(),
                                expected.real(), expected.imag()
                            );
                        }
                    }
                }
            }
        }
    }

    std::size_t global_errors = 0;
    MPI_Allreduce( &local_errors, &global_errors, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );

    const double avg_ms = mean( wall_ms );
    const double sd_ms  = stddev( wall_ms, avg_ms );

    double max_avg_ms = 0.0;
    MPI_Allreduce( &avg_ms, &max_avg_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

    if ( comm_info.myid == 0 )
    {
        const double send_mb  = static_cast<double>( total_send_bytes ) / 1.0e6;
        const double recv_mb  = static_cast<double>( total_recv_bytes ) / 1.0e6;
        const double total_mb = send_mb + recv_mb;
        const double bw_mb_s  = total_mb / ( max_avg_ms * 1.0e-3 );
        log.info_f(
            "test=egger-pencil-comm-3d, pattern=%s, method=%s, egger_opt=%d, large_count_transport=%s, "
            "grid=(%zu,%zu), sizes=(%zu,%zu,%zu), nz_complex=%zu, warmup=%d, times=%d, chunk_mib=%zu, "
            "validated=%d: avg_wall_ms=%.8e, stddev_wall_ms=%.8e, send_MB=%.8e, recv_MB=%.8e, "
            "total_bandwidth_MB_s=%.8e, global_errors=%zu",
            opt.pattern.c_str(), opt.method.c_str(), opt.egger_opt, opt.large_count_transport.c_str(), opt.p1, opt.p2,
            opt.nx, opt.ny, opt.nz, nz_complex, opt.warmup, opt.times, opt.chunk_mib, opt.egger_opt == 0 ? 1 : 0,
            max_avg_ms, sd_ms, send_mb, recv_mb, bw_mb_s, global_errors
        );
    }

    MPI_Comm_free( &comm1 );
    MPI_Comm_free( &comm2 );
    return global_errors == 0 ? 0 : 1;
}

} // namespace

int main( int argc, char *argv[] )
{
    int exit_code = 0;

    try
    {
        scfd::communication::mpi_wrap mpi( argc, argv, MPI_THREAD_MULTIPLE );
        const auto                    comm_info = mpi.comm_world();
        scfd::utils::log_mpi          log;

        fftm::test::detail::init_cuda_mpi_for_tests( log, comm_info );
        const test_options options = parse_options( argc, argv, comm_info.num_procs );
        exit_code                  = run_test( log, comm_info, options );
    }
    catch ( const std::exception &e )
    {
        std::cerr << scfd::utils::nested_exception_to_multistring( e ) << std::endl;
        exit_code = 1;
    }
    catch ( ... )
    {
        std::cerr << "unknown exception" << std::endl;
        exit_code = 1;
    }

    return exit_code;
}
