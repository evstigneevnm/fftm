#include <fftw3-mpi.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include <thread>
#include <vector>

#include <sched.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace
{

struct options_t
{
    int                    dimension = 0;
    std::vector<ptrdiff_t> sizes;
    int                    threads    = 1;
    int                    warmup     = 2;
    int                    times      = 5;
    double                 epsilon    = 1.0e-11;
    double                 planner_time_limit_seconds = FFTW_NO_TIMELIMIT;
    std::string            planner    = "measure";
    std::string            output_dir = ".";
};

struct stats_t
{
    double mean    = 0.0;
    double stddev  = 0.0;
    double median  = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double p95     = 0.0;
};

[[noreturn]] void fail( MPI_Comm comm, const std::string &message, int code = 2 )
{
    int rank = 0;
    MPI_Comm_rank( comm, &rank );
    if ( rank == 0 )
        std::cerr << "ERROR: " << message << '\n';
    MPI_Abort( comm, code );
    std::abort();
}

void usage( const char *program )
{
    std::cerr << "Usage: " << program << " --dimension 3|4 --size N [options]\n"
              << "       " << program << " --dimension 3|4 --sizes N0,N1,... [options]\n\n"
              << "Options:\n"
              << "  --threads N          FFTW threads per MPI rank (default: 1)\n"
              << "  --warmup N           Untimed forward/inverse pairs (default: 2)\n"
              << "  --times N            Timed forward/inverse pairs (default: 5)\n"
              << "  --planner NAME       estimate, measure, patient, or exhaustive\n"
              << "  --planner-time-limit SECONDS\n"
              << "                       Limit each FFTW planner call; omitted means unlimited\n"
              << "  --epsilon VALUE      Maximum accepted relative L2 error\n"
              << "  --output-dir PATH    CSV output directory\n";
}

long parse_long( const std::string &text, const char *name )
{
    char *end        = nullptr;
    errno            = 0;
    const long value = std::strtol( text.c_str(), &end, 10 );
    if ( errno != 0 || end == text.c_str() || *end != '\0' )
        throw std::invalid_argument( std::string( "invalid " ) + name + ": " + text );
    return value;
}

double parse_double( const std::string &text, const char *name )
{
    char *end          = nullptr;
    errno              = 0;
    const double value = std::strtod( text.c_str(), &end );
    if ( errno != 0 || end == text.c_str() || *end != '\0' || !std::isfinite( value ) )
        throw std::invalid_argument( std::string( "invalid " ) + name + ": " + text );
    return value;
}

std::vector<ptrdiff_t> parse_sizes( const std::string &text )
{
    std::vector<ptrdiff_t> result;
    std::size_t            begin = 0;
    while ( begin <= text.size() )
    {
        const std::size_t end   = text.find( ',', begin );
        const std::string item  = text.substr( begin, end == std::string::npos ? std::string::npos : end - begin );
        const long        value = parse_long( item, "size" );
        if ( value <= 0 )
            throw std::invalid_argument( "FFT dimensions must be positive" );
        result.push_back( static_cast<ptrdiff_t>( value ) );
        if ( end == std::string::npos )
            break;
        begin = end + 1;
    }
    return result;
}

options_t parse_options( int argc, char **argv )
{
    options_t options;
    ptrdiff_t isotropic_size = 0;
    for ( int i = 1; i < argc; ++i )
    {
        const std::string arg           = argv[i];
        auto              require_value = [&]( const char *name ) {
            if ( i + 1 >= argc )
                throw std::invalid_argument( std::string( "missing value for " ) + name );
            return std::string( argv[++i] );
        };

        if ( arg == "--dimension" )
            options.dimension = static_cast<int>( parse_long( require_value( "--dimension" ), "dimension" ) );
        else if ( arg == "--size" )
            isotropic_size = static_cast<ptrdiff_t>( parse_long( require_value( "--size" ), "size" ) );
        else if ( arg == "--sizes" )
            options.sizes = parse_sizes( require_value( "--sizes" ) );
        else if ( arg == "--threads" )
            options.threads = static_cast<int>( parse_long( require_value( "--threads" ), "threads" ) );
        else if ( arg == "--warmup" )
            options.warmup = static_cast<int>( parse_long( require_value( "--warmup" ), "warmup" ) );
        else if ( arg == "--times" )
            options.times = static_cast<int>( parse_long( require_value( "--times" ), "times" ) );
        else if ( arg == "--planner" )
            options.planner = require_value( "--planner" );
        else if ( arg == "--planner-time-limit" )
            options.planner_time_limit_seconds =
                parse_double( require_value( "--planner-time-limit" ), "planner time limit" );
        else if ( arg == "--epsilon" )
            options.epsilon = parse_double( require_value( "--epsilon" ), "epsilon" );
        else if ( arg == "--output-dir" )
            options.output_dir = require_value( "--output-dir" );
        else if ( arg == "--help" || arg == "-h" )
        {
            usage( argv[0] );
            std::exit( 0 );
        }
        else
            throw std::invalid_argument( "unknown option: " + arg );
    }

    if ( options.dimension != 3 && options.dimension != 4 )
        throw std::invalid_argument( "--dimension must be 3 or 4" );
    if ( options.sizes.empty() )
    {
        if ( isotropic_size <= 0 )
            throw std::invalid_argument( "--size or --sizes is required" );
        options.sizes.assign( static_cast<std::size_t>( options.dimension ), isotropic_size );
    }
    if ( options.sizes.size() != static_cast<std::size_t>( options.dimension ) )
        throw std::invalid_argument( "--sizes count must equal --dimension" );
    if ( options.threads <= 0 || options.warmup < 0 || options.times <= 0 )
        throw std::invalid_argument( "threads and times must be positive; warmup must be nonnegative" );
    if ( options.epsilon <= 0.0 )
        throw std::invalid_argument( "--epsilon must be positive" );
    if ( options.planner_time_limit_seconds != FFTW_NO_TIMELIMIT && options.planner_time_limit_seconds <= 0.0 )
        throw std::invalid_argument( "--planner-time-limit must be positive" );
    return options;
}

unsigned planner_flags( const std::string &name )
{
    if ( name == "estimate" )
        return FFTW_ESTIMATE;
    if ( name == "measure" )
        return FFTW_MEASURE;
    if ( name == "patient" )
        return FFTW_PATIENT;
    if ( name == "exhaustive" )
        return FFTW_EXHAUSTIVE;
    throw std::invalid_argument( "unsupported planner: " + name );
}

ptrdiff_t checked_product( const std::vector<ptrdiff_t> &values, std::size_t begin = 0 )
{
    ptrdiff_t result = 1;
    for ( std::size_t i = begin; i < values.size(); ++i )
    {
        if ( values[i] <= 0 || result > std::numeric_limits<ptrdiff_t>::max() / values[i] )
            throw std::overflow_error( "FFT point count exceeds ptrdiff_t" );
        result *= values[i];
    }
    return result;
}

std::size_t checked_size_t( ptrdiff_t value, const char *name )
{
    if ( value < 0 ||
         static_cast<std::uintmax_t>( value ) > static_cast<std::uintmax_t>( std::numeric_limits<std::size_t>::max() ) )
        throw std::overflow_error( std::string( name ) + " exceeds size_t" );
    return static_cast<std::size_t>( value );
}

double input_value( std::uint64_t global_index )
{
    const std::uint64_t code = ( global_index * UINT64_C( 17 ) + UINT64_C( 13 ) ) & UINT64_C( 1023 );
    return ( static_cast<double>( code ) - 511.5 ) / 512.0;
}

void initialize_input(
    double *input, ptrdiff_t local_n0, ptrdiff_t local_0_start, ptrdiff_t rows_per_slab, ptrdiff_t logical_last_size,
    ptrdiff_t physical_last_stride, int threads
)
{
    const ptrdiff_t     local_rows = local_n0 * rows_per_slab;
    const ptrdiff_t     count      = local_rows * logical_last_size;
    const std::uint64_t global_row_start =
        static_cast<std::uint64_t>( local_0_start ) * static_cast<std::uint64_t>( rows_per_slab );
#pragma omp parallel for schedule( static ) num_threads( threads )
    for ( ptrdiff_t i = 0; i < count; ++i )
    {
        const ptrdiff_t     row    = i / logical_last_size;
        const ptrdiff_t     column = i % logical_last_size;
        const std::uint64_t global_index =
            ( global_row_start + static_cast<std::uint64_t>( row ) ) * static_cast<std::uint64_t>( logical_last_size ) +
            static_cast<std::uint64_t>( column );
        input[row * physical_last_stride + column] = input_value( global_index );
    }
}

void parallel_zero( double *data, ptrdiff_t count, int threads )
{
#pragma omp parallel for schedule( static ) num_threads( threads )
    for ( ptrdiff_t i = 0; i < count; ++i )
        data[i] = 0.0;
}

stats_t statistics( const std::vector<double> &values )
{
    if ( values.empty() )
        throw std::logic_error( "cannot calculate statistics for an empty sample set" );
    stats_t result;
    result.mean = std::accumulate( values.begin(), values.end(), 0.0 ) / static_cast<double>( values.size() );
    if ( values.size() > 1 )
    {
        double sum = 0.0;
        for ( const double value : values )
        {
            const double delta = value - result.mean;
            sum += delta * delta;
        }
        result.stddev = std::sqrt( sum / static_cast<double>( values.size() - 1 ) );
    }
    std::vector<double> sorted = values;
    std::sort( sorted.begin(), sorted.end() );
    result.minimum           = sorted.front();
    result.maximum           = sorted.back();
    const std::size_t middle = sorted.size() / 2;
    result.median            = sorted.size() % 2 == 0 ? 0.5 * ( sorted[middle - 1] + sorted[middle] ) : sorted[middle];
    const std::size_t p95_index = std::min(
        sorted.size() - 1, static_cast<std::size_t>( std::ceil( 0.95 * static_cast<double>( sorted.size() ) ) ) - 1
    );
    result.p95 = sorted[p95_index];
    return result;
}

std::string join_sizes( const std::vector<ptrdiff_t> &sizes, const char *separator )
{
    std::ostringstream stream;
    for ( std::size_t i = 0; i < sizes.size(); ++i )
    {
        if ( i != 0 )
            stream << separator;
        stream << sizes[i];
    }
    return stream.str();
}

std::string cpu_model()
{
    std::ifstream input( "/proc/cpuinfo" );
    std::string   line;
    while ( std::getline( input, line ) )
    {
        const std::string prefix = "model name";
        if ( line.compare( 0, prefix.size(), prefix ) != 0 )
            continue;
        const std::size_t colon = line.find( ':' );
        if ( colon == std::string::npos )
            continue;
        std::string       value = line.substr( colon + 1 );
        const std::size_t first = value.find_first_not_of( " \t" );
        return first == std::string::npos ? std::string() : value.substr( first );
    }
    return "";
}

std::string environment_value( const char *name )
{
    const char *value = std::getenv( name );
    return value == nullptr ? std::string() : std::string( value );
}

std::string mpi_library_version()
{
    char value[MPI_MAX_LIBRARY_VERSION_STRING] = {};
    int  length                                = 0;
    if ( MPI_Get_library_version( value, &length ) != MPI_SUCCESS )
        return "";
    std::string       result( value, static_cast<std::size_t>( length ) );
    const std::size_t null_position = result.find( '\0' );
    if ( null_position != std::string::npos )
        result.resize( null_position );
    for ( char &character : result )
    {
        if ( character == '\n' || character == '\r' || character == '\t' )
            character = ' ';
    }
    while ( !result.empty() && ( result.back() == '\n' || result.back() == '\r' || result.back() == ' ' ) )
        result.pop_back();
    return result;
}

int allowed_cpu_count()
{
    cpu_set_t mask;
    CPU_ZERO( &mask );
    if ( sched_getaffinity( 0, sizeof( mask ), &mask ) != 0 )
        return 0;
    return CPU_COUNT( &mask );
}

std::string csv_escape( const std::string &value )
{
    if ( value.find_first_of( ",\"\n" ) == std::string::npos )
        return value;
    std::string escaped = "\"";
    for ( const char c : value )
    {
        if ( c == '"' )
            escaped += '"';
        escaped += c;
    }
    escaped += '"';
    return escaped;
}

bool file_is_empty( const std::string &path )
{
    struct stat info;
    return stat( path.c_str(), &info ) != 0 || info.st_size == 0;
}

void ensure_directory( const std::string &path )
{
    if ( path.empty() || path == "." )
        return;
    std::string current;
    if ( path.front() == '/' )
        current = "/";
    std::size_t begin = path.front() == '/' ? 1 : 0;
    while ( begin <= path.size() )
    {
        const std::size_t end  = path.find( '/', begin );
        const std::string part = path.substr( begin, end == std::string::npos ? std::string::npos : end - begin );
        if ( !part.empty() )
        {
            if ( current.size() > 1 && current.back() != '/' )
                current += '/';
            current += part;
            if ( mkdir( current.c_str(), 0775 ) != 0 && errno != EEXIST )
                throw std::runtime_error( "failed to create output directory " + current );
        }
        if ( end == std::string::npos )
            break;
        begin = end + 1;
    }
}

std::string unique_nodes( MPI_Comm comm )
{
    int rank = 0;
    int size = 0;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &size );
    char hostname[MPI_MAX_PROCESSOR_NAME] = {};
    int  length                           = 0;
    MPI_Get_processor_name( hostname, &length );
    std::vector<char> gathered( rank == 0 ? static_cast<std::size_t>( size ) * MPI_MAX_PROCESSOR_NAME : 0 );
    MPI_Gather(
        hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, rank == 0 ? gathered.data() : nullptr, MPI_MAX_PROCESSOR_NAME,
        MPI_CHAR, 0, comm
    );
    if ( rank != 0 )
        return "";
    std::set<std::string> nodes;
    for ( int i = 0; i < size; ++i )
        nodes.insert( std::string( gathered.data() + static_cast<std::size_t>( i ) * MPI_MAX_PROCESSOR_NAME ) );
    std::ostringstream stream;
    for ( auto it = nodes.begin(); it != nodes.end(); ++it )
    {
        if ( it != nodes.begin() )
            stream << ';';
        stream << *it;
    }
    return stream.str();
}

std::string make_run_id( int dimension, const std::vector<ptrdiff_t> &sizes, int ranks, int threads )
{
    std::ostringstream stream;
    stream << "d" << dimension << "_n" << join_sizes( sizes, "x" ) << "_r" << ranks << "_t" << threads << "_p"
           << getpid();
    return stream.str();
}

} // namespace

int main( int argc, char **argv )
{
    for ( int i = 1; i < argc; ++i )
    {
        if ( std::strcmp( argv[i], "--help" ) == 0 || std::strcmp( argv[i], "-h" ) == 0 )
        {
            usage( argv[0] );
            return 0;
        }
    }

    int provided = MPI_THREAD_SINGLE;
    if ( MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided ) != MPI_SUCCESS )
        return 2;

    MPI_Comm comm  = MPI_COMM_WORLD;
    int      rank  = 0;
    int      ranks = 0;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &ranks );

    options_t options;
    try
    {
        options = parse_options( argc, argv );
        if ( provided < MPI_THREAD_FUNNELED )
            fail( comm, "MPI does not provide MPI_THREAD_FUNNELED" );
    }
    catch ( const std::exception &error )
    {
        if ( rank == 0 )
        {
            std::cerr << "ERROR: " << error.what() << '\n';
            usage( argv[0] );
        }
        MPI_Finalize();
        return 2;
    }

    const int launch_allowed_cpus = allowed_cpu_count();
    int       min_launch_allowed_cpus = 0;
    int       max_launch_allowed_cpus = 0;
    MPI_Allreduce( &launch_allowed_cpus, &min_launch_allowed_cpus, 1, MPI_INT, MPI_MIN, comm );
    MPI_Allreduce( &launch_allowed_cpus, &max_launch_allowed_cpus, 1, MPI_INT, MPI_MAX, comm );
    if ( min_launch_allowed_cpus < options.threads )
    {
        std::ostringstream message;
        message << "rank CPU affinity exposes only " << min_launch_allowed_cpus
                << " CPUs, fewer than the requested " << options.threads << " FFTW threads";
        fail( comm, message.str() );
    }

    if ( fftw_init_threads() == 0 )
        fail( comm, "fftw_init_threads failed" );
    fftw_mpi_init();
    fftw_plan_with_nthreads( options.threads );

    if ( rank == 0 )
    {
        std::cerr << "FFTW_MPI_PROGRESS phase=setup dimension=" << options.dimension
                  << " sizes=" << join_sizes( options.sizes, "x" ) << " ranks=" << ranks
                  << " threads=" << options.threads << " planner=" << options.planner
                  << " planner_time_limit_seconds=" << options.planner_time_limit_seconds << std::endl;
    }

    std::vector<ptrdiff_t> complex_sizes = options.sizes;
    complex_sizes.back()                 = options.sizes.back() / 2 + 1;

    ptrdiff_t       local_n0      = 0;
    ptrdiff_t       local_0_start = 0;
    ptrdiff_t       local_n1      = 0;
    ptrdiff_t       local_1_start = 0;
    const ptrdiff_t alloc_local   = fftw_mpi_local_size_transposed(
        options.dimension, complex_sizes.data(), comm, &local_n0, &local_0_start, &local_n1, &local_1_start
    );
    if ( alloc_local <= 0 )
        fail( comm, "FFTW returned a nonpositive local allocation" );

    ptrdiff_t slab_points   = 0;
    ptrdiff_t rows_per_slab = 0;
    ptrdiff_t total_points  = 0;
    try
    {
        slab_points = checked_product( options.sizes, 1 );
        std::vector<ptrdiff_t> row_dimensions( options.sizes.begin() + 1, options.sizes.end() - 1 );
        rows_per_slab = checked_product( row_dimensions );
        total_points  = checked_product( options.sizes );
    }
    catch ( const std::exception &error )
    {
        fail( comm, error.what() );
    }
    if ( local_n0 > std::numeric_limits<ptrdiff_t>::max() / slab_points )
        fail( comm, "local real allocation overflows ptrdiff_t" );
    const ptrdiff_t logical_real_local   = local_n0 * slab_points;
    const ptrdiff_t physical_last_stride = 2 * complex_sizes.back();
    if ( local_n0 > std::numeric_limits<ptrdiff_t>::max() / rows_per_slab )
        fail( comm, "local real row count overflows ptrdiff_t" );
    const ptrdiff_t local_real_rows = local_n0 * rows_per_slab;
    if ( local_real_rows > std::numeric_limits<ptrdiff_t>::max() / physical_last_stride )
        fail( comm, "padded local real allocation overflows ptrdiff_t" );
    const ptrdiff_t physical_real_local = local_real_rows * physical_last_stride;
    if ( alloc_local > std::numeric_limits<ptrdiff_t>::max() / 2 )
        fail( comm, "local FFTW allocation overflows ptrdiff_t" );
    const ptrdiff_t real_alloc_local = std::max( physical_real_local, 2 * alloc_local );

    double       *input    = fftw_alloc_real( checked_size_t( real_alloc_local, "real allocation" ) );
    double       *output   = fftw_alloc_real( checked_size_t( real_alloc_local, "real allocation" ) );
    fftw_complex *spectrum = fftw_alloc_complex( checked_size_t( alloc_local, "complex allocation" ) );
    if ( input == nullptr || output == nullptr || spectrum == nullptr )
        fail( comm, "FFTW data allocation failed" );
    parallel_zero( input, real_alloc_local, options.threads );
    parallel_zero( output, real_alloc_local, options.threads );
    parallel_zero( reinterpret_cast<double *>( spectrum ), 2 * alloc_local, options.threads );
    const int post_openmp_allowed_cpus = allowed_cpu_count();
    int       min_post_openmp_allowed_cpus = 0;
    int       max_post_openmp_allowed_cpus = 0;
    MPI_Allreduce( &post_openmp_allowed_cpus, &min_post_openmp_allowed_cpus, 1, MPI_INT, MPI_MIN, comm );
    MPI_Allreduce( &post_openmp_allowed_cpus, &max_post_openmp_allowed_cpus, 1, MPI_INT, MPI_MAX, comm );
    if ( min_post_openmp_allowed_cpus < options.threads )
    {
        std::ostringstream message;
        message << "OpenMP initialization reduced rank CPU affinity to " << min_post_openmp_allowed_cpus
                << " CPUs, fewer than the requested " << options.threads << " FFTW threads";
        fail( comm, message.str() );
    }
    if ( rank == 0 )
    {
        std::cerr << "FFTW_MPI_AFFINITY launch_allowed_cpus=" << min_launch_allowed_cpus << ':'
                  << max_launch_allowed_cpus << " post_openmp_allowed_cpus=" << min_post_openmp_allowed_cpus << ':'
                  << max_post_openmp_allowed_cpus << std::endl;
    }

    unsigned base_flags = 0;
    try
    {
        base_flags = planner_flags( options.planner );
    }
    catch ( const std::exception &error )
    {
        fail( comm, error.what() );
    }

    if ( rank == 0 )
    {
        const unsigned long long local_bytes =
            static_cast<unsigned long long>( 2 ) * static_cast<unsigned long long>( real_alloc_local ) *
                sizeof( double ) +
            static_cast<unsigned long long>( alloc_local ) * sizeof( fftw_complex );
        std::cerr << "FFTW_MPI_PROGRESS phase=plan_forward_begin local_alloc_mib="
                  << static_cast<double>( local_bytes ) / ( 1024.0 * 1024.0 ) << std::endl;
    }
    fftw_set_timelimit( options.planner_time_limit_seconds );
    MPI_Barrier( comm );
    const double plan_forward_start = MPI_Wtime();
    fftw_plan    forward            = fftw_mpi_plan_dft_r2c(
        options.dimension, options.sizes.data(), input, spectrum, comm,
        base_flags | FFTW_PRESERVE_INPUT | FFTW_MPI_TRANSPOSED_OUT
    );
    const double plan_forward_local_ms = 1000.0 * ( MPI_Wtime() - plan_forward_start );
    double       plan_forward_ms       = 0.0;
    MPI_Allreduce( &plan_forward_local_ms, &plan_forward_ms, 1, MPI_DOUBLE, MPI_MAX, comm );
    if ( forward == nullptr )
        fail( comm, "failed to create FFTW MPI forward plan" );

    if ( rank == 0 )
        std::cerr << "FFTW_MPI_PROGRESS phase=plan_forward_end elapsed_ms=" << plan_forward_ms << std::endl;
    if ( rank == 0 )
        std::cerr << "FFTW_MPI_PROGRESS phase=plan_backward_begin" << std::endl;
    fftw_set_timelimit( options.planner_time_limit_seconds );
    MPI_Barrier( comm );
    const double plan_backward_start = MPI_Wtime();
    fftw_plan    backward            = fftw_mpi_plan_dft_c2r(
        options.dimension, options.sizes.data(), spectrum, output, comm, base_flags | FFTW_MPI_TRANSPOSED_IN
    );
    const double plan_backward_local_ms = 1000.0 * ( MPI_Wtime() - plan_backward_start );
    double       plan_backward_ms       = 0.0;
    MPI_Allreduce( &plan_backward_local_ms, &plan_backward_ms, 1, MPI_DOUBLE, MPI_MAX, comm );
    if ( backward == nullptr )
        fail( comm, "failed to create FFTW MPI backward plan" );

    if ( rank == 0 )
        std::cerr << "FFTW_MPI_PROGRESS phase=plan_backward_end elapsed_ms=" << plan_backward_ms << std::endl;

    initialize_input(
        input, local_n0, local_0_start, rows_per_slab, options.sizes.back(), physical_last_stride, options.threads
    );
    parallel_zero( output, real_alloc_local, options.threads );

    if ( rank == 0 )
        std::cerr << "FFTW_MPI_PROGRESS phase=warmup_begin iterations=" << options.warmup << std::endl;
    for ( int iteration = 0; iteration < options.warmup; ++iteration )
    {
        MPI_Barrier( comm );
        fftw_execute( forward );
        fftw_execute( backward );
        if ( rank == 0 )
            std::cerr << "FFTW_MPI_PROGRESS phase=warmup_iteration_end iteration=" << iteration << std::endl;
    }
    if ( rank == 0 )
        std::cerr << "FFTW_MPI_PROGRESS phase=warmup_end" << std::endl;

    std::vector<double> pair_samples;
    std::vector<double> forward_samples;
    std::vector<double> backward_samples;
    if ( rank == 0 )
    {
        pair_samples.reserve( static_cast<std::size_t>( options.times ) );
        forward_samples.reserve( static_cast<std::size_t>( options.times ) );
        backward_samples.reserve( static_cast<std::size_t>( options.times ) );
    }

    if ( rank == 0 )
        std::cerr << "FFTW_MPI_PROGRESS phase=timed_begin iterations=" << options.times << std::endl;
    for ( int iteration = 0; iteration < options.times; ++iteration )
    {
        MPI_Barrier( comm );
        const double pair_start = MPI_Wtime();
        fftw_execute( forward );
        const double forward_end = MPI_Wtime();
        fftw_execute( backward );
        const double pair_end = MPI_Wtime();

        double local_ms[3] = {
            1000.0 * ( forward_end - pair_start ), 1000.0 * ( pair_end - forward_end ),
            1000.0 * ( pair_end - pair_start )
        };
        double global_ms[3] = {};
        MPI_Reduce( local_ms, global_ms, 3, MPI_DOUBLE, MPI_MAX, 0, comm );
        if ( rank == 0 )
        {
            forward_samples.push_back( global_ms[0] );
            backward_samples.push_back( global_ms[1] );
            pair_samples.push_back( global_ms[2] );
            std::cerr << "FFTW_MPI_PROGRESS phase=timed_iteration_end iteration=" << iteration
                      << " forward_ms=" << global_ms[0] << " backward_ms=" << global_ms[1]
                      << " pair_ms=" << global_ms[2] << std::endl;
        }
    }
    if ( rank == 0 )
        std::cerr << "FFTW_MPI_PROGRESS phase=timed_end" << std::endl;

    long double         local_diff2      = 0.0L;
    long double         local_reference2 = 0.0L;
    double              local_max_abs    = 0.0;
    const std::uint64_t global_row_begin =
        static_cast<std::uint64_t>( local_0_start ) * static_cast<std::uint64_t>( rows_per_slab );
    for ( ptrdiff_t i = 0; i < logical_real_local; ++i )
    {
        const ptrdiff_t     row          = i / options.sizes.back();
        const ptrdiff_t     column       = i % options.sizes.back();
        const std::uint64_t global_index = ( global_row_begin + static_cast<std::uint64_t>( row ) ) *
                                               static_cast<std::uint64_t>( options.sizes.back() ) +
                                           static_cast<std::uint64_t>( column );
        const long double expected =
            static_cast<long double>( input_value( global_index ) ) * static_cast<long double>( total_points );
        const long double difference =
            static_cast<long double>( output[row * physical_last_stride + column] ) - expected;
        local_diff2 += difference * difference;
        local_reference2 += expected * expected;
        local_max_abs = std::max( local_max_abs, std::abs( static_cast<double>( difference ) ) );
    }
    long double global_diff2      = 0.0L;
    long double global_reference2 = 0.0L;
    double      global_max_abs    = 0.0;
    MPI_Allreduce( &local_diff2, &global_diff2, 1, MPI_LONG_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( &local_reference2, &global_reference2, 1, MPI_LONG_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( &local_max_abs, &global_max_abs, 1, MPI_DOUBLE, MPI_MAX, comm );
    const double relative_l2 = std::sqrt( static_cast<double>( global_diff2 / std::max( global_reference2, 1.0L ) ) );

    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split_type( comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm );
    int local_ranks = 0;
    MPI_Comm_size( local_comm, &local_ranks );
    int min_local_ranks = 0;
    int max_local_ranks = 0;
    MPI_Allreduce( &local_ranks, &min_local_ranks, 1, MPI_INT, MPI_MIN, comm );
    MPI_Allreduce( &local_ranks, &max_local_ranks, 1, MPI_INT, MPI_MAX, comm );
    MPI_Comm_free( &local_comm );

    const std::string nodes = unique_nodes( comm );
    const int node_count    = rank == 0 ? static_cast<int>( std::count( nodes.begin(), nodes.end(), ';' ) + 1 ) : 0;
    const unsigned long long local_bytes =
        static_cast<unsigned long long>( 2 ) * static_cast<unsigned long long>( real_alloc_local ) * sizeof( double ) +
        static_cast<unsigned long long>( alloc_local ) * sizeof( fftw_complex );
    unsigned long long max_rank_bytes = 0;
    unsigned long long total_bytes    = 0;
    MPI_Reduce( &local_bytes, &max_rank_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm );
    MPI_Reduce( &local_bytes, &total_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm );

    if ( rank == 0 )
    {
        try
        {
            ensure_directory( options.output_dir );
            const stats_t     pair           = statistics( pair_samples );
            const stats_t     forward_stats  = statistics( forward_samples );
            const stats_t     backward_stats = statistics( backward_samples );
            const std::string run_id         = make_run_id( options.dimension, options.sizes, ranks, options.threads );
            const std::string summary_path   = options.output_dir + "/benchmark_fftw_mpi_cpu.csv";
            const bool        write_summary_header = file_is_empty( summary_path );
            std::ofstream     summary( summary_path, std::ios::app );
            if ( !summary )
                throw std::runtime_error( "cannot open " + summary_path );
            if ( write_summary_header )
            {
                summary << "run_id,dimension,sizes,num_ranks,num_nodes,min_ranks_per_node,"
                        << "max_ranks_per_node,threads_per_rank,allowed_cpus_per_rank,planner,"
                        << "transposed_layout,warmup,times,total_points,local_n0,local_0_start,"
                        << "local_n1_transposed,local_1_start_transposed,alloc_local_complex,"
                        << "local_real_rows,real_last_stride,"
                        << "max_rank_alloc_mib,total_alloc_gib,plan_forward_ms,plan_backward_ms,"
                        << "avg_forward_ms,stddev_forward_ms,avg_backward_ms,stddev_backward_ms,"
                        << "avg_pair_ms,stddev_pair_ms,median_pair_ms,min_pair_ms,max_pair_ms,"
                        << "p95_pair_ms,relative_l2,max_abs_error,fftw_version,mpi_library,"
                        << "slurm_job_id,cpu_model,nodes,planner_time_limit_seconds,"
                        << "max_allowed_cpus_per_rank,post_openmp_min_allowed_cpus_per_rank,"
                        << "post_openmp_max_allowed_cpus_per_rank\n";
            }
            summary << std::setprecision( 12 ) << csv_escape( run_id ) << ',' << options.dimension << ','
                    << csv_escape( join_sizes( options.sizes, "x" ) ) << ',' << ranks << ',' << node_count << ','
                    << min_local_ranks << ',' << max_local_ranks << ',' << options.threads << ','
                    << min_launch_allowed_cpus
                    << ',' << csv_escape( options.planner ) << ",1," << options.warmup << ',' << options.times << ','
                    << total_points << ',' << local_n0 << ',' << local_0_start << ',' << local_n1 << ','
                    << local_1_start << ',' << alloc_local << ',' << local_real_rows << ',' << physical_last_stride
                    << ',' << static_cast<double>( max_rank_bytes ) / ( 1024.0 * 1024.0 ) << ','
                    << static_cast<double>( total_bytes ) / ( 1024.0 * 1024.0 * 1024.0 ) << ',' << plan_forward_ms
                    << ',' << plan_backward_ms << ',' << forward_stats.mean << ',' << forward_stats.stddev << ','
                    << backward_stats.mean << ',' << backward_stats.stddev << ',' << pair.mean << ',' << pair.stddev
                    << ',' << pair.median << ',' << pair.minimum << ',' << pair.maximum << ',' << pair.p95 << ','
                    << relative_l2 << ',' << global_max_abs << ',' << csv_escape( fftw_version ) << ','
                    << csv_escape( mpi_library_version() ) << ',' << csv_escape( environment_value( "SLURM_JOB_ID" ) )
                    << ',' << csv_escape( cpu_model() ) << ',' << csv_escape( nodes ) << ','
                    << options.planner_time_limit_seconds << ',' << max_launch_allowed_cpus << ','
                    << min_post_openmp_allowed_cpus << ',' << max_post_openmp_allowed_cpus << '\n';

            const std::string iteration_path         = options.output_dir + "/fftw_mpi_cpu_iterations.csv";
            const bool        write_iteration_header = file_is_empty( iteration_path );
            std::ofstream     iterations( iteration_path, std::ios::app );
            if ( !iterations )
                throw std::runtime_error( "cannot open " + iteration_path );
            if ( write_iteration_header )
                iterations << "run_id,iteration,forward_ms,backward_ms,pair_ms\n";
            for ( std::size_t i = 0; i < pair_samples.size(); ++i )
            {
                iterations << csv_escape( run_id ) << ',' << i << ',' << std::setprecision( 12 ) << forward_samples[i]
                           << ',' << backward_samples[i] << ',' << pair_samples[i] << '\n';
            }

            std::cout << std::setprecision( 9 ) << "FFTW_MPI_RESULT dimension=" << options.dimension
                      << " sizes=" << join_sizes( options.sizes, "x" ) << " ranks=" << ranks << " nodes=" << node_count
                      << " ranks_per_node=" << min_local_ranks << ':' << max_local_ranks
                      << " threads=" << options.threads << " planner=" << options.planner
                      << " planner_time_limit_seconds=" << options.planner_time_limit_seconds
                      << " avg_pair_ms=" << pair.mean << " stddev_pair_ms=" << pair.stddev
                      << " relative_l2=" << relative_l2 << " launch_allowed_cpus=" << min_launch_allowed_cpus << ':'
                      << max_launch_allowed_cpus << " post_openmp_allowed_cpus=" << min_post_openmp_allowed_cpus << ':'
                      << max_post_openmp_allowed_cpus
                      << " max_rank_alloc_mib=" << static_cast<double>( max_rank_bytes ) / ( 1024.0 * 1024.0 ) << '\n';
        }
        catch ( const std::exception &error )
        {
            fail( comm, error.what() );
        }
    }

    int validation_failed = relative_l2 > options.epsilon ? 1 : 0;
    MPI_Bcast( &validation_failed, 1, MPI_INT, 0, comm );

    fftw_destroy_plan( forward );
    fftw_destroy_plan( backward );
    fftw_free( input );
    fftw_free( output );
    fftw_free( spectrum );
    fftw_mpi_cleanup();
    fftw_cleanup_threads();
    MPI_Finalize();
    return validation_failed;
}
