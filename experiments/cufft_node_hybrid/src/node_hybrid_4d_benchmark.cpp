#include "node_fft_4d_backend.hpp"

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <scfd/communication/mpi_comm.h>
#include <scfd/communication/mpi_wrap.h>

namespace
{

using fftm::experiments::node_hybrid::node_fft_4d_backend;
using fftm::experiments::node_hybrid::shape_4d;
using fftm::experiments::node_hybrid::validation_result;
using comm_t = scfd::communication::mpi_comm_info;
using clock_type = std::chrono::steady_clock;

struct options
{
    shape_4d shape{ 64, 64, 64, 64 };
    int gpus_per_process = 0;
    int warmup = 2;
    int iterations = 5;
    double epsilon = 1.0e-11;
    bool validate = true;
    std::string output = "node_hybrid_cufft_xt_4d.csv";
    std::string label = "unspecified";
};

struct iteration_sample
{
    int iteration = 0;
    double forward_yzw_ms = 0.0;
    double forward_layout_ms = 0.0;
    double forward_x_ms = 0.0;
    double inverse_x_ms = 0.0;
    double inverse_layout_ms = 0.0;
    double inverse_yzw_ms = 0.0;
    double total_ms = 0.0;
};

double elapsed_ms( clock_type::time_point start, clock_type::time_point stop )
{
    return std::chrono::duration<double, std::milli>( stop - start ).count();
}

std::vector<std::string> split( const std::string &value, char delimiter )
{
    std::vector<std::string> result;
    std::stringstream stream( value );
    std::string item;
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
    if ( consumed != value.size() || parsed < 0 ||
         parsed > std::numeric_limits<int>::max() )
    {
        throw std::invalid_argument(
            std::string( name ) + " must be a nonnegative integer"
        );
    }
    return static_cast<int>( parsed );
}

shape_4d parse_shape( const std::string &value )
{
    const auto items = split( value, ',' );
    if ( items.size() != 4 )
        throw std::invalid_argument( "--shape requires X,Y,Z,W" );
    return shape_4d{
        parse_positive_size( items[0], "shape X" ),
        parse_positive_size( items[1], "shape Y" ),
        parse_positive_size( items[2], "shape Z" ),
        parse_positive_size( items[3], "shape W" )
    };
}

void print_usage( const char *program )
{
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --size N                    Hypercubic transform size\n"
        << "  --shape X,Y,Z,W             Explicit transform dimensions\n"
        << "  --gpus-per-process N        GPUs used by the node plan; 0 means all visible\n"
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
                throw std::invalid_argument(
                    std::string( "missing value after " ) + name
                );
            return argv[++index];
        };

        if ( argument == "--size" )
        {
            const std::size_t size =
                parse_positive_size( require_value( "--size" ), "size" );
            result.shape = shape_4d{ size, size, size, size };
        }
        else if ( argument == "--shape" )
        {
            result.shape = parse_shape( require_value( "--shape" ) );
        }
        else if ( argument == "--gpus-per-process" )
        {
            result.gpus_per_process = parse_nonnegative_int(
                require_value( "--gpus-per-process" ), "gpus per process"
            );
        }
        else if ( argument == "--warmup" )
        {
            result.warmup =
                parse_nonnegative_int( require_value( "--warmup" ), "warmup" );
        }
        else if ( argument == "--iterations" )
        {
            result.iterations = parse_nonnegative_int(
                require_value( "--iterations" ), "iterations"
            );
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
    if ( !( result.epsilon > 0.0 ) )
        throw std::invalid_argument( "epsilon must be positive" );
    return result;
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
         summary_path.compare(
             summary_path.size() - suffix.size(), suffix.size(), suffix
         ) == 0 )
    {
        return summary_path.substr( 0, summary_path.size() - suffix.size() ) +
               ".samples.csv";
    }
    return summary_path + ".samples.csv";
}

template <class Function>
double time_stage( node_fft_4d_backend &backend, Function function )
{
    const auto start = clock_type::now();
    function();
    backend.synchronize();
    return elapsed_ms( start, clock_type::now() );
}

iteration_sample execute_pair(
    node_fft_4d_backend &backend, int iteration, bool print_progress
)
{
    iteration_sample sample;
    sample.iteration = iteration;
    const auto total_start = clock_type::now();

    if ( print_progress )
        std::cout << "NODE_HYBRID_4D_STAGE iteration=" << iteration
                  << " stage=forward_yzw begin=1" << std::endl;
    sample.forward_yzw_ms =
        time_stage( backend, [&]() { backend.forward_yzw(); } );

    if ( print_progress )
        std::cout << "NODE_HYBRID_4D_STAGE iteration=" << iteration
                  << " stage=forward_layout begin=1" << std::endl;
    sample.forward_layout_ms =
        time_stage( backend, [&]() { backend.transpose_x_to_z(); } );

    if ( print_progress )
        std::cout << "NODE_HYBRID_4D_STAGE iteration=" << iteration
                  << " stage=forward_x begin=1" << std::endl;
    sample.forward_x_ms =
        time_stage( backend, [&]() { backend.forward_x(); } );

    if ( print_progress )
        std::cout << "NODE_HYBRID_4D_STAGE iteration=" << iteration
                  << " stage=inverse_x begin=1" << std::endl;
    sample.inverse_x_ms =
        time_stage( backend, [&]() { backend.inverse_x(); } );

    if ( print_progress )
        std::cout << "NODE_HYBRID_4D_STAGE iteration=" << iteration
                  << " stage=inverse_layout begin=1" << std::endl;
    sample.inverse_layout_ms =
        time_stage( backend, [&]() { backend.transpose_z_to_x(); } );

    if ( print_progress )
        std::cout << "NODE_HYBRID_4D_STAGE iteration=" << iteration
                  << " stage=inverse_yzw begin=1" << std::endl;
    sample.inverse_yzw_ms =
        time_stage( backend, [&]() { backend.inverse_yzw(); } );
    sample.total_ms = elapsed_ms( total_start, clock_type::now() );
    return sample;
}

void write_results(
    const options &opts, const node_fft_4d_backend &backend,
    const std::vector<iteration_sample> &samples,
    const validation_result &validation, double plan_ms, double initialize_ms
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
        throw std::runtime_error( "failed to open 4D summary output" );
    summary
        << "label,backend,spectral_layout,nodes,ranks,gpus_per_process,"
           "nx,ny,nz,nw,logical_real_bytes,allocated_data_bytes,"
           "max_data_bytes_per_device,shared_work_bytes,"
           "max_work_bytes_per_device,forward_3d_work_bytes,inverse_3d_work_bytes,"
           "x_fft_work_bytes,transpose_bytes,warmup,iterations,plan_ms,"
           "initialize_ms,avg_pair_ms,min_pair_ms,stddev_pair_ms,"
           "relative_l2,max_abs,valid\n";
    summary << std::setprecision( 12 )
            << opts.label << ',' << backend.name() << ','
            << backend.spectral_layout_name() << ",1,1,"
            << backend.device_count() << ','
            << opts.shape.x << ',' << opts.shape.y << ',' << opts.shape.z << ','
            << opts.shape.w << ',' << backend.logical_real_bytes() << ','
            << backend.allocated_data_bytes() << ','
            << backend.max_allocated_data_bytes_per_device() << ','
            << backend.shared_work_bytes() << ','
            << backend.max_shared_work_bytes_per_device() << ','
            << backend.forward_3d_work_bytes() << ','
            << backend.inverse_3d_work_bytes() << ','
            << backend.x_fft_work_bytes() << ','
            << backend.transpose_bytes() << ','
            << opts.warmup << ',' << opts.iterations << ',' << plan_ms << ','
            << initialize_ms << ',' << average << ',' << minimum << ','
            << deviation << ',' << validation.relative_l2 << ','
            << validation.max_abs << ','
            << ( !opts.validate || validation.relative_l2 <= opts.epsilon ? 1 : 0 )
            << '\n';

    std::ofstream output(
        samples_path( opts.output ).c_str(), std::ios::out | std::ios::trunc
    );
    if ( !output )
        throw std::runtime_error( "failed to open 4D sample output" );
    output
        << "iteration,forward_yzw_ms,forward_layout_ms,forward_x_ms,"
           "inverse_x_ms,inverse_layout_ms,inverse_yzw_ms,total_ms\n";
    output << std::setprecision( 12 );
    for ( const auto &sample : samples )
    {
        output << sample.iteration << ',' << sample.forward_yzw_ms << ','
               << sample.forward_layout_ms << ',' << sample.forward_x_ms << ','
               << sample.inverse_x_ms << ',' << sample.inverse_layout_ms << ','
               << sample.inverse_yzw_ms << ',' << sample.total_ms << '\n';
    }
}

int run( int argc, char **argv, const comm_t &comm )
{
    const options opts = parse_options( argc, argv );
    if ( comm.num_procs != 1 )
        throw std::invalid_argument(
            "standalone 4D node-hybrid benchmark requires exactly one MPI process"
        );

    const int available_devices =
        fftm::experiments::node_hybrid::available_cuda_device_count();
    const int requested_devices =
        opts.gpus_per_process == 0 ? available_devices : opts.gpus_per_process;
    if ( requested_devices < 2 || requested_devices > available_devices )
        throw std::invalid_argument(
            "gpus per process must select between 2 and all visible GPUs"
        );
    std::vector<int> devices( static_cast<std::size_t>( requested_devices ) );
    std::iota( devices.begin(), devices.end(), 0 );

    std::cout << "NODE_HYBRID_4D_SETUP label=" << opts.label
              << " shape=" << opts.shape.x << 'x' << opts.shape.y << 'x'
              << opts.shape.z << 'x' << opts.shape.w
              << " gpus_per_process=" << requested_devices
              << " layout=native-xzwy" << std::endl;
    std::cout << "NODE_HYBRID_4D_PLAN begin=1" << std::endl;
    const auto plan_start = clock_type::now();
    auto backend =
        fftm::experiments::node_hybrid::make_cuda_cufft_xt_3d_1d_backend(
            opts.shape, devices
        );
    backend->synchronize();
    const double plan_ms = elapsed_ms( plan_start, clock_type::now() );
    std::cout << "NODE_HYBRID_4D_PLAN done=1 plan_ms=" << plan_ms
              << " allocated_data_bytes=" << backend->allocated_data_bytes()
              << " shared_work_bytes=" << backend->shared_work_bytes()
              << " max_device_bytes="
              << backend->max_allocated_data_bytes_per_device() +
                     backend->max_shared_work_bytes_per_device()
              << std::endl;

    constexpr std::uint64_t seed = 1;
    std::cout << "NODE_HYBRID_4D_INITIALIZE begin=1 logical_bytes="
              << backend->logical_real_bytes() << std::endl;
    const auto initialize_start = clock_type::now();
    backend->initialize_input( seed );
    const double initialize_ms =
        elapsed_ms( initialize_start, clock_type::now() );
    std::cout << "NODE_HYBRID_4D_INITIALIZE done=1 initialize_ms="
              << initialize_ms << std::endl;

    std::vector<iteration_sample> samples;
    samples.reserve( static_cast<std::size_t>( opts.iterations ) );
    const int total_iterations = opts.warmup + opts.iterations;
    for ( int iteration = 0; iteration < total_iterations; ++iteration )
    {
        const bool measured = iteration >= opts.warmup;
        const int sample_index = measured ? iteration - opts.warmup : iteration;
        iteration_sample sample =
            execute_pair( *backend, sample_index, true );
        backend->normalize_inverse_output();
        backend->synchronize();
        if ( measured )
            samples.push_back( sample );
        std::cout << "NODE_HYBRID_4D_ITERATION index=" << sample_index
                  << " measured=" << ( measured ? 1 : 0 )
                  << " total_ms=" << sample.total_ms << std::endl;
    }

    validation_result validation;
    if ( opts.validate )
    {
        std::cout << "NODE_HYBRID_4D_VALIDATE begin=1" << std::endl;
        backend->initialize_input( seed );
        execute_pair( *backend, 0, false );
        validation = backend->validate_input( seed );
        std::cout << std::setprecision( 12 )
                  << "NODE_HYBRID_4D_VALIDATE done=1 relative_l2="
                  << validation.relative_l2 << " max_abs="
                  << validation.max_abs << std::endl;
    }

    write_results(
        opts, *backend, samples, validation, plan_ms, initialize_ms
    );
    const bool valid = !opts.validate || validation.relative_l2 <= opts.epsilon;
    std::cout << std::setprecision( 12 )
              << "NODE_HYBRID_4D_RESULT label=" << opts.label
              << " relative_l2=" << validation.relative_l2
              << " max_abs=" << validation.max_abs
              << " valid=" << ( valid ? 1 : 0 )
              << " output=" << opts.output << std::endl;
    return valid ? 0 : 2;
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
            std::cerr << "NODE_HYBRID_4D_ERROR rank=" << comm.myid
                      << " message=" << error.what() << std::endl;
            MPI_Abort( comm.comm, 1 );
            return 1;
        }
    }
    catch ( const std::exception &error )
    {
        std::cerr << "NODE_HYBRID_4D_ERROR rank=uninitialized message="
                  << error.what() << std::endl;
        return 1;
    }
}
