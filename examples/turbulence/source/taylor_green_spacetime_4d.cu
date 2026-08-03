#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/system_timer_event.h>

#include <detail/array_arrangers.h>
#include <fftm.hpp>
#include <fftm_backend.hpp>

#include "snapshot_writer.h"
#include "spacetime_kernels.h"
#include "spacetime_options.h"

namespace
{

using real_t        = double;
using fft_backend_t = fftm::device_backend::fft<real_t>;
using runtime_api_t = typename fft_backend_t::runtime_api;
using backend_t     = fftm::device_backend::scfd_backend;
using memory_t      = typename backend_t::memory_type;
using reduce_t      = typename backend_t::reduce_type;
using for_each_3d_t = typename backend_t::template for_each_nd_type<3, int>;
using for_each_4d_t = typename backend_t::template for_each_nd_type<4, int>;
using index3_t      = scfd::static_vec::vec<int, 3>;
using index4_t      = scfd::static_vec::vec<int, 4>;
using rect3_t       = scfd::static_vec::rect<int, 3>;
using rect4_t       = scfd::static_vec::rect<int, 4>;

struct snapshot_record
{
    std::size_t step = 0;
    double      time = 0;
    std::string field;
    std::string file;
};

std::vector<std::string> split_csv_line( const std::string &line )
{
    std::vector<std::string> fields;
    std::stringstream        stream( line );
    std::string              value;
    while ( std::getline( stream, value, ',' ) )
        fields.push_back( value );
    return fields;
}

std::vector<snapshot_record> load_snapshot_manifest(
    const fftm::examples::turbulence::spacetime_options &options
)
{
    const std::string path = options.input_dir + "/snapshots.csv";
    std::ifstream     input( path.c_str() );
    if ( !input )
        throw std::runtime_error( "Failed to open snapshot manifest: " + path );

    std::string line;
    if ( !std::getline( input, line ) )
        throw std::runtime_error( "Missing snapshot manifest header in " + path );
    if ( !line.empty() && line[line.size() - 1] == '\r' )
        line.erase( line.size() - 1 );
    if ( line != "step,time,field,file" )
        throw std::runtime_error( "Unexpected snapshot manifest header in " + path );

    std::vector<snapshot_record> matching;
    while ( std::getline( input, line ) )
    {
        if ( !line.empty() && line[line.size() - 1] == '\r' )
            line.erase( line.size() - 1 );
        if ( line.empty() )
            continue;
        const auto fields = split_csv_line( line );
        if ( fields.size() != 4 )
            throw std::runtime_error( "Malformed snapshot manifest row: " + line );
        snapshot_record record;
        record.step  = static_cast<std::size_t>( std::strtoull( fields[0].c_str(), nullptr, 10 ) );
        record.time  = std::strtod( fields[1].c_str(), nullptr );
        record.field = fields[2];
        record.file  = fields[3];
        if ( record.field == options.field )
            matching.push_back( record );
    }

    if ( options.frame_offset + options.frames > matching.size() )
    {
        throw std::runtime_error(
            "Requested " + std::to_string( options.frames ) + " frames at offset " +
            std::to_string( options.frame_offset ) + ", but the manifest contains only " +
            std::to_string( matching.size() ) + " matching '" + options.field + "' snapshots"
        );
    }
    std::vector<snapshot_record> selected(
        matching.begin() + static_cast<std::ptrdiff_t>( options.frame_offset ),
        matching.begin() + static_cast<std::ptrdiff_t>( options.frame_offset + options.frames )
    );

    if ( selected.size() > 1 )
    {
        const double dt = selected[1].time - selected[0].time;
        if ( dt <= 0 )
            throw std::runtime_error( "Snapshot times must be strictly increasing" );
        for ( std::size_t i = 2; i < selected.size(); ++i )
        {
            const double current = selected[i].time - selected[i - 1].time;
            if ( std::abs( current - dt ) > 1.0e-8 * std::max( 1.0, std::abs( dt ) ) )
                throw std::runtime_error( "4D FFT requires uniformly spaced snapshot times" );
        }
    }
    return selected;
}

bool wrap_mpi_processes_over_gpus_enabled()
{
    const char *value = std::getenv( "FFTM_WRAP_PROCS_GPUS" );
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

template <class Array>
rect3_t range_3d( const Array &array )
{
    const auto size = array.size_nd();
    return rect3_t(
        index3_t( 0, 0, 0 ),
        index3_t( static_cast<int>( size[0] ), static_cast<int>( size[1] ), static_cast<int>( size[2] ) )
    );
}

template <class Array>
rect4_t range_4d( const Array &array )
{
    const auto size = array.size_nd();
    return rect4_t(
        index4_t( 0, 0, 0, 0 ),
        index4_t(
            static_cast<int>( size[0] ), static_cast<int>( size[1] ), static_cast<int>( size[2] ),
            static_cast<int>( size[3] )
        )
    );
}

fftm::fftm_init_options production_4d_options()
{
    return fftm::production_options_4d(
        fftm::transform_strategy_4d_mpi::slab_slab,
        fftm::fftm_4d_spectral_layout::native_xzwy,
        true
    );
}

template <class HostSnapshot>
void read_local_snapshot(
    const scfd::communication::mpi_comm_info &comm, const std::string &path, std::size_t global_size,
    std::size_t start_x, std::size_t start_y, std::size_t start_z, HostSnapshot &host,
    std::vector<float> &global
)
{
    using clock_t = std::chrono::steady_clock;
    const auto begin = clock_t::now();
    const std::uintmax_t expected_bytes =
        static_cast<std::uintmax_t>( global_size ) * static_cast<std::uintmax_t>( global_size ) *
        static_cast<std::uintmax_t>( global_size ) * sizeof( float );
    const std::size_t expected_count = static_cast<std::size_t>( expected_bytes / sizeof( float ) );
    if ( expected_count > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
        throw std::overflow_error( "Snapshot broadcast exceeds the MPI int-count limit" );
    if ( global.size() != expected_count )
        throw std::logic_error( "Snapshot broadcast buffer has the wrong size" );

    int read_ok = 1;
    if ( comm.myid == 0 )
    {
        std::ifstream input( path.c_str(), std::ios::binary | std::ios::ate );
        if ( !input )
        {
            read_ok = 0;
            std::cerr << "TAYLOR_GREEN_SNAPSHOT_READ_ERROR path=" << path
                      << " message=\"failed to open snapshot\"" << std::endl;
        }
        else
        {
            const std::streamoff actual_bytes = input.tellg();
            if ( actual_bytes < 0 || static_cast<std::uintmax_t>( actual_bytes ) != expected_bytes )
            {
                read_ok = 0;
                std::cerr << "TAYLOR_GREEN_SNAPSHOT_READ_ERROR path=" << path
                          << " expected_bytes=" << expected_bytes << " actual_bytes=" << actual_bytes
                          << std::endl;
            }
            else
            {
                input.seekg( 0, std::ios::beg );
                input.read(
                    reinterpret_cast<char *>( global.data() ),
                    static_cast<std::streamsize>( expected_bytes )
                );
                if ( !input )
                {
                    read_ok = 0;
                    std::cerr << "TAYLOR_GREEN_SNAPSHOT_READ_ERROR path=" << path
                              << " message=\"failed to read complete snapshot\"" << std::endl;
                }
            }
        }
    }
    comm.bcast( &read_ok, 1, 0 );
    if ( !read_ok )
        throw std::runtime_error( "Rank 0 failed to read snapshot: " + path );
    const auto read = clock_t::now();

    comm.bcast( global.data(), static_cast<int>( expected_count ), 0 );
    const auto broadcast = clock_t::now();
    const auto local = host.size_nd();
    for ( std::size_t z = 0; z < static_cast<std::size_t>( local[2] ); ++z )
    {
        for ( std::size_t y = 0; y < static_cast<std::size_t>( local[1] ); ++y )
        {
            const std::size_t offset =
                start_x + global_size * ( start_y + y + global_size * ( start_z + z ) );
            const index3_t local_index( 0, static_cast<int>( y ), static_cast<int>( z ) );
            std::copy(
                global.begin() + static_cast<std::ptrdiff_t>( offset ),
                global.begin() + static_cast<std::ptrdiff_t>(
                                     offset + static_cast<std::size_t>( local[0] )
                                 ),
                &host( local_index )
            );
        }
    }
    if ( comm.myid == 0 )
    {
        const auto finish = clock_t::now();
        const auto milliseconds = []( const clock_t::time_point &first, const clock_t::time_point &last ) {
            return std::chrono::duration<double, std::milli>( last - first ).count();
        };
        std::cout << "TAYLOR_GREEN_SNAPSHOT_READ_RESULT path=" << path
                  << " read_ms=" << milliseconds( begin, read )
                  << " broadcast_ms=" << milliseconds( read, broadcast )
                  << " unpack_ms=" << milliseconds( broadcast, finish )
                  << " bytes=" << expected_bytes << std::endl;
    }
}

int run_analysis(
    const fftm::examples::turbulence::spacetime_options &options,
    const scfd::communication::mpi_comm_info &comm, scfd::utils::log_mpi &log
)
{
    using strategy_3d_t = fftm::strategy_3d_slab_pencil<fftm::mpi_transpose_3d_mode::p2p_waitany>;
    using strategy_4d_t = fftm::strategy_4d_slab_slab_mpi<fftm::mpi_transpose_3d_mode::p2p_waitany>;
    using fftm_t = fftm::fftm<
        fft_backend_t, scfd::communication::mpi_comm_info, backend_t, strategy_3d_t, scfd::utils::log_mpi,
        strategy_4d_t>;
    using field4_t = typename fftm_t::template real_array_t<4>;
    using hat4_t   = typename fftm_t::template complex_array_t<4>;
    using snapshot_device_t =
        scfd::arrays::tensor_array_nd<float, 3, memory_t, scfd::arrays::custom_arranger_012_t>;
    using snapshot_host_t = scfd::arrays::tensor_array_nd<
        float, 3, typename memory_t::host_memory_type, scfd::arrays::custom_arranger_012_t>;
    using metric4_t =
        scfd::arrays::tensor_array_nd<real_t, 4, memory_t, scfd::arrays::custom_arranger_0213_t>;

    const auto records = load_snapshot_manifest( options );
    const double sample_dt = records.size() > 1 ? records[1].time - records[0].time : 1.0;

    fftm::processor_grid grid;
    grid.init( 1, static_cast<std::size_t>( comm.num_procs ), 1 );
    fftm::global_sizes sizes;
    sizes.init( options.spatial_size, options.spatial_size, options.spatial_size, options.frames );

    fftm_t fft( comm, log );
    fft.template init<4>( grid, sizes, production_4d_options() );

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm );
    partitioning.init( grid, sizes );
    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    const auto input_sizes     = fft.get_local_input_sizes_4d();
    const auto spectral_sizes  = fft.get_local_spectral_sizes_4d();
    const auto spectral_starts = fft.get_local_spectral_starts_4d();
    const auto &input_part     = fft.input_partition();

    const std::size_t local_nx = std::get<0>( input_sizes );
    const std::size_t local_ny = std::get<1>( input_sizes );
    const std::size_t local_nz = std::get<2>( input_sizes );
    const std::size_t start_x  = input_part.start_x[myid_i];
    const std::size_t start_y  = input_part.start_y[myid_j];
    const std::size_t start_z  = input_part.start_z[myid_k];

    field4_t field;
    hat4_t   spectrum_storage;
    field.init( local_nx, local_ny, local_nz, options.frames );
    spectrum_storage.init(
        std::get<0>( spectral_sizes ), std::get<1>( spectral_sizes ), std::get<2>( spectral_sizes ),
        std::get<3>( spectral_sizes )
    );

    snapshot_device_t snapshot_device;
    snapshot_host_t   snapshot_host;
    snapshot_device.init( local_nx, local_ny, local_nz );
    snapshot_host.init( local_nx, local_ny, local_nz );
    const std::size_t global_snapshot_count =
        options.spatial_size * options.spatial_size * options.spatial_size;
    std::vector<float> global_snapshot( global_snapshot_count );

    for_each_3d_t for_each_3d;
    for_each_4d_t for_each_4d;
    reduce_t      reduce;
    for_each_3d.block_size = 128;
    for_each_4d.block_size = 128;

    runtime_api_t::device_synchronize();
    scfd::utils::system_timer_event load_start, load_finish;
    load_start.record();
    for ( std::size_t frame = 0; frame < records.size(); ++frame )
    {
        const std::string path = options.input_dir + "/" + records[frame].file;
        read_local_snapshot(
            comm, path, options.spatial_size, start_x, start_y, start_z, snapshot_host,
            global_snapshot
        );
        memory_t::copy_from_host(
            snapshot_device.size() * sizeof( float ), static_cast<const void *>( snapshot_host.raw_ptr() ),
            static_cast<void *>( snapshot_device.raw_ptr() )
        );
        for_each_3d(
            fftm::examples::turbulence::insert_snapshot_functor<snapshot_device_t, field4_t>{
                snapshot_device, field, static_cast<int>( frame ) },
            range_3d( snapshot_device )
        );
        for_each_3d.wait();
        if ( comm.myid == 0 && ( frame + 1 == records.size() || ( frame + 1 ) % 10 == 0 ) )
            log.info_f( "SPACETIME_LOAD frames=%zu/%zu", frame + 1, records.size() );
    }

    real_t window_scale = real_t( 1 );
    if ( options.hann_window )
    {
        real_t square_sum = real_t( 0 );
        for ( std::size_t frame = 0; frame < options.frames; ++frame )
        {
            const real_t phase = real_t( 6.283185307179586476925286766559 ) * static_cast<real_t>( frame ) /
                                 static_cast<real_t>( options.frames - 1 );
            const real_t window = real_t( 0.5 ) * ( real_t( 1 ) - std::cos( phase ) );
            square_sum += window * window;
        }
        window_scale = std::sqrt( static_cast<real_t>( options.frames ) / square_sum );
    }
    for_each_3d(
        fftm::examples::turbulence::temporal_preprocess_functor<real_t, field4_t>{
            field, static_cast<int>( options.frames ), options.subtract_mean, options.hann_window, window_scale },
        rect3_t(
            index3_t( 0, 0, 0 ),
            index3_t( static_cast<int>( local_nx ), static_cast<int>( local_ny ), static_cast<int>( local_nz ) )
        )
    );
    for_each_3d.wait();
    runtime_api_t::device_synchronize();
    load_finish.record();
    const double load_ms = load_finish.elapsed_time( load_start );

    runtime_api_t::device_synchronize();
    scfd::utils::system_timer_event fft_start, fft_finish;
    fft_start.record();
    auto native_spectrum = fft.make_native_spectral_view_4d( spectrum_storage );
    fft.forward_native_spectral_4d( field, native_spectrum );
    runtime_api_t::device_synchronize();
    fft_finish.record();
    const double forward_ms = fft_finish.elapsed_time( fft_start );

    const std::size_t spectral_elements =
        std::get<0>( spectral_sizes ) * std::get<1>( spectral_sizes ) * std::get<2>( spectral_sizes ) *
        std::get<3>( spectral_sizes );
    if ( spectral_elements > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
        throw std::runtime_error( "Local 4D spectrum exceeds the current SCFD reduction index range" );
    metric4_t metric(
        field.raw_ptr(), std::get<0>( spectral_sizes ), std::get<1>( spectral_sizes ),
        std::get<2>( spectral_sizes ), std::get<3>( spectral_sizes )
    );

    const int start_native_x = static_cast<int>( std::get<0>( spectral_starts ) );
    const int start_native_z = static_cast<int>( std::get<1>( spectral_starts ) );
    const int start_native_w = static_cast<int>( std::get<2>( spectral_starts ) );
    const int start_native_y = static_cast<int>( std::get<3>( spectral_starts ) );

    for_each_4d(
        fftm::examples::turbulence::spectral_power_functor<real_t, decltype( native_spectrum ), metric4_t>{
            native_spectrum, metric, static_cast<int>( options.spatial_size ),
            static_cast<int>( options.spatial_size ), static_cast<int>( options.spatial_size ),
            static_cast<int>( options.frames ), start_native_x, start_native_z, start_native_w, start_native_y, 0,
            static_cast<int>( options.frames / 2 ), 0 },
        range_4d( native_spectrum )
    );
    for_each_4d.wait();
    const real_t total_power_local =
        reduce( static_cast<int>( spectral_elements ), metric.raw_ptr(), real_t( 0 ) );
    const real_t total_power = comm.all_reduce_sum( total_power_local );

    for_each_4d(
        fftm::examples::turbulence::spectral_power_functor<real_t, decltype( native_spectrum ), metric4_t>{
            native_spectrum, metric, static_cast<int>( options.spatial_size ),
            static_cast<int>( options.spatial_size ), static_cast<int>( options.spatial_size ),
            static_cast<int>( options.frames ), start_native_x, start_native_z, start_native_w, start_native_y,
            static_cast<int>( options.mode_min ), static_cast<int>( options.mode_max ),
            static_cast<int>( options.spatial_cutoff ) },
        range_4d( native_spectrum )
    );
    for_each_4d.wait();
    const real_t selected_power_local =
        reduce( static_cast<int>( spectral_elements ), metric.raw_ptr(), real_t( 0 ) );
    const real_t selected_power = comm.all_reduce_sum( selected_power_local );

    for_each_4d(
        fftm::examples::turbulence::spacetime_bandpass_functor<decltype( native_spectrum )>{
            native_spectrum, static_cast<int>( options.spatial_size ), static_cast<int>( options.spatial_size ),
            static_cast<int>( options.spatial_size ), start_native_x, start_native_z, start_native_w, start_native_y,
            static_cast<int>( options.mode_min ), static_cast<int>( options.mode_max ),
            static_cast<int>( options.spatial_cutoff ) },
        range_4d( native_spectrum )
    );
    for_each_4d.wait();

    runtime_api_t::device_synchronize();
    scfd::utils::system_timer_event inverse_start, inverse_finish;
    inverse_start.record();
    fft.backward_native_spectral_4d( native_spectrum, field );
    const real_t normalization =
        real_t( 1 ) /
        static_cast<real_t>(
            options.spatial_size * options.spatial_size * options.spatial_size * options.frames
        );
    for_each_4d(
        fftm::examples::turbulence::scale_spacetime_functor<real_t, field4_t>{ field, normalization },
        range_4d( field )
    );
    for_each_4d.wait();
    runtime_api_t::device_synchronize();
    inverse_finish.record();
    const double inverse_ms = inverse_finish.elapsed_time( inverse_start );

    fftm::examples::turbulence::snapshot_writer<memory_t> writer(
        comm, options.output_dir, options.spatial_size, options.spatial_size, options.spatial_size,
        options.spatial_size, start_x, start_y, local_nx, local_ny
    );
    if ( comm.myid == 0 )
    {
        std::ofstream selected_manifest( ( options.output_dir + "/selected_frames.csv" ).c_str() );
        if ( !selected_manifest )
            throw std::runtime_error( "Failed to create selected_frames.csv" );
        selected_manifest << "frame,step,time,field,file\n";
        for ( std::size_t frame = 0; frame < records.size(); ++frame )
        {
            selected_manifest << frame << ',' << records[frame].step << ',' << std::setprecision( 17 )
                              << records[frame].time << ',' << records[frame].field << ','
                              << records[frame].file << '\n';
        }
    }
    const std::string output_field =
        "filtered_" + options.field + "_m" + std::to_string( options.mode_min ) + "_" +
        std::to_string( options.mode_max );
    for ( std::size_t frame = 0; frame < options.frames; frame += options.write_every )
    {
        for_each_3d(
            fftm::examples::turbulence::extract_snapshot_functor<
                field4_t, typename fftm::examples::turbulence::snapshot_writer<memory_t>::device_array_t>{
                field, writer.device_array(), static_cast<int>( frame ) },
            range_3d( writer.device_array() )
        );
        for_each_3d.wait();
        writer.write( comm, output_field, records[frame].step, records[frame].time );
    }

    const real_t retained_fraction = total_power == real_t( 0 ) ? real_t( 0 ) : selected_power / total_power;
    if ( comm.myid == 0 )
    {
        const double fundamental_angular_frequency =
            6.283185307179586476925286766559 / ( static_cast<double>( options.frames ) * sample_dt );
        std::ofstream metadata( ( options.output_dir + "/analysis.json" ).c_str() );
        if ( !metadata )
            throw std::runtime_error( "Failed to create 4D analysis metadata" );
        metadata << "{\n"
                 << "  \"format\": \"fftm-taylor-green-spacetime-v1\",\n"
                 << "  \"input_field\": \"" << options.field << "\",\n"
                 << "  \"shape\": [" << options.spatial_size << ", " << options.spatial_size << ", "
                 << options.spatial_size << ", " << options.frames << "],\n"
                 << "  \"frame_offset\": " << options.frame_offset << ",\n"
                 << "  \"write_every\": " << options.write_every << ",\n"
                 << "  \"first_time\": " << std::setprecision( 17 ) << records.front().time << ",\n"
                 << "  \"last_time\": " << records.back().time << ",\n"
                 << "  \"sample_dt\": " << sample_dt << ",\n"
                 << "  \"temporal_mode_range\": [" << options.mode_min << ", " << options.mode_max << "],\n"
                 << "  \"angular_frequency_range\": ["
                 << fundamental_angular_frequency * static_cast<double>( options.mode_min ) << ", "
                 << fundamental_angular_frequency * static_cast<double>( options.mode_max ) << "],\n"
                 << "  \"spatial_cutoff\": " << options.spatial_cutoff << ",\n"
                 << "  \"subtract_mean\": " << ( options.subtract_mean ? "true" : "false" ) << ",\n"
                 << "  \"hann_window\": " << ( options.hann_window ? "true" : "false" ) << ",\n"
                 << "  \"retained_spectral_power_fraction\": " << static_cast<double>( retained_fraction ) << ",\n"
                 << "  \"load_preprocess_ms\": " << load_ms << ",\n"
                 << "  \"forward_4d_ms\": " << forward_ms << ",\n"
                 << "  \"inverse_4d_ms\": " << inverse_ms << "\n"
                 << "}\n";
        log.info_f(
            "SPACETIME_RESULT shape=%zux%zux%zux%zu ranks=%d modes=%zu:%zu retained=%.9g "
            "load_ms=%.9g forward_ms=%.9g inverse_ms=%.9g output=%s",
            options.spatial_size, options.spatial_size, options.spatial_size, options.frames, comm.num_procs,
            options.mode_min, options.mode_max, static_cast<double>( retained_fraction ), load_ms, forward_ms,
            inverse_ms, options.output_dir.c_str()
        );
    }
    return 0;
}

} // namespace

int main( int argc, char **argv )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    auto                          comm = mpi.comm_world();
    scfd::utils::log_mpi          log;
    try
    {
        fftm::device_backend::init_mpi( log, comm, 0, wrap_mpi_processes_over_gpus_enabled() );
        const auto options = fftm::examples::turbulence::parse_spacetime_options( argc, argv );
        return run_analysis( options, comm, log );
    }
    catch ( const std::exception &error )
    {
        log.error( scfd::utils::nested_exception_to_multistring( error ) );
        return 1;
    }
}
