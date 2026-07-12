#ifndef __FFTM_TESTS_DETAIL_FFTM_3D_NATIVE_PENCIL_SCHEDULE_CHECK_H__
#define __FFTM_TESTS_DETAIL_FFTM_3D_NATIVE_PENCIL_SCHEDULE_CHECK_H__

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <scfd/communication/mpi_comm_info.h>
#include <scfd/utils/log_mpi.h>

#include <detail/mpi_transpose_3d_pencil_pencil_plan_state.h>
#include <external_wrap/cufft_wrap_many.h>

#include "fft_benchmark_common.h"
#include "fftm_3d_test_options.h"

namespace fftm
{
namespace test
{
namespace detail
{
namespace native_pencil_schedule
{

inline const char *transpose_name( long long value )
{
    return value == 1 ? "first" : "second";
}

inline const char *direction_name( long long value )
{
    return value == 0 ? "forward" : "backward";
}

inline const char *operation_name( long long value )
{
    return value == 0 ? "recv" : "send";
}

inline std::vector<std::string> split_csv_line( const std::string &line )
{
    std::vector<std::string> out;
    std::string              value;
    bool                     quoted = false;
    for ( std::size_t i = 0; i < line.size(); ++i )
    {
        const char ch = line[i];
        if ( ch == '"' )
        {
            if ( quoted && i + 1 < line.size() && line[i + 1] == '"' )
            {
                value.push_back( '"' );
                ++i;
            }
            else
            {
                quoted = !quoted;
            }
        }
        else if ( ch == ',' && !quoted )
        {
            out.push_back( value );
            value.clear();
        }
        else
        {
            value.push_back( ch );
        }
    }
    out.push_back( value );
    return out;
}

using csv_row_t = std::map<std::string, std::string>;

inline std::vector<csv_row_t> read_csv_rows( const std::string &path )
{
    std::ifstream in( path.c_str() );
    if ( !in )
        throw std::runtime_error( "cannot open CSV reference '" + path + "'" );

    std::string line;
    if ( !std::getline( in, line ) )
        return std::vector<csv_row_t>();

    const std::vector<std::string> header = split_csv_line( line );
    std::vector<csv_row_t>         rows;
    while ( std::getline( in, line ) )
    {
        if ( line.empty() )
            continue;
        const std::vector<std::string> values = split_csv_line( line );
        csv_row_t                      row;
        for ( std::size_t i = 0; i < header.size() && i < values.size(); ++i )
            row[header[i]] = values[i];
        rows.push_back( row );
    }
    return rows;
}

inline std::string csv_get( const csv_row_t &row, const std::string &key )
{
    const csv_row_t::const_iterator it = row.find( key );
    if ( it == row.end() )
        return std::string();
    return it->second;
}

inline bool path_exists_for_check( const std::string &path )
{
    return path_exists( path );
}

inline std::string reference_path( const std::string &reference_dir, const std::string &filename )
{
    return join_path( reference_dir, filename );
}

inline std::uint32_t native_pencil_fnv1a32( const std::string &value )
{
    std::uint32_t hash = 2166136261u;
    for ( char ch : value )
    {
        hash ^= static_cast<unsigned char>( ch );
        hash *= 16777619u;
    }
    return hash;
}

inline std::string native_pencil_hex8( std::uint32_t value )
{
    static const char digits[] = "0123456789abcdef";
    std::string       out( 8, '0' );
    for ( int i = 7; i >= 0; --i )
    {
        out[static_cast<std::size_t>( i )] = digits[value & 0x0fu];
        value >>= 4;
    }
    return out;
}

template <class Options>
inline std::string native_pencil_csv_suffix( const Options &options )
{
    std::ostringstream key;
    key << options.p1 << 'x' << options.p2 << '|' << pencil_layout_name( options.pencil_layout ) << '|'
        << options.nx << 'x' << options.ny << 'x' << options.nz;
    std::ostringstream ss;
    ss << "_g" << ( options.p1 * options.p2 ) << "p" << options.p1 << 'x' << options.p2 << '_'
       << pencil_layout_name( options.pencil_layout ) << "_h" << native_pencil_hex8( native_pencil_fnv1a32( key.str() ) )
       << ".csv";
    return ss.str();
}

template <class Writer>
inline void write_native_csv_file( const std::string &path, const std::string &label, Writer writer )
{
    std::ofstream out( path.c_str(), std::ios::out );
    if ( !out )
        throw std::runtime_error( "cannot open " + label + " CSV: " + path );
    writer( out );
}

template <class Options, class Writer>
inline void write_native_csv_aliases(
    const Options &options, const std::string &generic_filename, const std::string &unique_stem,
    const std::string &label, Writer writer
)
{
    ensure_directory_exists( options.directory );
    const std::string generic_path = join_path( options.directory, generic_filename );
    const std::string unique_path  = join_path( options.directory, unique_stem + native_pencil_csv_suffix( options ) );
    write_native_csv_file( generic_path, label, writer );
    if ( unique_path != generic_path )
        write_native_csv_file( unique_path, label, writer );
}

template <class Options>
inline void write_plan_csv( const Options &options, const std::vector<long long> &gathered, int world_size )
{
    write_native_csv_aliases(
        options, "native_pencil_plan.csv", "native_pencil_plan", "native pencil plan",
        [&]( std::ostream &out ) {
            out << "benchmark,num_gpus,opt,p1,p2,nx,ny,nz,rank,pidx_i,pidx_j,"
                << "input_start_x,input_size_x,input_start_y,input_size_y,input_size_z,"
                << "stage1_start_x,stage1_size_x,stage1_size_y,stage1_start_z,stage1_size_z,"
                << "output_size_x,output_start_y,output_size_y,output_start_z,output_size_z,directory\n";

            const int values_per_record = 18;
            for ( int rank = 0; rank < world_size; ++rank )
            {
                const long long *row = &gathered[static_cast<std::size_t>( rank ) * values_per_record];
                out << "fftm-native-pencil-plan," << world_size << ','
                    << pencil_layout_name( options.pencil_layout ) << ',' << options.p1 << ',' << options.p2 << ','
                    << options.nx << ',' << options.ny << ',' << options.nz;
                for ( int i = 0; i < values_per_record; ++i )
                    out << ',' << row[i];
                out << ',' << options.directory << '\n';
            }
        }
    );
}

template <class Options>
inline void write_schedule_csv( const Options &options, const std::vector<long long> &gathered, int records_per_rank,
                                int world_size )
{
    write_native_csv_aliases(
        options, "native_pencil_schedule.csv", "native_pencil_schedule", "native pencil schedule",
        [&]( std::ostream &out ) {
            out << "benchmark,num_gpus,opt,p1,p2,rank,pidx_i,pidx_j,transpose,direction,operation,order_index,peer,"
                << "offset_elems,bytes,mpi_peer,mpi_tag,directory\n";

            const int values_per_record = 12;
            for ( int rank = 0; rank < world_size; ++rank )
            {
                for ( int rec = 0; rec < records_per_rank; ++rec )
                {
                    const long long *row =
                        &gathered[( static_cast<std::size_t>( rank ) * records_per_rank + rec ) *
                                  values_per_record];
                    out << "fftm-native-pencil-plan," << world_size << ','
                        << pencil_layout_name( options.pencil_layout ) << ',' << options.p1 << ',' << options.p2
                        << ',' << row[0] << ',' << row[1] << ',' << row[2] << ',' << transpose_name( row[3] )
                        << ',' << direction_name( row[4] ) << ',' << operation_name( row[5] ) << ',' << row[6]
                        << ',' << row[7] << ',' << row[8] << ',' << row[9] << ',' << row[10] << ',' << row[11]
                        << ',' << options.directory << '\n';
                }
            }
        }
    );
}

template <class Options>
inline void write_rank_device_map_csv(
    const Options &options, const std::vector<long long> &gathered, const std::vector<char> &gathered_bus,
    int world_size
)
{
    write_native_csv_aliases(
        options, "native_rank_device_map.csv", "native_rank_device_map", "native rank-device map",
        [&]( std::ostream &out ) {
            out << "benchmark,num_gpus,p1,p2,opt,rank,pidx_i,pidx_j,selected_device,visible_device_count,pci_bus_id,"
                << "directory\n";
            for ( int rank = 0; rank < world_size; ++rank )
            {
                const long long *row = &gathered[static_cast<std::size_t>( rank ) * 5];
                const char      *bus = &gathered_bus[static_cast<std::size_t>( rank ) * 64];
                out << "fftm-native-pencil-plan," << world_size << ',' << options.p1 << ',' << options.p2 << ','
                    << pencil_layout_name( options.pencil_layout ) << ',' << row[0] << ',' << row[1] << ','
                    << row[2] << ',' << row[3] << ',' << row[4] << ',' << bus << ',' << options.directory << '\n';
            }
        }
    );
}

inline std::string projected_schedule_row( const csv_row_t &row )
{
    static const char *keys[] = {
        "rank",      "pidx_i",      "pidx_j", "transpose", "direction", "operation",
        "order_index", "peer",      "offset_elems", "bytes", "mpi_peer", "mpi_tag"
    };
    std::ostringstream ss;
    for ( std::size_t i = 0; i < sizeof( keys ) / sizeof( keys[0] ); ++i )
    {
        if ( i != 0 )
            ss << ',';
        ss << csv_get( row, keys[i] );
    }
    return ss.str();
}

inline std::string projected_rank_device_row( const csv_row_t &row )
{
    static const char *keys[] = {
        "rank", "pidx_i", "pidx_j", "selected_device", "visible_device_count", "pci_bus_id"
    };
    std::ostringstream ss;
    for ( std::size_t i = 0; i < sizeof( keys ) / sizeof( keys[0] ); ++i )
    {
        if ( i != 0 )
            ss << ',';
        ss << csv_get( row, keys[i] );
    }
    return ss.str();
}

inline std::string projected_plan_row( const csv_row_t &row )
{
    static const char *keys[] = {
        "rank",
        "pidx_i",
        "pidx_j",
        "input_start_x",
        "input_size_x",
        "input_start_y",
        "input_size_y",
        "input_size_z",
        "stage1_start_x",
        "stage1_size_x",
        "stage1_size_y",
        "stage1_start_z",
        "stage1_size_z",
        "output_size_x",
        "output_start_y",
        "output_size_y",
        "output_start_z",
        "output_size_z"
    };
    std::ostringstream ss;
    for ( std::size_t i = 0; i < sizeof( keys ) / sizeof( keys[0] ); ++i )
    {
        if ( i != 0 )
            ss << ',';
        ss << csv_get( row, keys[i] );
    }
    return ss.str();
}

inline bool compare_projected_csv(
    const std::string &label,
    const std::vector<csv_row_t> &reference_rows,
    const std::vector<csv_row_t> &native_rows,
    std::string ( *project )( const csv_row_t & ),
    std::ostringstream &report
)
{
    bool ok = true;
    if ( reference_rows.size() != native_rows.size() )
    {
        report << label << ": row count mismatch reference=" << reference_rows.size()
               << " native=" << native_rows.size() << '\n';
        ok = false;
    }

    const std::size_t count = std::min( reference_rows.size(), native_rows.size() );
    int               mismatches = 0;
    for ( std::size_t i = 0; i < count; ++i )
    {
        const std::string ref = project( reference_rows[i] );
        const std::string got = project( native_rows[i] );
        if ( ref == got )
            continue;
        if ( mismatches < 20 )
        {
            report << label << ": row " << i << " mismatch\n"
                   << "  reference: " << ref << '\n'
                   << "  native:    " << got << '\n';
        }
        ++mismatches;
        ok = false;
    }
    if ( mismatches > 20 )
        report << label << ": " << ( mismatches - 20 ) << " additional mismatches suppressed\n";
    return ok;
}

inline std::vector<csv_row_t> read_optional_plan_reference( const std::string &reference_dir )
{
    const std::string native_path = reference_path( reference_dir, "native_pencil_plan.csv" );
    if ( path_exists_for_check( native_path ) )
        return read_csv_rows( native_path );
    const std::string generic_path = reference_path( reference_dir, "pencil_plan.csv" );
    if ( path_exists_for_check( generic_path ) )
        return read_csv_rows( generic_path );
    return std::vector<csv_row_t>();
}

inline ::fftm::detail::pencil_pencil_plan_layout plan_layout_from_options( const fftm_3d_pencil_layout_kind layout )
{
    if ( layout == fftm_3d_pencil_layout_kind::opt1 )
        return ::fftm::detail::pencil_pencil_plan_layout::opt1;
    if ( layout == fftm_3d_pencil_layout_kind::opt0 )
        return ::fftm::detail::pencil_pencil_plan_layout::opt0;
    throw std::logic_error( "native pencil schedule check requires explicit --pencil-layout opt0 or opt1" );
}

template <class Options>
inline int run( scfd::utils::log_mpi &log, const Options &options,
                const scfd::communication::mpi_comm_info &comm_info )
{
    if ( !options.write_native_pencil_schedule && !options.check_native_pencil_schedule )
        return 0;

    if ( options.strategy != fftm_3d_strategy_kind::pencil_pencil )
        throw std::logic_error( "native pencil schedule check requires --strategy pencil-pencil" );
    if ( options.pencil_layout == fftm_3d_pencil_layout_kind::legacy ||
         options.pencil_layout == fftm_3d_pencil_layout_kind::auto_select )
        throw std::logic_error( "native pencil schedule check requires explicit --pencil-layout opt0 or opt1" );
    if ( options.p1 * options.p2 != static_cast<std::size_t>( comm_info.num_procs ) )
        throw std::logic_error( "native pencil schedule check requires P1*P2 == MPI size" );

    processor_grid grid;
    grid.init( options.p1, options.p2 );
    global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );

    ::fftm::detail::mpi_transpose_3d_pencil_pencil_plan_state plan_state;
    plan_state.init(
        grid, sizes, comm_info.myid, comm_info.num_procs, plan_layout_from_options( options.pencil_layout ),
        sizeof( typename ::fftm::wrap::cufft_wrap_many<double>::complex )
    );

    const std::vector<long long> local_plan =
        ::fftm::detail::flatten_pencil_plan_record( plan_state.local_plan_record() );
    const int plan_values = static_cast<int>( local_plan.size() );
    std::vector<long long> gathered_plan( static_cast<std::size_t>( comm_info.num_procs ) * plan_values, 0 );
    comm_info.all_gather( local_plan.data(), plan_values, gathered_plan.data(), plan_values );

    const std::vector<long long> local_schedule =
        ::fftm::detail::flatten_pencil_schedule_records( plan_state.local_schedule_records() );
    const int schedule_values = static_cast<int>( local_schedule.size() );
    std::vector<long long> gathered_schedule(
        static_cast<std::size_t>( comm_info.num_procs ) * schedule_values, 0
    );
    comm_info.all_gather( local_schedule.data(), schedule_values, gathered_schedule.data(), schedule_values );

    const int selected_device = ::fftm::wrap::cuda_runtime_api::get_device();
    int       visible_device_count = 0;
    cudaGetDeviceCount( &visible_device_count );
    char pci_bus_id[64] = "n/a";
    if ( selected_device >= 0 )
        cudaDeviceGetPCIBusId( pci_bus_id, sizeof( pci_bus_id ), selected_device );

    const long long local_rank_device[5] = {
        comm_info.myid, plan_state.rank_i(), plan_state.rank_j(), selected_device, visible_device_count
    };
    std::vector<long long> gathered_rank_device( static_cast<std::size_t>( comm_info.num_procs ) * 5, 0 );
    comm_info.all_gather( local_rank_device, 5, gathered_rank_device.data(), 5 );

    std::vector<char> gathered_bus( static_cast<std::size_t>( comm_info.num_procs ) * 64, '\0' );
    comm_info.all_gather( pci_bus_id, 64, gathered_bus.data(), 64 );

    int local_failure = 0;
    if ( comm_info.myid == 0 )
    {
        const int records_per_rank = schedule_values / 12;
        write_plan_csv( options, gathered_plan, comm_info.num_procs );
        write_schedule_csv( options, gathered_schedule, records_per_rank, comm_info.num_procs );
        write_rank_device_map_csv( options, gathered_rank_device, gathered_bus, comm_info.num_procs );

        std::ostringstream report;
        bool               ok = true;
        if ( options.check_native_pencil_schedule )
        {
            if ( options.native_pencil_reference_dir.empty() )
                throw std::logic_error(
                    "--check-native-pencil-reference-dir requires a non-empty reference directory"
                );

            const std::string reference_schedule = reference_path( options.native_pencil_reference_dir,
                                                                   "pencil_schedule.csv" );
            const std::string reference_rank_map = reference_path( options.native_pencil_reference_dir,
                                                                   "rank_device_map.csv" );

            ok &= compare_projected_csv(
                "schedule", read_csv_rows( reference_schedule ),
                read_csv_rows( join_path( options.directory, "native_pencil_schedule.csv" ) ),
                projected_schedule_row, report
            );
            if ( options.skip_native_pencil_rank_device_check )
            {
                report << "rank_device_map: skipped by --skip-native-pencil-rank-device-check\n";
            }
            else
            {
                ok &= compare_projected_csv(
                    "rank_device_map", read_csv_rows( reference_rank_map ),
                    read_csv_rows( join_path( options.directory, "native_rank_device_map.csv" ) ),
                    projected_rank_device_row, report
                );
            }

            const std::vector<csv_row_t> reference_plan = read_optional_plan_reference(
                options.native_pencil_reference_dir
            );
            if ( !reference_plan.empty() )
            {
                ok &= compare_projected_csv(
                    "plan", reference_plan, read_csv_rows( join_path( options.directory, "native_pencil_plan.csv" ) ),
                    projected_plan_row, report
                );
            }
            else
            {
                report << "plan: no optional reference plan CSV found; wrote native_pencil_plan.csv only\n";
            }

            if ( ok )
            {
                log.info(
                    "native_pencil_schedule_check status=pass step=3 reference_dir=" +
                    options.native_pencil_reference_dir
                );
            }
            else
            {
                log.error(
                    "native_pencil_schedule_check status=fail step=3 reference_dir=" +
                    options.native_pencil_reference_dir + "\n" + report.str()
                );
                local_failure = 1;
            }
        }
        else
        {
            log.info( "native_pencil_schedule_check status=written step=3" );
        }
    }

    const int global_failure = comm_info.all_reduce_max( local_failure );
    return global_failure == 0 ? 0 : 1;
}

} // namespace native_pencil_schedule
} // namespace detail
} // namespace test
} // namespace fftm

#endif
