#ifndef FFTM_EXAMPLES_TURBULENCE_SNAPSHOT_WRITER_H
#define FFTM_EXAMPLES_TURBULENCE_SNAPSHOT_WRITER_H

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include <scfd/arrays/tensor_array_nd.h>
#include <detail/array_arrangers.h>

namespace fftm
{
namespace examples
{
namespace turbulence
{

inline void create_directory_component( const std::string &path )
{
    if ( path.empty() )
        return;
    if ( ::mkdir( path.c_str(), 0775 ) == 0 || errno == EEXIST )
        return;
    throw std::runtime_error( "Failed to create directory '" + path + "': " + std::strerror( errno ) );
}

inline void create_directories( const std::string &path )
{
    std::string current;
    if ( !path.empty() && path[0] == '/' )
        current = "/";
    std::size_t begin = path.empty() || path[0] != '/' ? 0 : 1;
    while ( begin <= path.size() )
    {
        const std::size_t end = path.find( '/', begin );
        const std::string component = path.substr( begin, end == std::string::npos ? std::string::npos : end - begin );
        if ( !component.empty() )
        {
            if ( !current.empty() && current[current.size() - 1] != '/' )
                current += '/';
            current += component;
            create_directory_component( current );
        }
        if ( end == std::string::npos )
            break;
        begin = end + 1;
    }
}

template <class DeviceMemory>
class snapshot_writer
{
public:
    using device_array_t =
        scfd::arrays::tensor_array_nd<float, 3, DeviceMemory, scfd::arrays::custom_arranger_012_t>;
    using host_memory_t = typename DeviceMemory::host_memory_type;
    using host_array_t =
        scfd::arrays::tensor_array_nd<float, 3, host_memory_t, scfd::arrays::custom_arranger_012_t>;

    template <class Comm>
    snapshot_writer(
        const Comm &comm, const std::string &output_dir, std::size_t source_nx, std::size_t source_ny,
        std::size_t source_nz, std::size_t target_size, std::size_t source_start_x, std::size_t source_start_y,
        std::size_t source_local_nx, std::size_t source_local_ny
    )
        : output_dir_( output_dir ), target_size_( target_size )
    {
        if ( target_size_ == 0 )
            throw std::logic_error( "snapshot_writer requires a positive target size" );
        if ( source_nx % target_size_ != 0 || source_ny % target_size_ != 0 || source_nz % target_size_ != 0 )
            throw std::logic_error( "snapshot_writer requires exact integer decimation" );

        stride_x_ = static_cast<int>( source_nx / target_size_ );
        stride_y_ = static_cast<int>( source_ny / target_size_ );
        stride_z_ = static_cast<int>( source_nz / target_size_ );

        const std::size_t first_x = first_selected_index_( source_start_x, stride_x_ );
        const std::size_t first_y = first_selected_index_( source_start_y, stride_y_ );
        source_offset_x_ = static_cast<int>( first_x - source_start_x );
        source_offset_y_ = static_cast<int>( first_y - source_start_y );
        source_offset_z_ = 0;
        output_start_x_  = first_x / static_cast<std::size_t>( stride_x_ );
        output_start_y_  = first_y / static_cast<std::size_t>( stride_y_ );
        output_start_z_  = 0;
        output_size_x_   = selected_count_( source_start_x, source_local_nx, stride_x_ );
        output_size_y_   = selected_count_( source_start_y, source_local_ny, stride_y_ );
        output_size_z_   = target_size_;

        device_.init( output_size_x_, output_size_y_, output_size_z_ );
        host_.init( output_size_x_, output_size_y_, output_size_z_ );

        if ( comm.myid == 0 )
            create_directories( output_dir_ );
        comm.barrier();

        initialize_io_layout_( comm );
        write_layout_metadata_( comm, source_nx, source_ny, source_nz );
    }

    device_array_t &device_array()
    {
        return device_;
    }

    int source_offset_x() const
    {
        return source_offset_x_;
    }

    int source_offset_y() const
    {
        return source_offset_y_;
    }

    int source_offset_z() const
    {
        return source_offset_z_;
    }

    int stride_x() const
    {
        return stride_x_;
    }

    int stride_y() const
    {
        return stride_y_;
    }

    int stride_z() const
    {
        return stride_z_;
    }

    std::size_t output_size_x() const
    {
        return output_size_x_;
    }

    std::size_t output_size_y() const
    {
        return output_size_y_;
    }

    std::size_t output_size_z() const
    {
        return output_size_z_;
    }

    template <class Comm>
    void write( const Comm &comm, const std::string &field, std::size_t step, double time )
    {
        using clock_t = std::chrono::steady_clock;
        const auto begin = clock_t::now();
        if ( comm.myid == 0 )
        {
            std::cout << "TAYLOR_GREEN_SNAPSHOT_BEGIN field=" << field << " step=" << step
                      << " time=" << std::setprecision( 17 ) << time << " target=" << target_size_ << "^3"
                      << std::endl;
        }

        DeviceMemory::copy_to_host(
            device_.size() * sizeof( float ), static_cast<const void *>( device_.raw_ptr() ),
            static_cast<void *>( host_.raw_ptr() )
        );
        const auto copied = clock_t::now();

        comm.gatherv(
            host_.raw_ptr(), static_cast<int>( host_.size() ),
            comm.myid == 0 ? gathered_.data() : static_cast<float *>( nullptr ), gather_counts_.data(),
            gather_displacements_.data(), 0
        );
        const auto gathered = clock_t::now();

        const std::string stem = snapshot_stem_( field, step );
        const std::string raw_name = stem + ".raw";
        const std::string raw_path = output_dir_ + "/" + raw_name;

        int         write_ok = 1;
        std::string write_error;
        if ( comm.myid == 0 )
        {
            try
            {
                assemble_global_snapshot_();
                write_raw_atomically_( raw_path );
            }
            catch ( const std::exception &error )
            {
                write_ok    = 0;
                write_error = error.what();
                std::cerr << "TAYLOR_GREEN_SNAPSHOT_ERROR field=" << field << " step=" << step
                          << " phase=commit message=\"" << write_error << '"' << std::endl;
            }
        }
        comm.bcast( &write_ok, 1, 0 );
        if ( !write_ok )
        {
            throw std::runtime_error(
                comm.myid == 0 ? write_error : "Snapshot commit failed on rank 0"
            );
        }
        const auto committed = clock_t::now();

        int metadata_ok = 1;
        if ( comm.myid == 0 )
        {
            try
            {
                write_xdmf_( stem, raw_name, field, time );
                std::ofstream times(
                    ( output_dir_ + "/snapshots.csv" ).c_str(), std::ios::out | std::ios::app
                );
                if ( !times )
                    throw std::runtime_error( "Failed to append snapshot manifest" );
                times << step << ',' << std::setprecision( 17 ) << time << ',' << field << ',' << raw_name
                      << '\n';
                times.close();
                if ( !times )
                    throw std::runtime_error( "Failed to commit snapshot manifest" );
            }
            catch ( const std::exception &error )
            {
                metadata_ok = 0;
                std::cerr << "TAYLOR_GREEN_SNAPSHOT_ERROR field=" << field << " step=" << step
                          << " phase=metadata message=\"" << error.what() << '"' << std::endl;
            }
        }
        comm.bcast( &metadata_ok, 1, 0 );
        if ( !metadata_ok )
            throw std::runtime_error( "Snapshot metadata commit failed on rank 0" );
        comm.barrier();

        if ( comm.myid == 0 )
        {
            const auto finish = clock_t::now();
            const auto milliseconds = []( const clock_t::time_point &first, const clock_t::time_point &last ) {
                return std::chrono::duration<double, std::milli>( last - first ).count();
            };
            std::cout << "TAYLOR_GREEN_SNAPSHOT_RESULT field=" << field << " step=" << step
                      << " time=" << std::setprecision( 17 ) << time
                      << " copy_ms=" << milliseconds( begin, copied )
                      << " gather_ms=" << milliseconds( copied, gathered )
                      << " commit_ms=" << milliseconds( gathered, committed )
                      << " total_ms=" << milliseconds( begin, finish )
                      << " bytes=" << global_.size() * sizeof( float ) << std::endl;
        }
    }

private:
    template <class Comm>
    void initialize_io_layout_( const Comm &comm )
    {
        long long local[6] = {
            static_cast<long long>( output_start_x_ ), static_cast<long long>( output_start_y_ ),
            static_cast<long long>( output_start_z_ ), static_cast<long long>( output_size_x_ ),
            static_cast<long long>( output_size_y_ ), static_cast<long long>( output_size_z_ )
        };
        pieces_.resize( static_cast<std::size_t>( comm.num_procs ) * 6 );
        comm.all_gather( local, 6, pieces_.data(), 6 );

        gather_counts_.resize( static_cast<std::size_t>( comm.num_procs ) );
        gather_displacements_.resize( static_cast<std::size_t>( comm.num_procs ) );
        std::size_t gathered_count = 0;
        for ( int rank = 0; rank < comm.num_procs; ++rank )
        {
            const std::size_t offset = static_cast<std::size_t>( rank ) * 6;
            const std::size_t count =
                static_cast<std::size_t>( pieces_[offset + 3] ) *
                static_cast<std::size_t>( pieces_[offset + 4] ) *
                static_cast<std::size_t>( pieces_[offset + 5] );
            if ( count > static_cast<std::size_t>( std::numeric_limits<int>::max() ) ||
                 gathered_count > static_cast<std::size_t>( std::numeric_limits<int>::max() ) - count )
            {
                throw std::overflow_error(
                    "Snapshot root-gather count exceeds the MPI int-count limit"
                );
            }
            gather_counts_[rank]        = static_cast<int>( count );
            gather_displacements_[rank] = static_cast<int>( gathered_count );
            gathered_count += count;
        }

        const std::size_t global_count = target_size_ * target_size_ * target_size_;
        if ( gathered_count != global_count )
            throw std::logic_error( "Snapshot pieces do not cover the requested target volume" );
        if ( static_cast<std::size_t>( host_.size() ) !=
             static_cast<std::size_t>( gather_counts_[comm.myid] ) )
        {
            throw std::logic_error( "Snapshot local buffer and gathered piece sizes differ" );
        }
        if ( comm.myid == 0 )
        {
            gathered_.resize( gathered_count );
            global_.resize( global_count );
        }
    }

    void assemble_global_snapshot_()
    {
        for ( std::size_t rank = 0; rank < gather_counts_.size(); ++rank )
        {
            const std::size_t offset = rank * 6;
            const std::size_t start_x = static_cast<std::size_t>( pieces_[offset] );
            const std::size_t start_y = static_cast<std::size_t>( pieces_[offset + 1] );
            const std::size_t start_z = static_cast<std::size_t>( pieces_[offset + 2] );
            const std::size_t size_x  = static_cast<std::size_t>( pieces_[offset + 3] );
            const std::size_t size_y  = static_cast<std::size_t>( pieces_[offset + 4] );
            const std::size_t size_z  = static_cast<std::size_t>( pieces_[offset + 5] );
            const std::size_t source_base =
                static_cast<std::size_t>( gather_displacements_[rank] );

            for ( std::size_t z = 0; z < size_z; ++z )
            {
                for ( std::size_t y = 0; y < size_y; ++y )
                {
                    const std::size_t source = source_base + size_x * ( y + size_y * z );
                    const std::size_t destination =
                        start_x + target_size_ *
                                      ( start_y + y + target_size_ * ( start_z + z ) );
                    std::copy(
                        gathered_.begin() + static_cast<std::ptrdiff_t>( source ),
                        gathered_.begin() + static_cast<std::ptrdiff_t>( source + size_x ),
                        global_.begin() + static_cast<std::ptrdiff_t>( destination )
                    );
                }
            }
        }
    }

    void write_raw_atomically_( const std::string &raw_path )
    {
        const std::string temporary_path = raw_path + ".partial";
        {
            std::ofstream output(
                temporary_path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc
            );
            if ( !output )
                throw std::runtime_error( "Failed to open temporary snapshot: " + temporary_path );
            output.write(
                reinterpret_cast<const char *>( global_.data() ),
                static_cast<std::streamsize>( global_.size() * sizeof( float ) )
            );
            output.close();
            if ( !output )
                throw std::runtime_error( "Failed to write temporary snapshot: " + temporary_path );
        }
        if ( std::rename( temporary_path.c_str(), raw_path.c_str() ) != 0 )
        {
            throw std::runtime_error(
                "Failed to rename temporary snapshot '" + temporary_path + "' to '" + raw_path +
                "': " + std::strerror( errno )
            );
        }
    }

    static std::size_t first_selected_index_( std::size_t start, int stride )
    {
        const std::size_t stride_value = static_cast<std::size_t>( stride );
        return ( ( start + stride_value - 1 ) / stride_value ) * stride_value;
    }

    static std::size_t selected_count_( std::size_t start, std::size_t count, int stride )
    {
        if ( count == 0 )
            return 0;
        const std::size_t first = first_selected_index_( start, stride );
        const std::size_t end   = start + count;
        if ( first >= end )
            return 0;
        return 1 + ( end - 1 - first ) / static_cast<std::size_t>( stride );
    }

    static std::string snapshot_stem_( const std::string &field, std::size_t step )
    {
        std::ostringstream stream;
        stream << field << "_s" << std::setw( 8 ) << std::setfill( '0' ) << step;
        return stream.str();
    }

    template <class Comm>
    void write_layout_metadata_(
        const Comm &comm, std::size_t source_nx, std::size_t source_ny, std::size_t source_nz
    )
    {
        if ( comm.myid != 0 )
            return;

        std::ofstream metadata( ( output_dir_ + "/layout.json" ).c_str() );
        if ( !metadata )
            throw std::runtime_error( "Failed to create snapshot layout metadata" );
        metadata << "{\n"
                 << "  \"format\": \"fftm-taylor-green-snapshot-v1\",\n"
                 << "  \"source_shape\": [" << source_nx << ", " << source_ny << ", " << source_nz << "],\n"
                 << "  \"snapshot_shape\": [" << target_size_ << ", " << target_size_ << ", " << target_size_
                 << "],\n"
                 << "  \"storage_order\": \"x-fastest\",\n"
                 << "  \"scalar_type\": \"float32\",\n"
                 << "  \"domain\": [0.0, 6.283185307179586, 0.0, 6.283185307179586, 0.0, "
                    "6.283185307179586],\n"
                 << "  \"low_pass_cutoff\": " << target_size_ / 3 << ",\n"
                 << "  \"pieces\": [\n";
        for ( int rank = 0; rank < comm.num_procs; ++rank )
        {
            const std::size_t offset = static_cast<std::size_t>( rank ) * 6;
            metadata << "    {\"rank\": " << rank << ", \"start\": [" << pieces_[offset] << ", "
                     << pieces_[offset + 1] << ", " << pieces_[offset + 2] << "], \"size\": ["
                     << pieces_[offset + 3] << ", " << pieces_[offset + 4] << ", "
                     << pieces_[offset + 5] << "]}";
            metadata << ( rank + 1 == comm.num_procs ? "\n" : ",\n" );
        }
        metadata << "  ]\n}\n";

        std::ofstream times( ( output_dir_ + "/snapshots.csv" ).c_str(), std::ios::out | std::ios::trunc );
        if ( !times )
            throw std::runtime_error( "Failed to create snapshot manifest" );
        times << "step,time,field,file\n";
    }

    void write_xdmf_(
        const std::string &stem, const std::string &raw_name, const std::string &field, double time
    ) const
    {
        const double spacing = 6.283185307179586476925286766559 / static_cast<double>( target_size_ );
        std::ofstream xdmf( ( output_dir_ + "/" + stem + ".xdmf" ).c_str() );
        if ( !xdmf )
            throw std::runtime_error( "Failed to create XDMF snapshot metadata" );
        xdmf << "<?xml version=\"1.0\" ?>\n"
             << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
             << "<Xdmf Version=\"3.0\">\n"
             << "  <Domain>\n"
             << "    <Grid Name=\"" << field << "\" GridType=\"Uniform\">\n"
             << "      <Time Value=\"" << std::setprecision( 17 ) << time << "\"/>\n"
             << "      <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"" << target_size_ << ' '
             << target_size_ << ' ' << target_size_ << "\"/>\n"
             << "      <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
             << "        <DataItem Dimensions=\"3\" Format=\"XML\">0 0 0</DataItem>\n"
             << "        <DataItem Dimensions=\"3\" Format=\"XML\">" << spacing << ' ' << spacing << ' '
             << spacing << "</DataItem>\n"
             << "      </Geometry>\n"
             << "      <Attribute Name=\"" << field << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
             << "        <DataItem Dimensions=\"" << target_size_ << ' ' << target_size_ << ' ' << target_size_
             << "\" NumberType=\"Float\" Precision=\"4\" Endian=\"Little\" Format=\"Binary\">"
             << raw_name << "</DataItem>\n"
             << "      </Attribute>\n"
             << "    </Grid>\n"
             << "  </Domain>\n"
             << "</Xdmf>\n";
    }

    std::string output_dir_;
    std::size_t target_size_ = 0;
    int         stride_x_ = 1;
    int         stride_y_ = 1;
    int         stride_z_ = 1;
    int         source_offset_x_ = 0;
    int         source_offset_y_ = 0;
    int         source_offset_z_ = 0;
    std::size_t output_start_x_ = 0;
    std::size_t output_start_y_ = 0;
    std::size_t output_start_z_ = 0;
    std::size_t output_size_x_ = 0;
    std::size_t output_size_y_ = 0;
    std::size_t output_size_z_ = 0;
    device_array_t device_;
    host_array_t   host_;
    std::vector<long long> pieces_;
    std::vector<int>       gather_counts_;
    std::vector<int>       gather_displacements_;
    std::vector<float>     gathered_;
    std::vector<float>     global_;
};

} // namespace turbulence
} // namespace examples
} // namespace fftm

#endif
