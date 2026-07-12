#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <cufft.h>
#include <cuda_runtime.h>

#include <scfd/backend/cuda.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda_mpi.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/system_timer_event.h>

#include <external_wrap/cufft_wrap_many.h>
#include <fftm.hpp>

#include "detail/fft_benchmark_common.h"
#include "detail/fft_benchmark_options.h"
#include "detail/fftm_3d_native_pencil_schedule_check.h"
#include "detail/mpi_cuda_test_init.h"
#include "detail/test_memory_profile_helpers.h"

#if defined(FFTM_ENABLE_FFTM3D_SCFD_FFT_BACKEND)
#include <fftm3d_benchmark_3d_api.hpp>
#endif

namespace
{

using T             = double;
using base_fft_t    = fftm::wrap::cufft_wrap_many<T>;
using runtime_api_t = typename base_fft_t::runtime_api;
using backend_t     = scfd::backend::cuda;
using memory_t      = backend_t::memory_type;
using reduce_t      = backend_t::reduce_type;
using for_each_t    = backend_t::template for_each_nd_type<3, int>;
using idx_t         = scfd::static_vec::vec<int, 3>;
using rect_t        = scfd::static_vec::rect<int, 3>;
using options_t     = fftm::test::detail::fftm_3d_benchmark_options<T>;
using strategy_kind = fftm::test::detail::fftm_3d_strategy_kind;

const char *stage_timer_strategy_name( strategy_kind strategy );

std::string to_arg( std::size_t value )
{
    return std::to_string( static_cast<unsigned long long>( value ) );
}

std::string to_arg( int value )
{
    return std::to_string( value );
}

std::string getenv_or_empty( const char *name )
{
    const char *value = std::getenv( name );
    return value == nullptr ? std::string() : std::string( value );
}

bool env_truthy( const char *name )
{
    const std::string value = getenv_or_empty( name );
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES" ||
           value == "on" || value == "ON";
}

std::string trim_copy( const std::string &value )
{
    const std::string whitespace = " \t\r\n";
    const std::size_t begin      = value.find_first_not_of( whitespace );
    if ( begin == std::string::npos )
        return std::string();
    const std::size_t end = value.find_last_not_of( whitespace );
    return value.substr( begin, end - begin + 1 );
}

std::vector<std::string> split_csv_simple( const std::string &line )
{
    std::vector<std::string> values;
    std::string              current;
    for ( char ch : line )
    {
        if ( ch == ',' )
        {
            values.push_back( trim_copy( current ) );
            current.clear();
        }
        else
        {
            current.push_back( ch );
        }
    }
    values.push_back( trim_copy( current ) );
    return values;
}

std::string run_command_capture_first_line( const std::string &command, int &return_code )
{
    return_code = -1;
    FILE *pipe = popen( command.c_str(), "r" );
    if ( pipe == nullptr )
        return std::string();

    std::array<char, 4096> buffer;
    std::string            output;
    if ( fgets( buffer.data(), static_cast<int>( buffer.size() ), pipe ) != nullptr )
    {
        output = buffer.data();
    }
    return_code = pclose( pipe );
    return trim_copy( output );
}

int local_rank_from_environment()
{
    const char *names[] = { "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "PMI_LOCAL_RANK", "PMIX_LOCAL_RANK" };
    for ( const char *name : names )
    {
        const std::string value = getenv_or_empty( name );
        if ( !value.empty() )
            return std::atoi( value.c_str() );
    }
    return -1;
}

std::string node_name_from_environment()
{
    const char *names[] = { "SLURMD_NODENAME", "HOSTNAME" };
    for ( const char *name : names )
    {
        const std::string value = getenv_or_empty( name );
        if ( !value.empty() )
            return value;
    }
    char hostname[256] = { 0 };
    if ( gethostname( hostname, sizeof( hostname ) ) == 0 && hostname[0] != '\0' )
        return std::string( hostname );
    const std::string host = getenv_or_empty( "HOST" );
    if ( !host.empty() )
        return host;
    return "unknown";
}

std::string selected_device_pci_bus_id( int device )
{
    char bus_id[64] = { 0 };
    const cudaError_t err = cudaDeviceGetPCIBusId( bus_id, static_cast<int>( sizeof( bus_id ) ), device );
    if ( err != cudaSuccess )
    {
        cudaGetLastError();
        return std::string();
    }
    return std::string( bus_id );
}

void append_gpu_telemetry_row(
    const options_t &options, const scfd::communication::mpi_comm_info &comm_info, const char *phase
)
{
    if ( !options.enable_gpu_telemetry && !env_truthy( "FFTM_ENABLE_GPU_TELEMETRY" ) )
        return;

    try
    {
        const int selected_device = runtime_api_t::get_device();
        const std::string selected_pci = selected_device_pci_bus_id( selected_device );

        std::ostringstream cmd;
        cmd << "nvidia-smi --id=" << selected_device
            << " --query-gpu=index,name,uuid,pci.bus_id,pstate,clocks.sm,clocks.mem,temperature.gpu,"
               "power.draw,power.limit,memory.used,memory.total,utilization.gpu,utilization.memory "
               "--format=csv,noheader,nounits 2>/dev/null";
        int smi_rc = -1;
        const std::string smi_line = run_command_capture_first_line( cmd.str(), smi_rc );
        std::vector<std::string> smi = split_csv_simple( smi_line );
        smi.resize( 14 );

        const std::string header =
            "source,phase,rank,num_gpus,local_rank,node,selected_device,selected_pci_bus_id,cuda_visible_devices,"
            "nvidia_visible_devices,backend,strategy,mode,pencil_layout,pencil_pipeline,"
            "fft_exec_no_sync,"
            "native_opt0_reference_y_plan_bundle,native_opt0_raw_y_plan_bundle,"
            "native_opt0_y_plan_bundle_stream_first,native_opt0_raw_y_plan_bundle_reference_streams,"
            "native_opt0_reference_local_plan_context,native_opt0_y_no_sync_exec,p1,p2,nx,ny,nz,"
            "nvidia_smi_available,nvidia_smi_rc,"
            "smi_index,smi_name,smi_uuid,smi_pci_bus_id,pstate,clocks_sm_mhz,clocks_mem_mhz,"
            "temperature_gpu_c,power_draw_w,power_limit_w,memory_used_mib,memory_total_mib,"
            "utilization_gpu_pct,utilization_memory_pct";

        std::ostringstream row;
        row << fftm::test::detail::csv_quote( "fftm-gpu-telemetry" ) << ','
            << fftm::test::detail::csv_quote( phase == nullptr ? "" : phase ) << ',' << comm_info.myid << ','
            << comm_info.num_procs << ',' << local_rank_from_environment() << ','
            << fftm::test::detail::csv_quote( node_name_from_environment() ) << ',' << selected_device << ','
            << fftm::test::detail::csv_quote( selected_pci ) << ','
            << fftm::test::detail::csv_quote( getenv_or_empty( "CUDA_VISIBLE_DEVICES" ) ) << ','
            << fftm::test::detail::csv_quote( getenv_or_empty( "NVIDIA_VISIBLE_DEVICES" ) ) << ','
            << fftm::test::detail::csv_quote( fftm::test::detail::fftm_3d_backend_name( options.backend ) ) << ','
            << fftm::test::detail::csv_quote( stage_timer_strategy_name( options.strategy ) ) << ','
            << fftm::test::detail::csv_quote( fftm::mpi_transpose_3d_mode_name( options.mode ) ) << ','
            << fftm::test::detail::csv_quote( fftm::test::detail::pencil_layout_name( options.pencil_layout ) )
            << ',' << fftm::test::detail::csv_quote(
                   fftm::test::detail::pencil_pipeline_name( options.pencil_pipeline ) )
            << ',' << ( options.use_fft_exec_no_sync ? 1 : 0 )
            << ',' << ( options.use_native_opt0_reference_y_plan_bundle ? 1 : 0 )
            << ',' << ( options.use_native_opt0_raw_y_plan_bundle ? 1 : 0 )
            << ',' << ( options.use_native_opt0_y_plan_bundle_stream_first ? 1 : 0 )
            << ',' << ( options.use_native_opt0_raw_y_plan_bundle_reference_streams ? 1 : 0 )
            << ',' << ( options.use_native_opt0_reference_local_plan_context ? 1 : 0 )
            << ',' << ( options.use_native_opt0_y_no_sync_exec ? 1 : 0 )
            << ',' << options.p1 << ',' << options.p2 << ',' << options.nx << ',' << options.ny << ',' << options.nz << ','
            << ( smi_line.empty() ? 0 : 1 ) << ',' << smi_rc;
        for ( const auto &value : smi )
        {
            row << ',' << fftm::test::detail::csv_quote( value );
        }

        fftm::test::detail::append_csv_row(
            options.directory, "gpu_telemetry_r" + std::to_string( comm_info.myid ) + ".csv", header, row.str()
        );
    }
    catch ( const std::exception &ex )
    {
        std::cerr << "WARNING: failed to write GPU telemetry on rank " << comm_info.myid << ": " << ex.what()
                  << std::endl;
    }
}

[[noreturn]] void throw_cuda_error( cudaError_t err, const char *expr )
{
    std::ostringstream os;
    os << expr << " failed: " << cudaGetErrorString( err );
    throw std::runtime_error( os.str() );
}

void check_cuda( cudaError_t err, const char *expr )
{
    if ( err != cudaSuccess )
        throw_cuda_error( err, expr );
}

void check_cufft( cufftResult err, const char *expr )
{
    if ( err != CUFFT_SUCCESS )
    {
        std::ostringstream os;
        os << expr << " failed with cuFFT status " << static_cast<int>( err );
        throw std::runtime_error( os.str() );
    }
}

struct device_buffer
{
    void       *ptr   = nullptr;
    std::size_t bytes = 0;

    device_buffer() = default;

    explicit device_buffer( std::size_t requested_bytes )
    {
        allocate( requested_bytes );
    }

    device_buffer( const device_buffer & ) = delete;
    device_buffer &operator=( const device_buffer & ) = delete;

    device_buffer( device_buffer &&other ) noexcept : ptr( other.ptr ), bytes( other.bytes )
    {
        other.ptr   = nullptr;
        other.bytes = 0;
    }

    device_buffer &operator=( device_buffer &&other ) noexcept
    {
        if ( this != &other )
        {
            release();
            ptr         = other.ptr;
            bytes       = other.bytes;
            other.ptr   = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    ~device_buffer()
    {
        release();
    }

    void allocate( std::size_t requested_bytes )
    {
        release();
        bytes = requested_bytes;
        if ( bytes != 0 )
            check_cuda( cudaMalloc( &ptr, bytes ), "cudaMalloc(y-cross buffer)" );
    }

    void release() noexcept
    {
        if ( ptr != nullptr )
        {
            cudaFree( ptr );
            ptr = nullptr;
        }
        bytes = 0;
    }
};

struct y_cross_topology
{
    const char         *owner = "";
    cufftDoubleComplex *forward_input = nullptr;
    cufftDoubleComplex *forward_output = nullptr;
    cufftDoubleComplex *backward_input = nullptr;
    cufftDoubleComplex *backward_output = nullptr;
    void              *work_base = nullptr;
    void              *first_work = nullptr;
    void              *last_work = nullptr;
};

class raw_reference_y_plan_array
{
public:
    raw_reference_y_plan_array() = default;

    raw_reference_y_plan_array( const raw_reference_y_plan_array & ) = delete;
    raw_reference_y_plan_array &operator=( const raw_reference_y_plan_array & ) = delete;

    ~raw_reference_y_plan_array()
    {
        for ( cufftHandle handle : handles_ )
        {
            if ( handle != 0 )
                cufftDestroy( handle );
        }
        for ( cudaStream_t stream : streams_ )
        {
            if ( stream != nullptr )
                cudaStreamDestroy( stream );
        }
    }

    void init(
        long long int n, long long int inembed, long long int istride, long long int idist,
        long long int onembed, long long int ostride, long long int odist, long long int batch,
        std::size_t plan_count
    )
    {
        handles_.reserve( plan_count );
        streams_.reserve( plan_count );
        long long int n_arr[1]       = { n };
        long long int inembed_arr[1] = { inembed };
        long long int onembed_arr[1] = { onembed };
        for ( std::size_t i = 0; i < plan_count; ++i )
        {
            cudaStream_t stream = nullptr;
            check_cuda( cudaStreamCreate( &stream ), "cudaStreamCreate(reference y-cross)" );
            streams_.push_back( stream );

            cufftHandle handle = 0;
            check_cufft( cufftCreate( &handle ), "cufftCreate(reference y-cross)" );
            handles_.push_back( handle );
            check_cufft( cufftSetAutoAllocation( handle, 0 ), "cufftSetAutoAllocation(reference y-cross)" );
            std::size_t work_size = 0;
            check_cufft(
                cufftMakePlanMany64(
                    handle, 1, n_arr, inembed_arr, istride, idist, onembed_arr, ostride, odist,
                    CUFFT_Z2Z, batch, &work_size
                ),
                "cufftMakePlanMany64(reference y-cross)"
            );
            check_cufft( cufftSetStream( handle, stream ), "cufftSetStream(reference y-cross)" );
            work_stride_bytes_ = std::max( work_stride_bytes_, work_size );
        }
    }

    void bind_work_areas( void *base, std::size_t stride )
    {
        char *raw = static_cast<char *>( base );
        const std::size_t actual_stride = std::max( stride, work_stride_bytes_ );
        for ( std::size_t i = 0; i < handles_.size(); ++i )
        {
            check_cufft(
                cufftSetWorkArea( handles_[i], static_cast<void *>( raw + i * actual_stride ) ),
                "cufftSetWorkArea(reference y-cross)"
            );
        }
    }

    void exec( int direction, cufftDoubleComplex *in, cufftDoubleComplex *out, const std::vector<std::size_t> &offsets )
    {
        if ( offsets.size() != handles_.size() )
            throw std::logic_error( "reference y-cross plan/offset count mismatch" );
        for ( std::size_t i = 0; i < handles_.size(); ++i )
        {
            check_cufft(
                cufftExecZ2Z( handles_[i], in + offsets[i], out + offsets[i], direction ),
                "cufftExecZ2Z(reference y-cross)"
            );
        }
    }

    void exec_repeated(
        int direction, cufftDoubleComplex *in, cufftDoubleComplex *out, const std::vector<std::size_t> &offsets,
        std::size_t repeats
    )
    {
        for ( std::size_t repeat = 0; repeat < repeats; ++repeat )
            exec( direction, in, out, offsets );
    }

    void synchronize_streams()
    {
        for ( cudaStream_t stream : streams_ )
            check_cuda( cudaStreamSynchronize( stream ), "cudaStreamSynchronize(reference y-cross)" );
    }

    std::size_t size() const
    {
        return handles_.size();
    }

    std::size_t work_stride_bytes() const
    {
        return work_stride_bytes_;
    }

    std::uintptr_t first_plan_token() const
    {
        return handles_.empty() ? 0 : static_cast<std::uintptr_t>( handles_.front() );
    }

    std::uintptr_t last_plan_token() const
    {
        return handles_.empty() ? 0 : static_cast<std::uintptr_t>( handles_.back() );
    }

    std::uintptr_t first_stream_token() const
    {
        return streams_.empty() ? 0 : reinterpret_cast<std::uintptr_t>( streams_.front() );
    }

    std::uintptr_t last_stream_token() const
    {
        return streams_.empty() ? 0 : reinterpret_cast<std::uintptr_t>( streams_.back() );
    }

    const std::vector<cudaStream_t> &streams() const
    {
        return streams_;
    }

private:
    std::vector<cufftHandle> handles_;
    std::vector<cudaStream_t> streams_;
    std::size_t              work_stride_bytes_ = 0;
};

class cuda_stream_array
{
public:
    cuda_stream_array() = default;

    explicit cuda_stream_array( std::size_t count )
    {
        init( count );
    }

    cuda_stream_array( const cuda_stream_array & ) = delete;
    cuda_stream_array &operator=( const cuda_stream_array & ) = delete;

    ~cuda_stream_array()
    {
        for ( cudaStream_t stream : streams_ )
        {
            if ( stream != nullptr )
                cudaStreamDestroy( stream );
        }
    }

    void init( std::size_t count )
    {
        streams_.reserve( count );
        for ( std::size_t i = 0; i < count; ++i )
        {
            cudaStream_t stream = nullptr;
            check_cuda( cudaStreamCreate( &stream ), "cudaStreamCreate(native y-cross)" );
            streams_.push_back( stream );
        }
    }

    const std::vector<cudaStream_t> &streams() const
    {
        return streams_;
    }

    std::uintptr_t first_token() const
    {
        return streams_.empty() ? 0 : reinterpret_cast<std::uintptr_t>( streams_.front() );
    }

    std::uintptr_t last_token() const
    {
        return streams_.empty() ? 0 : reinterpret_cast<std::uintptr_t>( streams_.back() );
    }

private:
    std::vector<cudaStream_t> streams_;
};

std::string y_cross_filename( int rank )
{
    return "y_cross_r" + std::to_string( rank ) + ".csv";
}

std::uintptr_t pointer_token( const void *ptr )
{
    return reinterpret_cast<std::uintptr_t>( ptr );
}

int run_native_opt0_y_cross_microbench(
    scfd::utils::log_mpi &log, const options_t &options, const scfd::communication::mpi_comm_info &comm_info
)
{
    if ( options.run_all )
        throw std::logic_error( "--native-opt0-y-cross-microbench does not support --strategy all" );
    if ( options.strategy != strategy_kind::pencil_pencil )
        throw std::logic_error( "--native-opt0-y-cross-microbench is only valid for strategy=pencil-pencil" );
    if ( options.pencil_layout != fftm::test::detail::fftm_3d_pencil_layout_kind::opt0 )
        throw std::logic_error( "--native-opt0-y-cross-microbench requires --pencil-layout opt0" );
    if ( options.p1 * options.p2 != static_cast<std::size_t>( comm_info.num_procs ) )
        throw std::logic_error( "P1*P2 must equal the number of MPI processes" );
    fftm::test::detail::ensure_directory_exists( options.directory );

    fftm::processor_grid grid;
    grid.init( options.p1, options.p2 );
    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );
    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( grid, sizes );

    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();
    (void)myid_k;
    fftm::partition input_dim;
    fftm::partition transposed_dim;
    fftm::partition output_dim;
    std::tie( input_dim, transposed_dim, output_dim ) = partitioning.get_partitioning_3D();

    const std::size_t x_size = input_dim.size_x[myid_i];
    const std::size_t z_size = transposed_dim.size_z[myid_j];
    const std::size_t y_size = options.ny;
    const std::size_t plan_count = std::min( x_size, z_size );
    const std::size_t batch = std::max( x_size, z_size );
    const std::size_t nembed = ( x_size <= z_size ) ? 1 : z_size * y_size;
    const std::size_t offset_step = ( x_size <= z_size ) ? z_size * y_size : 1;
    const std::size_t buffer_elems = x_size * y_size * z_size;
    if ( plan_count == 0 || buffer_elems == 0 )
        throw std::logic_error( "invalid empty Y cross microbench geometry" );

    std::vector<std::size_t> offsets;
    offsets.reserve( plan_count );
    for ( std::size_t i = 0; i < plan_count; ++i )
        offsets.push_back( i * offset_step );

    const std::size_t complex_bytes = buffer_elems * sizeof( cufftDoubleComplex );
    const std::size_t ref_work_offset_slots = 3;
    const std::size_t ref_domain_bytes = complex_bytes;
    const std::size_t ref_work_base_offset = ref_work_offset_slots * ref_domain_bytes;
    const std::size_t work_stride = static_cast<std::size_t>( 1 ) << 20;
    const std::size_t work_span_bytes = plan_count * work_stride;

    device_buffer ref_workspace( ref_work_base_offset + work_span_bytes );
    device_buffer ref_output( complex_bytes );
    device_buffer native_input( complex_bytes );
    device_buffer native_output( complex_bytes );
    device_buffer native_work( work_span_bytes );

    y_cross_topology ref_topology;
    ref_topology.owner = "reference-buffer";
    ref_topology.forward_input = static_cast<cufftDoubleComplex *>( ref_workspace.ptr );
    ref_topology.forward_output = static_cast<cufftDoubleComplex *>( ref_output.ptr );
    ref_topology.backward_input = ref_topology.forward_output;
    ref_topology.backward_output = ref_topology.forward_input;
    ref_topology.work_base = static_cast<void *>( static_cast<char *>( ref_workspace.ptr ) + ref_work_base_offset );
    ref_topology.first_work = ref_topology.work_base;
    ref_topology.last_work = static_cast<void *>(
        static_cast<char *>( ref_topology.work_base ) + ( plan_count - 1 ) * work_stride
    );

    y_cross_topology native_topology;
    native_topology.owner = "native-buffer";
    native_topology.forward_input = static_cast<cufftDoubleComplex *>( native_input.ptr );
    native_topology.forward_output = static_cast<cufftDoubleComplex *>( native_output.ptr );
    native_topology.backward_input = native_topology.forward_output;
    native_topology.backward_output = native_topology.forward_input;
    native_topology.work_base = native_work.ptr;
    native_topology.first_work = native_work.ptr;
    native_topology.last_work = static_cast<void *>( static_cast<char *>( native_work.ptr ) + ( plan_count - 1 ) * work_stride );

    const std::string header =
        "source,run_label,rank,factory_mode,direction,variant,plan_owner,native_factory,buffer_owner,iterations,warmup,total_ms,avg_ms,"
        "event_total_ms,event_avg_ms,host_overhead_total_ms,host_overhead_avg_ms,plan_count,n,inembed,istride,"
        "idist,onembed,ostride,odist,batch,batches_offset_elems,buffer_elems,complex_bytes,work_stride_bytes,"
        "input_token,output_token,work_base_token,first_work_token,last_work_token,first_plan_token,last_plan_token,"
        "first_stream_token,last_stream_token,pidx_i,pidx_j,directory";

    const auto append_row = [&]( const char *factory_mode, const char *direction_name, const char *variant,
                                 const char *plan_owner, const char *native_factory, const y_cross_topology &topology,
                                 double total_ms, double event_total_ms, std::uintptr_t first_plan_token,
                                 std::uintptr_t last_plan_token, std::uintptr_t first_stream_token,
                                 std::uintptr_t last_stream_token, const void *in_ptr, const void *out_ptr ) {
        const double avg_ms = total_ms / static_cast<double>( options.native_opt0_y_microbench_iterations );
        const double event_avg_ms =
            event_total_ms / static_cast<double>( options.native_opt0_y_microbench_iterations );
        const double host_overhead_total_ms = total_ms - event_total_ms;
        const double host_overhead_avg_ms =
            host_overhead_total_ms / static_cast<double>( options.native_opt0_y_microbench_iterations );
        std::ostringstream label;
        label << "y_cross_g" << comm_info.num_procs << "_p" << options.p1 << "x" << options.p2 << "_"
              << options.nx << "x" << options.ny << "x" << options.nz;

        std::ostringstream row;
        row << "fftm-native-y-cross," << label.str() << ',' << comm_info.myid << ',' << factory_mode << ','
            << direction_name << ',' << variant << ',' << plan_owner << ',' << native_factory << ','
            << topology.owner << ','
            << options.native_opt0_y_microbench_iterations << ',' << options.native_opt0_y_microbench_warmup << ','
            << total_ms << ',' << avg_ms << ',' << event_total_ms << ',' << event_avg_ms << ','
            << host_overhead_total_ms << ',' << host_overhead_avg_ms << ',' << plan_count << ',' << y_size << ','
            << nembed << ',' << z_size << ',' << nembed << ',' << nembed << ',' << z_size << ',' << nembed << ','
            << batch << ',' << offset_step << ',' << buffer_elems << ',' << complex_bytes << ',' << work_stride << ','
            << pointer_token( in_ptr ) << ',' << pointer_token( out_ptr ) << ',' << pointer_token( topology.work_base )
            << ',' << pointer_token( topology.first_work ) << ',' << pointer_token( topology.last_work ) << ','
            << first_plan_token << ',' << last_plan_token << ',' << first_stream_token << ',' << last_stream_token
            << ',' << myid_i << ',' << myid_j << ',' << fftm::test::detail::csv_quote( options.directory );
        fftm::test::detail::append_csv_row(
            options.directory, y_cross_filename( comm_info.myid ), header, row.str()
        );
    };

    const auto init_reference_plans = [&]( raw_reference_y_plan_array &reference_plans ) {
        reference_plans.init(
            static_cast<long long int>( y_size ), static_cast<long long int>( nembed ),
            static_cast<long long int>( z_size ), static_cast<long long int>( nembed ),
            static_cast<long long int>( nembed ), static_cast<long long int>( z_size ),
            static_cast<long long int>( nembed ), static_cast<long long int>( batch ), plan_count
        );
        if ( reference_plans.work_stride_bytes() > work_stride )
            throw std::logic_error( "reference Y cross work area exceeds diagnostic fixed stride" );
    };

    const auto make_native_bundle = [&]( base_fft_t &native_fft, cuda_stream_array &native_streams,
                                         const char *native_factory ) {
        base_fft_t::c2c_plan_array_id_t native_bundle = base_fft_t::invalid_c2c_plan_array_id();
        if ( std::string( native_factory ) == "exact-raw-reference" )
        {
            native_bundle = native_fft.make_reference_style_raw_c2c_plan_array_1D_diagnostic_with_streams(
                static_cast<long long int>( y_size ), static_cast<long long int>( nembed ),
                static_cast<long long int>( z_size ), static_cast<long long int>( nembed ),
                static_cast<long long int>( nembed ), static_cast<long long int>( z_size ),
                static_cast<long long int>( nembed ), static_cast<long long int>( batch ), offsets,
                native_streams.streams()
            );
        }
        else
        {
            native_bundle = native_fft.make_raw_c2c_plan_array_1D_diagnostic_with_streams(
                static_cast<long long int>( y_size ), static_cast<long long int>( nembed ),
                static_cast<long long int>( z_size ), static_cast<long long int>( nembed ),
                static_cast<long long int>( nembed ), static_cast<long long int>( z_size ),
                static_cast<long long int>( nembed ), static_cast<long long int>( batch ), offsets,
                native_streams.streams()
            );
        }
        if ( native_fft.c2c_plan_array_work_size( native_bundle ) > work_stride )
            throw std::logic_error( "native Y cross work area exceeds diagnostic fixed stride" );
        return native_bundle;
    };

    const auto make_native_owned_streams_bundle = [&]( base_fft_t &native_fft ) {
        auto native_bundle = native_fft.make_reference_style_raw_c2c_plan_array_1D_diagnostic_with_owned_streams(
            static_cast<long long int>( y_size ), static_cast<long long int>( nembed ),
            static_cast<long long int>( z_size ), static_cast<long long int>( nembed ),
            static_cast<long long int>( nembed ), static_cast<long long int>( z_size ),
            static_cast<long long int>( nembed ), static_cast<long long int>( batch ), offsets
        );
        if ( native_fft.c2c_plan_array_work_size( native_bundle ) > work_stride )
            throw std::logic_error( "native owned-streams Y cross work area exceeds diagnostic fixed stride" );
        return native_bundle;
    };

    const auto make_native_direct_c_bundle = [&]( base_fft_t &native_fft ) {
        auto native_bundle = native_fft.make_direct_c_raw_c2c_plan_array_1D_diagnostic_with_owned_streams(
            static_cast<long long int>( y_size ), static_cast<long long int>( nembed ),
            static_cast<long long int>( z_size ), static_cast<long long int>( nembed ),
            static_cast<long long int>( nembed ), static_cast<long long int>( z_size ),
            static_cast<long long int>( nembed ), static_cast<long long int>( batch ), offsets
        );
        if ( native_fft.c2c_plan_array_work_size( native_bundle ) > work_stride )
            throw std::logic_error( "native direct-C Y cross work area exceeds diagnostic fixed stride" );
        return native_bundle;
    };

    const auto make_native_minimal_reference_bundle = [&]( base_fft_t &native_fft ) {
        auto native_bundle = native_fft.make_minimal_reference_c2c_plan_array_1D_diagnostic(
            static_cast<long long int>( y_size ), static_cast<long long int>( nembed ),
            static_cast<long long int>( z_size ), static_cast<long long int>( nembed ),
            static_cast<long long int>( nembed ), static_cast<long long int>( z_size ),
            static_cast<long long int>( nembed ), static_cast<long long int>( batch ), offsets
        );
        if ( native_fft.minimal_reference_c2c_plan_array_work_size( native_bundle ) > work_stride )
            throw std::logic_error( "native minimal-reference Y cross work area exceeds diagnostic fixed stride" );
        return native_bundle;
    };

    const auto run_reference_timed = [&]( const char *factory_mode, raw_reference_y_plan_array &reference_plans,
                                          const char *direction_name, const char *variant,
                                          const y_cross_topology &topology, int cufft_direction ) {
        reference_plans.bind_work_areas( topology.work_base, work_stride );
        cufftDoubleComplex *in_ptr = cufft_direction == CUFFT_FORWARD ? topology.forward_input : topology.backward_input;
        cufftDoubleComplex *out_ptr = cufft_direction == CUFFT_FORWARD ? topology.forward_output : topology.backward_output;

        check_cuda( cudaDeviceSynchronize(), "cudaDeviceSynchronize(y-cross pre)" );
        for ( int i = 0; i < options.native_opt0_y_microbench_warmup; ++i )
        {
            reference_plans.exec( cufft_direction, in_ptr, out_ptr, offsets );
        }
        reference_plans.synchronize_streams();
        check_cuda( cudaDeviceSynchronize(), "cudaDeviceSynchronize(y-cross warmup)" );

        cudaEvent_t event_start = nullptr;
        cudaEvent_t event_stop = nullptr;
        check_cuda( cudaEventCreate( &event_start ), "cudaEventCreate(y-cross start)" );
        check_cuda( cudaEventCreate( &event_stop ), "cudaEventCreate(y-cross stop)" );
        scfd::utils::system_timer_event begin;
        scfd::utils::system_timer_event end;
        begin.record();
        check_cuda( cudaEventRecord( event_start, 0 ), "cudaEventRecord(y-cross start)" );
        reference_plans.exec_repeated(
            cufft_direction, in_ptr, out_ptr, offsets,
            static_cast<std::size_t>( options.native_opt0_y_microbench_iterations )
        );
        check_cuda( cudaEventRecord( event_stop, 0 ), "cudaEventRecord(y-cross stop)" );
        reference_plans.synchronize_streams();
        check_cuda( cudaEventSynchronize( event_stop ), "cudaEventSynchronize(y-cross stop)" );
        float event_ms = 0.0f;
        check_cuda( cudaEventElapsedTime( &event_ms, event_start, event_stop ), "cudaEventElapsedTime(y-cross)" );
        check_cuda( cudaDeviceSynchronize(), "cudaDeviceSynchronize(y-cross post)" );
        end.record();
        check_cuda( cudaEventDestroy( event_start ), "cudaEventDestroy(y-cross start)" );
        check_cuda( cudaEventDestroy( event_stop ), "cudaEventDestroy(y-cross stop)" );

        append_row(
            factory_mode, direction_name, variant, "reference-plan", "reference-raw", topology, end.elapsed_time( begin ),
            static_cast<double>( event_ms ), reference_plans.first_plan_token(), reference_plans.last_plan_token(),
            reference_plans.first_stream_token(), reference_plans.last_stream_token(), in_ptr, out_ptr
        );
    };

    const auto run_native_timed = [&]( const char *factory_mode, const char *native_factory, base_fft_t &native_fft,
                                       base_fft_t::c2c_plan_array_id_t native_bundle, const char *direction_name,
                                       const char *variant, const y_cross_topology &topology,
                                       ::fftm::direction native_direction, bool no_sync_exec ) {
        native_fft.bind_c2c_plan_array_work_areas( native_bundle, topology.work_base, work_stride );
        cufftDoubleComplex *in_ptr =
            native_direction == ::fftm::direction::C2CF ? topology.forward_input : topology.backward_input;
        cufftDoubleComplex *out_ptr =
            native_direction == ::fftm::direction::C2CF ? topology.forward_output : topology.backward_output;

        check_cuda( cudaDeviceSynchronize(), "cudaDeviceSynchronize(y-cross native pre)" );
        for ( int i = 0; i < options.native_opt0_y_microbench_warmup; ++i )
        {
            if ( no_sync_exec )
                native_fft.exec_c2c_plan_array_direction_no_sync(
                    native_bundle, native_direction, in_ptr, out_ptr
                );
            else
                native_fft.exec_c2c_plan_array_direction( native_bundle, native_direction, in_ptr, out_ptr );
        }
        native_fft.synchronize_c2c_plan_array_streams( native_bundle );
        check_cuda( cudaDeviceSynchronize(), "cudaDeviceSynchronize(y-cross native warmup)" );

        cudaEvent_t event_start = nullptr;
        cudaEvent_t event_stop = nullptr;
        check_cuda( cudaEventCreate( &event_start ), "cudaEventCreate(y-cross native start)" );
        check_cuda( cudaEventCreate( &event_stop ), "cudaEventCreate(y-cross native stop)" );
        scfd::utils::system_timer_event begin;
        scfd::utils::system_timer_event end;
        begin.record();
        check_cuda( cudaEventRecord( event_start, 0 ), "cudaEventRecord(y-cross native start)" );
        if ( no_sync_exec )
            native_fft.exec_c2c_plan_array_direction_no_sync_repeated(
                native_bundle, native_direction, in_ptr, out_ptr,
                static_cast<std::size_t>( options.native_opt0_y_microbench_iterations )
            );
        else
            native_fft.exec_c2c_plan_array_direction_repeated(
                native_bundle, native_direction, in_ptr, out_ptr,
                static_cast<std::size_t>( options.native_opt0_y_microbench_iterations )
            );
        check_cuda( cudaEventRecord( event_stop, 0 ), "cudaEventRecord(y-cross native stop)" );
        native_fft.synchronize_c2c_plan_array_streams( native_bundle );
        check_cuda( cudaEventSynchronize( event_stop ), "cudaEventSynchronize(y-cross native stop)" );
        float event_ms = 0.0f;
        check_cuda( cudaEventElapsedTime( &event_ms, event_start, event_stop ), "cudaEventElapsedTime(y-cross native)" );
        check_cuda( cudaDeviceSynchronize(), "cudaDeviceSynchronize(y-cross native post)" );
        end.record();
        check_cuda( cudaEventDestroy( event_start ), "cudaEventDestroy(y-cross native start)" );
        check_cuda( cudaEventDestroy( event_stop ), "cudaEventDestroy(y-cross native stop)" );

        append_row(
            factory_mode, direction_name, variant, "native-plan", native_factory, topology, end.elapsed_time( begin ),
            static_cast<double>( event_ms ), native_fft.c2c_plan_array_first_handle_token( native_bundle ),
            native_fft.c2c_plan_array_last_handle_token( native_bundle ),
            native_fft.c2c_plan_array_first_stream_token( native_bundle ),
            native_fft.c2c_plan_array_last_stream_token( native_bundle ), in_ptr, out_ptr
        );
    };

    const auto run_native_minimal_timed = [&](
        const char *factory_mode, const char *native_factory, base_fft_t &native_fft,
        base_fft_t::minimal_reference_c2c_plan_array_id_t native_bundle, const char *direction_name,
        const char *variant, const y_cross_topology &topology, ::fftm::direction native_direction,
        bool no_sync_exec
    ) {
        native_fft.bind_minimal_reference_c2c_plan_array_work_areas( native_bundle, topology.work_base, work_stride );
        cufftDoubleComplex *in_ptr =
            native_direction == ::fftm::direction::C2CF ? topology.forward_input : topology.backward_input;
        cufftDoubleComplex *out_ptr =
            native_direction == ::fftm::direction::C2CF ? topology.forward_output : topology.backward_output;

        check_cuda( cudaDeviceSynchronize(), "cudaDeviceSynchronize(y-cross native minimal pre)" );
        for ( int i = 0; i < options.native_opt0_y_microbench_warmup; ++i )
        {
            if ( no_sync_exec )
                native_fft.exec_minimal_reference_c2c_plan_array_direction_no_sync(
                    native_bundle, native_direction, in_ptr, out_ptr
                );
            else
                native_fft.exec_minimal_reference_c2c_plan_array_direction(
                    native_bundle, native_direction, in_ptr, out_ptr
                );
        }
        native_fft.synchronize_minimal_reference_c2c_plan_array_streams( native_bundle );
        check_cuda( cudaDeviceSynchronize(), "cudaDeviceSynchronize(y-cross native minimal warmup)" );

        cudaEvent_t event_start = nullptr;
        cudaEvent_t event_stop = nullptr;
        check_cuda( cudaEventCreate( &event_start ), "cudaEventCreate(y-cross native minimal start)" );
        check_cuda( cudaEventCreate( &event_stop ), "cudaEventCreate(y-cross native minimal stop)" );
        scfd::utils::system_timer_event begin;
        scfd::utils::system_timer_event end;
        begin.record();
        check_cuda( cudaEventRecord( event_start, 0 ), "cudaEventRecord(y-cross native minimal start)" );
        if ( no_sync_exec )
            native_fft.exec_minimal_reference_c2c_plan_array_direction_no_sync_repeated(
                native_bundle, native_direction, in_ptr, out_ptr,
                static_cast<std::size_t>( options.native_opt0_y_microbench_iterations )
            );
        else
            native_fft.exec_minimal_reference_c2c_plan_array_direction_repeated(
                native_bundle, native_direction, in_ptr, out_ptr,
                static_cast<std::size_t>( options.native_opt0_y_microbench_iterations )
            );
        check_cuda( cudaEventRecord( event_stop, 0 ), "cudaEventRecord(y-cross native minimal stop)" );
        native_fft.synchronize_minimal_reference_c2c_plan_array_streams( native_bundle );
        check_cuda( cudaEventSynchronize( event_stop ), "cudaEventSynchronize(y-cross native minimal stop)" );
        float event_ms = 0.0f;
        check_cuda(
            cudaEventElapsedTime( &event_ms, event_start, event_stop ),
            "cudaEventElapsedTime(y-cross native minimal)"
        );
        check_cuda( cudaDeviceSynchronize(), "cudaDeviceSynchronize(y-cross native minimal post)" );
        end.record();
        check_cuda( cudaEventDestroy( event_start ), "cudaEventDestroy(y-cross native minimal start)" );
        check_cuda( cudaEventDestroy( event_stop ), "cudaEventDestroy(y-cross native minimal stop)" );

        append_row(
            factory_mode, direction_name, variant, "native-plan", native_factory, topology, end.elapsed_time( begin ),
            static_cast<double>( event_ms ),
            native_fft.minimal_reference_c2c_plan_array_first_handle_token( native_bundle ),
            native_fft.minimal_reference_c2c_plan_array_last_handle_token( native_bundle ),
            native_fft.minimal_reference_c2c_plan_array_first_stream_token( native_bundle ),
            native_fft.minimal_reference_c2c_plan_array_last_stream_token( native_bundle ), in_ptr, out_ptr
        );
    };

    const y_cross_topology *topologies[] = { &ref_topology, &native_topology };
    const auto run_reference_rows = [&]( const char *factory_mode, raw_reference_y_plan_array &reference_plans ) {
        for ( const y_cross_topology *topology : topologies )
        {
            const bool ref_buffer = std::string( topology->owner ) == "reference-buffer";
            const char *ref_variant = ref_buffer ? "ref_plan/ref_buffer" : "ref_plan/native_buffer";
            run_reference_timed( factory_mode, reference_plans, "forward", ref_variant, *topology, CUFFT_FORWARD );
            run_reference_timed( factory_mode, reference_plans, "backward", ref_variant, *topology, CUFFT_INVERSE );
        }
    };

    const auto run_native_rows = [&]( const char *factory_mode, const char *native_factory, base_fft_t &native_fft,
                                      base_fft_t::c2c_plan_array_id_t native_bundle, bool no_sync_exec ) {
        for ( const y_cross_topology *topology : topologies )
        {
            const bool ref_buffer = std::string( topology->owner ) == "reference-buffer";
            const char *native_variant = ref_buffer ? "native_plan/ref_buffer" : "native_plan/native_buffer";
            run_native_timed(
                factory_mode, native_factory, native_fft, native_bundle, "forward", native_variant, *topology,
                ::fftm::direction::C2CF, no_sync_exec
            );
            run_native_timed(
                factory_mode, native_factory, native_fft, native_bundle, "backward", native_variant, *topology,
                ::fftm::direction::C2CB, no_sync_exec
            );
        }
    };

    const auto run_native_minimal_rows = [&](
        const char *factory_mode, const char *native_factory, base_fft_t &native_fft,
        base_fft_t::minimal_reference_c2c_plan_array_id_t native_bundle, bool no_sync_exec
    ) {
        for ( const y_cross_topology *topology : topologies )
        {
            const bool ref_buffer = std::string( topology->owner ) == "reference-buffer";
            const char *native_variant =
                ref_buffer ? "native_minimal_plan/ref_buffer" : "native_minimal_plan/native_buffer";
            run_native_minimal_timed(
                factory_mode, native_factory, native_fft, native_bundle, "forward", native_variant, *topology,
                ::fftm::direction::C2CF, no_sync_exec
            );
            run_native_minimal_timed(
                factory_mode, native_factory, native_fft, native_bundle, "backward", native_variant, *topology,
                ::fftm::direction::C2CB, no_sync_exec
            );
        }
    };

    const std::string factory_mode = options.native_opt0_y_cross_factory_mode == "single"
                                         ? std::string( "ref-then-native-current" )
                                         : options.native_opt0_y_cross_factory_mode;

    const auto starts_with = []( const std::string &value, const char *prefix ) {
        return value.rfind( prefix, 0 ) == 0;
    };
    const auto ends_with = []( const std::string &value, const char *suffix ) {
        const std::string suffix_str( suffix );
        return value.size() >= suffix_str.size() &&
               value.compare( value.size() - suffix_str.size(), suffix_str.size(), suffix_str ) == 0;
    };
    const auto contains = []( const std::string &value, const char *needle ) {
        return value.find( needle ) != std::string::npos;
    };
    const auto native_factory_label = [&]( bool no_sync_exec ) {
        if ( contains( factory_mode, "direct" ) )
            return no_sync_exec ? "direct-c-raw-nosync" : "direct-c-raw";
        if ( contains( factory_mode, "minimal" ) )
            return no_sync_exec ? "minimal-reference-bundle-nosync" : "minimal-reference-bundle";
        if ( contains( factory_mode, "owned" ) )
            return no_sync_exec ? "exact-owned-streams-nosync" : "exact-owned-streams";
        if ( contains( factory_mode, "exact" ) )
            return no_sync_exec ? "exact-raw-reference-nosync" : "exact-raw-reference";
        return no_sync_exec ? "current-nosync" : "current";
    };
    const auto run_selected_native_rows = [&]( const char *mode_name ) {
        const bool no_sync_exec = contains( factory_mode, "-nosync" );
        const bool minimal = contains( factory_mode, "minimal" );
        const bool direct_c = contains( factory_mode, "direct" );
        const bool owned_streams = contains( factory_mode, "owned" );
        const char *native_factory = native_factory_label( no_sync_exec );
        base_fft_t native_fft;
        cuda_stream_array native_streams;
        if ( minimal )
        {
            auto native_bundle = make_native_minimal_reference_bundle( native_fft );
            run_native_minimal_rows( mode_name, native_factory, native_fft, native_bundle, no_sync_exec );
        }
        else
        {
            auto native_bundle = direct_c ? make_native_direct_c_bundle( native_fft )
                                 : owned_streams ? make_native_owned_streams_bundle( native_fft )
                                                 : ( native_streams.init( plan_count ),
                                                     make_native_bundle( native_fft, native_streams, native_factory ) );
            run_native_rows( mode_name, native_factory, native_fft, native_bundle, no_sync_exec );
        }
    };
    const auto run_selected_reference_rows = [&]( const char *mode_name ) {
        raw_reference_y_plan_array reference_plans;
        init_reference_plans( reference_plans );
        run_reference_rows( mode_name, reference_plans );
    };

    if ( factory_mode == "ref-only" )
    {
        run_selected_reference_rows( factory_mode.c_str() );
    }
    else if ( starts_with( factory_mode, "native-" ) && ends_with( factory_mode, "-only" ) )
    {
        run_selected_native_rows( factory_mode.c_str() );
    }
    else if ( starts_with( factory_mode, "ref-then-native-" ) )
    {
        run_selected_reference_rows( factory_mode.c_str() );
        run_selected_native_rows( factory_mode.c_str() );
    }
    else if ( starts_with( factory_mode, "native-" ) && ends_with( factory_mode, "-then-ref" ) )
    {
        run_selected_native_rows( factory_mode.c_str() );
        run_selected_reference_rows( factory_mode.c_str() );
    }
    else
    {
        throw std::logic_error( "unknown native opt0 Y cross factory mode" );
    }

    comm_info.barrier();
    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "benchmark=fftm-3d, diagnostic=native_opt0_y_cross_microbench, factory_mode=%s, grid=(%zu,%zu), "
            "Nx=%zu, Ny=%zu, Nz=%zu, iterations=%d, warmup=%d: complete",
            factory_mode.c_str(),
            options.p1, options.p2, options.nx, options.ny, options.nz,
            options.native_opt0_y_microbench_iterations, options.native_opt0_y_microbench_warmup
        );
    }
    return 0;
}

int fftm3d_opt_for_options( const options_t &options )
{
    using layout_kind = fftm::test::detail::fftm_3d_pencil_layout_kind;
    if ( options.pencil_layout == layout_kind::opt1 )
        return 1;
    if ( options.pencil_layout == layout_kind::legacy )
        throw std::logic_error( "fftm3d-scfd-fft-facade backend does not support pencil-layout=legacy" );
    return 0;
}

std::string fftm3d_backend_run_id( const options_t &options, int opt )
{
    std::ostringstream os;
    os << "fftm_backend_g" << ( options.p1 * options.p2 ) << "_p" << options.p1 << "x" << options.p2
       << "_opt" << opt << "_" << options.nx << "x" << options.ny << "x" << options.nz;
    return os.str();
}

int run_fftm3d_scfd_fft_facade_backend(
    scfd::utils::log_mpi &log, const options_t &options, const scfd::communication::mpi_comm_info &comm_info
)
{
#if defined(FFTM_ENABLE_FFTM3D_SCFD_FFT_BACKEND)
    (void)log;
    if ( options.run_all )
        throw std::logic_error( "fftm3d-scfd-fft-facade backend does not support --strategy all" );
    if ( options.strategy != strategy_kind::pencil_pencil )
        throw std::logic_error( "fftm3d-scfd-fft-facade backend is only valid for strategy=pencil-pencil" );
    if ( options.mode != fftm::mpi_transpose_3d_mode::p2p_waitall &&
         options.mode != fftm::mpi_transpose_3d_mode::p2p_waitany )
        throw std::logic_error( "fftm3d-scfd-fft-facade backend is only valid for P2P modes" );
    if ( options.p1 * options.p2 != static_cast<std::size_t>( comm_info.num_procs ) )
        throw std::logic_error( "P1*P2 must equal the number of MPI processes" );

    const int opt = fftm3d_opt_for_options( options );

    std::vector<std::string> args;
    args.reserve( 52 );
    const std::string run_id = fftm3d_backend_run_id( options, opt );
    args.push_back( "fftm3d_backend_for_fftm" );
    args.push_back( "--grid" );
    args.push_back( to_arg( options.p1 ) );
    args.push_back( to_arg( options.p2 ) );
    args.push_back( "--pencil-layout" );
    args.push_back( opt == 1 ? "opt1" : "opt0" );
    args.push_back( "--comm-method1" );
    args.push_back( "Peer2Peer" );
    args.push_back( "--send-method1" );
    args.push_back( "Sync" );
    args.push_back( "--comm-method2" );
    args.push_back( "Peer2Peer" );
    args.push_back( "--send-method2" );
    args.push_back( "Sync" );
    args.push_back( options.direct_p2p_cuda_aware ? "--cuda-aware" : "--no-cuda-aware" );
    args.push_back( "--storage" );
    args.push_back( "scfd-tensor" );
    args.push_back( "--mpi-backend" );
    args.push_back( "scfd" );
    args.push_back( "--runtime-backend" );
    args.push_back( "scfd" );
    args.push_back( "--input-mode" );
    args.push_back( "fftm-compatible" );
    args.push_back( "--egger-variant" );
    args.push_back( "scfd-fft-facade" );
    args.push_back( "--diagnostic-stage" );
    args.push_back( "fftm-backend" );
    args.push_back( "--run-id" );
    args.push_back( run_id );
    args.push_back( "--directory" );
    args.push_back( options.directory );
    args.push_back( options.print_pencil_schedule ? "--write-schedule-csv" : "--no-schedule-csv" );
    args.push_back( "--no-iteration-csv" );
    args.push_back(
        options.enable_fftm3d_backend_stage_timers ? "--enable-egger-stage-timers"
                                                   : "--disable-egger-stage-timers"
    );
    args.push_back(
        ( options.enable_local_fft_diagnostics || options.native_opt0_y_microbench )
            ? "--enable-local-fft-diagnostics"
            : "--disable-local-fft-diagnostics"
    );
    if ( options.native_opt0_y_microbench )
    {
        args.push_back( "--y-microbench" );
        args.push_back( "--y-microbench-iterations" );
        args.push_back( to_arg( options.native_opt0_y_microbench_iterations ) );
        args.push_back( "--y-microbench-warmup" );
        args.push_back( to_arg( options.native_opt0_y_microbench_warmup ) );
    }
    args.push_back( "--times" );
    args.push_back( to_arg( options.times ) );
    args.push_back( "--warmup" );
    args.push_back( to_arg( options.warmup ) );
    args.push_back( to_arg( options.nx ) );
    args.push_back( to_arg( options.ny ) );
    args.push_back( to_arg( options.nz ) );

    std::vector<char *> argv;
    argv.reserve( args.size() );
    for ( std::string &arg : args )
        argv.push_back( const_cast<char *>( arg.c_str() ) );

    if ( comm_info.myid == 0 )
    {
        std::cout << "INFO: benchmark=fftm-3d, backend="
                  << fftm::test::detail::fftm_3d_backend_name( options.backend )
                  << ", strategy=pencil-pencil, mode="
                  << fftm::mpi_transpose_3d_mode_name( options.mode ) << ", grid=(" << options.p1 << ","
                  << options.p2 << "), Nx=" << options.nx << ", Ny=" << options.ny << ", Nz=" << options.nz
                  << ", warmup=" << options.warmup << ", times=" << options.times << ", pencil_layout="
                  << ( opt == 1 ? "opt1" : "opt0" ) << ", pencil_pipeline=scfd-fft-facade: launching=1"
                  << ", fftm3d_backend_stage_timers="
                  << ( options.enable_fftm3d_backend_stage_timers ? 1 : 0 )
                  << ", fftm3d_input_mode=fftm-compatible"
                  << std::endl;
    }
    return fftm3d::benchmark::run_3d_with_existing_mpi(
        static_cast<int>( argv.size() ), argv.data(), comm_info
    );
#else
    (void)log;
    (void)options;
    (void)comm_info;
    throw std::logic_error(
        "fftm3d-scfd-fft-facade backend was requested, but this binary was built without "
        "FFTM_ENABLE_FFTM3D_SCFD_FFT_BACKEND"
    );
#endif
}

std::string native_stage_times_filename( int rank )
{
    return "native_stage_times_r" + std::to_string( rank ) + ".csv";
}

const char *stage_timer_strategy_name( strategy_kind strategy )
{
    switch ( strategy )
    {
    case strategy_kind::slab_pencil:
        return "slab-pencil";
    case strategy_kind::pencil_slab:
        return "pencil-slab";
    case strategy_kind::pencil_pencil:
        return "pencil-pencil";
    }
    return "unknown";
}

void append_native_stage_timer_rows(
    const options_t &options, int num_procs, int rank, int iteration, double wall_ms,
    const std::vector<fftm::fftm_native_stage_timing> &rows
)
{
    if ( rows.empty() )
        return;

    const std::string header =
        "source,rank,num_gpus,iteration,stage_index,stage,stage_ms,wall_ms,strategy,mode,pencil_layout,"
        "pencil_pipeline,large_count_p2p_transport,fft_exec_no_sync,native_backward_second_peer_loop,"
        "native_opt0_default_z_layout,"
        "native_opt0_reference_y_buffer_topology,native_opt0_compact_y_workarea,native_opt0_tight_y_plan_sequence,"
        "native_opt0_shared_y_plan_handles,native_opt0_y_group_device_sync,"
        "native_opt0_y_no_sync_exec,"
        "native_opt0_raw_y_plan_array_executor,native_opt0_reference_y_plan_lifecycle,"
        "native_opt0_reference_y_plan_bundle,native_opt0_raw_y_plan_bundle,"
        "native_opt0_y_plan_bundle_stream_first,native_opt0_raw_y_plan_bundle_reference_streams,"
        "native_opt0_reference_local_plan_context,directory";

    for ( std::size_t i = 0; i < rows.size(); ++i )
    {
        std::ostringstream row;
        row << fftm::test::detail::csv_quote( "fftm-native-stage" ) << ',' << rank << ',' << num_procs << ','
            << iteration << ',' << i << ',' << fftm::test::detail::csv_quote( rows[i].stage ) << ','
            << rows[i].ms << ',' << wall_ms << ','
            << fftm::test::detail::csv_quote( stage_timer_strategy_name( options.strategy ) ) << ','
            << fftm::test::detail::csv_quote( fftm::mpi_transpose_3d_mode_name( options.mode ) ) << ','
            << fftm::test::detail::csv_quote( fftm::test::detail::pencil_layout_name( options.pencil_layout ) )
            << ',' << fftm::test::detail::csv_quote(
                   fftm::test::detail::pencil_pipeline_name( options.pencil_pipeline ) )
            << ',' << fftm::test::detail::csv_quote(
                   fftm::fftm_3d_large_count_p2p_transport_name( options.large_count_p2p_transport ) )
            << ',' << ( options.use_fft_exec_no_sync ? 1 : 0 )
            << ',' << ( options.use_native_backward_second_peer_loop ? 1 : 0 ) << ','
            << ( options.use_native_opt0_default_z_layout ? 1 : 0 ) << ','
            << ( options.use_native_opt0_reference_y_buffer_topology ? 1 : 0 ) << ','
            << ( options.use_native_opt0_compact_y_workarea ? 1 : 0 ) << ','
            << ( options.use_native_opt0_tight_y_plan_sequence ? 1 : 0 ) << ','
            << ( options.use_native_opt0_shared_y_plan_handles ? 1 : 0 ) << ','
            << ( options.use_native_opt0_y_group_device_sync ? 1 : 0 ) << ','
            << ( options.use_native_opt0_y_no_sync_exec ? 1 : 0 ) << ','
            << ( options.use_native_opt0_raw_y_plan_array_executor ? 1 : 0 ) << ','
            << ( options.use_native_opt0_reference_y_plan_lifecycle ? 1 : 0 ) << ','
            << ( options.use_native_opt0_reference_y_plan_bundle ? 1 : 0 ) << ','
            << ( options.use_native_opt0_raw_y_plan_bundle ? 1 : 0 ) << ','
            << ( options.use_native_opt0_y_plan_bundle_stream_first ? 1 : 0 ) << ','
            << ( options.use_native_opt0_raw_y_plan_bundle_reference_streams ? 1 : 0 ) << ','
            << ( options.use_native_opt0_reference_local_plan_context ? 1 : 0 ) << ','
            << fftm::test::detail::csv_quote( options.directory );

        fftm::test::detail::append_csv_row(
            options.directory, native_stage_times_filename( rank ), header, row.str()
        );
    }
}

template <class Strategy>
int run_benchmark_case(
    scfd::utils::log_mpi &log, const options_t &options, const scfd::communication::mpi_comm_info &comm_info
)
{
    using fftm_t =
        fftm::fftm<base_fft_t, scfd::communication::mpi_comm_info, backend_t, Strategy, scfd::utils::log_mpi>;
    using strategy_traits = fftm::detail::fftm_3d_strategy_traits<Strategy>;
    using real_array_t  = typename fftm_t::template real_array_t<3>;
    using hat_array_t   = typename fftm_t::template complex_array_t<3>;
    using error_array_t = scfd::arrays::array_nd<T, 1, memory_t>;

    fftm::processor_grid grid;
    grid.init( options.p1, options.p2 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );

    if ( options.enable_local_fft_diagnostics || options.native_opt0_y_microbench )
    {
        fftm::test::detail::ensure_directory_exists(
            options.local_fft_diagnostics_directory.empty() ? options.directory
                                                           : options.local_fft_diagnostics_directory
        );
    }

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( grid, sizes );

    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();
    (void)myid_k;

    fftm::partition input_dim;
    fftm::partition transpose1_dim;
    fftm::partition output_dim;
    std::tie( input_dim, transpose1_dim, output_dim ) = partitioning.get_partitioning_3D();

    real_array_t work;
    hat_array_t  hat;

    auto checked_mul = []( std::size_t lhs, std::size_t rhs, const char *what ) -> std::size_t {
        if ( lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs )
        {
            std::ostringstream ss;
            ss << what << " overflows size_t";
            throw std::overflow_error( ss.str() );
        }
        return lhs * rhs;
    };
    auto tuple_elems = [&]( const auto &dims, const char *what ) -> std::size_t {
        const std::size_t xy = checked_mul( std::get<0>( dims ), std::get<1>( dims ), what );
        return checked_mul( xy, std::get<2>( dims ), what );
    };
    const auto preinit_in_sizes =
        std::make_tuple( input_dim.size_x[myid_i], input_dim.size_y[myid_j], input_dim.size_z[0] );
    const bool optimized_pencil_public_output =
        strategy_traits::family == fftm::transform_strategy_3d::pencil_pencil && strategy_traits::optimized_layout;
    const auto preinit_out_sizes = optimized_pencil_public_output
                                       ? std::make_tuple(
                                             output_dim.size_x[0], output_dim.size_y[myid_i], output_dim.size_z[myid_j]
                                         )
                                       : std::make_tuple(
                                             output_dim.size_x[0], output_dim.size_z[myid_j], output_dim.size_y[myid_i]
                                         );
    const std::size_t reserve_bytes = static_cast<std::size_t>( 512 ) * static_cast<std::size_t>( 1024 ) *
                                      static_cast<std::size_t>( 1024 );
    const std::size_t preinit_real_elems =
        tuple_elems( preinit_in_sizes, "benchmark real tensor element count" );
    const std::size_t preinit_complex_elems =
        tuple_elems( preinit_out_sizes, "benchmark complex tensor element count" );
    const std::size_t preinit_real_bytes =
        checked_mul( preinit_real_elems, sizeof( typename fftm_t::real ), "benchmark real tensor bytes" );
    const std::size_t preinit_complex_bytes =
        checked_mul( preinit_complex_elems, sizeof( typename fftm_t::complex ), "benchmark complex tensor bytes" );
    std::size_t public_tensor_reserve_bytes =
        checked_mul( 1, preinit_real_bytes, "benchmark public tensor memory reserve" );
    if ( preinit_complex_bytes > std::numeric_limits<std::size_t>::max() - public_tensor_reserve_bytes )
        throw std::overflow_error( "benchmark public tensor memory reserve overflows size_t" );
    public_tensor_reserve_bytes += preinit_complex_bytes;
    if ( reserve_bytes > std::numeric_limits<std::size_t>::max() - public_tensor_reserve_bytes )
        throw std::overflow_error( "benchmark public tensor memory reserve overflows size_t" );
    public_tensor_reserve_bytes += reserve_bytes;

    auto init_options = fftm::test::detail::make_fftm_init_options( options );
    if ( init_options.native_opt0_memory_feasibility_reserve_bytes < public_tensor_reserve_bytes )
    {
        init_options.native_opt0_memory_feasibility_reserve_bytes = public_tensor_reserve_bytes;
    }

    fftm_t distributed_fft( comm_info, log );
    distributed_fft.template init<3>( grid, sizes, init_options );

    const auto  in_sizes   = distributed_fft.get_local_input_sizes();
    const auto  out_sizes  = distributed_fft.get_local_output_sizes();
    const auto &input_part = distributed_fft.input_partition();

    const std::size_t real_elems    = tuple_elems( in_sizes, "benchmark real tensor element count" );
    const std::size_t complex_elems = tuple_elems( out_sizes, "benchmark complex tensor element count" );
    const std::size_t real_bytes    = checked_mul( real_elems, sizeof( typename fftm_t::real ), "benchmark real tensor bytes" );
    const std::size_t complex_bytes =
        checked_mul( complex_elems, sizeof( typename fftm_t::complex ), "benchmark complex tensor bytes" );
    std::size_t required_bytes = checked_mul( 1, real_bytes, "benchmark tensor memory preflight" );
    if ( complex_bytes > std::numeric_limits<std::size_t>::max() - required_bytes )
        throw std::overflow_error( "benchmark tensor memory preflight overflows size_t" );
    required_bytes += complex_bytes;
    if ( reserve_bytes > std::numeric_limits<std::size_t>::max() - required_bytes )
        throw std::overflow_error( "benchmark tensor memory preflight reserve overflows size_t" );
    required_bytes += reserve_bytes;
    const auto mem_info = runtime_api_t::get_device_memory_info();
    if ( mem_info.free_bytes_known && mem_info.free_bytes < required_bytes )
    {
        std::ostringstream ss;
        ss << "FFTM benchmark memory preflight failed before allocating public tensors: free="
           << ( mem_info.free_bytes / ( 1024 * 1024 ) ) << " MiB, required="
           << ( required_bytes / ( 1024 * 1024 ) ) << " MiB, tensor_bytes="
           << ( ( real_bytes + complex_bytes ) / ( 1024 * 1024 ) )
           << " MiB, reserve=" << ( reserve_bytes / ( 1024 * 1024 ) ) << " MiB";
        throw std::runtime_error( ss.str() );
    }

    work.init( std::get<0>( in_sizes ), std::get<1>( in_sizes ), std::get<2>( in_sizes ) );
    hat.init( std::get<0>( out_sizes ), std::get<1>( out_sizes ), std::get<2>( out_sizes ) );

    if ( options.native_opt0_y_microbench )
    {
        runtime_api_t::device_synchronize();
        distributed_fft.run_native_opt0_y_same_buffer_microbench(
            hat, static_cast<std::size_t>( options.native_opt0_y_microbench_iterations ),
            static_cast<std::size_t>( options.native_opt0_y_microbench_warmup )
        );
        runtime_api_t::device_synchronize();
        if ( comm_info.myid == 0 )
        {
            log.info_f(
                "benchmark=fftm-3d, diagnostic=native_opt0_y_microbench, grid=(%zu,%zu), Nx=%zu, Ny=%zu, Nz=%zu, "
                "iterations=%d, warmup=%d: complete",
                options.p1, options.p2, options.nx, options.ny, options.nz,
                options.native_opt0_y_microbench_iterations, options.native_opt0_y_microbench_warmup
            );
        }
        return 0;
    }

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    const T        normalization = T( 1 ) / static_cast<T>( options.nx * options.ny * options.nz );
    std::vector<T> wall_times;
    wall_times.reserve( static_cast<std::size_t>( options.times ) );
    T    max_norm          = T( 0 );
    bool validation_failed = false;

    auto run_untimed_iteration = [&]( int iter, unsigned long long seed_base ) {
        const unsigned long long seed = seed_base + static_cast<unsigned long long>( iter );

        for_each(
            fftm::test::detail::fill_random_real_3d_functor<T, idx_t, real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), 0 },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

        runtime_api_t::device_synchronize();
        distributed_fft.forward( work, hat );
        distributed_fft.backward( hat, work );
        runtime_api_t::device_synchronize();
    };

    for ( int iter = 0; iter < options.contiguous_forward_send_registration_warmups; ++iter )
    {
        run_untimed_iteration( iter, 0x2718281828459045ull );
    }

    for ( int iter = 0; iter < options.warmup; ++iter )
    {
        run_untimed_iteration( iter, 0x3141592653589793ull );
    }

    for ( int iter = 0; iter < options.times; ++iter )
    {
        const unsigned long long seed = 0x3141592653589793ull + static_cast<unsigned long long>( iter );

        for_each(
            fftm::test::detail::fill_random_real_3d_functor<T, idx_t, real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), 0 },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

	        scfd::utils::system_timer_event t0, t1;
	        runtime_api_t::device_synchronize();
            if ( options.enable_native_stage_timers )
            {
                distributed_fft.begin_native_stage_timing_iteration( iter );
            }
	        t0.record();
	        distributed_fft.forward( work, hat );
	        distributed_fft.backward( hat, work );
	        runtime_api_t::device_synchronize();
	        t1.record();
	        const T wall_ms = static_cast<T>( t1.elapsed_time( t0 ) );
	        wall_times.push_back( wall_ms );
            if ( options.enable_native_stage_timers )
            {
                append_native_stage_timer_rows(
                    options, comm_info.num_procs, comm_info.myid, iter, static_cast<double>( wall_ms ),
                    distributed_fft.native_stage_timings()
                );
                distributed_fft.end_native_stage_timing_iteration();
            }

        for_each(
            fftm::test::detail::scale_real_functor<T, idx_t, real_array_t>{ work, normalization },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

        for_each(
            fftm::test::detail::overwrite_with_random_diff_square_3d_functor<T, idx_t, real_array_t>{
                work, seed, static_cast<int>( input_part.start_x[myid_i] ),
                static_cast<int>( input_part.start_y[myid_j] ), 0 },
            fftm::test::detail::make_range_3d<idx_t, rect_t>( work )
        );
        for_each.wait();

        const T local_diff_sq  = reduce( work.size(), work.raw_ptr(), T( 0 ) );
        const T global_diff_sq = comm_info.all_reduce_sum( local_diff_sq );
        const T diff_l2        = std::sqrt( global_diff_sq / static_cast<T>( options.nx * options.ny * options.nz ) );
        if ( diff_l2 > max_norm )
            max_norm = diff_l2;

        if ( diff_l2 > options.epsilon && comm_info.myid == 0 )
        {
            log.warning_f(
                "strategy=%s, mode=%s, iteration=%d: l2_diff=%.8e exceeded epsilon=%.8e", fftm_t::strategy_name(),
                fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_3d ), iter, diff_l2, options.epsilon
            );
        }
        if ( diff_l2 > options.epsilon )
            validation_failed = true;
    }

    const auto stats = fftm::test::detail::compute_timing_statistics( wall_times );
    fftm::test::detail::log_tracked_memory_with_external_mpi(
        log, comm_info, distributed_fft, "benchmark=fftm-3d",
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( work ), fftm::test::detail::array_bytes( hat ) ) ),
        static_cast<typename fftm_t::memory_profile_bytes_t>( fftm::test::detail::sum_bytes(
            fftm::test::detail::array_bytes( work ), fftm::test::detail::array_bytes( hat ) ) )
    );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "benchmark=fftm-3d, strategy=%s, mode=%s, grid=(%zu,%zu), Nx=%zu, Ny=%zu, Nz=%zu, warmup=%d, times=%d, "
            "pencil_layout=%s, pencil_pipeline=%s, persistent_p2p=%d, ready_p2p_send=%d, "
            "large_count_p2p_transport=%s, fft_exec_no_sync=%d, stable_forward_byte_send_buffer=%d, "
            "ready_stable_forward_byte_send_buffer=%d, contiguous_forward_byte_send=%d, "
            "physical_forward_peer_exchange=%d, native_backward_second_peer_loop=%d, "
            "native_opt0_default_z_layout=%d, native_opt0_reference_y_buffer_topology=%d, "
            "native_opt0_compact_y_workarea=%d, "
            "native_opt0_tight_y_plan_sequence=%d, native_opt0_shared_y_plan_handles=%d, "
            "native_opt0_y_group_device_sync=%d, native_opt0_y_no_sync_exec=%d, "
            "native_opt0_raw_y_plan_array_executor=%d, "
            "native_opt0_reference_y_plan_lifecycle=%d, native_opt0_reference_y_plan_bundle=%d, "
            "native_opt0_raw_y_plan_bundle=%d, native_opt0_y_plan_bundle_stream_first=%d, "
            "native_opt0_raw_y_plan_bundle_reference_streams=%d, native_opt0_reference_local_plan_context=%d, "
            "contiguous_forward_send_mode=%s, contiguous_forward_send_chunk_mib=%zu, "
            "contiguous_forward_send_registration_warmups=%d: "
            "avg_wall_ms=%.8e, stddev_wall_ms=%.8e",
            fftm_t::strategy_name(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_3d ), options.p1,
            options.p2, options.nx, options.ny, options.nz, options.warmup, options.times,
            fftm::test::detail::pencil_layout_name( options.pencil_layout ),
            fftm::test::detail::pencil_pipeline_name( options.pencil_pipeline ),
            options.use_persistent_p2p ? 1 : 0, options.use_ready_p2p_send ? 1 : 0,
            fftm::fftm_3d_large_count_p2p_transport_name( options.large_count_p2p_transport ),
            options.use_fft_exec_no_sync ? 1 : 0,
            options.use_stable_forward_byte_send_buffer ? 1 : 0,
            options.use_ready_stable_forward_byte_send_buffer ? 1 : 0,
            options.use_contiguous_forward_byte_send ? 1 : 0,
            options.use_physical_forward_peer_exchange ? 1 : 0,
            options.use_native_backward_second_peer_loop ? 1 : 0,
            options.use_native_opt0_default_z_layout ? 1 : 0,
            options.use_native_opt0_reference_y_buffer_topology ? 1 : 0,
            options.use_native_opt0_compact_y_workarea ? 1 : 0,
            options.use_native_opt0_tight_y_plan_sequence ? 1 : 0,
            options.use_native_opt0_shared_y_plan_handles ? 1 : 0,
            options.use_native_opt0_y_group_device_sync ? 1 : 0,
            options.use_native_opt0_y_no_sync_exec ? 1 : 0,
            options.use_native_opt0_raw_y_plan_array_executor ? 1 : 0,
            options.use_native_opt0_reference_y_plan_lifecycle ? 1 : 0,
            options.use_native_opt0_reference_y_plan_bundle ? 1 : 0,
            options.use_native_opt0_raw_y_plan_bundle ? 1 : 0,
            options.use_native_opt0_y_plan_bundle_stream_first ? 1 : 0,
            options.use_native_opt0_raw_y_plan_bundle_reference_streams ? 1 : 0,
            options.use_native_opt0_reference_local_plan_context ? 1 : 0,
            fftm::fftm_3d_contiguous_forward_send_mode_name( options.contiguous_forward_send_mode ),
            options.contiguous_forward_send_chunk_bytes / ( static_cast<std::size_t>( 1024 ) *
                                                            static_cast<std::size_t>( 1024 ) ),
            options.contiguous_forward_send_registration_warmups, stats.mean, stats.stddev
        );

        std::ostringstream row;
        row << fftm::test::detail::csv_quote( "fftm-3d" ) << ',' << comm_info.num_procs << ','
            << fftm::test::detail::csv_quote( fftm_t::strategy_name() ) << ','
            << fftm::test::detail::csv_quote( fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_3d ) ) << ','
            << fftm::test::detail::csv_quote( fftm::test::detail::pencil_layout_name( options.pencil_layout ) ) << ','
            << fftm::test::detail::csv_quote( fftm::test::detail::pencil_pipeline_name( options.pencil_pipeline ) ) << ','
            << ( options.use_persistent_p2p ? 1 : 0 ) << ',' << ( options.use_ready_p2p_send ? 1 : 0 ) << ','
            << fftm::test::detail::csv_quote(
                   fftm::fftm_3d_large_count_p2p_transport_name( options.large_count_p2p_transport ) )
            << ',' << ( options.use_fft_exec_no_sync ? 1 : 0 )
            << ',' << ( options.use_stable_forward_byte_send_buffer ? 1 : 0 ) << ','
            << ( options.use_ready_stable_forward_byte_send_buffer ? 1 : 0 ) << ','
            << ( options.use_contiguous_forward_byte_send ? 1 : 0 ) << ','
            << ( options.use_physical_forward_peer_exchange ? 1 : 0 ) << ','
            << ( options.use_native_backward_second_peer_loop ? 1 : 0 ) << ','
            << ( options.use_native_opt0_default_z_layout ? 1 : 0 ) << ','
            << ( options.use_native_opt0_reference_y_buffer_topology ? 1 : 0 ) << ','
            << ( options.use_native_opt0_compact_y_workarea ? 1 : 0 ) << ','
            << ( options.use_native_opt0_tight_y_plan_sequence ? 1 : 0 ) << ','
            << ( options.use_native_opt0_shared_y_plan_handles ? 1 : 0 ) << ','
            << ( options.use_native_opt0_y_group_device_sync ? 1 : 0 ) << ','
            << ( options.use_native_opt0_y_no_sync_exec ? 1 : 0 ) << ','
            << ( options.use_native_opt0_raw_y_plan_array_executor ? 1 : 0 ) << ','
            << ( options.use_native_opt0_reference_y_plan_lifecycle ? 1 : 0 ) << ','
            << ( options.use_native_opt0_reference_y_plan_bundle ? 1 : 0 ) << ','
            << ( options.use_native_opt0_raw_y_plan_bundle ? 1 : 0 ) << ','
            << ( options.use_native_opt0_y_plan_bundle_stream_first ? 1 : 0 ) << ','
            << ( options.use_native_opt0_raw_y_plan_bundle_reference_streams ? 1 : 0 ) << ','
            << ( options.use_native_opt0_reference_local_plan_context ? 1 : 0 ) << ','
            << fftm::test::detail::csv_quote(
                   fftm::fftm_3d_contiguous_forward_send_mode_name( options.contiguous_forward_send_mode ) )
            << ','
            << options.contiguous_forward_send_chunk_bytes / ( static_cast<std::size_t>( 1024 ) *
                                                               static_cast<std::size_t>( 1024 ) )
            << ',' << options.contiguous_forward_send_registration_warmups << ','
            << options.p1 << ',' << options.p2 << ',' << 1 << ',' << options.nx << ',' << options.ny << ','
            << options.nz << ',' << 0 << ',' << options.times << ',' << options.warmup << ',' << options.epsilon
            << ',' << stats.mean << ',' << stats.stddev << ',' << max_norm << ','
            << fftm::test::detail::csv_quote( options.directory );

        fftm::test::detail::append_csv_row(
            options.directory, "benchmark_fftm_3d.csv",
            "benchmark,num_gpus,strategy,mode,pencil_layout,pencil_pipeline,persistent_p2p,ready_p2p_send,large_count_p2p_transport,fft_exec_no_sync,stable_forward_byte_send_buffer,ready_stable_forward_byte_send_buffer,contiguous_forward_byte_send,physical_forward_peer_exchange,native_backward_second_peer_loop,native_opt0_default_z_layout,native_opt0_reference_y_buffer_topology,native_opt0_compact_y_workarea,native_opt0_tight_y_plan_sequence,native_opt0_shared_y_plan_handles,native_opt0_y_group_device_sync,native_opt0_y_no_sync_exec,native_opt0_raw_y_plan_array_executor,native_opt0_reference_y_plan_lifecycle,native_opt0_reference_y_plan_bundle,native_opt0_raw_y_plan_bundle,native_opt0_y_plan_bundle_stream_first,native_opt0_raw_y_plan_bundle_reference_streams,native_opt0_reference_local_plan_context,contiguous_forward_send_mode,contiguous_forward_send_chunk_mib,contiguous_forward_send_registration_warmups,p1,p2,p3,nx,ny,nz,nw,times,warmup,epsilon,avg_wall_ms,stddev_wall_ms,"
            "max_l2_diff,directory",
            row.str()
        );
    }

    if ( validation_failed && comm_info.myid == 0 )
    {
        log.error_f(
            "benchmark=fftm-3d validation failed: max_l2_diff=%.8e exceeded epsilon=%.8e", max_norm,
            options.epsilon
        );
    }

    return validation_failed ? 1 : 0;
}

template <fftm::mpi_transpose_3d_mode Mode>
int dispatch_mode(
    strategy_kind strategy, scfd::utils::log_mpi &log, const options_t &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    switch ( strategy )
    {
    case strategy_kind::slab_pencil:
        return run_benchmark_case<fftm::strategy_3d_slab_pencil<Mode>>( log, options, comm_info );
    case strategy_kind::pencil_slab:
        return run_benchmark_case<fftm::strategy_3d_pencil_slab<Mode>>( log, options, comm_info );
    case strategy_kind::pencil_pencil:
        if ( options.pencil_layout == fftm::test::detail::fftm_3d_pencil_layout_kind::legacy )
        {
            return run_benchmark_case<fftm::strategy_3d_pencil_pencil<Mode, false>>( log, options, comm_info );
        }
        return run_benchmark_case<fftm::strategy_3d_pencil_pencil<Mode, true>>( log, options, comm_info );
    }
    return 1;
}

int dispatch_strategy(
    strategy_kind strategy, scfd::utils::log_mpi &log, const options_t &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    switch ( options.mode )
    {
    case fftm::mpi_transpose_3d_mode::p2p_waitall:
        return dispatch_mode<fftm::mpi_transpose_3d_mode::p2p_waitall>( strategy, log, options, comm_info );
    case fftm::mpi_transpose_3d_mode::p2p_waitany:
        return dispatch_mode<fftm::mpi_transpose_3d_mode::p2p_waitany>( strategy, log, options, comm_info );
    case fftm::mpi_transpose_3d_mode::alltoallv:
        return dispatch_mode<fftm::mpi_transpose_3d_mode::alltoallv>( strategy, log, options, comm_info );
    case fftm::mpi_transpose_3d_mode::alltoallw:
        return dispatch_mode<fftm::mpi_transpose_3d_mode::alltoallw>( strategy, log, options, comm_info );
    }
    return 1;
}

} // namespace

int main( int argc, char *argv[] )
{
    scfd::communication::mpi_wrap mpi( argc, argv, MPI_THREAD_MULTIPLE );
    auto                          comm_info = mpi.comm_world();
    scfd::utils::log_mpi          log;

    try
    {
        fftm::test::detail::init_cuda_mpi_for_tests( log, comm_info );
        const options_t options = fftm::test::detail::parse_fftm_3d_benchmark_options<T>(
            argc, argv, comm_info.num_procs, "test_benchmark_fftm_3D.bin"
        );

        if ( options.p1 * options.p2 != static_cast<std::size_t>( comm_info.num_procs ) )
            throw std::logic_error( "P1*P2 must equal the number of MPI processes" );

        const int native_schedule_status =
            fftm::test::detail::native_pencil_schedule::run( log, options, comm_info );
        if ( native_schedule_status != 0 )
            return native_schedule_status;
        if ( options.native_pencil_schedule_check_only )
            return 0;

        append_gpu_telemetry_row( options, comm_info, "pre" );

        int status = 0;
        if ( options.native_opt0_y_cross_microbench )
        {
            status = run_native_opt0_y_cross_microbench( log, options, comm_info );
            append_gpu_telemetry_row( options, comm_info, "post" );
            return status;
        }

        if ( options.backend != fftm::test::detail::fftm_3d_backend_kind::native )
        {
            status = run_fftm3d_scfd_fft_facade_backend( log, options, comm_info );
            append_gpu_telemetry_row( options, comm_info, "post" );
            return status;
        }

        if ( options.run_all )
        {
            status |= dispatch_strategy( strategy_kind::slab_pencil, log, options, comm_info );
            status |= dispatch_strategy( strategy_kind::pencil_slab, log, options, comm_info );
            status |= dispatch_strategy( strategy_kind::pencil_pencil, log, options, comm_info );
            append_gpu_telemetry_row( options, comm_info, "post" );
            return status;
        }

        status = dispatch_strategy( options.strategy, log, options, comm_info );
        append_gpu_telemetry_row( options, comm_info, "post" );
        return status;
    }
    catch ( const std::exception &ex )
    {
        log.error( scfd::utils::nested_exception_to_multistring( ex ) );
        return 1;
    }
}
