#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>


#include <scfd/arrays/array_nd.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/rect.h>
#include <scfd/static_vec/vec.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/nested_exception_to_multistring.h>
#include <scfd/utils/scalar_traits.h>

#include <fftm.hpp>
#include <ffts.hpp>

#include "detail/fft_test_backend.h"

namespace
{

using T          = double;
using base_fft_t = fftm::test::detail::fft_test_wrap_many<T>;
using backend_t  = fftm::test::detail::fft_test_backend;
using memory_t   = backend_t::memory_type;
using reduce_t   = backend_t::reduce_type;
using for_each_t = backend_t::template for_each_nd_type<3, int>;
using idx_t      = scfd::static_vec::vec<int, 3>;
using rect_t     = scfd::static_vec::rect<int, 3>;
using ref_ffts_t = fftm::ffts<base_fft_t, backend_t>;

enum class strategy_kind
{
    slab_pencil,
    pencil_slab,
    pencil_pencil
};

struct test_options
{
    strategy_kind               strategy  = strategy_kind::slab_pencil;
    bool                        run_all   = false;
    fftm::mpi_transpose_3d_mode mode      = fftm::mpi_transpose_3d_mode::alltoallv;
    std::size_t                 nx        = 16;
    std::size_t                 ny        = 18;
    std::size_t                 nz        = 20;
    std::size_t                 p1        = 0;
    std::size_t                 p2        = 0;
    T                           threshold = T( 1.0e-11 );
    bool                                      use_direct_backward_receive = false;
    bool                                      direct_p2p_cuda_aware       = true;
    bool                                      use_p2p_byte_transfer       = false;
    bool                                      print_pencil_schedule       = false;
    bool                                      use_direct_forward_byte_receive = false;
    bool                                      use_stable_forward_byte_send_buffer = false;
    bool                                      use_ready_stable_forward_byte_send_buffer = false;
    bool                                      use_contiguous_forward_byte_send = false;
    bool                                      use_physical_forward_peer_exchange = false;
    fftm::fftm_3d_contiguous_forward_send_mode contiguous_forward_send_mode =
        fftm::fftm_3d_contiguous_forward_send_mode::single;
    std::size_t                               contiguous_forward_send_chunk_bytes =
        static_cast<std::size_t>( 1 ) << 30;
    fftm::fftm_3d_large_count_p2p_transport  large_count_p2p_transport =
        fftm::fftm_3d_large_count_p2p_transport::hindexed;
    bool                                      use_large_count_datatype_cache = false;
    bool                                      use_native_backward_second_peer_loop = false;
    fftm::fftm_3d_pencil_pipeline            pencil_pipeline = fftm::fftm_3d_pencil_pipeline::staged;
};

std::pair<std::size_t, std::size_t> choose_pencil_grid( std::size_t num_procs )
{
    std::size_t p1 = 1;
    for ( std::size_t d = 1; d * d <= num_procs; ++d )
    {
        if ( num_procs % d == 0 )
            p1 = d;
    }
    return std::make_pair( p1, num_procs / p1 );
}

test_options parse_options( int argc, char *argv[], int num_procs )
{
    test_options options;
    int          argi = 1;

    while ( argi < argc )
    {
        const std::string arg = argv[argi];

        if ( arg == "--strategy" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --strategy" );
            const std::string value = argv[argi + 1];
            if ( value == "slab-pencil" )
                options.strategy = strategy_kind::slab_pencil;
            else if ( value == "pencil-slab" )
                options.strategy = strategy_kind::pencil_slab;
            else if ( value == "pencil-pencil" )
                options.strategy = strategy_kind::pencil_pencil;
            else if ( value == "all" )
                options.run_all = true;
            else
                throw std::logic_error( "Unknown strategy '" + value + "'" );
            argi += 2;
        }
        else if ( arg == "--mode" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --mode" );
            const std::string value = argv[argi + 1];
            if ( value == "p2p-waitall" )
                options.mode = fftm::mpi_transpose_3d_mode::p2p_waitall;
            else if ( value == "p2p-waitany" )
                options.mode = fftm::mpi_transpose_3d_mode::p2p_waitany;
            else if ( value == "alltoallv" )
                options.mode = fftm::mpi_transpose_3d_mode::alltoallv;
            else if ( value == "alltoallw" )
                options.mode = fftm::mpi_transpose_3d_mode::alltoallw;
            else
                throw std::logic_error( "Unknown mode '" + value + "'" );
            argi += 2;
        }
        else if ( arg == "--grid" )
        {
            if ( argi + 2 >= argc )
                throw std::logic_error( "Missing values for --grid P1 P2" );
            options.p1 = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            options.p2 = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
            argi += 3;
        }
        else if ( arg == "--threshold" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --threshold" );
            options.threshold = static_cast<T>( std::atof( argv[argi + 1] ) );
            argi += 2;
        }
        else if ( arg == "--use-direct-backward-receive" )
        {
            options.use_direct_backward_receive = true;
            argi += 1;
        }
        else if ( arg == "--no-direct-backward-receive" )
        {
            options.use_direct_backward_receive = false;
            argi += 1;
        }
        else if ( arg == "--direct-p2p-cuda-aware" || arg == "--direct-p2p-CUDA-aware" )
        {
            options.direct_p2p_cuda_aware = true;
            argi += 1;
        }
        else if ( arg == "--no-direct-p2p-cuda-aware" || arg == "--no-direct-p2p-CUDA-aware" )
        {
            options.direct_p2p_cuda_aware = false;
            argi += 1;
        }
        else if ( arg == "--use-p2p-byte-transfer" )
        {
            options.use_p2p_byte_transfer = true;
            argi += 1;
        }
        else if ( arg == "--no-p2p-byte-transfer" )
        {
            options.use_p2p_byte_transfer = false;
            argi += 1;
        }
        else if ( arg == "--print-pencil-schedule" )
        {
            options.print_pencil_schedule = true;
            argi += 1;
        }
        else if ( arg == "--no-print-pencil-schedule" )
        {
            options.print_pencil_schedule = false;
            argi += 1;
        }
        else if ( arg == "--use-direct-forward-byte-receive" )
        {
            options.use_direct_forward_byte_receive = true;
            argi += 1;
        }
        else if ( arg == "--no-direct-forward-byte-receive" )
        {
            options.use_direct_forward_byte_receive = false;
            argi += 1;
        }
        else if ( arg == "--use-stable-forward-byte-send-buffer" )
        {
            options.use_stable_forward_byte_send_buffer = true;
            argi += 1;
        }
        else if ( arg == "--no-stable-forward-byte-send-buffer" )
        {
            options.use_stable_forward_byte_send_buffer = false;
            argi += 1;
        }
        else if ( arg == "--use-ready-stable-forward-byte-send-buffer" )
        {
            options.use_ready_stable_forward_byte_send_buffer = true;
            argi += 1;
        }
        else if ( arg == "--no-ready-stable-forward-byte-send-buffer" )
        {
            options.use_ready_stable_forward_byte_send_buffer = false;
            argi += 1;
        }
        else if ( arg == "--use-contiguous-forward-byte-send" )
        {
            options.use_contiguous_forward_byte_send = true;
            argi += 1;
        }
        else if ( arg == "--no-contiguous-forward-byte-send" )
        {
            options.use_contiguous_forward_byte_send = false;
            argi += 1;
        }
        else if ( arg == "--use-physical-forward-peer-exchange" )
        {
            options.use_physical_forward_peer_exchange = true;
            argi += 1;
        }
        else if ( arg == "--no-physical-forward-peer-exchange" )
        {
            options.use_physical_forward_peer_exchange = false;
            argi += 1;
        }
        else if ( arg == "--contiguous-forward-send-mode" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --contiguous-forward-send-mode" );
            const std::string value = argv[argi + 1];
            if ( value == "single" )
                options.contiguous_forward_send_mode = fftm::fftm_3d_contiguous_forward_send_mode::single;
            else if ( value == "chunked" )
                options.contiguous_forward_send_mode = fftm::fftm_3d_contiguous_forward_send_mode::chunked;
            else
                throw std::logic_error( "Unknown contiguous forward send mode '" + value + "'" );
            argi += 2;
        }
        else if ( arg == "--contiguous-forward-send-chunk-mib" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --contiguous-forward-send-chunk-mib" );
            const std::size_t mib = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
            if ( mib == 0 )
                throw std::logic_error( "--contiguous-forward-send-chunk-mib must be positive" );
            options.contiguous_forward_send_chunk_bytes = mib * static_cast<std::size_t>( 1024 ) *
                                                          static_cast<std::size_t>( 1024 );
            argi += 2;
        }
        else if ( arg == "--large-count-p2p-transport" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --large-count-p2p-transport" );
            const std::string value = argv[argi + 1];
            if ( value == "hindexed" )
                options.large_count_p2p_transport = fftm::fftm_3d_large_count_p2p_transport::hindexed;
            else if ( value == "mpi-count" || value == "mpi_count" )
                options.large_count_p2p_transport = fftm::fftm_3d_large_count_p2p_transport::mpi_count;
            else if ( value == "element-count" || value == "element_count" || value == "complex-count" ||
                      value == "complex_count" )
                options.large_count_p2p_transport = fftm::fftm_3d_large_count_p2p_transport::element_count;
            else if ( value == "chunked" )
                options.large_count_p2p_transport = fftm::fftm_3d_large_count_p2p_transport::chunked;
            else
                throw std::logic_error( "Unknown large-count P2P transport '" + value + "'" );
            argi += 2;
        }
        else if ( arg == "--use-large-count-datatype-cache" )
        {
            options.use_large_count_datatype_cache = true;
            argi += 1;
        }
        else if ( arg == "--no-large-count-datatype-cache" )
        {
            options.use_large_count_datatype_cache = false;
            argi += 1;
        }
        else if ( arg == "--use-native-backward-second-peer-loop" )
        {
            options.use_native_backward_second_peer_loop = true;
            argi += 1;
        }
        else if ( arg == "--no-native-backward-second-peer-loop" )
        {
            options.use_native_backward_second_peer_loop = false;
            argi += 1;
        }
        else if ( arg == "--pencil-pipeline" )
        {
            if ( argi + 1 >= argc )
                throw std::logic_error( "Missing value for --pencil-pipeline" );
            const std::string value = argv[argi + 1];
            if ( value == "staged" )
                options.pencil_pipeline = fftm::fftm_3d_pencil_pipeline::staged;
            else if ( value == "fused" )
                options.pencil_pipeline = fftm::fftm_3d_pencil_pipeline::fused;
            else if ( value == "reference" || value == "egger" )
                options.pencil_pipeline = fftm::fftm_3d_pencil_pipeline::reference;
            else if ( value == "reference-parity" || value == "reference_parity" ||
                      value == "egger-parity" || value == "egger_parity" )
                options.pencil_pipeline = fftm::fftm_3d_pencil_pipeline::reference_parity;
            else
                throw std::logic_error( "Unknown pencil pipeline '" + value + "'" );
            argi += 2;
        }
        else
        {
            break;
        }
    }

    if ( argc - argi == 3 )
    {
        options.nx = static_cast<std::size_t>( std::strtoull( argv[argi], NULL, 10 ) );
        options.ny = static_cast<std::size_t>( std::strtoull( argv[argi + 1], NULL, 10 ) );
        options.nz = static_cast<std::size_t>( std::strtoull( argv[argi + 2], NULL, 10 ) );
    }
    else if ( argc != argi )
    {
        throw std::logic_error(
            "USAGE: test_fftm_3D_compare.bin [--strategy slab-pencil|pencil-slab|pencil-pencil|all] "
            "[--mode p2p-waitall|p2p-waitany|alltoallv|alltoallw] [--grid P1 P2] [--threshold eps] "
            "[--use-direct-backward-receive|--no-direct-backward-receive] "
	            "[--direct-p2p-cuda-aware|--no-direct-p2p-cuda-aware] "
            "[--use-p2p-byte-transfer|--no-p2p-byte-transfer] "
            "[--print-pencil-schedule|--no-print-pencil-schedule] "
            "[--use-direct-forward-byte-receive|--no-direct-forward-byte-receive] "
            "[--use-stable-forward-byte-send-buffer|--no-stable-forward-byte-send-buffer] "
            "[--use-ready-stable-forward-byte-send-buffer|--no-ready-stable-forward-byte-send-buffer] "
            "[--use-contiguous-forward-byte-send|--no-contiguous-forward-byte-send] "
            "[--use-physical-forward-peer-exchange|--no-physical-forward-peer-exchange] "
            "[--contiguous-forward-send-mode single|chunked] "
            "[--contiguous-forward-send-chunk-mib MiB] "
            "[--large-count-p2p-transport hindexed|mpi-count|element-count|chunked] "
            "[--use-large-count-datatype-cache|--no-large-count-datatype-cache] "
            "[--use-native-backward-second-peer-loop|--no-native-backward-second-peer-loop] "
            "[--pencil-pipeline staged|fused|egger|egger-parity] [Nx Ny Nz]"
        );
    }

    if ( options.p1 == 0 || options.p2 == 0 )
    {
        const auto pencil_grid = choose_pencil_grid( static_cast<std::size_t>( num_procs ) );
        if ( options.strategy == strategy_kind::slab_pencil )
        {
            options.p1 = static_cast<std::size_t>( num_procs );
            options.p2 = 1;
        }
        else if ( options.strategy == strategy_kind::pencil_slab )
        {
            options.p1 = 1;
            options.p2 = static_cast<std::size_t>( num_procs );
        }
        else
        {
            options.p1 = pencil_grid.first;
            options.p2 = pencil_grid.second;
        }
    }

    return options;
}

fftm::fftm_init_options make_init_options( const test_options &options )
{
    fftm::fftm_init_options init_options;
    init_options.reporting = fftm::profiling_reporting_options();
    init_options.execution.use_direct_backward_receive = options.use_direct_backward_receive;
    init_options.execution.direct_p2p_cuda_aware      = options.direct_p2p_cuda_aware;
    init_options.execution.use_p2p_byte_transfer      = options.use_p2p_byte_transfer;
    init_options.diagnostics.print_pencil_schedule      = options.print_pencil_schedule;
    init_options.diagnostics.use_direct_forward_byte_receive = options.use_direct_forward_byte_receive;
    init_options.diagnostics.use_stable_forward_byte_send_buffer = options.use_stable_forward_byte_send_buffer;
    init_options.diagnostics.use_ready_stable_forward_byte_send_buffer = options.use_ready_stable_forward_byte_send_buffer;
    init_options.diagnostics.use_contiguous_forward_byte_send = options.use_contiguous_forward_byte_send;
    init_options.diagnostics.use_physical_forward_peer_exchange = options.use_physical_forward_peer_exchange;
    init_options.diagnostics.contiguous_forward_send_mode = options.contiguous_forward_send_mode;
    init_options.diagnostics.contiguous_forward_send_chunk_bytes = options.contiguous_forward_send_chunk_bytes;
    init_options.execution.large_count_p2p_transport = options.large_count_p2p_transport;
    init_options.diagnostics.use_large_count_datatype_cache = options.use_large_count_datatype_cache;
    init_options.diagnostics.use_native_backward_second_peer_loop = options.use_native_backward_second_peer_loop;
    init_options.pencil_pipeline_3d         = options.pencil_pipeline;
    return init_options;
}

template <class Array>
rect_t make_range( const Array &array )
{
    const auto sz = array.size_nd();
    return rect_t(
        idx_t( 0, 0, 0 ), idx_t( static_cast<int>( sz[0] ), static_cast<int>( sz[1] ), static_cast<int>( sz[2] ) )
    );
}

__DEVICE_TAG__ T sample_value( T x, T y, T z )
{
    return scfd::utils::scalar_traits<T>::sin( x ) + T( 0.5 ) * scfd::utils::scalar_traits<T>::cos( y ) -
           T( 0.25 ) * scfd::utils::scalar_traits<T>::sin( T( 2 ) * z ) +
           T( 0.125 ) * scfd::utils::scalar_traits<T>::sin( x + y - z );
}

template <class Array>
struct fill_input_functor
{
    Array array;
    T     hx;
    T     hy;
    T     hz;
    T     x0;
    T     y0;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const T x    = x0 + hx * static_cast<T>( idx[0] );
        const T y    = y0 + hy * static_cast<T>( idx[1] );
        const T z    = hz * static_cast<T>( idx[2] );
        array( idx ) = sample_value( x, y, z );
    }
};

template <class LocalArray, class RefArray, class ErrorArray>
struct compare_output_functor
{
    LocalArray local;
    RefArray   reference;
    ErrorArray diff_sq;
    ErrorArray ref_sq;
    int        y_start;
    int        z_start;
    bool       output_is_xyz;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const auto actual   = local( idx );
        const auto expected = output_is_xyz ? reference( idx[0], y_start + idx[1], z_start + idx[2] )
                                            : reference( idx[0], y_start + idx[2], z_start + idx[1] );

        const T diff_re = actual.x - expected.x;
        const T diff_im = actual.y - expected.y;
        const T ref_re  = expected.x;
        const T ref_im  = expected.y;

        diff_sq( idx ) = diff_re * diff_re + diff_im * diff_im;
        ref_sq( idx )  = ref_re * ref_re + ref_im * ref_im;
    }
};

template <class LocalArray, class RefArray, class ErrorArray>
struct compare_real_output_functor
{
    LocalArray local;
    RefArray   reference;
    ErrorArray diff_sq;
    ErrorArray ref_sq;
    int        x_start;
    int        y_start;
    int        z_start;

    __DEVICE_TAG__ void operator()( const idx_t &idx ) const
    {
        const T actual   = local( idx );
        const T expected = reference( x_start + idx[0], y_start + idx[1], z_start + idx[2] );

        const T diff = actual - expected;

        diff_sq( idx ) = diff * diff;
        ref_sq( idx )  = expected * expected;
    }
};

template <class Strategy>
int run_compare(
    scfd::utils::log_mpi &log, const test_options &options, const scfd::communication::mpi_comm_info &comm_info
)
{
    using fftm_t =
        fftm::fftm<base_fft_t, scfd::communication::mpi_comm_info, backend_t, Strategy, scfd::utils::log_mpi>;
    using local_real_t     = typename fftm_t::template real_array_t<3>;
    using local_hat_t      = typename fftm_t::template complex_array_t<3>;
    using ref_real_t       = typename ref_ffts_t::template real_array_t<3>;
    using ref_hat_t        = typename ref_ffts_t::template complex_array_t<3>;
    using err_hat_array_t  = scfd::arrays::array_nd<T, 3, memory_t, scfd::arrays::custom_arranger_201_t>;
    using err_real_array_t = scfd::arrays::array_nd<T, 3, memory_t, scfd::arrays::custom_arranger_102_t>;

    fftm::processor_grid grid;
    grid.init( options.p1, options.p2 );

    fftm::global_sizes sizes;
    sizes.init( options.nx, options.ny, options.nz );

    fftm::fft_partitioning<scfd::communication::mpi_comm_info> partitioning( comm_info );
    partitioning.init( grid, sizes );
    int myid_i = 0, myid_j = 0, myid_k = 0;
    std::tie( myid_i, myid_j, myid_k ) = partitioning.get_my_grid();

    fftm_t     distributed_fft( comm_info, log );
    ref_ffts_t reference_fft;

    distributed_fft.template init<3>( grid, sizes, make_init_options( options ) );
    reference_fft.init( options.nx, options.ny, options.nz );

    local_real_t local_in;
    local_real_t local_back;
    local_hat_t  local_out;
    ref_real_t   ref_in;
    ref_real_t   ref_back;
    ref_hat_t    ref_out;

    const auto local_in_sizes  = distributed_fft.get_local_input_sizes();
    const auto local_out_sizes = distributed_fft.get_local_output_sizes();

    local_in.init( std::get<0>( local_in_sizes ), std::get<1>( local_in_sizes ), std::get<2>( local_in_sizes ) );
    local_back.init( std::get<0>( local_in_sizes ), std::get<1>( local_in_sizes ), std::get<2>( local_in_sizes ) );
    local_out.init( std::get<0>( local_out_sizes ), std::get<1>( local_out_sizes ), std::get<2>( local_out_sizes ) );
    ref_in.init( options.nx, options.ny, options.nz );
    ref_back.init( options.nx, options.ny, options.nz );
    ref_out.init( options.nx, options.ny, options.nz / 2 + 1 );

    const T lx = T( 2 ) * scfd::utils::scalar_traits<T>::pi();
    const T hx = lx / static_cast<T>( options.nx );
    const T hy = lx / static_cast<T>( options.ny );
    const T hz = lx / static_cast<T>( options.nz );

    for_each_t for_each;
    reduce_t   reduce;
    for_each.block_size = 128;

    for_each( fill_input_functor<ref_real_t>{ ref_in, hx, hy, hz, T( 0 ), T( 0 ) }, make_range( ref_in ) );
    for_each.wait();

    const auto &input_part = distributed_fft.input_partition();
    for_each(
        fill_input_functor<local_real_t>{
            local_in, hx, hy, hz, hx * static_cast<T>( input_part.start_x[myid_i] ),
            hy * static_cast<T>( input_part.start_y[myid_j] ) },
        make_range( local_in )
    );
    for_each.wait();

    reference_fft.forward( ref_in, ref_out );
    distributed_fft.forward( local_in, local_out );

    err_hat_array_t forward_diff_sq;
    err_hat_array_t forward_ref_sq;
    forward_diff_sq.init(
        std::get<0>( local_out_sizes ), std::get<1>( local_out_sizes ), std::get<2>( local_out_sizes )
    );
    forward_ref_sq.init(
        std::get<0>( local_out_sizes ), std::get<1>( local_out_sizes ), std::get<2>( local_out_sizes )
    );

    const auto &output_part = distributed_fft.output_partition();
    for_each(
        compare_output_functor<local_hat_t, ref_hat_t, err_hat_array_t>{
            local_out, ref_out, forward_diff_sq, forward_ref_sq, static_cast<int>( output_part.start_y[myid_i] ),
            static_cast<int>( output_part.start_z[myid_j] ),
            fftm_t::strategy_family_3d == fftm::transform_strategy_3d::pencil_pencil &&
                fftm_t::strategy_3d_optimized_layout },
        make_range( local_out )
    );
    for_each.wait();

    const T local_forward_diff_sq  = reduce( forward_diff_sq.size(), forward_diff_sq.raw_ptr(), T( 0 ) );
    const T local_forward_ref_sq   = reduce( forward_ref_sq.size(), forward_ref_sq.raw_ptr(), T( 0 ) );
    const T global_forward_diff_sq = comm_info.all_reduce_sum( local_forward_diff_sq );
    const T global_forward_ref_sq  = comm_info.all_reduce_sum( local_forward_ref_sq );
    const T forward_relative_error = std::sqrt( global_forward_diff_sq / global_forward_ref_sq );

    reference_fft.backward( ref_out, ref_back );
    distributed_fft.backward( local_out, local_back );

    err_real_array_t backward_diff_sq;
    err_real_array_t backward_ref_sq;
    backward_diff_sq.init(
        std::get<0>( local_in_sizes ), std::get<1>( local_in_sizes ), std::get<2>( local_in_sizes )
    );
    backward_ref_sq.init( std::get<0>( local_in_sizes ), std::get<1>( local_in_sizes ), std::get<2>( local_in_sizes ) );

    for_each(
        compare_real_output_functor<local_real_t, ref_real_t, err_real_array_t>{
            local_back, ref_back, backward_diff_sq, backward_ref_sq, static_cast<int>( input_part.start_x[myid_i] ),
            static_cast<int>( input_part.start_y[myid_j] ), static_cast<int>( input_part.start_z[myid_k] ) },
        make_range( local_back )
    );
    for_each.wait();

    const T local_backward_diff_sq  = reduce( backward_diff_sq.size(), backward_diff_sq.raw_ptr(), T( 0 ) );
    const T local_backward_ref_sq   = reduce( backward_ref_sq.size(), backward_ref_sq.raw_ptr(), T( 0 ) );
    const T global_backward_diff_sq = comm_info.all_reduce_sum( local_backward_diff_sq );
    const T global_backward_ref_sq  = comm_info.all_reduce_sum( local_backward_ref_sq );
    const T backward_relative_error = std::sqrt( global_backward_diff_sq / global_backward_ref_sq );

    if ( comm_info.myid == 0 )
    {
        log.info_f(
            "strategy=%s, mode=%s, grid=(%zu,%zu), sizes=(%zu,%zu,%zu), forward_rel_l2=%.8e, backward_rel_l2=%.8e",
            fftm_t::strategy_name(), fftm::mpi_transpose_3d_mode_name( fftm_t::transpose_mode_3d ), options.p1,
            options.p2, options.nx, options.ny, options.nz, forward_relative_error, backward_relative_error
        );
    }

    return ( forward_relative_error <= options.threshold && backward_relative_error <= options.threshold ) ? 0 : 1;
}

template <fftm::mpi_transpose_3d_mode Mode>
int run_for_strategy_kind(
    strategy_kind strategy, scfd::utils::log_mpi &log, const test_options &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    switch ( strategy )
    {
    case strategy_kind::slab_pencil:
        return run_compare<fftm::strategy_3d_slab_pencil<Mode>>( log, options, comm_info );
    case strategy_kind::pencil_slab:
        return run_compare<fftm::strategy_3d_pencil_slab<Mode>>( log, options, comm_info );
    case strategy_kind::pencil_pencil:
        return run_compare<fftm::strategy_3d_pencil_pencil<Mode>>( log, options, comm_info );
    }
    return 1;
}

int dispatch_mode(
    strategy_kind strategy, scfd::utils::log_mpi &log, const test_options &options,
    const scfd::communication::mpi_comm_info &comm_info
)
{
    switch ( options.mode )
    {
    case fftm::mpi_transpose_3d_mode::p2p_waitall:
        return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::p2p_waitall>( strategy, log, options, comm_info );
    case fftm::mpi_transpose_3d_mode::p2p_waitany:
        return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::p2p_waitany>( strategy, log, options, comm_info );
    case fftm::mpi_transpose_3d_mode::alltoallv:
        return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::alltoallv>( strategy, log, options, comm_info );
    case fftm::mpi_transpose_3d_mode::alltoallw:
        return run_for_strategy_kind<fftm::mpi_transpose_3d_mode::alltoallw>( strategy, log, options, comm_info );
    }
    return 1;
}

} // namespace

int main( int argc, char *argv[] )
{
    scfd::communication::mpi_wrap mpi( argc, argv );
    auto                          comm_info = mpi.comm_world();
    scfd::utils::log_mpi          log;

    try
    {
        fftm::test::detail::init_fft_test_mpi( log, comm_info );

        const test_options options = parse_options( argc, argv, comm_info.num_procs );

        if ( options.p1 * options.p2 != static_cast<std::size_t>( comm_info.num_procs ) )
            throw std::logic_error( "P1*P2 must equal the number of MPI processes" );

        int failed = 0;
        if ( options.run_all )
        {
            failed = std::max( failed, dispatch_mode( strategy_kind::slab_pencil, log, options, comm_info ) );
            failed = std::max( failed, dispatch_mode( strategy_kind::pencil_slab, log, options, comm_info ) );
            failed = std::max( failed, dispatch_mode( strategy_kind::pencil_pencil, log, options, comm_info ) );
        }
        else
        {
            failed = dispatch_mode( options.strategy, log, options, comm_info );
        }

        if ( comm_info.myid == 0 )
        {
            if ( failed == 0 )
                log.info( "PASSED" );
            else
                log.error( "FAILED" );
        }

        return failed;
    }
    catch ( const std::exception &e )
    {
        log.error( scfd::utils::nested_exception_to_multistring( e ) );
        return 1;
    }
}
