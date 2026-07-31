#include "node_fft_4d_backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cufftXt.h>

namespace fftm
{
namespace experiments
{
namespace node_hybrid
{
namespace
{

constexpr unsigned int transpose_tile = 32;
constexpr unsigned int transpose_rows = 8;

void check_cuda( cudaError_t result, const char *expression )
{
    if ( result == cudaSuccess )
        return;
    std::ostringstream message;
    message << expression << " failed: " << cudaGetErrorString( result );
    throw std::runtime_error( message.str() );
}

void check_cufft( cufftResult result, const char *expression )
{
    if ( result == CUFFT_SUCCESS )
        return;
    std::ostringstream message;
    message << expression << " failed with cuFFT status " << static_cast<int>( result );
    throw std::runtime_error( message.str() );
}

#define FFTM_NODE_4D_CUDA_CALL( expression ) check_cuda( ( expression ), #expression )
#define FFTM_NODE_4D_CUFFT_CALL( expression ) check_cufft( ( expression ), #expression )

std::size_t checked_add( std::size_t lhs, std::size_t rhs, const char *label )
{
    if ( rhs > std::numeric_limits<std::size_t>::max() - lhs )
        throw std::overflow_error( std::string( label ) + " size overflow" );
    return lhs + rhs;
}

std::size_t checked_multiply( std::size_t lhs, std::size_t rhs, const char *label )
{
    if ( lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs )
        throw std::overflow_error( std::string( label ) + " size overflow" );
    return lhs * rhs;
}

long long checked_long_long( std::size_t value, const char *label )
{
    if ( value > static_cast<std::size_t>( std::numeric_limits<long long>::max() ) )
        throw std::overflow_error( std::string( label ) + " exceeds the cuFFT 64-bit API" );
    return static_cast<long long>( value );
}

std::size_t partition_size( std::size_t total, std::size_t part, std::size_t parts )
{
    const std::size_t base = total / parts;
    const std::size_t remainder = total % parts;
    return base + ( part < remainder ? 1 : 0 );
}

std::size_t partition_start( std::size_t total, std::size_t part, std::size_t parts )
{
    const std::size_t base = total / parts;
    const std::size_t remainder = total % parts;
    return part * base + std::min( part, remainder );
}

std::size_t complex_w_size( const shape_4d &shape )
{
    return shape.w / 2 + 1;
}

std::size_t padded_real_w_size( const shape_4d &shape )
{
    return checked_multiply( complex_w_size( shape ), 2, "4D in-place real padding" );
}

std::size_t logical_point_count( const shape_4d &shape )
{
    return checked_multiply(
        checked_multiply( shape.x, shape.y, "4D logical shape" ),
        checked_multiply( shape.z, shape.w, "4D logical shape" ),
        "4D logical shape"
    );
}

std::size_t padded_complex_count( const shape_4d &shape )
{
    return checked_multiply(
        checked_multiply( shape.x, shape.y, "4D complex shape" ),
        checked_multiply( shape.z, complex_w_size( shape ), "4D complex shape" ),
        "4D complex shape"
    );
}

__host__ __device__ double deterministic_input_value( std::size_t index, std::uint64_t seed )
{
    const std::uint64_t mask = 1023;
    const std::uint64_t value =
        ( ( static_cast<std::uint64_t>( index ) & mask ) * 17 +
          ( seed & mask ) * 131 + 29 ) &
        mask;
    return ( static_cast<double>( value ) - 511.5 ) / 1024.0;
}

__global__ void initialize_real_4d_segment(
    double *data, std::size_t local_x, std::size_t x_start, std::size_t y_size,
    std::size_t z_size, std::size_t w_size, std::size_t padded_w,
    std::uint64_t seed
)
{
    const std::size_t storage_count = local_x * y_size * z_size * padded_w;
    for ( std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
          linear < storage_count;
          linear += static_cast<std::size_t>( blockDim.x ) * gridDim.x )
    {
        std::size_t remainder = linear;
        const std::size_t w = remainder % padded_w;
        remainder /= padded_w;
        const std::size_t z = remainder % z_size;
        remainder /= z_size;
        const std::size_t y = remainder % y_size;
        const std::size_t local_x_index = remainder / y_size;
        if ( w < w_size )
        {
            const std::size_t x = x_start + local_x_index;
            const std::size_t logical =
                w + w_size * ( z + z_size * ( y + y_size * x ) );
            data[linear] = deterministic_input_value( logical, seed );
        }
        else
        {
            data[linear] = 0.0;
        }
    }
}

__global__ void scale_real_4d_segment( double *data, std::size_t count, double factor )
{
    for ( std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
          index < count;
          index += static_cast<std::size_t>( blockDim.x ) * gridDim.x )
    {
        data[index] *= factor;
    }
}

__device__ void atomic_max_nonnegative_double( double *address, double value )
{
    auto *bits = reinterpret_cast<unsigned long long *>( address );
    unsigned long long current = *bits;
    while ( __longlong_as_double( static_cast<long long>( current ) ) < value )
    {
        const unsigned long long assumed = current;
        current = atomicCAS(
            bits, assumed,
            static_cast<unsigned long long>( __double_as_longlong( value ) )
        );
        if ( current == assumed )
            break;
    }
}

__global__ void validate_real_4d_segment(
    const double *data, std::size_t local_x, std::size_t x_start,
    std::size_t y_size, std::size_t z_size, std::size_t w_size,
    std::size_t padded_w, std::uint64_t seed, double inverse_scale,
    double *error_norm2, double *reference_norm2, double *maximum_error
)
{
    const std::size_t logical_count = local_x * y_size * z_size * w_size;
    double local_error = 0.0;
    double local_reference = 0.0;
    double local_maximum = 0.0;
    for ( std::size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
          linear < logical_count;
          linear += static_cast<std::size_t>( blockDim.x ) * gridDim.x )
    {
        std::size_t remainder = linear;
        const std::size_t w = remainder % w_size;
        remainder /= w_size;
        const std::size_t z = remainder % z_size;
        remainder /= z_size;
        const std::size_t y = remainder % y_size;
        const std::size_t local_x_index = remainder / y_size;
        const std::size_t x = x_start + local_x_index;
        const std::size_t logical =
            w + w_size * ( z + z_size * ( y + y_size * x ) );
        const std::size_t storage =
            w + padded_w * ( z + z_size * ( y + y_size * local_x_index ) );
        const double expected = deterministic_input_value( logical, seed );
        const double actual = data[storage] * inverse_scale;
        const double error = actual - expected;
        local_error += error * error;
        local_reference += expected * expected;
        local_maximum = fmax( local_maximum, fabs( error ) );
    }

    __shared__ double errors[256];
    __shared__ double references[256];
    __shared__ double maxima[256];
    const unsigned int lane = threadIdx.x;
    errors[lane] = local_error;
    references[lane] = local_reference;
    maxima[lane] = local_maximum;
    __syncthreads();
    for ( unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2 )
    {
        if ( lane < stride )
        {
            errors[lane] += errors[lane + stride];
            references[lane] += references[lane + stride];
            maxima[lane] = fmax( maxima[lane], maxima[lane + stride] );
        }
        __syncthreads();
    }
    if ( lane == 0 )
    {
        atomicAdd( error_norm2, errors[0] );
        atomicAdd( reference_norm2, references[0] );
        atomic_max_nonnegative_double( maximum_error, maxima[0] );
    }
}

__global__ void transpose_x_slab_to_native_xzwy(
    const cufftDoubleComplex *source, cufftDoubleComplex *destination,
    std::size_t local_x, std::size_t x_start, std::size_t global_x,
    std::size_t y_size, std::size_t global_z, std::size_t local_z,
    std::size_t z_start, std::size_t complex_w
)
{
    __shared__ cufftDoubleComplex tile[transpose_tile][transpose_tile + 1];
    const std::size_t yz_count = y_size * local_z;
    for ( std::size_t yz = blockIdx.z; yz < yz_count; yz += gridDim.z )
    {
        const std::size_t z_local = yz % local_z;
        const std::size_t y = yz / local_z;
        const std::size_t z = z_start + z_local;
        const std::size_t x_base = blockIdx.y * transpose_tile;
        const std::size_t w_base = blockIdx.x * transpose_tile;

        for ( unsigned int row = threadIdx.y; row < transpose_tile; row += transpose_rows )
        {
            const std::size_t x_local = x_base + row;
            const std::size_t w = w_base + threadIdx.x;
            if ( x_local < local_x && w < complex_w )
            {
                const std::size_t source_offset =
                    w + complex_w *
                            ( z + global_z * ( y + y_size * x_local ) );
                tile[row][threadIdx.x] = source[source_offset];
            }
        }
        __syncthreads();

        for ( unsigned int row = threadIdx.y; row < transpose_tile; row += transpose_rows )
        {
            const std::size_t x_local = x_base + threadIdx.x;
            const std::size_t w = w_base + row;
            if ( x_local < local_x && w < complex_w )
            {
                const std::size_t x = x_start + x_local;
                const std::size_t destination_offset =
                    x + global_x *
                            ( w + complex_w * ( z_local + local_z * y ) );
                destination[destination_offset] = tile[threadIdx.x][row];
            }
        }
        __syncthreads();
    }
}

__global__ void transpose_native_xzwy_to_x_slab(
    const cufftDoubleComplex *source, cufftDoubleComplex *destination,
    std::size_t local_z, std::size_t z_start, std::size_t global_z,
    std::size_t local_x, std::size_t x_start, std::size_t global_x,
    std::size_t y_size, std::size_t complex_w
)
{
    __shared__ cufftDoubleComplex tile[transpose_tile][transpose_tile + 1];
    const std::size_t yz_count = y_size * local_z;
    for ( std::size_t yz = blockIdx.z; yz < yz_count; yz += gridDim.z )
    {
        const std::size_t z_local = yz % local_z;
        const std::size_t y = yz / local_z;
        const std::size_t z = z_start + z_local;
        const std::size_t x_base = blockIdx.x * transpose_tile;
        const std::size_t w_base = blockIdx.y * transpose_tile;

        for ( unsigned int row = threadIdx.y; row < transpose_tile; row += transpose_rows )
        {
            const std::size_t w = w_base + row;
            const std::size_t x_local = x_base + threadIdx.x;
            if ( w < complex_w && x_local < local_x )
            {
                const std::size_t x = x_start + x_local;
                const std::size_t source_offset =
                    x + global_x *
                            ( w + complex_w * ( z_local + local_z * y ) );
                tile[row][threadIdx.x] = source[source_offset];
            }
        }
        __syncthreads();

        for ( unsigned int row = threadIdx.y; row < transpose_tile; row += transpose_rows )
        {
            const std::size_t x_local = x_base + row;
            const std::size_t w = w_base + threadIdx.x;
            if ( x_local < local_x && w < complex_w )
            {
                const std::size_t destination_offset =
                    w + complex_w *
                            ( z + global_z * ( y + y_size * x_local ) );
                destination[destination_offset] = tile[threadIdx.x][row];
            }
        }
        __syncthreads();
    }
}

class cuda_cufft_xt_3d_1d_backend final : public node_fft_4d_backend
{
public:
    cuda_cufft_xt_3d_1d_backend(
        const shape_4d &shape, std::vector<int> device_ordinals
    )
        : shape_( shape ), devices_( std::move( device_ordinals ) )
    {
        validate_configuration_();
        const std::size_t count = devices_.size();
        forward_3d_work_sizes_.assign( count, 0 );
        inverse_3d_work_sizes_.assign( count, 0 );
        x_work_sizes_.assign( count, 0 );
        shared_work_sizes_.assign( count, 0 );
        shared_work_.assign( count, nullptr );
        native_xzwy_.assign( count, nullptr );
        x_plans_.assign( count, 0 );
        streams_.assign( count, nullptr );

        try
        {
            create_streams_and_peer_access_();
            create_yzw_plans_();
            create_x_plans_();
            allocate_and_bind_shared_work_();
            FFTM_NODE_4D_CUFFT_CALL(
                cufftXtMalloc( forward_yzw_plan_, &x_slab_, CUFFT_XT_FORMAT_INPLACE )
            );
            allocate_native_xzwy_();
            verify_storage_();
        }
        catch ( ... )
        {
            release_();
            throw;
        }
    }

    ~cuda_cufft_xt_3d_1d_backend() override
    {
        release_();
    }

    const char *name() const override
    {
        return "cuda-cufft-xt-yzw-plus-x";
    }

    const char *spectral_layout_name() const override
    {
        return "native-xzwy";
    }

    shape_4d shape() const override
    {
        return shape_;
    }

    int device_count() const override
    {
        return static_cast<int>( devices_.size() );
    }

    std::size_t logical_real_bytes() const override
    {
        return checked_multiply(
            logical_point_count( shape_ ), sizeof( double ), "4D logical bytes"
        );
    }

    std::size_t allocated_data_bytes() const override
    {
        return checked_add(
            descriptor_bytes_( x_slab_ ),
            checked_multiply(
                padded_complex_count( shape_ ), sizeof( cufftDoubleComplex ),
                "native xzwy bytes"
            ),
            "4D allocated data"
        );
    }

    std::size_t max_allocated_data_bytes_per_device() const override
    {
        ensure_descriptor_();
        std::size_t maximum = 0;
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            const std::size_t native_bytes = checked_multiply(
                native_xzwy_count_( index ), sizeof( cufftDoubleComplex ),
                "native xzwy per-device bytes"
            );
            maximum = std::max(
                maximum,
                checked_add(
                    x_slab_->descriptor->size[index], native_bytes,
                    "4D per-device data"
                )
            );
        }
        return maximum;
    }

    std::size_t shared_work_bytes() const override
    {
        return sum_sizes_( shared_work_sizes_ );
    }

    std::size_t max_shared_work_bytes_per_device() const override
    {
        return shared_work_sizes_.empty()
                   ? 0
                   : *std::max_element(
                         shared_work_sizes_.begin(), shared_work_sizes_.end()
                     );
    }

    std::size_t forward_3d_work_bytes() const override
    {
        return sum_sizes_( forward_3d_work_sizes_ );
    }

    std::size_t inverse_3d_work_bytes() const override
    {
        return sum_sizes_( inverse_3d_work_sizes_ );
    }

    std::size_t x_fft_work_bytes() const override
    {
        return sum_sizes_( x_work_sizes_ );
    }

    std::size_t transpose_bytes() const override
    {
        return checked_multiply(
            padded_complex_count( shape_ ), sizeof( cufftDoubleComplex ),
            "4D transpose bytes"
        );
    }

    void initialize_input( std::uint64_t seed ) override
    {
        ensure_descriptor_();
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[index] ) );
            const std::size_t local_x = x_count_( index );
            const std::size_t storage_count = checked_multiply(
                checked_multiply( local_x, shape_.y, "4D input segment" ),
                checked_multiply(
                    shape_.z, padded_real_w_size( shape_ ), "4D input segment"
                ),
                "4D input segment"
            );
            const unsigned int blocks = launch_blocks_( storage_count );
            initialize_real_4d_segment<<<blocks, 256, 0, streams_[index]>>>(
                static_cast<double *>( x_slab_->descriptor->data[index] ),
                local_x, x_start_( index ), shape_.y, shape_.z, shape_.w,
                padded_real_w_size( shape_ ), seed
            );
            FFTM_NODE_4D_CUDA_CALL( cudaGetLastError() );
        }
        synchronize();
    }

    void forward_yzw() override
    {
        ensure_descriptor_();
        FFTM_NODE_4D_CUFFT_CALL(
            cufftXtExecDescriptorD2Z( forward_yzw_plan_, x_slab_, x_slab_ )
        );
    }

    void transpose_x_to_z() override
    {
        ensure_descriptor_();
        const dim3 block( transpose_tile, transpose_rows, 1 );
        for ( std::size_t source = 0; source < devices_.size(); ++source )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[source] ) );
            const auto *source_pointer = static_cast<const cufftDoubleComplex *>(
                x_slab_->descriptor->data[source]
            );
            const std::size_t local_x = x_count_( source );
            for ( std::size_t destination = 0; destination < devices_.size(); ++destination )
            {
                const std::size_t local_z = z_count_( destination );
                const std::size_t yz_count =
                    checked_multiply( shape_.y, local_z, "forward transpose yz" );
                const dim3 grid(
                    grid_tiles_( complex_w_size( shape_ ) ),
                    grid_tiles_( local_x ),
                    static_cast<unsigned int>( std::min<std::size_t>( yz_count, 65535 ) )
                );
                transpose_x_slab_to_native_xzwy<<<grid, block, 0, streams_[source]>>>(
                    source_pointer, native_xzwy_[destination], local_x,
                    x_start_( source ), shape_.x, shape_.y, shape_.z, local_z,
                    z_start_( destination ), complex_w_size( shape_ )
                );
                FFTM_NODE_4D_CUDA_CALL( cudaGetLastError() );
            }
        }
    }

    void forward_x() override
    {
        execute_x_( CUFFT_FORWARD );
    }

    void inverse_x() override
    {
        execute_x_( CUFFT_INVERSE );
    }

    void transpose_z_to_x() override
    {
        ensure_descriptor_();
        const dim3 block( transpose_tile, transpose_rows, 1 );
        for ( std::size_t source = 0; source < devices_.size(); ++source )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[source] ) );
            const std::size_t local_z = z_count_( source );
            const std::size_t yz_count =
                checked_multiply( shape_.y, local_z, "inverse transpose yz" );
            for ( std::size_t destination = 0; destination < devices_.size(); ++destination )
            {
                const std::size_t local_x = x_count_( destination );
                const dim3 grid(
                    grid_tiles_( local_x ),
                    grid_tiles_( complex_w_size( shape_ ) ),
                    static_cast<unsigned int>( std::min<std::size_t>( yz_count, 65535 ) )
                );
                transpose_native_xzwy_to_x_slab<<<grid, block, 0, streams_[source]>>>(
                    native_xzwy_[source],
                    static_cast<cufftDoubleComplex *>(
                        x_slab_->descriptor->data[destination]
                    ),
                    local_z, z_start_( source ), shape_.z, local_x,
                    x_start_( destination ), shape_.x, shape_.y,
                    complex_w_size( shape_ )
                );
                FFTM_NODE_4D_CUDA_CALL( cudaGetLastError() );
            }
        }
    }

    void inverse_yzw() override
    {
        ensure_descriptor_();
        FFTM_NODE_4D_CUFFT_CALL(
            cufftXtExecDescriptorZ2D( inverse_yzw_plan_, x_slab_, x_slab_ )
        );
    }

    void normalize_inverse_output() override
    {
        const double factor =
            1.0 / static_cast<double>( logical_point_count( shape_ ) );
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[index] ) );
            const std::size_t count = expected_x_slab_storage_count_( index );
            scale_real_4d_segment<<<launch_blocks_( count ), 256, 0, streams_[index]>>>(
                static_cast<double *>( x_slab_->descriptor->data[index] ),
                count, factor
            );
            FFTM_NODE_4D_CUDA_CALL( cudaGetLastError() );
        }
    }

    void synchronize() override
    {
        for ( int device : devices_ )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( device ) );
            FFTM_NODE_4D_CUDA_CALL( cudaDeviceSynchronize() );
        }
    }

    validation_result validate_input( std::uint64_t seed ) override
    {
        synchronize();
        double error_norm2 = 0.0;
        double reference_norm2 = 0.0;
        double maximum_error = 0.0;
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[index] ) );
            double *device_result = nullptr;
            FFTM_NODE_4D_CUDA_CALL(
                cudaMalloc( reinterpret_cast<void **>( &device_result ), 3 * sizeof( double ) )
            );
            try
            {
                FFTM_NODE_4D_CUDA_CALL(
                    cudaMemsetAsync(
                        device_result, 0, 3 * sizeof( double ), streams_[index]
                    )
                );
                const std::size_t local_count = checked_multiply(
                    checked_multiply( x_count_( index ), shape_.y, "validation" ),
                    checked_multiply( shape_.z, shape_.w, "validation" ),
                    "validation"
                );
                validate_real_4d_segment<<<
                    launch_blocks_( local_count ), 256, 0, streams_[index]>>>(
                    static_cast<const double *>( x_slab_->descriptor->data[index] ),
                    x_count_( index ), x_start_( index ), shape_.y, shape_.z,
                    shape_.w, padded_real_w_size( shape_ ), seed,
                    1.0 / static_cast<double>( logical_point_count( shape_ ) ),
                    device_result, device_result + 1, device_result + 2
                );
                FFTM_NODE_4D_CUDA_CALL( cudaGetLastError() );
                double host_result[3] = { 0.0, 0.0, 0.0 };
                FFTM_NODE_4D_CUDA_CALL(
                    cudaMemcpyAsync(
                        host_result, device_result, sizeof( host_result ),
                        cudaMemcpyDeviceToHost, streams_[index]
                    )
                );
                FFTM_NODE_4D_CUDA_CALL( cudaStreamSynchronize( streams_[index] ) );
                error_norm2 += host_result[0];
                reference_norm2 += host_result[1];
                maximum_error = std::max( maximum_error, host_result[2] );
            }
            catch ( ... )
            {
                cudaFree( device_result );
                throw;
            }
            FFTM_NODE_4D_CUDA_CALL( cudaFree( device_result ) );
        }

        validation_result result;
        result.relative_l2 =
            reference_norm2 == 0.0
                ? std::sqrt( error_norm2 )
                : std::sqrt( error_norm2 / reference_norm2 );
        result.max_abs = maximum_error;
        return result;
    }

private:
    void validate_configuration_() const
    {
        if ( shape_.x == 0 || shape_.y == 0 || shape_.z == 0 || shape_.w == 0 )
            throw std::invalid_argument( "4D transform dimensions must be positive" );
        if ( shape_.w % 2 != 0 )
            throw std::invalid_argument( "4D in-place D2Z requires an even W dimension" );
        if ( devices_.size() < 2 )
            throw std::invalid_argument( "4D node-hybrid test requires at least two GPUs" );
        if ( devices_.size() > shape_.x || devices_.size() > shape_.z )
            throw std::invalid_argument( "4D X and Z dimensions must cover every GPU" );

        int available = 0;
        FFTM_NODE_4D_CUDA_CALL( cudaGetDeviceCount( &available ) );
        for ( int device : devices_ )
        {
            if ( device < 0 || device >= available )
                throw std::invalid_argument( "requested 4D CUDA device is not visible" );
        }
    }

    void create_streams_and_peer_access_()
    {
        for ( std::size_t source = 0; source < devices_.size(); ++source )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[source] ) );
            FFTM_NODE_4D_CUDA_CALL(
                cudaStreamCreateWithFlags( &streams_[source], cudaStreamNonBlocking )
            );
            for ( std::size_t destination = 0; destination < devices_.size(); ++destination )
            {
                if ( source == destination )
                    continue;
                int accessible = 0;
                FFTM_NODE_4D_CUDA_CALL(
                    cudaDeviceCanAccessPeer(
                        &accessible, devices_[source], devices_[destination]
                    )
                );
                if ( accessible == 0 )
                    throw std::runtime_error(
                        "selected GPUs do not provide complete peer access"
                    );
                const cudaError_t enabled =
                    cudaDeviceEnablePeerAccess( devices_[destination], 0 );
                if ( enabled == cudaErrorPeerAccessAlreadyEnabled )
                    cudaGetLastError();
                else
                    FFTM_NODE_4D_CUDA_CALL( enabled );
            }
        }
    }

    void create_yzw_plans_()
    {
        long long dimensions[3] = {
            checked_long_long( shape_.y, "Y dimension" ),
            checked_long_long( shape_.z, "Z dimension" ),
            checked_long_long( shape_.w, "W dimension" )
        };
        long long real_embed[3] = {
            dimensions[0], dimensions[1],
            checked_long_long( padded_real_w_size( shape_ ), "padded W" )
        };
        long long complex_embed[3] = {
            dimensions[0], dimensions[1],
            checked_long_long( complex_w_size( shape_ ), "complex W" )
        };
        const long long real_distance = checked_long_long(
            checked_multiply(
                checked_multiply( shape_.y, shape_.z, "YZW real distance" ),
                padded_real_w_size( shape_ ), "YZW real distance"
            ),
            "YZW real distance"
        );
        const long long complex_distance = checked_long_long(
            checked_multiply(
                checked_multiply( shape_.y, shape_.z, "YZW complex distance" ),
                complex_w_size( shape_ ), "YZW complex distance"
            ),
            "YZW complex distance"
        );
        const long long batch = checked_long_long( shape_.x, "YZW batch" );

        FFTM_NODE_4D_CUFFT_CALL( cufftCreate( &forward_yzw_plan_ ) );
        FFTM_NODE_4D_CUFFT_CALL( cufftSetAutoAllocation( forward_yzw_plan_, 0 ) );
        FFTM_NODE_4D_CUFFT_CALL(
            cufftXtSetGPUs(
                forward_yzw_plan_, static_cast<int>( devices_.size() ),
                devices_.data()
            )
        );
        FFTM_NODE_4D_CUFFT_CALL( cufftMakePlanMany64(
            forward_yzw_plan_, 3, dimensions, real_embed, 1, real_distance,
            complex_embed, 1, complex_distance, CUFFT_D2Z, batch,
            forward_3d_work_sizes_.data()
        ) );

        FFTM_NODE_4D_CUFFT_CALL( cufftCreate( &inverse_yzw_plan_ ) );
        FFTM_NODE_4D_CUFFT_CALL( cufftSetAutoAllocation( inverse_yzw_plan_, 0 ) );
        FFTM_NODE_4D_CUFFT_CALL(
            cufftXtSetGPUs(
                inverse_yzw_plan_, static_cast<int>( devices_.size() ),
                devices_.data()
            )
        );
        FFTM_NODE_4D_CUFFT_CALL( cufftMakePlanMany64(
            inverse_yzw_plan_, 3, dimensions, complex_embed, 1,
            complex_distance, real_embed, 1, real_distance, CUFFT_Z2D, batch,
            inverse_3d_work_sizes_.data()
        ) );
    }

    void create_x_plans_()
    {
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[index] ) );
            FFTM_NODE_4D_CUFFT_CALL( cufftCreate( &x_plans_[index] ) );
            FFTM_NODE_4D_CUFFT_CALL(
                cufftSetAutoAllocation( x_plans_[index], 0 )
            );
            long long dimension[1] = {
                checked_long_long( shape_.x, "X dimension" )
            };
            long long embed[1] = { dimension[0] };
            const std::size_t batches = checked_multiply(
                checked_multiply(
                    shape_.y, z_count_( index ), "local X FFT batch"
                ),
                complex_w_size( shape_ ), "local X FFT batch"
            );
            FFTM_NODE_4D_CUFFT_CALL( cufftMakePlanMany64(
                x_plans_[index], 1, dimension, embed, 1, dimension[0],
                embed, 1, dimension[0], CUFFT_Z2Z,
                checked_long_long( batches, "local X FFT batch" ),
                &x_work_sizes_[index]
            ) );
            FFTM_NODE_4D_CUFFT_CALL(
                cufftSetStream( x_plans_[index], streams_[index] )
            );
        }
    }

    void allocate_and_bind_shared_work_()
    {
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            shared_work_sizes_[index] = std::max(
                forward_3d_work_sizes_[index],
                std::max( inverse_3d_work_sizes_[index], x_work_sizes_[index] )
            );
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[index] ) );
            if ( shared_work_sizes_[index] != 0 )
            {
                FFTM_NODE_4D_CUDA_CALL(
                    cudaMalloc( &shared_work_[index], shared_work_sizes_[index] )
                );
            }
        }
        FFTM_NODE_4D_CUFFT_CALL(
            cufftXtSetWorkArea( forward_yzw_plan_, shared_work_.data() )
        );
        FFTM_NODE_4D_CUFFT_CALL(
            cufftXtSetWorkArea( inverse_yzw_plan_, shared_work_.data() )
        );
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[index] ) );
            FFTM_NODE_4D_CUFFT_CALL(
                cufftSetWorkArea( x_plans_[index], shared_work_[index] )
            );
        }
    }

    void allocate_native_xzwy_()
    {
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[index] ) );
            const std::size_t count = native_xzwy_count_( index );
            FFTM_NODE_4D_CUDA_CALL( cudaMalloc(
                reinterpret_cast<void **>( &native_xzwy_[index] ),
                checked_multiply(
                    count, sizeof( cufftDoubleComplex ), "native xzwy segment"
                )
            ) );
        }
    }

    void verify_storage_() const
    {
        ensure_descriptor_();
        if ( x_slab_->descriptor->nGPUs != static_cast<int>( devices_.size() ) )
            throw std::runtime_error( "4D cuFFT Xt descriptor GPU count mismatch" );
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            if ( x_slab_->descriptor->GPUs[index] != devices_[index] )
                throw std::runtime_error( "4D cuFFT Xt descriptor device order mismatch" );
            const std::size_t expected = checked_multiply(
                expected_x_slab_storage_count_( index ), sizeof( double ),
                "expected X-slab segment"
            );
            if ( x_slab_->descriptor->size[index] < expected )
                throw std::runtime_error( "4D cuFFT Xt X-slab segment is undersized" );
            if ( native_xzwy_[index] == nullptr )
                throw std::runtime_error( "4D native xzwy segment is missing" );
        }
    }

    void execute_x_( int direction )
    {
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            FFTM_NODE_4D_CUDA_CALL( cudaSetDevice( devices_[index] ) );
            FFTM_NODE_4D_CUFFT_CALL( cufftExecZ2Z(
                x_plans_[index], native_xzwy_[index], native_xzwy_[index],
                direction
            ) );
        }
    }

    std::size_t x_count_( std::size_t index ) const
    {
        return partition_size( shape_.x, index, devices_.size() );
    }

    std::size_t x_start_( std::size_t index ) const
    {
        return partition_start( shape_.x, index, devices_.size() );
    }

    std::size_t z_count_( std::size_t index ) const
    {
        return partition_size( shape_.z, index, devices_.size() );
    }

    std::size_t z_start_( std::size_t index ) const
    {
        return partition_start( shape_.z, index, devices_.size() );
    }

    std::size_t expected_x_slab_storage_count_( std::size_t index ) const
    {
        return checked_multiply(
            checked_multiply( x_count_( index ), shape_.y, "X-slab storage" ),
            checked_multiply(
                shape_.z, padded_real_w_size( shape_ ), "X-slab storage"
            ),
            "X-slab storage"
        );
    }

    std::size_t native_xzwy_count_( std::size_t index ) const
    {
        return checked_multiply(
            checked_multiply( shape_.x, z_count_( index ), "native xzwy" ),
            checked_multiply(
                complex_w_size( shape_ ), shape_.y, "native xzwy"
            ),
            "native xzwy"
        );
    }

    static unsigned int launch_blocks_( std::size_t count )
    {
        const std::size_t required = ( count + 255 ) / 256;
        return static_cast<unsigned int>(
            std::max<std::size_t>( 1, std::min<std::size_t>( required, 65535 ) )
        );
    }

    static unsigned int grid_tiles_( std::size_t count )
    {
        const std::size_t tiles = ( count + transpose_tile - 1 ) / transpose_tile;
        if ( tiles == 0 || tiles > 65535 )
            throw std::overflow_error( "4D transpose grid exceeds CUDA limits" );
        return static_cast<unsigned int>( tiles );
    }

    static std::size_t sum_sizes_( const std::vector<std::size_t> &sizes )
    {
        std::size_t total = 0;
        for ( std::size_t size : sizes )
            total = checked_add( total, size, "cuFFT work-size sum" );
        return total;
    }

    static std::size_t descriptor_bytes_( const cudaLibXtDesc *descriptor )
    {
        if ( descriptor == nullptr || descriptor->descriptor == nullptr )
            return 0;
        std::size_t total = 0;
        for ( int index = 0; index < descriptor->descriptor->nGPUs; ++index )
        {
            total = checked_add(
                total, descriptor->descriptor->size[index],
                "cuFFT descriptor-size sum"
            );
        }
        return total;
    }

    void ensure_descriptor_() const
    {
        if ( x_slab_ == nullptr || x_slab_->descriptor == nullptr )
            throw std::logic_error( "4D cuFFT Xt descriptor is not initialized" );
    }

    void release_() noexcept
    {
        for ( int device : devices_ )
        {
            cudaSetDevice( device );
            cudaDeviceSynchronize();
        }
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            cudaSetDevice( devices_[index] );
            if ( native_xzwy_[index] != nullptr )
            {
                cudaFree( native_xzwy_[index] );
                native_xzwy_[index] = nullptr;
            }
        }
        if ( x_slab_ != nullptr )
        {
            cufftXtFree( x_slab_ );
            x_slab_ = nullptr;
        }
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            cudaSetDevice( devices_[index] );
            if ( shared_work_[index] != nullptr )
            {
                cudaFree( shared_work_[index] );
                shared_work_[index] = nullptr;
            }
            if ( x_plans_[index] != 0 )
            {
                cufftDestroy( x_plans_[index] );
                x_plans_[index] = 0;
            }
        }
        if ( inverse_yzw_plan_ != 0 )
        {
            cufftDestroy( inverse_yzw_plan_ );
            inverse_yzw_plan_ = 0;
        }
        if ( forward_yzw_plan_ != 0 )
        {
            cufftDestroy( forward_yzw_plan_ );
            forward_yzw_plan_ = 0;
        }
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            cudaSetDevice( devices_[index] );
            if ( streams_[index] != nullptr )
            {
                cudaStreamDestroy( streams_[index] );
                streams_[index] = nullptr;
            }
        }
    }

    shape_4d shape_;
    std::vector<int> devices_;
    std::vector<std::size_t> forward_3d_work_sizes_;
    std::vector<std::size_t> inverse_3d_work_sizes_;
    std::vector<std::size_t> x_work_sizes_;
    std::vector<std::size_t> shared_work_sizes_;
    std::vector<void *> shared_work_;
    std::vector<cufftDoubleComplex *> native_xzwy_;
    std::vector<cufftHandle> x_plans_;
    std::vector<cudaStream_t> streams_;
    cufftHandle forward_yzw_plan_ = 0;
    cufftHandle inverse_yzw_plan_ = 0;
    cudaLibXtDesc *x_slab_ = nullptr;
};

}

std::unique_ptr<node_fft_4d_backend>
make_cuda_cufft_xt_3d_1d_backend(
    const shape_4d &shape, const std::vector<int> &device_ordinals
)
{
    return std::unique_ptr<node_fft_4d_backend>(
        new cuda_cufft_xt_3d_1d_backend( shape, device_ordinals )
    );
}

}
}
}
