#include "node_fft_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
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

#define FFTM_NODE_HYBRID_CUDA_CALL( expression ) check_cuda( ( expression ), #expression )
#define FFTM_NODE_HYBRID_CUFFT_CALL( expression ) check_cufft( ( expression ), #expression )

__global__ void scale_real_values( double *values, std::size_t count, double factor )
{
    for ( std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
          index < count;
          index += static_cast<std::size_t>( blockDim.x ) * gridDim.x )
    {
        values[index] *= factor;
    }
}

std::size_t checked_multiply( std::size_t lhs, std::size_t rhs, const char *label )
{
    if ( lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs )
        throw std::overflow_error( std::string( label ) + " size overflow" );
    return lhs * rhs;
}

std::size_t point_count( const shape_3d &shape )
{
    return checked_multiply( checked_multiply( shape.x, shape.y, "3D shape" ), shape.z, "3D shape" );
}

std::size_t padded_real_z( const shape_3d &shape )
{
    return checked_multiply( shape.z / 2 + 1, 2, "in-place real padding" );
}

std::size_t padded_real_point_count( const shape_3d &shape )
{
    return checked_multiply(
        checked_multiply( shape.x, shape.y, "padded 3D shape" ),
        padded_real_z( shape ),
        "padded 3D shape"
    );
}

double input_value( std::size_t index, std::uint64_t seed )
{
    const std::uint64_t mask = 1023;
    const std::uint64_t value =
        ( ( index & mask ) * 17 + ( seed & mask ) * 131 + 29 ) & mask;
    return ( static_cast<double>( value ) - 511.5 ) / 1024.0;
}

struct free_deleter
{
    void operator()( double *pointer ) const
    {
        std::free( pointer );
    }
};

using host_buffer = std::unique_ptr<double, free_deleter>;

host_buffer allocate_host_buffer( std::size_t points )
{
    const std::size_t bytes = checked_multiply( points, sizeof( double ), "host buffer" );
    double           *pointer = static_cast<double *>( std::malloc( bytes ) );
    if ( pointer == nullptr )
        throw std::bad_alloc();
    return host_buffer( pointer );
}

class cuda_cufft_xt_backend final : public node_fft_backend
{
public:
    cuda_cufft_xt_backend( const shape_3d &shape, std::vector<int> device_ordinals )
        : shape_( shape ), devices_( std::move( device_ordinals ) )
    {
        if ( shape_.x == 0 || shape_.y == 0 || shape_.z == 0 )
            throw std::invalid_argument( "cuFFT Xt shape dimensions must be positive" );
        if ( shape_.x > static_cast<std::size_t>( std::numeric_limits<int>::max() ) ||
             shape_.y > static_cast<std::size_t>( std::numeric_limits<int>::max() ) ||
             shape_.z > static_cast<std::size_t>( std::numeric_limits<int>::max() ) )
        {
            throw std::invalid_argument( "cuFFT Xt shape exceeds the cufftMakePlan3d integer API" );
        }
        if ( devices_.size() < 2 )
            throw std::invalid_argument( "cuFFT Xt requires at least two visible GPUs" );

        int available = 0;
        FFTM_NODE_HYBRID_CUDA_CALL( cudaGetDeviceCount( &available ) );
        for ( int device : devices_ )
        {
            if ( device < 0 || device >= available )
                throw std::invalid_argument( "requested cuFFT Xt device is not visible" );
        }

        forward_work_sizes_.assign( devices_.size(), 0 );
        inverse_work_sizes_.assign( devices_.size(), 0 );

        try
        {
            FFTM_NODE_HYBRID_CUFFT_CALL( cufftCreate( &forward_plan_ ) );
            FFTM_NODE_HYBRID_CUFFT_CALL(
                cufftXtSetGPUs( forward_plan_, static_cast<int>( devices_.size() ), devices_.data() )
            );
            FFTM_NODE_HYBRID_CUFFT_CALL( cufftMakePlan3d(
                forward_plan_, static_cast<int>( shape_.x ), static_cast<int>( shape_.y ),
                static_cast<int>( shape_.z ), CUFFT_D2Z, forward_work_sizes_.data()
            ) );

            FFTM_NODE_HYBRID_CUFFT_CALL( cufftCreate( &inverse_plan_ ) );
            FFTM_NODE_HYBRID_CUFFT_CALL(
                cufftXtSetGPUs( inverse_plan_, static_cast<int>( devices_.size() ), devices_.data() )
            );
            FFTM_NODE_HYBRID_CUFFT_CALL( cufftMakePlan3d(
                inverse_plan_, static_cast<int>( shape_.x ), static_cast<int>( shape_.y ),
                static_cast<int>( shape_.z ), CUFFT_Z2D, inverse_work_sizes_.data()
            ) );

            FFTM_NODE_HYBRID_CUFFT_CALL(
                cufftXtMalloc( forward_plan_, &active_, CUFFT_XT_FORMAT_INPLACE )
            );
            FFTM_NODE_HYBRID_CUFFT_CALL(
                cufftXtMalloc( inverse_plan_, &spare_, CUFFT_XT_FORMAT_INPLACE_SHUFFLED )
            );
            verify_descriptors_();
        }
        catch ( ... )
        {
            release_();
            throw;
        }
    }

    ~cuda_cufft_xt_backend() override
    {
        release_();
    }

    const char *name() const override
    {
        return "cuda-cufft-xt-d2z-z2d";
    }

    shape_3d shape() const override
    {
        return shape_;
    }

    std::size_t logical_real_bytes() const override
    {
        return checked_multiply( point_count( shape_ ), sizeof( double ), "logical real data" );
    }

    std::size_t allocated_data_bytes() const override
    {
        return descriptor_bytes_( active_ ) + descriptor_bytes_( spare_ );
    }

    std::size_t forward_work_bytes() const override
    {
        return sum_sizes_( forward_work_sizes_ );
    }

    std::size_t inverse_work_bytes() const override
    {
        return sum_sizes_( inverse_work_sizes_ );
    }

    int device_count() const override
    {
        return static_cast<int>( devices_.size() );
    }

    void initialize_input( std::uint64_t seed ) override
    {
        const std::size_t rows =
            checked_multiply( shape_.x, shape_.y, "input row count" );
        const std::size_t padded_z = padded_real_z( shape_ );
        const std::size_t storage_points = padded_real_point_count( shape_ );
        auto              host = allocate_host_buffer( storage_points );

#pragma omp parallel for schedule(static)
        for ( std::size_t row = 0; row < rows; ++row )
        {
            const std::size_t logical_base = row * shape_.z;
            const std::size_t storage_base = row * padded_z;
            for ( std::size_t z = 0; z < shape_.z; ++z )
                host.get()[storage_base + z] = input_value( logical_base + z, seed );
            for ( std::size_t z = shape_.z; z < padded_z; ++z )
                host.get()[storage_base + z] = 0.0;
        }

        copy_natural_host_to_descriptor_( host.get(), active_ );
        synchronize();
    }

    void forward() override
    {
        FFTM_NODE_HYBRID_CUFFT_CALL( cufftXtExecDescriptorD2Z( forward_plan_, active_, active_ ) );
    }

    void inverse() override
    {
        FFTM_NODE_HYBRID_CUFFT_CALL( cufftXtExecDescriptorZ2D( inverse_plan_, active_, active_ ) );
    }

    void normalize_inverse_output() override
    {
        const double factor = 1.0 / static_cast<double>( point_count( shape_ ) );
        for ( const auto &segment : descriptor_segments_( active_ ) )
        {
            if ( segment.bytes % sizeof( double ) != 0 )
                throw std::runtime_error( "cuFFT Xt real segment is not double aligned" );
            const std::size_t count = segment.bytes / sizeof( double );
            if ( count == 0 )
                continue;

            FFTM_NODE_HYBRID_CUDA_CALL( cudaSetDevice( segment.device_ordinal ) );
            constexpr unsigned int threads = 256;
            const std::size_t required_blocks = ( count + threads - 1 ) / threads;
            const unsigned int blocks = static_cast<unsigned int>(
                std::min<std::size_t>( required_blocks, 65535 )
            );
            scale_real_values<<<blocks, threads>>>(
                static_cast<double *>( segment.data ), count, factor
            );
            FFTM_NODE_HYBRID_CUDA_CALL( cudaGetLastError() );
        }
    }

    void synchronize() override
    {
        for ( int device : devices_ )
        {
            FFTM_NODE_HYBRID_CUDA_CALL( cudaSetDevice( device ) );
            FFTM_NODE_HYBRID_CUDA_CALL( cudaDeviceSynchronize() );
        }
    }

    std::vector<device_segment_view> send_segments() override
    {
        return descriptor_segments_( active_ );
    }

    std::vector<device_segment_view> receive_segments() override
    {
        return descriptor_segments_( spare_ );
    }

    void select_device( int device_ordinal ) override
    {
        FFTM_NODE_HYBRID_CUDA_CALL( cudaSetDevice( device_ordinal ) );
    }

    void accept_received_spectrum() override
    {
        std::swap( active_, spare_ );
    }

    validation_result validate_input( std::uint64_t seed ) override
    {
        synchronize();
        const std::size_t logical_points = point_count( shape_ );
        const std::size_t rows =
            checked_multiply( shape_.x, shape_.y, "validation row count" );
        const std::size_t padded_z = padded_real_z( shape_ );
        const std::size_t storage_points = padded_real_point_count( shape_ );
        auto              host = allocate_host_buffer( storage_points );
        copy_natural_descriptor_to_host_( active_, host.get() );

        const double scale = static_cast<double>( logical_points );
        double       error_norm2 = 0.0;
        double       reference_norm2 = 0.0;
        double       max_abs = 0.0;

#pragma omp parallel for reduction( + : error_norm2, reference_norm2 ) reduction( max : max_abs ) schedule(static)
        for ( std::size_t row = 0; row < rows; ++row )
        {
            const std::size_t logical_base = row * shape_.z;
            const std::size_t storage_base = row * padded_z;
            for ( std::size_t z = 0; z < shape_.z; ++z )
            {
                const double expected = input_value( logical_base + z, seed );
                const double actual = host.get()[storage_base + z] / scale;
                const double error = actual - expected;
                error_norm2 += error * error;
                reference_norm2 += expected * expected;
                max_abs = std::max( max_abs, std::abs( error ) );
            }
        }

        validation_result result;
        result.relative_l2 =
            reference_norm2 == 0.0 ? std::sqrt( error_norm2 ) : std::sqrt( error_norm2 / reference_norm2 );
        result.max_abs = max_abs;
        return result;
    }

private:
    static std::size_t sum_sizes_( const std::vector<std::size_t> &sizes )
    {
        std::size_t total = 0;
        for ( std::size_t size : sizes )
        {
            if ( size > std::numeric_limits<std::size_t>::max() - total )
                throw std::overflow_error( "cuFFT Xt work-size sum overflow" );
            total += size;
        }
        return total;
    }

    static std::size_t descriptor_bytes_( const cudaLibXtDesc *descriptor )
    {
        if ( descriptor == nullptr || descriptor->descriptor == nullptr )
            return 0;
        std::size_t total = 0;
        for ( int index = 0; index < descriptor->descriptor->nGPUs; ++index )
        {
            const std::size_t size = descriptor->descriptor->size[index];
            if ( size > std::numeric_limits<std::size_t>::max() - total )
                throw std::overflow_error( "cuFFT Xt descriptor-size sum overflow" );
            total += size;
        }
        return total;
    }

    static std::vector<device_segment_view> descriptor_segments_( cudaLibXtDesc *descriptor )
    {
        if ( descriptor == nullptr || descriptor->descriptor == nullptr )
            throw std::logic_error( "cuFFT Xt descriptor is not initialized" );
        std::vector<device_segment_view> result;
        result.reserve( static_cast<std::size_t>( descriptor->descriptor->nGPUs ) );
        for ( int index = 0; index < descriptor->descriptor->nGPUs; ++index )
        {
            result.push_back( device_segment_view{
                descriptor->descriptor->data[index],
                descriptor->descriptor->size[index],
                descriptor->descriptor->GPUs[index]
            } );
        }
        return result;
    }

    std::size_t local_x_size_( std::size_t device_index ) const
    {
        const std::size_t device_count = devices_.size();
        const std::size_t base = shape_.x / device_count;
        const std::size_t remainder = shape_.x % device_count;
        return base + ( device_index < remainder ? 1 : 0 );
    }

    std::size_t natural_segment_bytes_( std::size_t device_index ) const
    {
        return checked_multiply(
            checked_multiply(
                local_x_size_( device_index ), shape_.y, "natural descriptor segment"
            ),
            checked_multiply(
                padded_real_z( shape_ ), sizeof( double ), "natural descriptor row"
            ),
            "natural descriptor segment"
        );
    }

    void copy_natural_host_to_descriptor_( const double *host, cudaLibXtDesc *descriptor )
    {
        const auto segments = descriptor_segments_( descriptor );
        std::size_t host_offset_bytes = 0;
        const char *host_bytes = reinterpret_cast<const char *>( host );
        for ( std::size_t index = 0; index < segments.size(); ++index )
        {
            const std::size_t bytes = natural_segment_bytes_( index );
            if ( segments[index].bytes < bytes )
                throw std::runtime_error( "cuFFT Xt natural input segment is undersized" );
            FFTM_NODE_HYBRID_CUDA_CALL( cudaSetDevice( segments[index].device_ordinal ) );
            FFTM_NODE_HYBRID_CUDA_CALL( cudaMemcpy(
                segments[index].data, host_bytes + host_offset_bytes,
                bytes, cudaMemcpyHostToDevice
            ) );
            host_offset_bytes += bytes;
        }
        if ( host_offset_bytes !=
             checked_multiply(
                 padded_real_point_count( shape_ ), sizeof( double ), "natural host input"
             ) )
        {
            throw std::logic_error( "cuFFT Xt natural input decomposition is incomplete" );
        }
    }

    void copy_natural_descriptor_to_host_( cudaLibXtDesc *descriptor, double *host )
    {
        const auto segments = descriptor_segments_( descriptor );
        std::size_t host_offset_bytes = 0;
        char *host_bytes = reinterpret_cast<char *>( host );
        for ( std::size_t index = 0; index < segments.size(); ++index )
        {
            const std::size_t bytes = natural_segment_bytes_( index );
            if ( segments[index].bytes < bytes )
                throw std::runtime_error( "cuFFT Xt natural output segment is undersized" );
            FFTM_NODE_HYBRID_CUDA_CALL( cudaSetDevice( segments[index].device_ordinal ) );
            FFTM_NODE_HYBRID_CUDA_CALL( cudaMemcpy(
                host_bytes + host_offset_bytes, segments[index].data,
                bytes, cudaMemcpyDeviceToHost
            ) );
            host_offset_bytes += bytes;
        }
        if ( host_offset_bytes !=
             checked_multiply(
                 padded_real_point_count( shape_ ), sizeof( double ), "natural host output"
             ) )
        {
            throw std::logic_error( "cuFFT Xt natural output decomposition is incomplete" );
        }
    }

    void verify_descriptors_() const
    {
        if ( active_ == nullptr || spare_ == nullptr || active_->descriptor == nullptr ||
             spare_->descriptor == nullptr )
        {
            throw std::runtime_error( "cuFFT Xt did not create both data descriptors" );
        }
        if ( active_->descriptor->nGPUs != static_cast<int>( devices_.size() ) ||
             spare_->descriptor->nGPUs != static_cast<int>( devices_.size() ) )
        {
            throw std::runtime_error( "cuFFT Xt descriptor GPU count mismatch" );
        }
        for ( std::size_t index = 0; index < devices_.size(); ++index )
        {
            if ( active_->descriptor->GPUs[index] != devices_[index] ||
                 spare_->descriptor->GPUs[index] != devices_[index] )
            {
                throw std::runtime_error( "cuFFT Xt descriptor device ordering mismatch" );
            }
            if ( active_->descriptor->size[index] != spare_->descriptor->size[index] )
            {
                throw std::runtime_error(
                    "cuFFT Xt input and shuffled descriptors have different segment sizes"
                );
            }
        }
    }

    void release_() noexcept
    {
        if ( active_ != nullptr )
        {
            cufftXtFree( active_ );
            active_ = nullptr;
        }
        if ( spare_ != nullptr )
        {
            cufftXtFree( spare_ );
            spare_ = nullptr;
        }
        if ( inverse_plan_ != 0 )
        {
            cufftDestroy( inverse_plan_ );
            inverse_plan_ = 0;
        }
        if ( forward_plan_ != 0 )
        {
            cufftDestroy( forward_plan_ );
            forward_plan_ = 0;
        }
    }

    shape_3d                shape_;
    std::vector<int>        devices_;
    std::vector<std::size_t> forward_work_sizes_;
    std::vector<std::size_t> inverse_work_sizes_;
    cufftHandle             forward_plan_ = 0;
    cufftHandle             inverse_plan_ = 0;
    cudaLibXtDesc          *active_ = nullptr;
    cudaLibXtDesc          *spare_ = nullptr;
};

}

int available_cuda_device_count()
{
    int count = 0;
    FFTM_NODE_HYBRID_CUDA_CALL( cudaGetDeviceCount( &count ) );
    return count;
}

std::unique_ptr<node_fft_backend>
make_cuda_cufft_xt_backend( const shape_3d &shape, const std::vector<int> &device_ordinals )
{
    return std::unique_ptr<node_fft_backend>(
        new cuda_cufft_xt_backend( shape, device_ordinals )
    );
}

}
}
}
