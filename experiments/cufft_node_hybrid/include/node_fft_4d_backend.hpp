#ifndef FFTM_EXPERIMENTS_CUFFT_NODE_HYBRID_NODE_FFT_4D_BACKEND_HPP
#define FFTM_EXPERIMENTS_CUFFT_NODE_HYBRID_NODE_FFT_4D_BACKEND_HPP

#include "node_fft_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace fftm
{
namespace experiments
{
namespace node_hybrid
{

struct shape_4d
{
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t z = 0;
    std::size_t w = 0;
};

class node_fft_4d_backend
{
public:
    virtual ~node_fft_4d_backend() = default;

    virtual const char *name() const = 0;
    virtual const char *spectral_layout_name() const = 0;
    virtual shape_4d shape() const = 0;
    virtual int device_count() const = 0;

    virtual std::size_t logical_real_bytes() const = 0;
    virtual std::size_t allocated_data_bytes() const = 0;
    virtual std::size_t max_allocated_data_bytes_per_device() const = 0;
    virtual std::size_t shared_work_bytes() const = 0;
    virtual std::size_t max_shared_work_bytes_per_device() const = 0;
    virtual std::size_t forward_3d_work_bytes() const = 0;
    virtual std::size_t inverse_3d_work_bytes() const = 0;
    virtual std::size_t x_fft_work_bytes() const = 0;
    virtual std::size_t transpose_bytes() const = 0;

    virtual void initialize_input( std::uint64_t seed ) = 0;

    virtual void forward_yzw() = 0;
    virtual void transpose_x_to_z() = 0;
    virtual void forward_x() = 0;
    virtual void inverse_x() = 0;
    virtual void transpose_z_to_x() = 0;
    virtual void inverse_yzw() = 0;

    virtual void normalize_inverse_output() = 0;
    virtual void synchronize() = 0;
    virtual validation_result validate_input( std::uint64_t seed ) = 0;
};

std::unique_ptr<node_fft_4d_backend>
make_cuda_cufft_xt_3d_1d_backend(
    const shape_4d &shape, const std::vector<int> &device_ordinals
);

}
}
}

#endif
