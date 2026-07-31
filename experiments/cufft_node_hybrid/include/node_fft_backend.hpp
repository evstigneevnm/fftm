#ifndef FFTM_EXPERIMENTS_CUFFT_NODE_HYBRID_NODE_FFT_BACKEND_HPP
#define FFTM_EXPERIMENTS_CUFFT_NODE_HYBRID_NODE_FFT_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fftm
{
namespace experiments
{
namespace node_hybrid
{

struct shape_3d
{
    std::size_t x = 0;
    std::size_t y = 0;
    std::size_t z = 0;
};

struct device_segment_view
{
    void       *data = nullptr;
    std::size_t bytes = 0;
    int         device_ordinal = -1;
};

struct validation_result
{
    double relative_l2 = 0.0;
    double max_abs = 0.0;
};

class node_fft_backend
{
public:
    virtual ~node_fft_backend() = default;

    virtual const char *name() const = 0;
    virtual shape_3d shape() const = 0;
    virtual std::size_t logical_real_bytes() const = 0;
    virtual std::size_t allocated_data_bytes() const = 0;
    virtual std::size_t forward_work_bytes() const = 0;
    virtual std::size_t inverse_work_bytes() const = 0;
    virtual int device_count() const = 0;

    virtual void initialize_input( std::uint64_t seed ) = 0;
    virtual void forward() = 0;
    virtual void inverse() = 0;
    virtual void normalize_inverse_output() = 0;
    virtual void synchronize() = 0;

    virtual std::vector<device_segment_view> send_segments() = 0;
    virtual std::vector<device_segment_view> receive_segments() = 0;
    virtual void select_device( int device_ordinal ) = 0;
    virtual void accept_received_spectrum() = 0;

    virtual validation_result validate_input( std::uint64_t seed ) = 0;
};

int available_cuda_device_count();

std::unique_ptr<node_fft_backend>
make_cuda_cufft_xt_backend( const shape_3d &shape, const std::vector<int> &device_ordinals );

}
}
}

#endif
