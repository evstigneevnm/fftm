#include <iostream>
#include <memory>

#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/backend/cuda.h>

#include <contrib/scfd/test/arrays/custom_index_fast_arranger.h>

#include <external_wrap/cufft_wrap_many.h>


namespace scfd
{
namespace arrays
{

template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_012_t = scfd::arrays::custom_index_fast_arranger<0, 1, 2>::type<Dims...>;

}
}


int main( int argc, char const *argv[] )
{
    using T         = double;
    const int dim   = 3;
    using backend_t = scfd::backend::cuda;
    using memory_t  = backend_t::memory_type;

    using base_fft_t = fftm::wrap::cufft_wrap_many<T>;
    using C          = typename base_fft_t::complex;

    using tensor3_r_t = scfd::arrays::tensor_array_nd<T, dim, memory_t, scfd::arrays::custom_arranger_012_t>;
    using tensor3_c_t = scfd::arrays::tensor_array_nd<C, dim, memory_t, scfd::arrays::custom_arranger_012_t>;

    base_fft_t fft;

    std::size_t Nx = 12, Ny = 10, Nz = 8, Nz_C = Nz / 2 + 1;

    tensor3_r_t t_in;
    tensor3_c_t t_out;
    t_in.init( Nx, Ny, Nz );
    t_out.init( Nx, Ny, Nz_C );

    fft.template add_plan_3D<fftm::direction::R2C>(
        "3D_r2c", static_cast<long long int>( Nx ), static_cast<long long int>( Ny ), static_cast<long long int>( Nz ),
        static_cast<long long int>( Nx ), static_cast<long long int>( Ny ), static_cast<long long int>( Nz ), 1,
        static_cast<long long int>( Nx * Ny * Nz ), static_cast<long long int>( Nx ), static_cast<long long int>( Ny ),
        static_cast<long long int>( Nz_C ), 1, static_cast<long long int>( Nx * Ny * Nz_C ), 1
    );

    fft.activate();
    auto fft_work_size = fft.get_work_size();
    std::cout << "work_size: " << fft_work_size * 1.0e-3 << "kB." << std::endl;

    fft.template exec<tensor3_r_t, tensor3_c_t>( "3D_r2c", t_in, t_out );

    return 0;
}
