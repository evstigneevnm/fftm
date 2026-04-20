#include <iostream>
#include <memory>
#include <thrust/complex.h>

#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/backend/cuda.h>

#include <contrib/scfd/test/arrays/custom_index_fast_arranger.h>

#include <external_wrap/cufft_wrap_many.h>


namespace scfd
{
namespace arrays
{


template <scfd::arrays::ordinal_type... Dims>
using custom_arranger_01_t = scfd::arrays::custom_index_fast_arranger<0, 1>::type<Dims...>;

}
}


int main( int argc, char const *argv[] )
{
    using T         = double;
    const int dim   = 2;
    using backend_t = scfd::backend::cuda;
    using memory_t  = backend_t::memory_type;

    using base_fft_t = fftm::wrap::cufft_wrap_many<T>;
    using C          = typename base_fft_t::complex;

    using tensor2_01_t = scfd::arrays::tensor_array_nd<T, dim, memory_t, scfd::arrays::custom_arranger_01_t>;
    using tensor2_c_t  = scfd::arrays::tensor_array_nd<C, dim, memory_t, scfd::arrays::custom_arranger_01_t>;


    base_fft_t fft;

    std::size_t Nx = 30, Ny = 50, Ny_C = Ny / 2 + 1;

    tensor2_01_t t_in;
    tensor2_c_t  t_out;
    t_in.init( Nx, Ny );
    t_out.init( Nx, Ny_C );

    fft.template add_plan_2D<fftm::direction::R2C>(
        "2D_r2c", static_cast<long long int>( Nx ), static_cast<long long int>( Ny ), static_cast<long long int>( Nx ),
        static_cast<long long int>( Ny ), 1, static_cast<long long int>( Nx * Ny ), static_cast<long long int>( Nx ),
        static_cast<long long int>( Ny_C ), 1, static_cast<long long int>( Nx * Ny_C ), 1
    );

    fft.activate();
    auto fft_work_size = fft.get_work_size();
    std::cout << "work_size: " << fft_work_size * 1.0e-3 << "kB." << std::endl;


    fft.template exec<tensor2_01_t, tensor2_c_t>( "2D_r2c", t_in, t_out );


    return 0;
}
