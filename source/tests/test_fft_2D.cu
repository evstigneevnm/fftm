#include <memory>
#include <thrust/complex.h>

#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/backend/cuda.h>

#include <contrib/scfd/test/arrays/custom_index_arranger.h>

#include <external_wrap/cufft_wrap_many.h>


namespace scfd
{
namespace arrays
{


template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1>
using custom_arranger_01_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, 0, 1 >;
template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1>
using custom_arranger_10_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, 1, 0 >;


template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1, scfd::arrays::ordinal_type Dim2>
using custom_arranger_012_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, Dim2, 0, 1, 2 >;

template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1, scfd::arrays::ordinal_type Dim2>
using custom_arranger_102_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, Dim2, 1, 0, 2 >;

template<scfd::arrays::ordinal_type Dim0, scfd::arrays::ordinal_type Dim1, scfd::arrays::ordinal_type Dim2>
using custom_arranger_201_t = scfd::arrays::custom_index_arranger<Dim0, Dim1, Dim2, 2, 0, 1 >;

}
}


int main(int argc, char const *argv[])
{
    using T = double;
    const int dim = 2;
    // using C = thrust::complex<T>;//std::complex<T>;
    using idx_t = scfd::static_vec::vec<int, dim>;
    using backend_t = scfd::backend::cuda;
    using for_each_t = backend_t::for_each_nd_type<dim>;
    using memory_t = backend_t::memory_type;

    // using tensor3_012_t = scfd::arrays::tensor_array_nd<T, dim, memory_t, scfd::arrays::custom_arranger_012_t>;
    // using tensor3_102_t = scfd::arrays::tensor_array_nd<T, dim, memory_t, scfd::arrays::custom_arranger_102_t>;
    // using tensor3_201_t = scfd::arrays::tensor_array_nd<T, dim, memory_t, scfd::arrays::custom_arranger_201_t>;
    
    using tensor2_01_t = scfd::arrays::tensor_array_nd<T, dim, memory_t, scfd::arrays::custom_arranger_01_t>;
    using tensor2_10_t = scfd::arrays::tensor_array_nd<T, dim, memory_t, scfd::arrays::custom_arranger_10_t>;

    using base_fft_t = fftm::wrap::cufft_wrap_many<T>;
    using C = typename base_fft_t::complex;

    base_fft_t fft;

    std::size_t Nx = 30, Ny = 50;


    
    // X-plan
    long long int n[2] = {static_cast<long long int>(Nx), static_cast<long long int>(Ny)};
    long long int inembed[2] = {1,1};  
    long long int istride[2] = {1,1}; 
    long long int idist[2] = {static_cast<long long int>(Nx), static_cast<long long int>(Ny)};
    long long int onembed[2] = {static_cast<long long int>(Ny), static_cast<long long int>(Nx)};
    long long int ostride[2] = {static_cast<long long int>(Ny), static_cast<long long int>(Nx)};
    long long int odist[2] = {1,1};
    long long int batch[2] = {static_cast<long long int>(Ny), static_cast<long long int>(Nx)};

    fft.add_plan_1D("1D_x_direction", n[0], inembed[0], istride[0], idist[0], onembed[0], ostride[0], odist[0], fftm::direction::R2C, batch[0]);


    // Y-plan
    fft.add_plan_1D("1D_y_direction", n[0], inembed[0], istride[0], idist[0], onembed[0], ostride[0], odist[0], fftm::direction::C2C, batch[0]);

    fft.activate();
    auto fft_work_size = fft.get_work_size();
    std::cout << "work_size: " << fft_work_size*1.0e-6 << "MB." << std::endl;

    

    

    
    return 0;
}