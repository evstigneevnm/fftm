#include <memory>
#include <external_wrap/cufft_wrap_many.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/mpi_comm.h>
#include <scfd/communication/mpi_comm_info.h>
#include <scfd/backend/cuda.h>
#include <scfd/backend/omp.h>

#include <scfd/arrays/array_nd.h>

#include <fftm.hpp>

#include <complex>

int main(int argc, char *argv[])
{
    using T = double;
    using C = std::complex<T>;
    using comm_info_type = scfd::communication::mpi_comm_info;
    using base_fft_t = fftm::wrap::cufft_wrap_many<T>;
    // using backend_t = scfd::backend::cuda;
    using backend_t = scfd::backend::omp;
    using fftm_t = fftm::fftm<base_fft_t, comm_info_type, backend_t>;

    using array_R_type = scfd::arrays::array_nd<T ,3, backend_t::memory_type>;
    using array_C_type = scfd::arrays::array_nd<C ,3, backend_t::memory_type>;

    scfd::communication::mpi_wrap mpi(argc, argv);
    comm_info_type comm_info = mpi.comm_world();

    auto base_fft = std::make_shared<base_fft_t>();
    auto fftm = std::make_shared<fftm_t>(base_fft, comm_info);
    fftm->init({3,3},{128,128,128});
    auto input_size = fftm->get_local_input_sizes();
    auto output_size = fftm->get_local_output_sizes();

    std::cout << comm_info.myid << " (input): "  << std::get<0>(input_size) << " " << std::get<1>(input_size) << " " <<  std::get<2>(input_size)  << std::endl;
    std::cout << comm_info.myid << " (output): "  << std::get<0>(output_size) << " " << std::get<1>(output_size) << " " <<  std::get<2>(output_size)  << std::endl;
    array_R_type in;
    array_C_type out;
    in.init( std::get<0>(input_size), std::get<1>(input_size), std::get<2>(input_size) );
    out.init( std::get<0>(output_size), std::get<1>(output_size), std::get<2>(output_size) );

    // fftm->init_test();


    fftm->forwardR(in, out);
    
    return 0;
}