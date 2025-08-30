#include <memory>
#include <external_wrap/cufft_wrap_many.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/communication/mpi_comm_info.h>
#include <fftm.hpp>


int main(int argc, char *argv[])
{
    using comm_info_type = scfd::communication::mpi_comm_info;
    using base_fft_t = fftm::wrap::cufft_wrap_many<double>;
    using fftm_t = fftm::fftm<base_fft_t, comm_info_type>;

    scfd::communication::mpi_wrap mpi(argc, argv);
    comm_info_type comm_info = mpi.comm_world();
    
    auto base_fft = std::make_shared<base_fft_t>();
    auto fftm = std::make_shared<fftm_t>(base_fft, comm_info);
    fftm->init_test();
    
    return 0;
}