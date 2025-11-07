#ifndef __FFTM_FFTM_HPP__
#define __FFTM_FFTM_HPP__

#include <iostream>
#include <memory>
#include <utility>
#include <scfd/utils/log_mpi.h>
#include <scfd/utils/safe_call.h>
#include "fft_partitioning.h"
#include "mpi_fft_distributor.h"

namespace fftm
{

template <class BaseFFT, class MPIComm, class Backend, class Log = scfd::utils::log_mpi>
class fftm
{
    using T = typename BaseFFT::real;
    using C = typename BaseFFT::complex;
    using memory_t = typename Backend::memory_type;
    using for_each_t = typename Backend::for_each_type<std::size_t>;
    using for_each_nd3_t = typename Backend::for_each_nd_type<3, std::size_t>;
    using reduce_type = typename Backend::reduce_type;
    using partitioning_t = fft_partitioning<MPIComm>;
    using distributor_t  = communication::mpi_fft_distributor<T, C, MPIComm, Log, memory_t, for_each_t>; 


public:
    fftm(std::shared_ptr<BaseFFT> &base_fft, const MPIComm& mpi, const Log& log = scfd::utils::log_mpi() ):
    base_fft_(base_fft),
    mpi_(mpi),
    log_(log), 
    init_done(false)
    {
        #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        log_.info("no gpu aware mpi.");
        #else
        log_.info("using gpu aware mpi.");
        #endif 
        partitioning_ = std::make_shared<partitioning_t>(mpi_);
        distributor_ = std::make_shared<distributor_t>(mpi_, log_);
    }
    ~fftm()
    {}

    auto get_local_input_sizes() 
    {
        if (!init_done)
        {
            throw std::logic_error("Call init() first");
        }
        auto szs = std::get<0>( partitioning_->get_partitioning_3D() );
        using sz_t = std::size_t;
        return std::tuple< sz_t,sz_t,sz_t >(szs.size_x[myid_i_], szs.size_y[myid_j_], szs.size_z[0]);
    }
    auto get_local_output_sizes() 
        {
        if (!init_done)
        {
            throw std::logic_error("Call init() first");
        }        
        auto szs = std::get<2>( partitioning_->get_partitioning_3D() );
        using sz_t = std::size_t;
        return  std::tuple< sz_t,sz_t,sz_t >( szs.size_x[0], szs.size_y[myid_i_], szs.size_z[myid_j_] );
    }

    void init(const processor_grid& pg, const global_sizes& gs)
    {
        partitioning_->init(pg, gs);
        std::tie(myid_i_, myid_j_, myid_k_) = partitioning_->get_my_grid();
        bool _4D = gs.is_4D();
        if(_4D)
        {

        }
        else
        {

            distributor_->init(partitioning_);

            auto part = partitioning_->get_partitioning_3D();
            auto input_dim = std::get<0>(part);
            auto transpose1 = std::get<1>(part);
            auto transpose2 = std::get<2>(part);

            long long int batch[3] = {static_cast<long long int>(input_dim.size_y[myid_j_]*input_dim.size_x[myid_i_]), 
            static_cast<long long int>(transpose1.size_z[myid_j_]*transpose1.size_x[myid_i_]), 
            static_cast<long long int>(transpose2.size_z[myid_j_]*transpose2.size_y[myid_i_])};
            long long int n[3] = {static_cast<long long int>(transpose2.size_x[0]), static_cast<long long int>(transpose1.size_y[0]), 
                static_cast<long long int>(input_dim.size_z[0])};
            long long int inembed[3] = {1, 1, 1};
            long long int onembed[3] = {static_cast<long long int>(transpose2.size_z[myid_j_]*transpose2.size_y[myid_i_]), 
                static_cast<long long int>(transpose1.size_z[myid_j_]*transpose1.size_x[myid_i_]),
                static_cast<long long int>(input_dim.size_y[myid_j_]*input_dim.size_x[myid_i_])};
            long long int idist[3] = {static_cast<long long int>(transpose2.size_x[0]), 
                static_cast<long long int>(transpose1.size_y[0]), 
                static_cast<long long int>(input_dim.size_z[0])};
            long long int odist[3] = {1, 1, 1};   

/*            
            base_fft_->add_plan_1D("R2C_1_direct", n[2], inembed[2], inembed[2], idist[2], onembed[2], onembed[2], odist[2], direction::R2C, batch[0]);
            base_fft_->add_plan_1D("C2R_1_inverse",n[2], onembed[2], onembed[2], odist[2], inembed[2], inembed[2], idist[2], direction::C2R, batch[0]);

            base_fft_->add_plan_1D("C2C_2_direct", n[1], inembed[1], inembed[1], idist[1], onembed[1], onembed[1], odist[1], direction::C2C, batch[1]);
            base_fft_->add_plan_1D("C2C_2_inverse", n[1], onembed[1], onembed[1], odist[1], inembed[1], inembed[1], idist[1], direction::C2C, batch[1]);

            base_fft_->add_plan_1D("C2C_3_direct", n[0], inembed[0], inembed[0], idist[0], onembed[0], onembed[0], odist[0], direction::C2C, batch[0]);
            base_fft_->add_plan_1D("C2C_3_inverse", n[0], onembed[0], onembed[0], odist[0], inembed[0], inembed[0], idist[0], direction::C2C, batch[0]);
            
            base_fft_->activate();
*/

            init_done = true;
        }
        
    }

    template<class ArrayIn, class ArrayOut>
    void forwardR(const ArrayIn& in, ArrayOut& out)
    {
        // void (*func_ptr)(void*);
        // func_ptr = (void (*)(void*)) distributor_->template first_transpose< ArrayIn, ArrayOut>;
        // SCFD_SAFE_CALL( func_ptr(in, out, true) );

        SCFD_SAFE_CALL( (distributor_->template first_transpose< ArrayIn, ArrayOut>( in, out, true )) );
        
    }


    void init_test()
    {
        
// pidx:1, pidx_i:0, pidx_j:1, batch[0]:175104, batch[1]:87552, batch[2]:87552
// pidx:1, pidx_i:0, pidx_j:1, n[0]:1024, n[1]:1024, n[2]:1024
// pidx:1, pidx_i:0, pidx_j:1, inembed[0]:1, inembed[1]:1, inembed[2]:1
// pidx:1, pidx_i:0, pidx_j:1, onembed[0]:87552, onembed[1]:87552, onembed[2]:175104
// pidx:1, pidx_i:0, pidx_j:1, idist[0]:1024, idist[1]:1024, idist[2]:1024
// pidx:1, pidx_i:0, pidx_j:1, odist[0]:1, odist[1]:1, odist[2]:1

// cufftMakePlanMany64(planR2C, 1, &n[2], //plan, rank, *n
            // &inembed[2], inembed[2], idist[2], //*inembed, istride, idist
            // &onembed[2], onembed[2], odist[2], //*onembed, ostride, odist
            // cuFFT<T>::R2Ctype, batch[0], &ws_r2c)

        long long int n = 1024;
        long long int inembed = 1;
        long long int istride = 1;
        long long int idist = 1024;
        long long int onembed = 87552;
        long long int ostride = 87552;
        long long int odist = 1; 
        long long int batch = 175104;

        base_fft_->add_plan_1D("test_R2C",n, inembed, istride, idist, onembed, ostride, odist, direction::R2C, batch );
        base_fft_->add_plan_1D("test_C2R",n, inembed, istride, idist, onembed, ostride, odist, direction::C2R, batch );
        base_fft_->add_plan_1D("test_C2C",n, inembed, istride, idist, onembed, ostride, odist, direction::C2C, batch );
        base_fft_->activate();

        try
        {
            base_fft_->add_plan_1D("test_R2C",n, inembed, istride, idist, onembed, ostride, odist, direction::R2C, batch );
        }
        catch(const std::logic_error& e)
        {
            std::cout << "test logic 1: " << e.what() << std::endl;
        }
        try
        {
            base_fft_->activate();
        }
        catch(const std::logic_error& e)
        {
            std::cout << "test logic 2: " << e.what() << std::endl;
        }
        auto wsize = base_fft_->get_work_size();
        std::cout << "work_size: " << wsize*1.0e-9 << "GB." << std::endl;

    }
    
private:
    std::shared_ptr<BaseFFT> base_fft_;
    Log log_;
    MPIComm mpi_;
    std::shared_ptr<partitioning_t> partitioning_;
    int myid_i_, myid_j_, myid_k_;
    std::shared_ptr<distributor_t> distributor_;
    bool init_done;

};



}

#endif // __FFTM_FFTM_HPP__