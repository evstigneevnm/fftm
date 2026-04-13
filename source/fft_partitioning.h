#ifndef __FFTM_FFT_PARTITIONING_H__
#define __FFTM_FFT_PARTITIONING_H__


#include <stdexcept>
#include <vector>
namespace fftm
{


struct processor_grid
{

    int prod() const
    {
        _4D = p3 > 0;
        return p1*p2*( p3 == 0?1:p3  );
    }
    void init(std::size_t p1, std::size_t p2, std::size_t p3 = 0)
    {
        this->p1 = p1;
        this->p2 = p2;
        this->p3 = p3;
        _4D = p3 > 0;
    }
    std::size_t p1{0}, p2{0}, p3{0};
    mutable bool _4D;
};
struct global_sizes
{

    bool is_4D() const
    {
        _4D = (Nw > 0);
        return _4D;
    }
    void init(std::size_t Nx, std::size_t Ny, std::size_t Nz, std::size_t Nw = 0)
    {
        this->Nx = Nx;
        this->Ny = Ny;
        this->Nz = Nz;
        this->Nw = Nw;
        _4D = Nw > 0;
    }    
    ~global_sizes() = default;
    std::size_t Nx{0}, Ny{0}, Nz{0}, Nw{0};
    mutable bool _4D;
};



struct partition
{
    void compute_offsets(bool use_4D) 
    {
        start_x.clear();
        start_y.clear();
        start_z.clear();
        start_w.clear();
        computeStart(size_x, start_x);
        computeStart(size_y, start_y);
        computeStart(size_z, start_z);
        if(use_4D)
        {
            computeStart(size_w, start_w);
        }
    }

    std::vector<std::size_t> size_x;
    std::vector<std::size_t> size_y;
    std::vector<std::size_t> size_z;
    std::vector<std::size_t> size_w;

    std::vector<std::size_t> start_x;
    std::vector<std::size_t> start_y;
    std::vector<std::size_t> start_z;
    std::vector<std::size_t> start_w;

    template<class MPIComm>
    void debug_plot(const MPIComm& mpi_, std::tuple<int,int,int> processor_grid)
    {
        int myid_i,myid_j, myid_k;
        std::tie(myid_i, myid_j, myid_k) = processor_grid;
        for(int k=0; k<mpi_.num_procs; k++)
        {
            mpi_.barrier();
            if (mpi_.myid != k)
                continue;
            std::cout << "---" << std::endl;
            for(int j = 0; j< size_x.size(); j++)
            {
                std::cout << "myid = " << mpi_.myid << ", myid_grid = (" <<  myid_i << "," << myid_j << "," << myid_k << "), j = " << j << ", size_x[j] = " << size_x[j] << std::endl;
            }
            for(int j = 0; j< size_y.size(); j++)
            {
                std::cout << "myid = " << mpi_.myid << ", myid_grid = (" <<  myid_i << "," << myid_j << "," << myid_k << "), j = " << j << ", size_y[j] = " << size_y[j] << std::endl;
            }
            for(int j = 0; j< size_z.size(); j++)
            {
                std::cout << "myid = " << mpi_.myid << ", myid_grid = (" <<  myid_i << "," << myid_j << "," << myid_k << "), j = " << j << ", size_z[j] = " << size_z[j] << std::endl;
            }
            if(size_w.size() > 0)
            {
                for(int j = 0; j< size_w.size(); j++)
                {
                    std::cout << "myid = " << mpi_.myid << ", myid_grid = (" <<  myid_i << "," << myid_j << "," << myid_k << "), j = " << j << ", size_w[j] = " << size_w[j] << std::endl;
                }
            }
        }
        mpi_.barrier();        
    }

private:
    void computeStart(const std::vector<std::size_t>& size, std::vector<std::size_t>& start)
    {
        std::size_t offset = 0;
        for (std::size_t j = 0; j < size.size(); j++)
        {
            start.push_back(offset);
            offset += size[j];
        }
    }



};


template<class MPIComm>
class fft_partitioning
{
public:
    fft_partitioning(const MPIComm& mpi):
    mpi_(mpi)
    {}
    ~fft_partitioning() = default;

    // 1D FFT is applied to the right-most axis direction using the rest in batch mode, i.e. when we use "xyz", then fft is applied to "z" direction.
    // 3D transopse is as follows: xyz->xzy>zyx.
    // 4D transpose is as follows: xyzw->xywz->xzwy->yzwx
    void init(const processor_grid& p_g, const global_sizes& g_s)
    {
        p_grid_.init( p_g.p1, p_g.p2, p_g.p3 );
        g_sizes_.init(g_s.Nx, g_s.Ny, g_s.Nz, g_s.Nw);
        if(p_grid_.prod() != mpi_.num_procs)
        {
            throw std::logic_error("product of process grids (" + std::to_string(p_grid_.prod()) + ") != number of mpi processes (" + std::to_string(mpi_.num_procs) + ").");
        }
        if( ((g_sizes_.Nw == 0)&&(p_grid_.p3 != 0)) || ((g_sizes_.Nw != 0)&&(p_grid_.p3 == 0)) )
        {
            
            throw std::logic_error("fft_partitioning: g_sizes_.Nw = " + std::to_string(g_sizes_.Nw) + " and p_grid_.p3 = " +  std::to_string(p_grid_.p3) );
        }
        else if( g_sizes_._4D)
        {
            // myid = myid_i*p2*p3+myid_j*p3+myid_k
            myid_i = mpi_.myid/(p_grid_.p2*p_grid_.p3);
            myid_j = mpi_.myid%(p_grid_.p2*p_grid_.p3)/p_grid_.p3;
            myid_k = mpi_.myid%p_grid_.p3;
            init_4D();
            _4D = true;
        }
        else
        {
            // myid = myid_i*p2+myid_j
            myid_i = mpi_.myid/p_grid_.p2;
            myid_j = mpi_.myid%p_grid_.p2;
            myid_k = 0;
            init_3D();
            _4D = false;
        }        
    }
    void init_3D()
    {
        input_dim.size_x.resize(p_grid_.p1, g_sizes_.Nx/p_grid_.p1);
        for (std::size_t j = 0; j < g_sizes_.Nx%p_grid_.p1; j++)
            input_dim.size_x[j]++;

        input_dim.size_y.resize(p_grid_.p2, g_sizes_.Ny/p_grid_.p2);
        for (std::size_t j = 0; j < g_sizes_.Ny%p_grid_.p2; j++)
            input_dim.size_y[j]++;

        input_dim.size_z.resize(1, g_sizes_.Nz);    
        input_dim.compute_offsets(g_sizes_._4D);
        // input_dim.debug_plot(mpi_, {myid_i, myid_j, myid_k});

        transpose1.size_x = input_dim.size_x;
        transpose1.size_y.resize(1, g_sizes_.Ny);
        transpose1.size_z.resize(p_grid_.p2, (g_sizes_.Nz/2+1)/p_grid_.p2);
        for (std::size_t k = 0; k < (g_sizes_.Nz/2+1)%p_grid_.p2; k++)
            transpose1.size_z[k]++;      
        transpose1.compute_offsets(g_sizes_._4D);
        // transpose1.debug_plot(mpi_, {myid_i, myid_j, myid_k});
        
        transpose2.size_x.resize(1, g_sizes_.Nx);
        transpose2.size_y.resize(p_grid_.p1, g_sizes_.Ny/p_grid_.p1);
        for (std::size_t j = 0; j < g_sizes_.Ny%p_grid_.p1; j++)
            transpose2.size_y[j]++;
        transpose2.size_z = transpose1.size_z;
        transpose2.compute_offsets(g_sizes_._4D);
        // transpose2.debug_plot(mpi_, {myid_i, myid_j, myid_k});
    
    }

    void init_4D()
    {
        input_dim.size_x.resize(p_grid_.p1, g_sizes_.Nx/p_grid_.p1);
        for (std::size_t j = 0; j < g_sizes_.Nx%p_grid_.p1; j++)
            input_dim.size_x[j]++;
        input_dim.size_y.resize(p_grid_.p2, g_sizes_.Ny/p_grid_.p2);
        for (std::size_t j = 0; j < g_sizes_.Ny%p_grid_.p2; j++)
            input_dim.size_y[j]++;
        input_dim.size_z.resize(p_grid_.p3, g_sizes_.Nz/p_grid_.p3);
        for (std::size_t j = 0; j < g_sizes_.Nz%p_grid_.p3; j++)
            input_dim.size_z[j]++;      
        input_dim.size_w.resize(1, g_sizes_.Nw); 
        input_dim.compute_offsets(g_sizes_._4D);
        input_dim.debug_plot(mpi_, {myid_i, myid_j, myid_k});

        transpose1.size_x = input_dim.size_x;
        transpose1.size_y = input_dim.size_y;
        input_dim.size_z.resize(1, g_sizes_.Nz);
        transpose1.size_w.resize(p_grid_.p3, (g_sizes_.Nw/2+1)/p_grid_.p3);
        for (std::size_t k = 0; k < (g_sizes_.Nw/2+1)%p_grid_.p3; k++)
            transpose1.size_w[k]++;
        transpose1.compute_offsets(g_sizes_._4D);
        transpose1.debug_plot(mpi_, {myid_i, myid_j, myid_k});
        
        transpose2.size_x = transpose1.size_x;
        transpose2.size_y.resize(1, g_sizes_.Ny);
        transpose2.size_z.resize(p_grid_.p2, (g_sizes_.Nz)/p_grid_.p2);
        for (std::size_t k = 0; k < (g_sizes_.Nz)%p_grid_.p2; k++)
            transpose2.size_z[k]++;     
        transpose2.size_w = transpose1.size_w;          
        transpose2.compute_offsets(g_sizes_._4D);
        transpose2.debug_plot(mpi_, {myid_i, myid_j, myid_k});

        transpose3.size_x.resize(1, g_sizes_.Nx);
        transpose3.size_y.resize(p_grid_.p1, g_sizes_.Ny/p_grid_.p1);
        for (std::size_t j = 0; j < g_sizes_.Ny%p_grid_.p1; j++)
            transpose3.size_y[j]++;            
        transpose3.size_z = transpose2.size_z;
        transpose3.size_w = transpose2.size_w;   
        transpose3.compute_offsets(g_sizes_._4D);
        transpose3.debug_plot(mpi_, {myid_i, myid_j, myid_k});            
    }



    std::tuple<int, int, int> get_my_grid() const 
    {
        return {myid_i, myid_j, myid_k};
    }

    std::tuple<partition, partition, partition> get_partitioning_3D() const
    {
        return {input_dim, transpose1, transpose2};
    }
    std::tuple<partition, partition, partition, partition> get_partitioning_4D() const
    {
        return {input_dim, transpose1, transpose2, transpose3};
    }
    processor_grid get_process_grid() const
    {
        return p_grid_;
    }



private:
    
    int myid_i, myid_j, myid_k;

    partition input_dim, transpose1, transpose2, transpose3;

    MPIComm mpi_;
    processor_grid p_grid_;
    global_sizes g_sizes_;
    bool _4D;


};




}


#endif // __FFTM_FFT_PARTITIONING_H__
