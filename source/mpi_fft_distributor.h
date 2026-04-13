#ifndef __FFTM_MPI_FFT_DISTRIBUTOR_H__
#define __FFTM_MPI_FFT_DISTRIBUTOR_H__

#include <algorithm>
#include <mpi.h>
#include <scfd/communication/mpi_comm.h>

#include <scfd/arrays/array_nd.h>
#ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
#include <scfd/arrays/array_nd_visible.h>
#endif

#include "fft_partitioning.h"

namespace fftm
{
namespace communication
{


namespace detail
{
 __global__ void copy3D_kernel(
    char* dst_base, size_t dst_pitch, size_t dst_y_pitch,
    const char* src_base, size_t src_pitch, size_t src_y_pitch,
    size_t elem_size, size_t width, size_t height, size_t depth)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        const char* src_slice = src_base + z * src_y_pitch + y * src_pitch;
        char* dst_slice       = dst_base + z * dst_y_pitch + y * dst_pitch;

        const char* src_elem = src_slice + x * elem_size;
        char* dst_elem       = dst_slice + x * elem_size;

        for (size_t i = 0; i < elem_size; ++i)
            dst_elem[i] = src_elem[i];
    }
}



inline void getPitchedPtrs(const cudaMemcpy3DParms& p,
                           char*& dst_base, size_t& dst_pitch, size_t& dst_y_pitch,
                           const char*& src_base, size_t& src_pitch, size_t& src_y_pitch)
{
    // destination
    dst_base = (char*)p.dstPtr.ptr +
               p.dstPos.z * p.dstPtr.pitch * p.dstPtr.ysize +
               p.dstPos.y * p.dstPtr.pitch +
               p.dstPos.x;

    dst_pitch   = p.dstPtr.pitch;
    dst_y_pitch = p.dstPtr.pitch * p.dstPtr.ysize;

    // source
    src_base = (const char*)p.srcPtr.ptr +
               p.srcPos.z * p.srcPtr.pitch * p.srcPtr.ysize +
               p.srcPos.y * p.srcPtr.pitch +
               p.srcPos.x;

    src_pitch   = p.srcPtr.pitch;
    src_y_pitch = p.srcPtr.pitch * p.srcPtr.ysize;
}


inline void launchMemcpy3DKernel(const cudaMemcpy3DParms& p, cudaStream_t stream)
{
    size_t elem_size = sizeof(C_t);  // could be generalized with a param

    size_t width  = p.extent.width / elem_size;  
    size_t height = p.extent.height;
    size_t depth  = p.extent.depth;

    char* dst_base; size_t dst_pitch, dst_y_pitch;
    const char* src_base; size_t src_pitch, src_y_pitch;

    getPitchedPtrs(p, dst_base, dst_pitch, dst_y_pitch,
                      src_base, src_pitch, src_y_pitch);

    dim3 block(8, 8, 8);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        (depth  + block.z - 1) / block.z
    );

    switch (p.kind) {
        case cudaMemcpyDeviceToDevice:
            copy3D_kernel<<<grid, block, 0, stream>>>(
                dst_base, dst_pitch, dst_y_pitch,
                src_base, src_pitch, src_y_pitch,
                elem_size, width, height, depth);
            break;

        case cudaMemcpyDeviceToHost: {
            // Allocate temp buffer on device
            size_t buf_size = width * height * depth * elem_size;
            char* tmp_dev;
            CUDA_CALL(cudaMalloc(&tmp_dev, buf_size));

            // Pack device data into contiguous buffer
            copy3D_kernel<<<grid, block, 0, stream>>>(
                tmp_dev, width * elem_size, width * height * elem_size,
                src_base, src_pitch, src_y_pitch,
                elem_size, width, height, depth);

            // Async copy to host
            CUDA_CALL(cudaMemcpyAsync(p.dstPtr.ptr, tmp_dev, buf_size, cudaMemcpyDeviceToHost, stream));
            CUDA_CALL(cudaFree(tmp_dev));
            break;
        }

        case cudaMemcpyHostToDevice: {
            // Allocate temp buffer on device
            size_t buf_size = width * height * depth * elem_size;
            char* tmp_dev;
            CUDA_CALL(cudaMalloc(&tmp_dev, buf_size));

            // Async copy from host to staging buffer
            CUDA_CALL(cudaMemcpyAsync(tmp_dev, p.srcPtr.ptr, buf_size, cudaMemcpyHostToDevice, stream));

            // Scatter into pitched destination
            copy3D_kernel<<<grid, block, 0, stream>>>(
                dst_base, dst_pitch, dst_y_pitch,
                tmp_dev, width * elem_size, width * height * elem_size,
                elem_size, width, height, depth);

            CUDA_CALL(cudaFree(tmp_dev));
            break;
        }

        case cudaMemcpyHostToHost:
            // Just CPU memcpy (blocking)
            std::memcpy(p.dstPtr.ptr, p.srcPtr.ptr,
                        width * height * depth * elem_size);
            break;
    }

    CUDA_CALL(cudaGetLastError());
}

CUDA_CALL(cudaMemcpy3DAsync(&cpy_params, streams[p_j]));
launchMemcpy3DKernel(cpy_params, streams[p_j]);
CUDA_CALL(cudaDeviceSynchronize());


}


template<class R, class C, class MPIComm, class Log, class Memory, class ForEach>
class mpi_fft_distributor
{
    using partition_t = fft_partitioning<MPIComm>;
    using mpi_comm_t = scfd::communication::mpi_comm;
    using local_sizes_t = ::fftm::partition;
    //types for buffers
    using vis_array_type = scfd::arrays::array_nd_visible<char, 1, Memory>; 
    using array_type = scfd::arrays::array_nd<char ,1,Memory>; //should be char to be used for both C and R

public:
    mpi_fft_distributor(const MPIComm& mpi, const Log& log):
    mpi_(mpi),
    log_(log)
    { }
    ~mpi_fft_distributor()
    { }

    void init(std::shared_ptr<partition_t> partition_p)
    {
        partition_ = partition_p;
        
        std::tie(myid_i, myid_j, myid_k) = partition_->get_my_grid();
        process_grid_ = partition_->get_process_grid();
        
        std::tie(input_dim, transposed1_dim, transposed2_dim) = partition_->get_partitioning_3D();

        comm1_ = std::make_shared<mpi_comm_t>( std::move(mpi_.split(myid_i, myid_j)) );
        comm2_ = std::make_shared<mpi_comm_t>( std::move(mpi_.split(myid_j, myid_i)) );
        std::cout << "myid: " <<  mpi_.myid << ", comm1_->comm(): " << comm1_->comm() << ", comm2_->comm(): " << comm2_->comm() << std::endl;

        comm_order1_.clear();
        for (int j = 1; j < process_grid_.p2; j++)
        {
            comm_order1_.push_back((myid_j+j)%process_grid_.p2);
        }

        comm_order2_.clear();
        for (int i = 1; i < process_grid_.p1; i++)
        {
            comm_order2_.push_back((myid_i+i)%process_grid_.p1);
        }

        send_req.resize(std::max(process_grid_.p1, process_grid_.p2), scfd::communication::detail::mpi_request());
        recv_req.resize(std::max(process_grid_.p1, process_grid_.p2), scfd::communication::detail::mpi_request());

        domainsize_ = std::max(input_dim.size_x[myid_i]*input_dim.size_y[myid_j]*(input_dim.size_z[0]/2+1), 
        transposed1_dim.size_x[myid_i]*transposed1_dim.size_y[0]*transposed1_dim.size_z[myid_j]);
        domainsize_ = std::max(domainsize_, transposed2_dim.size_x[0]*transposed2_dim.size_y[myid_i]*transposed2_dim.size_z[myid_j]);
        
        std::cout << "domainsize_ = " << domainsize_ << std::endl;
        buf_ = std::make_unique<packet_bucket>(sizeof(C)*domainsize_);
        

    }

    template<class ArrayIn, class ArrayOut> //assume that those are SCFD arrays, because we need view_t here if CUDA_AWARE is not supported
    void first_transpose(const ArrayIn& in, ArrayOut& out, bool forward)
    {
        //TODO in scfd arrays for 'sync_to/from_array'->memory::cuda::copy_to_host add async and stream version!!!  Now we continue without it, but it will be slower, than pure CUDA version!
        //https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_g732efed5ab5cb184c920a21eb36e8ce4.html#g732efed5ab5cb184c920a21eb36e8ce4
        using Tin = typename ArrayIn::value_type;
        using Tout = typename ArrayOut::value_type;
        Tin* in_ptr;
        #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            using view_t = typename ArrayIn::view_type;
            view_t in_view =  in.create_view();
            in_ptr = in_view.raw_ptr();
        #else
            in_ptr = in.raw_ptr();
        #endif
        for (std::size_t i = 0; i < comm_order1_.size(); i++)
        {
            auto p_j = comm_order1_[i];
            // Start non-blocking MPI recv
            std::size_t islice = transposed1_dim.size_x[myid_i]*input_dim.start_y[p_j]*transposed1_dim.size_z[myid_j];
            std::size_t isize = transposed1_dim.size_x[myid_i]*input_dim.size_y[p_j]*transposed1_dim.size_z[myid_j];
            std::size_t oslice = transposed1_dim.start_z[p_j]*input_dim.size_y[myid_j]*input_dim.size_x[myid_i];
            std::size_t osize = transposed1_dim.size_z[p_j]*input_dim.size_y[myid_j]*input_dim.size_x[myid_i];

            //int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)            
            scfd::communication::detail::irecv(
                &buf_->buf()[islice], sizeof(Tout)*isize,
                scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                p_j, p_j, comm1_->comm(), &recv_req[p_j]
            );


            // int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
            scfd::communication::detail::isend(
                &in_ptr[oslice], sizeof(Tin)*osize,
                scfd::communication::detail::mpi_data_type<char>::mpi_type(),
                p_j, myid_j, comm1_->comm(), &send_req[p_j]
            );

        }
        
        cudaMemcpy3DParms cpy_params = {0};
        cpy_params.srcPos = make_cudaPos(0, 0, transposed_dim.start_z[pidx_j]);
        cpy_params.srcPtr = make_cudaPitchedPtr(complex, input_dim.size_y[pidx_j]*sizeof(C_t), input_dim.size_y[pidx_j], input_dim.size_x[pidx_i]);
        // struct cudaPitchedPtr make_cudaPitchedPtr   (   void *      d, size_t      pitch, size_t      xsz, size_t      ysz  )   
        // cudaPitchedPtr is a structure,which contains a pointer to the memory, the pitch (the width in bytes of each row), and the height (number of rows). This is useful for 2D arrays where each row may not be tightly packed in memory.
        cpy_params.extent = make_cudaExtent(input_dim.size_y[pidx_j]*sizeof(C_t), transposed_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);
        // cudaExtent is a structure that describes the size of a 3D region in memory, including width, height, and depth. It is used to specify the dimensions of the memory copy operation.




        int p = 0,count=0;
        do
        {
            p = scfd::communication::detail::waitany(process_grid_.p2, recv_req.data());
            if (p == MPI_UNDEFINED)
            {
                break;
            }
            count++;
            std::cout << "count = " << count << std::endl;
        }
        while (p != MPI_UNDEFINED);
        std::cout << "process " << mpi_.myid << " recieved " << count << " data packets" << std::endl;



        scfd::communication::detail::waitall(process_grid_.p2, send_req.data());
        for (std::size_t i = 0; i < comm_order1_.size(); i++)
        {
            auto p_j = comm_order1_[i];
            int flag_recv, flag_send;
            SCFD_MPI_SAFE_CALL(
                MPI_Test(
                    recv_req[p_j].native_ptr(), &flag_recv,
                    scfd::communication::detail::raw_status( static_cast<scfd::communication::detail::mpi_status *>( nullptr ) )
                )
            );
            SCFD_MPI_SAFE_CALL(
                MPI_Test(
                    send_req[p_j].native_ptr(), &flag_send,
                    scfd::communication::detail::raw_status( static_cast<scfd::communication::detail::mpi_status *>( nullptr ) )
                )
            );
            std::cout << "myid  = " << mpi_.myid << ", flag_recv: " << flag_recv << ", flag_send: " << flag_send << std::endl;
        }





    }




private:
    MPIComm mpi_;
    std::shared_ptr<mpi_comm_t> comm1_, comm2_;
    Log log_;
    std::shared_ptr<partition_t> partition_;
    std::vector<int> comm_order1_;
    std::vector<int> comm_order2_;
    int myid_i, myid_j, myid_k;
    std::vector<scfd::communication::detail::mpi_request> send_req;
    std::vector<scfd::communication::detail::mpi_request> recv_req;
    local_sizes_t input_dim, transposed1_dim, transposed2_dim;
    std::size_t domainsize_;
    processor_grid process_grid_;

    //in FFT both C and R are required to be sent by MPI. Hence the buffer type will be 'char' until further modified.
    struct packet_bucket    
    {
        #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        using buf_array_type = vis_array_type;
        #else
        using buf_array_type = array_type;
        #endif
        std::unique_ptr<buf_array_type>  data_buf;
        std::size_t loc_size;

        packet_bucket(const std::size_t& loc_size_p) : 
        loc_size(loc_size_p),
        data_buf(std::make_unique<buf_array_type>())
        {
            data_buf->init(loc_size_p);
        }

        #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
        array_type  buf_array_device()const
        {
            return data_buf->array();
        }
        #else
        array_type  buf_array_device()const
        {
            return *data_buf;
        }
        #endif
        char        *buf()const
        {
            return data_buf->raw_ptr();
        }
        std::size_t buf_size()const
        {
            return data_buf->total_size()*sizeof(char); // =) char is guaranteed to be 1 byte, but 1 byte can be anythng.
        }
        void        sync_from_array(const ForEach &for_each, const array_type &array)const
        {
            // detail::copy_array_nd_rect(
            //     for_each, array, loc_rect, buf_array_device()
            // );
            // #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            // data_buf->sync_from_array();
            // #endif
        }
        void        sync_to_array(const ForEach &for_each, const array_type &array)const
        {
            // #ifndef SCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
            // data_buf->sync_to_array();
            // #endif
            // //std::cout << "sync_to_array" << std::endl;
            // //auto i1 = loc_rect.i1,i2 = loc_rect.i2;
            // //ord_vec_t i1,i2;
            // auto array_sz = array.size_nd(), array_i1 = array.indexes0_nd();
            // //std::cout << "array: sz = " << array_sz[0] << "," << array_sz[1] << "," << array_sz[2] << std::endl;
            // //std::cout << "array: i1 = " << array_i1[0] << "," << array_i1[1] << "," << array_i1[2] << std::endl;
            // //std::cout << "loc_rect: i1 = " << i1[0] << "," << i1[1] << "," << i1[2] << std::endl;
            // //std::cout << "loc_rect: i2 = " << i2[0] << "," << i2[1] << "," << i2[2] << std::endl;
            // detail::copy_array_nd_rect(
            //     for_each, buf_array_device(), loc_rect, array
            // );
        }
    };
    std::unique_ptr<packet_bucket> buf_;


};



}
}

#endif // __FFTM_MPI_FFT_DISTRIBUTOR_H__
