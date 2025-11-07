#include <cuda.h>
#include <complex>
#include <scfd/arrays/array_nd.h>
#include <scfd/backend/cuda.h>
#include <scfd/utils/cuda_safe_call.h>

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


template <class T>
inline void launchMemcpy3DKernel(const cudaMemcpy3DParms& p, cudaStream_t stream)
{
    size_t elem_size = sizeof(T);  // could be generalized with a param

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
            CUDA_SAFE_CALL(cudaMalloc(&tmp_dev, buf_size));

            // Pack device data into contiguous buffer
            copy3D_kernel<<<grid, block, 0, stream>>>(
                tmp_dev, width * elem_size, width * height * elem_size,
                src_base, src_pitch, src_y_pitch,
                elem_size, width, height, depth);

            // Async copy to host
            CUDA_SAFE_CALL(cudaMemcpyAsync(p.dstPtr.ptr, tmp_dev, buf_size, cudaMemcpyDeviceToHost, stream));
            CUDA_SAFE_CALL(cudaFree(tmp_dev));
            break;
        }

        case cudaMemcpyHostToDevice: {
            // Allocate temp buffer on device
            size_t buf_size = width * height * depth * elem_size;
            char* tmp_dev;
            CUDA_SAFE_CALL(cudaMalloc(&tmp_dev, buf_size));

            // Async copy from host to staging buffer
            CUDA_SAFE_CALL(cudaMemcpyAsync(tmp_dev, p.srcPtr.ptr, buf_size, cudaMemcpyHostToDevice, stream));

            // Scatter into pitched destination
            copy3D_kernel<<<grid, block, 0, stream>>>(
                dst_base, dst_pitch, dst_y_pitch,
                tmp_dev, width * elem_size, width * height * elem_size,
                elem_size, width, height, depth);

            CUDA_SAFE_CALL(cudaFree(tmp_dev));
            break;
        }

        case cudaMemcpyHostToHost:
            // Just CPU memcpy (blocking)
            std::memcpy(p.dstPtr.ptr, p.srcPtr.ptr,
                        width * height * depth * elem_size);
            break;
    }

    CUDA_SAFE_CALL(cudaGetLastError());
}


struct partition
{
    void compute_offsets()
    {
        computeStart(size_x, start_x);
        computeStart(size_y, start_y);
        computeStart(size_z, start_z);
    }

    std::vector<std::size_t> size_x;
    std::vector<std::size_t> size_y;
    std::vector<std::size_t> size_z;
    std::vector<std::size_t> size_w;

    std::vector<std::size_t> start_x;
    std::vector<std::size_t> start_y;
    std::vector<std::size_t> start_z;
    std::vector<std::size_t> start_w;

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

int main(int argc, char const *argv[]) 
{
    static const int dim = 3;
    using T = double;
    using C = std::complex<T>;
    using idx_t = scfd::static_vec::vec<int,dim>;
    using backend_t = scfd::backend::cuda;
    using array_R_type = scfd::arrays::array_nd<T ,dim, backend_t::memory_type>;
    using array_C_type = scfd::arrays::array_nd<C ,dim, backend_t::memory_type>;

    std::size_t Nx =30, Ny=20, Nz=40;
    bool cuda_aware = true;

    partition input_dim, transposed_dim;

    array_C_type send_array;
    array_C_type temp_array;


    auto send_ptr = send_array.raw_ptr();
    auto temp_ptr = temp_array.raw_ptr();


    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.dstPos = make_cudaPos(0, 0, 0);
    cpy_params.dstPtr = make_cudaPitchedPtr(&send_ptr[transposed_dim.size_x[pidx_i]*input_dim.start_y[p_j]*transposed_dim.size_z[pidx_j]],
        input_dim.size_y[p_j]*sizeof(C), input_dim.size_y[p_j], input_dim.size_x[pidx_i]);
    cpy_params.srcPos = make_cudaPos(input_dim.start_y[p_j]*sizeof(C), 0, 0);
    cpy_params.srcPtr = make_cudaPitchedPtr(temp_ptr, Ny*sizeof(C), Ny, transposed_dim.size_x[pidx_i]);
    cpy_params.extent = make_cudaExtent(input_dim.size_y[p_j]*sizeof(C), input_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);
    cpy_params.kind   = cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;    

    CUDA_SAFE_CALL(cudaMemcpy3DAsync(&cpy_params));
    launchMemcpy3DKernel<C>(cpy_params, NULL);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    return 0;
}