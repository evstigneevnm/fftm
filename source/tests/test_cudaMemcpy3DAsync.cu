#include <iostream>
#include <cuda.h>
#include <thrust/complex.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/backend/cuda.h>
#include <scfd/utils/cuda_safe_call.h>
#include <scfd/utils/safe_call.h>
#include <scfd/utils/device_tag.h>
#include <scfd/utils/init_cuda.h>
#include <scfd/for_each/cuda_nd.h>
#include <scfd/for_each/cuda_nd_impl.cuh>
#include <scfd/utils/cuda_timer_event.h>
#include <scfd/utils/system_timer_event.h>

//----


namespace io
{

template<class T, class Array>
void write_out_pos_file_scal_3D_point(const std::string& filename, const Array& U_in, std::size_t Nx_, std::size_t Ny_, std::size_t Nz_)
{
    using array_C_view_type = typename Array::view_type;

    array_C_view_type U(U_in);

    std::size_t Nx = Nx_, Ny = Ny_, Nz = Nz_;


    FILE *stream = std::fopen( filename.c_str(), "w" );
    if(stream == NULL)
    {
        throw std::runtime_error("error creating file: " + filename);
    }
    fclose(stream);


    stream=std::fopen(filename.c_str(), "a" );
    if(stream == NULL)
    {
        throw std::runtime_error("error opening file: " + filename);
    }

    fprintf( stream, "View");
    fprintf( stream, " '");
    fprintf( stream, "%s %i", filename.c_str(), 0);
    fprintf( stream, "' {\n");
    fprintf( stream, "TIME{0};\n");

    for(int j=0;j<Nx;j++){
        double xm = j;
        double xp = j+1;
        for(int k=0;k<Ny;k++){
            double ym = k;
            double yp = k+1;                
            for(int l=0;l<Nz;l++){
                double zm =l;
                double zp = l+1;

                    T par_x_mmm=0.0;
                    T par_x_pmm=0.0;
                    T par_x_ppm=0.0;
                    T par_x_ppp=0.0;
                    T par_x_mpp=0.0;
                    T par_x_mmp=0.0;
                    T par_x_pmp=0.0;
                    T par_x_mpm=0.0;


                    par_x_mmm=U(j,k,l).real();
                    par_x_pmm=U(j,k,l).real();
                    par_x_ppm=U(j,k,l).real();
                    par_x_ppp=U(j,k,l).real();
                    par_x_mpp=U(j,k,l).real();
                    par_x_mmp=U(j,k,l).real();
                    par_x_pmp=U(j,k,l).real();
                    par_x_mpm=U(j,k,l).real();
                                



                    fprintf( stream, "SH(%le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le, %le)",
                    xm, ym, zm,  
                    xp, ym, zm,  
                    xp, yp, zm,  
                    xm, yp, zm, 
                    xm, ym, zp, 
                    xp, ym, zp, 
                    xp, yp, zp,
                    xm, yp, zp);


                    fprintf( stream,"{");
                    fprintf(stream, "%le,",par_x_mmm);
                    fprintf(stream, "%le,",par_x_pmm);
                    fprintf(stream, "%le,",par_x_ppm);
                    fprintf(stream, "%le,",par_x_mpm);
                    fprintf(stream, "%le,",par_x_mmp);
                    fprintf(stream, "%le,",par_x_pmp);
                    fprintf(stream, "%le,",par_x_ppp);
                    fprintf(stream, "%le",par_x_mpp);
                    fprintf(stream, "};\n");

                // }
            }
        }
    }
    fflush(stream);  //flush all to disk

    fprintf( stream, "};");

    fclose( stream );

    std::cout << filename << " output done." << std::endl;


}
}

//-----

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

    std::cout << "width: " << width << ", height: " << height << ", depth: " << depth << std::endl;

    char* dst_base; size_t dst_pitch, dst_y_pitch;
    const char* src_base; size_t src_pitch, src_y_pitch;

    getPitchedPtrs(p, dst_base, dst_pitch, dst_y_pitch,
                      src_base, src_pitch, src_y_pitch);

    dim3 block(16, 8, 8);
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

template<class C>
cudaMemcpy3DParms set_params(int Nx, int Ny, int Nz, C* send_ptr, C* recv_ptr, partition input_dim, partition transposed_dim, bool cuda_aware, int pidx_i, int pidx_j, int p_j)
{

// cudaMemcpy3DParms Struct Reference
// Data Fields
// struct cudaArray *  dstArray
// struct cudaPos  dstPos
// struct cudaPitchedPtr   dstPtr
// struct cudaExtent   extent
// enum cudaMemcpyKind     kind
// struct cudaArray *  srcArray
// struct cudaPos  srcPos
// struct cudaPitchedPtr   srcPtr

// cudaPitchedPtr Struct Reference
// [Data types used by CUDA Runtime]
// Data Fields
// size_t  pitch
// void *  ptr
// size_t  xsize
// size_t  ysize

    std::cout << std::endl;
    for(int j = 0; j<input_dim.start_x.size(); j++)
    {
        std::cout << "index: "<< j << ", input_dim.start_x: " << input_dim.start_x[j] << ", input_dim.start_y: " << input_dim.start_y[j] << ", input_dim.start_z: " << input_dim.start_z[j] << std::endl;
    }
    
    for(int j = 0; j<input_dim.start_x.size(); j++)
    {
        std::cout << "index: "<< j << ", input_dim.size_x: " << input_dim.size_x[j] << ", input_dim.size_y: " << input_dim.size_y[j] << ", input_dim.size_z: " << input_dim.size_z[j] << std::endl;
    }

    for(int j = 0; j<input_dim.start_x.size(); j++)
    {
        std::cout << "index: "<< j << ", transposed_dim.start_x: " << transposed_dim.start_x[j] << ", transposed_dim.start_y: " << transposed_dim.start_y[j] << ", transposed_dim.start_z: " << transposed_dim.start_z[j] << std::endl;
    }

    for(int j = 0; j<input_dim.start_x.size(); j++)
    {
        std::cout << "index: "<< j << ", transposed_dim.size_x: " << transposed_dim.size_x[j] << ", transposed_dim.size_y: " << transposed_dim.size_y[j] << ", transposed_dim.size_z: " << transposed_dim.size_z[j] << std::endl;
    }


    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.dstPos = make_cudaPos(0, 0, 0);

// struct cudaPitchedPtr {
//     void *ptr;
//     size_t pitch; //bytes
//     size_t xsize; //elements
//     size_t ysize; //elements
// };

    cpy_params.dstPtr = make_cudaPitchedPtr(&recv_ptr[transposed_dim.size_x[pidx_i]*input_dim.start_y[p_j]*transposed_dim.size_z[pidx_j]], input_dim.size_y[p_j]*sizeof(C), input_dim.size_y[p_j], input_dim.size_x[pidx_i]);

    cpy_params.srcPos = make_cudaPos(input_dim.start_y[p_j]*sizeof(C), 0, 0);
    cpy_params.srcPtr = make_cudaPitchedPtr(send_ptr, Ny*sizeof(C), Ny, transposed_dim.size_x[pidx_i]);

// The extent field defines the dimensions of the transferred area in elements. If a CUDA array is participating in the copy, the extent is defined in terms of that array's elements. If no CUDA array is participating in the copy then the extents are defined in elements of unsigned char. 

    cpy_params.extent = make_cudaExtent(input_dim.size_y[p_j]*sizeof(C), input_dim.size_x[pidx_i], transposed_dim.size_z[pidx_j]);

    cpy_params.kind   = cuda_aware ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost; 
    
    std::cout << "send_ptr: " << send_ptr << ", cpy_params.srcArray: " << cpy_params.srcPtr.ptr << std::endl;
    return cpy_params;

}


template<class Idx, class Array>
struct test_diff
{
    test_diff(const Array& _f1, const Array& _f2, Array& _res) : f1(_f1), f2(_f2), res(_res) {}
    Array f1, f2, res;

    __DEVICE_TAG__ void operator()(const Idx& idx)
    {
        res(idx) =  f1(idx) - f2(idx);
    }

};


template<class Idx, class Array, int axis0, int axis1, int axis2>
struct manual_transpose
{
    manual_transpose(const Array& _f1, Array& _res) : f1(_f1), res(_res) {}
    Array f1, res;

    __DEVICE_TAG__ void operator()(const Idx& idx)
    {
        auto i = idx[axis0];
        auto j = idx[axis1];
        auto k = idx[axis2];
        res(i,j,k) =  f1(idx);
    }

};

template<class T, class Array>
T get_norm(const Array& temp_array, std::size_t sz)
{
    using array_C_view_type = typename Array::view_type;
    array_C_view_type temp_array_view(temp_array);

    T norm = 0.0;
    #pragma omp parallel for reduction (+: norm)
    for(std::size_t j = 0; j<sz; j++)
    {
        norm += thrust::abs(temp_array_view.raw_ptr()[j]);
    }

    return norm;
}





int main(int argc, char const *argv[]) 
{
    static const int dim = 3;
    using T = double;
    using C = thrust::complex<T>;//std::complex<T>;
    using idx_t = scfd::static_vec::vec<int, dim>;
    using backend_t = scfd::backend::cuda;
    using for_each_t = backend_t::for_each_nd_type<dim>;
    using array_R_type = scfd::arrays::array_nd<T, dim, backend_t::memory_type>;
    using array_C_type = scfd::arrays::array_nd<C, dim, backend_t::memory_type>;
    using array_C_view_type = array_C_type::view_type;
    using timer_t = scfd::utils::cuda_timer_event;
    
    
    timer_t cuda3d_s, cuda3d_e, my_s, my_e;

    std::size_t Nx =20, Ny=30, Nz=50;
    bool cuda_aware = true;

    int pidx_i = 0, pidx_j = 0;
    int p_j = 0;

    // scfd::utils::init_cuda_persistent();

    partition input_dim, transposed_dim;

    input_dim.size_x.resize(1, Nx);
    input_dim.size_y.resize(1, Ny);
    input_dim.size_z.resize(1, Nz);
    input_dim.compute_offsets();
    transposed_dim.size_x.resize(1, Nx);
    transposed_dim.size_y.resize(1, Nz);
    transposed_dim.size_z.resize(1, Ny);
    transposed_dim.compute_offsets();


    array_C_type send_array;
    array_C_type temp_array;
    array_C_type temp_array_check;
    array_C_type diff_array;
    array_C_type temp_manual;

    SCFD_SAFE_CALL(send_array.init(Nx,Ny,Nz));
    SCFD_SAFE_CALL(temp_array.init(Nx,Nz,Ny));
    SCFD_SAFE_CALL(temp_array_check.init(Nx,Nz,Ny));
    SCFD_SAFE_CALL( diff_array.init(Nx,Nz,Ny) );
    SCFD_SAFE_CALL( temp_manual.init(Nx,Nz,Ny) );


    array_C_view_type send_array_view(send_array);
    // array_C_view_type temp_array_check_view(temp_array_check);
    for(std::size_t j = 0; j<Nx; j++)
    for(std::size_t k = 0; k<Ny; k++)
    for(std::size_t l = 0; l<Nz; l++)
    {
        send_array_view(j,k,l) = C(k,0);
        // temp_array_check_view.raw_ptr()[j] = C(0.0, 0.1*j);
    }
    send_array_view.release(true);
    // temp_array_check_view.release(true);

    C* send_ptr = send_array.raw_ptr();
    C* temp_ptr = temp_array.raw_ptr();
    C* temp_ptr_check = temp_array_check.raw_ptr();

    auto papars_1 = set_params<C>(Nx, Ny, Nz, send_ptr, temp_ptr_check, input_dim, transposed_dim, cuda_aware, pidx_i, pidx_j, p_j);
    auto papars_2 = set_params<C>(Nx, Ny, Nz, send_ptr, temp_ptr, input_dim, transposed_dim, cuda_aware, pidx_i, pidx_j, p_j);
    
    std::cout << "send_ptr: " << send_ptr << ", papars_1.srcArray: " << papars_1.srcPtr.ptr << std::endl;



    cuda3d_s.record();
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    CUDA_SAFE_CALL(cudaMemcpy3DAsync(&papars_1) ); //, stream)
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
    cuda3d_e.record();

    my_s.record();
    SCFD_SAFE_CALL(launchMemcpy3DKernel<C>(papars_2, NULL) );
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    my_e.record();

    for_each_t for_each;
    scfd::static_vec::rect<int, dim> range_T(idx_t(0,0,0), idx_t(Nx, Nz, Ny));
    scfd::static_vec::rect<int, dim> range(idx_t(0,0,0), idx_t(Nx, Ny, Nz));
    for_each( test_diff<idx_t, array_C_type>(temp_array, temp_array_check, diff_array), range_T);
    for_each.wait();
    for_each( manual_transpose<idx_t, array_C_type, 0, 2, 1>(send_array, temp_manual), range);
    for_each.wait();


    
    auto norm_diff_array =  get_norm<T, decltype(temp_array)>(diff_array, Nx*Ny*Nz);
    auto norm_temp_array =  get_norm<T, decltype(temp_array)>(temp_array, Nx*Ny*Nz);

    std::cout << "result norm: " << norm_temp_array << " norm diff: " << norm_diff_array << std::endl;
    
    std::cout << "cudaMemcpy3DAsync: " << cuda3d_e.elapsed_time(cuda3d_s) << " launchMemcpy3DKernel: " << my_e.elapsed_time(my_s) << std::endl;


    io::write_out_pos_file_scal_3D_point<T, array_C_type>("send_array.pos", send_array, Nx, Ny, Nz);
    io::write_out_pos_file_scal_3D_point<T, array_C_type>("temp_array.pos", temp_array, Nx, Nz, Ny);
    io::write_out_pos_file_scal_3D_point<T, array_C_type>("temp_array_check.pos", temp_array_check, Nx, Nz, Ny);
    io::write_out_pos_file_scal_3D_point<T, array_C_type>("temp_manual.pos", temp_manual, Nx, Nz, Ny);

    return 0;
}