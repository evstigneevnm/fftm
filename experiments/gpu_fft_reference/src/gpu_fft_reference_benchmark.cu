#include <GPU_FFT/GPU_FFT.h>

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GPU_FFT_UPSTREAM_COMMIT
#define GPU_FFT_UPSTREAM_COMMIT "unknown"
#endif

namespace
{

struct options_t
{
    std::int64_t size = 0;
    int warmup = 5;
    int times = 30;
    double epsilon = 1.0e-11;
    std::string label = "gpu_fft_reference";
    std::string summary_path;
    std::string iterations_path;
};

struct validation_partial_t
{
    double error_squared;
    double reference_squared;
    double max_abs_error;
};

[[noreturn]] void fail(const std::string &message, int rank, int code = 2)
{
    std::cerr << "GPU_FFT_REFERENCE_ERROR rank=" << rank << " message=" << message
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, code);
    std::abort();
}

void check_cuda(cudaError_t status, const char *expression, int rank)
{
    if (status != cudaSuccess)
    {
        std::ostringstream stream;
        stream << expression << " failed: " << cudaGetErrorString(status);
        fail(stream.str(), rank, 3);
    }
}

void check_mpi(int status, const char *expression, int rank)
{
    if (status != MPI_SUCCESS)
    {
        char error[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        MPI_Error_string(status, error, &length);
        std::ostringstream stream;
        stream << expression << " failed: " << std::string(error, error + length);
        fail(stream.str(), rank, 4);
    }
}

#define GPU_FFT_CUDA(call, rank) check_cuda((call), #call, (rank))
#define GPU_FFT_MPI(call, rank) check_mpi((call), #call, (rank))

void print_usage(std::ostream &stream, const char *program)
{
    stream
        << "Usage: " << program << " --size N [options]\n"
        << "\n"
        << "Times a double-precision GPU-FFT R2C/normalize/C2R pair.\n"
        << "\n"
        << "Options:\n"
        << "  --size N             Cubic global size; the MPI rank count must divide N\n"
        << "  --warmup N           Untimed transform pairs (default: 5)\n"
        << "  --times N            Measured transform pairs (default: 30)\n"
        << "  --epsilon VALUE      Maximum accepted relative L2 error (default: 1e-11)\n"
        << "  --label TEXT         Compact case label\n"
        << "  --summary PATH       Write one summary CSV row\n"
        << "  --iterations PATH    Write per-iteration max-rank wall times\n"
        << "  --help                Show this message\n";
}

options_t parse_options(int argc, char **argv, int rank)
{
    options_t options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument(argv[index]);
        auto value = [&](const char *name) -> std::string {
            if (++index >= argc)
            {
                fail(std::string("missing value for ") + name, rank);
            }
            return argv[index];
        };

        if (argument == "--size")
        {
            options.size = std::stoll(value("--size"));
        }
        else if (argument == "--warmup")
        {
            options.warmup = std::stoi(value("--warmup"));
        }
        else if (argument == "--times")
        {
            options.times = std::stoi(value("--times"));
        }
        else if (argument == "--epsilon")
        {
            options.epsilon = std::stod(value("--epsilon"));
        }
        else if (argument == "--label")
        {
            options.label = value("--label");
        }
        else if (argument == "--summary")
        {
            options.summary_path = value("--summary");
        }
        else if (argument == "--iterations")
        {
            options.iterations_path = value("--iterations");
        }
        else if (argument == "--help")
        {
            if (rank == 0)
            {
                print_usage(std::cout, argv[0]);
            }
            MPI_Finalize();
            std::exit(0);
        }
        else
        {
            fail("unknown option: " + argument, rank);
        }
    }

    if (options.size <= 0 || options.warmup < 0 || options.times <= 0 ||
        !(options.epsilon > 0.0))
    {
        fail("invalid size, warmup, times, or epsilon", rank);
    }
    return options;
}

__device__ inline double expected_value(
    std::uint64_t global_x,
    std::uint64_t y,
    std::uint64_t z)
{
    const std::uint64_t value = (17ULL * global_x + 13ULL * y + 7ULL * z) % 101ULL;
    return (static_cast<double>(value) - 50.0) / 50.0;
}

__global__ void initialize_input(
    double *data,
    std::uint64_t physical_count,
    std::uint64_t size,
    std::uint64_t local_x_start)
{
    const std::uint64_t thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    for (std::uint64_t linear = thread; linear < physical_count; linear += stride)
    {
        const std::uint64_t z = linear % size;
        const std::uint64_t plane = linear / size;
        const std::uint64_t y = plane % size;
        const std::uint64_t local_x = plane / size;
        const std::uint64_t padded = plane * (size + 2ULL) + z;
        data[padded] = expected_value(local_x_start + local_x, y, z);
    }
}

__global__ void normalize_spectrum(
    cufftDoubleComplex *data,
    std::uint64_t count,
    double scale)
{
    const std::uint64_t thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = thread; index < count; index += stride)
    {
        data[index].x *= scale;
        data[index].y *= scale;
    }
}

__global__ void validate_input(
    const double *data,
    validation_partial_t *partials,
    std::uint64_t physical_count,
    std::uint64_t size,
    std::uint64_t local_x_start)
{
    extern __shared__ double shared[];
    double *error_squared = shared;
    double *reference_squared = shared + blockDim.x;
    double *max_abs_error = shared + 2 * blockDim.x;

    const std::uint64_t thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    double local_error_squared = 0.0;
    double local_reference_squared = 0.0;
    double local_max_abs_error = 0.0;

    for (std::uint64_t linear = thread; linear < physical_count; linear += stride)
    {
        const std::uint64_t z = linear % size;
        const std::uint64_t plane = linear / size;
        const std::uint64_t y = plane % size;
        const std::uint64_t local_x = plane / size;
        const std::uint64_t padded = plane * (size + 2ULL) + z;
        const double reference = expected_value(local_x_start + local_x, y, z);
        const double difference = data[padded] - reference;
        local_error_squared += difference * difference;
        local_reference_squared += reference * reference;
        local_max_abs_error = fmax(local_max_abs_error, fabs(difference));
    }

    error_squared[threadIdx.x] = local_error_squared;
    reference_squared[threadIdx.x] = local_reference_squared;
    max_abs_error[threadIdx.x] = local_max_abs_error;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (threadIdx.x < offset)
        {
            error_squared[threadIdx.x] += error_squared[threadIdx.x + offset];
            reference_squared[threadIdx.x] += reference_squared[threadIdx.x + offset];
            max_abs_error[threadIdx.x] =
                fmax(max_abs_error[threadIdx.x], max_abs_error[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        partials[blockIdx.x] = {
            error_squared[0], reference_squared[0], max_abs_error[0]};
    }
}

std::string csv_escape(const std::string &value)
{
    if (value.find_first_of(",\"\n") == std::string::npos)
    {
        return value;
    }
    std::string escaped = "\"";
    for (char character : value)
    {
        if (character == '"')
        {
            escaped += '"';
        }
        escaped += character;
    }
    escaped += '"';
    return escaped;
}

double percentile(std::vector<double> values, double fraction)
{
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size()))) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

std::string device_uuid(const cudaDeviceProp &properties)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (unsigned char byte : properties.uuid.bytes)
    {
        stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
}

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 0;
    GPU_FFT_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank), rank);
    GPU_FFT_MPI(MPI_Comm_size(MPI_COMM_WORLD, &ranks), rank);

    const options_t options = parse_options(argc, argv, rank);
    if (options.size % ranks != 0)
    {
        std::ostringstream stream;
        stream << "GPU-FFT slab decomposition requires size % ranks == 0; size="
               << options.size << " ranks=" << ranks;
        fail(stream.str(), rank);
    }

    MPI_Comm local_comm = MPI_COMM_NULL;
    GPU_FFT_MPI(
        MPI_Comm_split_type(
            MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm),
        rank);
    int local_rank = 0;
    int local_ranks = 0;
    GPU_FFT_MPI(MPI_Comm_rank(local_comm, &local_rank), rank);
    GPU_FFT_MPI(MPI_Comm_size(local_comm, &local_ranks), rank);

    int device_count = 0;
    GPU_FFT_CUDA(cudaGetDeviceCount(&device_count), rank);
    const int device = device_count == 1 ? 0 : local_rank;
    if (device < 0 || device >= device_count)
    {
        std::ostringstream stream;
        stream << "local rank " << local_rank << " has no visible CUDA device; visible="
               << device_count;
        fail(stream.str(), rank);
    }
    GPU_FFT_CUDA(cudaSetDevice(device), rank);

    cudaDeviceProp properties = {};
    GPU_FFT_CUDA(cudaGetDeviceProperties(&properties, device), rank);
    size_t free_before = 0;
    size_t total_memory = 0;
    GPU_FFT_CUDA(cudaMemGetInfo(&free_before, &total_memory), rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME] = {};
    int processor_name_length = 0;
    GPU_FFT_MPI(MPI_Get_processor_name(processor_name, &processor_name_length), rank);
    std::cerr << "GPU_FFT_REFERENCE_DEVICE host=" << processor_name
              << " rank=" << rank
              << " local_rank=" << local_rank
              << " local_ranks=" << local_ranks
              << " visible_devices=" << device_count
              << " device_ordinal=" << device
              << " name=" << properties.name
              << " uuid=" << device_uuid(properties)
              << " memory_mib=" << (total_memory / (1024ULL * 1024ULL))
              << std::endl;

    const std::uint64_t size = static_cast<std::uint64_t>(options.size);
    const std::uint64_t local_x = size / static_cast<std::uint64_t>(ranks);
    const std::uint64_t physical_count = local_x * size * size;
    const std::uint64_t complex_count = local_x * size * (size / 2ULL + 1ULL);
    const std::uint64_t complex_bytes = complex_count * sizeof(cufftDoubleComplex);
    if (complex_count > std::numeric_limits<std::size_t>::max() /
                            sizeof(cufftDoubleComplex))
    {
        fail("local allocation size overflows size_t", rank);
    }

    cufftDoubleComplex *data = nullptr;
    GPU_FFT_CUDA(
        cudaMalloc(reinterpret_cast<void **>(&data),
                   static_cast<std::size_t>(complex_bytes)),
        rank);
    GPU_FFT_CUDA(
        cudaMemset(data, 0, static_cast<std::size_t>(complex_bytes)), rank);

    constexpr int threads = 256;
    constexpr int blocks = 4096;
    initialize_input<<<blocks, threads>>>(
        reinterpret_cast<double *>(data),
        physical_count,
        size,
        static_cast<std::uint64_t>(rank) * local_x);
    GPU_FFT_CUDA(cudaGetLastError(), rank);
    GPU_FFT_CUDA(cudaDeviceSynchronize(), rank);

    MPI_Comm communicator = MPI_COMM_WORLD;
    GPU_FFT::INIT_GPU_FFT<TRANSITIONS::T1_d, TRANSITIONS::T2_d>(
        options.size,
        options.size,
        options.size,
        ranks,
        rank,
        communicator);
    GPU_FFT_CUDA(cudaGetLastError(), rank);
    GPU_FFT_CUDA(cudaDeviceSynchronize(), rank);

    size_t free_after_init = 0;
    size_t ignored_total = 0;
    GPU_FFT_CUDA(cudaMemGetInfo(&free_after_init, &ignored_total), rank);

    const double scale = 1.0 / static_cast<double>(size * size * size);
    auto transform_pair = [&]() {
        GPU_FFT::GPU_FFT_R2C<TRANSITIONS::T1_d, TRANSITIONS::T2_d>(
            reinterpret_cast<double *>(data));
        normalize_spectrum<<<blocks, threads>>>(data, complex_count, scale);
        GPU_FFT_CUDA(cudaGetLastError(), rank);
        GPU_FFT::GPU_FFT_C2R<TRANSITIONS::T1_d, TRANSITIONS::T2_d>(data);
        GPU_FFT_CUDA(cudaGetLastError(), rank);
        GPU_FFT_CUDA(cudaDeviceSynchronize(), rank);
    };

    for (int iteration = 0; iteration < options.warmup; ++iteration)
    {
        GPU_FFT_MPI(MPI_Barrier(MPI_COMM_WORLD), rank);
        transform_pair();
    }

    std::vector<double> pair_times_ms;
    if (rank == 0)
    {
        pair_times_ms.reserve(static_cast<std::size_t>(options.times));
    }
    for (int iteration = 0; iteration < options.times; ++iteration)
    {
        GPU_FFT_MPI(MPI_Barrier(MPI_COMM_WORLD), rank);
        const double begin = MPI_Wtime();
        transform_pair();
        const double local_elapsed_ms = (MPI_Wtime() - begin) * 1000.0;
        double max_elapsed_ms = 0.0;
        GPU_FFT_MPI(
            MPI_Reduce(
                &local_elapsed_ms,
                &max_elapsed_ms,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                0,
                MPI_COMM_WORLD),
            rank);
        if (rank == 0)
        {
            pair_times_ms.push_back(max_elapsed_ms);
            std::cout << "GPU_FFT_REFERENCE_ITERATION label=" << options.label
                      << " index=" << iteration
                      << " max_pair_ms=" << std::setprecision(12) << max_elapsed_ms
                      << std::endl;
        }
    }

    validation_partial_t *device_partials = nullptr;
    GPU_FFT_CUDA(
        cudaMalloc(
            reinterpret_cast<void **>(&device_partials),
            blocks * sizeof(validation_partial_t)),
        rank);
    validate_input<<<blocks, threads, 3 * threads * sizeof(double)>>>(
        reinterpret_cast<const double *>(data),
        device_partials,
        physical_count,
        size,
        static_cast<std::uint64_t>(rank) * local_x);
    GPU_FFT_CUDA(cudaGetLastError(), rank);
    std::vector<validation_partial_t> host_partials(blocks);
    GPU_FFT_CUDA(
        cudaMemcpy(
            host_partials.data(),
            device_partials,
            blocks * sizeof(validation_partial_t),
            cudaMemcpyDeviceToHost),
        rank);

    double local_sums[2] = {0.0, 0.0};
    double local_max_abs_error = 0.0;
    for (const validation_partial_t &partial : host_partials)
    {
        local_sums[0] += partial.error_squared;
        local_sums[1] += partial.reference_squared;
        local_max_abs_error = std::max(local_max_abs_error, partial.max_abs_error);
    }
    double global_sums[2] = {0.0, 0.0};
    double global_max_abs_error = 0.0;
    GPU_FFT_MPI(
        MPI_Reduce(local_sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD),
        rank);
    GPU_FFT_MPI(
        MPI_Reduce(
            &local_max_abs_error,
            &global_max_abs_error,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD),
        rank);

    int min_local_ranks = 0;
    int max_local_ranks = 0;
    GPU_FFT_MPI(
        MPI_Reduce(&local_ranks, &min_local_ranks, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD),
        rank);
    GPU_FFT_MPI(
        MPI_Reduce(&local_ranks, &max_local_ranks, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD),
        rank);

    std::vector<char> processor_names;
    if (rank == 0)
    {
        processor_names.resize(static_cast<std::size_t>(ranks) * MPI_MAX_PROCESSOR_NAME);
    }
    GPU_FFT_MPI(
        MPI_Gather(
            processor_name,
            MPI_MAX_PROCESSOR_NAME,
            MPI_CHAR,
            rank == 0 ? processor_names.data() : nullptr,
            MPI_MAX_PROCESSOR_NAME,
            MPI_CHAR,
            0,
            MPI_COMM_WORLD),
        rank);

    int result = 0;
    if (rank == 0)
    {
        std::vector<std::string> hosts;
        for (int process = 0; process < ranks; ++process)
        {
            const char *name = processor_names.data() +
                               static_cast<std::size_t>(process) * MPI_MAX_PROCESSOR_NAME;
            hosts.emplace_back(name);
        }
        std::sort(hosts.begin(), hosts.end());
        hosts.erase(std::unique(hosts.begin(), hosts.end()), hosts.end());

        const double average_ms =
            std::accumulate(pair_times_ms.begin(), pair_times_ms.end(), 0.0) /
            static_cast<double>(pair_times_ms.size());
        double variance = 0.0;
        for (double value : pair_times_ms)
        {
            const double difference = value - average_ms;
            variance += difference * difference;
        }
        variance /= static_cast<double>(pair_times_ms.size());
        const double standard_deviation_ms = std::sqrt(variance);
        const auto extrema = std::minmax_element(pair_times_ms.begin(), pair_times_ms.end());
        const double median_ms = percentile(pair_times_ms, 0.5);
        const double p95_ms = percentile(pair_times_ms, 0.95);
        const long double points =
            static_cast<long double>(size) * size * size;
        const long double pair_flops =
            10.0L * points * std::log2(points);
        const double tflops = static_cast<double>(
            pair_flops / (static_cast<long double>(average_ms) * 1.0e9L));
        const double relative_l2 =
            std::sqrt(global_sums[0] / global_sums[1]);
        const bool valid = std::isfinite(relative_l2) && relative_l2 <= options.epsilon;
        result = valid ? 0 : 5;

        if (!options.iterations_path.empty())
        {
            std::ofstream output(options.iterations_path);
            if (!output)
            {
                fail("cannot open iterations CSV: " + options.iterations_path, rank);
            }
            output << "label,size,nodes,ranks,iteration,max_pair_ms\n";
            output << std::setprecision(17);
            for (std::size_t index = 0; index < pair_times_ms.size(); ++index)
            {
                output << csv_escape(options.label) << ',' << size << ',' << hosts.size()
                       << ',' << ranks << ',' << index << ',' << pair_times_ms[index] << '\n';
            }
        }

        if (!options.summary_path.empty())
        {
            std::ofstream output(options.summary_path);
            if (!output)
            {
                fail("cannot open summary CSV: " + options.summary_path, rank);
            }
            output
                << "label,size,nodes,ranks,ranks_per_node_min,ranks_per_node_max,warmup,times,"
                << "avg_pair_ms,stddev_pair_ms,min_pair_ms,median_pair_ms,p95_pair_ms,max_pair_ms,"
                << "effective_tflops,relative_l2,max_abs_error,epsilon,valid,device_name,"
                << "device_memory_mib,free_before_mib,free_after_init_mib,local_data_mib,"
                << "upstream_commit\n";
            output << std::setprecision(17)
                   << csv_escape(options.label) << ',' << size << ',' << hosts.size() << ','
                   << ranks << ',' << min_local_ranks << ',' << max_local_ranks << ','
                   << options.warmup << ',' << options.times << ',' << average_ms << ','
                   << standard_deviation_ms << ',' << *extrema.first << ',' << median_ms << ','
                   << p95_ms << ',' << *extrema.second << ',' << tflops << ',' << relative_l2
                   << ',' << global_max_abs_error << ',' << options.epsilon << ','
                   << (valid ? 1 : 0) << ',' << csv_escape(properties.name) << ','
                   << (total_memory / (1024.0 * 1024.0)) << ','
                   << (free_before / (1024.0 * 1024.0)) << ','
                   << (free_after_init / (1024.0 * 1024.0)) << ','
                   << (complex_bytes / (1024.0 * 1024.0)) << ','
                   << GPU_FFT_UPSTREAM_COMMIT << '\n';
        }

        std::cout << std::setprecision(12)
                  << "GPU_FFT_REFERENCE_RESULT label=" << options.label
                  << " size=" << size
                  << " nodes=" << hosts.size()
                  << " ranks=" << ranks
                  << " avg_pair_ms=" << average_ms
                  << " median_pair_ms=" << median_ms
                  << " p95_pair_ms=" << p95_ms
                  << " effective_tflops=" << tflops
                  << " relative_l2=" << relative_l2
                  << " valid=" << (valid ? 1 : 0)
                  << std::endl;
    }

    GPU_FFT_MPI(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD), rank);
    GPU_FFT_CUDA(cudaFree(device_partials), rank);
    GPU_FFT_CUDA(cudaFree(data), rank);
    GPU_FFT_MPI(MPI_Comm_free(&local_comm), rank);
    MPI_Finalize();
    return result;
}
