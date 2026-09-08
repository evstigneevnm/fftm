#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <mpi.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/utils/log_mpi.h>
#include <fftm_backend.hpp>

int main(int argc, char **argv)
{
    scfd::communication::mpi_wrap mpi(argc, argv);
    auto comm = mpi.comm_world();
    scfd::utils::log_mpi log;
    int rank = 0, ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try
    {
        const bool direct = argc == 2 && std::string(argv[1]) == "device-aware";
        if (argc != 2 || (!direct && std::string(argv[1]) != "host-staged"))
            throw std::runtime_error("probe requires device-aware or host-staged");
        const int device = fftm::device_backend::init_mpi(log, comm, 0, false);
        using runtime = fftm::device_backend::fft<double>::runtime_api;
        using memory = fftm::device_backend::scfd_backend::memory_type;
        const auto identity = runtime::get_hardware_identity();
        const auto available = runtime::get_device_memory_info();
        if (!available.free_bytes_known || available.free_bytes < 2048ULL * 1024 * 1024)
            throw std::runtime_error("capsule requires at least 2048 MiB free per selected GPU");

        // Rank mapping must select distinct physical devices, not merely distinct ordinals.
        char pci[128] = {};
        std::snprintf(pci, sizeof(pci), "%s", identity.pci_bus_id.c_str());
        std::string peers(static_cast<std::size_t>(ranks) * sizeof(pci), '\0');
        MPI_Allgather(pci, sizeof(pci), MPI_CHAR, &peers[0], sizeof(pci), MPI_CHAR, MPI_COMM_WORLD);
        for (int a = 0; a < ranks; ++a)
            for (int b = a + 1; b < ranks; ++b)
                if (std::string(peers.data() + a * sizeof(pci)) == peers.data() + b * sizeof(pci))
                    throw std::runtime_error("two ranks selected the same physical GPU");

        constexpr int count = 16384;
        scfd::arrays::array_nd<int, 1, memory> send, receive;
        scfd::arrays::array_nd<int, 1, typename memory::host_memory_type> host_send, host_receive;
        send.init(count);
        receive.init(count);
        host_send.init(count);
        host_receive.init(count);
        for (int i = 0; i < count; ++i)
            host_send.raw_ptr()[i] = rank * 17 + i % 97;
        memory::copy_from_host(count * sizeof(int), host_send.raw_ptr(), send.raw_ptr());
        runtime::device_synchronize();
        const int previous = (rank + ranks - 1) % ranks;
        if (direct)
            MPI_Sendrecv(send.raw_ptr(), count, MPI_INT, (rank + 1) % ranks, 1,
                         receive.raw_ptr(), count, MPI_INT, previous, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
        {
            memory::copy_to_host(count * sizeof(int), send.raw_ptr(), host_send.raw_ptr());
            MPI_Sendrecv(host_send.raw_ptr(), count, MPI_INT, (rank + 1) % ranks, 1,
                         host_receive.raw_ptr(), count, MPI_INT, previous, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            memory::copy_from_host(count * sizeof(int), host_receive.raw_ptr(), receive.raw_ptr());
        }
        memory::copy_to_host(count * sizeof(int), receive.raw_ptr(), host_receive.raw_ptr());
        for (int i = 0; i < count; ++i)
            if (host_receive.raw_ptr()[i] != previous * 17 + i % 97)
                throw std::runtime_error("MPI buffer roundtrip mismatch");
        if (direct)
        {
            // UCX send/receive can work even when MPI's collective self-copy lacks GPU support.
            const int counts[] = {count}, displacements[] = {0};
            MPI_Alltoallv(send.raw_ptr(), counts, displacements, MPI_INT,
                          receive.raw_ptr(), counts, displacements, MPI_INT, MPI_COMM_SELF);
            memory::copy_to_host(count * sizeof(int), receive.raw_ptr(), host_receive.raw_ptr());
            for (int i = 0; i < count; ++i)
                if (host_receive.raw_ptr()[i] != rank * 17 + i % 97)
                    throw std::runtime_error("MPI collective GPU self-copy mismatch");
        }
        for (int r = 0; r < ranks; ++r)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (r == rank)
                std::cout << "CAPSULE_DEVICE {\"rank\":" << rank << ",\"device\":" << device
                          << ",\"backend\":" << std::quoted(identity.backend)
                          << ",\"name\":" << std::quoted(identity.device_name)
                          << ",\"pci\":" << std::quoted(identity.pci_bus_id)
                          << ",\"uuid\":" << std::quoted(identity.device_uuid)
                          << ",\"architecture\":" << std::quoted(identity.architecture)
                          << ",\"free_bytes\":" << available.free_bytes
                          << ",\"transport\":" << std::quoted(argv[1]) << "}" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "CAPSULE_PROBE_PASS" << std::endl;
    }
    catch (const std::exception &error)
    {
        std::cerr << "CAPSULE_PROBE_ERROR rank=" << rank << " " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
