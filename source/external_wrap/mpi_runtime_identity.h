#ifndef __FFTM_EXTERNAL_WRAP_MPI_RUNTIME_IDENTITY_H__
#define __FFTM_EXTERNAL_WRAP_MPI_RUNTIME_IDENTITY_H__

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <mpi.h>
#include <scfd/communication/mpi_comm_info.h>

namespace fftm
{
namespace wrap
{

struct mpi_runtime_identity
{
    std::string              library_version = "unavailable";
    std::vector<std::string> rank_node_names;
    std::vector<std::string> unique_node_names;
    std::vector<int>         ranks_per_node;
    bool                     distributed_details_available = false;
};

template <class MPIComm>
inline std::vector<std::string> all_gather_runtime_strings( const MPIComm &, const std::string &local_value )
{
    return std::vector<std::string>( 1, local_value );
}

inline std::vector<std::string> all_gather_runtime_strings(
    const scfd::communication::mpi_comm_info &comm, const std::string &local_value
)
{
    const int local_size = static_cast<int>( local_value.size() );
    std::vector<int> sizes( static_cast<std::size_t>( comm.num_procs ), 0 );
    SCFD_MPI_SAFE_CALL( MPI_Allgather( &local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm.comm ) );

    std::vector<int> offsets( sizes.size(), 0 );
    int              total_size = 0;
    for ( std::size_t i = 0; i < sizes.size(); ++i )
    {
        offsets[i] = total_size;
        total_size += sizes[i];
    }

    std::vector<char> packed( static_cast<std::size_t>( std::max( total_size, 1 ) ), '\0' );
    SCFD_MPI_SAFE_CALL( MPI_Allgatherv(
        local_value.data(), local_size, MPI_CHAR, packed.data(), sizes.data(), offsets.data(), MPI_CHAR, comm.comm
    ) );

    std::vector<std::string> result( sizes.size() );
    for ( std::size_t i = 0; i < sizes.size(); ++i )
        result[i].assign( packed.data() + offsets[i], static_cast<std::size_t>( sizes[i] ) );
    return result;
}

template <class MPIComm>
inline mpi_runtime_identity query_mpi_runtime_identity( const MPIComm & )
{
    return mpi_runtime_identity{};
}

inline mpi_runtime_identity query_mpi_runtime_identity( const scfd::communication::mpi_comm_info &comm )
{
    mpi_runtime_identity result;

    char library[MPI_MAX_LIBRARY_VERSION_STRING] = {};
    int  library_size = 0;
    SCFD_MPI_SAFE_CALL( MPI_Get_library_version( library, &library_size ) );
    while ( library_size > 0 && library[library_size - 1] == '\0' )
        --library_size;
    result.library_version.assign( library, static_cast<std::size_t>( library_size ) );

    char node[MPI_MAX_PROCESSOR_NAME] = {};
    int  node_size = 0;
    SCFD_MPI_SAFE_CALL( MPI_Get_processor_name( node, &node_size ) );
    result.rank_node_names = all_gather_runtime_strings(
        comm, std::string( node, static_cast<std::size_t>( node_size ) )
    );

    std::map<std::string, int> node_ranks;
    for ( const auto &name : result.rank_node_names )
        ++node_ranks[name];
    for ( const auto &entry : node_ranks )
    {
        result.unique_node_names.push_back( entry.first );
        result.ranks_per_node.push_back( entry.second );
    }
    result.distributed_details_available = true;
    return result;
}

} // namespace wrap
} // namespace fftm

#endif
